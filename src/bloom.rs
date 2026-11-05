//! Bloom post-process (see shaders/bloom.wgsl): bright-pass + mip pyramid of
//! downsamples, then additive tent upsamples. The compositor adds the result to the
//! scene before tone mapping, so it works the same for SDR and HDR output.

use egui_wgpu::wgpu;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MAX_LEVELS: usize = 7;

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BloomSettings {
    pub enabled: bool,
    pub intensity: f32,
    /// Linear brightness where bloom starts (1.0 = paper white).
    pub threshold: f32,
    /// Soft-knee width as a fraction of the threshold.
    pub knee: f32,
    /// Upsample filter radius in texels; larger spreads the glow further.
    pub radius: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self { enabled: true, intensity: 0.6, threshold: 1.0, knee: 0.5, radius: 1.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PassUniform {
    texel: [f32; 2],
    threshold: f32,
    knee: f32,
    radius: f32,
    _pad: [f32; 3],
}

struct Level {
    view: wgpu::TextureView,
    size: [u32; 2],
    /// Uniform for the pass that writes this level (prefilter / downsample).
    down_buf: wgpu::Buffer,
    /// Uniform + bind group reading this level (as the source of the next downsample
    /// and of the upsample into the level above).
    read_buf: wgpu::Buffer,
    read_bg: wgpu::BindGroup,
}

pub struct Bloom {
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    levels: Vec<Level>,
    scene_size: [u32; 2],
}

impl Bloom {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let pipeline = |entry: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            prefilter: pipeline("fs_prefilter", None),
            down: pipeline("fs_down", None),
            up: pipeline("fs_up", Some(additive)),
            bgl,
            sampler,
            levels: Vec::new(),
            scene_size: [0, 0],
        }
    }

    fn uniform(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom pass"),
            size: size_of::<PassUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn bind_group(&self, device: &wgpu::Device, src: &wgpu::TextureView, buf: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
            ],
        })
    }

    /// (Re)builds the pyramid for a scene of `size`: half resolution down to ~4 px.
    /// Returns true if it was rebuilt (contents undefined until the next `run`).
    pub fn ensure(&mut self, device: &wgpu::Device, size: [u32; 2]) -> bool {
        if self.scene_size == size && !self.levels.is_empty() {
            return false;
        }
        self.scene_size = size;
        self.levels.clear();
        let mut s = [(size[0] / 2).max(1), (size[1] / 2).max(1)];
        while self.levels.len() < MAX_LEVELS && (self.levels.is_empty() || s[0].min(s[1]) >= 4) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bloom mip"),
                size: wgpu::Extent3d { width: s[0], height: s[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let read_buf = Self::uniform(device);
            let read_bg = self.bind_group(device, &view, &read_buf);
            self.levels.push(Level { view, size: s, down_buf: Self::uniform(device), read_buf, read_bg });
            s = [(s[0] / 2).max(1), (s[1] / 2).max(1)];
        }
        true
    }

    /// The bloom result (pyramid level 0, half resolution).
    pub fn view(&self) -> &wgpu::TextureView {
        &self.levels[0].view
    }

    /// Each level adds its own energy on the way up; scale the sum back to ~1.
    pub fn normalization(&self) -> f32 {
        1.0 / self.levels.len().max(1) as f32
    }

    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &wgpu::TextureView,
        s: &BloomSettings,
    ) {
        let texel = |size: [u32; 2]| [1.0 / size[0] as f32, 1.0 / size[1] as f32];
        let write = |buf: &wgpu::Buffer, src_size: [u32; 2]| {
            let u = PassUniform {
                texel: texel(src_size),
                threshold: s.threshold,
                knee: s.knee,
                radius: s.radius,
                _pad: [0.0; 3],
            };
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&u));
        };
        let pass = |encoder: &mut wgpu::CommandEncoder,
                    target: &wgpu::TextureView,
                    pipeline: &wgpu::RenderPipeline,
                    bg: &wgpu::BindGroup,
                    load: wgpu::LoadOp<wgpu::Color>| {
            let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            p.set_pipeline(pipeline);
            p.set_bind_group(0, bg, &[]);
            p.draw(0..3, 0..1);
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);

        // Prefilter: scene -> level 0.
        let l0 = &self.levels[0];
        write(&l0.down_buf, self.scene_size);
        let scene_bg = self.bind_group(device, scene, &l0.down_buf);
        pass(encoder, &l0.view, &self.prefilter, &scene_bg, clear);

        // Reading a level uses the same uniform (its texel size, plus the radius) for
        // the downsample out of it and the upsample out of it.
        for level in &self.levels {
            write(&level.read_buf, level.size);
        }
        // Downsample: level i-1 -> level i.
        for i in 1..self.levels.len() {
            let (src, dst) = (&self.levels[i - 1], &self.levels[i]);
            pass(encoder, &dst.view, &self.down, &src.read_bg, clear);
        }
        // Upsample: level i -> level i-1, added onto its downsampled content.
        for i in (1..self.levels.len()).rev() {
            let (src, dst) = (&self.levels[i], &self.levels[i - 1]);
            pass(encoder, &dst.view, &self.up, &src.read_bg, wgpu::LoadOp::Load);
        }
    }
}
