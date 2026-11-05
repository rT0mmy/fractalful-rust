//! The world graph on the GPU: sky, sun, ambient light and fog.
//!
//! Each update (when the world changed, see `App::frame`):
//! 1. `cs_settings` evaluates the World Output's sun / fog / ambient values into the
//!    lighting buffer (`WorldLight` in world.wgsl);
//! 2. the world graph is rendered into a small equirectangular environment map, and
//!    its mips are built (rough reflections and fog sample them);
//! 3. `cs_sh` projects a mip onto spherical harmonics: ambient light for any normal.
//!
//! The fractal shader reads the world through two bind groups: group 2 (the world
//! graph's parameters and image files, to draw the background exactly) and group 3
//! (the lighting buffer and the environment map).

use egui_wgpu::wgpu;

use crate::graph::codegen::{LIVE_OFFSET, MAX_PARAM_VEC4S, WORLD_IMAGES, WorldCode};
use crate::images::Images;
use crate::renderer::{NODE_LIBRARY, WORLD_LIBRARY, splice};

const ENV_TEMPLATE: &str = include_str!("shaders/world_env.wgsl");
const COMPUTE_TEMPLATE: &str = include_str!("shaders/world_compute.wgsl");
pub const DOWNSAMPLE_SHADER: &str = include_str!("shaders/env_down.wgsl");

const ENV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ENV_SIZE: [u32; 2] = [256, 128];
/// Mip levels (256x128 down to 2x1); must match ENV_MAX_LOD + 1 in world_compute.wgsl.
const ENV_MIPS: u32 = 8;
/// The mip the spherical harmonics are computed from (64x32).
const SH_MIP: u32 = 2;
/// Size of `WorldLight` in world.wgsl: 3 vec4 + 9 vec4.
const LIGHT_SIZE: u64 = 12 * 16;

struct Pipelines {
    env: wgpu::RenderPipeline,
    settings: wgpu::ComputePipeline,
    sh: wgpu::ComputePipeline,
}

pub struct WorldRenderer {
    pub world_bgl: wgpu::BindGroupLayout,
    pub light_bgl: wgpu::BindGroupLayout,
    env_light_bgl: wgpu::BindGroupLayout,
    compute_light_bgl: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    down: wgpu::RenderPipeline,
    params_buf: wgpu::Buffer,
    time_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    env_mips: Vec<wgpu::TextureView>,
    /// Reads mip i - 1, for building mip i.
    down_bgs: Vec<wgpu::BindGroup>,
    world_bg: wgpu::BindGroup,
    light_bg: wgpu::BindGroup,
    env_light_bg: wgpu::BindGroup,
    compute_light_bg: wgpu::BindGroup,
    /// Files bound to the image slots, and the image cache version they were bound at.
    bound: (Vec<Option<String>>, u64),
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

pub fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}

/// The shader of a world pass: `template` with the node and world libraries and the
/// generated world graph at its `//@WORLD` line.
fn world_source(template: &str, world_wgsl: &str) -> String {
    splice(template, "//@WORLD", &[NODE_LIBRARY, WORLD_LIBRARY, world_wgsl])
}

/// Fullscreen render pipeline drawing `entry` of `module` into `format`.
pub fn fullscreen(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry: &str,
    bgls: &[Option<&wgpu::BindGroupLayout>],
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: bgls, ..Default::default() });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState { module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn draw(encoder: &mut wgpu::CommandEncoder, label: &str, target: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, bgs: &[&wgpu::BindGroup]) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    for (i, bg) in bgs.iter().enumerate() {
        pass.set_bind_group(i as u32, *bg, &[]);
    }
    pass.draw(0..3, 0..1);
}

impl WorldRenderer {
    /// `code` must compile (it's the initial world graph).
    pub fn new(device: &wgpu::Device, images: &Images, code: &WorldCode) -> Self {
        let frag_comp = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let world_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world bgl"),
            entries: &[
                uniform_entry(0, frag_comp),
                texture_entry(1, frag_comp),
                texture_entry(2, frag_comp),
                sampler_entry(3, frag_comp),
                uniform_entry(4, frag_comp),
            ],
        });
        let light_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world light bgl"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::FRAGMENT), texture_entry(1, wgpu::ShaderStages::FRAGMENT)],
        });
        let env_light_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env light bgl"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::FRAGMENT)],
        });
        let compute_light_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world compute bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let down_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("downsample bgl"),
            entries: &[texture_entry(0, wgpu::ShaderStages::FRAGMENT), sampler_entry(1, wgpu::ShaderStages::FRAGMENT)],
        });

        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        let params_buf = buffer("world params", (MAX_PARAM_VEC4S * 16) as u64, wgpu::BufferUsages::UNIFORM);
        let time_buf = buffer("world time", 16, wgpu::BufferUsages::UNIFORM);
        let light_buf = buffer("world light", LIGHT_SIZE, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("world sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let env = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment map"),
            size: wgpu::Extent3d { width: ENV_SIZE[0], height: ENV_SIZE[1], depth_or_array_layers: 1 },
            mip_level_count: ENV_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let env_mips: Vec<wgpu::TextureView> = (0..ENV_MIPS)
            .map(|m| env.create_view(&wgpu::TextureViewDescriptor { base_mip_level: m, mip_level_count: Some(1), ..Default::default() }))
            .collect();
        let env_full = env.create_view(&Default::default());
        let down_bgs = (1..ENV_MIPS as usize)
            .map(|m| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("env mip"),
                    layout: &down_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&env_mips[m - 1]) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                    ],
                })
            })
            .collect();
        let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world light"),
            layout: &light_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&env_full) },
            ],
        });
        let env_light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("env light"),
            layout: &env_light_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() }],
        });
        let compute_light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world compute"),
            layout: &compute_light_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&env_mips[SH_MIP as usize]) },
            ],
        });
        let down_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("env downsample"),
            source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_SHADER.into()),
        });
        let down = fullscreen(device, "env downsample", &down_module, "fs_main", &[Some(&down_bgl)], ENV_FORMAT);
        let world_bg = Self::world_bind_group(device, &world_bgl, &params_buf, &time_buf, &sampler, [images.view(None), images.view(None)]);
        let pipelines = Self::pipelines(device, &world_bgl, &env_light_bgl, &compute_light_bgl, &code.wgsl)
            .unwrap_or_else(|e| panic!("initial world shader failed to compile: {e}"));
        Self {
            world_bgl,
            light_bgl,
            env_light_bgl,
            compute_light_bgl,
            pipelines,
            down,
            params_buf,
            time_buf,
            sampler,
            env_mips,
            down_bgs,
            world_bg,
            light_bg,
            env_light_bg,
            compute_light_bg,
            bound: (Vec::new(), u64::MAX),
        }
    }

    fn world_bind_group(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        time: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        images: [&wgpu::TextureView; WORLD_IMAGES],
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(images[0]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(images[1]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: time.as_entire_binding() },
            ],
        })
    }

    fn pipelines(
        device: &wgpu::Device,
        world_bgl: &wgpu::BindGroupLayout,
        env_light_bgl: &wgpu::BindGroupLayout,
        compute_light_bgl: &wgpu::BindGroupLayout,
        world_wgsl: &str,
    ) -> Result<Pipelines, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let env_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world env"),
            source: wgpu::ShaderSource::Wgsl(world_source(ENV_TEMPLATE, world_wgsl).into()),
        });
        let env = fullscreen(device, "world env", &env_module, "fs_env", &[Some(world_bgl), Some(env_light_bgl)], ENV_FORMAT);
        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world compute"),
            source: wgpu::ShaderSource::Wgsl(world_source(COMPUTE_TEMPLATE, world_wgsl).into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world compute"),
            bind_group_layouts: &[Some(world_bgl), Some(compute_light_bgl)],
            ..Default::default()
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &compute_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines { env, settings: compute("cs_settings"), sh: compute("cs_sh") };
        match pollster::block_on(scope.pop()) {
            None => Ok(pipelines),
            Some(e) => Err(e.to_string()),
        }
    }

    /// Swaps in a new world graph. On error the previous one stays active.
    pub fn set_graph(&mut self, device: &wgpu::Device, code: &WorldCode) -> Result<(), String> {
        self.pipelines = Self::pipelines(device, &self.world_bgl, &self.env_light_bgl, &self.compute_light_bgl, &code.wgsl)?;
        Ok(())
    }

    /// Binds the image files of the world graph's slots (rebuilt only when the files or
    /// the loaded images change). Returns true if the binding changed.
    pub fn bind_images(&mut self, device: &wgpu::Device, images: &mut Images, files: &[Option<String>]) -> bool {
        if self.bound.0 == files && self.bound.1 == images.version {
            return false;
        }
        for f in files.iter().flatten() {
            images.request(f);
        }
        let view = |i: usize| images.view(files.get(i).and_then(|f| f.as_deref()));
        self.world_bg = Self::world_bind_group(device, &self.world_bgl, &self.params_buf, &self.time_buf, &self.sampler, [view(0), view(1)]);
        self.bound = (files.to_vec(), images.version);
        true
    }

    /// Uploads the world graph's inline values and live node outputs.
    pub fn write_params(&self, queue: &wgpu::Queue, params: &[f32], live: &[f32], time: f32) {
        let mut packed = [0f32; MAX_PARAM_VEC4S * 4];
        let n = params.len().min(LIVE_OFFSET);
        packed[..n].copy_from_slice(&params[..n]);
        let m = live.len().min(packed.len() - LIVE_OFFSET);
        packed[LIVE_OFFSET..LIVE_OFFSET + m].copy_from_slice(&live[..m]);
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(&self.time_buf, 0, bytemuck::cast_slice(&[time, 0.0, 0.0, 0.0]));
    }

    /// Recomputes the lighting and the environment map from the current parameters.
    pub fn update(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("world settings"), timestamp_writes: None });
            pass.set_pipeline(&self.pipelines.settings);
            pass.set_bind_group(0, &self.world_bg, &[]);
            pass.set_bind_group(1, &self.compute_light_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        draw(encoder, "world env", &self.env_mips[0], &self.pipelines.env, &[&self.world_bg, &self.env_light_bg]);
        for m in 1..ENV_MIPS as usize {
            draw(encoder, "env mip", &self.env_mips[m], &self.down, &[&self.down_bgs[m - 1]]);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("world sh"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.sh);
        pass.set_bind_group(0, &self.world_bg, &[]);
        pass.set_bind_group(1, &self.compute_light_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Bind groups 2 (world) and 3 (lighting) of the fractal shader.
    pub fn bind_groups(&self) -> [&wgpu::BindGroup; 2] {
        [&self.world_bg, &self.light_bg]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::codegen::generate_world;
    use crate::graph::{Graph, LegacyLook, NodeKind};

    fn validate(src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
    }

    /// Every world node, in both world passes and the fractal shader.
    #[test]
    fn world_shaders_validate() {
        let mut g = Graph::world_from(&LegacyLook::default());
        let out = g.find(NodeKind::WorldOutput).unwrap().id;
        let sky = g.find(NodeKind::SkyGradient).unwrap().id;
        let phys = g.add(NodeKind::PhysicalSky, [0.0; 2]);
        let env = g.add(NodeKind::EnvTexture, [0.0; 2]);
        let env2 = g.add(NodeKind::EnvTexture, [0.0; 2]);
        let stars = g.add(NodeKind::StarField, [0.0; 2]);
        let dir = g.add(NodeKind::Direction, [0.0; 2]);
        let sun = g.add(NodeKind::SunInfo, [0.0; 2]);
        let osc = g.add(NodeKind::Oscillator, [0.0; 2]);
        let mut sum = sky;
        for n in [phys, env, env2, stars] {
            let b = g.add(NodeKind::Blend, [0.0; 2]);
            g.node_mut(b).unwrap().set(0, &[1.0]);
            assert!(g.connect(sum, 0, b, 2));
            assert!(g.connect(n, 0, b, 3));
            sum = b;
        }
        let noise = g.add(NodeKind::NoiseTexture, [0.0; 2]);
        assert!(g.connect(dir, 0, noise, 0));
        assert!(g.connect(noise, 0, stars, 1));
        assert!(g.connect(sun, 1, phys, 2));
        assert!(g.connect(sum, 0, out, 0));
        assert!(g.connect(osc, 0, out, crate::graph::WORLD_SUN_ELEVATION));
        assert!(g.connect(sun, 2, out, crate::graph::WORLD_SUN_COLOR));
        let code = generate_world(&g).unwrap();
        assert_eq!(crate::graph::codegen::image_files(&g).len(), 2);
        validate(&world_source(ENV_TEMPLATE, &code.wgsl));
        validate(&world_source(COMPUTE_TEMPLATE, &code.wgsl));
        let material = crate::graph::codegen::generate(&crate::graph::preset("Mandelbulb").0).unwrap();
        validate(&crate::renderer::assemble_shader(&material, &code.wgsl));

        // Material-only nodes are refused.
        let geo = g.add(NodeKind::Geometry, [0.0; 2]);
        g.connect(geo, 2, out, crate::graph::WORLD_STRENGTH);
        assert!(generate_world(&g).err().unwrap().contains("color graph"));
    }

    #[test]
    fn downsample_validates() {
        validate(DOWNSAMPLE_SHADER);
    }
}
