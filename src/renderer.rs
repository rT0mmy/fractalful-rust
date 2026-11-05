use egui_wgpu::wgpu;

use crate::graph::codegen::{LIVE_OFFSET, MAX_PARAM_VEC4S};

const FRACTAL_TEMPLATE: &str = include_str!("shaders/fractal.wgsl");
pub const NODE_LIBRARY: &str = include_str!("shaders/nodes.wgsl");
pub const WORLD_LIBRARY: &str = include_str!("shaders/world.wgsl");

/// Running mean: blend the new sample in with the constant set per pass (1 / (n + 1)).
const ACCUMULATE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Constant,
        dst_factor: wgpu::BlendFactor::OneMinusConstant,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent::REPLACE,
};

pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Cone pre-pass target: one safe start distance per COARSE_BLOCK^2 pixels.
const COARSE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// G-buffer: hit distance (f32 bits, for deep zooms) and an octahedral normal, packed into
/// 8 bytes (a 16-byte float G-buffer cost ~10% on bandwidth-heavy scenes).
const GBUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Half-resolution shadow and ambient occlusion.
const VIS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Must match COARSE_BLOCK in fractal.wgsl.
const COARSE_BLOCK: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct Params {
    pub cam_pos: [f32; 4],
    pub cam_fwd: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub resolution: [f32; 4],
    pub base_color: [f32; 4],
    pub misc: [f32; 4],
    pub shadow: [f32; 4],
    pub jitter: [f32; 4],
    pub shade: [f32; 4],
}

impl Params {
    /// Everything that affects the image, excluding render size and sub-pixel jitter.
    /// When this is unchanged between frames, samples can be accumulated.
    pub fn scene_key(&self) -> Params {
        let mut k = *self;
        k.resolution[0] = 0.0;
        k.resolution[1] = 0.0;
        k.jitter = [0.0; 4];
        k
    }
}

/// Raymarches the fractal into an offscreen target at `render_scale` of the window
/// size (the compositor upscales it). Rendering below native resolution is the single
/// biggest performance lever for raymarching.
///
/// The target holds *linear, scene-referred* color (not tone mapped), so highlights can
/// exceed 1.0 for HDR output. Alpha < 0.5 marks debug views that must not be tone mapped.
///
/// The target doubles as an accumulation buffer: while the scene is static, jittered
/// frames are blended in with weight 1/(n+1) (a running mean, done by fixed-function
/// blending), which anti-aliases sub-pixel fractal detail.
///
/// Each sample runs four passes (see fractal.wgsl): cone pre-pass, march into a
/// G-buffer, shadows + AO at half resolution, shading.
pub struct FractalRenderer {
    pipelines: Pipelines,
    uniform_buf: wgpu::Buffer,
    graph_params_buf: wgpu::Buffer,
    uniform_bgl: wgpu::BindGroupLayout,
    uniform_bg: wgpu::BindGroup,
    coarse_bgl: wgpu::BindGroupLayout,
    /// World graph resources (group 2) and the lighting derived from them (group 3),
    /// see world.rs.
    world_bgl: wgpu::BindGroupLayout,
    light_bgl: wgpu::BindGroupLayout,
    /// Group 1 of the light pass (G-buffer) and the shading pass (G-buffer + light).
    gbuf_bgl: wgpu::BindGroupLayout,
    shade_bgl: wgpu::BindGroupLayout,
    target: Option<(wgpu::Texture, wgpu::TextureView, [u32; 2])>,
    /// Size-dependent intermediate targets and the bind groups reading them.
    passes: Option<PassTargets>,
    /// Frames rendered: rotates which pixel of each 2x2 block the light pass shades.
    frame: std::cell::Cell<u32>,
}

struct Pipelines {
    coarse: wgpu::RenderPipeline,
    gbuffer: wgpu::RenderPipeline,
    light: wgpu::RenderPipeline,
    shade: wgpu::RenderPipeline,
}

struct PassTargets {
    coarse: wgpu::TextureView,
    /// Group 1 of the march (the cone pre-pass result).
    coarse_bg: wgpu::BindGroup,
    gbuf: wgpu::TextureView,
    vis: wgpu::TextureView,
    light_bg: wgpu::BindGroup,
    shade_bg: wgpu::BindGroup,
}

impl FractalRenderer {
    /// `graph_wgsl` / `world_wgsl` are the generated material and world graph code; they
    /// must compile. `world_bgl` / `light_bgl` come from the `WorldRenderer`.
    pub fn new(
        device: &wgpu::Device,
        world_bgl: &wgpu::BindGroupLayout,
        light_bgl: &wgpu::BindGroupLayout,
        graph_wgsl: &str,
        world_wgsl: &str,
    ) -> Self {
        let graph_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("graph params"),
            size: (MAX_PARAM_VEC4S * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fractal params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fractal bgl"),
            entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fractal bg"),
            layout: &uniform_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: graph_params_buf.as_entire_binding() },
            ],
        });

        let coarse_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("coarse bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let gbuf = texture(1, wgpu::TextureSampleType::Uint);
        let gbuf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gbuffer bgl"), entries: &[gbuf] });
        let shade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shade bgl"),
            entries: &[gbuf, texture(2, wgpu::TextureSampleType::Float { filterable: false })],
        });
        let layouts = Layouts { uniform: &uniform_bgl, coarse: &coarse_bgl, gbuf: &gbuf_bgl, shade: &shade_bgl, world: world_bgl, light: light_bgl };
        let pipelines = fractal_pipelines(device, &layouts, graph_wgsl, world_wgsl)
            .unwrap_or_else(|e| panic!("initial fractal shader failed to compile: {e}"));
        Self {
            pipelines,
            uniform_buf,
            graph_params_buf,
            uniform_bgl,
            uniform_bg,
            coarse_bgl,
            world_bgl: world_bgl.clone(),
            light_bgl: light_bgl.clone(),
            gbuf_bgl,
            shade_bgl,
            target: None,
            passes: None,
            frame: std::cell::Cell::new(0),
        }
    }

    /// Swaps in new material / world graph code. On error the previous pipeline stays
    /// active.
    pub fn set_graph(&mut self, device: &wgpu::Device, graph_wgsl: &str, world_wgsl: &str) -> Result<(), String> {
        let layouts = Layouts {
            uniform: &self.uniform_bgl,
            coarse: &self.coarse_bgl,
            gbuf: &self.gbuf_bgl,
            shade: &self.shade_bgl,
            world: &self.world_bgl,
            light: &self.light_bgl,
        };
        self.pipelines = fractal_pipelines(device, &layouts, graph_wgsl, world_wgsl)?;
        Ok(())
    }

    /// Uploads the graph's inline values and the live (audio) node outputs.
    pub fn write_graph_params(&self, queue: &wgpu::Queue, params: &[f32], live: &[f32]) {
        let mut packed = [0f32; MAX_PARAM_VEC4S * 4];
        let n = params.len().min(LIVE_OFFSET);
        packed[..n].copy_from_slice(&params[..n]);
        let m = live.len().min(packed.len() - LIVE_OFFSET);
        packed[LIVE_OFFSET..LIVE_OFFSET + m].copy_from_slice(&live[..m]);
        queue.write_buffer(&self.graph_params_buf, 0, bytemuck::cast_slice(&packed));
    }

    /// (Re)creates the offscreen target only when its size actually changes.
    /// Returns true if it was recreated (its contents are then undefined).
    pub fn ensure_target(&mut self, device: &wgpu::Device, size: [u32; 2]) -> bool {
        let size = [size[0].max(1), size[1].max(1)];
        if self.target.as_ref().is_some_and(|t| t.2 == size) {
            return false;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fractal target"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        self.target = Some((tex, view, size));

        let texture = |label: &str, size: [u32; 2], format: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let coarse = texture("coarse cone depth", [size[0].div_ceil(COARSE_BLOCK), size[1].div_ceil(COARSE_BLOCK)], COARSE_FORMAT);
        let gbuf = texture("g-buffer", size, GBUFFER_FORMAT);
        let vis = texture("shadow / ao", [size[0].div_ceil(2), size[1].div_ceil(2)], VIS_FORMAT);
        fn entry(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }
        }
        let bg = |label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries })
        };
        let coarse_bg = bg("coarse bg", &self.coarse_bgl, &[entry(0, &coarse)]);
        let light_bg = bg("light bg", &self.gbuf_bgl, &[entry(1, &gbuf)]);
        let shade_bg = bg("shade bg", &self.shade_bgl, &[entry(1, &gbuf), entry(2, &vis)]);
        self.passes = Some(PassTargets { coarse, coarse_bg, gbuf, vis, light_bg, shade_bg });
        true
    }

    pub fn target_size(&self) -> [u32; 2] {
        self.target.as_ref().map_or([0, 0], |t| t.2)
    }

    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.target.as_ref().expect("ensure_target not called").1
    }

    /// `sample`: index of this sample in the accumulation (0 = start over), or `None`
    /// to skip raymarching entirely and keep the converged image. `moving_camera` re-runs
    /// the cone pre-pass for every sample (motion blur moves the camera between them).
    /// `world` holds the world graph's bind groups (2 and 3). `timer` marks each pass.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: &Params,
        world: [&wgpu::BindGroup; 2],
        sample: Option<u32>,
        moving_camera: bool,
        mut timer: Option<&mut crate::profiler::GpuTimer>,
    ) {
        let Some(n) = sample else { return };
        let t = self.passes.as_ref().expect("ensure_target not called");
        // jitter.w counts frames: the light pass rotates its 2x2 pixel with it.
        let frame = self.frame.get().wrapping_add(1);
        self.frame.set(frame);
        let mut params = *params;
        params.jitter[3] = (frame % 4) as f32;
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&params));

        let pl = &self.pipelines;
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        let mut draw = |label: &'static str, view: &wgpu::TextureView, load, pipeline: &wgpu::RenderPipeline, groups: &[(u32, &wgpu::BindGroup)], blend: f64| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_blend_constant(wgpu::Color { r: blend, g: blend, b: blend, a: blend });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.uniform_bg, &[]);
            for (i, g) in groups {
                pass.set_bind_group(*i, *g, &[]);
            }
            pass.draw(0..3, 0..1);
            drop(pass);
            if let Some(t) = timer.as_deref_mut() {
                t.mark(encoder, label);
            }
        };
        // The cone covers the jitter of later accumulation samples, so the pre-pass
        // only runs when the view changes (the first sample).
        if n == 0 || moving_camera {
            draw("cone pre-pass", &t.coarse, clear, &pl.coarse, &[(3, world[1])], 1.0);
        }
        draw("march", &t.gbuf, clear, &pl.gbuffer, &[(1, &t.coarse_bg), (3, world[1])], 1.0);
        draw("shadows + ao", &t.vis, clear, &pl.light, &[(1, &t.light_bg), (3, world[1])], 1.0);
        let load = if n == 0 { clear } else { wgpu::LoadOp::Load };
        let w = 1.0 / (n as f64 + 1.0);
        let target = &self.target.as_ref().expect("ensure_target not called").1;
        draw("shade", target, load, &pl.shade, &[(1, &t.shade_bg), (2, world[0]), (3, world[1])], w);
    }
}

pub fn save_png(path: &std::path::Path, [w, h]: [u32; 2], rgba: &[u8]) -> std::io::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)?;
    Ok(())
}

/// Splices the node and world libraries and the generated material and world graph
/// code into the fractal template at its `//@GRAPH` line.
pub fn assemble_shader(graph_wgsl: &str, world_wgsl: &str) -> String {
    splice(FRACTAL_TEMPLATE, "//@GRAPH", &[NODE_LIBRARY, WORLD_LIBRARY, graph_wgsl, world_wgsl])
}

/// Replaces the `marker` line of `template` with `parts` (matched per line, so CRLF
/// checkouts work too).
pub fn splice(template: &str, marker: &str, parts: &[&str]) -> String {
    let mut out = String::with_capacity(template.len() + parts.iter().map(|p| p.len() + 1).sum::<usize>());
    let mut found = false;
    for line in template.lines() {
        if line.trim() == marker {
            for p in parts {
                out.push_str(p);
                out.push('\n');
            }
            found = true;
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    assert!(found, "shader template lost its {marker} marker");
    out
}

/// Bind group layouts of the fractal passes: group 0 (uniforms), group 1 (per pass: the
/// cone result, the G-buffer, or the G-buffer + shadows / AO), 2 (world), 3 (lighting).
struct Layouts<'a> {
    uniform: &'a wgpu::BindGroupLayout,
    coarse: &'a wgpu::BindGroupLayout,
    gbuf: &'a wgpu::BindGroupLayout,
    shade: &'a wgpu::BindGroupLayout,
    world: &'a wgpu::BindGroupLayout,
    light: &'a wgpu::BindGroupLayout,
}

/// The four fractal pipelines, from one shader module.
fn fractal_pipelines(device: &wgpu::Device, l: &Layouts, graph_wgsl: &str, world_wgsl: &str) -> Result<Pipelines, String> {
    let source = assemble_shader(graph_wgsl, world_wgsl);
    // Catch validation errors instead of hitting wgpu's default panic handler.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fractal"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = |label: &str, entry: &str, bgls: &[Option<&wgpu::BindGroupLayout>], target: wgpu::ColorTargetState| {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: bgls,
            ..Default::default()
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
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
                targets: &[Some(target)],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    // Every pass reads the fog distance / sun from the lighting group.
    let pipelines = Pipelines {
        coarse: pipeline("cone pre-pass", "fs_coarse", &[Some(l.uniform), None, None, Some(l.light)], COARSE_FORMAT.into()),
        gbuffer: pipeline("march", "fs_gbuffer", &[Some(l.uniform), Some(l.coarse), None, Some(l.light)], GBUFFER_FORMAT.into()),
        light: pipeline("shadows + ao", "fs_light", &[Some(l.uniform), Some(l.gbuf), None, Some(l.light)], VIS_FORMAT.into()),
        shade: pipeline(
            "shade",
            "fs_shade",
            &[Some(l.uniform), Some(l.shade), Some(l.world), Some(l.light)],
            wgpu::ColorTargetState { format: TARGET_FORMAT, blend: Some(ACCUMULATE_BLEND), write_mask: wgpu::ColorWrites::ALL },
        ),
    };
    match pollster::block_on(scope.pop()) {
        None => Ok(pipelines),
        Some(e) => Err(e.to_string()),
    }
}

pub(crate) fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bgl: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        ..Default::default()
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}
