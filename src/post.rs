//! The processing graph on the GPU: runs a compiled `PostPlan` (see
//! `codegen::generate_post`) on the rendered image, before the compositor maps it to the
//! display.
//!
//! Each `Step::Stage` is a fullscreen pass writing its own half-float texture at the
//! render resolution. Filters run in passes of their own: Blur is a separable gaussian
//! (at reduced resolution for large radii, so the radius, given in % of the image height,
//! looks the same at any resolution); Bloom reuses the bloom pyramid (bloom.rs).
//! Filter settings that aren't per-pixel (blur radius, bloom threshold / knee / radius)
//! are evaluated on the CPU, so they can still be animated or driven.

use egui_wgpu::wgpu;

use crate::bloom::{Bloom, BloomSettings};
use crate::graph::codegen::{Filter, LIVE_OFFSET, MAX_PARAM_VEC4S, PostPlan, Step, Tex};
use crate::graph::Graph;
use crate::images::Images;
use crate::world::{DOWNSAMPLE_SHADER, fullscreen, sampler_entry, texture_entry};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Largest blur sigma (in texels) run without downsampling first.
const MAX_SIGMA: f32 = 6.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostUniform {
    size: [f32; 4],
    misc: [f32; 4],
    scales: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    step: [f32; 2],
    sigma: f32,
    taps: f32,
}

fn target(device: &wgpu::Device, label: &str, size: [u32; 2]) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn draw(encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, bg: &wgpu::BindGroup) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("post pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.draw(0..3, 0..1);
}

/// A blur's textures: the downsampled chain, then horizontal / vertical passes at its
/// smallest level.
struct BlurRes {
    /// Render size and levels of downsampling they were made for.
    key: ([u32; 2], u32),
    chain: Vec<wgpu::TextureView>,
    tmp: wgpu::TextureView,
    out: wgpu::TextureView,
    /// Uniforms of the horizontal and vertical pass.
    bufs: [wgpu::Buffer; 2],
}

enum FilterRes {
    Blur(Option<BlurRes>),
    Bloom(Box<Bloom>),
}

/// Per-frame settings of a filter, evaluated on the CPU.
#[derive(Clone, Copy)]
enum FilterSettings {
    /// Sigma in texels of the render.
    Blur(f32),
    Bloom(BloomSettings),
}

struct Compiled {
    plan: PostPlan,
    stages: Vec<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
}

pub struct PostChain {
    compiled: Option<Compiled>,
    params_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    size: [u32; 2],
    /// The file of each image slot (set every frame: files change without a recompile).
    pub files: Vec<Option<String>>,
    stage_targets: Vec<wgpu::TextureView>,
    filters: Vec<FilterRes>,
    down: wgpu::RenderPipeline,
    down_bgl: wgpu::BindGroupLayout,
    blur: wgpu::RenderPipeline,
    blur_bgl: wgpu::BindGroupLayout,
}

impl PostChain {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let down_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post downsample bgl"),
            entries: &[texture_entry(0, frag), sampler_entry(1, frag)],
        });
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur bgl"),
            entries: &[
                texture_entry(0, frag),
                sampler_entry(1, frag),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: frag,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
        };
        let down = fullscreen(device, "post downsample", &module("downsample", DOWNSAMPLE_SHADER), "fs_main", &[Some(&down_bgl)], FORMAT);
        let blur = fullscreen(device, "blur", &module("blur", include_str!("shaders/blur.wgsl")), "fs_main", &[Some(&blur_bgl)], FORMAT);
        Self {
            compiled: None,
            params_buf: buffer("post params", MAX_PARAM_VEC4S * 16),
            uniform_buf: buffer("post uniform", size_of::<PostUniform>()),
            sampler,
            size: [0, 0],
            files: Vec::new(),
            stage_targets: Vec::new(),
            filters: Vec::new(),
            down,
            down_bgl,
            blur,
            blur_bgl,
        }
    }

    /// Swaps in a compiled processing graph. On error the previous one stays active.
    pub fn set_plan(&mut self, device: &wgpu::Device, plan: PostPlan) -> Result<(), String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let frag = wgpu::ShaderStages::FRAGMENT;
        let stages = plan
            .stages
            .iter()
            .enumerate()
            .map(|(i, st)| {
                let uniform = |binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: frag,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                };
                let mut entries = vec![uniform(0), uniform(1), sampler_entry(2, frag)];
                entries.extend((0..st.inputs.len()).map(|k| texture_entry(3 + k as u32, frag)));
                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("post stage bgl"), entries: &entries });
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("post stage {i}")),
                    source: wgpu::ShaderSource::Wgsl(st.wgsl.as_str().into()),
                });
                let pipeline = fullscreen(device, "post stage", &module, "fs_main", &[Some(&bgl)], FORMAT);
                (pipeline, bgl)
            })
            .collect();
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(e.to_string());
        }
        self.filters = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Filter { kind: Filter::Blur, .. } => Some(FilterRes::Blur(None)),
                Step::Filter { kind: Filter::Bloom, .. } => Some(FilterRes::Bloom(Box::new(Bloom::new(device)))),
                Step::Stage(_) => None,
            })
            .collect();
        self.compiled = Some(Compiled { plan, stages });
        self.size = [0, 0];
        Ok(())
    }

    /// True when the output is just the render (nothing to run).
    pub fn is_identity(&self) -> bool {
        self.compiled.as_ref().is_none_or(|c| c.plan.output == Tex::Render)
    }

    fn ensure_size(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        let Some(c) = &self.compiled else { return };
        if self.size == size && self.stage_targets.len() == c.plan.stages.len() {
            return;
        }
        self.size = size;
        self.stage_targets = (0..c.plan.stages.len()).map(|_| target(device, "post stage", size)).collect();
        for f in &mut self.filters {
            if let FilterRes::Blur(res) = f {
                *res = None;
            }
        }
    }

    /// Runs the processing graph on `render` (the fractal target, `size`). `g` is the
    /// processing graph (for filter settings), `t` the animation time.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        render: &wgpu::TextureView,
        size: [u32; 2],
        images: &Images,
        g: &Graph,
        t: f32,
        live: &[f32],
    ) {
        if self.is_identity() {
            return;
        }
        self.ensure_size(device, size);
        let steps = self.compiled.as_ref().unwrap().plan.steps.clone();

        // Filter settings, and blur textures for their current downsampling.
        let mut settings = Vec::new();
        for step in &steps {
            let Step::Filter { kind, pass, node, .. } = *step else { continue };
            let v = |j: usize| crate::graph::eval::input_value(g, node, j, t, live);
            let s = match kind {
                Filter::Blur => FilterSettings::Blur(v(1).max(0.0) / 100.0 * size[1] as f32),
                Filter::Bloom => FilterSettings::Bloom(BloomSettings {
                    enabled: true,
                    intensity: 1.0,
                    threshold: v(2).max(0.0),
                    knee: v(3).clamp(0.0, 1.0),
                    radius: v(4).clamp(0.5, 4.0),
                }),
            };
            match (&mut self.filters[pass], s) {
                (FilterRes::Blur(res), FilterSettings::Blur(sigma)) => Self::blur_textures(device, res, size, sigma),
                (FilterRes::Bloom(b), _) => {
                    b.ensure(device, size);
                }
                _ => {}
            }
            settings.push((pass, s));
        }

        let mut scales = [0f32; 16];
        for (p, f) in self.filters.iter().enumerate().take(16) {
            if let FilterRes::Bloom(b) = f {
                scales[p] = b.normalization();
            }
        }
        let u = PostUniform {
            size: [size[0] as f32, size[1] as f32, 1.0 / size[0].max(1) as f32, 1.0 / size[1].max(1) as f32],
            misc: [t, 0.0, size[0] as f32 / size[1].max(1) as f32, 0.0],
            scales,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        let mut packed = [0f32; MAX_PARAM_VEC4S * 4];
        let values = g.packed_values();
        let n = values.len().min(LIVE_OFFSET);
        packed[..n].copy_from_slice(&values[..n]);
        let m = live.len().min(packed.len() - LIVE_OFFSET);
        packed[LIVE_OFFSET..LIVE_OFFSET + m].copy_from_slice(&live[..m]);
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&packed));

        let c = self.compiled.as_ref().unwrap();
        for step in &steps {
            match *step {
                Step::Stage(i) => {
                    let (pipeline, bgl) = &c.stages[i];
                    let views: Vec<&wgpu::TextureView> = c.plan.stages[i].inputs.iter().map(|t| self.resolve(*t, render, images)).collect();
                    let mut entries = vec![
                        wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ];
                    entries.extend(views.iter().enumerate().map(|(k, v)| wgpu::BindGroupEntry {
                        binding: 3 + k as u32,
                        resource: wgpu::BindingResource::TextureView(v),
                    }));
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("post stage"), layout: bgl, entries: &entries });
                    draw(encoder, &self.stage_targets[i], pipeline, &bg);
                }
                Step::Filter { pass, input, .. } => {
                    let input = self.resolve(input, render, images);
                    let s = settings.iter().find(|(p, _)| *p == pass).map(|(_, s)| *s);
                    match (&self.filters[pass], s) {
                        (FilterRes::Blur(Some(res)), Some(FilterSettings::Blur(sigma))) => self.run_blur(device, queue, encoder, res, input, sigma),
                        (FilterRes::Bloom(b), Some(FilterSettings::Bloom(s))) => b.run(device, queue, encoder, input, &s),
                        _ => {}
                    }
                }
            }
        }
    }

    /// The processed image (the render itself when there's nothing to do).
    pub fn output<'a>(&'a self, render: &'a wgpu::TextureView, images: &'a Images) -> &'a wgpu::TextureView {
        match &self.compiled {
            Some(c) => self.resolve(c.plan.output, render, images),
            None => render,
        }
    }

    fn resolve<'a>(&'a self, t: Tex, render: &'a wgpu::TextureView, images: &'a Images) -> &'a wgpu::TextureView {
        match t {
            Tex::Render => render,
            Tex::Stage(i) => self.stage_targets.get(i).unwrap_or(render),
            Tex::Pass(p) => match self.filters.get(p) {
                Some(FilterRes::Blur(Some(res))) => &res.out,
                Some(FilterRes::Bloom(b)) => b.view(),
                _ => render,
            },
            Tex::Image(slot) => images.view(self.files.get(slot).and_then(|f| f.as_deref())),
        }
    }

    /// (Re)creates a blur's textures when the render size or its downsampling changed.
    fn blur_textures(device: &wgpu::Device, res: &mut Option<BlurRes>, size: [u32; 2], sigma: f32) {
        let mut levels = 0;
        while sigma / (1u32 << levels) as f32 > MAX_SIGMA && levels < 6 {
            levels += 1;
        }
        if res.as_ref().is_some_and(|r| r.key == (size, levels)) {
            return;
        }
        let level = |l: u32| [(size[0] >> l).max(1), (size[1] >> l).max(1)];
        let buffer = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("blur pass"),
                size: size_of::<BlurUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        *res = Some(BlurRes {
            key: (size, levels),
            chain: (1..=levels).map(|l| target(device, "blur level", level(l))).collect(),
            tmp: target(device, "blur tmp", level(levels)),
            out: target(device, "blur out", level(levels)),
            bufs: [buffer(), buffer()],
        });
    }

    fn run_blur(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        res: &BlurRes,
        input: &wgpu::TextureView,
        sigma: f32,
    ) {
        let (size, levels) = res.key;
        let tex_bg = |layout: &wgpu::BindGroupLayout, src: &wgpu::TextureView, buf: Option<&wgpu::Buffer>| {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ];
            if let Some(b) = buf {
                entries.push(wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() });
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("blur"), layout, entries: &entries })
        };
        let mut src = input;
        for view in &res.chain {
            draw(encoder, view, &self.down, &tex_bg(&self.down_bgl, src, None));
            src = view;
        }
        let w = [(size[0] >> levels).max(1) as f32, (size[1] >> levels).max(1) as f32];
        let s = sigma / (1u32 << levels) as f32;
        let taps = (s * 3.0).ceil().clamp(1.0, 24.0);
        let uniforms = [
            BlurUniform { step: [1.0 / w[0], 0.0], sigma: s, taps },
            BlurUniform { step: [0.0, 1.0 / w[1]], sigma: s, taps },
        ];
        for (buf, u) in res.bufs.iter().zip(&uniforms) {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(u));
        }
        draw(encoder, &res.tmp, &self.blur, &tex_bg(&self.blur_bgl, src, Some(&res.bufs[0])));
        draw(encoder, &res.out, &self.blur, &tex_bg(&self.blur_bgl, &res.tmp, Some(&res.bufs[1])));
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::codegen::{Step, Tex, generate_post};
    use crate::graph::{Graph, LegacyLook, NodeKind};

    fn validate(src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
    }

    fn filters(plan: &crate::graph::codegen::PostPlan) -> usize {
        plan.steps.iter().filter(|s| matches!(s, Step::Filter { .. })).count()
    }

    #[test]
    fn blur_validates() {
        validate(include_str!("shaders/blur.wgsl"));
    }

    /// The default chain: render -> bloom -> output is one bloom filter plus one pass.
    #[test]
    fn default_graph_plan() {
        let g = Graph::post_from(&LegacyLook::default());
        let plan = generate_post(&g).unwrap();
        assert_eq!(filters(&plan), 1);
        assert!(matches!(plan.steps[0], Step::Filter { input: Tex::Render, .. }));
        assert_eq!(plan.output, Tex::Stage(0));
        for s in &plan.stages {
            validate(&s.wgsl);
        }
        // Muting the bloom leaves the render untouched: nothing to run.
        let mut g = g;
        let bloom = g.find(NodeKind::Bloom).unwrap().id;
        g.node_mut(bloom).unwrap().muted = true;
        let plan = generate_post(&g).unwrap();
        assert_eq!(filters(&plan), 0);
        assert_eq!(plan.stages.len(), 1, "a muted bloom passes the render through a plain pass");
    }

    /// Every processing node in one graph: per-pixel chains, sampled effects, filters
    /// feeding filters, images, drivers.
    #[test]
    fn every_post_node_compiles() {
        use NodeKind::*;
        let mut g = Graph::post_from(&LegacyLook::default());
        let out = g.find(PostOutput).unwrap().id;
        let render = g.find(RenderLayers).unwrap().id;
        let bloom = g.find(Bloom).unwrap().id;
        let mut prev = (bloom, 0);
        for kind in [
            Exposure, BrightContrast, Curves, ColorBalance, Invert, Gamma, Posterize, Vignette, FilmGrain, LensDistortion,
            Blur, Pixelate, Sharpen, Kaleidoscope, Blur, HueSaturation,
        ] {
            let n = g.add(kind, [0.0; 2]);
            assert!(g.connect(prev.0, prev.1, n, 0), "{kind:?}");
            prev = (n, 0);
        }
        // Mask and depth drive things; an image is blended in; screen coords feed noise.
        let blend = g.add(Blend, [0.0; 2]);
        g.node_mut(blend).unwrap().set(0, &[5.0]);
        assert!(g.connect(prev.0, prev.1, blend, 2));
        let img = g.add(ImageTexture, [0.0; 2]);
        let coords = g.add(ScreenCoords, [0.0; 2]);
        assert!(g.connect(coords, 1, img, 0));
        assert!(g.connect(img, 0, blend, 3));
        assert!(g.connect(render, 2, blend, 1));
        let bw = g.add(RgbToBw, [0.0; 2]);
        assert!(g.connect(render, 0, bw, 0));
        let noise = g.add(NoiseTexture, [0.0; 2]);
        assert!(g.connect(coords, 2, noise, 1));
        let vig = g.nodes.iter().find(|n| n.kind == Vignette).unwrap().id;
        assert!(g.connect(noise, 0, vig, 1));
        let osc = g.add(Oscillator, [0.0; 2]);
        assert!(g.connect(osc, 0, bloom, 1));
        let glow_blur = g.add(Blur, [0.0; 2]);
        assert!(g.connect(bloom, 1, glow_blur, 0));
        let mb = g.add(MotionBlur, [0.0; 2]);
        assert!(g.connect(blend, 0, mb, 0));
        let add = g.add(Blend, [0.0; 2]);
        g.node_mut(add).unwrap().set(0, &[1.0]);
        assert!(g.connect(mb, 0, add, 2));
        assert!(g.connect(glow_blur, 0, add, 3));
        assert!(g.connect(add, 0, out, 0));
        assert!(g.motion_blur_shutter() > 0.0);

        let plan = generate_post(&g).unwrap();
        assert_eq!(filters(&plan), 4, "bloom + three blurs");
        assert_eq!(crate::graph::codegen::image_files(&g).len(), 1);
        for s in &plan.stages {
            validate(&s.wgsl);
        }
        // Every texture a step reads was written by an earlier step.
        let mut written = vec![Tex::Render, Tex::Image(0)];
        for step in &plan.steps {
            match step {
                Step::Stage(i) => {
                    for t in &plan.stages[*i].inputs {
                        assert!(written.contains(t), "stage {i} reads {t:?} before it exists");
                    }
                    written.push(Tex::Stage(*i));
                }
                Step::Filter { pass, input, .. } => {
                    assert!(written.contains(input), "filter {pass} reads {input:?} before it exists");
                    written.push(Tex::Pass(*pass));
                }
            }
        }
        assert!(written.contains(&plan.output));
    }

    #[test]
    fn wrong_graph_nodes_are_errors() {
        let mut g = Graph::post_from(&LegacyLook::default());
        let out = g.find(NodeKind::PostOutput).unwrap().id;
        let geo = g.add(NodeKind::Geometry, [0.0; 2]);
        assert!(g.connect(geo, 0, out, 0));
        assert!(generate_post(&g).err().unwrap().contains("color graph"));
    }
}
