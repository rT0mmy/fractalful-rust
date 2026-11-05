//! Display output: SDR / HDR10 / scRGB surface configuration and the final composite
//! pass (see composite.wgsl).
//!
//! The UI is drawn into its own sRGB layer and composited at paper-white brightness,
//! so it looks the same in every output mode.

use egui_wgpu::wgpu;

use crate::renderer::fullscreen_pipeline;

/// Format of the offscreen layer egui renders into.
pub const UI_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputMode {
    Sdr,
    /// BT.2100 PQ on a 10-bit swapchain.
    Hdr10,
    /// Extended linear sRGB on a 16-bit float swapchain (Windows' native HDR format).
    ScRgb,
}

impl OutputMode {
    pub const ALL: [OutputMode; 3] = [Self::Sdr, Self::Hdr10, Self::ScRgb];

    pub fn label(self) -> &'static str {
        match self {
            Self::Sdr => "SDR",
            Self::Hdr10 => "HDR10 (PQ, 10-bit)",
            Self::ScRgb => "scRGB (linear, 16-bit float)",
        }
    }
}

/// What the surface and the display can do, queried once at startup.
pub struct DisplayCaps {
    sdr_format: wgpu::TextureFormat,
    hdr10: bool,
    scrgb: bool,
    /// The OS reports the display is currently in HDR mode.
    pub hdr_active: bool,
    pub peak_nits: Option<f32>,
    pub sdr_white_nits: Option<f32>,
}

impl DisplayCaps {
    pub fn query(surface: &wgpu::Surface, adapter: &wgpu::Adapter) -> Self {
        let caps = surface.get_capabilities(adapter);
        let has = |format, space| {
            caps.format_capabilities.iter().any(|f| f.format == format && f.color_spaces.contains(space))
        };
        let info = surface.display_hdr_info(adapter);
        let lum = info.luminance.as_ref();
        Self {
            sdr_format: caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]),
            hdr10: has(wgpu::TextureFormat::Rgb10a2Unorm, wgpu::SurfaceColorSpaces::BT2100_PQ),
            scrgb: has(wgpu::TextureFormat::Rgba16Float, wgpu::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR),
            hdr_active: info.coarse.and_then(|c| c.high_dynamic_range).unwrap_or(false),
            peak_nits: lum.and_then(|l| l.max_nits).filter(|&n| n > 0.0),
            sdr_white_nits: lum.and_then(|l| l.sdr_white_nits).filter(|&n| n > 0.0),
        }
    }

    pub fn supports(&self, mode: OutputMode) -> bool {
        match mode {
            OutputMode::Sdr => true,
            OutputMode::Hdr10 => self.hdr10,
            OutputMode::ScRgb => self.scrgb,
        }
    }

    /// HDR10 when the display is in HDR mode and supports it, otherwise SDR.
    pub fn default_mode(&self) -> OutputMode {
        if self.hdr_active && self.hdr10 { OutputMode::Hdr10 } else { OutputMode::Sdr }
    }

    pub fn surface_format(&self, mode: OutputMode) -> (wgpu::TextureFormat, wgpu::SurfaceColorSpace) {
        match mode {
            OutputMode::Sdr => (self.sdr_format, wgpu::SurfaceColorSpace::Srgb),
            OutputMode::Hdr10 => (wgpu::TextureFormat::Rgb10a2Unorm, wgpu::SurfaceColorSpace::Bt2100Pq),
            OutputMode::ScRgb => (wgpu::TextureFormat::Rgba16Float, wgpu::SurfaceColorSpace::ExtendedSrgbLinear),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    mode: f32,
    paper_white: f32,
    peak: f32,
    srgb_encode: f32,
    /// View transform (see graph::VIEW_TRANSFORMS).
    view: f32,
    no_ui: f32,
    /// Fraction of the output's width / height the scene covers (letterboxing).
    fit: [f32; 2],
}

/// Offscreen composite target (see `Compositor::render_offscreen`).
#[derive(Clone, Copy)]
pub enum Offscreen {
    Sdr,
    Hdr10 { paper_white: f32, peak: f32 },
}

/// Scale of the scene inside the output so it keeps its aspect ratio (bars on the
/// sides or top and bottom). Near-equal aspects (the window at a reduced render scale
/// rounds to whole pixels) fill the output exactly.
fn fit(scene: [u32; 2], out: [u32; 2]) -> [f32; 2] {
    let aspect = |s: [u32; 2]| s[0].max(1) as f32 / s[1].max(1) as f32;
    let r = aspect(scene) / aspect(out);
    if (r - 1.0).abs() < 0.02 {
        [1.0, 1.0]
    } else if r > 1.0 {
        [1.0, 1.0 / r]
    } else {
        [r, 1.0]
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Inputs for a composite: the linear scene (after the processing graph) and how to map
/// it to the display.
pub struct Scene<'a> {
    pub view: &'a wgpu::TextureView,
    pub size: [u32; 2],
    /// Index into graph::VIEW_TRANSFORMS.
    pub view_transform: usize,
}

pub struct Compositor {
    bgl: wgpu::BindGroupLayout,
    /// Offscreen pipelines (screenshots, video frames) per format, created on first use.
    offscreen_pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    sampler: wgpu::Sampler,
    config_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    format: wgpu::TextureFormat,
    ui_layer: Option<(wgpu::TextureView, [u32; 2])>,
}

impl Compositor {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bgl"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite config"),
            size: size_of::<Config>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = Self::pipeline(device, &bgl, format);
        Self { bgl, offscreen_pipelines: Vec::new(), sampler, config_buf, pipeline, format, ui_layer: None }
    }

    fn pipeline(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        fullscreen_pipeline(device, "composite", include_str!("shaders/composite.wgsl"), bgl, format, None)
    }

    /// Rebuilds the pipeline when the swapchain format changes (output mode switch).
    pub fn set_format(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        if format != self.format {
            self.pipeline = Self::pipeline(device, &self.bgl, format);
            self.format = format;
        }
    }

    /// The UI layer, (re)created to match the window size.
    pub fn ui_layer(&mut self, device: &wgpu::Device, size: [u32; 2]) -> &wgpu::TextureView {
        if self.ui_layer.as_ref().is_none_or(|l| l.1 != size) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ui layer"),
                size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: UI_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.ui_layer = Some((tex.create_view(&Default::default()), size));
        }
        &self.ui_layer.as_ref().unwrap().0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        out: &wgpu::TextureView,
        out_size: [u32; 2],
        mode: OutputMode,
        paper_white: f32,
        peak: f32,
    ) {
        let config = Config {
            mode: match mode {
                OutputMode::Sdr => 0.0,
                OutputMode::Hdr10 => 1.0,
                OutputMode::ScRgb => 2.0,
            },
            paper_white,
            peak,
            srgb_encode: (mode == OutputMode::Sdr && !self.format.is_srgb()) as u32 as f32,
            view: scene.view_transform as f32,
            no_ui: 0.0,
            fit: fit(scene.size, out_size),
        };
        self.composite(device, queue, encoder, &self.pipeline, scene, out, &config);
    }

    /// Renders the scene (without UI) into an offscreen image of `size` and
    /// reads it back: RGBA8 sRGB for `Offscreen::Sdr`, or RGBA16 (little-endian,
    /// BT.2020 + PQ encoded) for `Offscreen::Hdr10`. Blocks on the GPU.
    pub fn render_offscreen(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        size: [u32; 2],
        kind: Offscreen,
    ) -> Vec<u8> {
        let (format, bpp, config) = match kind {
            Offscreen::Sdr => (wgpu::TextureFormat::Rgba8UnormSrgb, 4, (0.0, 200.0, 200.0)),
            // f16 keeps ~11 bits of the 0..1 PQ signal: enough for 10-bit video.
            Offscreen::Hdr10 { paper_white, peak } => (wgpu::TextureFormat::Rgba16Float, 8, (1.0, paper_white, peak)),
        };
        // The UI layer is ignored (no_ui) but must be bound; it may not exist yet when
        // the first frame of a command-line export renders.
        if self.ui_layer.is_none() {
            self.ui_layer(device, [1, 1]);
        }
        if !self.offscreen_pipelines.iter().any(|(f, _)| *f == format) {
            self.offscreen_pipelines.push((format, Self::pipeline(device, &self.bgl, format)));
        }
        let pipeline = &self.offscreen_pipelines.iter().find(|(f, _)| *f == format).unwrap().1;
        let (w, h) = (size[0].max(1), size[1].max(1));
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen composite"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let config = Config {
            mode: config.0,
            paper_white: config.1,
            peak: config.2,
            srgb_encode: 0.0,
            view: scene.view_transform as f32,
            no_ui: 1.0,
            fit: [1.0; 2],
        };
        let row = (w * bpp).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let view = tex.create_view(&Default::default());
        self.composite(device, queue, &mut encoder, pipeline, scene, &view, &config);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map offscreen buffer"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range().expect("mapped range");
        let rows = (0..h).flat_map(|y| data[(y * row) as usize..][..(w * bpp) as usize].iter().copied());
        match kind {
            Offscreen::Sdr => rows.collect(),
            // f16 -> u16 (0..1 -> 0..65535), little-endian, for ffmpeg's rgba64le.
            Offscreen::Hdr10 { .. } => rows
                .collect::<Vec<u8>>()
                .chunks_exact(2)
                .flat_map(|c| {
                    let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                    ((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes()
                })
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        scene: &Scene,
        out: &wgpu::TextureView,
        config: &Config,
    ) {
        queue.write_buffer(&self.config_buf, 0, bytemuck::bytes_of(config));
        let ui = &self.ui_layer.as_ref().expect("ui_layer not created").0;
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(ui) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.config_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn bloom_shader_validates() {
        let src = include_str!("shaders/bloom.wgsl");
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap();
    }

    #[test]
    fn composite_shader_validates() {
        let src = include_str!("shaders/composite.wgsl");
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap();
    }
}
