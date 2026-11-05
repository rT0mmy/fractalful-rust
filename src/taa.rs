//! Temporal anti-aliasing and upscaling (see shaders/taa.wgsl): resolves the fractal
//! render, at whatever render scale dynamic resolution picked, into a history at window
//! resolution. While moving, each frame adds one jittered sample per render pixel and
//! the history carries the rest, so detail stays stable instead of shimmering; while
//! still, the exact accumulation takes over.

use egui_wgpu::wgpu;

use crate::renderer::{Params, TARGET_FORMAT};
use crate::world::{fullscreen, sampler_entry, texture_entry};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct TaaUniform {
    cam: [[f32; 4]; 4],
    prev: [[f32; 4]; 4],
    sizes: [f32; 4],
    misc: [f32; 4],
    flags: [f32; 4],
}

/// How the current render relates to the history.
#[derive(Clone, Copy)]
pub enum Mode {
    /// One jittered sample per render pixel, offset by `jitter` render pixels.
    Temporal { jitter: [f32; 2] },
    /// The render is an exact accumulation; fade the history over to it by `weight`.
    Still { weight: f32 },
}

pub struct Taa {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    /// Ping-pong history (the output of the last two frames).
    hist: Vec<wgpu::TextureView>,
    size: [u32; 2],
    /// Which history the last frame wrote.
    current: usize,
    /// The other history holds a usable previous frame.
    valid: bool,
    /// The camera of the last resolved frame.
    prev: Option<[[f32; 4]; 4]>,
}

fn camera(p: &Params) -> [[f32; 4]; 4] {
    [p.cam_pos, p.cam_fwd, p.cam_right, p.cam_up]
}

impl Taa {
    pub fn new(device: &wgpu::Device) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa bgl"),
            entries: &[
                texture_entry(0, frag),
                texture_entry(1, frag),
                sampler_entry(2, frag),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: frag,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/taa.wgsl").into()),
        });
        let pipeline = fullscreen(device, "taa", &module, "fs_main", &[Some(&bgl)], TARGET_FORMAT);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa uniform"),
            size: size_of::<TaaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bgl, sampler, uniform, hist: Vec::new(), size: [0, 0], current: 0, valid: false, prev: None }
    }

    /// Forgets the history (the next frame starts from its own samples).
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Resolves `render` (`render_size`, rendered with `params`) into the history at
    /// `size`. Returns the resolved image.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        render: &wgpu::TextureView,
        render_size: [u32; 2],
        size: [u32; 2],
        params: &Params,
        mode: Mode,
    ) -> &wgpu::TextureView {
        if self.size != size || self.hist.is_empty() {
            self.hist = (0..2)
                .map(|_| {
                    device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("taa history"),
                            size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: TARGET_FORMAT,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        })
                        .create_view(&Default::default())
                })
                .collect();
            self.size = size;
            self.valid = false;
        }
        let cam = camera(params);
        let (misc, prev) = match mode {
            Mode::Temporal { jitter } => ([jitter[0], jitter[1], 0.0, 0.0], self.prev.unwrap_or(cam)),
            Mode::Still { weight } => ([0.0, 0.0, 1.0, weight.clamp(0.0, 1.0)], cam),
        };
        let u = TaaUniform {
            cam,
            prev,
            sizes: [render_size[0] as f32, render_size[1] as f32, size[0] as f32, size[1] as f32],
            misc,
            flags: [(self.valid && self.prev.is_some()) as u32 as f32, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let (read, write) = (self.current, 1 - self.current);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(render) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.hist[read]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.uniform.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hist[write],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.current = write;
        self.valid = true;
        self.prev = Some(cam);
        &self.hist[write]
    }

    /// The last resolved image.
    pub fn output(&self) -> Option<&wgpu::TextureView> {
        self.hist.get(self.current).filter(|_| self.valid)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn taa_shader_validates() {
        let src = include_str!("shaders/taa.wgsl");
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
    }
}
