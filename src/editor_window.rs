//! A separate OS window hosting the node editor (its own surface and egui context,
//! sharing the main GPU device). Redraws only when it has input or egui asks for a
//! repaint, so an idle editor window costs nothing.

use std::sync::Arc;
use std::time::Instant;

use egui_wgpu::wgpu;
use egui_winit::winit;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub struct EditorWindow {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: egui_wgpu::Renderer,
    state: egui_winit::State,
    needs_repaint: bool,
    repaint_at: Option<Instant>,
    /// egui texture updates not yet applied (kept across skipped frames).
    textures: egui::TexturesDelta,
}

impl EditorWindow {
    pub fn new(
        event_loop: &ActiveEventLoop,
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
    ) -> Result<Self, String> {
        let attrs = Window::default_attributes()
            .with_title("fractalful — node editor")
            .with_inner_size(winit::dpi::LogicalSize::new(1100, 600));
        let window = Arc::new(event_loop.create_window(attrs).map_err(|e| e.to_string())?);
        let surface = instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let caps = surface.get_capabilities(adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            // No vsync: presenting two vsynced swapchains per frame would halve the
            // main window's frame rate while the editor is being used.
            present_mode: wgpu::PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(device, &config);
        let renderer = egui_wgpu::Renderer::new(device, format, Default::default());
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        let state = egui_winit::State::new(
            ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        Ok(Self { window, surface, config, renderer, state, needs_repaint: true, repaint_at: None, textures: Default::default() })
    }

    pub fn on_event(&mut self, device: &wgpu::Device, event: &WindowEvent) {
        let _ = self.state.on_window_event(&self.window, event);
        if let WindowEvent::Resized(size) = event {
            if size.width > 0 && size.height > 0 {
                self.config.width = size.width;
                self.config.height = size.height;
                self.surface.configure(device, &self.config);
            }
        }
        self.needs_repaint = true;
    }

    pub fn ctx(&self) -> &egui::Context {
        self.state.egui_ctx()
    }

    /// Repaint now if there was input or egui scheduled one (animations, tooltips).
    pub fn wants_repaint(&self) -> bool {
        self.needs_repaint || self.repaint_at.is_some_and(|t| Instant::now() >= t)
    }

    pub fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mut ui: impl FnMut(&mut egui::Ui)) {
        self.needs_repaint = false;
        self.repaint_at = None;

        let raw_input = self.state.take_egui_input(&self.window);
        let ctx = self.state.egui_ctx().clone();
        let mut full_output = ctx.run_ui(raw_input, |ui_root| ui(ui_root));
        self.textures.append(std::mem::take(&mut full_output.textures_delta));
        self.state.handle_platform_output(&self.window, full_output.platform_output);
        if let Some(vp) = full_output.viewport_output.get(&egui::ViewportId::ROOT) {
            if vp.repaint_delay.is_zero() {
                self.needs_repaint = true;
            } else if vp.repaint_delay.as_secs() < 3600 {
                self.repaint_at = Some(Instant::now() + vp.repaint_delay);
            }
        }
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(device, &self.config);
                self.needs_repaint = true;
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in self.textures.set.drain() {
            for d in &delta {
                self.renderer.update_texture(device, queue, id, d);
            }
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        let extra = self.renderer.update_buffers(device, queue, &mut encoder, &paint_jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("editor window pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &paint_jobs, &screen);
        }
        for id in self.textures.free.drain() {
            self.renderer.free_texture(&id);
        }
        queue.submit(extra.into_iter().chain(std::iter::once(encoder.finish())));
        self.window.pre_present_notify();
        queue.present(frame);
    }
}
