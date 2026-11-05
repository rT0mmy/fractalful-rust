// Release builds open no console window (see platform.rs); debug builds keep one.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod anim;
mod bloom;
mod camera;
mod display;
mod editor_window;
mod audio;
mod graph;
mod images;
mod modal;
mod paths;
mod platform;
mod post;
mod profiler;
mod project;
mod renderer;
mod theme;
mod timeline;
mod toast;
mod taa;
mod video;
mod world;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_wgpu::wgpu;
use egui_winit::winit;
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use bloom::BloomSettings;
use camera::Camera;
use display::{Compositor, DisplayCaps, OutputMode, UI_FORMAT};
use editor_window::EditorWindow;
use graph::{Graph, GraphKind, Graphs};
use graph::editor::EditorState;
use graph::history::History;
use modal::{Action, Kind, Message, Modals};
use project::{Project, Session};
use renderer::{FractalRenderer, Params, save_png};
use toast::Toasts;
use video::{CameraMode, Studio, VideoFormat, VideoJob, keyframe_camera};

/// Shader debug visualizations; values must match `misc.z` handling in fractal.wgsl.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DebugView {
    Shaded = 0,
    Steps = 1,
    Normals = 2,
    Shadow = 3,
    Occlusion = 4,
}

impl DebugView {
    const ALL: [DebugView; 5] = [Self::Shaded, Self::Steps, Self::Normals, Self::Shadow, Self::Occlusion];

    fn label(self) -> &'static str {
        match self {
            Self::Shaded => "Shaded",
            Self::Steps => "Step-count heatmap",
            Self::Normals => "Normals",
            Self::Shadow => "Shadow only",
            Self::Occlusion => "Ambient occlusion only",
        }
    }
}

/// Camera speed range (world units per second): tiny for deep zooms into detail.
const MIN_SPEED: f32 = 1e-6;
const MAX_SPEED: f32 = 100.0;

const SKY_TOP: [f32; 3] = [0.10, 0.13, 0.20];
const SKY_BOTTOM: [f32; 3] = [0.02, 0.02, 0.035];

/// Render settings; saved with projects (see project.rs). Missing fields load as defaults.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
    iterations: u32,
    max_steps: u32,
    detail: f32,
    render_scale: f32,
    dynamic_resolution: bool,
    /// Temporal anti-aliasing / upscaling while the view changes (see taa.rs).
    taa: bool,
    target_fps: f32,
    vsync: bool,
    shadows: bool,
    shadow_steps: u32,
    shadow_iterations: u32,
    shadow_softness: f32,
    ao_strength: f32,
    /// Color of the built-in material (graphs without a Material Output).
    base_color: [f32; 3],
    reflections: bool,
    reflection_steps: u32,
    // Sky, light, fog and bloom from before the world and processing graphs replaced
    // them: read from old projects to build matching graphs (`Project::graphs`), never
    // written.
    #[serde(skip_serializing)]
    fog: f32,
    #[serde(skip_serializing)]
    sky_top: [f32; 3],
    #[serde(skip_serializing)]
    sky_bottom: [f32; 3],
    #[serde(skip_serializing)]
    sky_solid: bool,
    #[serde(skip_serializing)]
    light_intensity: f32,
    #[serde(skip_serializing)]
    bloom: BloomSettings,
    #[serde(skip_serializing)]
    light_yaw: f32,
    #[serde(skip_serializing)]
    light_pitch: f32,
    // Display-specific and diagnostic settings stay with the machine / session.
    #[serde(skip)]
    debug_view: DebugView,
    #[serde(skip)]
    output_mode: OutputMode,
    #[serde(skip)]
    paper_white_nits: f32,
    #[serde(skip)]
    peak_nits: f32,
    accumulate: bool,
    max_samples: u32,
    /// Camera speed in world units per second (scroll wheel adjusts it).
    move_speed: f32,
    /// Mouse-look multiplier.
    look_sensitivity: f32,
    /// Animation clock for driver nodes.
    playing: bool,
    time_speed: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            iterations: 10,
            max_steps: 160,
            detail: 1.0,
            render_scale: 1.0,
            dynamic_resolution: true,
            taa: true,
            target_fps: 60.0,
            vsync: true,
            shadows: true,
            shadow_steps: 32,
            shadow_iterations: 6,
            shadow_softness: 8.0,
            ao_strength: 1.5,
            fog: 0.02,
            base_color: [0.35, 0.45, 0.75],
            sky_top: SKY_TOP,
            sky_bottom: SKY_BOTTOM,
            sky_solid: false,
            reflections: true,
            reflection_steps: 96,
            light_intensity: 1.0,
            bloom: BloomSettings::default(),
            light_yaw: 0.8,
            light_pitch: 0.7,
            debug_view: DebugView::Shaded,
            output_mode: OutputMode::Sdr,
            paper_white_nits: 200.0,
            peak_nits: 1000.0,
            accumulate: true,
            max_samples: 64,
            move_speed: 0.5,
            look_sensitivity: 1.0,
            playing: true,
            time_speed: 1.0,
        }
    }
}

impl Settings {
    /// The look these settings had before the world and processing graphs existed.
    fn legacy_look(&self, shutter: f32) -> graph::LegacyLook {
        let b = &self.bloom;
        graph::LegacyLook {
            sky_top: self.sky_top,
            sky_bottom: self.sky_bottom,
            sky_solid: self.sky_solid,
            light_yaw: self.light_yaw,
            light_pitch: self.light_pitch,
            light_intensity: self.light_intensity,
            fog: self.fog,
            bloom_enabled: b.enabled,
            bloom: [b.intensity, b.threshold, b.knee, b.radius],
            shutter,
        }
    }
}

struct Gpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    fractal: FractalRenderer,
    compositor: Compositor,
    world: world::WorldRenderer,
    /// The world graph code the fractal shader was last built with (the last one that
    /// compiled, so a broken world graph doesn't take the fractal down with it).
    world_wgsl: String,
    post: post::PostChain,
    taa: taa::Taa,
    /// GPU time per stage, when the adapter supports timestamps.
    timer: Option<profiler::GpuTimer>,
    images: images::Images,
    caps: DisplayCaps,
    output_mode: OutputMode,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    /// egui texture updates not yet applied (kept across skipped frames).
    textures: egui::TexturesDelta,
}

impl Gpu {
    fn new(window: Arc<Window>, graph_wgsl: &str, world_code: &graph::codegen::WorldCode) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no suitable GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: profiler::GpuTimer::features(&adapter),
                ..Default::default()
            }))
            .expect("request device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let display = DisplayCaps::query(&surface, &adapter);
        let output_mode = OutputMode::Sdr;
        let (format, color_space) = display.surface_format(output_mode);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space,
        };
        surface.configure(&device, &config);

        let images = images::Images::new(&device, &queue);
        let world = world::WorldRenderer::new(&device, &images, world_code);
        let fractal = FractalRenderer::new(&device, &world.world_bgl, &world.light_bgl, graph_wgsl, &world_code.wgsl);
        let compositor = Compositor::new(&device, format);
        let post = post::PostChain::new(&device);
        let taa = taa::Taa::new(&device);
        let timer = profiler::GpuTimer::new(&device, &queue);
        // egui draws into its own sRGB layer; the compositor places it at paper white.
        let egui_renderer = egui_wgpu::Renderer::new(&device, UI_FORMAT, Default::default());
        let egui_ctx = egui::Context::default();
        theme::apply(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );

        log_adapter(&adapter);
        println!(
            "display: HDR {} (HDR10 {}, scRGB {}), peak {:?} nits, SDR white {:?} nits",
            if display.hdr_active { "on" } else { "off" },
            if display.supports(OutputMode::Hdr10) { "supported" } else { "unsupported" },
            if display.supports(OutputMode::ScRgb) { "supported" } else { "unsupported" },
            display.peak_nits,
            display.sdr_white_nits,
        );
        Self {
            instance,
            adapter,
            window,
            surface,
            device,
            queue,
            config,
            fractal,
            compositor,
            world,
            world_wgsl: world_code.wgsl.clone(),
            post,
            taa,
            timer,
            images,
            caps: display,
            output_mode,
            egui_renderer,
            egui_state,
            textures: egui::TexturesDelta::default(),
        }
    }

    /// Reconfigures the swapchain for SDR / HDR10 / scRGB. Unsupported modes fall back to SDR.
    fn set_output_mode(&mut self, mode: OutputMode) {
        let mode = if self.caps.supports(mode) { mode } else { OutputMode::Sdr };
        if mode == self.output_mode {
            return;
        }
        let (format, color_space) = self.caps.surface_format(mode);
        self.config.format = format;
        self.config.color_space = color_space;
        self.surface.configure(&self.device, &self.config);
        self.compositor.set_format(&self.device, format);
        self.output_mode = mode;
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn set_vsync(&mut self, vsync: bool) {
        let mode = if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        if self.config.present_mode != mode {
            self.config.present_mode = mode;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

fn log_adapter(adapter: &wgpu::Adapter) {
    let info = adapter.get_info();
    println!("GPU: {} ({:?})", info.name, info.backend);
}

#[derive(Default)]
struct App {
    gpu: Option<Gpu>,
    camera: Option<Camera>,
    settings: Settings,
    keys: HashSet<KeyCode>,
    /// Mouse look (RMB held): the cursor is hidden and confined, and raw mouse motion
    /// turns the camera; on release it reappears where it was.
    looking: bool,
    cursor_pos: Option<PhysicalPosition<f64>>,
    look_anchor: Option<PhysicalPosition<f64>>,
    last_frame: Option<Instant>,
    frame_ms: f32,
    surface_de: f32,
    /// Animation time in seconds (advances while playing).
    anim_time: f32,
    bench: Option<Bench>,
    exit_requested: bool,
    screenshot: Option<std::path::PathBuf>,
    /// Scene key of the previous frame; equal keys mean the image can keep accumulating.
    last_key: Option<Params>,
    /// Number of samples accumulated into the fractal target so far.
    samples: u32,
    last_graph_key: Option<(u64, Vec<f32>, u64, Vec<f32>, u64)>,
    /// What the world lighting / processing were last computed from (see `fingerprint`):
    /// they're only recomputed when it changes.
    last_world_key: Option<u64>,
    last_post_key: Option<u64>,
    /// Frames rendered while the view changes (picks the temporal jitter).
    taa_frame: u32,
    doc: GraphDoc,
    editor_window: Option<EditorWindow>,
    studio: Studio,
    timeline: timeline::Timeline,
    /// Animation time the camera was last placed on its keyframe path at; the path is
    /// re-applied when the time moves (play, scrub) or the keys change.
    follow_time: Option<f32>,
    path_dirty: bool,
    video_job: Option<VideoJob>,
    /// Command-line export (FRACTALFUL_EXPORT): start on the first frame, exit when done.
    auto_export: Option<bool>,
    toasts: Toasts,
    /// Screen-wide message boxes; while one is open the app ignores its own input.
    modals: Modals,
    /// A message box button chosen last frame, handled at the start of the next.
    modal_action: Option<Action>,
    /// When a screenshot was taken (the view flashes white for a moment).
    flash: Option<Instant>,
    /// First frame drawn: the view fades in from black.
    first_frame: Option<Instant>,
    restore_maximized: bool,
    /// The loaded song, its analysis and playback.
    audio: audio::Audio,
    /// Tab hides every panel for an unobstructed view.
    ui_hidden: bool,
    project: ProjectUi,
    /// Autosave the session and restore it at the next launch (off for the
    /// command-line test / export modes so they never touch the user's session).
    persist: bool,
    last_autosave: Option<Instant>,
    /// Last session JSON written, to skip identical rewrites.
    saved_session: String,
    /// Window size restored from the session, applied when the window is created.
    restore_window_size: Option<[u32; 2]>,
    title: String,
}

/// Project name and what the Project panel / Ctrl+S asked for this frame.
#[derive(Default)]
struct ProjectUi {
    /// Project (or preset) name: title bar, and the default file name when saving.
    name: String,
    /// Stem of the project file this scene was opened from or last saved to. Saving
    /// over any *other* existing file asks first.
    file: Option<String>,
    action: Option<ProjectAction>,
    /// Saved projects and when they were saved, refreshed when the Open menu opens.
    saved: Vec<(String, std::time::SystemTime)>,
}

enum ProjectAction {
    Save,
    Load(String),
}

const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(20);

/// What the Video export panel asked for this frame.
#[derive(PartialEq)]
enum VideoAction {
    None,
    Start,
    Cancel,
}

/// The node graphs plus their editor and compile state.
struct GraphDoc {
    graphs: Graphs,
    editor: EditorState,
    show_editor: bool,
    /// `structure_rev` of each graph (by `GraphKind::index`) currently compiled.
    compiled: [Option<u64>; 3],
    compiled_may_reflect: bool,
    /// Compile error of each graph (the previous version stays in use meanwhile).
    errors: [Option<String>; 3],
    /// Editor lives in its own OS window instead of inside the main one.
    detached: bool,
    /// Set by the UI; applied by the event loop, which can create windows.
    detach_request: Option<bool>,
    /// Fade the docked editor while the pointer isn't over it.
    fade_unfocused: bool,
    faded_opacity: f32,
    /// Docked editor window rect and engagement from the previous frame.
    editor_rect: Option<egui::Rect>,
    engaged: bool,
    history: History,
    /// (playhead bits, key fingerprint) the keyframed inputs were last written at.
    anim_applied: Option<(u32, u64)>,
    /// Re-apply the keyframed inputs next frame (after undo, loading a graph, ...).
    anim_dirty: bool,
}

impl GraphDoc {
    fn undo(&mut self, toasts: &mut Toasts) {
        self.anim_dirty = true;
        match self.history.undo(&mut self.graphs) {
            true => toasts.info("undo", "Undo"),
            false => toasts.info("undo", "Nothing to undo"),
        }
    }

    fn redo(&mut self, toasts: &mut Toasts) {
        self.anim_dirty = true;
        match self.history.redo(&mut self.graphs) {
            true => toasts.info("undo", "Redo"),
            false => toasts.info("undo", "Nothing to redo"),
        }
    }

    /// Replaces the material graph (a preset), keeping the revision increasing so it
    /// gets recompiled.
    fn replace_material(&mut self, g: Graph) {
        self.replace(GraphKind::Material, g);
    }

    fn replace(&mut self, kind: GraphKind, g: Graph) {
        let old = self.graphs.get_mut(kind);
        let rev = old.structure_rev + 1;
        *old = g;
        old.structure_rev = rev;
        self.anim_dirty = true;
        self.editor.reframe(kind);
    }

    /// Replaces every graph (opening a project).
    fn replace_graphs(&mut self, g: Graphs) {
        let Graphs { material, world, post } = g;
        self.replace(GraphKind::Material, material);
        self.replace(GraphKind::World, world);
        self.replace(GraphKind::Post, post);
    }
}

/// Ctrl+Z: undo, Ctrl+Y / Ctrl+Shift+Z: redo, in whichever window has keyboard focus
/// (text fields keep their own undo).
fn undo_shortcuts(ctx: &egui::Context, doc: &mut GraphDoc, toasts: &mut Toasts) {
    if ctx.text_edit_focused() {
        return;
    }
    let (z, y, shift) = ctx.input(|i| {
        (i.modifiers.command && i.key_pressed(egui::Key::Z), i.modifiers.command && i.key_pressed(egui::Key::Y), i.modifiers.shift)
    });
    if z && !shift {
        doc.undo(toasts);
    } else if y || z {
        doc.redo(toasts);
    }
}

impl Default for GraphDoc {
    fn default() -> Self {
        Self {
            graphs: Graphs::default(),
            editor: EditorState::default(),
            show_editor: false,
            compiled: [None; 3],
            compiled_may_reflect: false,
            errors: [None, None, None],
            detached: false,
            detach_request: None,
            fade_unfocused: true,
            faded_opacity: 0.3,
            editor_rect: None,
            engaged: false,
            history: History::default(),
            anim_applied: None,
            anim_dirty: true,
        }
    }
}

/// Node editor contents, shared by the docked egui window and the separate OS window.
/// `fade` is how far the docked editor is faded out (always 0 when detached).
fn editor_contents(ui: &mut egui::Ui, doc: &mut GraphDoc, toasts: &mut Toasts, fade: f32, time: f32) {
    ui.horizontal(|ui| {
        // One canvas per graph, like Blender's shader / world / compositor editors.
        for k in GraphKind::ALL {
            let failed = doc.errors[k.index()].is_some();
            let mut text = egui::RichText::new(k.label()).strong();
            if failed {
                text = text.color(theme::ERROR);
            }
            let hint = match k {
                GraphKind::Material => "The fractal: its shape (domain, iteration) and surface material",
                GraphKind::World => "Sky, sun, ambient light and fog",
                GraphKind::Post => "Image processing: bloom, curves, blur, lens effects, view transform",
            };
            ui.selectable_value(&mut doc.editor.mode, k, text).on_hover_text(hint);
        }
        let mode = doc.editor.mode;
        let names = match mode {
            GraphKind::World => graph::looks::WORLD_PRESETS,
            GraphKind::Post => graph::looks::POST_PRESETS,
            GraphKind::Material => &[],
        };
        if !names.is_empty() {
            ui.menu_button("Load preset", |ui| {
                for &name in names {
                    if ui.button(name).clicked() {
                        let g = if mode == GraphKind::World { graph::looks::world_preset(name) } else { graph::looks::post_preset(name) };
                        doc.replace(mode, g);
                        toasts.info("preset", format!("{} preset: {name}  (Ctrl+Z to go back)", mode.label()));
                        ui.close();
                    }
                }
            });
        }
        ui.separator();
        if ui.add_enabled(doc.history.can_undo(), egui::Button::new("Undo")).on_hover_text("Ctrl+Z").clicked() {
            doc.undo(toasts);
        }
        if ui.add_enabled(doc.history.can_redo(), egui::Button::new("Redo")).on_hover_text("Ctrl+Y / Ctrl+Shift+Z").clicked() {
            doc.redo(toasts);
        }
        ui.separator();
        if doc.detached {
            if ui.button("Dock").on_hover_text("Move the editor back into the main window").clicked() {
                doc.detach_request = Some(false);
            }
        } else {
            if ui.button("Pop out").on_hover_text("Move the editor into its own window").clicked() {
                doc.detach_request = Some(true);
            }
            ui.checkbox(&mut doc.fade_unfocused, "Fade when not hovered");
            ui.add_enabled(
                doc.fade_unfocused,
                egui::Slider::new(&mut doc.faded_opacity, 0.0..=1.0).text("faded opacity"),
            );
        }
        ui.separator();
        match &doc.errors[doc.editor.mode.index()] {
            Some(e) => ui.colored_label(egui::Color32::LIGHT_RED, format!("Shader error (previous graph still used): {e}")),
            None => ui.weak("Scroll: zoom   Shift+A: add   Right-drag: box select   Ctrl+right-drag: cut links   Home / F: frame"),
        };
    });
    graph::editor::show(ui, &mut doc.graphs, &mut doc.editor, fade, doc.faded_opacity, time);
    if let Some(n) = doc.editor.notice.take() {
        toasts.info("nodes", n);
    }
}

/// `FRACTALFUL_BENCH=1`: fixed camera, vsync off, full resolution.
/// Measures average frame time with shadows off, then on, prints both and exits.
#[derive(Default)]
struct Bench {
    frame: u32,
    sum_ms: [f64; 2],
}

impl Bench {
    const WARMUP: u32 = 60;
    const MEASURE: u32 = 300;
    const PHASE: u32 = Self::WARMUP + Self::MEASURE;
}

impl App {
    fn frame(&mut self) {
        let now = Instant::now();
        let dt = self.last_frame.map_or(1.0 / 60.0, |t| (now - t).as_secs_f32()).min(0.1);
        self.last_frame = Some(now);
        // Exponential moving average keeps the readout and the resolution controller stable.
        self.frame_ms = self.frame_ms * 0.9 + dt * 1000.0 * 0.1;
        if self.persist && self.video_job.is_none() && self.last_autosave.is_none_or(|t| t.elapsed() > AUTOSAVE_INTERVAL) {
            self.last_autosave = Some(now);
            self.save_session();
        }
        if let Some(result) = self.audio.poll() {
            match &result {
                Ok(msg) => println!("audio: {msg}"),
                Err(e) => eprintln!("audio: {e}"),
            }
            self.toasts.result("audio", result);
        }
        if let Some(action) = self.project.action.take() {
            match action {
                ProjectAction::Save => self.save_project(false),
                ProjectAction::Load(name) => self.load_project(&name),
            }
        }
        if let Some(action) = self.modal_action.take() {
            self.on_modal_action(action);
        }
        // A message box takes over input: stop flying and give the cursor back.
        if self.modals.is_open() {
            self.keys.clear();
            if self.looking {
                self.set_looking(false);
            }
        }

        let s = &mut self.settings;
        if let Some(b) = self.bench.as_mut() {
            let phase = (b.frame / Bench::PHASE) as usize;
            if phase >= 2 {
                let n = Bench::MEASURE as f64;
                println!("bench  shadows off: {:.2} ms", b.sum_ms[0] / n);
                println!("bench  shadows on:  {:.2} ms", b.sum_ms[1] / n);
                if let Some(t) = self.gpu.as_ref().and_then(|g| g.timer.as_ref()) {
                    let stages: Vec<String> = t.stages.iter().map(|(l, ms)| format!("{l} {ms:.2}")).collect();
                    println!("bench  gpu (shadows on): {:.2} ms = {}", t.total_ms(), stages.join(", "));
                }
                self.exit_requested = true;
                return;
            }
            if b.frame % Bench::PHASE >= Bench::WARMUP {
                b.sum_ms[phase] += dt as f64 * 1000.0;
            }
            if b.frame % Bench::PHASE == Bench::PHASE - 1 {
                let dir = std::env::var_os("FRACTALFUL_BENCH_SHOTS");
                if let Some(dir) = dir {
                    self.screenshot = Some(std::path::Path::new(&dir).join(format!("bench_{phase}.png")));
                }
            }
            b.frame += 1;
            s.vsync = false;
            s.dynamic_resolution = false;
            // FRACTALFUL_BENCH_SCALE=<0.25..1> renders below window resolution;
            // FRACTALFUL_TAA=0 turns temporal AA off.
            s.render_scale = std::env::var("FRACTALFUL_BENCH_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            s.taa = std::env::var("FRACTALFUL_TAA").as_deref() != Ok("0");
            s.shadows = phase == 1;
            s.accumulate = std::env::var_os("FRACTALFUL_BENCH_ACCUM").is_some();
        }

        // Recompile a graph only when its structure changed; parameter values are
        // uploaded every frame and never need a recompile.
        let doc = &mut self.doc;
        if let Some(gpu) = self.gpu.as_mut() {
            compile_graphs(gpu, doc);
        }
        let (bailout, bound, de_mode) = doc.graphs.material.output_settings();
        if s.playing && self.video_job.is_none() {
            self.anim_time += dt * s.time_speed;
            self.audio.follow(&mut self.anim_time);
            let v = &self.studio.settings;
            let (start, end) = (v.start, v.start + v.duration);
            if v.loop_playback && self.anim_time > end {
                self.anim_time = start;
            } else if v.loop_playback && s.time_speed < 0.0 && self.anim_time < start {
                self.anim_time = end;
            }
        }
        // Only feed time when a connected driver uses it, so static scenes keep
        // accumulating anti-aliasing samples instead of restarting every frame. (The
        // processing graph runs after accumulation, so its animation doesn't count.)
        let animated = doc.graphs.material.uses_time() || doc.graphs.world.uses_time();
        let time = match &self.video_job {
            Some(job) => job.sample_time(0, 1),
            None if animated => self.anim_time,
            None => 0.0,
        };
        // Keyframed node inputs: written into the graph when the playhead moves or the
        // keys change, so a value edited while paused stays (until then, or keyed).
        let anim_t = if self.video_job.is_some() { time } else { self.anim_time };
        let anim_key = (anim_t.to_bits(), doc.graphs.anim_fingerprint());
        if doc.graphs.is_animated() && (doc.anim_applied != Some(anim_key) || doc.anim_dirty) {
            doc.graphs.animate(anim_t);
        }
        doc.anim_applied = Some(anim_key);
        doc.anim_dirty = false;
        // Audio node outputs at this time go into the live slots of the graph uniform.
        let live = self.audio.live_values(&doc.graphs.material, time);
        let world_live = self.audio.live_values(&doc.graphs.world, time);
        let post_live = self.audio.live_values(&doc.graphs.post, anim_t);
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.fractal.write_graph_params(&gpu.queue, &doc.graphs.material.packed_values(), &live);
        }
        // Repeating scenes are unbounded: march up to the view distance instead.
        let clip = doc.graphs.material.unbounded_view_distance().map_or(bound, |d| -d);

        // Fly camera at the fixed speed set with the scroll wheel.
        let cam = self.camera.get_or_insert_with(Camera::new);
        // Camera animation: playing, scrubbing or editing keys puts the view on the
        // path; while paused and untouched, the camera can be flown off it (and keyed).
        let studio = &mut self.studio;
        let follow = self.video_job.is_none() && studio.camera_animated();
        if follow && (self.follow_time != Some(self.anim_time) || self.path_dirty) {
            if let Some(c) = keyframe_camera(&studio.keyframes, self.anim_time) {
                *cam = c;
            }
        }
        self.follow_time = Some(self.anim_time);
        self.path_dirty = false;
        self.surface_de = graph::eval::distance(&doc.graphs.material, cam.pos, s.iterations, time, &live).max(1e-6);
        let (fwd, right, _) = cam.basis();
        let mut mv = Vec3::ZERO;
        for (key, dir) in [
            (KeyCode::KeyW, fwd),
            (KeyCode::KeyS, -fwd),
            (KeyCode::KeyD, right),
            (KeyCode::KeyA, -right),
            (KeyCode::KeyE, Vec3::Y),
            (KeyCode::KeyQ, -Vec3::Y),
        ] {
            if self.keys.contains(&key) {
                mv += dir;
            }
        }
        // Ctrl+S etc. are shortcuts, not movement.
        let ctrl = self.keys.contains(&KeyCode::ControlLeft) || self.keys.contains(&KeyCode::ControlRight);
        // FRACTALFUL_BENCH_MOVE="speed,turn": the bench flies forward (units/s) and turns
        // (rad/s), to measure and look at rendering in motion.
        if let Some([speed, turn]) = self.bench.as_ref().and_then(|_| {
            let v: Vec<f32> = std::env::var("FRACTALFUL_BENCH_MOVE").ok()?.split(',').filter_map(|c| c.trim().parse().ok()).collect();
            Some([*v.first()?, v.get(1).copied().unwrap_or(0.0)])
        }) {
            let step = 1.0 / 60.0;
            cam.pos += fwd * speed * step;
            cam.yaw += turn * step;
        }
        if mv != Vec3::ZERO && self.video_job.is_none() && !ctrl {
            cam.pos += mv.normalize() * s.move_speed * dt;
        }
        // Auto-key: while paused, a view that left the path is keyed at the playhead.
        if follow && self.timeline.auto_key && !s.playing {
            let on_path = keyframe_camera(&studio.keyframes, self.anim_time).is_some_and(|p| same_view(&p, cam));
            if !on_path {
                studio.add_keyframe(self.anim_time, *cam);
            }
        }
        // During export the job drives the camera; the interactive one is left alone.
        let view_cam = match &self.video_job {
            Some(job) => job.camera_at(time),
            None => *cam,
        };
        let (fwd, right, up) = view_cam.basis();

        let mut params = Params {
            cam_pos: [view_cam.pos.x, view_cam.pos.y, view_cam.pos.z, (view_cam.fov_deg.to_radians() * 0.5).tan()],
            cam_fwd: [fwd.x, fwd.y, fwd.z, time],
            cam_right: [right.x, right.y, right.z, de_mode as f32],
            cam_up: [up.x, up.y, up.z, s.detail],
            resolution: [0.0, 0.0, s.max_steps as f32, s.iterations as f32],
            base_color: [s.base_color[0], s.base_color[1], s.base_color[2], s.ao_strength],
            misc: [s.shadows as u32 as f32, clip, s.debug_view as u32 as f32, bailout],
            shadow: [s.shadow_steps as f32, s.shadow_iterations.min(s.iterations) as f32, s.shadow_softness, bound],
            jitter: [0.0; 4],
            shade: [s.reflections as u32 as f32, s.reflection_steps as f32, 0.0, 0.0],
        };

        // Progressive refinement: while nothing changes, keep adding jittered samples
        // at full resolution; any change restarts at the (dynamic) moving resolution.
        let key = params.scene_key();
        // The world lights the fractal, so its values (and loaded images) count too.
        let images_version = self.gpu.as_ref().map_or(0, |g| g.images.version);
        let graph_key = (
            doc.graphs.material.structure_rev,
            doc.graphs.material.packed_values(),
            doc.graphs.world.structure_rev,
            doc.graphs.world.packed_values(),
            images_version,
        );
        let still = s.accumulate
            && self.last_key.is_some_and(|k| bytemuck::bytes_of(&k) == bytemuck::bytes_of(&key))
            && self.last_graph_key.as_ref() == Some(&graph_key);
        self.last_key = Some(key);
        self.last_graph_key = Some(graph_key);
        if !still {
            self.samples = 0;
            if s.dynamic_resolution {
                let target_ms = 1000.0 / s.target_fps;
                if self.frame_ms > target_ms * 1.05 {
                    s.render_scale -= 0.25 * dt;
                } else if self.frame_ms < target_ms * 0.8 {
                    s.render_scale += 0.1 * dt;
                }
                s.render_scale = s.render_scale.clamp(0.25, 1.0);
            }
        }

        let Some(gpu) = self.gpu.as_mut() else { return };
        gpu.set_vsync(s.vsync);
        gpu.set_output_mode(s.output_mode);
        s.output_mode = gpu.output_mode;

        // --- egui ---
        let raw_input = gpu.egui_state.take_egui_input(&gpu.window);
        let ctx = gpu.egui_state.egui_ctx().clone();
        let target_size = gpu.fractal.target_size();
        let stats = Stats {
            video: self.video_job.as_ref().map(|j| (j.frame, j.total, j.eta())),
            animated,
            frame_ms: self.frame_ms,
            gpu: gpu.timer.as_ref().map(|t| t.stages.clone()).unwrap_or_default(),
            surface_de: self.surface_de,
            target_size,
            samples: self.samples,
        };
        let mut video_action = VideoAction::None;
        // A command-line export waits for its song to finish analyzing.
        if self.auto_export == Some(false) && self.audio.loading().is_none() {
            self.auto_export = Some(true);
            video_action = VideoAction::Start;
        }
        // The song plays along only at normal speed (no time-stretching).
        let clock_ok = (s.time_speed - 1.0).abs() < 1e-3 && self.video_job.is_none();
        let anim_time = &mut self.anim_time;
        let audio = &mut self.audio;
        let (toasts, proj, ui_hidden) = (&mut self.toasts, &mut self.project, self.ui_hidden);
        let (modals, modal_action) = (&mut self.modals, &mut self.modal_action);
        // Keyboard shortcuts read raw input, which a message box doesn't block by itself.
        let blocked = modals.is_open();
        let tl = &mut self.timeline;
        let exporting = self.video_job.is_some();
        let mut keys_changed = false;
        let mut shot_clicked = false;
        let shot = &mut shot_clicked;
        let first_frame = *self.first_frame.get_or_insert(now);
        let flash = self.flash;
        let mut full_output = ctx.run_pass(raw_input, |ctx| {
            // The area left above the timeline; floating windows are kept inside it.
            let free = ctx.root_ui(|ui| {
                if !ui_hidden && tl.visible {
                    egui::Panel::bottom("timeline").resizable(true).default_size(150.0).show(ui, |ui| {
                        let changes = tl.show(
                            ui,
                            timeline::Ctx {
                                time: &mut *anim_time,
                                playing: &mut s.playing,
                                speed: &mut s.time_speed,
                                studio: &mut *studio,
                                graph: &mut doc.graphs,
                                cam: &*cam,
                                song: audio.track.as_ref().map(|t| (&t.analysis, audio.settings.offset)),
                                busy: exporting,
                            },
                        );
                        keys_changed |= changes.keys;
                        if let Some(m) = changes.message {
                            toasts.info("timeline", m);
                        }
                    });
                }
                ui.available_rect_before_wrap()
            });
            if !blocked {
                let changes = tl.shortcuts(
                    ctx,
                    timeline::Ctx {
                        time: &mut *anim_time,
                        playing: &mut s.playing,
                        speed: &mut s.time_speed,
                        studio: &mut *studio,
                        graph: &mut doc.graphs,
                        cam: &*cam,
                        song: None,
                        busy: exporting,
                    },
                );
                keys_changed |= changes.keys;
                if let Some(m) = changes.message {
                    toasts.info("timeline", m);
                }
            }
            if !ui_hidden && !exporting && tl.show_path {
                timeline::draw_path(ctx, &studio.keyframes, &view_cam, !studio.camera_animated());
            }
            if !ui_hidden {
                let panels = Panels {
                    s: &mut *s,
                    cam: &mut *cam,
                    st: &stats,
                    doc: &mut *doc,
                    caps: &gpu.caps,
                    anim_time: &mut *anim_time,
                    studio: &mut *studio,
                    toasts: &mut *toasts,
                    modals: &mut *modals,
                    proj: &mut *proj,
                    shot: &mut *shot,
                    audio: &mut *audio,
                    clock_ok,
                    timeline: &mut tl.visible,
                    area: free,
                };
                debug_window(ctx, panels, &mut video_action);
            }
            if !blocked {
                undo_shortcuts(ctx, doc, toasts);
                if !ctx.text_edit_focused() && ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::S)) {
                    proj.action = Some(ProjectAction::Save);
                }
            }
            screen_effects(ctx, first_frame, flash);
            if let Some(a) = modals.show(ctx) {
                *modal_action = Some(a);
            }
            toasts.show(ctx, free);
        });
        if shot_clicked {
            self.screenshot = Some(screenshot_path());
        }
        self.path_dirty |= keys_changed;
        // Auto-key: animated node values edited while paused are keyed at the playhead.
        if self.timeline.auto_key && !s.playing && self.video_job.is_none() {
            doc.graphs.autokey(self.anim_time);
        }
        self.audio.drive(self.anim_time, s.playing && clock_ok);
        // Snapshot the graph for undo once an edit has settled (not mid-drag / typing).
        let busy = |c: &egui::Context| c.input(|i| i.pointer.any_down()) || c.text_edit_focused();
        if !busy(&ctx) && !self.editor_window.as_ref().is_some_and(|ew| busy(ew.ctx())) {
            doc.history.commit(&doc.graphs);
        }
        match video_action {
            VideoAction::Start => match {
                // Motion blur comes from the processing graph (FRACTALFUL_EXPORT_SHUTTER overrides it).
                if std::env::var_os("FRACTALFUL_EXPORT_SHUTTER").is_none() {
                    studio.settings.shutter = doc.graphs.post.motion_blur_shutter();
                }
                VideoJob::start(studio, *cam, self.audio.soundtrack(studio.settings.start))
            } {
                Ok(job) => {
                    studio.status = None;
                    self.video_job = Some(job);
                }
                Err(e) => {
                    if self.auto_export.is_some() {
                        println!("export: {e}");
                        self.exit_requested = true;
                    } else {
                        self.modals.push(
                            Message::new("export", Kind::Error, "Couldn't start the export", "Nothing was rendered.")
                                .details(e.clone()),
                        );
                    }
                    studio.status = Some(Err(e));
                }
            },
            VideoAction::Cancel => {
                if let Some(job) = self.video_job.take() {
                    studio.status = Some(Ok(job.cancel()));
                }
            }
            VideoAction::None => {}
        }
        gpu.egui_state.handle_platform_output(&gpu.window, full_output.platform_output);
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        // Texture updates wait for a frame that draws (a skipped one would lose them).
        gpu.textures.append(std::mem::take(&mut full_output.textures_delta));

        // --- render ---
        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());

        let (w, h) = (gpu.config.width, gpu.config.height);
        // Quantize so the target isn't reallocated on every tiny scale change.
        let scale = if still { 1.0 } else { (s.render_scale * 20.0).round() / 20.0 };
        let rw = ((w as f32 * scale) as u32).max(1);
        let rh = ((h as f32 * scale) as u32).max(1);
        if gpu.fractal.ensure_target(&gpu.device, [rw, rh]) {
            self.samples = 0;
        }
        params.resolution[0] = rw as f32;
        params.resolution[1] = rh as f32;

        let sample = (self.samples < s.max_samples.max(1)).then_some(self.samples);
        // Temporal AA: debug views are shown raw, exports accumulate exactly.
        let temporal = s.taa && s.debug_view == DebugView::Shaded && self.video_job.is_none();
        if self.samples > 0 {
            let [jx, jy] = halton23(self.samples);
            params.jitter = [jx - 0.5, jy - 0.5, self.samples as f32, 0.0];
        } else if temporal && !still {
            // A new sub-pixel position every frame for the history to gather.
            self.taa_frame = self.taa_frame.wrapping_add(1);
            let [jx, jy] = halton23(self.taa_frame % 16 + 1);
            params.jitter = [jx - 0.5, jy - 0.5, 0.0, 0.0];
        }

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        if let Some(t) = gpu.timer.as_mut() {
            t.begin(&mut encoder);
        }
        // Image files: upload finished ones, bind the ones the graphs use now.
        for result in gpu.images.poll(&gpu.device, &gpu.queue) {
            self.toasts.result("image", result);
        }
        let world_files = graph::codegen::image_files(&doc.graphs.world);
        gpu.world.bind_images(&gpu.device, &mut gpu.images, &world_files);
        gpu.post.files = graph::codegen::image_files(&doc.graphs.post);
        for f in gpu.post.files.iter().flatten() {
            gpu.images.request(f);
        }
        // World lighting: recomputed when the world (or its time, audio, images) changes.
        let world_key = fingerprint(
            &[&doc.graphs.world.packed_values(), &world_live],
            &[doc.graphs.world.structure_rev, time.to_bits() as u64, gpu.images.version, doc.compiled[1].unwrap_or(0)],
        );
        if self.last_world_key != Some(world_key) && self.video_job.is_none() {
            gpu.world.write_params(&gpu.queue, &doc.graphs.world.packed_values(), &world_live, time);
            gpu.world.update(&mut encoder);
            self.last_world_key = Some(world_key);
        }
        if let Some(t) = gpu.timer.as_mut() {
            t.mark(&mut encoder, "world");
        }
        let sample = if let Some(job) = self.video_job.as_mut() {
            // Export: render one finished video frame (all its samples) this app frame;
            // the window shows it as a preview.
            self.samples = 0;
            self.last_world_key = None;
            self.last_post_key = None;
            // Each sample (motion blur) gets the graph and audio values for its own instant.
            let (graphs, audio) = (&doc.graphs, &self.audio);
            let upload = |gpu: &Gpu, t: f32, encoder: &mut wgpu::CommandEncoder| {
                gpu.fractal.write_graph_params(&gpu.queue, &graphs.material.packed_at(t), &audio.live_values(&graphs.material, t));
                gpu.world.write_params(&gpu.queue, &graphs.world.packed_at(t), &audio.live_values(&graphs.world, t), t);
                gpu.world.update(encoder);
            };
            let post = (&graphs.post, audio);
            let result = render_video_frame(gpu, job, &params, post, s.paper_white_nits, s.peak_nits, &upload);
            let finished = match result {
                Ok(()) if job.done() => Some(self.video_job.take().unwrap().finish()),
                Ok(()) => None,
                Err(e) => {
                    self.video_job.take().map(VideoJob::cancel);
                    Some(Err(e))
                }
            };
            if let Some(status) = finished {
                if self.auto_export.is_some() {
                    println!("export: {}", status.as_ref().unwrap_or_else(|e| e));
                    self.exit_requested = true;
                } else {
                    self.modals.push(export_message(&studio.settings, &status));
                }
                studio.status = Some(status);
            }
            None
        } else {
            gpu.fractal.render(&gpu.queue, &mut encoder, &params, gpu.world.bind_groups(), sample, false, gpu.timer.as_mut());
            if sample.is_some() {
                self.samples += 1;
            }
            sample
        };
        // Temporal resolve into the window-sized history (a converged still keeps it).
        let mut taa_ran = false;
        if !temporal {
            gpu.taa.invalidate();
        } else if sample.is_some() || gpu.taa.output().is_none() {
            let mode = if still {
                // Fade from the history over to the exact accumulation in a few samples.
                taa::Mode::Still { weight: (self.samples as f32 / 8.0).min(1.0) }
            } else {
                taa::Mode::Temporal { jitter: [params.jitter[0], params.jitter[1]] }
            };
            let render = gpu.fractal.target_view();
            gpu.taa.run(&gpu.device, &gpu.queue, &mut encoder, render, gpu.fractal.target_size(), [w, h], &params, mode);
            taa_ran = true;
        }
        if let Some(t) = gpu.timer.as_mut() {
            t.mark(&mut encoder, "temporal");
        }

        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: full_output.pixels_per_point };
        for (id, delta) in gpu.textures.set.drain() {
            for d in &delta {
                gpu.egui_renderer.update_texture(&gpu.device, &gpu.queue, id, d);
            }
        }
        let extra = gpu.egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &paint_jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: gpu.compositor.ui_layer(&gpu.device, [w, h]),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            gpu.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        for id in gpu.textures.free.drain() {
            gpu.egui_renderer.free_texture(&id);
        }
        if let Some(t) = gpu.timer.as_mut() {
            t.mark(&mut encoder, "ui");
        }
        // Processing graph: only re-run when the render or its own inputs changed; a
        // converged still costs nothing extra. Debug views are shown raw.
        let processed = s.debug_view == DebugView::Shaded;
        // The resolved history (window size) when temporal AA runs, else the render.
        let (source, size) = match gpu.taa.output().filter(|_| temporal) {
            Some(v) => (v, [w, h]),
            None => (gpu.fractal.target_view(), gpu.fractal.target_size()),
        };
        let post_key = fingerprint(
            &[&doc.graphs.post.packed_values(), &post_live],
            &[doc.graphs.post.structure_rev, anim_t.to_bits() as u64, gpu.images.version, ((size[0] as u64) << 32) | size[1] as u64],
        );
        if processed && self.video_job.is_none() && (sample.is_some() || taa_ran || self.last_post_key != Some(post_key)) {
            gpu.post.run(&gpu.device, &gpu.queue, &mut encoder, source, size, &gpu.images, &doc.graphs.post, anim_t, &post_live);
            self.last_post_key = Some(post_key);
        }
        if let Some(t) = gpu.timer.as_mut() {
            t.mark(&mut encoder, "processing");
        }
        let scene = display::Scene {
            view: if processed { gpu.post.output(source, &gpu.images) } else { source },
            size,
            view_transform: doc.graphs.post.view_transform(),
        };
        gpu.compositor.draw(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &scene,
            &view,
            [w, h],
            gpu.output_mode,
            s.paper_white_nits,
            s.peak_nits,
        );
        if let Some(t) = gpu.timer.as_mut() {
            t.mark(&mut encoder, "display");
            t.end(&mut encoder);
        }

        gpu.queue.submit(extra.into_iter().chain(std::iter::once(encoder.finish())));
        if let Some(t) = gpu.timer.as_mut() {
            t.collect(&gpu.device);
        }
        if let Some(path) = self.screenshot.take() {
            let size = scene.size;
            let rgba = gpu.compositor.render_offscreen(&gpu.device, &gpu.queue, &scene, size, display::Offscreen::Sdr);
            match save_png(&path, size, &rgba) {
                Ok(()) => {
                    println!("saved {}", path.display());
                    let refined = if self.samples > 1 { format!(", {} samples", self.samples) } else { String::new() };
                    let file = path.file_name().map_or(String::new(), |f| f.to_string_lossy().into_owned());
                    self.toasts.success("shot", format!("Screenshot saved: {file} ({}×{}{refined})", size[0], size[1]));
                    self.flash = Some(Instant::now());
                }
                Err(e) => {
                    eprintln!("screenshot failed: {e}");
                    self.toasts.error("shot", format!("Screenshot failed: {e}"));
                }
            }
        }
        gpu.window.pre_present_notify();
        gpu.queue.present(frame);

        if let Some(ew) = self.editor_window.as_mut().filter(|ew| ew.wants_repaint()) {
            let toasts = &mut self.toasts;
            let anim_time = self.anim_time;
            ew.draw(&gpu.device, &gpu.queue, |ui| {
                undo_shortcuts(ui.ctx(), doc, toasts);
                egui::CentralPanel::default().show(ui, |ui| editor_contents(ui, doc, toasts, 0.0, anim_time));
            });
        }

        let title = match self.project.name.as_str() {
            "" => "fractalful".to_string(),
            name => format!("{name} — fractalful"),
        };
        if title != self.title {
            gpu.window.set_title(&title);
            self.title = title;
        }
    }

    /// Applies a pop-out / dock request from the UI (needs the event loop to create windows).
    fn apply_detach_request(&mut self, event_loop: &ActiveEventLoop) {
        let Some(detach) = self.doc.detach_request.take() else { return };
        let Some(gpu) = self.gpu.as_ref() else { return };
        if detach && self.editor_window.is_none() {
            match EditorWindow::new(event_loop, &gpu.instance, &gpu.adapter, &gpu.device) {
                Ok(ew) => {
                    self.editor_window = Some(ew);
                    self.doc.detached = true;
                }
                Err(e) => eprintln!("could not open the node editor window: {e}"),
            }
        } else if !detach {
            self.editor_window = None;
            self.doc.detached = false;
            self.doc.show_editor = true;
        }
    }

    fn project_snapshot(&self) -> Project {
        let camera = self.camera.unwrap_or_else(|| start_camera(&self.doc.graphs.material));
        let mut p = Project::new(&self.doc.graphs, self.settings.clone(), camera);
        p.anim_time = self.anim_time;
        p.video = self.studio.settings.clone();
        p.keyframes = self.studio.keyframes.clone();
        p.audio = self.audio.settings.clone();
        p
    }

    fn apply_project(&mut self, p: Project) {
        self.doc.replace_graphs(p.graphs());
        // Display settings belong to this machine, the debug view to this session.
        let s = &self.settings;
        self.settings = Settings {
            debug_view: s.debug_view,
            output_mode: s.output_mode,
            paper_white_nits: s.paper_white_nits,
            peak_nits: s.peak_nits,
            ..p.settings
        };
        self.camera = Some(p.camera);
        self.anim_time = p.anim_time;
        self.studio.settings = p.video;
        self.studio.keyframes = p.keyframes;
        self.audio.apply_settings(p.audio);
    }

    /// Saves under the project name. Replacing a saved project other than the one this
    /// scene came from asks first, unless `force`.
    fn save_project(&mut self, force: bool) {
        if self.project.name.trim().is_empty() {
            self.project.name = "untitled".into();
        }
        let path = project::project_path(&self.project.name);
        let stem = project::file_stem(&self.project.name);
        if !force && path.exists() && self.project.file.as_deref() != Some(stem.as_str()) {
            self.modals.push(
                Message::new(
                    "overwrite",
                    Kind::Warning,
                    format!("Replace \"{stem}\"?"),
                    "A saved project with this name already exists. Replacing it overwrites its graph, \
                     camera, keyframes and settings with this scene.",
                )
                .button("Cancel", Action::Close)
                .danger("Replace", Action::OverwriteProject),
            );
            return;
        }
        let result = project::write(&path, &project::to_json(&self.project_snapshot()));
        if result.is_ok() {
            self.project.file = Some(stem);
        }
        self.toasts.result("project", result.map(|()| format!("Saved {}", path.display())));
    }

    fn load_project(&mut self, name: &str) {
        self.load_project_file(&project::project_path(name));
    }

    fn load_project_file(&mut self, path: &std::path::Path) {
        match project::read::<Project>(path) {
            Ok(p) => {
                let name = path.file_stem().map_or(String::new(), |n| n.to_string_lossy().into_owned());
                self.apply_project(p);
                self.toasts.success("project", format!("Opened {name}"));
                // Only files in the projects folder are "this project" for saving.
                let in_projects = path.parent().is_some_and(|d| d == paths::projects_dir());
                self.project.file = in_projects.then(|| name.clone());
                self.project.name = name;
            }
            Err(e) => self.modals.push(
                Message::new("open", Kind::Error, "Couldn't open the project", "The file couldn't be read as a fractalful project.")
                    .details(e),
            ),
        }
    }

    /// Runs what a message box button asked for.
    fn on_modal_action(&mut self, action: Action) {
        let result = match action {
            Action::Close => Ok(()),
            Action::ReconnectOutputs => {
                self.doc.undo(&mut self.toasts);
                self.toasts.success("undo", "Outputs reconnected");
                Ok(())
            }
            Action::Open(path) => platform::open(&path),
            Action::Reveal(path) => platform::reveal(&path),
            Action::OverwriteProject => {
                self.save_project(true);
                Ok(())
            }
            Action::Quit => {
                if let Some(job) = self.video_job.take() {
                    self.studio.status = Some(Ok(job.cancel()));
                }
                if self.persist {
                    self.save_session();
                }
                self.exit_requested = true;
                Ok(())
            }
        };
        if let Err(e) = result {
            self.toasts.error("open", e);
        }
    }

    /// Writes the session file if anything changed since the last write.
    fn save_session(&mut self) {
        let window = self.gpu.as_ref().map(|g| &g.window);
        let maximized = window.map_or(self.restore_maximized, |w| w.is_maximized());
        // Maximized / fullscreen sizes are the screen's: keep the size it restores to
        // (tracked on resize, see `window_event`).
        let window_size = self.restore_window_size;
        let session = Session {
            project: self.project_snapshot(),
            name: self.project.name.clone(),
            window_size,
            show_editor: self.doc.show_editor,
            editor_detached: self.doc.detached,
            maximized,
            file: self.project.file.clone(),
        };
        let json = project::to_json(&session);
        if json != self.saved_session {
            match project::write(&project::session_path(), &json) {
                Ok(()) => self.saved_session = json,
                Err(e) => eprintln!("session: {e}"),
            }
        }
    }

    /// Restores the last session, if there is one (before the window exists). After an
    /// unclean exit (crash, or a hang ended from Task Manager) the graph is probably what
    /// took the app down, so it comes back with its outputs cut (every node is kept, and
    /// the message box offers to reconnect them).
    fn restore_session(&mut self, last_run: project::LastRun) {
        let crashed = match last_run {
            project::LastRun::Unclean(started) => Some(started),
            _ => None,
        };
        let path = project::session_path();
        if !path.exists() {
            if let Some(started) = crashed {
                self.modals.push(crash_message(0, started.and_then(platform::panic_since), None));
            }
            return;
        }
        match project::read::<Session>(&path) {
            Ok(session) => {
                self.apply_project(session.project);
                self.project.name = session.name;
                self.project.file = session.file;
                self.restore_window_size = session.window_size;
                self.restore_maximized = session.maximized;
                self.doc.show_editor = session.show_editor;
                if session.editor_detached {
                    self.doc.detach_request = Some(true);
                }
                if let Some(started) = crashed {
                    let backup = project::back_up_session("crashed");
                    // The intact graph becomes the undo step before the cut.
                    self.doc.history.commit(&self.doc.graphs);
                    let cut = self.doc.graphs.material.cut_outputs();
                    self.modals.push(crash_message(cut, started.and_then(platform::panic_since), backup));
                }
            }
            Err(e) => {
                eprintln!("session: {e}");
                let backup = project::back_up_session("unreadable");
                let mut text = e;
                if let Some(b) = backup {
                    text += &format!("\n\nThe file was copied to {}.", b.display());
                }
                self.modals.push(
                    Message::new(
                        "session",
                        Kind::Error,
                        "Couldn't restore the last session",
                        "The saved session couldn't be read, so fractalful started with the default scene.",
                    )
                    .details(text),
                );
            }
        }
    }

    /// Starts / ends mouse look: hides and confines the cursor (camera turns by raw
    /// mouse motion, see `device_event`), then puts it back where it was.
    fn set_looking(&mut self, on: bool) {
        if on == self.looking {
            return;
        }
        self.looking = on;
        let Some(gpu) = self.gpu.as_ref() else { return };
        let w = &gpu.window;
        if on {
            self.look_anchor = self.cursor_pos;
            if w.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                let _ = w.set_cursor_grab(CursorGrabMode::Confined);
            }
            w.set_cursor_visible(false);
        } else {
            let _ = w.set_cursor_grab(CursorGrabMode::None);
            w.set_cursor_visible(true);
            if let Some(p) = self.look_anchor.take() {
                let _ = w.set_cursor_position(p);
            }
        }
    }
}

/// Recompiles the graphs whose structure changed. The world compiles first: the fractal
/// shader embeds it (and keeps the last world code that compiled, so a broken world graph
/// shows its error without taking the fractal down).
fn compile_graphs(gpu: &mut Gpu, doc: &mut GraphDoc) {
    let gs = &doc.graphs;
    let (m, w, p) = (GraphKind::Material.index(), GraphKind::World.index(), GraphKind::Post.index());
    let world_changed = doc.compiled[w] != Some(gs.world.structure_rev);
    if world_changed {
        doc.compiled[w] = Some(gs.world.structure_rev);
        let result = graph::codegen::generate_world(&gs.world).and_then(|code| gpu.world.set_graph(&gpu.device, &code).map(|()| code));
        match result {
            Ok(code) => {
                doc.errors[w] = None;
                gpu.world_wgsl = code.wgsl;
            }
            Err(e) => doc.errors[w] = Some(e),
        }
    }
    // Roughness crossing the reflection cutoff changes the specialized shader.
    let may_reflect = gs.material.may_reflect();
    if world_changed || doc.compiled[m] != Some(gs.material.structure_rev) || doc.compiled_may_reflect != may_reflect {
        doc.compiled[m] = Some(gs.material.structure_rev);
        doc.compiled_may_reflect = may_reflect;
        doc.errors[m] = graph::codegen::generate(&gs.material)
            .and_then(|wgsl| gpu.fractal.set_graph(&gpu.device, &wgsl, &gpu.world_wgsl))
            .err();
    }
    if doc.compiled[p] != Some(gs.post.structure_rev) {
        doc.compiled[p] = Some(gs.post.structure_rev);
        doc.errors[p] = graph::codegen::generate_post(&gs.post).and_then(|plan| gpu.post.set_plan(&gpu.device, plan)).err();
    }
}

/// A hash of `floats` (by their bits) and `ints`, to tell whether a pass's inputs changed.
fn fingerprint(floats: &[&[f32]], ints: &[u64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for f in floats {
        f.len().hash(&mut h);
        for v in *f {
            v.to_bits().hash(&mut h);
        }
    }
    ints.hash(&mut h);
    h.finish()
}

/// Points `params` at `cam` and animation time `time` (other fields untouched).
fn apply_camera(params: &mut Params, cam: &Camera, time: f32) {
    let (fwd, right, up) = cam.basis();
    params.cam_pos = [cam.pos.x, cam.pos.y, cam.pos.z, (cam.fov_deg.to_radians() * 0.5).tan()];
    params.cam_fwd = [fwd.x, fwd.y, fwd.z, time];
    params.cam_right = [right.x, right.y, right.z, params.cam_right[3]];
    params.cam_up = [up.x, up.y, up.z, params.cam_up[3]];
}

/// Renders the job's current frame at the video resolution: accumulates all samples
/// (spreading time and camera over the shutter for motion blur), runs the processing
/// graph, composites offscreen and hands the pixels to the encoder. `graph_params_at`
/// uploads the graphs' values for an instant (and updates the world lighting).
#[allow(clippy::too_many_arguments)]
fn render_video_frame(
    gpu: &mut Gpu,
    job: &mut VideoJob,
    base: &Params,
    (post, audio): (&Graph, &audio::Audio),
    paper_white: f32,
    peak: f32,
    graph_params_at: &dyn Fn(&Gpu, f32, &mut wgpu::CommandEncoder),
) -> Result<(), String> {
    let vs = job.settings.clone();
    let size = [vs.width, vs.height];
    gpu.fractal.ensure_target(&gpu.device, size);
    let samples = vs.samples.max(1);
    let blur = vs.shutter > 0.0 && samples > 1;
    for n in 0..samples {
        let t = if blur { job.sample_time(n, samples) } else { job.sample_time(0, 1) };
        let mut p = *base;
        apply_camera(&mut p, &job.camera_at(t), t);
        p.resolution[0] = size[0] as f32;
        p.resolution[1] = size[1] as f32;
        p.jitter = if n == 0 {
            [0.0; 4]
        } else {
            let [jx, jy] = halton23(n);
            [jx - 0.5, jy - 0.5, n as f32, 0.0]
        };
        // One submit per sample: each needs its own uniform values.
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        graph_params_at(gpu, t, &mut encoder);
        gpu.fractal.render(&gpu.queue, &mut encoder, &p, gpu.world.bind_groups(), Some(n), blur && job.camera_moves(), None);
        gpu.queue.submit([encoder.finish()]);
    }
    let t = job.sample_time(0, 1);
    let live = audio.live_values(post, t);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let render = gpu.fractal.target_view();
    gpu.post.run(&gpu.device, &gpu.queue, &mut encoder, render, size, &gpu.images, post, t, &live);
    gpu.queue.submit([encoder.finish()]);
    let scene = display::Scene {
        view: gpu.post.output(gpu.fractal.target_view(), &gpu.images),
        size,
        view_transform: post.view_transform(),
    };
    let kind = if vs.format.is_hdr() {
        display::Offscreen::Hdr10 { paper_white, peak }
    } else {
        display::Offscreen::Sdr
    };
    let pixels = gpu.compositor.render_offscreen(&gpu.device, &gpu.queue, &scene, size, kind);
    job.push_frame(&pixels)
}

fn video_panel(ui: &mut egui::Ui, studio: &mut Studio, timeline_visible: &mut bool, st: &Stats, shutter: f32, action: &mut VideoAction) {
    if let Some((frame, total, eta)) = st.video {
        ui.add(egui::ProgressBar::new(frame as f32 / total as f32).show_percentage());
        let eta = eta.map_or("estimating…".into(), |s| format!("{:.0}:{:02} left", s as u32 / 60, s as u32 % 60));
        ui.label(format!("frame {frame} / {total}, {eta}"));
        if ui.button("Cancel").clicked() {
            *action = VideoAction::Cancel;
        }
        return;
    }
    let v = &mut studio.settings;
    ui.horizontal(|ui| {
        ui.label("size");
        ui.add(egui::DragValue::new(&mut v.width).range(16..=7680));
        ui.label("×");
        ui.add(egui::DragValue::new(&mut v.height).range(16..=4320));
        for (name, w, h) in [("720p", 1280, 720), ("1080p", 1920, 1080), ("4K", 3840, 2160)] {
            if ui.small_button(name).clicked() {
                (v.width, v.height) = (w, h);
            }
        }
    });
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut v.fps).range(1..=240).suffix(" fps"));
        ui.add(egui::DragValue::new(&mut v.duration).range(0.1..=3600.0).speed(0.1).suffix(" s"));
        ui.label("from");
        ui.add(egui::DragValue::new(&mut v.start).range(0.0..=3600.0).speed(0.1).suffix(" s"));
    });
    ui.add(egui::Slider::new(&mut v.samples, 1..=256).logarithmic(true).text("samples / frame"));
    if shutter > 0.0 {
        ui.small(format!("Motion blur: shutter {shutter:.2} (Motion Blur node in the Processing graph)"));
    } else {
        ui.small("Motion blur: off (add a Motion Blur node to the Processing graph)");
    }
    egui::ComboBox::from_label("format").selected_text(v.format.label()).show_ui(ui, |ui| {
        for f in VideoFormat::ALL {
            ui.selectable_value(&mut v.format, f, f.label());
        }
    });
    if v.format != VideoFormat::PngSequence {
        ui.add(egui::Slider::new(&mut v.crf, 0..=40).text("quality (CRF, lower = better)"));
    }
    ui.horizontal(|ui| {
        ui.label("output");
        ui.text_edit_singleline(&mut v.output);
    });
    ui.small(format!("→ {}", v.output_path().display()));
    if v.format != VideoFormat::PngSequence {
        ui.checkbox(&mut v.include_audio, "Include the song").on_hover_text("When a song is loaded (Audio panel)");
    }

    ui.separator();
    egui::ComboBox::from_label("camera").selected_text(v.camera.label()).show_ui(ui, |ui| {
        for m in CameraMode::ALL {
            ui.selectable_value(&mut v.camera, m, m.label());
        }
    });
    match v.camera {
        CameraMode::Orbit => {
            ui.add(egui::Slider::new(&mut v.orbit_turns, -3.0..=3.0).text("turns"));
            ui.small("Orbits the origin from the current camera position.");
        }
        CameraMode::Keyframes => {
            let n = studio.keyframes.len();
            ui.small(format!("{n} camera key{}: edit them in the timeline (I inserts one at the playhead).", if n == 1 { "" } else { "s" }));
            if !*timeline_visible && ui.button("Show timeline").clicked() {
                *timeline_visible = true;
            }
        }
        CameraMode::Current => {}
    }
    let v = &studio.settings;
    ui.separator();
    ui.small(format!(
        "{} frames × {} samples, animation time {:.2} → {:.2} s",
        v.frame_count(),
        v.samples,
        v.start,
        v.start + v.duration
    ));
    if ui.button("Render video").clicked() {
        *action = VideoAction::Start;
    }
    match &studio.status {
        Some(Ok(msg)) => {
            ui.colored_label(egui::Color32::LIGHT_GREEN, msg);
        }
        Some(Err(e)) => {
            ui.colored_label(egui::Color32::LIGHT_RED, e);
        }
        None => {}
    }
}

/// Low-discrepancy 2D sequence for sub-pixel jitter (Halton bases 2 and 3), in [0, 1).
fn halton23(i: u32) -> [f32; 2] {
    let radical = |mut i: u32, base: u32| {
        let (mut f, mut r) = (1.0f32, 0.0f32);
        while i > 0 {
            f /= base as f32;
            r += f * (i % base) as f32;
            i /= base;
        }
        r
    };
    [radical(i, 2), radical(i, 3)]
}

struct Stats {
    /// GPU milliseconds per stage of the frame (empty without timestamp support).
    gpu: Vec<(&'static str, f32)>,
    /// Export progress: (frame, total, seconds remaining).
    video: Option<(u32, u32, Option<f32>)>,
    animated: bool,
    frame_ms: f32,
    surface_de: f32,
    target_size: [u32; 2],
    samples: u32,
}

/// Everything the main window's panels edit.
struct Panels<'a> {
    s: &'a mut Settings,
    cam: &'a mut Camera,
    st: &'a Stats,
    doc: &'a mut GraphDoc,
    caps: &'a DisplayCaps,
    anim_time: &'a mut f32,
    studio: &'a mut Studio,
    toasts: &'a mut Toasts,
    modals: &'a mut Modals,
    proj: &'a mut ProjectUi,
    /// Set by the screenshot button.
    shot: &'a mut bool,
    audio: &'a mut audio::Audio,
    /// The song can play along (animation speed 1, not exporting).
    clock_ok: bool,
    /// The timeline bar is shown.
    timeline: &'a mut bool,
    /// Screen area not covered by the timeline: windows stay inside it.
    area: egui::Rect,
}

fn debug_window(ctx: &egui::Context, panels: Panels, video_action: &mut VideoAction) {
    let Panels { s, cam, st, doc, caps, anim_time, studio, toasts, modals, proj, shot, audio, clock_ok, timeline, area } = panels;
    if !doc.detached {
        // "Engaged": pointer over the editor, or still dragging / typing / in a menu
        // after starting there. While not engaged, the editor fades out.
        let hovered = doc.editor_rect.zip(ctx.pointer_hover_pos()).is_some_and(|(r, p)| r.contains(p));
        let busy = ctx.input(|i| i.pointer.any_down()) || ctx.memory(|m| m.focused().is_some());
        doc.engaged = hovered || (doc.engaged && busy) || egui::Popup::is_any_open(ctx);
        let fade = if doc.fade_unfocused { ctx.animate_bool(egui::Id::unique("node editor fade"), !doc.engaged) } else { 0.0 };

        let style = ctx.global_style();
        let frame = egui::Frame::window(&style)
            .fill(style.visuals.window_fill.gamma_multiply(1.0 - fade))
            .shadow(if fade > 0.5 { egui::Shadow::NONE } else { style.visuals.window_shadow });
        let mut open = doc.show_editor;
        let response = egui::Window::new("Node editor")
            .open(&mut open)
            .constrain_to(area)
            .frame(frame)
            .default_size([900.0, 420.0])
            .resizable(true)
            .show(ctx, |ui| editor_contents(ui, doc, toasts, fade, *anim_time));
        doc.show_editor = open;
        doc.editor_rect = response.map(|r| r.response.rect);
    }

    egui::Window::new("Controls").default_width(280.0).constrain_to(area).show(ctx, |ui| {
        let fps = 1000.0 / st.frame_ms.max(0.001);
        // Green at the target frame rate or better, amber when close, red when struggling.
        let target = if s.dynamic_resolution { s.target_fps } else { 60.0 };
        let fps_color = match fps / target {
            r if r >= 0.95 => theme::SUCCESS,
            r if r >= 0.6 => theme::WARNING,
            _ => theme::ERROR,
        };
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(format!("{fps:.0} fps")).color(fps_color).strong());
            ui.weak(format!("{:.2} ms", st.frame_ms));
            if !st.gpu.is_empty() {
                let total: f32 = st.gpu.iter().map(|g| g.1).sum();
                let detail: Vec<String> = st.gpu.iter().map(|(l, ms)| format!("{l}: {ms:.2} ms")).collect();
                ui.weak(format!("GPU {total:.2} ms")).on_hover_text(detail.join("\n"));
            }
        });
        ui.label(format!("render target: {}x{}", st.target_size[0], st.target_size[1]));
        if s.accumulate {
            let done = st.samples as f32 / s.max_samples.max(1) as f32;
            let bar = egui::ProgressBar::new(done.min(1.0)).desired_height(6.0).desired_width(ui.available_width());
            ui.add(bar).on_hover_text(format!("Refinement: {} of {} samples", st.samples, s.max_samples));
        } else {
            ui.label(format!("samples: {}/{}", st.samples, s.max_samples));
        }
        ui.label(format!("distance to surface: {:.2e}", st.surface_de));
        ui.label(format!("pos: {:.5} {:.5} {:.5}", cam.pos.x, cam.pos.y, cam.pos.z));
        ui.horizontal(|ui| {
            if ui.button("Screenshot").on_hover_text("F12").clicked() {
                *shot = true;
            }
            if ui.button("Shortcuts").on_hover_text("F1").clicked() {
                modals.push(shortcuts_message());
            }
            ui.weak("Tab: hide UI");
        });

        ui.collapsing("Project", |ui| project_panel(ui, proj));
        ui.collapsing("Performance", |ui| {
            ui.checkbox(&mut s.vsync, "VSync");
            ui.checkbox(&mut s.accumulate, "Progressive refinement (AA when still)");
            ui.add_enabled(s.accumulate, egui::Slider::new(&mut s.max_samples, 1..=256).text("max samples"));
            ui.checkbox(&mut s.dynamic_resolution, "Dynamic resolution");
            ui.checkbox(&mut s.taa, "Temporal AA + upscaling").on_hover_text(
                "While the view changes, each frame adds jittered samples to a history at full resolution: \
                 detail stays stable instead of shimmering, and dynamic resolution can drop further without blur",
            );
            ui.add_enabled(s.dynamic_resolution, egui::Slider::new(&mut s.target_fps, 30.0..=240.0).text("target fps"));
            ui.add_enabled(!s.dynamic_resolution, egui::Slider::new(&mut s.render_scale, 0.25..=1.0).text("render scale"));
            ui.add(egui::Slider::new(&mut s.max_steps, 16..=512).text("max steps"));
            ui.add(egui::Slider::new(&mut s.detail, 0.25..=8.0).logarithmic(true).text("LOD (px per hit)"));
            egui::ComboBox::from_label("view")
                .selected_text(s.debug_view.label())
                .show_ui(ui, |ui| {
                    for v in DebugView::ALL {
                        ui.selectable_value(&mut s.debug_view, v, v.label());
                    }
                });
        });
        ui.collapsing("Animation", |ui| {
            ui.checkbox(timeline, "Timeline (bottom bar)").on_hover_text("Transport, scene range and camera keyframes");
            ui.horizontal(|ui| {
                let icon = if s.playing { timeline::Icon::Pause } else { timeline::Icon::Play };
                if timeline::icon_button(ui, icon).on_hover_text("Play / pause (Space)").clicked() {
                    s.playing = !s.playing;
                }
                ui.label("t =");
                ui.add(egui::DragValue::new(anim_time).speed(0.02).suffix(" s").max_decimals(2));
                if timeline::icon_button(ui, timeline::Icon::ToStart).on_hover_text("Back to 0").clicked() {
                    *anim_time = 0.0;
                }
            });
            ui.add(egui::Slider::new(&mut s.time_speed, -4.0..=4.0).text("speed"));
            if !st.animated {
                ui.small("No driver nodes connected: the image is static.");
            }
        });
        egui::CollapsingHeader::new("Audio")
            .open(std::mem::take(&mut audio.reveal).then_some(true))
            .show(ui, |ui| audio::ui::panel(ui, audio, anim_time, clock_ok));
        let shutter = doc.graphs.post.motion_blur_shutter();
        ui.collapsing("Video export", |ui| video_panel(ui, studio, timeline, st, shutter, video_action));
        ui.collapsing("Fractal", |ui| {
            ui.horizontal(|ui| {
                ui.toggle_value(&mut doc.show_editor, "Node editor (N)");
                ui.menu_button("Load preset", |ui| {
                    // One submenu per category; the current scene's preset is highlighted.
                    for (category, names) in graph::PRESET_CATEGORIES {
                        let current = names.contains(&proj.name.as_str()) && proj.file.is_none();
                        let label = egui::RichText::new(format!("{category}  ({})", names.len()));
                        ui.menu_button(if current { label.strong() } else { label }, |ui| {
                            for &name in *names {
                                let selected = proj.file.is_none() && proj.name == name;
                                if ui.selectable_label(selected, name).clicked() {
                                    let (g, iters) = graph::preset(name);
                                    doc.replace_material(g);
                                    s.iterations = iters;
                                    *cam = start_camera(&doc.graphs.material);
                                    proj.name = name.to_string();
                                    proj.file = None;
                                    toasts.info("preset", format!("{name}  (Ctrl+Z to go back)"));
                                    ui.close();
                                }
                            }
                        });
                    }
                });
            });
            ui.add(egui::Slider::new(&mut s.iterations, 1..=64).text("iterations"));
            for k in GraphKind::ALL {
                if let Some(e) = &doc.errors[k.index()] {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("{} graph failed to compile (see node editor)", k.label()));
                    ui.small(e.lines().next().unwrap_or_default());
                }
            }
        });
        ui.collapsing("Display", |ui| {
            egui::ComboBox::from_label("output")
                .selected_text(s.output_mode.label())
                .show_ui(ui, |ui| {
                    for m in OutputMode::ALL {
                        ui.add_enabled_ui(caps.supports(m), |ui| {
                            ui.selectable_value(&mut s.output_mode, m, m.label());
                        });
                    }
                });
            let hdr = s.output_mode != OutputMode::Sdr;
            ui.add_enabled(hdr, egui::Slider::new(&mut s.paper_white_nits, 80.0..=500.0).text("paper white (nits)"));
            ui.add_enabled(
                hdr,
                egui::Slider::new(&mut s.peak_nits, 100.0..=10000.0).logarithmic(true).text("peak (nits)"),
            );
            if s.peak_nits < s.paper_white_nits {
                s.peak_nits = s.paper_white_nits;
            }
            let fmt = |n: Option<f32>| n.map_or("unknown".to_string(), |n| format!("{n:.0}"));
            ui.small(format!(
                "Display: HDR {}, peak {} nits, SDR white {} nits",
                if caps.hdr_active { "on" } else { "off" },
                fmt(caps.peak_nits),
                fmt(caps.sdr_white_nits),
            ));
            if !caps.hdr_active {
                ui.small("Turn on \"Use HDR\" in Windows display settings for HDR output.");
            }
        });
        ui.collapsing("Shading", |ui| {
            ui.horizontal(|ui| {
                for (k, what) in [(GraphKind::World, "Sky & light…"), (GraphKind::Post, "Effects…")] {
                    let hint = match k {
                        GraphKind::World => "Sky, sun, ambient light and fog: the World graph of the node editor",
                        _ => "Bloom, curves, blur, lens effects, view transform: the Processing graph of the node editor",
                    };
                    if ui.button(what).on_hover_text(hint).clicked() {
                        doc.editor.mode = k;
                        doc.show_editor = true;
                    }
                }
            });
            ui.checkbox(&mut s.shadows, "Soft shadows");
            ui.add_enabled_ui(s.shadows, |ui| {
                ui.add(egui::Slider::new(&mut s.shadow_steps, 4..=128).text("shadow steps"));
                ui.add(egui::Slider::new(&mut s.shadow_iterations, 1..=32).text("shadow iterations"));
                ui.add(egui::Slider::new(&mut s.shadow_softness, 2.0..=64.0).logarithmic(true).text("shadow sharpness"));
            });
            ui.add(egui::Slider::new(&mut s.ao_strength, 0.0..=6.0).text("AO"));
            ui.horizontal(|ui| {
                ui.label("base color");
                ui.color_edit_button_rgb(&mut s.base_color).on_hover_text("Used when the material graph has no Material Output");
            });
            ui.checkbox(&mut s.reflections, "Ray-traced reflections");
            ui.add_enabled(
                s.reflections,
                egui::Slider::new(&mut s.reflection_steps, 16..=256).text("reflection steps"),
            );
        });
        ui.collapsing("Camera", |ui| {
            ui.add(egui::Slider::new(&mut cam.fov_deg, 20.0..=110.0).text("fov"));
            ui.add(
                egui::Slider::new(&mut s.move_speed, MIN_SPEED..=MAX_SPEED)
                    .logarithmic(true)
                    .max_decimals(6)
                    .text("speed (units/s)"),
            );
            ui.small("Scroll: ±5% per notch, Shift + scroll: ±25%");
            ui.add(egui::Slider::new(&mut s.look_sensitivity, 0.1..=4.0).logarithmic(true).text("look sensitivity"));
            if ui.button("Reset camera").on_hover_text("Back to this fractal's starting view").clicked() {
                *cam = start_camera(&doc.graphs.material);
            }
        });
    });
}

const SHORTCUTS: &[(&str, &str)] = &[
    ("F1", "this list"),
    ("W A S D", "move"),
    ("Q / E", "down / up"),
    ("Right mouse drag", "look around"),
    ("Scroll", "speed ±5% (Shift: ±25%)"),
    ("Space", "play / pause (with the song)"),
    ("Tab", "hide / show UI"),
    ("F11", "fullscreen (Esc leaves)"),
    ("F12", "screenshot"),
    ("N", "node editor"),
    ("I", "insert camera key at the playhead"),
    ("Left / Right", "step a frame (Shift: range start / end)"),
    ("Up / Down", "next / previous camera key"),
    ("Del / X (timeline)", "delete selected keys"),
    ("Ctrl+C / X / V", "copy / cut / paste nodes"),
    ("Ctrl+D", "duplicate nodes"),
    ("M, Ctrl+L", "mute / lock nodes"),
    ("Ctrl+A, Del / X", "select all / delete nodes"),
    ("Scroll (node editor)", "zoom; Home / F frame all / selected"),
    ("Shift+A", "search and add a node"),
    ("Right drag (node editor)", "box select (Shift adds, Alt removes)"),
    ("Ctrl+right drag", "cut links"),
    ("Ctrl+Z", "undo graph edit"),
    ("Ctrl+Y, Ctrl+Shift+Z", "redo"),
    ("Ctrl+S", "save project"),
];

fn shortcuts_message() -> Message {
    let rows = SHORTCUTS.iter().map(|(k, w)| (k.to_string(), w.to_string())).collect();
    Message::new("shortcuts", Kind::Info, "Keyboard shortcuts", "").rows(rows).primary("Close", Action::Close)
}

/// What happened when the last run didn't exit cleanly, after `cut` links into the
/// output nodes were removed. `panic` is its crash log entry, if it crashed with one.
fn crash_message(cut: usize, panic: Option<String>, backup: Option<std::path::PathBuf>) -> Message {
    let (title, mut body) = match &panic {
        Some(_) => ("fractalful crashed last time", "The last session stopped with an internal error.".to_string()),
        None => (
            "fractalful didn't close properly",
            "The last session ended unexpectedly. It most likely froze and was closed with End task, \
             or the computer shut down while it was open."
                .to_string(),
        ),
    };
    if cut > 0 {
        body += "\n\nIn case the node graph caused it, its outputs were disconnected before anything ran, \
                 so you're looking at a plain stand-in shape. All your nodes are still there.";
    }
    let mut details = panic.unwrap_or_default();
    if let Some(b) = backup {
        details += &format!("{}The session as it was before is saved in {}.", if details.is_empty() { "" } else { "\n\n" }, b.display());
    }
    let mut m = Message::new("crash", Kind::Warning, title, body);
    if !details.is_empty() {
        m = m.details(details);
    }
    match cut {
        0 => m.primary("OK", Action::Close),
        _ => m.button("Reconnect outputs", Action::ReconnectOutputs).primary("Keep disconnected", Action::Close),
    }
}

/// A finished (or failed) export, with buttons to look at the result.
fn export_message(v: &video::VideoSettings, status: &Result<String, String>) -> Message {
    let path = v.output_path();
    match status {
        Ok(msg) if v.format == VideoFormat::PngSequence => Message::new("export", Kind::Success, "Frames exported", capitalize(msg))
            .button("Close", Action::Close)
            .primary("Open folder", Action::Open(path)),
        Ok(msg) => Message::new("export", Kind::Success, "Video exported", capitalize(msg))
            .button("Close", Action::Close)
            .button("Show in folder", Action::Reveal(path.clone()))
            .primary("Play video", Action::Open(path)),
        Err(e) => Message::new("export", Kind::Error, "The export failed", "The video couldn't be finished.").details(e.clone()),
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
}

/// Full-view overlays under the UI: the fade-in from black at launch, and the white
/// flash when a screenshot is taken.
fn screen_effects(ctx: &egui::Context, first_frame: Instant, flash: Option<Instant>) {
    let fade_in = 1.0 - (first_frame.elapsed().as_secs_f32() / 0.6).min(1.0);
    let flash = flash.map_or(0.0, |t| 1.0 - (t.elapsed().as_secs_f32() / 0.35).min(1.0));
    if fade_in <= 0.0 && flash <= 0.0 {
        return;
    }
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::unique("screen effects")));
    let rect = ctx.content_rect();
    if fade_in > 0.0 {
        painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha((egui::emath::easing::quadratic_in(fade_in) * 255.0) as u8));
    }
    if flash > 0.0 {
        painter.rect_filled(rect, 0.0, egui::Color32::from_white_alpha((egui::emath::easing::quadratic_in(flash) * 110.0) as u8));
    }
    ctx.request_repaint();
}

/// The camera views are the same (to well below anything visible).
fn same_view(a: &Camera, b: &Camera) -> bool {
    (a.pos - b.pos).length() <= 1e-6 * (1.0 + a.pos.length())
        && (a.yaw - b.yaw).abs() < 1e-5
        && (a.pitch - b.pitch).abs() < 1e-5
        && (a.fov_deg - b.fov_deg).abs() < 1e-4
}

/// The starting view for a graph: a preset's own hint, or in front of the fractal.
fn start_camera(g: &Graph) -> Camera {
    let mut cam = Camera::new();
    let (pos, yaw, pitch) = g.suggested_camera();
    (cam.pos, cam.yaw, cam.pitch) = (Vec3::from(pos), yaw, pitch);
    cam
}

/// A new screenshot file name that doesn't overwrite an earlier one.
/// A new file in Documents\fractalful\screenshots (created if needed).
fn screenshot_path() -> std::path::PathBuf {
    let dir = paths::screenshots_dir();
    let _ = std::fs::create_dir_all(&dir);
    let stamp = chrono_stamp();
    (1..)
        .map(|n| match n {
            1 => format!("screenshot_{stamp}.png"),
            n => format!("screenshot_{stamp}_{n}.png"),
        })
        .map(|name| dir.join(name))
        .find(|p| !p.exists())
        .expect("unbounded")
}

fn project_panel(ui: &mut egui::Ui, proj: &mut ProjectUi) {
    ui.horizontal(|ui| {
        ui.label("name");
        let field = ui.add(egui::TextEdit::singleline(&mut proj.name).hint_text("untitled").desired_width(140.0));
        let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if ui.button("Save").on_hover_text("Ctrl+S").clicked() || enter {
            proj.action = Some(ProjectAction::Save);
        }
    });
    ui.horizontal(|ui| {
        let menu = ui.menu_button("Open…", |ui| {
            if proj.saved.is_empty() {
                ui.weak(format!("No projects in {} yet", paths::projects_dir().display()));
            }
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                egui::Grid::new("saved projects").num_columns(2).spacing([14.0, 2.0]).show(ui, |ui| {
                    for (name, saved) in &proj.saved {
                        let current = proj.file.as_deref() == Some(name.as_str());
                        if ui.selectable_label(current, name).clicked() {
                            proj.action = Some(ProjectAction::Load(name.clone()));
                            ui.close();
                        }
                        ui.weak(project::ago(*saved));
                        ui.end_row();
                    }
                });
            });
        });
        if menu.response.clicked() {
            proj.saved = project::list();
        }
        if ui.button("Show folder").clicked() {
            let dir = paths::projects_dir();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::process::Command::new("explorer").arg(&dir).spawn();
        }
    });
    ui.small(format!("→ {}", project::project_path(&proj.name).display()));
    ui.small("The session is also saved automatically and restored at the next launch.");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let mut attrs = Window::default_attributes()
                .with_title("fractalful")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 800))
                // Below this the panels no longer fit next to each other.
                .with_min_inner_size(winit::dpi::LogicalSize::new(640, 400))
                .with_maximized(self.restore_maximized);
            if let Some([w, h]) = self.restore_window_size.filter(|s| s[0] >= 320 && s[1] >= 200) {
                attrs = attrs.with_inner_size(winit::dpi::PhysicalSize::new(w, h));
            }
            let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
            // A restored graph that no longer compiles falls back to the default one.
            let material = graph::codegen::generate(&self.doc.graphs.material);
            let world = graph::codegen::generate_world(&self.doc.graphs.world);
            if material.is_err() || world.is_err() {
                let mut details = [material.as_ref().err(), world.as_ref().err()].into_iter().flatten().cloned().collect::<Vec<_>>().join("\n");
                if let Some(b) = self.persist.then(|| project::back_up_session("broken")).flatten() {
                    details += &format!("\n\nThe session was copied to {}.", b.display());
                }
                self.modals.push(
                    Message::new(
                        "compile",
                        Kind::Error,
                        "The restored scene doesn't compile",
                        "A node graph from the last session can't be turned into a shader anymore, \
                         so fractalful started with the default one instead.",
                    )
                    .details(details),
                );
            }
            let wgsl = material.unwrap_or_else(|_| {
                self.doc.replace_material(Graph::default());
                self.camera = Some(start_camera(&self.doc.graphs.material));
                graph::codegen::generate(&self.doc.graphs.material).expect("default graph compiles")
            });
            let world = world.unwrap_or_else(|_| {
                self.doc.replace(GraphKind::World, Graphs::default().world);
                graph::codegen::generate_world(&self.doc.graphs.world).expect("default world compiles")
            });
            let gpu = Gpu::new(window, &wgsl, &world);
            // Start in HDR10 when the display is in HDR mode, with its own luminance levels.
            self.settings.output_mode = gpu.caps.default_mode();
            self.settings.paper_white_nits = gpu.caps.sdr_white_nits.unwrap_or(200.0).clamp(80.0, 500.0);
            self.settings.peak_nits = gpu.caps.peak_nits.unwrap_or(1000.0).max(self.settings.paper_white_nits);
            self.gpu = Some(gpu);
            self.doc.compiled[GraphKind::Material.index()] = Some(self.doc.graphs.material.structure_rev);
            self.doc.compiled[GraphKind::World.index()] = Some(self.doc.graphs.world.structure_rev);
            self.doc.compiled_may_reflect = self.doc.graphs.material.may_reflect();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if let Some(ew) = self.editor_window.as_mut().filter(|ew| ew.window.id() == id) {
            if matches!(event, WindowEvent::CloseRequested) {
                // Closing the separate window docks the editor again.
                self.doc.detach_request = Some(false);
                self.apply_detach_request(event_loop);
            } else if let Some(gpu) = &self.gpu {
                ew.on_event(&gpu.device, &event);
            }
            return;
        }
        let blocked = self.modals.is_open();
        let Some(gpu) = self.gpu.as_mut() else { return };
        // While a message box is open, the app's own keys and mouse are off.
        let consumed = gpu.egui_state.on_window_event(&gpu.window, &event).consumed || blocked;

        match event {
            WindowEvent::CloseRequested if self.video_job.is_some() && self.auto_export.is_none() => {
                let (frame, total) = self.video_job.as_ref().map_or((0, 0), |j| (j.frame, j.total));
                self.modals.push(
                    Message::new(
                        "quit",
                        Kind::Warning,
                        "A video is still exporting",
                        format!(
                            "Frame {frame} of {total}. Quitting now stops the export; the frames rendered so far \
                             are kept (an MP4 is finalized up to that point)."
                        ),
                    )
                    .danger("Stop and quit", Action::Quit)
                    .primary("Keep rendering", Action::Close),
                );
            }
            WindowEvent::CloseRequested => {
                if self.persist {
                    self.save_session();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                let normal = !gpu.window.is_maximized() && gpu.window.fullscreen().is_none() && gpu.window.is_minimized() != Some(true);
                if normal && size.width > 0 && size.height > 0 {
                    self.restore_window_size = Some([size.width, size.height]);
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                self.apply_detach_request(event_loop);
                if self.exit_requested {
                    event_loop.exit();
                } else if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed if !consumed && code == KeyCode::KeyN => match &self.editor_window {
                            Some(ew) => ew.window.focus_window(),
                            None => self.doc.show_editor = !self.doc.show_editor,
                        },
                        ElementState::Pressed if !consumed && code == KeyCode::F1 && !event.repeat => {
                            self.modals.push(shortcuts_message());
                        }
                        ElementState::Pressed if !consumed && code == KeyCode::F12 && !event.repeat => {
                            self.screenshot = Some(screenshot_path());
                        }
                        ElementState::Pressed if !consumed && code == KeyCode::Space && !event.repeat => {
                            self.settings.playing = !self.settings.playing;
                        }
                        ElementState::Pressed if !consumed && code == KeyCode::Tab && !event.repeat => {
                            self.ui_hidden = !self.ui_hidden;
                            if self.ui_hidden {
                                self.toasts.info("ui", "UI hidden (Tab to show)");
                            }
                        }
                        ElementState::Pressed if !consumed && code == KeyCode::F11 && !event.repeat => {
                            let full = gpu.window.fullscreen().is_none();
                            gpu.window.set_fullscreen(full.then_some(Fullscreen::Borderless(None)));
                        }
                        ElementState::Pressed if !consumed && code == KeyCode::Escape && gpu.window.fullscreen().is_some() => {
                            gpu.window.set_fullscreen(None);
                        }
                        ElementState::Pressed if !consumed => {
                            self.keys.insert(code);
                        }
                        ElementState::Released => {
                            self.keys.remove(&code);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { button: MouseButton::Right, state, .. } => {
                self.set_looking(state == ElementState::Pressed && !consumed);
            }
            WindowEvent::CursorMoved { position, .. } => self.cursor_pos = Some(position),
            WindowEvent::MouseWheel { delta, .. } if !consumed => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                // Fine steps by default; Shift for coarse ones.
                let shift = self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight);
                let step: f32 = if shift { 1.25 } else { 1.05 };
                let speed = (self.settings.move_speed * step.powf(y)).clamp(MIN_SPEED, MAX_SPEED);
                self.settings.move_speed = speed;
                let text = if speed >= 0.01 { format!("speed {speed:.3} units/s") } else { format!("speed {speed:.2e} units/s") };
                self.toasts.info("speed", text);
            }
            // Drop a song to load it, or a saved project to open it.
            WindowEvent::DroppedFile(path) => {
                if audio::decode::is_audio(&path) {
                    self.audio.load(path);
                } else if path.extension().is_some_and(|e| e == "json") {
                    self.load_project_file(&path);
                } else {
                    self.toasts.error("drop", format!("Can't open {} (songs: FLAC, MP3, WAV; projects: .json)", path.display()));
                }
            }
            WindowEvent::Focused(false) => {
                self.keys.clear();
                self.set_looking(false);
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        // Raw motion keeps turning when the hidden cursor would hit the window edge.
        if let (DeviceEvent::MouseMotion { delta: (dx, dy) }, true) = (event, self.looking) {
            if let Some(cam) = self.camera.as_mut() {
                let k = 0.004 * self.settings.look_sensitivity;
                cam.rotate(dx as f32 * k, dy as f32 * k);
            }
        }
    }
}

fn chrono_stamp() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn main() {
    platform::attach_parent_console();
    platform::install_crash_log();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    // Command-line runs (export, preset, bench) start clean and leave the session alone;
    // FRACTALFUL_FRESH=1 does the same for an interactive run.
    let scripted = ["FRACTALFUL_EXPORT", "FRACTALFUL_PRESET", "FRACTALFUL_PROJECT", "FRACTALFUL_BENCH", "FRACTALFUL_FRESH"];
    app.persist = !scripted.iter().any(|v| std::env::var_os(v).is_some());
    // Held until the event loop returns; left behind if the process is killed.
    let mut run_lock = None;
    if app.persist {
        let (lock, last_run) = project::RunLock::claim(project::RunLock::path());
        run_lock = lock;
        app.restore_session(last_run);
        if let Some((n, from)) = paths::migrate_old_projects() {
            let msg = format!("Copied {n} project(s) from {} to {}", from.display(), paths::projects_dir().display());
            println!("{msg}");
            app.toasts.info("migrate", msg);
        }
    }
    // FRACTALFUL_PROJECT=<file.json> opens a saved project (the session is left alone).
    if let Some(path) = std::env::var_os("FRACTALFUL_PROJECT") {
        app.load_project_file(std::path::Path::new(&path));
    }
    // FRACTALFUL_EXPORT=<path> renders a video and exits. Optional overrides:
    // FRACTALFUL_EXPORT_FORMAT=h264|hdr10|png, _SIZE=WxH, _FPS, _DURATION, _SAMPLES,
    // _SHUTTER, _CAMERA=current|orbit.
    if let Ok(path) = std::env::var("FRACTALFUL_EXPORT") {
        let env = |k: &str| std::env::var(format!("FRACTALFUL_EXPORT_{k}")).ok();
        let v = &mut app.studio.settings;
        v.output = std::path::absolute(&path).map_or(path, |p| p.display().to_string());
        match env("FORMAT").as_deref() {
            Some("hdr10") => v.format = VideoFormat::Hdr10,
            Some("png") => v.format = VideoFormat::PngSequence,
            _ => v.format = VideoFormat::H264,
        }
        if let Some((w, h)) = env("SIZE").as_deref().and_then(|s| s.split_once('x')) {
            (v.width, v.height) = (w.parse().unwrap_or(v.width), h.parse().unwrap_or(v.height));
        }
        v.fps = env("FPS").and_then(|s| s.parse().ok()).unwrap_or(v.fps);
        v.duration = env("DURATION").and_then(|s| s.parse().ok()).unwrap_or(v.duration);
        v.samples = env("SAMPLES").and_then(|s| s.parse().ok()).unwrap_or(v.samples);
        v.shutter = env("SHUTTER").and_then(|s| s.parse().ok()).unwrap_or(v.shutter);
        if env("CAMERA").as_deref() == Some("orbit") {
            v.camera = CameraMode::Orbit;
        }
        app.auto_export = Some(false);
    }
    // FRACTALFUL_TIME=<seconds> pauses the animation clock at that time.
    if let Some(t) = std::env::var("FRACTALFUL_TIME").ok().and_then(|t| t.parse().ok()) {
        app.anim_time = t;
        app.settings.playing = false;
    }
    // FRACTALFUL_AUDIO=<file> loads a song (FLAC / MP3 / WAV) at startup.
    if let Some(path) = std::env::var_os("FRACTALFUL_AUDIO") {
        app.audio.load(path.into());
    }
    // FRACTALFUL_EDITOR=docked|detached[:world|:post] opens the node editor at startup
    // (in that graph's mode).
    if let Ok(v) = std::env::var("FRACTALFUL_EDITOR") {
        let (how, mode) = v.split_once(':').unwrap_or((&v, ""));
        match how {
            "docked" => app.doc.show_editor = true,
            "detached" => app.doc.detach_request = Some(true),
            _ => {}
        }
        app.doc.editor.mode = match mode {
            "world" => GraphKind::World,
            "post" => GraphKind::Post,
            _ => GraphKind::Material,
        };
    }
    // FRACTALFUL_WORLD / FRACTALFUL_POST=<name> start with a world / processing preset
    // (see graph::looks).
    if let Ok(name) = std::env::var("FRACTALFUL_WORLD") {
        app.doc.graphs.world = graph::looks::world_preset(&name);
    }
    if let Ok(name) = std::env::var("FRACTALFUL_POST") {
        app.doc.graphs.post = graph::looks::post_preset(&name);
    }
    // FRACTALFUL_PRESET=<name> starts from a built-in graph (see graph::PRESET_CATEGORIES).
    if let Ok(name) = std::env::var("FRACTALFUL_PRESET") {
        if !graph::presets().any(|p| p == name) {
            eprintln!("preset: no preset named {name:?}; using the default Mandelbulb");
        }
        let (g, iters) = graph::preset(&name);
        let mut cam = Camera::new();
        let (pos, yaw, pitch) = g.suggested_camera();
        (cam.pos, cam.yaw, cam.pitch) = (Vec3::from(pos), yaw, pitch);
        app.doc.graphs.material = g;
        app.settings.iterations = iters;
        app.camera = Some(cam);
    }
    if std::env::var_os("FRACTALFUL_BENCH").is_some() {
        app.bench = Some(Bench::default());
        // FRACTALFUL_BENCH_VIEW=<n> picks a DebugView by index.
        if let Some(v) = std::env::var("FRACTALFUL_BENCH_VIEW").ok().and_then(|v| v.parse::<usize>().ok()) {
            app.settings.debug_view = DebugView::ALL[v.min(DebugView::ALL.len() - 1)];
        }
        // FRACTALFUL_BENCH_POS="x,y,z[,yaw,pitch]" overrides the camera.
        if let Some(pos) = std::env::var("FRACTALFUL_BENCH_POS").ok() {
            let v: Vec<f32> = pos.split(',').filter_map(|c| c.trim().parse().ok()).collect();
            if let [x, y, z, ..] = v[..] {
                let mut cam = Camera::new();
                cam.pos = Vec3::new(x, y, z);
                if let [_, _, _, yaw, pitch] = v[..] {
                    cam.yaw = yaw;
                    cam.pitch = pitch;
                }
                app.camera = Some(cam);
            }
        }
    }
    event_loop.run_app(&mut app).expect("run app");
    drop(run_lock);
}
