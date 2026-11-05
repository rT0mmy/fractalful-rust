//! Blender-style timeline: a bar along the bottom of the main window with transport
//! controls, a time ruler and playhead, the camera keyframe track, and (with a song
//! loaded) its waveform and beat grid.
//!
//! The scene range is the video export range (start + duration), so what loops in the
//! timeline is what gets rendered. Camera keys are inserted at the playhead from the
//! current view (I); while camera animation is on, scrubbing or playing moves the view
//! along the path, and flying while paused leaves the path until the next scrub (or
//! keys the new view right away with Auto-key).
//!
//! The Nodes row sums up the keyframes of node inputs (keyed in the node editor): one
//! diamond per time that has any, which selects, moves and deletes all keys there.
//!
//! Mouse: drag the ruler (or waveform) to scrub; click a key to select it (Shift / Ctrl
//! adds), drag to move the selection, double-click to jump to it, right-click for its
//! menu; drag empty track space to box-select. Scroll zooms around the pointer,
//! Shift+scroll / middle drag pans, Home fits the scene range.

use egui::{Color32, FontId, Key, Modifiers, PointerButton, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, pos2, vec2};

use crate::audio::analysis::{Analysis, Trigger};
use crate::camera::Camera;
use crate::graph::{AnimSnapshot, Graphs};
use crate::video::{CameraMode, Interp, KEY_EPSILON, Keyframe, Studio, keyframe_camera, sort_keys};

const LABEL_W: f32 = 84.0;
const RULER_H: f32 = 22.0;
const KEY_ROW_H: f32 = 26.0;
const KEY_R: f32 = 6.0;
const PLAYHEAD: Color32 = Color32::from_rgb(80, 140, 255);
const KEY_SELECTED: Color32 = Color32::from_rgb(255, 165, 40);

#[derive(Default)]
enum Drag {
    #[default]
    None,
    Scrub,
    /// Moving the selected keys: the time grabbed, and each moved key's original time.
    Keys { grab: f32, orig: Vec<(usize, f32)> },
    /// Moving the selected node key times: the graph's keys when the drag started (each
    /// frame moves from there), and the offset applied so far.
    Params { grab: f32, snapshot: AnimSnapshot, delta: f32 },
    Box { start: Pos2 },
}

pub struct Timeline {
    pub visible: bool,
    /// Visible time span in seconds; `None` fits the scene range on the next frame.
    view: Option<(f32, f32)>,
    drag: Drag,
    /// Draw the camera path in the viewport.
    pub show_path: bool,
    /// Blender's record button: while paused, a changed view is keyed at the playhead.
    pub auto_key: bool,
    /// Time under the pointer when the context menu opened.
    menu_t: f32,
    /// Selected times in the Nodes row.
    param_sel: Vec<f32>,
    /// The tracks area last frame (I over it keys the camera, like over the viewport).
    tracks_rect: Option<Rect>,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            visible: true,
            view: None,
            drag: Drag::None,
            show_path: true,
            auto_key: false,
            menu_t: 0.0,
            param_sel: Vec::new(),
            tracks_rect: None,
        }
    }
}

/// What the timeline edits.
pub struct Ctx<'a> {
    pub time: &'a mut f32,
    pub playing: &'a mut bool,
    pub speed: &'a mut f32,
    pub studio: &'a mut Studio,
    /// The node graphs, for their keyframed inputs.
    pub graph: &'a mut Graphs,
    /// The current view, for inserting keys.
    pub cam: &'a Camera,
    /// The song's analysis and the animation time where it starts.
    pub song: Option<(&'a Analysis, f32)>,
    /// Exporting: everything is read-only.
    pub busy: bool,
}

/// What changed, for the app to react to.
#[derive(Default)]
pub struct Changes {
    /// Keys were added, moved, removed or edited: the camera should re-follow the path.
    pub keys: bool,
    pub message: Option<String>,
}

fn snap(t: f32, fps: f32) -> f32 {
    (t * fps).round() / fps
}

fn range(studio: &Studio) -> (f32, f32) {
    let v = &studio.settings;
    (v.start, v.start + v.duration)
}

/// Readable time: "1:05.2" past a minute, else "12.34 s".
pub fn fmt_time(t: f32) -> String {
    if t.abs() >= 60.0 {
        format!("{}{}:{:04.1}", if t < 0.0 { "-" } else { "" }, (t.abs() / 60.0) as u32, t.abs() % 60.0)
    } else {
        format!("{t:.2} s")
    }
}

impl Timeline {
    /// The whole bar: transport row, then the tracks. Returns what changed.
    pub fn show(&mut self, ui: &mut egui::Ui, c: Ctx) -> Changes {
        let mut changes = Changes::default();
        let changes = &mut changes;
        let Ctx { time, playing, speed, studio, graph, cam, song, busy } = c;
        let fps = studio.settings.fps.max(1) as f32;
        let (start, end) = range(studio);
        let view = *self.view.get_or_insert_with(|| fit(start, end));
        let mut fit_view = false;

        // ---- transport row (wraps on narrow windows) ----
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal_wrapped(|ui| {
                if icon_button(ui, Icon::ToStart).on_hover_text("Jump to start (Shift+Left)").clicked() {
                    *time = start;
                }
                if icon_button(ui, Icon::PrevKey).on_hover_text("Previous key (Down)").clicked() {
                    if let Some(t) = prev_key(&all_key_times(studio, graph), *time) {
                        *time = t;
                    }
                }
                let play = if *playing { Icon::Pause } else { Icon::Play };
                if icon_button(ui, play).on_hover_text("Play / pause (Space)").clicked() {
                    *playing = !*playing;
                }
                if icon_button(ui, Icon::NextKey).on_hover_text("Next key (Up)").clicked() {
                    if let Some(t) = next_key(&all_key_times(studio, graph), *time) {
                        *time = t;
                    }
                }
                if icon_button(ui, Icon::ToEnd).on_hover_text("Jump to end (Shift+Right)").clicked() {
                    *time = end;
                }
                ui.add(egui::DragValue::new(time).speed(1.0 / fps).range(0.0..=36000.0).custom_formatter(|t, _| fmt_time(t as f32)))
                    .on_hover_text("Current time (Left / Right: step a frame)");
                ui.weak(format!("frame {}", (*time * fps).round() as i64));
                ui.separator();
                ui.label("speed");
                ui.add(egui::DragValue::new(speed).speed(0.01).range(-8.0..=8.0).max_decimals(2).suffix("×"));
                ui.checkbox(&mut studio.settings.loop_playback, "Loop").on_hover_text("Play the scene range in a loop");
                ui.separator();
                let v = &mut studio.settings;
                ui.label("range");
                let mut s = v.start;
                if ui.add(egui::DragValue::new(&mut s).speed(1.0 / fps).range(0.0..=36000.0).custom_formatter(|t, _| fmt_time(t as f32))).changed() {
                    v.start = s;
                }
                ui.label("–");
                let mut e = v.start + v.duration;
                if ui
                    .add(egui::DragValue::new(&mut e).speed(1.0 / fps).range(v.start + 0.1..=36000.0).custom_formatter(|t, _| fmt_time(t as f32)))
                    .on_hover_text("Scene range = the video export range")
                    .changed()
                {
                    v.duration = e - v.start;
                }
                ui.add(egui::DragValue::new(&mut v.fps).range(1..=240).suffix(" fps"));
                ui.separator();

                // Camera animation.
                let mut animated = v.camera == CameraMode::Keyframes;
                let toggle = ui
                    .toggle_value(&mut animated, "Animate camera")
                    .on_hover_text("The camera follows its keyframes when scrubbing, playing and exporting");
                if toggle.changed() {
                    v.camera = if animated { CameraMode::Keyframes } else { CameraMode::Current };
                    changes.keys = true;
                }
                if ui.button("Insert key").on_hover_text("Key the current view at the playhead (I)").clicked() {
                    insert_key(studio, *time, cam, changes);
                }
                let selected = studio.keyframes.iter().filter(|k| k.selected).count();
                if ui.add_enabled(selected > 0, egui::Button::new("Delete")).on_hover_text("Delete selected keys (Del over the timeline)").clicked() {
                    delete_selected(studio, changes);
                }
                // Interpolation of the selected keys.
                ui.add_enabled_ui(selected > 0, |ui| {
                    let current = studio.keyframes.iter().find(|k| k.selected).map_or(Interp::Smooth, |k| k.interp);
                    let mixed = studio.keyframes.iter().filter(|k| k.selected).any(|k| k.interp != current);
                    egui::ComboBox::from_id_salt("key interp")
                        .width(110.0)
                        .selected_text(if mixed { "(mixed)" } else { current.label() })
                        .show_ui(ui, |ui| {
                            for i in Interp::ALL {
                                if ui.selectable_label(!mixed && current == i, i.label()).clicked() {
                                    for k in studio.keyframes.iter_mut().filter(|k| k.selected) {
                                        k.interp = i;
                                    }
                                    changes.keys = true;
                                }
                            }
                        })
                        .response
                        .on_hover_text("Interpolation from the selected keys to the next ones");
                });
                let rec = egui::Button::new(egui::RichText::new("Auto-key").color(if self.auto_key {
                    Color32::from_rgb(255, 90, 90)
                } else {
                    ui.visuals().text_color()
                }))
                .selected(self.auto_key);
                if ui.add(rec).on_hover_text("While paused, moving the camera keys it at the playhead").clicked() {
                    self.auto_key = !self.auto_key;
                    if self.auto_key && studio.settings.camera != CameraMode::Keyframes {
                        studio.settings.camera = CameraMode::Keyframes;
                    }
                }
                ui.toggle_value(&mut self.show_path, "Path").on_hover_text("Show the camera path in the viewport");
                if ui.button("Fit").on_hover_text("Fit the scene range in view (Home)").clicked() {
                    fit_view = true;
                }
            });
        });

        // ---- tracks ----
        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(vec2(avail.x, avail.y.max(RULER_H + 2.0 * KEY_ROW_H + 8.0)), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let tracks = Rect::from_min_max(pos2(rect.left() + LABEL_W, rect.top()), rect.max);
        let ruler = Rect::from_min_size(tracks.min, vec2(tracks.width(), RULER_H));
        self.tracks_rect = Some(tracks);
        let key_row = Rect::from_min_size(pos2(tracks.left(), ruler.bottom()), vec2(tracks.width(), KEY_ROW_H));
        let param_row = Rect::from_min_size(pos2(tracks.left(), key_row.bottom()), vec2(tracks.width(), KEY_ROW_H));
        let audio_row = Rect::from_min_max(pos2(tracks.left(), param_row.bottom() + 2.0), tracks.max);
        let (v0, v1) = view;
        let x_of = |t: f32| tracks.left() + (t - v0) / (v1 - v0) * tracks.width();
        let t_of = |x: f32| v0 + (x - tracks.left()) / tracks.width() * (v1 - v0);

        // Backgrounds and row labels.
        let bg = ui.visuals().extreme_bg_color;
        painter.rect_filled(rect, 2.0, bg);
        painter.rect_filled(ruler, 0.0, ui.visuals().faint_bg_color);
        let label = |r: Rect, text: &str| {
            painter.text(pos2(rect.left() + 8.0, r.center().y), egui::Align2::LEFT_CENTER, text, FontId::proportional(12.0), Color32::from_gray(170));
        };
        label(key_row, "Camera");
        label(param_row, "Nodes");
        if song.is_some() && audio_row.height() > 6.0 {
            label(audio_row, "Song");
        }
        // Outside the scene range is shaded.
        let shade = Color32::from_black_alpha(90);
        let (xs, xe) = (x_of(start), x_of(end));
        if xs > tracks.left() {
            painter.rect_filled(Rect::from_min_max(tracks.min, pos2(xs.min(tracks.right()), tracks.bottom())), 0.0, shade);
        }
        if xe < tracks.right() {
            painter.rect_filled(Rect::from_min_max(pos2(xe.max(tracks.left()), tracks.top()), tracks.max), 0.0, shade);
        }

        // Ruler ticks.
        let px_per_s = tracks.width() / (v1 - v0);
        let step = [1.0 / fps, 2.0 / fps, 5.0 / fps, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0]
            .into_iter()
            .find(|s| s * px_per_s >= 70.0)
            .unwrap_or(1200.0);
        let mut t = (v0 / step).floor() * step;
        while t <= v1 {
            let x = x_of(t);
            if x >= tracks.left() {
                painter.vline(x, tracks.y_range(), Stroke::new(1.0, Color32::from_white_alpha(10)));
                painter.vline(x, ruler.bottom() - 6.0..=ruler.bottom(), Stroke::new(1.0, Color32::from_gray(120)));
                let text = if step >= 1.0 && t.abs() >= 60.0 { fmt_time(t) } else if step >= 1.0 { format!("{t:.0}s") } else { format!("{t:.2}") };
                painter.text(pos2(x + 3.0, ruler.top() + 2.0), egui::Align2::LEFT_TOP, text, FontId::proportional(11.0), Color32::from_gray(150));
            }
            t += step;
        }

        // Song waveform and beat grid.
        if let Some((a, offset)) = song.filter(|_| audio_row.height() > 6.0) {
            let mid = audio_row.center().y;
            let half = audio_row.height() * 0.45;
            let wave = Color32::from_rgb(70, 120, 110);
            let mut x = audio_row.left();
            while x < audio_row.right() {
                let st = t_of(x) - offset;
                if (0.0..=a.duration).contains(&st) {
                    let h = a.level(0, st).clamp(0.0, 1.0) * half;
                    painter.vline(x, mid - h..=mid + h, Stroke::new(1.0, wave));
                }
                x += 1.0;
            }
            for (trigger, len, color) in [(Trigger::Beat, 0.25, Color32::from_white_alpha(40)), (Trigger::Bar, 0.5, Color32::from_white_alpha(90))] {
                for &e in a.events(trigger) {
                    let x = x_of(e + offset);
                    if audio_row.x_range().contains(x) {
                        painter.vline(x, audio_row.top()..=audio_row.top() + audio_row.height() * len, Stroke::new(1.0, color));
                    }
                }
            }
        }

        // Camera keys: interpolation bars between neighbours, then diamonds.
        let keys = &studio.keyframes;
        let ky = key_row.center().y;
        let animated = studio.settings.camera == CameraMode::Keyframes;
        for w in keys.windows(2) {
            let (a, b) = (x_of(w[0].time), x_of(w[1].time));
            let (width, color) = match w[0].interp {
                Interp::Constant => (4.0, Color32::from_gray(110)),
                Interp::Linear => (1.5, Color32::from_rgb(110, 170, 110)),
                Interp::Ease => (1.5, Color32::from_rgb(110, 150, 210)),
                Interp::Smooth => (1.5, Color32::from_gray(120)),
            };
            painter.line_segment([pos2(a.max(tracks.left()), ky), pos2(b.min(tracks.right()), ky)], Stroke::new(width, color.gamma_multiply(if animated { 1.0 } else { 0.4 })));
        }
        let pointer = ui.ctx().pointer_latest_pos();
        // Hit-testing works on a snapshot of the key times, so keys can be edited below.
        let key_times: Vec<f32> = keys.iter().map(|k| k.time).collect();
        let hit_key = |p: Pos2| -> Option<usize> {
            if !key_row.expand(4.0).contains(p) {
                return None;
            }
            key_times
                .iter()
                .enumerate()
                .map(|(i, &t)| (i, (x_of(t) - p.x).abs()))
                .filter(|(_, d)| *d <= KEY_R + 3.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        };
        let hovered_key = pointer.and_then(hit_key);
        for (i, k) in keys.iter().enumerate() {
            let x = x_of(k.time);
            if !tracks.x_range().contains(x) {
                continue;
            }
            let fill = if k.selected { KEY_SELECTED } else { Color32::from_gray(205) };
            let fill = if animated { fill } else { fill.gamma_multiply(0.5) };
            let stroke = Stroke::new(if hovered_key == Some(i) { 2.0 } else { 1.0 }, Color32::BLACK);
            painter.add(Shape::convex_polygon(
                vec![pos2(x, ky - KEY_R), pos2(x + KEY_R, ky), pos2(x, ky + KEY_R), pos2(x - KEY_R, ky)],
                fill,
                stroke,
            ));
        }
        if !animated && !keys.is_empty() {
            painter.text(
                pos2(tracks.right() - 6.0, key_row.center().y),
                egui::Align2::RIGHT_CENTER,
                "camera animation off",
                FontId::proportional(11.0),
                Color32::from_gray(140),
            );
        }

        // Nodes row: one diamond per time with node keys.
        let param_times = graph.key_times();
        self.param_sel.retain(|s| param_times.iter().any(|t| (t - s).abs() < KEY_EPSILON));
        let py = param_row.center().y;
        if let (Some(a), Some(b)) = (param_times.first(), param_times.last()) {
            painter.line_segment([pos2(x_of(*a).max(tracks.left()), py), pos2(x_of(*b).min(tracks.right()), py)], Stroke::new(1.0, Color32::from_gray(90)));
        }
        let hit_param = |p: Pos2| -> Option<f32> {
            if !param_row.expand(4.0).contains(p) {
                return None;
            }
            param_times
                .iter()
                .map(|&t| (t, (x_of(t) - p.x).abs()))
                .filter(|(_, d)| *d <= KEY_R + 3.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(t, _)| t)
        };
        let hovered_param = pointer.and_then(hit_param);
        for &t in &param_times {
            let x = x_of(t);
            if !tracks.x_range().contains(x) {
                continue;
            }
            let selected = self.param_sel.iter().any(|s| (s - t).abs() < KEY_EPSILON);
            let fill = if selected { KEY_SELECTED } else { Color32::from_rgb(170, 200, 230) };
            let r = KEY_R - 0.5;
            let stroke = Stroke::new(if hovered_param == Some(t) { 2.0 } else { 1.0 }, Color32::BLACK);
            painter.add(Shape::convex_polygon(vec![pos2(x, py - r), pos2(x + r, py), pos2(x, py + r), pos2(x - r, py)], fill, stroke));
        }
        if param_times.is_empty() {
            painter.text(
                pos2(tracks.left() + 8.0, py),
                egui::Align2::LEFT_CENTER,
                "hover a node value and press I to keyframe it",
                FontId::proportional(11.0),
                Color32::from_gray(110),
            );
        }

        // ---- interaction ----
        let mods = ui.input(|i| i.modifiers);
        let additive = mods.shift || mods.command;
        if !busy {
            if resp.drag_started_by(PointerButton::Primary) {
                let p = resp.interact_pointer_pos().unwrap_or_default();
                self.drag = if let Some(i) = hit_key(p) {
                    if !studio.keyframes[i].selected {
                        select_only(&mut studio.keyframes, i, additive);
                        if !additive {
                            self.param_sel.clear();
                        }
                    }
                    let orig = studio.keyframes.iter().enumerate().filter(|(_, k)| k.selected).map(|(i, k)| (i, k.time)).collect();
                    Drag::Keys { grab: t_of(p.x), orig }
                } else if let Some(t) = hit_param(p) {
                    if !self.param_sel.iter().any(|s| (s - t).abs() < KEY_EPSILON) {
                        if !additive {
                            self.param_sel.clear();
                            studio.keyframes.iter_mut().for_each(|k| k.selected = false);
                        }
                        self.param_sel.push(t);
                    }
                    let snapshot = graph.anim_snapshot();
                    Drag::Params { grab: t_of(p.x), snapshot, delta: 0.0 }
                } else if key_row.contains(p) || param_row.contains(p) {
                    Drag::Box { start: p }
                } else if tracks.contains(p) {
                    Drag::Scrub
                } else {
                    Drag::None
                };
            }
            // Clicks: on the ruler / waveform scrub, on a key select it.
            if resp.clicked() {
                let p = resp.interact_pointer_pos().unwrap_or_default();
                if let Some(i) = hit_key(p) {
                    if additive {
                        studio.keyframes[i].selected ^= true;
                    } else {
                        select_only(&mut studio.keyframes, i, false);
                        self.param_sel.clear();
                    }
                } else if let Some(t) = hit_param(p) {
                    let at = self.param_sel.iter().position(|s| (s - t).abs() < KEY_EPSILON);
                    if additive {
                        match at {
                            Some(i) => {
                                self.param_sel.remove(i);
                            }
                            None => self.param_sel.push(t),
                        }
                    } else {
                        self.param_sel = vec![t];
                        studio.keyframes.iter_mut().for_each(|k| k.selected = false);
                    }
                } else if key_row.contains(p) || param_row.contains(p) {
                    if !additive {
                        studio.keyframes.iter_mut().for_each(|k| k.selected = false);
                        self.param_sel.clear();
                    }
                } else if tracks.contains(p) {
                    *time = snap(t_of(p.x), fps).max(0.0);
                }
            }
            if resp.double_clicked() {
                if let Some(i) = resp.interact_pointer_pos().and_then(hit_key) {
                    *time = studio.keyframes[i].time;
                } else if let Some(t) = resp.interact_pointer_pos().and_then(hit_param) {
                    *time = t;
                }
            }
            match &mut self.drag {
                Drag::Params { grab, snapshot, delta } => {
                    if let Some(p) = pointer {
                        let min_sel = self.param_sel.iter().copied().fold(f32::INFINITY, f32::min);
                        let d = snap(t_of(p.x) - *grab, fps).max(-min_sel);
                        // Start from the keys as they were, so the move is exact.
                        graph.restore_anim(snapshot);
                        graph.move_keys(&self.param_sel, d);
                        *delta = d;
                        painter.text(
                            p + vec2(10.0, -14.0),
                            egui::Align2::LEFT_BOTTOM,
                            format!("{}{:.2} s", if d >= 0.0 { "+" } else { "" }, d),
                            FontId::proportional(12.0),
                            Color32::WHITE,
                        );
                    }
                }
                Drag::Scrub => {
                    if let Some(p) = pointer {
                        *time = snap(t_of(p.x), fps).max(0.0);
                    }
                }
                Drag::Keys { grab, orig } => {
                    if let Some(p) = pointer {
                        // Move by whole frames, keeping each key on its own frame grid.
                        let delta = snap(t_of(p.x) - *grab, fps);
                        let min_orig = orig.iter().map(|o| o.1).fold(f32::INFINITY, f32::min);
                        let delta = delta.max(-min_orig);
                        for &(i, t0) in orig.iter() {
                            studio.keyframes[i].time = t0 + delta;
                        }
                        changes.keys = true;
                        painter.text(
                            p + vec2(10.0, -14.0),
                            egui::Align2::LEFT_BOTTOM,
                            format!("{}{:.2} s", if delta >= 0.0 { "+" } else { "" }, delta),
                            FontId::proportional(12.0),
                            Color32::WHITE,
                        );
                    }
                }
                Drag::Box { start } => {
                    if let Some(p) = pointer {
                        let r = Rect::from_two_pos(*start, p);
                        painter.rect(r, 0.0, Color32::from_rgba_unmultiplied(120, 160, 255, 30), Stroke::new(1.0, Color32::from_rgb(120, 160, 255)), StrokeKind::Inside);
                    }
                }
                Drag::None => {}
            }
            if resp.drag_stopped() {
                match std::mem::take(&mut self.drag) {
                    Drag::Keys { .. } => finish_move(studio),
                    Drag::Params { delta, .. } => {
                        for s in &mut self.param_sel {
                            *s = (*s + delta).max(0.0);
                        }
                    }
                    Drag::Box { start } => {
                        let end_p = pointer.unwrap_or(start);
                        let (a, b) = (t_of(start.x.min(end_p.x)), t_of(start.x.max(end_p.x)));
                        let (y0, y1) = (start.y.min(end_p.y), start.y.max(end_p.y));
                        let covers = |r: Rect| y0 <= r.bottom() && y1 >= r.top();
                        if covers(key_row) || !additive {
                            for k in &mut studio.keyframes {
                                let inside = covers(key_row) && k.time >= a && k.time <= b;
                                k.selected = if additive { k.selected || inside } else { inside };
                            }
                        }
                        if !additive {
                            self.param_sel.clear();
                        }
                        if covers(param_row) {
                            for &t in param_times.iter().filter(|&&t| t >= a && t <= b) {
                                if !self.param_sel.iter().any(|s| (s - t).abs() < KEY_EPSILON) {
                                    self.param_sel.push(t);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Right-click menus.
            if resp.secondary_clicked() {
                let p = resp.interact_pointer_pos().unwrap_or_default();
                if let Some(i) = hit_key(p) {
                    if !studio.keyframes[i].selected {
                        select_only(&mut studio.keyframes, i, false);
                        self.param_sel.clear();
                    }
                } else if let Some(t) = hit_param(p) {
                    if !self.param_sel.iter().any(|s| (s - t).abs() < KEY_EPSILON) {
                        self.param_sel = vec![t];
                        studio.keyframes.iter_mut().for_each(|k| k.selected = false);
                    }
                }
                self.menu_t = snap(t_of(p.x), fps).max(0.0);
            }
            let menu_t = self.menu_t;
            let param_sel = &mut self.param_sel;
            resp.context_menu(|ui| {
                if !param_sel.is_empty() {
                    let n = param_sel.len();
                    ui.weak(if n == 1 { format!("node keys at {}", fmt_time(param_sel[0])) } else { format!("node keys at {n} times") });
                    if n == 1 && ui.button("Jump to keys").clicked() {
                        *time = param_sel[0];
                        ui.close();
                    }
                    ui.menu_button("Interpolation", |ui| {
                        for i in Interp::ALL {
                            if ui.button(i.label()).clicked() {
                                graph.set_interp_at(param_sel, i);
                                ui.close();
                            }
                        }
                    });
                    if ui.add(egui::Button::new("Delete").shortcut_text("Del")).clicked() {
                        let gone = graph.delete_keys_at(param_sel);
                        param_sel.clear();
                        changes.message = Some(format!("Deleted {gone} node key{}", if gone == 1 { "" } else { "s" }));
                        ui.close();
                    }
                    ui.separator();
                }
                let selected = studio.keyframes.iter().filter(|k| k.selected).count();
                if selected > 0 {
                    ui.weak(if selected == 1 { "1 key".to_string() } else { format!("{selected} keys") });
                    if selected == 1 && ui.button("Jump to key").clicked() {
                        *time = studio.keyframes.iter().find(|k| k.selected).unwrap().time;
                        ui.close();
                    }
                    if selected == 1 && ui.button("Set to current view").on_hover_text("Replace this key's camera with the current view").clicked() {
                        if let Some(k) = studio.keyframes.iter_mut().find(|k| k.selected) {
                            k.camera = *cam;
                        }
                        changes.keys = true;
                        ui.close();
                    }
                    ui.menu_button("Interpolation", |ui| {
                        for i in Interp::ALL {
                            if ui.button(i.label()).clicked() {
                                studio.keyframes.iter_mut().filter(|k| k.selected).for_each(|k| k.interp = i);
                                changes.keys = true;
                                ui.close();
                            }
                        }
                    });
                    if ui.add(egui::Button::new("Delete").shortcut_text("Del")).clicked() {
                        delete_selected(studio, changes);
                        ui.close();
                    }
                    ui.separator();
                }
                if ui.button(format!("Insert key at {}", fmt_time(menu_t))).on_hover_text("From the current view").clicked() {
                    insert_key(studio, menu_t, cam, changes);
                    ui.close();
                }
                if ui.button("Set range start here").clicked() {
                    let v = &mut studio.settings;
                    let end = v.start + v.duration;
                    v.start = menu_t.min(end - 0.1);
                    v.duration = end - v.start;
                    ui.close();
                }
                if ui.button("Set range end here").clicked() {
                    let v = &mut studio.settings;
                    v.duration = (menu_t - v.start).max(0.1);
                    ui.close();
                }
                if ui.button("Select all keys").clicked() {
                    studio.keyframes.iter_mut().for_each(|k| k.selected = true);
                    ui.close();
                }
            });
        }

        // Zoom / pan.
        let mut view = view;
        if resp.hovered() {
            let (scroll, shift) = ui.input(|i| (i.smooth_scroll_delta, i.modifiers.shift));
            let pan_px = if shift { scroll.y + scroll.x } else { scroll.x };
            if pan_px != 0.0 {
                let dt = -pan_px / tracks.width() * (view.1 - view.0);
                view = (view.0 + dt, view.1 + dt);
            }
            if !shift && scroll.y != 0.0 {
                let pivot = pointer.map_or((view.0 + view.1) * 0.5, |p| t_of(p.x));
                let k = (-scroll.y * 0.003).exp();
                let span = ((view.1 - view.0) * k).clamp(0.2, 7200.0);
                let f = (pivot - view.0) / (view.1 - view.0);
                view = (pivot - f * span, pivot + (1.0 - f) * span);
            }
        }
        if resp.dragged_by(PointerButton::Middle) {
            let dt = -resp.drag_delta().x / tracks.width() * (view.1 - view.0);
            view = (view.0 + dt, view.1 + dt);
        }
        if fit_view || (resp.hovered() && ui.input(|i| i.key_pressed(Key::Home))) {
            view = fit(start, end);
        }
        self.view = Some(view);

        // Playhead on top of everything.
        let px = x_of(*time);
        if tracks.x_range().contains(px) {
            painter.vline(px, tracks.y_range(), Stroke::new(1.5, PLAYHEAD));
            let text = format!("{}", (*time * fps).round() as i64);
            let galley = painter.layout_no_wrap(text, FontId::proportional(11.0), Color32::WHITE);
            let r = Rect::from_center_size(pos2(px, ruler.center().y), galley.size() + vec2(8.0, 4.0));
            painter.rect_filled(r, 3.0, PLAYHEAD);
            painter.galley(r.center() - galley.size() * 0.5, galley, Color32::WHITE);
        }

        // Delete over the timeline removes the selected keys.
        if !busy && resp.hovered() && !ui.ctx().egui_wants_keyboard_input() {
            if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Delete) || i.consume_key(Modifiers::NONE, Key::X)) {
                delete_selected(studio, changes);
                if !self.param_sel.is_empty() {
                    let gone = graph.delete_keys_at(&self.param_sel);
                    self.param_sel.clear();
                    changes.message = Some(format!("Deleted {gone} node key{}", if gone == 1 { "" } else { "s" }));
                }
            }
        }
        std::mem::take(changes)
    }

    /// App-wide timeline shortcuts (not while typing): I inserts a camera key, Left /
    /// Right step a frame (Shift: range start / end), Up / Down jump between keys.
    pub fn shortcuts(&mut self, ctx: &egui::Context, c: Ctx) -> Changes {
        let mut changes = Changes::default();
        if c.busy || ctx.egui_wants_keyboard_input() {
            return changes;
        }
        let fps = c.studio.settings.fps.max(1) as f32;
        let (start, end) = range(c.studio);
        let Ctx { time, studio, graph, cam, .. } = c;
        // Over a UI field, I belongs to the field (the node editor keys it); over the
        // viewport or the timeline tracks it keys the camera.
        let over_tracks = self.tracks_rect.zip(ctx.pointer_hover_pos()).is_some_and(|(r, p)| r.contains(p));
        let camera_i = !ctx.is_pointer_over_egui() || over_tracks;
        let keys = all_key_times(studio, graph);
        ctx.input_mut(|i| {
            if camera_i && i.consume_key(Modifiers::NONE, Key::I) {
                insert_key(studio, *time, cam, &mut changes);
            }
            if i.consume_key(Modifiers::SHIFT, Key::ArrowLeft) {
                *time = start;
            }
            if i.consume_key(Modifiers::SHIFT, Key::ArrowRight) {
                *time = end;
            }
            if i.consume_key(Modifiers::NONE, Key::ArrowLeft) {
                *time = (snap(*time, fps) - 1.0 / fps).max(0.0);
            }
            if i.consume_key(Modifiers::NONE, Key::ArrowRight) {
                *time = snap(*time, fps) + 1.0 / fps;
            }
            if i.consume_key(Modifiers::NONE, Key::ArrowUp) {
                if let Some(t) = next_key(&keys, *time) {
                    *time = t;
                }
            }
            if i.consume_key(Modifiers::NONE, Key::ArrowDown) {
                if let Some(t) = prev_key(&keys, *time) {
                    *time = t;
                }
            }
        });
        changes
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Icon {
    ToStart,
    PrevKey,
    Play,
    Pause,
    NextKey,
    ToEnd,
}

/// A transport button with its icon drawn as shapes (the bundled fonts lack these glyphs).
pub fn icon_button(ui: &mut egui::Ui, icon: Icon) -> egui::Response {
    let resp = ui.add(egui::Button::new("").min_size(vec2(26.0, 20.0)));
    let color = if ui.is_enabled() { ui.visuals().text_color() } else { ui.visuals().weak_text_color() };
    let p = ui.painter();
    let c = resp.rect.center();
    let tri = |dir: f32, x: f32| {
        let pts = vec![pos2(x - 4.0 * dir, c.y - 5.0), pos2(x + 4.0 * dir, c.y), pos2(x - 4.0 * dir, c.y + 5.0)];
        Shape::convex_polygon(pts, color, Stroke::NONE)
    };
    let diamond = |x: f32| {
        let pts = vec![pos2(x, c.y - 4.5), pos2(x + 4.5, c.y), pos2(x, c.y + 4.5), pos2(x - 4.5, c.y)];
        Shape::convex_polygon(pts, color, Stroke::NONE)
    };
    let bar = |x: f32| Rect::from_center_size(pos2(x, c.y), vec2(2.0, 10.0));
    match icon {
        Icon::Play => {
            p.add(tri(1.0, c.x + 1.0));
        }
        Icon::Pause => {
            p.rect_filled(Rect::from_center_size(pos2(c.x - 2.5, c.y), vec2(3.0, 10.0)), 0.0, color);
            p.rect_filled(Rect::from_center_size(pos2(c.x + 2.5, c.y), vec2(3.0, 10.0)), 0.0, color);
        }
        Icon::ToStart => {
            p.rect_filled(bar(c.x - 5.0), 0.0, color);
            p.add(tri(-1.0, c.x + 1.5));
        }
        Icon::ToEnd => {
            p.add(tri(1.0, c.x - 1.5));
            p.rect_filled(bar(c.x + 5.0), 0.0, color);
        }
        Icon::PrevKey => {
            p.add(tri(-1.0, c.x - 3.0));
            p.add(diamond(c.x + 4.5));
        }
        Icon::NextKey => {
            p.add(diamond(c.x - 4.5));
            p.add(tri(1.0, c.x + 3.0));
        }
    }
    resp
}

/// The scene range with a little margin.
fn fit(start: f32, end: f32) -> (f32, f32) {
    let pad = (end - start).max(0.5) * 0.04;
    (start - pad, end + pad)
}

/// Camera and node key times, sorted.
fn all_key_times(studio: &Studio, graph: &Graphs) -> Vec<f32> {
    let mut t: Vec<f32> = studio.keyframes.iter().map(|k| k.time).chain(graph.key_times()).collect();
    t.sort_by(f32::total_cmp);
    t
}

fn next_key(times: &[f32], t: f32) -> Option<f32> {
    times.iter().copied().find(|&k| k > t + KEY_EPSILON)
}

fn prev_key(times: &[f32], t: f32) -> Option<f32> {
    times.iter().rev().copied().find(|&k| k < t - KEY_EPSILON)
}

fn select_only(keys: &mut [Keyframe], i: usize, additive: bool) {
    for (j, k) in keys.iter_mut().enumerate() {
        if j == i {
            k.selected = true;
        } else if !additive {
            k.selected = false;
        }
    }
}

/// Keys the current view at `t`. The first key turns camera animation on.
fn insert_key(studio: &mut Studio, t: f32, cam: &Camera, changes: &mut Changes) {
    let first = studio.keyframes.is_empty();
    let replaced = studio.keyframes.iter().any(|k| (k.time - t).abs() < KEY_EPSILON);
    studio.add_keyframe(t, *cam);
    if first {
        studio.settings.camera = CameraMode::Keyframes;
    }
    changes.keys = true;
    changes.message = Some(format!("{} camera key at {}", if replaced { "Updated" } else { "Inserted" }, fmt_time(t)));
}

fn delete_selected(studio: &mut Studio, changes: &mut Changes) {
    let before = studio.keyframes.len();
    studio.keyframes.retain(|k| !k.selected);
    let n = before - studio.keyframes.len();
    if n > 0 {
        changes.keys = true;
        changes.message = Some(format!("Deleted {n} key{}", if n == 1 { "" } else { "s" }));
    }
}

/// After a move: re-sort, and where a moved key landed on another key, the moved one
/// wins (like Blender).
fn finish_move(studio: &mut Studio) {
    let keys = &mut studio.keyframes;
    let moved: Vec<f32> = keys.iter().filter(|k| k.selected).map(|k| k.time).collect();
    keys.retain(|k| k.selected || !moved.iter().any(|&t| (t - k.time).abs() < KEY_EPSILON));
    sort_keys(keys);
}

/// Draws the camera keyframe path into the viewport (behind all windows), as seen
/// from `view`: the path, a dot per key (selected ones orange) and each key's view
/// direction.
pub fn draw_path(ctx: &egui::Context, keys: &[Keyframe], view: &Camera, dim: bool) {
    let (Some(first), Some(last)) = (keys.first(), keys.last()) else { return };
    let screen = ctx.content_rect();
    let (fwd, right, up) = view.basis();
    let tan = (view.fov_deg.to_radians() * 0.5).tan();
    let aspect = screen.width() / screen.height();
    // Same mapping as `camera_ray` in fractal.wgsl, inverted.
    let project = |p: glam::Vec3| -> Option<Pos2> {
        let d = p - view.pos;
        let z = d.dot(fwd);
        if z < 1e-5 {
            return None;
        }
        let x = d.dot(right) / (z * tan) / aspect;
        let y = d.dot(up) / (z * tan);
        Some(pos2(screen.left() + (x + 1.0) * 0.5 * screen.width(), screen.top() + (1.0 - y) * 0.5 * screen.height()))
    };
    let painter = ctx.layer_painter(egui::LayerId::background());
    let alpha = if dim { 0.35 } else { 1.0 };
    let line = Stroke::new(2.0, Color32::from_rgb(255, 210, 120).gamma_multiply(0.8 * alpha));
    let samples = 64 * keys.len().max(2);
    let mut prev: Option<Pos2> = None;
    for s in 0..=samples {
        let t = first.time + (last.time - first.time) * s as f32 / samples as f32;
        let p = keyframe_camera(keys, t).and_then(|c| project(c.pos));
        if let (Some(a), Some(b)) = (prev, p) {
            // Skip segments that wrap around behind the camera.
            if a.distance(b) < screen.width() {
                painter.line_segment([a, b], line);
            }
        }
        prev = p;
    }
    for k in keys {
        let Some(p) = project(k.camera.pos) else { continue };
        let color = if k.selected { KEY_SELECTED } else { Color32::WHITE };
        let len = 0.15 * (k.camera.pos - view.pos).length();
        if let Some(tip) = project(k.camera.pos + k.camera.basis().0 * len) {
            painter.line_segment([p, tip], Stroke::new(1.5, color.gamma_multiply(0.8 * alpha)));
        }
        painter.circle(p, 5.0, color.gamma_multiply(alpha), Stroke::new(1.0, Color32::BLACK.gamma_multiply(alpha)));
    }
}
