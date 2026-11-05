//! The Audio panel: load a song, see its beats and drum hits around the playhead,
//! scrub, and tune tempo / hit detection / volume.

use egui::{Color32, Pos2, Rect, Sense, Stroke, pos2, vec2};

use super::analysis::{Analysis, BAND_COUNT};
use super::{Audio, beat_outputs};
use crate::graph::AUDIO_BANDS;

/// Seconds shown in the scrolling timeline, and where the playhead sits in it.
const SPAN: f32 = 6.0;
const PLAYHEAD: f32 = 0.35;

const BEAT: Color32 = Color32::from_rgb(120, 170, 255);
const KICK: Color32 = Color32::from_rgb(255, 120, 70);
const SNARE: Color32 = Color32::from_rgb(255, 220, 90);
const HAT: Color32 = Color32::from_rgb(110, 230, 200);

/// `clock_ok`: playback can follow the animation (speed 1, not exporting).
pub fn panel(ui: &mut egui::Ui, audio: &mut Audio, anim_time: &mut f32, clock_ok: bool) {
    ui.horizontal(|ui| {
        if ui.button("Open song…").on_hover_text("FLAC, MP3 or WAV. You can also drop a file on the window").clicked() {
            audio.pick_file();
        }
        if audio.track.is_some() && ui.button("Remove").clicked() {
            audio.unload();
        }
    });
    if let Some(path) = audio.loading() {
        let name = path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(format!("Analyzing {name}…"));
        });
    }
    let Some(track) = &audio.track else {
        ui.small("No song loaded: Audio Level and Beat nodes output 0 until one is.");
        return;
    };

    let a = &track.analysis;
    let offset = audio.settings.offset;
    let song_t = *anim_time - offset;
    ui.horizontal(|ui| {
        ui.strong(&track.name);
        ui.weak(format!("{} / {}", mmss(song_t), mmss(a.duration)));
    });
    if let Some(t) = timeline(ui, a, song_t) {
        *anim_time = t + offset;
    }
    meters(ui, a, song_t);

    let s = &mut audio.settings;
    let mut reanalyze = false;
    ui.horizontal(|ui| {
        ui.label("tempo");
        let mut bpm = a.bpm;
        let mut changed = ui.add(egui::DragValue::new(&mut bpm).range(40.0..=300.0).speed(0.1).max_decimals(1).suffix(" BPM")).changed();
        if ui.small_button("½").on_hover_text("Half tempo").clicked() {
            bpm *= 0.5;
            changed = true;
        }
        if ui.small_button("×2").on_hover_text("Double tempo").clicked() {
            bpm *= 2.0;
            changed = true;
        }
        if changed {
            s.bpm = Some(bpm);
            reanalyze = true;
        }
        if s.bpm.is_some() && ui.small_button("auto").on_hover_text(format!("Detected: {:.1} BPM", a.detected_bpm)).clicked() {
            s.bpm = None;
            reanalyze = true;
        }
    });
    reanalyze |= ui
        .add(egui::Slider::new(&mut s.sensitivity, 0.0..=1.0).text("hit sensitivity"))
        .on_hover_text("Higher finds quieter kicks / snares / hi-hats")
        .changed();
    ui.horizontal(|ui| {
        ui.add_enabled(!s.muted, egui::Slider::new(&mut s.volume, 0.0..=1.0).text("volume"));
        ui.checkbox(&mut s.muted, "mute");
    });
    ui.horizontal(|ui| {
        ui.label("song starts at t =");
        ui.add(egui::DragValue::new(&mut s.offset).speed(0.01).suffix(" s").max_decimals(3));
    });
    if !clock_ok {
        ui.small("Playback is silent unless the animation speed is 1.");
    }
    if reanalyze {
        audio.reanalyze();
    }
}

fn mmss(t: f32) -> String {
    let s = t.max(0.0) as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Scrolling view around the playhead plus a whole-song overview; returns a new song
/// time when the user scrubs or clicks.
fn timeline(ui: &mut egui::Ui, a: &Analysis, song_t: f32) -> Option<f32> {
    let width = ui.available_width().max(120.0);
    let mut seek = None;

    let (rect, resp) = ui.allocate_exact_size(vec2(width, 52.0), Sense::click_and_drag());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, Color32::from_gray(18));
    let t0 = song_t - SPAN * PLAYHEAD;
    let x_of = |t: f32| rect.left() + (t - t0) / SPAN * rect.width();
    let t_of = |x: f32| t0 + (x - rect.left()) / rect.width() * SPAN;
    // Loudness (full band) and bass underneath.
    for (band, color) in [(0, Color32::from_gray(70)), (1, Color32::from_rgb(120, 60, 50))] {
        let mut x = rect.left();
        while x < rect.right() {
            let h = a.level(band, t_of(x)) * (rect.height() - 6.0);
            p.line_segment([pos2(x, rect.bottom()), pos2(x, rect.bottom() - h)], Stroke::new(2.0, color));
            x += 2.0;
        }
    }
    let visible = |ev: &[f32]| {
        let (lo, hi) = (ev.partition_point(|&e| e < t0), ev.partition_point(|&e| e < t0 + SPAN));
        ev[lo..hi].to_vec()
    };
    for t in visible(&a.beats) {
        let bar = a.bars.binary_search_by(|b| b.total_cmp(&t)).is_ok();
        let stroke = if bar { Stroke::new(2.0, BEAT) } else { Stroke::new(1.0, BEAT.gamma_multiply(0.5)) };
        p.vline(x_of(t), rect.y_range(), stroke);
    }
    for (hits, color, y) in [(&a.hits[2], HAT, 0.2), (&a.hits[1], SNARE, 0.5), (&a.hits[0], KICK, 0.8)] {
        for t in visible(hits) {
            p.circle_filled(pos2(x_of(t), rect.top() + rect.height() * y), 2.5, color);
        }
    }
    let head = x_of(song_t);
    p.vline(head, rect.y_range(), Stroke::new(2.0, Color32::WHITE));
    if resp.dragged() {
        seek = Some(song_t - resp.drag_delta().x / rect.width() * SPAN);
    } else if resp.clicked() {
        seek = resp.interact_pointer_pos().map(|pos| t_of(pos.x));
    }
    resp.on_hover_text("Drag to scrub. Lines: beats (bright = bar). Dots: hi-hat, snare, kick");

    // Overview of the whole song: click or drag to jump.
    let (rect, resp) = ui.allocate_exact_size(vec2(width, 18.0), Sense::click_and_drag());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3.0, Color32::from_gray(14));
    let points = (rect.width() / 2.0) as usize;
    for (i, v) in a.overview(0, points).into_iter().enumerate() {
        let x = rect.left() + i as f32 * 2.0 + 1.0;
        let h = v * (rect.height() - 2.0);
        p.line_segment([pos2(x, rect.center().y - h * 0.5), pos2(x, rect.center().y + h * 0.5)], Stroke::new(2.0, Color32::from_gray(90)));
    }
    let frac = |x: f32| ((x - rect.left()) / rect.width()).clamp(0.0, 1.0);
    let window = Rect::from_x_y_ranges(
        rect.left() + (t0 / a.duration).clamp(0.0, 1.0) * rect.width()..=rect.left() + ((t0 + SPAN) / a.duration).clamp(0.0, 1.0) * rect.width(),
        rect.y_range(),
    );
    p.rect_stroke(window, 2.0, Stroke::new(1.0, Color32::from_gray(150)), egui::StrokeKind::Inside);
    if let Some(pos) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
        seek = Some(frac(pos.x) * a.duration);
    }
    seek
}

/// Band level meters and beat / bar lights at the playhead.
fn meters(ui: &mut egui::Ui, a: &Analysis, song_t: f32) {
    ui.horizontal(|ui| {
        for band in 0..BAND_COUNT {
            let (rect, resp) = ui.allocate_exact_size(vec2(10.0, 26.0), Sense::hover());
            let p = ui.painter();
            p.rect_filled(rect, 2.0, Color32::from_gray(25));
            let level = a.envelope(band, 0.01, 0.15, song_t);
            let fill = Rect::from_min_max(pos2(rect.left(), rect.bottom() - level * rect.height()), rect.max);
            p.rect_filled(fill, 2.0, Color32::from_rgb(90, 200, 120).lerp_to_gamma(Color32::from_rgb(255, 90, 60), level));
            resp.on_hover_text(AUDIO_BANDS[band]);
        }
        ui.add_space(8.0);
        let beat = beat_outputs(&a.beats, 1, 0.25, song_t);
        let bar = beat_outputs(&a.bars, 1, 0.5, song_t);
        for (pulse, color, label) in [(beat[0], BEAT, "beat"), (bar[0], Color32::WHITE, "bar")] {
            let (rect, resp) = ui.allocate_exact_size(vec2(18.0, 26.0), Sense::hover());
            let c: Pos2 = rect.center();
            ui.painter().circle(c, 7.0, color.gamma_multiply(0.15 + 0.85 * pulse), Stroke::new(1.0, color.gamma_multiply(0.5)));
            resp.on_hover_text(label);
        }
        ui.small(format!("beat {}  bar {}", beat[1] as u32, bar[1] as u32));
    });
    ui.weak(format!("{} kicks, {} snares, {} hi-hats found", a.hits[0].len(), a.hits[1].len(), a.hits[2].len()));
}
