//! Short-lived notifications at the bottom of the view ("Saved …", "Undo", camera
//! speed). Shown even while the rest of the UI is hidden. Things that need reading or
//! a decision go in a message box instead (see `modal`).

use std::time::{Duration, Instant};

use egui::{Color32, Margin, Shadow, Stroke, vec2};

use crate::theme;

const LIFETIME: Duration = Duration::from_millis(2500);
const FADE: f32 = 0.4;
const APPEAR: f32 = 0.18;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Info,
    Success,
    Error,
}

struct Toast {
    /// Toasts with the same key replace each other (e.g. repeated speed changes).
    key: &'static str,
    text: String,
    kind: Kind,
    shown: Instant,
    life: Duration,
}

#[derive(Default)]
pub struct Toasts {
    items: Vec<Toast>,
}

impl Toasts {
    pub fn info(&mut self, key: &'static str, text: impl Into<String>) {
        self.push(key, text.into(), Kind::Info, LIFETIME);
    }

    /// Something finished well (saved, exported): green accent.
    pub fn success(&mut self, key: &'static str, text: impl Into<String>) {
        self.push(key, text.into(), Kind::Success, LIFETIME);
    }

    /// Errors stay twice as long.
    pub fn error(&mut self, key: &'static str, text: impl Into<String>) {
        self.push(key, text.into(), Kind::Error, LIFETIME * 2);
    }

    /// Reports a result: the success message, or the error.
    pub fn result(&mut self, key: &'static str, r: Result<String, String>) {
        match r {
            Ok(t) => self.success(key, t),
            Err(e) => self.error(key, e),
        }
    }

    fn push(&mut self, key: &'static str, text: String, kind: Kind, life: Duration) {
        // A replaced toast (same key) keeps its place on screen instead of popping in again.
        let shown = match self.items.iter().position(|t| t.key == key) {
            Some(i) => {
                let old = self.items.remove(i);
                let age = old.shown.elapsed().as_secs_f32().min(APPEAR);
                Instant::now() - Duration::from_secs_f32(age)
            }
            None => Instant::now(),
        };
        self.items.push(Toast { key, text, kind, shown, life });
    }

    /// Stacks the toasts centered just above `bottom` (the top of the timeline, if shown).
    pub fn show(&mut self, ctx: &egui::Context, area: egui::Rect) {
        self.items.retain(|t| t.shown.elapsed() < t.life);
        let Some(newest) = self.items.last() else { return };
        // The stack rises into place as the newest toast appears.
        let rise = 1.0 - egui::emath::easing::cubic_out((newest.shown.elapsed().as_secs_f32() / APPEAR).min(1.0));
        egui::Area::new(egui::Id::unique("toasts"))
            .pivot(egui::Align2::CENTER_BOTTOM)
            .fixed_pos(egui::pos2(area.center().x, area.bottom() - 18.0 + rise * 10.0))
            .interactable(false)
            .order(egui::Order::Tooltip)
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| self.draw(ui, (area.width() - 48.0).clamp(200.0, 760.0)));
            });
        ctx.request_repaint();
    }

    fn draw(&self, ui: &mut egui::Ui, max_width: f32) {
        ui.spacing_mut().item_spacing.y = 6.0;
        for t in &self.items {
            let left = t.life.as_secs_f32() - t.shown.elapsed().as_secs_f32();
            let appear = (t.shown.elapsed().as_secs_f32() / APPEAR).min(1.0);
            let accent = match t.kind {
                Kind::Info => theme::ACCENT,
                Kind::Success => theme::SUCCESS,
                Kind::Error => theme::ERROR,
            };
            ui.scope(|ui| {
                ui.multiply_opacity((left / FADE).min(1.0) * appear);
                let frame = egui::Frame::new()
                    .fill(theme::CARD)
                    .stroke(Stroke::new(1.0, theme::CARD_STROKE))
                    .corner_radius(theme::RADIUS)
                    .shadow(Shadow { offset: [0, 4], blur: 14, spread: 0, color: Color32::from_black_alpha(110) })
                    .inner_margin(Margin { left: 16, right: 14, top: 7, bottom: 7 });
                let r = frame.show(ui, |ui| {
                    let color = if t.kind == Kind::Error { theme::lighten(theme::ERROR, 0.35) } else { theme::TEXT_STRONG };
                    // Laid out here so the frame hugs the text, wrapping only past max_width.
                    let font = egui::TextStyle::Body.resolve(ui.style());
                    let galley = ui.painter().layout(t.text.clone(), font, color, max_width);
                    ui.label(galley);
                });
                // Accent bar on the left edge.
                let rect = r.response.rect;
                let bar = egui::Rect::from_min_size(rect.min + vec2(5.0, 6.0), vec2(3.0, rect.height() - 12.0));
                ui.painter().rect_filled(bar, 2, accent);
            });
        }
    }
}
