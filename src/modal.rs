//! Screen-wide message boxes: the whole window dims behind a centered card with an
//! icon, title, message and buttons. For things worth stopping for (what happened
//! after a crash, a finished export, quitting mid-export); passing notices stay toasts.
//!
//! Messages queue and show one at a time. The card eases in over the fading backdrop
//! and out again once answered; Enter picks the highlighted button, Esc cancels, and a
//! click outside gives the card a small shake instead of dismissing it by accident.
//! While one is open the app ignores its own keys and mouse (see `is_open`).

use std::collections::VecDeque;
use std::path::PathBuf;

use egui::{Align, Align2, Color32, CornerRadius, Frame, Layout, Margin, RichText, Shadow, Stroke, vec2};

use crate::theme;

/// Seconds for the card to ease in or out.
const EASE_TIME: f32 = 0.2;
/// The UI is blended in linear light, so this dims the view to roughly half its brightness.
const BACKDROP_ALPHA: f32 = 210.0;
const WIDTH: f32 = 440.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Info,
    Success,
    Warning,
    Error,
}

impl Kind {
    fn color(self) -> Color32 {
        match self {
            Kind::Info => theme::ACCENT,
            Kind::Success => theme::SUCCESS,
            Kind::Warning => theme::WARNING,
            Kind::Error => theme::ERROR,
        }
    }
}

/// What a button asks the app to do (handled in `App::modal_action`).
#[derive(Clone, PartialEq, Debug)]
pub enum Action {
    Close,
    /// Crash recovery: undo the cut, linking the graph's outputs again.
    ReconnectOutputs,
    /// Opens a file or folder with its default app.
    Open(PathBuf),
    /// Shows a file selected in Explorer.
    Reveal(PathBuf),
    /// Saves the project, replacing the file that's already there.
    OverwriteProject,
    /// Stops a running export and quits.
    Quit,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Normal,
    Primary,
    Danger,
}

struct Button {
    label: String,
    action: Action,
    role: Role,
}

pub struct Message {
    /// Pushing a message whose key is already queued replaces nothing (no duplicates).
    key: &'static str,
    kind: Kind,
    title: String,
    body: String,
    /// Two-column table under the text (e.g. keyboard shortcuts).
    rows: Vec<(String, String)>,
    /// Monospace text in a collapsed "Details" section (error output, a panic).
    details: Option<String>,
    /// Shown left to right; the first Primary one answers Enter (never a Danger one).
    buttons: Vec<Button>,
}

impl Message {
    pub fn new(key: &'static str, kind: Kind, title: impl Into<String>, body: impl Into<String>) -> Self {
        Self { key, kind, title: title.into(), body: body.into(), rows: Vec::new(), details: None, buttons: Vec::new() }
    }

    pub fn rows(mut self, rows: Vec<(String, String)>) -> Self {
        self.rows = rows;
        self
    }

    pub fn details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    pub fn button(self, label: impl Into<String>, action: Action) -> Self {
        self.with(label, action, Role::Normal)
    }

    /// The highlighted button.
    pub fn primary(self, label: impl Into<String>, action: Action) -> Self {
        self.with(label, action, Role::Primary)
    }

    /// A highlighted button for something destructive (red).
    pub fn danger(self, label: impl Into<String>, action: Action) -> Self {
        self.with(label, action, Role::Danger)
    }

    fn with(mut self, label: impl Into<String>, action: Action, role: Role) -> Self {
        self.buttons.push(Button { label: label.into(), action, role });
        self
    }

    /// Enter: the highlighted button, so a reflexive Enter never does something destructive.
    fn default_action(&self) -> Action {
        self.buttons.iter().find(|b| b.role == Role::Primary).map_or(Action::Close, |b| b.action.clone())
    }

    /// Esc: the button that just closes, else nothing destructive happens either.
    fn cancel_action(&self) -> Action {
        let close = self.buttons.iter().any(|b| b.action == Action::Close);
        if close || self.buttons.len() != 1 { Action::Close } else { self.buttons[0].action.clone() }
    }
}

#[derive(Default)]
pub struct Modals {
    queue: VecDeque<Message>,
    /// The front message was answered and is easing out.
    leaving: bool,
    /// Card visibility, 0..1 (eased when drawn).
    card: f32,
    backdrop: f32,
    /// Shake after a click outside the card, 1 → 0.
    shake: f32,
}

impl Modals {
    pub fn push(&mut self, mut m: Message) {
        if self.queue.iter().any(|q| q.key == m.key) {
            return;
        }
        if m.buttons.is_empty() {
            m = m.primary("OK", Action::Close);
        }
        self.queue.push_back(m);
    }

    /// A message is showing (or easing out): the app should ignore its own input.
    pub fn is_open(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Draws the front message; returns the action of a button that was just chosen.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Action> {
        if self.queue.is_empty() {
            self.backdrop = 0.0;
            return None;
        }
        let dt = ctx.input(|i| i.stable_dt).min(0.05);
        let step = |v: &mut f32, up: bool| {
            let delta = if up { dt } else { -dt } / EASE_TIME;
            *v = (*v + delta).clamp(0.0, 1.0);
        };
        let last = self.leaving && self.queue.len() == 1;
        step(&mut self.card, !self.leaving);
        step(&mut self.backdrop, !last);
        self.shake = (self.shake - dt * 2.5).max(0.0);
        if self.leaving && self.card == 0.0 {
            self.queue.pop_front();
            self.leaving = false;
            if self.queue.is_empty() {
                return None;
            }
        }
        ctx.request_repaint();

        let msg = self.queue.front().expect("checked above");
        let t = egui::emath::easing::cubic_out(self.card);
        let shake = (self.shake * 26.0).sin() * 7.0 * self.shake;
        let id = egui::Id::unique("fractalful modal");
        let area = egui::Modal::default_area(id).anchor(Align2::CENTER_CENTER, vec2(shake, (1.0 - t) * 18.0));
        let backdrop = Color32::from_black_alpha((egui::emath::easing::quadratic_out(self.backdrop) * BACKDROP_ALPHA) as u8);
        let response = egui::Modal::new(id).area(area).backdrop_color(backdrop).frame(Frame::NONE).show(ctx, |ui| {
            ui.multiply_opacity(t);
            card(ui, msg)
        });

        if self.leaving {
            return None;
        }
        let mut chosen = response.inner;
        if response.backdrop_response.clicked() {
            self.shake = 1.0;
        }
        if chosen.is_none() && response.is_top_modal {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
                chosen = Some(msg.cancel_action());
            } else if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)) {
                chosen = Some(msg.default_action());
            }
        }
        if chosen.is_some() {
            self.leaving = true;
        }
        chosen
    }
}

fn card(ui: &mut egui::Ui, msg: &Message) -> Option<Action> {
    let accent = msg.kind.color();
    let frame = Frame::new()
        .fill(theme::CARD)
        .stroke(Stroke::new(1.0, theme::CARD_STROKE))
        .corner_radius(theme::RADIUS + 1)
        .shadow(Shadow { offset: [0, 18], blur: 48, spread: 2, color: Color32::from_black_alpha(150) })
        .inner_margin(Margin { left: 24, right: 24, top: 24, bottom: 18 });
    let mut chosen = None;
    let shown = frame.show(ui, |ui| {
        ui.set_width(WIDTH);
        ui.horizontal(|ui| {
            icon(ui, msg.kind);
            ui.add_space(4.0);
            ui.label(RichText::new(&msg.title).size(19.0).color(theme::TEXT_STRONG));
        });
        ui.add_space(10.0);
        if !msg.body.is_empty() {
            ui.label(RichText::new(&msg.body).size(14.0).color(theme::TEXT).line_height(Some(20.0)));
        }
        if !msg.rows.is_empty() {
            ui.add_space(4.0);
            egui::ScrollArea::vertical().max_height(ui.ctx().content_rect().height() * 0.55).show(ui, |ui| {
                let col = (WIDTH - 20.0) / 2.0;
                egui::Grid::new("modal rows").num_columns(2).min_col_width(col).spacing([20.0, 6.0]).striped(true).show(ui, |ui| {
                    for (a, b) in &msg.rows {
                        ui.label(RichText::new(a).strong().color(theme::TEXT_STRONG));
                        ui.label(RichText::new(b).color(theme::TEXT));
                        ui.end_row();
                    }
                });
            });
        }
        if let Some(details) = &msg.details {
            ui.add_space(6.0);
            egui::CollapsingHeader::new(RichText::new("Details").color(theme::TEXT)).show(ui, |ui| {
                egui::Frame::new().fill(ui.visuals().extreme_bg_color).corner_radius(theme::RADIUS).inner_margin(8).show(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(170.0).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.add(egui::Label::new(RichText::new(details).monospace().size(11.5)).wrap());
                    });
                });
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(details.clone());
                }
            });
        }
        ui.add_space(18.0);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            for b in msg.buttons.iter().rev() {
                if button(ui, b) {
                    chosen = Some(b.action.clone());
                }
            }
        });
    });
    // Accent line along the top edge, between the rounded corners.
    let r = shown.response.rect;
    let strip = egui::Rect::from_min_max(r.min + vec2(16.0, 0.5), egui::pos2(r.max.x - 16.0, r.min.y + 3.5));
    ui.painter().rect_filled(strip, CornerRadius { nw: 0, ne: 0, sw: 2, se: 2 }, accent);
    chosen
}

fn icon(ui: &mut egui::Ui, kind: Kind) {
    let (rect, _) = ui.allocate_exact_size(vec2(30.0, 30.0), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    let color = kind.color();
    p.circle(c, 14.0, color.gamma_multiply(0.18), Stroke::new(1.5, color));
    let stroke = Stroke::new(2.2, color);
    match kind {
        Kind::Success => {
            p.line_segment([c + vec2(-6.0, 0.5), c + vec2(-1.8, 4.8)], stroke);
            p.line_segment([c + vec2(-1.8, 4.8), c + vec2(6.5, -4.5)], stroke);
        }
        Kind::Error => {
            p.line_segment([c + vec2(-4.5, -4.5), c + vec2(4.5, 4.5)], stroke);
            p.line_segment([c + vec2(4.5, -4.5), c + vec2(-4.5, 4.5)], stroke);
        }
        // "!" for warnings, "i" for info, drawn so they don't depend on font glyphs.
        Kind::Warning | Kind::Info => {
            let (dot, bar) = if kind == Kind::Warning { (5.5, [-6.5, 1.5]) } else { (-5.5, [-1.5, 6.5]) };
            p.circle_filled(c + vec2(0.0, dot), 1.6, color);
            p.line_segment([c + vec2(0.0, bar[0]), c + vec2(0.0, bar[1])], stroke);
        }
    }
}

fn button(ui: &mut egui::Ui, b: &Button) -> bool {
    let fill = match b.role {
        Role::Normal => None,
        Role::Primary => Some(theme::ACCENT),
        Role::Danger => Some(theme::ERROR),
    };
    ui.scope(|ui| {
        if let Some(fill) = fill {
            let w = &mut ui.visuals_mut().widgets;
            w.inactive.weak_bg_fill = fill.gamma_multiply(0.85);
            w.hovered.weak_bg_fill = fill;
            w.active.weak_bg_fill = theme::lighten(fill, 0.15);
            w.hovered.bg_stroke = Stroke::new(1.0, theme::lighten(fill, 0.4));
            for v in [&mut w.inactive, &mut w.hovered, &mut w.active] {
                v.fg_stroke.color = Color32::WHITE;
            }
        }
        let text = RichText::new(&b.label).size(14.0);
        ui.add(egui::Button::new(text).min_size(vec2(96.0, 32.0)).corner_radius(theme::RADIUS)).clicked()
    })
    .inner
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_choose_sensible_buttons() {
        let m = Message::new("a", Kind::Warning, "t", "b").danger("Stop and quit", Action::Quit).primary("Keep", Action::Close);
        assert_eq!(m.default_action(), Action::Close, "Enter takes the primary button, never the destructive one");
        assert_eq!(m.cancel_action(), Action::Close);
        let replace = Message::new("c", Kind::Warning, "t", "b").button("Cancel", Action::Close).danger("Replace", Action::OverwriteProject);
        assert_eq!(replace.default_action(), Action::Close);
        let ok = Message::new("b", Kind::Info, "t", "b").primary("Open", Action::Open("x".into()));
        assert_eq!(ok.cancel_action(), Action::Open("x".into()), "a lone button is also the cancel");
    }

    #[test]
    fn same_key_is_not_queued_twice() {
        let mut m = Modals::default();
        m.push(Message::new("quit", Kind::Warning, "t", "b"));
        m.push(Message::new("quit", Kind::Warning, "t", "b"));
        assert_eq!(m.queue.len(), 1);
        assert_eq!(m.queue[0].buttons.len(), 1, "a message without buttons gets OK");
    }
}
