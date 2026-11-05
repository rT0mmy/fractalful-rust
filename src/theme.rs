//! The app's look: OLED black with one electric-blue accent, near-square corners, hairline
//! borders and no soft shadows. Applied to every egui context (main window and the
//! popped-out node editor) so they match.

use egui::style::HandleShape;
use egui::{Color32, CornerRadius, Margin, Shadow, Stroke, vec2};

pub const ACCENT: Color32 = Color32::from_rgb(46, 140, 255);
pub const SUCCESS: Color32 = Color32::from_rgb(70, 214, 150);
pub const WARNING: Color32 = Color32::from_rgb(255, 184, 64);
pub const ERROR: Color32 = Color32::from_rgb(255, 84, 96);

pub const TEXT: Color32 = Color32::from_rgb(200, 204, 212);
pub const TEXT_STRONG: Color32 = Color32::from_rgb(246, 248, 252);
/// Message box / toast background.
pub const CARD: Color32 = Color32::from_rgb(6, 7, 9);
pub const CARD_STROKE: Color32 = Color32::from_rgb(38, 44, 54);
/// Corner radius of widgets, cards and nodes: just enough to soften the pixel corner.
pub const RADIUS: u8 = 2;
/// Hairline border separating black surfaces from the black around them.
pub const BORDER: Color32 = Color32::from_rgb(34, 38, 46);

pub fn apply(ctx: &egui::Context) {
    // The renderer's UI layer is made for a dark theme; don't follow the OS setting.
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |s| {
        s.animation_time = 0.08;

        let sp = &mut s.spacing;
        sp.item_spacing = vec2(8.0, 5.0);
        sp.button_padding = vec2(9.0, 3.0);
        sp.window_margin = Margin::same(10);
        sp.menu_margin = Margin::same(6);
        sp.interact_size.y = 22.0;
        sp.slider_width = 120.0;
        sp.slider_rail_height = 4.0;

        let v = &mut s.visuals;
        let radius = CornerRadius::same(RADIUS);
        v.window_corner_radius = CornerRadius::same(RADIUS + 1);
        v.menu_corner_radius = radius;
        v.window_fill = Color32::BLACK;
        v.panel_fill = Color32::BLACK;
        v.window_stroke = Stroke::new(1.0, BORDER);
        // A tight, hard-edged shadow: depth without the soft glow.
        v.window_shadow = Shadow { offset: [0, 2], blur: 6, spread: 0, color: Color32::from_black_alpha(160) };
        v.popup_shadow = Shadow { offset: [0, 2], blur: 4, spread: 0, color: Color32::from_black_alpha(160) };
        v.extreme_bg_color = Color32::from_rgb(4, 5, 7);
        v.faint_bg_color = Color32::from_rgb(10, 12, 15);
        v.code_bg_color = Color32::from_rgb(14, 17, 22);
        v.hyperlink_color = Color32::from_rgb(110, 180, 255);
        v.warn_fg_color = WARNING;
        v.error_fg_color = ERROR;
        v.selection.bg_fill = Color32::from_rgb(18, 74, 150);
        v.selection.stroke = Stroke::new(1.0, Color32::from_rgb(214, 234, 255));
        // Sliders fill up to the handle in the selection color.
        v.slider_trailing_fill = true;
        v.handle_shape = HandleShape::Rect { aspect_ratio: 0.5 };

        let w = &mut v.widgets;
        w.noninteractive.bg_fill = Color32::from_rgb(8, 9, 12);
        w.noninteractive.weak_bg_fill = Color32::from_rgb(8, 9, 12);
        w.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        w.noninteractive.corner_radius = radius;

        w.inactive.bg_fill = Color32::from_rgb(20, 23, 29);
        w.inactive.weak_bg_fill = Color32::from_rgb(14, 16, 21);
        w.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(30, 34, 42));
        w.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(212, 216, 224));
        w.inactive.corner_radius = radius;

        w.hovered.bg_fill = Color32::from_rgb(22, 34, 52);
        w.hovered.weak_bg_fill = Color32::from_rgb(18, 28, 44);
        w.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
        w.hovered.fg_stroke = Stroke::new(1.5, TEXT_STRONG);
        w.hovered.corner_radius = radius;
        w.hovered.expansion = 0.0;

        w.active.bg_fill = Color32::from_rgb(16, 82, 170);
        w.active.weak_bg_fill = Color32::from_rgb(14, 70, 150);
        w.active.bg_stroke = Stroke::new(1.0, Color32::from_rgb(120, 190, 255));
        w.active.fg_stroke = Stroke::new(2.0, Color32::WHITE);
        w.active.corner_radius = radius;
        w.active.expansion = 0.0;

        w.open.bg_fill = Color32::from_rgb(12, 16, 22);
        w.open.weak_bg_fill = Color32::from_rgb(14, 20, 30);
        w.open.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.6));
        w.open.fg_stroke = Stroke::new(1.0, TEXT_STRONG);
        w.open.corner_radius = radius;
    });
}

/// `c` moved toward white by `t` (0..1), keeping its alpha.
pub fn lighten(c: Color32, t: f32) -> Color32 {
    let mix = |a: u8| (a as f32 + (255.0 - a as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(mix(c.r()), mix(c.g()), mix(c.b()), c.a())
}
