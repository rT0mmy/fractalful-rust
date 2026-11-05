//! Blender-style node editor on raw egui, with one canvas per graph kind (Material,
//! World, Processing; `EditorState::mode`), each keeping its own view and selection.
//!
//! Navigation: scroll (or Ctrl+scroll / pinch) zooms around the pointer; drag empty
//! canvas with the left or middle button to pan. Home frames every node, F (or numpad
//! period) the selection.
//!
//! Selection: click a node's header (Shift/Ctrl+click toggles). Drag with the right
//! button to box-select (Shift adds, Alt removes); Shift/Ctrl + left drag adds too.
//! Ctrl + right drag cuts the links it crosses, like Blender's knife.
//!
//! Links: drag from an output port to an input port. Drop a new link on empty canvas
//! to search for a node that is added already connected. Dragging a connected input
//! picks its link back up.
//!
//! Adding nodes: right-click the canvas for the menu, or Shift+A for a search box.
//! Right-click a node for its actions.
//!
//! Shortcuts while the pointer is over the canvas: Ctrl+C / Ctrl+X / Ctrl+V copy, cut,
//! paste (via the system clipboard, so it works between windows), Ctrl+D duplicate,
//! Delete / X delete, Ctrl+A select all, M mute, Ctrl+L lock, Esc deselect.
//!
//! Keyframes: every value row has a diamond. Click it (or hover the row and press I)
//! to key the value at the playhead, Alt+I removes that key, right-click it for more.
//! Rows are tinted like Blender's fields: green animated, yellow keyed on this frame,
//! orange edited but not keyed.
//!
//! `fade` (0..1) fades the editor out while it isn't being used: the canvas background
//! goes transparent and nodes drop to `faded_opacity`, so the fractal stays visible.

use std::sync::Arc;

use egui::epaint::CubicBezierShape;
use egui::{
    Color32, FontId, Key, Modifiers, PointerButton, Pos2, Rect, Sense, Stroke, StrokeKind, UiBuilder, Vec2, pos2, vec2,
};

use super::{
    CURVE_CHANNELS, CURVE_POINTS, Category, Clip, Graph, GraphKind, Graphs, InputDef, KeyState, Node, NodeId, NodeKind,
    SocketType,
};
use crate::anim::Interp;

const NODE_W: f32 = 230.0;
const HEADER: f32 = 24.0;
const ROW: f32 = 22.0;
const PORT_R: f32 = 5.0;
const MIN_ZOOM: f32 = 0.15;
const MAX_ZOOM: f32 = 2.5;
/// Below this zoom node rows are drawn as plain text instead of widgets.
const WIDGET_ZOOM: f32 = 0.45;
/// Offset of duplicated nodes from their originals.
const DUPLICATE_OFFSET: f32 = 40.0;
const KEY_ANIMATED: Color32 = Color32::from_rgb(90, 170, 90);
const KEY_ON: Color32 = Color32::from_rgb(220, 190, 60);
const KEY_CHANGED: Color32 = Color32::from_rgb(235, 130, 40);
const SELECT_FILL: Color32 = Color32::from_rgba_premultiplied(5, 22, 51, 40);
const SELECT_STROKE: Color32 = crate::theme::ACCENT;
const CUT_COLOR: Color32 = Color32::from_rgb(255, 90, 90);

/// A keyframe edit asked for on one input.
#[derive(Clone, Copy)]
enum KeyEdit {
    Insert,
    Delete,
    /// The diamond: delete the key at the playhead if there is one, else insert.
    Toggle,
    Clear,
    Interp(Interp),
}

/// Draws a keyframe diamond for `state`; returns its response (click / right-click).
fn key_diamond(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    center: Pos2,
    z: f32,
    state: KeyState,
    salt: (NodeId, usize),
) -> egui::Response {
    let rect = Rect::from_center_size(center, Vec2::splat(13.0 * z));
    let resp = ui.interact(rect, ui.scope_id().with(("key", salt)), Sense::click());
    let r = if resp.hovered() { 5.0 } else { 4.0 } * z;
    let pts = vec![center - vec2(0.0, r), center + vec2(r, 0.0), center + vec2(0.0, r), center - vec2(r, 0.0)];
    let (fill, stroke) = match state {
        KeyState::None => (Color32::TRANSPARENT, Color32::from_gray(if resp.hovered() { 200 } else { 95 })),
        KeyState::Animated => (Color32::TRANSPARENT, KEY_ANIMATED),
        KeyState::OnKey => (KEY_ON, Color32::BLACK),
        KeyState::Changed => (KEY_CHANGED, Color32::BLACK),
    };
    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::new(1.2, stroke)));
    resp
}

/// Pan, zoom and selection of one graph's canvas.
struct View {
    /// Screen offset of the graph origin from the canvas' top-left corner.
    pan: Vec2,
    zoom: f32,
    selected: Vec<NodeId>,
    /// Framed once (every node in view) when first shown.
    framed: bool,
}

impl Default for View {
    fn default() -> Self {
        Self { pan: Vec2::ZERO, zoom: 1.0, selected: Vec::new(), framed: false }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum BoxMode {
    Set,
    Add,
    Remove,
}

/// The add-node search box (Shift+A, or a link dropped on empty canvas).
struct Search {
    /// Screen position of the box; new nodes go there.
    at: Pos2,
    text: String,
    /// The link being dragged when it opened: the new node gets connected to it.
    from: Option<(NodeId, usize)>,
    /// Highlighted entry (arrow keys move it, Enter adds it).
    cursor: usize,
    opened: bool,
}

#[derive(Default)]
pub struct EditorState {
    /// Which graph the editor shows.
    pub mode: GraphKind,
    views: [View; 3],
    /// (node, output) a link is being dragged from, and whether it was picked up from
    /// an input (then dropping it on empty canvas just removes it).
    wire: Option<(NodeId, usize, bool)>,
    menu_pos: Pos2,
    /// Box selection in progress: where it started (screen space).
    box_sel: Option<(Pos2, BoxMode)>,
    /// Link cut in progress: the knife's path (screen space).
    cut: Option<Vec<Pos2>>,
    search: Option<Search>,
    /// Last copy, for when the system clipboard isn't available.
    clip: Option<Clip>,
    /// A file dialog open for a node's image (it runs on its own thread so rendering
    /// goes on): the graph, the node, and where the chosen path arrives.
    picking: Option<(GraphKind, NodeId, std::sync::mpsc::Receiver<Option<std::path::PathBuf>>)>,
    /// A message for the app to show as a toast (e.g. "Copied 3 nodes").
    pub notice: Option<String>,
}

impl EditorState {
    fn view(&mut self) -> &mut View {
        &mut self.views[self.mode.index()]
    }

    /// Nodes selected in the current mode.
    pub fn selected(&self) -> &[NodeId] {
        &self.views[self.mode.index()].selected
    }

    /// Frames every node of a `kind` graph when next shown (after it was replaced).
    pub fn reframe(&mut self, kind: GraphKind) {
        self.views[kind.index()] = View::default();
    }
}

/// Graph <-> screen mapping of the canvas.
#[derive(Clone, Copy)]
struct Xf {
    origin: Pos2,
    z: f32,
}

impl Xf {
    fn to_screen(self, p: [f32; 2]) -> Pos2 {
        self.origin + vec2(p[0], p[1]) * self.z
    }

    fn to_graph(self, p: Pos2) -> [f32; 2] {
        let g = (p - self.origin) / self.z;
        [g.x, g.y]
    }
}

/// Row layout of a node: outputs first, then the file picker, then the inputs
/// (Blender layout). Rows are in units of `ROW`.
struct Layout {
    file_row: Option<f32>,
    /// First row of each input.
    input_row: Vec<f32>,
    rows: f32,
}

fn layout(kind: NodeKind) -> Layout {
    let mut row = kind.outputs().len() as f32;
    let file_row = kind.has_file().then(|| {
        row += 1.0;
        row - 1.0
    });
    let mut input_row = Vec::new();
    for d in kind.inputs() {
        input_row.push(row);
        row += d.rows();
    }
    Layout { file_row, input_row, rows: row }
}

fn node_rect(xf: Xf, kind: NodeKind, pos: [f32; 2]) -> Rect {
    let h = HEADER + ROW * layout(kind).rows + 6.0;
    Rect::from_min_size(xf.to_screen(pos), vec2(NODE_W, h) * xf.z)
}

fn out_port(r: Rect, z: f32, k: usize) -> Pos2 {
    pos2(r.right(), r.top() + (HEADER + ROW * (k as f32 + 0.5)) * z)
}

fn in_port(r: Rect, z: f32, kind: NodeKind, j: usize) -> Pos2 {
    pos2(r.left(), r.top() + (HEADER + ROW * (layout(kind).input_row[j] + 0.5)) * z)
}

fn socket_color(ty: SocketType) -> Color32 {
    match ty {
        SocketType::State => Color32::from_rgb(0, 214, 163),
        SocketType::Float => Color32::from_gray(161),
        SocketType::Vec3 => Color32::from_rgb(199, 199, 41),
    }
}

fn header_color(cat: Category) -> Color32 {
    match cat {
        Category::Input => Color32::from_rgb(130, 40, 40),
        Category::Output => Color32::from_rgb(60, 60, 60),
        Category::Driver => Color32::from_rgb(150, 40, 110),
        Category::Audio => Color32::from_rgb(190, 70, 50),
        Category::Domain => Color32::from_rgb(70, 110, 50),
        Category::Fractal => Color32::from_rgb(150, 95, 40),
        Category::Fold => Color32::from_rgb(40, 110, 100),
        Category::Transform => Color32::from_rgb(50, 85, 135),
        Category::Combine => Color32::from_rgb(100, 70, 140),
        Category::Sky => Color32::from_rgb(45, 90, 160),
        Category::Adjust => Color32::from_rgb(110, 100, 40),
        Category::Filter => Color32::from_rgb(70, 60, 140),
        Category::Effect => Color32::from_rgb(130, 50, 120),
        Category::Math | Category::Utility => Color32::from_rgb(40, 90, 110),
        Category::Texture => Color32::from_rgb(140, 75, 50),
        Category::Color => Color32::from_rgb(120, 110, 30),
    }
}

fn wire_points(a: Pos2, b: Pos2, z: f32) -> [Pos2; 4] {
    let k = ((b.x - a.x).abs() * 0.5).max(40.0 * z);
    [a, a + vec2(k, 0.0), b - vec2(k, 0.0), b]
}

fn wire(painter: &egui::Painter, a: Pos2, b: Pos2, z: f32, color: Color32) {
    let stroke = Stroke::new((2.5 * z).clamp(1.0, 4.0), color);
    painter.add(CubicBezierShape::from_points_stroke(wire_points(a, b, z), false, Color32::TRANSPARENT, stroke));
}

/// The wire from `a` to `b` as a polyline, for hit tests.
fn wire_polyline(a: Pos2, b: Pos2, z: f32) -> Vec<Pos2> {
    let [p0, p1, p2, p3] = wire_points(a, b, z);
    (0..=24)
        .map(|i| {
            let t = i as f32 / 24.0;
            let u = 1.0 - t;
            let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
            pos2(
                w[0] * p0.x + w[1] * p1.x + w[2] * p2.x + w[3] * p3.x,
                w[0] * p0.y + w[1] * p1.y + w[2] * p2.y + w[3] * p3.y,
            )
        })
        .collect()
}

fn segments_cross(a: Pos2, b: Pos2, c: Pos2, d: Pos2) -> bool {
    let cross = |o: Pos2, p: Pos2, q: Pos2| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    let (d1, d2) = (cross(c, d, a), cross(c, d, b));
    let (d3, d4) = (cross(a, b, c), cross(a, b, d));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

fn polylines_cross(p: &[Pos2], q: &[Pos2]) -> bool {
    p.windows(2).any(|s| q.windows(2).any(|t| segments_cross(s[0], s[1], t[0], t[1])))
}

/// A small padlock drawn with shapes (the bundled fonts have no lock glyph).
fn padlock(painter: &egui::Painter, center: Pos2, z: f32, color: Color32) {
    let body = Rect::from_center_size(center + vec2(0.0, 2.5) * z, vec2(10.0, 7.0) * z);
    painter.rect_filled(body, 1.5 * z, color);
    let shackle: Vec<Pos2> = (0..=12)
        .map(|i| {
            let a = std::f32::consts::PI * (1.0 + i as f32 / 12.0);
            center + vec2(3.2 * a.cos(), -1.0 + 3.6 * a.sin()) * z
        })
        .collect();
    painter.add(egui::Shape::line(shackle, Stroke::new(1.6 * z, color)));
}

/// `style` with every size scaled by `z`, for widgets inside zoomed nodes.
fn scaled_style(style: &egui::Style, z: f32) -> Arc<egui::Style> {
    let mut s = style.clone();
    for f in s.text_styles.values_mut() {
        f.size *= z;
    }
    let sp = &mut s.spacing;
    sp.item_spacing *= z;
    sp.button_padding *= z;
    sp.interact_size *= z;
    sp.icon_width *= z;
    sp.icon_width_inner *= z;
    sp.icon_spacing *= z;
    sp.combo_width *= z;
    sp.slider_width *= z;
    Arc::new(s)
}

/// Monotone cubic through the curve's points (x sorted), linear beyond the ends.
/// Mirrors `curve_eval` in post.wgsl.
pub fn curve_eval(pts: &[[f32; 2]; CURVE_POINTS], x: f32) -> f32 {
    let n = CURVE_POINTS;
    let d: Vec<f32> = (0..n - 1).map(|k| (pts[k + 1][1] - pts[k][1]) / (pts[k + 1][0] - pts[k][0]).max(1e-5)).collect();
    let m = |k: usize| -> f32 {
        if k == 0 {
            d[0]
        } else if k == n - 1 {
            d[n - 2]
        } else if d[k - 1] * d[k] <= 0.0 {
            0.0
        } else {
            2.0 * d[k - 1] * d[k] / (d[k - 1] + d[k])
        }
    };
    if x <= pts[0][0] {
        return pts[0][1] + m(0) * (x - pts[0][0]);
    }
    if x >= pts[n - 1][0] {
        return pts[n - 1][1] + m(n - 1) * (x - pts[n - 1][0]);
    }
    let k = (0..n - 1).find(|&k| x <= pts[k + 1][0]).unwrap_or(n - 2);
    let h = (pts[k + 1][0] - pts[k][0]).max(1e-5);
    let t = (x - pts[k][0]) / h;
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * pts[k][1]
        + (t3 - 2.0 * t2 + t) * h * m(k)
        + (-2.0 * t3 + 3.0 * t2) * pts[k + 1][1]
        + (t3 - t2) * h * m(k + 1)
}

fn curve_points(vals: &[f32], ch: usize) -> [[f32; 2]; CURVE_POINTS] {
    std::array::from_fn(|k| {
        let o = (ch * CURVE_POINTS + k) * 2;
        [vals[o], vals[o + 1]]
    })
}

/// The RGB curves widget: channel buttons (C R G B) and a plot with draggable points.
/// Double-click the plot to reset the shown channel. Returns true if a value changed.
fn curve_widget(ui: &mut egui::Ui, rect: Rect, vals: &mut [f32], id: egui::Id, z: f32, enabled: bool) -> bool {
    let painter = ui.painter_at(rect.expand(4.0 * z));
    let ch_id = id.with("channel");
    let mut ch: usize = ui.data(|d| d.get_temp(ch_id)).unwrap_or(0);
    let names = ["C", "R", "G", "B"];
    let colors = [Color32::from_gray(220), Color32::from_rgb(230, 80, 80), Color32::from_rgb(90, 200, 90), Color32::from_rgb(90, 140, 240)];
    let tab_h = ROW * z;
    for (k, name) in names.iter().enumerate() {
        let r = Rect::from_min_size(rect.min + vec2(k as f32 * 26.0 * z, 0.0), vec2(24.0 * z, tab_h - 3.0 * z));
        let resp = ui.interact(r, id.with(("tab", k)), Sense::click());
        let active = ch == k;
        painter.rect_filled(r, 3.0 * z, if active { Color32::from_gray(80) } else { Color32::from_gray(50) });
        painter.text(r.center(), egui::Align2::CENTER_CENTER, *name, FontId::proportional(11.0 * z), colors[k]);
        if resp.clicked() {
            ch = k;
        }
    }
    ui.data_mut(|d| d.insert_temp(ch_id, ch));

    let plot = Rect::from_min_max(pos2(rect.left(), rect.top() + tab_h), rect.max).shrink(4.0 * z);
    painter.rect_filled(plot, 2.0 * z, Color32::from_gray(28));
    for i in 1..4 {
        let f = i as f32 / 4.0;
        let s = Stroke::new(1.0, Color32::from_gray(45));
        painter.vline(plot.left() + plot.width() * f, plot.y_range(), s);
        painter.hline(plot.x_range(), plot.bottom() - plot.height() * f, s);
    }
    let to_screen = |p: [f32; 2]| pos2(plot.left() + p[0] * plot.width(), plot.bottom() - p[1] * plot.height());
    // Other channels faintly behind the active one.
    for c in (0..CURVE_CHANNELS).filter(|&c| c != ch) {
        let pts = curve_points(vals, c);
        let line: Vec<Pos2> = (0..=48).map(|i| i as f32 / 48.0).map(|x| to_screen([x, curve_eval(&pts, x)])).collect();
        painter.add(egui::Shape::line(line, Stroke::new(1.0, colors[c].gamma_multiply(0.25))));
    }
    let pts = curve_points(vals, ch);
    let line: Vec<Pos2> = (0..=64)
        .map(|i| i as f32 / 64.0)
        .map(|x| to_screen([x, curve_eval(&pts, x).clamp(-0.05, 1.05)]))
        .collect();
    painter.add(egui::Shape::line(line, Stroke::new(1.5 * z.max(0.7), colors[ch])));

    let bg = ui.interact(plot, id.with("plot"), Sense::click());
    let mut changed = false;
    if enabled && bg.double_clicked() {
        let n = (CURVE_POINTS - 1) as f32;
        for k in 0..CURVE_POINTS {
            let o = (ch * CURVE_POINTS + k) * 2;
            vals[o] = k as f32 / n;
            vals[o + 1] = k as f32 / n;
        }
        changed = true;
    }
    for k in 0..CURVE_POINTS {
        let o = (ch * CURVE_POINTS + k) * 2;
        let c = to_screen([vals[o], vals[o + 1]]);
        let resp = ui.interact(Rect::from_center_size(c, Vec2::splat(12.0 * z.max(0.7))), id.with(("pt", ch, k)), Sense::drag());
        if enabled && resp.dragged() {
            let d = resp.drag_delta();
            let lo = if k == 0 { 0.0 } else { vals[o - 2] + 0.01 };
            let hi = if k == CURVE_POINTS - 1 { 1.0 } else { vals[o + 2] - 0.01 };
            // The end points stay on the edges; inner ones move between their neighbours.
            if k > 0 && k < CURVE_POINTS - 1 {
                vals[o] = (vals[o] + d.x / plot.width()).clamp(lo, hi);
            }
            vals[o + 1] = (vals[o + 1] - d.y / plot.height()).clamp(0.0, 1.0);
            changed = true;
        }
        let r = if resp.hovered() || resp.dragged() { 4.5 } else { 3.5 } * z.max(0.7);
        painter.circle(c, r, colors[ch], Stroke::new(1.0, Color32::BLACK));
    }
    changed
}

/// What a menu item or shortcut asked for this frame; applied after drawing.
#[derive(Clone, Copy, PartialEq)]
enum Action {
    Copy,
    Cut,
    Paste,
    Duplicate,
    Delete,
    ToggleMute,
    ToggleLock,
    ResetValues,
    Disconnect,
    SelectAll,
    FrameAll,
    FrameSelected,
    Search,
}

/// Node kinds the add menu / search offers in a `mode` graph.
fn addable_kinds(g: &Graph, mode: GraphKind) -> impl Iterator<Item = NodeKind> + '_ {
    NodeKind::ALL.iter().copied().filter(move |k| k.addable() && k.allowed_in(mode) && !(k.unique() && g.find(*k).is_some()))
}

/// First input of `kind` a link of type `ty` can go into (an exact type match first).
fn input_for(kind: NodeKind, ty: SocketType) -> Option<usize> {
    let ins = kind.inputs();
    ins.iter()
        .position(|d| d.linkable() && d.ty == ty)
        .or_else(|| ins.iter().position(|d| d.linkable() && ty.compatible(d.ty)))
}

/// Fits the view to `ids` (every node if empty).
fn frame(g: &Graph, view: &mut View, canvas: Rect, ids: &[NodeId]) {
    let xf = Xf { origin: Pos2::ZERO, z: 1.0 };
    let rects = g.nodes.iter().filter(|n| ids.is_empty() || ids.contains(&n.id)).map(|n| node_rect(xf, n.kind, n.pos));
    let Some(b) = rects.reduce(|a, b| a.union(b)) else { return };
    let margin = 40.0;
    let z = ((canvas.width() - 2.0 * margin) / b.width()).min((canvas.height() - 2.0 * margin) / b.height());
    view.zoom = z.clamp(MIN_ZOOM, 1.0);
    view.pan = canvas.center() - canvas.min - b.center().to_vec2() * view.zoom;
}

/// `time` is the playhead (animation time), where keyframes are inserted.
pub fn show(ui: &mut egui::Ui, graphs: &mut Graphs, st: &mut EditorState, fade: f32, faded_opacity: f32, time: f32) {
    // A file picked for a node (possibly in another mode by now).
    if let Some((kind, id, rx)) = &st.picking {
        if let Ok(file) = rx.try_recv() {
            if let (Some(path), Some(n)) = (file, graphs.get_mut(*kind).node_mut(*id)) {
                // The file is bound at render time; no recompile needed.
                n.file = Some(path.to_string_lossy().into_owned());
            }
            st.picking = None;
        } else {
            // The pop-out editor window only redraws on demand.
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    let mode = st.mode;
    let g = graphs.get_mut(mode);
    let typing = ui.ctx().memory(|m| m.focused().is_some());
    let mut key_edit: Option<(NodeId, usize, KeyEdit)> = None;
    let (canvas_rect, canvas) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
    let backdrop = 1.0 - fade;
    let bg_painter = ui.painter_at(canvas_rect);
    bg_painter.rect_filled(canvas_rect, 0.0, Color32::BLACK.gamma_multiply(backdrop));
    // Everything drawn after this (grid, wires, nodes, widgets) fades together.
    ui.multiply_opacity(1.0 + (faded_opacity - 1.0) * fade);
    let painter = ui.painter_at(canvas_rect);
    st.view().selected.retain(|id| g.node(*id).is_some());
    if !st.views[mode.index()].framed && canvas_rect.width() > 50.0 {
        frame(g, st.view(), canvas_rect, &[]);
        st.view().framed = true;
    }
    let mods = ui.input(|i| i.modifiers);
    let additive = mods.shift || mods.command;
    let pointer = ui.ctx().pointer_latest_pos();
    let hovered = ui.rect_contains_pointer(canvas_rect);

    // Zoom around the pointer: scroll wheel, Ctrl+scroll or pinch.
    if hovered && st.search.is_none() {
        let (scroll, pinch) = ui.input_mut(|i| {
            let s = i.smooth_scroll_delta.y;
            i.smooth_scroll_delta = Vec2::ZERO;
            (s, i.zoom_delta())
        });
        let factor = (scroll * 0.0018).exp() * pinch;
        if factor != 1.0 {
            if let Some(p) = pointer {
                let v = st.view();
                let new = (v.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
                let at = (p - canvas_rect.min - v.pan) / v.zoom;
                v.pan = p - canvas_rect.min - at * new;
                v.zoom = new;
            }
        }
    }
    // Pan: drag empty canvas with the left button, or the middle button anywhere.
    let middle = ui.input(|i| i.pointer.button_down(PointerButton::Middle));
    if hovered && middle {
        st.view().pan += ui.input(|i| i.pointer.delta());
    } else if st.box_sel.is_none() && canvas.dragged_by(PointerButton::Primary) {
        st.view().pan += canvas.drag_delta();
    }
    // Shift/Ctrl + left drag adds a box; right drag boxes (Shift adds, Alt removes);
    // Ctrl + right drag cuts links.
    if canvas.drag_started_by(PointerButton::Primary) && additive {
        st.box_sel = ui.ctx().pointer_interact_pos().map(|p| (p, BoxMode::Add));
    }
    if canvas.drag_started_by(PointerButton::Secondary) {
        let p = ui.ctx().pointer_interact_pos();
        if mods.command {
            st.cut = p.map(|p| vec![p]);
        } else {
            let m = if mods.shift {
                BoxMode::Add
            } else if mods.alt {
                BoxMode::Remove
            } else {
                BoxMode::Set
            };
            st.box_sel = p.map(|p| (p, m));
        }
    }
    if canvas.clicked() && !additive {
        st.view().selected.clear();
    }
    let z = st.views[mode.index()].zoom;
    let xf = Xf { origin: canvas_rect.min + st.views[mode.index()].pan, z };

    // Background grid (coarser when zoomed out so lines stay apart).
    let mut grid = 32.0 * z;
    while grid < 12.0 {
        grid *= 4.0;
    }
    let grid_stroke = Stroke::new(1.0, Color32::from_rgb(14, 17, 22).gamma_multiply(backdrop));
    let pan = st.views[mode.index()].pan;
    let mut x = canvas_rect.left() + pan.x.rem_euclid(grid);
    while x < canvas_rect.right() {
        painter.vline(x, canvas_rect.y_range(), grid_stroke);
        x += grid;
    }
    let mut y = canvas_rect.top() + pan.y.rem_euclid(grid);
    while y < canvas_rect.bottom() {
        painter.hline(canvas_rect.x_range(), y, grid_stroke);
        y += grid;
    }

    // The action, and whether it came from a menu (then it applies at the menu's spot).
    let mut action: Option<(Action, bool)> = None;
    let has_clip = st.clip.is_some();

    // Canvas menu: add a node, search, paste, select, frame.
    if canvas.secondary_clicked() {
        st.menu_pos = ui.ctx().pointer_interact_pos().unwrap_or(canvas_rect.center());
    }
    let menu_pos = st.menu_pos;
    let mut added: Option<NodeKind> = None;
    canvas.context_menu(|ui| {
        if ui.add(egui::Button::new("Search…").shortcut_text("Shift+A")).clicked() {
            action = Some((Action::Search, true));
            ui.close();
        }
        ui.separator();
        for cat in Category::ALL {
            let kinds: Vec<NodeKind> = addable_kinds(g, mode).filter(|k| k.category() == cat).collect();
            if kinds.is_empty() {
                continue;
            }
            ui.menu_button(cat.label(), |ui| {
                for kind in kinds {
                    let mut button = ui.button(kind.label());
                    if !kind.description().is_empty() {
                        button = button.on_hover_text(kind.description());
                    }
                    if button.clicked() {
                        added = Some(kind);
                        ui.close();
                    }
                }
            });
        }
        ui.separator();
        if ui.add_enabled(has_clip, egui::Button::new("Paste").shortcut_text("Ctrl+V")).clicked() {
            action = Some((Action::Paste, true));
            ui.close();
        }
        if ui.add(egui::Button::new("Select all").shortcut_text("Ctrl+A")).clicked() {
            action = Some((Action::SelectAll, true));
            ui.close();
        }
        if ui.add(egui::Button::new("Frame all").shortcut_text("Home")).clicked() {
            action = Some((Action::FrameAll, true));
            ui.close();
        }
    });
    if let Some(kind) = added {
        st.view().selected = vec![g.add(kind, xf.to_graph(menu_pos))];
    }

    // Links (under nodes), colored by the source socket type. Remember their paths for
    // the knife.
    let mut link_paths: Vec<(usize, Vec<Pos2>)> = Vec::new();
    for (li, l) in g.links.iter().enumerate() {
        if let (Some(a), Some(b)) = (g.node(l.from), g.node(l.to)) {
            let ra = node_rect(xf, a.kind, a.pos);
            let rb = node_rect(xf, b.kind, b.pos);
            let (pa, pb) = (out_port(ra, z, l.from_out), in_port(rb, z, b.kind, l.input));
            let color = socket_color(a.kind.outputs()[l.from_out].ty);
            wire(&painter, pa, pb, z, color);
            if st.cut.is_some() {
                link_paths.push((li, wire_polyline(pa, pb, z)));
            }
        }
    }

    // Nodes inside the box being dragged show as selected already.
    let box_rect = st.box_sel.map(|(start, _)| Rect::from_two_pos(start, pointer.unwrap_or(start)));
    let wire_ty = st.wire.and_then(|(from, k, _)| g.node(from).map(|n| n.kind.outputs()[k].ty));

    let mut inputs: Vec<(NodeId, usize, Pos2)> = Vec::new();
    let mut detach: Option<(NodeId, usize)> = None;
    let mut structural_edit = false;
    let mut move_by = Vec2::ZERO;
    let mut open_file: Option<NodeId> = None;
    let linked: Vec<(NodeId, usize)> = g.links.iter().map(|l| (l.to, l.input)).collect();
    let selection = st.views[mode.index()].selected.clone();
    let any_mutable = selection.iter().any(|id| g.node(*id).is_some_and(|n| n.kind.mutable()));
    let all_muted = selection.iter().filter_map(|id| g.node(*id)).filter(|n| n.kind.mutable()).all(|n| n.muted);
    let all_locked = !selection.is_empty() && selection.iter().filter_map(|id| g.node(*id)).all(|n| n.locked);
    let widgets = z >= WIDGET_ZOOM;
    let style = scaled_style(ui.style(), z);
    let font = |size: f32| FontId::proportional(size * z);
    let label_color = Color32::from_gray(210);

    for node in &mut g.nodes {
        let (id, kind) = (node.id, node.kind);
        let rect = node_rect(xf, kind, node.pos);
        if !rect.intersects(canvas_rect) {
            continue;
        }
        let lay = layout(kind);
        let in_box = box_rect.is_some_and(|b| b.intersects(rect));
        let selected = match st.box_sel {
            Some((_, BoxMode::Remove)) if in_box => false,
            Some((_, BoxMode::Set)) => in_box,
            _ => selection.contains(&id) || in_box,
        };

        painter.rect(
            rect,
            crate::theme::RADIUS as f32 * z,
            Color32::from_rgb(12, 14, 18),
            if selected { Stroke::new(1.5, crate::theme::ACCENT) } else { Stroke::new(1.0, crate::theme::BORDER) },
            StrokeKind::Outside,
        );
        let header = Rect::from_min_size(rect.min, vec2(rect.width(), HEADER * z));
        let head_color = if node.muted { Color32::from_gray(70) } else { header_color(kind.category()) };
        let r = (crate::theme::RADIUS as f32 * z).round() as u8;
        painter.rect_filled(header, egui::CornerRadius { nw: r, ne: r, sw: 0, se: 0 }, head_color);
        let title = if node.muted { format!("{} (muted)", kind.label()) } else { kind.label().to_string() };
        painter.text(
            header.left_center() + vec2(8.0 * z, 0.0),
            egui::Align2::LEFT_CENTER,
            title,
            font(13.0),
            if node.muted { Color32::from_gray(170) } else { Color32::WHITE },
        );
        if node.locked {
            padlock(&painter, header.right_center() - vec2(14.0 * z, 0.0), z, Color32::from_gray(230));
        }

        let mut hr = ui.interact(header, ui.scope_id().with(("node", id)), Sense::click_and_drag());
        if !kind.description().is_empty() {
            hr = hr.on_hover_text(kind.description());
        }
        // Selection: Shift/Ctrl+click toggles; grabbing an unselected node selects only it.
        if hr.clicked() {
            let sel = &mut st.view().selected;
            if additive {
                match sel.iter().position(|s| *s == id) {
                    Some(i) => {
                        sel.remove(i);
                    }
                    None => sel.push(id),
                }
            } else {
                *sel = vec![id];
            }
        }
        if (hr.drag_started_by(PointerButton::Primary) || hr.secondary_clicked()) && !selected {
            st.view().selected = vec![id];
        }
        // Right drag from a header starts a box / cut, like on empty canvas.
        if hr.drag_started_by(PointerButton::Secondary) {
            let p = ui.ctx().pointer_interact_pos();
            if mods.command {
                st.cut = p.map(|p| vec![p]);
            } else {
                st.box_sel = p.map(|p| (p, if mods.shift { BoxMode::Add } else { BoxMode::Set }));
            }
        }
        if hr.secondary_clicked() {
            st.menu_pos = ui.ctx().pointer_interact_pos().unwrap_or(header.center());
        }
        if hr.dragged_by(PointerButton::Primary) {
            move_by += hr.drag_delta() / z;
        }

        let n_sel = if selected { selection.len().max(1) } else { 1 };
        hr.context_menu(|ui| {
            if n_sel > 1 {
                ui.weak(format!("{n_sel} nodes"));
                ui.separator();
            }
            let mut item = |ui: &mut egui::Ui, enabled: bool, label: &str, keys: &str, a: Action| {
                if ui.add_enabled(enabled, egui::Button::new(label).shortcut_text(keys)).clicked() {
                    action = Some((a, true));
                    ui.close();
                }
            };
            item(ui, true, "Duplicate", "Ctrl+D", Action::Duplicate);
            item(ui, true, "Copy", "Ctrl+C", Action::Copy);
            item(ui, true, "Cut", "Ctrl+X", Action::Cut);
            item(ui, has_clip, "Paste", "Ctrl+V", Action::Paste);
            ui.separator();
            let mutable = if selected { any_mutable } else { kind.mutable() };
            let muted = if selected { all_muted } else { node.muted };
            let locked = if selected { all_locked } else { node.locked };
            item(ui, mutable, if muted && mutable { "Unmute" } else { "Mute (bypass)" }, "M", Action::ToggleMute);
            item(ui, true, if locked { "Unlock" } else { "Lock" }, "Ctrl+L", Action::ToggleLock);
            item(ui, true, "Reset values", "", Action::ResetValues);
            item(ui, true, "Disconnect links", "", Action::Disconnect);
            item(ui, true, "Frame selected", "F", Action::FrameSelected);
            ui.separator();
            if kind.required() && n_sel == 1 {
                ui.weak(format!("The {} can't be deleted", kind.label()));
            } else {
                item(ui, true, "Delete", "X / Del", Action::Delete);
            }
        });

        // A muted node passes its first input straight through: show it as a wire.
        if node.muted {
            wire(&painter, in_port(rect, z, kind, 0), out_port(rect, z, 0), z, Color32::from_rgb(200, 80, 80));
        }

        // Outputs: label right-aligned, port on the right edge.
        let port_r = (PORT_R * z).max(2.5);
        let hit = (16.0 * z).max(10.0);
        for (k, o) in kind.outputs().iter().enumerate() {
            let p = out_port(rect, z, k);
            painter.text(p - vec2(12.0 * z, 0.0), egui::Align2::RIGHT_CENTER, o.name, font(12.0), label_color);
            painter.circle(p, port_r, socket_color(o.ty), Stroke::new(1.0, Color32::BLACK));
            let port = Rect::from_center_size(p, Vec2::splat(hit));
            if ui.interact(port, ui.scope_id().with(("out", id, k)), Sense::drag()).drag_started_by(PointerButton::Primary) {
                st.wire = Some((id, k, false));
            }
        }

        // File picker row.
        if let Some(row) = lay.file_row {
            let y = rect.top() + (HEADER + ROW * (row + 0.5)) * z;
            let r = Rect::from_min_size(pos2(rect.left() + 12.0 * z, y - ROW * 0.5 * z + z), vec2(rect.width() - 24.0 * z, (ROW - 2.0) * z));
            let name = node.file.as_deref().map_or("no file".to_string(), |f| {
                std::path::Path::new(f).file_name().map_or(f.to_string(), |n| n.to_string_lossy().into_owned())
            });
            if widgets {
                let locked = node.locked;
                ui.scope_builder(UiBuilder::new().max_rect(r), |ui| {
                    ui.set_style(style.clone());
                    ui.horizontal_centered(|ui| {
                        if ui.add_enabled(!locked, egui::Button::new("Open…")).on_hover_text(node.file.clone().unwrap_or_default()).clicked() {
                            open_file = Some(id);
                        }
                        ui.add(egui::Label::new(egui::RichText::new(name).size(12.0 * z)).truncate());
                    });
                });
            } else {
                painter.text(r.left_center(), egui::Align2::LEFT_CENTER, name, font(12.0), label_color);
            }
        }

        // Inputs: port on the left edge, then a label (linked / state) or the inline value.
        for (j, def) in kind.inputs().iter().enumerate() {
            let p = in_port(rect, z, kind, j);
            let is_linked = linked.contains(&(id, j));
            if def.linkable() {
                let compatible = wire_ty.is_some_and(|ty| ty.compatible(def.ty));
                if compatible {
                    painter.circle_stroke(p, port_r + 3.0 * z, Stroke::new(1.5, Color32::from_white_alpha(140)));
                }
                painter.circle(p, port_r, socket_color(def.ty), Stroke::new(1.0, Color32::BLACK));
                let port = Rect::from_center_size(p, Vec2::splat(hit));
                if ui.interact(port, ui.scope_id().with(("in", id, j)), Sense::drag()).drag_started_by(PointerButton::Primary) {
                    detach = Some((id, j));
                }
                inputs.push((id, j, p));
            }
            if is_linked || def.label_only() {
                painter.text(p + vec2(12.0 * z, 0.0), egui::Align2::LEFT_CENTER, def.name, font(12.0), label_color);
                continue;
            }
            let o = node.value_offset(j);
            if def.curve {
                let top = rect.top() + (HEADER + ROW * lay.input_row[j]) * z;
                let r = Rect::from_min_max(pos2(rect.left() + 12.0 * z, top + 2.0 * z), pos2(rect.right() - 12.0 * z, top + ROW * def.rows() * z));
                let locked = node.locked;
                curve_widget(ui, r, &mut node.values[o..o + def.width()], ui.scope_id().with(("curve", id, j)), z, widgets && !locked);
                continue;
            }
            // Keyframe diamond (and the row tint) for numeric inputs.
            let keyable = node.keyable(j);
            let indent = if keyable { 24.0 } else { 12.0 } * z;
            let row_rect = Rect::from_min_size(
                pos2(rect.left() + indent, p.y - ROW * 0.5 * z + z),
                vec2(rect.width() - indent - 12.0 * z, (ROW - 2.0) * z),
            );
            if keyable {
                let state = node.key_state(j, time);
                let tint = match state {
                    KeyState::None => None,
                    KeyState::Animated => Some(KEY_ANIMATED),
                    KeyState::OnKey => Some(KEY_ON),
                    KeyState::Changed => Some(KEY_CHANGED),
                };
                if let Some(c) = tint {
                    painter.rect_filled(row_rect.expand2(vec2(2.0 * z, 0.0)), 3.0 * z, c.gamma_multiply(0.22));
                }
                let d = key_diamond(ui, &painter, pos2(rect.left() + 16.0 * z, p.y), z, state, (id, j));
                let d = d.on_hover_text("Keyframe at the playhead (I over the row; Alt+I removes). Right-click: more");
                if d.clicked() && !node.locked {
                    key_edit = Some((id, j, KeyEdit::Toggle));
                }
                let has_track = node.track(j).is_some();
                let on_key = node.track(j).is_some_and(|tr| tr.key_at(time).is_some());
                d.context_menu(|ui| {
                    let mut item = |ui: &mut egui::Ui, enabled: bool, label: &str, keys: &str, e: KeyEdit| {
                        if ui.add_enabled(enabled, egui::Button::new(label).shortcut_text(keys)).clicked() {
                            key_edit = Some((id, j, e));
                            ui.close();
                        }
                    };
                    item(ui, true, "Insert keyframe", "I", KeyEdit::Insert);
                    item(ui, on_key, "Delete keyframe", "Alt+I", KeyEdit::Delete);
                    item(ui, has_track, "Clear all keyframes", "", KeyEdit::Clear);
                    ui.add_enabled_ui(on_key, |ui| {
                        ui.menu_button("Interpolation", |ui| {
                            for i in Interp::ALL {
                                if ui.button(i.label()).clicked() {
                                    key_edit = Some((id, j, KeyEdit::Interp(i)));
                                    ui.close();
                                }
                            }
                        });
                    });
                });
                // I / Alt+I while hovering the row (consumed, so it doesn't key the camera).
                let row_hit = Rect::from_min_max(pos2(rect.left() + 8.0 * z, row_rect.top()), pos2(rect.right(), row_rect.bottom()));
                if !typing && !node.locked && ui.rect_contains_pointer(row_hit) {
                    ui.input_mut(|i| {
                        if i.consume_key(Modifiers::ALT, Key::I) {
                            key_edit = Some((id, j, KeyEdit::Delete));
                        } else if i.consume_key(Modifiers::NONE, Key::I) {
                            key_edit = Some((id, j, KeyEdit::Insert));
                        }
                    });
                }
            }
            let vals = &mut node.values[o..o + def.width()];
            if !widgets {
                row_text(&painter, row_rect, def, vals, z);
                continue;
            }
            let locked = node.locked;
            ui.scope_builder(UiBuilder::new().max_rect(row_rect), |ui| {
                ui.set_style(style.clone());
                ui.add_enabled_ui(!locked, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0 * z;
                        ui.label(egui::RichText::new(def.name).size(12.0 * z));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if !def.choices.is_empty() {
                                let before = vals[0];
                                let mut idx = (vals[0].max(0.0) as usize).min(def.choices.len() - 1);
                                egui::ComboBox::from_id_salt(("choice", id, j))
                                    .width(110.0 * z)
                                    .selected_text(def.choices[idx])
                                    .show_ui(ui, |ui| {
                                        for (k, c) in def.choices.iter().enumerate() {
                                            ui.selectable_value(&mut idx, k, *c);
                                        }
                                    });
                                vals[0] = idx as f32;
                                // Enum choices are baked into the shader (live nodes' are read on the CPU).
                                structural_edit |= vals[0] != before && !kind.is_live();
                            } else if def.color {
                                let mut c = [vals[0], vals[1], vals[2]];
                                ui.color_edit_button_rgb(&mut c);
                                vals.copy_from_slice(&c);
                            } else {
                                // Right-to-left layout: add z, y, x so they read x, y, z.
                                for v in vals.iter_mut().rev() {
                                    let mut dv = egui::DragValue::new(v).speed(def.speed).range(def.min..=def.max);
                                    if def.integer {
                                        dv = dv.fixed_decimals(0);
                                    } else if def.ty == SocketType::Vec3 {
                                        dv = dv.max_decimals(2);
                                    }
                                    ui.add(dv);
                                }
                            }
                        });
                    });
                });
            });
        }
    }
    if structural_edit {
        g.structure_rev += 1;
    }
    // The whole selection moves together; locked nodes stay put.
    if move_by != Vec2::ZERO {
        let sel = st.views[mode.index()].selected.clone();
        for n in g.nodes.iter_mut().filter(|n| sel.contains(&n.id) && !n.locked) {
            n.pos[0] += move_by.x;
            n.pos[1] += move_by.y;
        }
    }
    if let Some(id) = open_file.filter(|_| st.picking.is_none()) {
        if let Some(rx) = pick_file(g, id) {
            st.picking = Some((mode, id, rx));
        }
    }

    // Grabbing a connected input picks its wire back up (Blender-style re-routing).
    if let Some((to, j)) = detach {
        if let Some(src) = g.disconnect(to, j) {
            st.wire = Some((src.0, src.1, true));
        }
    }

    if let Some((from, k, picked_up)) = st.wire {
        let src = g.node(from).map(|n| (out_port(node_rect(xf, n.kind, n.pos), z, k), n.kind.outputs()[k].ty));
        if let (Some((a, ty)), Some(b)) = (src, pointer) {
            wire(&painter, a, b, z, socket_color(ty).gamma_multiply(1.4));
        }
        if ui.input(|i| i.pointer.any_released()) {
            if let Some(b) = pointer {
                let reach = (14.0 * z).max(9.0);
                let hit = inputs
                    .iter()
                    .filter(|(_, _, p)| p.distance(b) < reach)
                    .min_by(|x, y| x.2.distance(b).total_cmp(&y.2.distance(b)));
                let over_node = g.nodes.iter().any(|n| node_rect(xf, n.kind, n.pos).contains(b));
                if let Some(&(to, j, _)) = hit {
                    // Incompatible types and cycles are refused by `connect`.
                    g.connect(from, k, to, j);
                } else if !picked_up && !over_node && canvas_rect.contains(b) {
                    // A new link dropped on empty canvas: find a node to connect it to.
                    st.search = Some(Search { at: b, text: String::new(), from: Some((from, k)), cursor: 0, opened: false });
                }
            }
            st.wire = None;
        }
    }

    // Box selection: applied on release.
    if let Some((start, m)) = st.box_sel {
        let sel = Rect::from_two_pos(start, pointer.unwrap_or(start));
        painter.rect(sel, 0.0, SELECT_FILL, Stroke::new(1.0, SELECT_STROKE), StrokeKind::Inside);
        if !ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down()) {
            let inside: Vec<NodeId> = g.nodes.iter().filter(|n| sel.intersects(node_rect(xf, n.kind, n.pos))).map(|n| n.id).collect();
            let v = st.view();
            match m {
                BoxMode::Set => v.selected = inside,
                BoxMode::Add => v.selected.extend(inside.into_iter().filter(|id| !v.selected.contains(id)).collect::<Vec<_>>()),
                BoxMode::Remove => v.selected.retain(|id| !inside.contains(id)),
            }
            st.box_sel = None;
        }
    }

    // Link cutting: the knife's path, applied on release.
    if let Some(path) = st.cut.as_mut() {
        if let Some(p) = pointer.filter(|p| path.last().is_none_or(|l| l.distance(*p) > 3.0)) {
            path.push(p);
        }
        let dash = egui::Shape::dashed_line(path, Stroke::new(1.5, CUT_COLOR), 6.0, 4.0);
        painter.extend(dash);
        if !ui.input(|i| i.pointer.secondary_down()) {
            let cut: Vec<usize> = link_paths.iter().filter(|(_, w)| polylines_cross(path, w)).map(|(i, _)| *i).collect();
            if !cut.is_empty() {
                let mut i = 0;
                g.links.retain(|_| {
                    i += 1;
                    !cut.contains(&(i - 1))
                });
                g.structure_rev += 1;
                st.notice = Some(format!("Cut {} link{}", cut.len(), if cut.len() == 1 { "" } else { "s" }));
            }
            st.cut = None;
        }
    }

    if let Some((id, j, e)) = key_edit {
        if let Some(n) = g.node_mut(id) {
            match e {
                KeyEdit::Insert => n.insert_key(j, time),
                KeyEdit::Delete => {
                    n.delete_key(j, time);
                }
                KeyEdit::Toggle => {
                    // Edited away from the curve: clicking keys the new value.
                    if n.key_state(j, time) == KeyState::OnKey {
                        n.delete_key(j, time);
                    } else {
                        n.insert_key(j, time);
                    }
                }
                KeyEdit::Clear => {
                    n.clear_track(j);
                }
                KeyEdit::Interp(i) => {
                    if let Some(tr) = n.anim.iter_mut().find(|tr| tr.input == j) {
                        if let Some(k) = tr.key_at(time) {
                            tr.keys[k].interp = i;
                        }
                    }
                }
            }
        }
    }

    // Shortcuts apply only while the pointer is over this canvas (not when typing),
    // so Delete or Ctrl+C elsewhere in the app never touch the graph.
    if hovered && !typing && st.search.is_none() {
        let mut pasted: Option<String> = None;
        let mut deselect = false;
        ui.input_mut(|i| {
            for e in &i.events {
                match e {
                    egui::Event::Copy => action = Some((Action::Copy, false)),
                    egui::Event::Cut => action = Some((Action::Cut, false)),
                    egui::Event::Paste(text) => pasted = Some(text.clone()),
                    _ => {}
                }
            }
            let shortcuts = [
                (Modifiers::COMMAND, Key::D, Action::Duplicate),
                (Modifiers::COMMAND, Key::A, Action::SelectAll),
                (Modifiers::COMMAND, Key::L, Action::ToggleLock),
                (Modifiers::SHIFT, Key::A, Action::Search),
                (Modifiers::NONE, Key::Delete, Action::Delete),
                (Modifiers::NONE, Key::X, Action::Delete),
                (Modifiers::NONE, Key::M, Action::ToggleMute),
                (Modifiers::NONE, Key::Home, Action::FrameAll),
                (Modifiers::NONE, Key::F, Action::FrameSelected),
                (Modifiers::NONE, Key::Period, Action::FrameSelected),
            ];
            for (m, k, a) in shortcuts {
                if i.consume_key(m, k) {
                    action = Some((a, false));
                }
            }
            deselect = i.consume_key(Modifiers::NONE, Key::Escape);
        });
        if deselect {
            st.view().selected.clear();
        }
        // Clipboard text from another window or app instance takes priority over the
        // last internal copy; any other text is ignored.
        if let Some(text) = pasted {
            if let Some(clip) = Clip::from_text(&text) {
                st.clip = Some(clip);
                action = Some((Action::Paste, false));
            } else if st.clip.is_some() {
                action = Some((Action::Paste, false));
            }
        }
    }

    if let Some((a, from_menu)) = action {
        let at = if from_menu { st.menu_pos } else { pointer.filter(|p| canvas_rect.contains(*p)).unwrap_or(canvas_rect.center()) };
        match a {
            Action::FrameAll => frame(g, st.view(), canvas_rect, &[]),
            Action::FrameSelected => {
                let sel = st.views[mode.index()].selected.clone();
                frame(g, st.view(), canvas_rect, &sel);
            }
            Action::Search => {
                st.search = Some(Search { at, text: String::new(), from: None, cursor: 0, opened: false });
            }
            _ => apply(ui, g, st, a, xf.to_graph(at)),
        }
    }

    search_box(ui, g, st, xf);
}

/// Opens a file dialog for node `id`'s image on its own thread (rendering goes on);
/// the chosen path (or `None` if cancelled) arrives on the returned channel.
fn pick_file(g: &Graph, id: NodeId) -> Option<std::sync::mpsc::Receiver<Option<std::path::PathBuf>>> {
    let n = g.node(id)?;
    let (title, label, exts): (&str, &str, &[&str]) = match n.kind {
        NodeKind::EnvTexture => ("Open an HDR panorama", "HDR panoramas", &["hdr", "exr", "png", "jpg", "jpeg"]),
        _ => ("Open an image", "Images", &["png", "jpg", "jpeg", "hdr", "exr"]),
    };
    let dir = n.file.as_deref().and_then(|f| std::path::Path::new(f).parent()).map(|d| d.to_path_buf());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().set_title(title).add_filter(label, exts);
        if let Some(dir) = dir {
            dialog = dialog.set_directory(dir);
        }
        let _ = tx.send(dialog.pick_file());
    });
    Some(rx)
}

/// A value row as plain text (zoomed far out).
fn row_text(painter: &egui::Painter, r: Rect, def: &InputDef, vals: &[f32], z: f32) {
    let font = FontId::proportional(12.0 * z);
    let c = Color32::from_gray(190);
    painter.text(r.left_center(), egui::Align2::LEFT_CENTER, def.name, font.clone(), c);
    if def.color {
        let sw = Rect::from_min_size(pos2(r.right() - 30.0 * z, r.top() + 2.0 * z), vec2(30.0, r.height() / z - 4.0) * z);
        let rgb = egui::Rgba::from_rgb(vals[0], vals[1], vals[2]);
        painter.rect_filled(sw, 2.0 * z, Color32::from(rgb));
        return;
    }
    let text = if !def.choices.is_empty() {
        def.choices[(vals[0].max(0.0) as usize).min(def.choices.len() - 1)].to_string()
    } else {
        vals.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join("  ")
    };
    painter.text(r.right_center(), egui::Align2::RIGHT_CENTER, text, font, c);
}

/// The Shift+A search box: type to filter, arrows + Enter or click to add.
fn search_box(ui: &mut egui::Ui, g: &mut Graph, st: &mut EditorState, xf: Xf) {
    let mode = st.mode;
    let Some(s) = st.search.as_mut() else { return };
    let from_ty = s.from.and_then(|(n, k)| g.node(n).map(|n| n.kind.outputs()[k].ty));
    let needle = s.text.to_lowercase();
    let mut matches: Vec<NodeKind> = addable_kinds(g, mode)
        .filter(|k| from_ty.is_none_or(|ty| input_for(*k, ty).is_some()))
        .filter(|k| needle.is_empty() || k.label().to_lowercase().contains(&needle) || k.category().label().to_lowercase().contains(&needle))
        .collect();
    // Names starting with the text first.
    matches.sort_by_key(|k| !k.label().to_lowercase().starts_with(&needle));
    s.cursor = s.cursor.min(matches.len().saturating_sub(1));

    let mut chosen: Option<NodeKind> = None;
    let mut close = false;
    let area = egui::Area::new(egui::Id::unique("node search")).fixed_pos(s.at).order(egui::Order::Foreground).show(ui.ctx(), |ui| {
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_width(220.0);
            let edit = ui.add(egui::TextEdit::singleline(&mut s.text).hint_text("Search nodes…"));
            if !s.opened {
                edit.request_focus();
                s.opened = true;
            }
            ui.input_mut(|i| {
                if i.consume_key(Modifiers::NONE, Key::ArrowDown) {
                    s.cursor = (s.cursor + 1).min(matches.len().saturating_sub(1));
                }
                if i.consume_key(Modifiers::NONE, Key::ArrowUp) {
                    s.cursor = s.cursor.saturating_sub(1);
                }
                if i.consume_key(Modifiers::NONE, Key::Escape) {
                    close = true;
                }
            });
            if edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                chosen = matches.get(s.cursor).copied();
                close = chosen.is_none();
            }
            ui.separator();
            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                for (i, kind) in matches.iter().enumerate() {
                    let text = format!("{}  ·  {}", kind.label(), kind.category().label());
                    let mut resp = ui.selectable_label(i == s.cursor, text);
                    if !kind.description().is_empty() {
                        resp = resp.on_hover_text(kind.description());
                    }
                    if i == s.cursor {
                        resp.scroll_to_me(None);
                    }
                    if resp.clicked() {
                        chosen = Some(*kind);
                    }
                }
                if matches.is_empty() {
                    ui.weak("No matching nodes");
                }
            });
        });
    });
    // A click outside closes it.
    let outside = ui.input(|i| i.pointer.any_pressed())
        && ui.ctx().pointer_interact_pos().is_some_and(|p| !area.response.rect.contains(p));
    if let Some(kind) = chosen {
        let at = xf.to_graph(s.at);
        let id = g.add(kind, at);
        if let Some((from, k)) = s.from {
            if let Some(j) = from_ty.and_then(|ty| input_for(kind, ty)) {
                g.connect(from, k, id, j);
            }
        }
        st.view().selected = vec![id];
        st.search = None;
    } else if close || outside {
        st.search = None;
    }
}

/// Applies `a` to the selection. `at` is where pasted nodes go (graph coordinates).
fn apply(ui: &egui::Ui, g: &mut Graph, st: &mut EditorState, a: Action, at: [f32; 2]) {
    let mode = st.mode;
    let sel = st.selected().to_vec();
    let plural = |n: usize| if n == 1 { "node".to_string() } else { format!("{n} nodes") };
    match a {
        Action::Copy | Action::Cut => {
            let clip = g.copy(&sel);
            if clip.nodes.is_empty() {
                st.notice = Some("Nothing to copy (output nodes can't be copied)".into());
                return;
            }
            ui.ctx().copy_text(clip.to_text());
            st.notice = Some(format!("{} {}", if a == Action::Cut { "Cut" } else { "Copied" }, plural(clip.nodes.len())));
            st.clip = Some(clip);
            if a == Action::Cut {
                delete(g, st, &sel);
            }
        }
        Action::Paste => {
            if let Some(clip) = st.clip.clone() {
                let new = g.paste(&clip, at, mode);
                let skipped = clip.nodes.len() - new.len();
                st.view().selected = new;
                if skipped > 0 {
                    st.notice = Some(format!("Skipped {} that can't go in the {} graph", plural(skipped), mode.label()));
                }
            }
        }
        Action::Duplicate => {
            let clip = g.copy(&sel);
            let Some(min) = clip.nodes.iter().map(|n| n.pos).reduce(|m, p| [m[0].min(p[0]), m[1].min(p[1])]) else {
                return;
            };
            // Unique nodes (e.g. Motion Blur) can't be duplicated: leave them out.
            let clip = Clip { nodes: clip.nodes.into_iter().filter(|n| !n.kind.unique()).collect(), links: clip.links };
            let new = g.paste(&clip, [min[0] + DUPLICATE_OFFSET, min[1] + DUPLICATE_OFFSET], mode);
            // Like Blender, copies keep their links from nodes that weren't duplicated.
            let pairs: Vec<(NodeId, NodeId)> = clip.nodes.iter().map(|n| n.id).zip(new.iter().copied()).collect();
            let incoming: Vec<_> = g.links.iter().filter(|l| pairs.iter().any(|p| p.0 == l.to) && !sel.contains(&l.from)).copied().collect();
            for l in incoming {
                let to = pairs.iter().find(|p| p.0 == l.to).unwrap().1;
                g.connect(l.from, l.from_out, to, l.input);
            }
            st.view().selected = new;
        }
        Action::Delete => delete(g, st, &sel),
        Action::ToggleMute => {
            let targets: Vec<NodeId> = sel.iter().copied().filter(|id| g.node(*id).is_some_and(|n| n.kind.mutable())).collect();
            let mute = !targets.iter().all(|id| g.node(*id).unwrap().muted);
            for id in targets {
                g.node_mut(id).unwrap().muted = mute;
            }
            g.structure_rev += 1;
        }
        Action::ToggleLock => {
            let lock = !sel.iter().all(|id| g.node(*id).is_some_and(|n| n.locked));
            for id in &sel {
                if let Some(n) = g.node_mut(*id) {
                    n.locked = lock;
                }
            }
        }
        Action::ResetValues => {
            let mut structural = false;
            for id in &sel {
                if let Some(n) = g.node_mut(*id).filter(|n| !n.locked) {
                    // Enum choices are baked into the shader.
                    structural |= n.kind.inputs().iter().enumerate().any(|(j, d)| !d.choices.is_empty() && n.value(j) != d.default[0]);
                    n.values = Node::default_values(n.kind);
                }
            }
            if structural {
                g.structure_rev += 1;
            }
        }
        Action::Disconnect => {
            let before = g.links.len();
            g.links.retain(|l| !sel.contains(&l.from) && !sel.contains(&l.to));
            if g.links.len() != before {
                g.structure_rev += 1;
            }
        }
        Action::SelectAll => st.view().selected = g.nodes.iter().map(|n| n.id).collect(),
        Action::FrameAll | Action::FrameSelected | Action::Search => {}
    }
}

/// Deletes `ids`, except locked nodes and the graph's required output.
fn delete(g: &mut Graph, st: &mut EditorState, ids: &[NodeId]) {
    let mut kept_locked = 0;
    for &id in ids {
        match g.node(id) {
            Some(n) if n.locked => kept_locked += 1,
            Some(n) if !n.kind.required() => g.remove(id),
            _ => {}
        }
    }
    st.view().selected.retain(|id| g.node(*id).is_some());
    if kept_locked > 0 {
        st.notice = Some(format!("{kept_locked} locked node(s) not deleted (unlock with Ctrl+L)"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_curve_is_identity() {
        let def = &NodeKind::Curves.inputs()[2];
        assert!(def.curve);
        let vals = def.default_values();
        let pts = curve_points(&vals, 0);
        for x in [0.0, 0.1, 0.37, 0.5, 0.99, 1.0, 1.5] {
            assert!((curve_eval(&pts, x) - x).abs() < 1e-5, "{x}");
        }
    }

    /// Runs the editor headlessly in a 1000x700 egui context, one pass per call.
    struct Harness {
        ctx: egui::Context,
        gs: Graphs,
        st: EditorState,
        t: f64,
        mods: Modifiers,
    }

    impl Harness {
        fn new(mode: GraphKind) -> Self {
            let mut h = Self {
                ctx: egui::Context::default(),
                gs: Graphs::default(),
                st: EditorState { mode, ..Default::default() },
                t: 0.0,
                mods: Modifiers::NONE,
            };
            // The first passes lay out and frame the graph.
            h.pass(vec![]);
            h.pass(vec![]);
            h
        }

        fn pass(&mut self, mut events: Vec<egui::Event>) {
            self.t += 1.0 / 60.0;
            events.insert(0, egui::Event::ModifiersChanged(self.mods));
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 700.0))),
                time: Some(self.t),
                predicted_dt: 1.0 / 60.0,
                events,
                ..Default::default()
            };
            let (gs, st) = (&mut self.gs, &mut self.st);
            let mut out = self.ctx.run_pass(input, |ctx| ctx.root_ui(|ui| show(ui, gs, st, 0.0, 1.0, 0.0)));
            out.textures_delta.clear();
        }

        fn drag(&mut self, button: PointerButton, from: Pos2, to: Pos2) {
            let modifiers = self.mods;
            self.pass(vec![egui::Event::PointerMoved(from)]);
            self.pass(vec![egui::Event::PointerButton { pos: from, button, pressed: true, modifiers }]);
            for i in 1..=10 {
                self.pass(vec![egui::Event::PointerMoved(from + (to - from) * (i as f32 / 10.0))]);
            }
            self.pass(vec![egui::Event::PointerButton { pos: to, button, pressed: false, modifiers }]);
            self.pass(vec![]);
        }

        fn view(&self) -> &View {
            &self.st.views[self.st.mode.index()]
        }
    }

    #[test]
    fn scroll_zooms_around_the_pointer() {
        let mut h = Harness::new(GraphKind::Material);
        let (z0, pan0) = (h.view().zoom, h.view().pan);
        let p = pos2(500.0, 350.0);
        let under = |pan: Vec2, z: f32| (p.to_vec2() - pan) / z;
        h.pass(vec![egui::Event::PointerMoved(p)]);
        h.pass(vec![egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: vec2(0.0, 4.0),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
            modifiers: Modifiers::NONE,
        }]);
        for _ in 0..40 {
            h.pass(vec![]);
        }
        let v = h.view();
        assert!(v.zoom > z0 * 1.1, "zoom {z0} -> {}", v.zoom);
        let drift = (under(v.pan, v.zoom) - under(pan0, z0)).length();
        assert!(drift < 0.5, "the graph point under the pointer stays put (moved {drift})");
    }

    #[test]
    fn right_drag_box_selects_without_a_menu() {
        let mut h = Harness::new(GraphKind::Material);
        h.drag(PointerButton::Secondary, pos2(3.0, 3.0), pos2(997.0, 697.0));
        assert_eq!(h.view().selected.len(), h.gs.material.nodes.len(), "everything is inside");
        assert!(!egui::Popup::is_any_open(&h.ctx), "a right drag isn't a right click");
        // A plain box replaces the selection; Alt removes from it.
        h.drag(PointerButton::Secondary, pos2(3.0, 3.0), pos2(12.0, 12.0));
        assert!(h.view().selected.is_empty());
        h.st.views[0].selected = h.gs.material.nodes.iter().map(|n| n.id).collect();
        h.mods = Modifiers::ALT;
        h.drag(PointerButton::Secondary, pos2(3.0, 3.0), pos2(997.0, 697.0));
        assert!(h.view().selected.is_empty());
    }

    #[test]
    fn ctrl_right_drag_cuts_links() {
        let mut h = Harness::new(GraphKind::Post);
        let before = h.gs.post.links.len();
        h.mods = Modifiers::COMMAND;
        // Down the middle of the framed chain (Render Layers -> Bloom -> Motion Blur -> output).
        h.drag(PointerButton::Secondary, pos2(500.0, 3.0), pos2(500.0, 697.0));
        assert!(h.gs.post.links.len() < before, "{before} links before, {} after", h.gs.post.links.len());
        assert!(h.view().selected.is_empty(), "cutting doesn't select");
    }

    #[test]
    fn shift_a_searches_and_adds() {
        let mut h = Harness::new(GraphKind::World);
        let p = pos2(500.0, 600.0);
        h.pass(vec![egui::Event::PointerMoved(p)]);
        let key = |key: Key, modifiers: Modifiers| egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers };
        h.mods = Modifiers::SHIFT;
        h.pass(vec![key(Key::A, Modifiers::SHIFT)]);
        h.mods = Modifiers::NONE;
        assert!(h.st.search.is_some(), "Shift+A opens the search");
        h.pass(vec![]);
        h.pass(vec![egui::Event::Text("star".into())]);
        h.pass(vec![key(Key::Enter, Modifiers::NONE)]);
        h.pass(vec![]);
        assert!(h.st.search.is_none());
        let star = h.gs.world.find(NodeKind::StarField).expect("Star Field added");
        assert_eq!(h.view().selected, vec![star.id]);
        // Material-only nodes aren't offered in the world graph.
        assert!(addable_kinds(&h.gs.world, GraphKind::World).all(|k| k.allowed_in(GraphKind::World)));
        assert!(!addable_kinds(&h.gs.world, GraphKind::World).any(|k| k == NodeKind::Mandelbulb || k == NodeKind::Bloom));
    }

    #[test]
    fn crossing_segments() {
        assert!(segments_cross(pos2(0.0, 0.0), pos2(2.0, 2.0), pos2(0.0, 2.0), pos2(2.0, 0.0)));
        assert!(!segments_cross(pos2(0.0, 0.0), pos2(1.0, 0.0), pos2(0.0, 1.0), pos2(1.0, 1.0)));
    }
}
