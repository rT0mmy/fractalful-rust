//! Compiles node graphs to WGSL.
//!
//! The material graph becomes three functions that call into nodes.wgsl:
//! - `graph_domain(p) -> vec3<f32>`: the point to iterate (from Domain Output);
//! - `graph_step(s0, c, i) -> State`: one fractal iteration (from Iteration Output);
//! - `graph_material(o: Orbit) -> Material`: the surface material (from Material Output).
//!
//! The world graph (`generate_world`) becomes `graph_world(dir)`, the sky radiance in a
//! direction, and `graph_world_settings()`, the World Output's sun / fog / ambient
//! values (evaluated once per update on the GPU, see world.rs).
//!
//! The processing graph (`generate_post`) becomes a list of fullscreen passes (see
//! post.rs). Per-pixel nodes are fused into one pass; a node that samples its input
//! around the pixel (`ImageInput::Sampled`) or filters it in passes of its own
//! (`ImageInput::Pass`) needs that input rendered to a texture first, which ends a pass.
//!
//! Inline input values are read from a uniform array (`gp` / `wp` / `pp`) rather than
//! baked in, so only structural edits need a recompile. Enum inputs (e.g. the Math
//! operation) are baked as constants. `Alternate` / `Iteration Switch` compile to a
//! real if/else, so only the active branch executes.

use std::collections::HashMap;
use std::fmt::Write;

use super::{Context, Graph, ImageInput, LIVE_SLOTS, NodeId, NodeKind, SocketType};

/// Capacity of each graph's parameter uniform array, in vec4s.
pub const MAX_PARAM_VEC4S: usize = 256;
/// Index in the parameter array (as floats) of the first live slot; inline values come
/// before it.
pub const LIVE_OFFSET: usize = MAX_PARAM_VEC4S * 4 - LIVE_SLOTS;
/// Image files a world graph can read (Environment Texture nodes).
pub const WORLD_IMAGES: usize = 2;
/// Image files a processing graph can read (Image Texture nodes).
pub const POST_IMAGES: usize = 4;
/// Filter passes (blur, bloom) a processing graph can have.
pub const MAX_PASSES: usize = 16;

/// Inline value offset of each node, in `Graph::packed_values` order.
fn value_bases(g: &Graph) -> Result<HashMap<NodeId, usize>, String> {
    let total = g.packed_values().len();
    if total > LIVE_OFFSET {
        return Err(format!("too many inline values ({total} > {LIVE_OFFSET})"));
    }
    let mut base = HashMap::new();
    let mut off = 0;
    for n in &g.nodes {
        base.insert(n.id, off);
        off += n.values.len();
    }
    Ok(base)
}

/// The file of each image slot: nodes reading files, in node order. Files can change
/// without a recompile (the slots stay), so the renderer binds them from this every frame.
pub fn image_files(g: &Graph) -> Vec<Option<String>> {
    g.nodes.iter().filter(|n| n.kind.has_file()).map(|n| n.file.clone()).collect()
}

/// Image slot of each node reading a file (in node order, like `image_files`).
fn image_slots(g: &Graph, max: usize) -> Result<HashMap<NodeId, usize>, String> {
    let slots: HashMap<NodeId, usize> =
        g.nodes.iter().filter(|n| n.kind.has_file()).enumerate().map(|(i, n)| (n.id, i)).collect();
    if slots.len() > max {
        return Err(format!("too many image nodes ({} > {max})", slots.len()));
    }
    Ok(slots)
}

pub fn generate(g: &Graph) -> Result<String, String> {
    let base = value_bases(g)?;
    let live: HashMap<NodeId, usize> = g.live_slots()?.into_iter().collect();
    let images = HashMap::new();
    let out = g.find(NodeKind::IterOutput).ok_or("graph has no Iteration Output node")?;
    let mut src = String::new();

    let mut cg = Codegen::new(g, &base, &live, &images, Context::Domain, "gp");
    let result = match g.find(NodeKind::DomainOutput) {
        Some(d) => cg.input(d.id, 0, &mut HashMap::new(), 1)?,
        None => "p".into(),
    };
    writeln!(src, "fn graph_domain(p: vec3<f32>) -> vec3<f32> {{\n{}    return {result};\n}}\n", cg.body).unwrap();

    let mut cg = Codegen::new(g, &base, &live, &images, Context::Iteration, "gp");
    let result = cg.input(out.id, 0, &mut HashMap::new(), 1)?;
    writeln!(src, "fn graph_step(s0: State, c: vec3<f32>, i: i32) -> State {{\n{}    return {result};\n}}\n", cg.body)
        .unwrap();

    let mut cg = Codegen::new(g, &base, &live, &images, Context::Color, "gp");
    let result = match g.find(NodeKind::ColorOutput) {
        Some(co) => {
            let mut cache = HashMap::new();
            let args = (0..NodeKind::ColorOutput.inputs().len())
                .map(|j| cg.input(co.id, j, &mut cache, 1))
                .collect::<Result<Vec<_>, _>>()?;
            format!("make_material({})", args.join(", "))
        }
        // No Material Output: the original look (base color blended by orbit trap).
        None => "make_material(mix(u.base_color.rgb, vec3<f32>(0.9, 0.55, 0.25), pow(clamp(o.trap_r, 0.0, 1.0), 2.0)), \
                 0.0, 0.4, 0.5, vec3<f32>(0.0), 0.0, 0.0, vec3<f32>(1.0), 0.5)"
            .into(),
    };
    writeln!(src, "fn graph_material(o: Orbit) -> Material {{\n{}    return {result};\n}}", cg.body).unwrap();

    // The orbit-trap pass re-runs the whole iteration per hit pixel; skip it when the
    // color graph doesn't read Orbit Data (the built-in fallback coloring does).
    let needs_orbit = g.find(NodeKind::ColorOutput).is_none_or(|co| {
        g.upstream(co.id).iter().any(|&id| g.node(id).is_some_and(|n| n.kind == NodeKind::OrbitData))
    });
    writeln!(src, "const NEEDS_ORBIT: bool = {needs_orbit};").unwrap();
    writeln!(src, "const MAY_REFLECT: bool = {};", g.may_reflect()).unwrap();
    Ok(src)
}

/// A compiled world graph (its image files are bound by slot, see `image_files`).
pub struct WorldCode {
    pub wgsl: String,
}

/// Compiles the world graph (see the module docs). Without a World Output it's a black
/// sky with the default sun.
pub fn generate_world(g: &Graph) -> Result<WorldCode, String> {
    use super::*;
    let base = value_bases(g)?;
    let live: HashMap<NodeId, usize> = g.live_slots()?.into_iter().collect();
    let images = image_slots(g, WORLD_IMAGES)?;
    let mut src = String::new();
    let Some(out) = g.find(NodeKind::WorldOutput) else {
        src.push_str(
            "fn graph_world(dir: vec3<f32>) -> vec3<f32> { return vec3<f32>(0.0); }\n\
             fn graph_world_settings() -> WorldSettings {\n    \
             return WorldSettings(1.0, 2.0, 45.0, 40.0, vec3<f32>(1.0), 1.0, 0.0, 0.0);\n}\n",
        );
        return Ok(WorldCode { wgsl: src });
    };

    let mut cg = Codegen::new(g, &base, &live, &images, Context::World, "wp");
    let mut cache = HashMap::new();
    let color = cg.input(out.id, WORLD_COLOR, &mut cache, 1)?;
    let strength = cg.input(out.id, WORLD_STRENGTH, &mut cache, 1)?;
    writeln!(src, "fn graph_world(dir: vec3<f32>) -> vec3<f32> {{\n{}    return max({color}, vec3<f32>(0.0)) * max({strength}, 0.0);\n}}\n", cg.body)
        .unwrap();

    // Settings don't depend on a direction: anything reading it sees straight up.
    let mut cg = Codegen::new(g, &base, &live, &images, Context::World, "wp");
    let mut cache = HashMap::new();
    let settings = [
        WORLD_STRENGTH,
        WORLD_AMBIENT,
        WORLD_SUN_AZIMUTH,
        WORLD_SUN_ELEVATION,
        WORLD_SUN_COLOR,
        WORLD_SUN_STRENGTH,
        WORLD_SUN_SIZE,
        WORLD_FOG,
    ]
    .iter()
    .map(|&j| cg.input(out.id, j, &mut cache, 1))
    .collect::<Result<Vec<_>, _>>()?;
    writeln!(
        src,
        "fn graph_world_settings() -> WorldSettings {{\n    let dir = vec3<f32>(0.0, 1.0, 0.0);\n{}    return WorldSettings({});\n}}",
        cg.body,
        settings.join(", ")
    )
    .unwrap();
    Ok(WorldCode { wgsl: src })
}

/// A texture read by a processing pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Tex {
    /// The rendered fractal (rgb, alpha = distance).
    Render,
    /// The output of pass `stages[i]`.
    Stage(usize),
    /// The output of filter pass `i` (a blur, or a bloom's glow).
    Pass(usize),
    /// Image file slot `i`.
    Image(usize),
}

/// One fullscreen pass of the processing graph: a fragment shader reading `inputs`
/// (bound as `tex_0`, `tex_1`, ...) and writing its own texture.
pub struct Stage {
    pub wgsl: String,
    pub inputs: Vec<Tex>,
}

/// A filter run in passes of its own (see post.rs), reading `input`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Filter {
    Blur,
    Bloom,
}

#[derive(Clone, Copy, Debug)]
pub enum Step {
    Stage(usize),
    Filter { kind: Filter, pass: usize, input: Tex, node: NodeId },
}

/// A compiled processing graph: run `steps` in order; `output` holds the final image.
/// Image files are bound by slot (see `image_files`).
pub struct PostPlan {
    pub stages: Vec<Stage>,
    pub steps: Vec<Step>,
    pub output: Tex,
}

/// What a materialized input was computed from (to reuse its texture).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Source {
    Output(NodeId, usize),
    Unlinked(NodeId, usize),
}

#[derive(Default)]
struct PostState {
    stages: Vec<Stage>,
    steps: Vec<Step>,
    textures: HashMap<Source, Tex>,
    passes: HashMap<NodeId, usize>,
}

/// Compiles the processing graph (see the module docs).
pub fn generate_post(g: &Graph) -> Result<PostPlan, String> {
    let base = value_bases(g)?;
    let live: HashMap<NodeId, usize> = g.live_slots()?.into_iter().collect();
    let images = image_slots(g, POST_IMAGES)?;
    let out = g.find(NodeKind::PostOutput).ok_or("graph has no Image Output node")?;
    let mut state = PostState::default();
    let mut cg = Codegen::new(g, &base, &live, &images, Context::Post, "pp");
    cg.post = Some(&mut state);
    let output = cg.materialize(out.id, 0)?;
    let passes = state.passes.len();
    if passes > MAX_PASSES {
        return Err(format!("too many blur / bloom nodes ({passes} > {MAX_PASSES})"));
    }
    Ok(PostPlan { stages: state.stages, steps: state.steps, output })
}

/// Fragment shader of one processing pass (bindings: see post.rs).
fn post_stage_wgsl(body: &str, result: &str, textures: usize) -> String {
    let mut s = String::from(POST_HEADER);
    for i in 0..textures {
        writeln!(s, "@group(0) @binding({}) var tex_{i}: texture_2d<f32>;", 3 + i).unwrap();
    }
    s.push_str(crate::renderer::NODE_LIBRARY);
    s.push('\n');
    s.push_str(POST_LIBRARY);
    write!(
        s,
        "\n@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {{\n    let uv = in.uv;\n{body}    return vec4<f32>(max({result}, vec3<f32>(0.0)), 1.0);\n}}\n"
    )
    .unwrap();
    s
}

pub const POST_LIBRARY: &str = include_str!("../shaders/post.wgsl");

/// Bindings and helpers every processing pass starts with.
const POST_HEADER: &str = "\
struct PostUniform {
    size: vec4<f32>,               // width, height, 1 / width, 1 / height
    misc: vec4<f32>,               // time (s), frame index, aspect, unused
    scales: array<vec4<f32>, 4>,   // per filter pass: output scale (bloom normalization)
}
@group(0) @binding(0) var<uniform> pp: array<vec4<f32>, 256>;
@group(0) @binding(1) var<uniform> pu: PostUniform;
@group(0) @binding(2) var post_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return o;
}

fn time_s() -> f32 {
    return pu.misc.x;
}

fn pp_at(i: i32) -> f32 {
    return pp[i / 4][i % 4];
}

fn pass_scale(p: i32) -> f32 {
    return pu.scales[p / 4][p % 4];
}
";

/// Variables holding each output of an emitted node, scoped per branch.
type Cache = HashMap<NodeId, Vec<String>>;

struct Codegen<'a, 'p> {
    g: &'a Graph,
    base: &'a HashMap<NodeId, usize>,
    /// First live slot of each live (audio, Wiggle) node.
    live: &'a HashMap<NodeId, usize>,
    /// Image slot of each node reading a file.
    images: &'a HashMap<NodeId, usize>,
    ctx: Context,
    /// The parameter uniform array of this graph.
    params: &'static str,
    body: String,
    /// Processing graph: the passes built so far.
    post: Option<&'p mut PostState>,
    /// Processing graph: textures this pass reads (bound as `tex_0`, ...).
    textures: Vec<Tex>,
}

fn context_name(c: Context) -> &'static str {
    match c {
        Context::Domain => "domain",
        Context::Iteration => "iteration",
        Context::Color => "color",
        Context::World => "world",
        Context::Post => "processing",
        Context::Any => "any",
    }
}

fn convert(expr: &str, from: SocketType, to: SocketType) -> String {
    match (from, to) {
        (SocketType::Float, SocketType::Vec3) => format!("vec3<f32>({expr})"),
        (SocketType::Vec3, SocketType::Float) => format!("dot({expr}, vec3<f32>(1.0 / 3.0))"),
        _ => expr.to_string(),
    }
}

fn zero(ty: SocketType) -> String {
    match ty {
        SocketType::Float => "0.0".into(),
        _ => "vec3<f32>(0.0)".into(),
    }
}

impl<'a, 'p> Codegen<'a, 'p> {
    fn new(
        g: &'a Graph,
        base: &'a HashMap<NodeId, usize>,
        live: &'a HashMap<NodeId, usize>,
        images: &'a HashMap<NodeId, usize>,
        ctx: Context,
        params: &'static str,
    ) -> Self {
        Self { g, base, live, images, ctx, params, body: String::new(), post: None, textures: Vec::new() }
    }

    fn gp(&self, k: usize) -> String {
        format!("{}[{}][{}]", self.params, k / 4, k % 4)
    }

    /// Expression for input `j` of node `id`, typed as the input expects.
    fn input(&mut self, id: NodeId, j: usize, cache: &mut Cache, depth: usize) -> Result<String, String> {
        let node = self.g.node(id).expect("link to missing node");
        let def = &node.kind.inputs()[j];
        if def.curve {
            // The first value's index; the node's code reads the points from there.
            return Ok(format!("{}i", self.base[&id] + node.value_offset(j)));
        }
        if !def.linkable() {
            return Ok(format!("{:.1}", node.choice(j) as f32));
        }
        if let Some((src, k)) = self.g.source(id, j) {
            let outs = self.emit(src, cache, depth)?;
            let from = self.g.node(src).unwrap().kind.outputs()[k].ty;
            return Ok(convert(&outs[k], from, def.ty));
        }
        if def.implicit_position {
            return Ok(match self.ctx {
                Context::Domain => "p",
                Context::Iteration => "s0.z",
                Context::World => "dir",
                Context::Post => "vec3<f32>(uv, 0.0)",
                _ => "o.pos",
            }
            .into());
        }
        let o = self.base[&id] + node.value_offset(j);
        Ok(match def.ty {
            SocketType::State if self.ctx == Context::Iteration => "s0".into(),
            SocketType::State => return Err("state inputs only exist in the iteration graph".into()),
            SocketType::Float => self.gp(o),
            SocketType::Vec3 => format!("vec3<f32>({}, {}, {})", self.gp(o), self.gp(o + 1), self.gp(o + 2)),
        })
    }

    /// Name of the binding this pass reads `tex` through.
    fn tex_name(&mut self, tex: Tex) -> String {
        let i = match self.textures.iter().position(|t| *t == tex) {
            Some(i) => i,
            None => {
                self.textures.push(tex);
                self.textures.len() - 1
            }
        };
        format!("tex_{i}")
    }

    /// Processing graph: a texture holding input `j` of node `id` (an image), rendered
    /// by a new pass unless it already is one (the render, a filter's output, or an
    /// earlier pass computing the same thing).
    fn materialize(&mut self, id: NodeId, j: usize) -> Result<Tex, String> {
        let source = match self.g.source(id, j) {
            Some((src, k)) => {
                let n = self.g.node(src).unwrap();
                if !n.muted && k == 0 && n.kind == NodeKind::RenderLayers {
                    return Ok(Tex::Render);
                }
                if !n.muted && k == 0 && n.kind == NodeKind::Blur {
                    return Ok(Tex::Pass(self.schedule_filter(src)?));
                }
                Source::Output(src, k)
            }
            None => Source::Unlinked(id, j),
        };
        if let Some(t) = self.post.as_ref().and_then(|p| p.textures.get(&source)) {
            return Ok(*t);
        }
        let mut inner = Codegen::new(self.g, self.base, self.live, self.images, Context::Post, self.params);
        inner.post = self.post.as_deref_mut();
        let expr = inner.input(id, j, &mut HashMap::new(), 1)?;
        let expr = convert(&expr, self.g.node(id).unwrap().kind.inputs()[j].ty, SocketType::Vec3);
        let stage = Stage { wgsl: post_stage_wgsl(&inner.body, &expr, inner.textures.len()), inputs: inner.textures };
        let post = self.post.as_deref_mut().ok_or("images only exist in the processing graph")?;
        post.stages.push(stage);
        let tex = Tex::Stage(post.stages.len() - 1);
        post.steps.push(Step::Stage(post.stages.len() - 1));
        post.textures.insert(source, tex);
        Ok(tex)
    }

    /// Processing graph: schedules a blur / bloom node's filter passes (after the pass
    /// computing its input). Returns its pass index.
    fn schedule_filter(&mut self, id: NodeId) -> Result<usize, String> {
        if let Some(p) = self.post.as_ref().and_then(|p| p.passes.get(&id)) {
            return Ok(*p);
        }
        let input = self.materialize(id, 0)?;
        let kind = match self.g.node(id).unwrap().kind {
            NodeKind::Blur => Filter::Blur,
            _ => Filter::Bloom,
        };
        let post = self.post.as_deref_mut().ok_or("filters only exist in the processing graph")?;
        let pass = post.passes.len();
        post.passes.insert(id, pass);
        post.steps.push(Step::Filter { kind, pass, input, node: id });
        Ok(pass)
    }

    /// Emits `id` (and its dependencies) into `self.body`; returns one expression per
    /// output. `cache` is scoped: branches get a copy, so values computed inside an
    /// if-arm are never referenced from outside it.
    fn emit(&mut self, id: NodeId, cache: &mut Cache, depth: usize) -> Result<Vec<String>, String> {
        if let Some(v) = cache.get(&id) {
            return Ok(v.clone());
        }
        let node = self.g.node(id).expect("link to missing node");
        let kind = node.kind;
        if kind.context() != Context::Any && kind.context() != self.ctx {
            return Err(format!(
                "'{}' can only be used in the {} graph, but it is connected to the {} graph",
                kind.label(),
                context_name(kind.context()),
                context_name(self.ctx)
            ));
        }
        let indent = "    ".repeat(depth);
        let var = format!("v{id}");

        let vars = match kind {
            NodeKind::IterStart => vec!["s0".to_string()],
            _ if node.muted && kind.mutable() => {
                let mut outs = vec![self.input(id, 0, cache, depth)?];
                outs.extend(kind.outputs()[1..].iter().map(|o| zero(o.ty)));
                outs
            }
            NodeKind::Alternate | NodeKind::IterSwitch => {
                let cond = if kind == NodeKind::Alternate {
                    let (na, nb) = (self.input(id, 2, cache, depth)?, self.input(id, 3, cache, depth)?);
                    format!("alt_is_a(i, {na}, {nb})")
                } else {
                    format!("f32(i) < round({})", self.input(id, 2, cache, depth)?)
                };
                writeln!(self.body, "{indent}var {var}: State;\n{indent}if ({cond}) {{").unwrap();
                let a = self.input(id, 0, &mut cache.clone(), depth + 1)?;
                writeln!(self.body, "{indent}    {var} = {a};\n{indent}}} else {{").unwrap();
                let b = self.input(id, 1, &mut cache.clone(), depth + 1)?;
                writeln!(self.body, "{indent}    {var} = {b};\n{indent}}}").unwrap();
                vec![var]
            }
            // Computed on the CPU each frame: read the uploaded values.
            _ if kind.is_live() => {
                let first = LIVE_OFFSET + self.live[&id];
                (0..kind.outputs().len()).map(|k| self.gp(first + k)).collect()
            }
            NodeKind::RenderLayers => {
                let t = self.tex_name(Tex::Render);
                writeln!(self.body, "{indent}let {var} = textureSampleLevel({t}, post_samp, uv, 0.0);").unwrap();
                vec![
                    format!("{var}.rgb"),
                    format!("max({var}.a, 0.0)"),
                    format!("select(0.0, 1.0, {var}.a >= 0.0 && {var}.a < SKY_DEPTH)"),
                ]
            }
            NodeKind::ImageTexture => {
                let t = self.tex_name(Tex::Image(self.images[&id]));
                let a: Vec<String> = (0..3).map(|j| self.input(id, j, cache, depth)).collect::<Result<_, _>>()?;
                writeln!(self.body, "{indent}let {var} = image_texture({t}, post_samp, ({}).xy, {}, {});", a[0], a[1], int_lit(&a[2]))
                    .unwrap();
                vec![format!("{var}.rgb"), format!("{var}.a")]
            }
            NodeKind::EnvTexture => {
                let a: Vec<String> = (0..3).map(|j| self.input(id, j, cache, depth)).collect::<Result<_, _>>()?;
                let slot = self.images[&id];
                vec![format!("env_texture(world_img{slot}, world_samp, {}, {}) * max({}, 0.0)", a[0], a[1], a[2])]
            }
            _ if kind.image_input() == ImageInput::Sampled => {
                let t = self.materialize(id, 0)?;
                let t = self.tex_name(t);
                let mut args = vec![t, "post_samp".into(), "uv".into()];
                for j in 1..kind.inputs().len() {
                    args.push(self.input(id, j, cache, depth)?);
                }
                let f = match kind {
                    NodeKind::LensDistortion => "lens_distortion",
                    NodeKind::Pixelate => "pixelate",
                    NodeKind::Sharpen => "sharpen",
                    _ => "kaleidoscope",
                };
                vec![format!("{f}({})", args.join(", "))]
            }
            _ if kind.image_input() == ImageInput::Pass => {
                let p = self.schedule_filter(id)?;
                let out = self.tex_name(Tex::Pass(p));
                if kind == NodeKind::Blur {
                    vec![format!("textureSampleLevel({out}, post_samp, uv, 0.0).rgb")]
                } else {
                    let input = self.materialize(id, 0)?;
                    let input = self.tex_name(input);
                    let intensity = self.input(id, 1, cache, depth)?;
                    writeln!(
                        self.body,
                        "{indent}let {var}_g = textureSampleLevel({out}, post_samp, uv, 0.0).rgb * max({intensity}, 0.0) * pass_scale({p});"
                    )
                    .unwrap();
                    vec![format!("textureSampleLevel({input}, post_samp, uv, 0.0).rgb + {var}_g"), format!("{var}_g")]
                }
            }
            _ => {
                let args = (0..kind.inputs().len())
                    .map(|j| self.input(id, j, cache, depth))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(pre) = node_prelude(kind, &args) {
                    writeln!(self.body, "{indent}let {var} = {pre};").unwrap();
                }
                let exprs = node_exprs(kind, &args, &var);
                let mut vars = Vec::new();
                for (k, e) in exprs.into_iter().enumerate() {
                    let v = format!("{var}_{k}");
                    writeln!(self.body, "{indent}let {v} = {e};").unwrap();
                    vars.push(v);
                }
                vars
            }
        };
        cache.insert(id, vars.clone());
        Ok(vars)
    }
}

/// A value shared by several outputs, bound to `{var}` before them (e.g. the Voronoi
/// search, whose distances and cell all come from one pass).
fn node_prelude(kind: NodeKind, a: &[String]) -> Option<String> {
    use NodeKind::*;
    Some(match kind {
        NoiseTexture => format!("noise_point({}, {}, {}, {})", a[0], a[1], a[4], a[5]),
        VoronoiTexture => format!("voronoi(({}) * ({}), {}, tex_seed({}))", a[0], a[1], a[2], a[3]),
        WhiteNoise => format!("white_noise_hash({}, {})", a[0], a[1]),
        ScreenCoords => "screen_centered(uv)".into(),
        _ => return None,
    })
}

/// WGSL expression for each output, given the input expressions in order.
/// `var` is the node's variable prefix (output k is bound to `{var}_{k}`, and a
/// `node_prelude` value to `{var}`).
fn node_exprs(kind: NodeKind, a: &[String], var: &str) -> Vec<String> {
    use NodeKind::*;
    let call = |f: &str, with_c: bool| {
        let mut args = vec![a[0].clone()];
        if with_c {
            args.push("c".into());
        }
        args.extend(a[1..].iter().cloned());
        format!("{f}({})", args.join(", "))
    };
    let one = |s: String| vec![s];
    match kind {
        Mandelbulb => one(call("f_mandelbulb", true)),
        Mandelbox => one(call("f_mandelbox", true)),
        AmazingSurf => one(call("f_amazing_surf", true)),
        Bristorbrot => one(call("f_bristorbrot", true)),
        QuickDudley => one(call("f_quick_dudley", true)),
        Makin3d => one(call("f_makin3d", true)),
        PineTree => one(call("f_pine_tree", true)),
        PolyIfs => one(call("f_poly_ifs", false)),
        PolyFold => one(call("f_poly_fold", false)),
        PseudoKleinian => one(call("f_pseudo_kleinian", false)),
        Apollonian => one(call("f_apollonian", false)),
        FractFold => one(call("f_fract_fold", false)),
        Kaliset => one(call("f_kaliset", true)),
        AddC => one(call("f_add_c", true)),
        Menger => one(call("f_menger", false)),
        Sierpinski => one(call("f_sierpinski", false)),
        Kifs => one(call("f_kifs", false)),
        BoxFold => one(call("f_box_fold", false)),
        SphereFold => one(call("f_sphere_fold", false)),
        AbsFold => one(call("f_abs_fold", false)),
        OctaFold => one(call("f_octa_fold", false)),
        TetraFold => one(call("f_tetra_fold", false)),
        PlaneFold => one(call("f_plane_fold", false)),
        Scale => one(call("f_scale", false)),
        IfsScale => one(call("f_ifs_scale", false)),
        Rotate => one(call("f_rotate", false)),
        Translate => one(call("f_translate", false)),
        Twist => one(call("f_twist", false)),
        SphereInvert => one(call("f_sphere_invert", false)),
        Mix => one(format!("f_mix({}, {}, {})", a[0], a[1], a[2])),
        Xenodreambuie => one(call("f_xenodreambuie", true)),
        PolarFold => one(call("f_polar_fold", false)),
        SmoothAbsFold => one(call("f_smooth_abs_fold", false)),
        CylinderFold => one(call("f_cylinder_fold", false)),

        NoiseTexture => {
            let ch = |k: u32| format!("noise_channel({var}, {}, {}, {}, {k}u)", a[2], a[3], a[5]);
            vec![ch(0), format!("vec3<f32>({var}_0, {}, {})", ch(1), ch(2))]
        }
        VoronoiTexture => vec![
            format!("{var}.f1"),
            format!("{var}.f2 - {var}.f1"),
            format!("voronoi_color({var}, {})", a[3]),
            format!("{var}_2.x"),
        ],
        WaveTexture => one(format!("wave_texture({})", a.join(", "))),
        WhiteNoise => vec![format!("white_noise_value({var})"), format!("tex_unit3(tex_hash({var}))")],
        PolarRepeat => vec![
            format!("polar_fold({}, {}, {}, {})", a[0], a[1], a[2], a[3]),
            format!("polar_sector({}, {}, {})", a[0], a[1], a[2]),
        ],
        Blackbody => one(format!("blackbody({})", a[0])),
        HueSaturation => one(format!("hue_saturation({})", a.join(", "))),
        Blend => one(format!("blend({}, {}, {}, {})", int_lit(&a[0]), a[1], a[2], a[3])),
        RgbToBw => one(format!("dot({}, vec3<f32>(0.2126, 0.7152, 0.0722))", a[0])),

        IterInfo => vec!["f32(i)".into(), "length(s0.z)".into(), "s0.dr".into(), "s0.z".into(), "c".into()],
        OrbitData => ["o.trap_r", "o.trap.x", "o.trap.y", "o.trap.z", "o.iters"].map(String::from).to_vec(),
        Geometry => ["o.pos", "o.normal", "o.dist", "max(dot(o.normal, o.view), 0.0)"].map(String::from).to_vec(),
        Fresnel => one(format!("fresnel_schlick({}, o.normal, o.view)", a[0])),
        Value => one(a[0].clone()),
        Math => one(math_expr(op(&a[0]), &a[1], &a[2])),
        VectorMath => {
            let (x, y) = (&a[1], &a[2]);
            let v = match op(&a[0]) {
                0 => format!("({x}) + ({y})"),
                1 => format!("({x}) - ({y})"),
                2 => format!("({x}) * ({y})"),
                3 => format!("({x}) * ({})", a[3]),
                4 => format!("cross({x}, {y})"),
                5 => format!("normalize({x})"),
                _ => x.clone(),
            };
            let f = match op(&a[0]) {
                6 => format!("length({x})"),
                7 => format!("dot({x}, {y})"),
                _ => format!("length({var}_0)"),
            };
            vec![v, f]
        }
        MapRange => one(format!("map_range({}, {}, {}, {}, {}, {} > 0.5)", a[0], a[1], a[2], a[3], a[4], a[5])),
        Clamp => one(format!("clamp({}, {}, {})", a[0], a[1], a[2])),
        SeparateXyz => vec![format!("({}).x", a[0]), format!("({}).y", a[0]), format!("({}).z", a[0])],
        CombineXyz => one(format!("vec3<f32>({}, {}, {})", a[0], a[1], a[2])),
        CosinePalette => one(format!("cosine_palette({})", a.join(", "))),
        ColorRamp => one(format!("color_ramp({})", a.join(", "))),
        MixColor => one(format!("mix({}, {}, {})", a[1], a[2], a[0])),
        Hsv => one(format!("hsv_to_rgb({}, {}, {})", a[0], a[1], a[2])),
        RotateVector => one(format!("rotate_vec({}, {})", a[0], a[1])),

        Time => one(format!("time_s() * ({}) + ({})", a[0], a[1])),
        Oscillator => one(format!("oscillator({}, {}, {}, {}, {}, time_s())", int_lit(&a[0]), a[1], a[2], a[3], a[4])),
        Animate => one(format!(
            "animate({}, {}, {}, {}, {}, {}, time_s())",
            a[0],
            a[1],
            a[2],
            a[3],
            int_lit(&a[4]),
            int_lit(&a[5])
        )),
        Position => one("p".into()),
        Repeat => vec![
            format!("repeat_pos({}, {}, {}, {} > 0.5)", a[0], a[1], a[2], a[3]),
            format!("repeat_cell({}, {}, {})", a[0], a[1], a[2]),
        ],

        // World.
        SkyGradient => one(format!("sky_gradient({})", a.join(", "))),
        PhysicalSky => one(format!("physical_sky({}, world_sun_dir(), {}, {}, {}, {}, {})", a[0], a[1], a[2], a[3], a[4], a[5])),
        StarField => one(format!("star_field({}, time_s())", a.join(", "))),
        Direction => one("dir".into()),
        SunInfo => vec!["world_sun_dir()".into(), "world_sun_dir().y".into(), "world_sun_color()".into()],

        // Processing.
        ScreenCoords => vec![
            "vec3<f32>(uv, 0.0)".into(),
            format!("vec3<f32>({var}, 0.0)"),
            format!("length({var})"),
            format!("atan2({var}.y, {var}.x)"),
        ],
        Exposure => one(format!("({}) * exp2({})", a[0], a[1])),
        BrightContrast => one(format!("bright_contrast({}, {}, {})", a[0], a[1], a[2])),
        Curves => one(format!("rgb_curves({}, {}, {})", a[0], a[1], a[2])),
        ColorBalance => one(format!("color_balance({})", a.join(", "))),
        Invert => one(format!("mix({0}, max(vec3<f32>(1.0) - ({0}), vec3<f32>(0.0)), {1})", a[0], a[1])),
        Gamma => one(format!("pow(max({}, vec3<f32>(0.0)), vec3<f32>({}))", a[0], a[1])),
        Posterize => one(format!("posterize({}, {})", a[0], a[1])),
        Vignette => one(format!("vignette({}, uv, {}, {}, {}, {}, {})", a[0], a[1], a[2], a[3], a[4], a[5])),
        FilmGrain => one(format!("film_grain({}, uv, {}, {}, {} > 0.5, {} > 0.5)", a[0], a[1], a[2], a[3], a[4])),
        MotionBlur => one(a[0].clone()),

        IterStart | IterOutput | ColorOutput | DomainOutput | Alternate | IterSwitch | AudioLevel | Beat | Wiggle
        | WorldOutput | PostOutput | RenderLayers | ImageTexture | EnvTexture | LensDistortion | Pixelate | Sharpen
        | Kaleidoscope | Blur | Bloom => {
            unreachable!("handled by emit")
        }
    }
}

/// Enum inputs arrive as baked literals ("3.0").
fn op(literal: &str) -> usize {
    literal.parse::<f32>().unwrap_or(0.0) as usize
}

/// An enum literal as a WGSL i32.
fn int_lit(literal: &str) -> String {
    format!("{}i", op(literal))
}

fn math_expr(op: usize, a: &str, b: &str) -> String {
    let (a, b) = (format!("({a})"), format!("({b})"));
    match op {
        0 => format!("{a} + {b}"),
        1 => format!("{a} - {b}"),
        2 => format!("{a} * {b}"),
        3 => format!("safe_div({a}, {b})"),
        4 => format!("pow(max({a}, 0.0), {b})"),
        5 => format!("min({a}, {b})"),
        6 => format!("max({a}, {b})"),
        7 => format!("select(0.0, {a} - {b} * floor({a} / {b}), {b} != 0.0)"),
        8 => format!("sin({a})"),
        9 => format!("cos({a})"),
        10 => format!("abs({a})"),
        11 => format!("fract({a})"),
        12 => format!("floor({a})"),
        13 => format!("sqrt(max({a}, 0.0))"),
        14 => format!("log(max({a}, 1e-12))"),
        15 => format!("exp({a})"),
        16 => format!("tan({a})"),
        17 => format!("atan2({a}, {b})"),
        18 => format!("sign({a})"),
        19 => format!("select(0.0, abs(fract(({a} - {b}) / (2.0 * {b})) * 2.0 * {b} - {b}), {b} != 0.0)"),
        20 => format!("select(0.0, floor({a} / {b}) * {b}, {b} != 0.0)"),
        _ => format!("floor({a} + 0.5)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MATH_OPS, VECTOR_OPS, preset, presets};

    /// Assembles the full fractal shader exactly like the renderer and validates it.
    fn validate(g: &Graph) {
        let graph_wgsl = generate(g).expect("codegen");
        let world = generate_world(&Graph::world_from(&crate::graph::LegacyLook::default())).unwrap();
        let source = crate::renderer::assemble_shader(&graph_wgsl, &world.wgsl);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}\n--- generated ---\n{graph_wgsl}", e.emit_to_string(&source)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}\n--- generated ---\n{graph_wgsl}"));
    }

    /// Each listed preset name is spelled the same in `preset` (a typo would silently
    /// fall back to the default Mandelbulb) and listed once.
    #[test]
    fn preset_names_are_unique_and_handled() {
        let names: Vec<_> = presets().collect();
        let default = preset("Mandelbulb").0;
        for (i, name) in names.iter().enumerate() {
            assert!(!names[..i].contains(name), "{name} is listed twice");
            if *name != "Mandelbulb" {
                assert!(!preset(name).0.same_content(&default), "{name} isn't handled by preset()");
            }
        }
    }

    #[test]
    fn presets_compile() {
        for name in presets() {
            validate(&preset(name).0);
        }
    }

    /// Every state node chained in the iteration graph with float inputs driven by math
    /// nodes, plus every color/math/utility node in the color graph.
    #[test]
    fn every_node_kind_compiles() {
        use NodeKind::*;
        let mut g = Graph::new();
        let start = g.add(IterStart, [0.0; 2]);
        let out = g.add(IterOutput, [0.0; 2]);
        let info = g.add(IterInfo, [0.0; 2]);
        let mut prev = start;
        for &kind in NodeKind::ALL {
            let has_state_in = kind.inputs().first().is_some_and(|i| i.ty == SocketType::State);
            if !has_state_in || matches!(kind, IterOutput | Mix | Alternate | IterSwitch) {
                continue;
            }
            let n = g.add(kind, [0.0; 2]);
            assert!(g.connect(prev, 0, n, 0), "{kind:?}");
            // Drive the first float input from the iteration index, through a Math node.
            if let Some(j) = kind.inputs().iter().position(|i| i.ty == SocketType::Float && i.linkable()) {
                let m = g.add(Math, [0.0; 2]);
                g.connect(info, 0, m, 1);
                assert!(g.connect(m, 0, n, j));
            }
            prev = n;
        }
        let mix = g.add(Mix, [0.0; 2]);
        g.connect(prev, 0, mix, 0);
        g.connect(start, 0, mix, 1);
        let inner = g.add(Alternate, [0.0; 2]);
        g.connect(mix, 0, inner, 0);
        let switch = g.add(IterSwitch, [0.0; 2]);
        g.connect(inner, 0, switch, 0);
        g.connect(prev, 0, switch, 1); // a node shared by both branches
        g.connect(switch, 0, out, 0);

        // Color graph through every color, math and utility node.
        let orbit = g.add(OrbitData, [0.0; 2]);
        let geo = g.add(Geometry, [0.0; 2]);
        let mut chain = vec![];
        for (op, _) in MATH_OPS.iter().enumerate() {
            let m = g.add(Math, [0.0; 2]);
            g.node_mut(m).unwrap().set(0, &[op as f32]);
            g.connect(orbit, op % 5, m, 1);
            chain.push(m);
        }
        let mut f = chain[0];
        for &m in &chain[1..] {
            g.connect(f, 0, m, 2);
            f = m;
        }
        let mut v = geo;
        for (op, _) in VECTOR_OPS.iter().enumerate() {
            let vm = g.add(VectorMath, [0.0; 2]);
            g.node_mut(vm).unwrap().set(0, &[op as f32]);
            g.connect(v, 0, vm, 1);
            g.connect(geo, 1, vm, 2);
            v = vm;
        }
        let sep = g.add(SeparateXyz, [0.0; 2]);
        g.connect(v, 0, sep, 0);
        let comb = g.add(CombineXyz, [0.0; 2]);
        g.connect(sep, 2, comb, 0);
        g.connect(f, 0, comb, 1);
        let map = g.add(MapRange, [0.0; 2]);
        g.connect(sep, 0, map, 0);
        let clamp = g.add(Clamp, [0.0; 2]);
        g.connect(map, 0, clamp, 0);
        let value = g.add(Value, [0.0; 2]);
        g.connect(value, 0, clamp, 2);
        let pal = g.add(CosinePalette, [0.0; 2]);
        g.connect(clamp, 0, pal, 0);
        g.connect(comb, 0, pal, 4); // vector into a vector input
        let ramp = g.add(ColorRamp, [0.0; 2]);
        g.connect(v, 1, ramp, 0);
        let hsv = g.add(Hsv, [0.0; 2]);
        g.connect(geo, 2, hsv, 0);
        let mixc = g.add(MixColor, [0.0; 2]);
        g.connect(pal, 0, mixc, 0); // color into a float input (implicit average)
        let fres = g.add(Fresnel, [0.0; 2]);
        g.connect(geo, 3, fres, 0);
        g.connect(ramp, 0, mixc, 1);
        g.connect(hsv, 0, mixc, 2);
        let co = g.add(ColorOutput, [0.0; 2]);
        g.connect(mixc, 0, co, 0);
        // Drive every material input.
        for j in 1..NodeKind::ColorOutput.inputs().len() {
            g.connect(fres, 0, co, j);
        }

        // Domain graph with drivers: Position -> Repeat -> Rotate Vector (angles from
        // every waveform / play mode) -> Domain Output; Repeat's Cell also feeds a Math.
        let pos = g.add(Position, [0.0; 2]);
        let rep = g.add(Repeat, [0.0; 2]);
        g.node_mut(rep).unwrap().set(3, &[1.0]);
        g.connect(pos, 0, rep, 0);
        let mut v = rep;
        for wave in 0..crate::graph::WAVEFORMS.len() {
            let osc = g.add(Oscillator, [0.0; 2]);
            g.node_mut(osc).unwrap().set(0, &[wave as f32]);
            let anim = g.add(Animate, [0.0; 2]);
            g.node_mut(anim).unwrap().set(4, &[(wave % 3) as f32]);
            g.connect(osc, 0, anim, 0);
            let rot = g.add(RotateVector, [0.0; 2]);
            g.connect(v, 0, rot, 0);
            g.connect(anim, 0, rot, 1);
            v = rot;
        }
        let time = g.add(Time, [0.0; 2]);
        let m = g.add(Math, [0.0; 2]);
        g.connect(rep, 1, m, 1);
        g.connect(time, 0, m, 2);
        let scale = g.add(VectorMath, [0.0; 2]);
        g.node_mut(scale).unwrap().set(0, &[3.0]);
        g.connect(v, 0, scale, 1);
        g.connect(m, 0, scale, 3);
        let dout = g.add(DomainOutput, [0.0; 2]);
        g.connect(scale, 0, dout, 0);
        // Drivers also work in the iteration and color graphs.
        let osc = g.add(Oscillator, [0.0; 2]);
        g.connect(osc, 0, mixc, 0);
        validate(&g);
        assert!(g.uses_time());
        assert!(g.unbounded_view_distance().is_some());
    }

    #[test]
    fn position_outside_domain_graph_is_an_error() {
        let (mut g, _) = preset("Mandelbulb");
        let bulb = g.nodes.iter().find(|n| n.kind == NodeKind::Mandelbulb).unwrap().id;
        let rep = g.add(NodeKind::Repeat, [0.0; 2]);
        assert!(g.connect(rep, 0, bulb, 1));
        let err = generate(&g).unwrap_err();
        assert!(err.contains("domain graph"), "{err}");
    }

    #[test]
    fn disconnected_drivers_dont_animate() {
        let (mut g, _) = preset("Mandelbulb");
        assert!(!g.uses_time());
        g.add(NodeKind::Oscillator, [0.0; 2]);
        assert!(!g.uses_time(), "an unconnected driver must not stop accumulation");
        assert!(preset("Breathing bulb").0.uses_time());
    }

    /// Enum choices are baked into the code, so every value must compile.
    #[test]
    fn every_variant_compiles() {
        use NodeKind::*;
        let (mut g, _) = preset("Mandelbulb");
        let out = g.find(IterOutput).unwrap().id;
        let mut prev = g.find(IterStart).unwrap().id;
        for v in 0..crate::graph::BULB_VARIANTS.len() {
            let b = g.add(Mandelbulb, [0.0; 2]);
            g.node_mut(b).unwrap().set(3, &[v as f32]);
            g.connect(prev, 0, b, 0);
            prev = b;
        }
        for kind in [PolyFold, PolyIfs] {
            for k in 0..crate::graph::POLYHEDRA.len() {
                let n = g.add(kind, [0.0; 2]);
                g.node_mut(n).unwrap().set(1, &[k as f32]);
                g.connect(prev, 0, n, 0);
                prev = n;
            }
        }
        g.connect(prev, 0, out, 0);
        validate(&g);
    }

    /// Textures, polar nodes and the newer color nodes in all three graphs (unlinked
    /// vectors read each graph's own position), with every enum choice.
    #[test]
    fn texture_and_polar_nodes_compile_everywhere() {
        use NodeKind::*;
        let (mut g, _) = preset("Mandelbulb");
        let out = g.find(IterOutput).unwrap().id;
        let bulb = g.nodes.iter().find(|n| n.kind == Mandelbulb).unwrap().id;

        // Iteration: polar / cylinder folds on every axis before the bulb; noise of z
        // drives the c weight, Voronoi of c (from Iteration Info) the power.
        let mut prev = g.find(IterStart).unwrap().id;
        for axis in 0..crate::graph::AXES.len() {
            for kind in [PolarFold, CylinderFold] {
                let n = g.add(kind, [0.0; 2]);
                let j = if kind == PolarFold { 2 } else { 3 };
                g.node_mut(n).unwrap().set(j, &[axis as f32]);
                g.connect(prev, 0, n, 0);
                prev = n;
            }
        }
        g.connect(prev, 0, bulb, 0);
        let noise = g.add(NoiseTexture, [0.0; 2]);
        g.node_mut(noise).unwrap().set(4, &[0.5]); // distortion
        assert!(g.connect(noise, 0, bulb, 2));
        let info = g.add(IterInfo, [0.0; 2]);
        let vor = g.add(VoronoiTexture, [0.0; 2]);
        assert!(g.connect(info, 4, vor, 0));
        assert!(g.connect(vor, 1, bulb, 1));
        let white = g.add(WhiteNoise, [0.0; 2]);
        assert!(g.connect(info, 3, white, 0));
        let xeno = g.add(Xenodreambuie, [0.0; 2]);
        g.connect(bulb, 0, xeno, 0);
        g.connect(white, 0, xeno, 2);
        g.connect(xeno, 0, out, 0);

        // Domain: Position -> Polar Repeat (segments wiggling) + noise color -> output.
        let pos = g.add(Position, [0.0; 2]);
        let polar = g.add(PolarRepeat, [0.0; 2]);
        g.connect(pos, 0, polar, 0);
        let wig = g.add(Wiggle, [0.0; 2]);
        g.connect(wig, 0, polar, 1);
        let dnoise = g.add(NoiseTexture, [0.0; 2]);
        let add = g.add(VectorMath, [0.0; 2]);
        g.connect(polar, 0, add, 1);
        g.connect(dnoise, 1, add, 2);
        let dout = g.add(DomainOutput, [0.0; 2]);
        g.connect(add, 0, dout, 0);
        let sector = g.add(PolarRepeat, [0.0; 2]); // unlinked position, Sector output used
        assert!(g.connect(sector, 1, polar, 1));
        assert!(!g.connect(sector, 1, wig, 0), "Wiggle settings are CPU knobs");
        g.connect(wig, 0, add, 3);

        // Color: every wave type x profile summed into the hue shift.
        let co = g.find(ColorOutput).unwrap().id;
        let cvor = g.add(VoronoiTexture, [0.0; 2]);
        let hue = g.add(HueSaturation, [0.0; 2]);
        g.connect(cvor, 2, hue, 0);
        let mut sum = cvor;
        let mut sum_out = 3;
        for kind in 0..crate::graph::WAVE_TYPES.len() {
            for profile in 0..crate::graph::WAVEFORMS.len() {
                let w = g.add(WaveTexture, [0.0; 2]);
                g.node_mut(w).unwrap().set(1, &[kind as f32]);
                g.node_mut(w).unwrap().set(2, &[profile as f32]);
                g.node_mut(w).unwrap().set(4, &[(profile % 2) as f32]);
                let m = g.add(Math, [0.0; 2]);
                g.connect(sum, sum_out, m, 1);
                g.connect(w, 0, m, 2);
                (sum, sum_out) = (m, 0);
            }
        }
        g.connect(sum, 0, hue, 1);
        assert!(g.connect(hue, 0, co, 0));
        let bb = g.add(Blackbody, [0.0; 2]);
        let cnoise = g.add(NoiseTexture, [0.0; 2]);
        g.connect(cnoise, 0, bb, 0);
        g.connect(bb, 0, co, 4);
        let cwhite = g.add(WhiteNoise, [0.0; 2]);
        g.connect(cwhite, 1, co, 7);

        let src = generate(&g).unwrap();
        for (needle, what) in [("noise_point(s0.z", "iteration"), ("polar_fold(p,", "domain"), ("(o.pos)", "color")] {
            assert!(src.contains(needle), "{what} graph should read its own position ({needle})\n{src}");
        }
        validate(&g);
    }

    /// Copy + paste brings the links between copied nodes along, never output nodes,
    /// and survives the clipboard text round trip.
    #[test]
    fn copy_paste_keeps_internal_links() {
        use crate::graph::Clip;
        let (mut g, _) = preset("Breathing bulb");
        let bulb = g.nodes.iter().find(|n| n.kind == NodeKind::Mandelbulb).unwrap().id;
        let osc = g.find(NodeKind::Oscillator).unwrap().id;
        let out = g.find(NodeKind::IterOutput).unwrap().id;
        g.node_mut(bulb).unwrap().locked = true;
        let clip = g.copy(&[bulb, osc, out]);
        assert_eq!(clip.nodes.len(), 2, "Iteration Output is never copied");
        let clip = Clip::from_text(&clip.to_text()).expect("round trip");
        assert!(Clip::from_text("just some text").is_none());

        let before = g.nodes.len();
        let new = g.paste(&clip, [500.0, 500.0], crate::graph::GraphKind::Material);
        assert_eq!(g.nodes.len(), before + 2);
        let of = |kind| *new.iter().find(|&&id| g.node(id).unwrap().kind == kind).unwrap();
        let (nb, no) = (of(NodeKind::Mandelbulb), of(NodeKind::Oscillator));
        assert!(!g.node(nb).unwrap().locked, "pasted nodes can be moved into place");
        assert!(g.links.iter().any(|l| l.from == no && l.to == nb), "oscillator -> bulb link copied");
        assert!(!g.links.iter().any(|l| l.to == nb && l.input == 0), "no link from outside the copy");
        validate(&g);
    }

    /// Keyframed inputs: sampled into the values without a recompile; `packed_at`
    /// samples without touching the graph; key states follow edits; auto-key keys them.
    #[test]
    fn keyframed_inputs_animate() {
        use crate::graph::KeyState;
        let (mut g, _) = preset("Mandelbulb");
        let bulb = g.nodes.iter().find(|n| n.kind == NodeKind::Mandelbulb).unwrap().id;
        let rev = g.structure_rev;
        let n = g.node_mut(bulb).unwrap();
        assert!(n.keyable(1) && !n.keyable(0) && !n.keyable(3), "power yes; state and enum inputs no");
        n.set(1, &[8.0]);
        n.insert_key(1, 0.0);
        n.set(1, &[4.0]);
        n.insert_key(1, 2.0);
        assert_eq!(g.structure_rev, rev, "keying never recompiles");
        assert!(g.uses_time() && g.is_animated());

        g.animate(1.0);
        let mid = g.node(bulb).unwrap().value(1);
        assert!(mid > 4.0 && mid < 8.0, "power at t=1: {mid}");
        assert_eq!(g.node(bulb).unwrap().key_state(1, 1.0), KeyState::Animated);
        assert_eq!(g.node(bulb).unwrap().key_state(1, 2.0), KeyState::Changed, "value is the t=1 one");

        let o = g.packed_values().len() - g.nodes.iter().skip_while(|n| n.id != bulb).map(|n| n.values.len()).sum::<usize>()
            + g.node(bulb).unwrap().value_offset(1);
        assert_eq!(g.packed_at(2.0)[o], 4.0);
        assert_eq!(g.node(bulb).unwrap().value(1), mid, "packed_at leaves the graph alone");

        // Edit while paused at t=1: Changed until keyed (auto-key does it).
        g.node_mut(bulb).unwrap().set(1, &[6.5]);
        assert_eq!(g.node(bulb).unwrap().key_state(1, 1.0), KeyState::Changed);
        assert!(g.autokey(1.0));
        assert_eq!(g.node(bulb).unwrap().key_state(1, 1.0), KeyState::OnKey);
        assert_eq!(g.key_times(), vec![0.0, 1.0, 2.0]);
        validate(&g);
    }

    /// Undo ignores values that only moved with the playhead, but records key edits.
    #[test]
    fn history_ignores_playback_but_records_keys() {
        let (mut g, _) = preset("Mandelbulb");
        let bulb = g.nodes.iter().find(|n| n.kind == NodeKind::Mandelbulb).unwrap().id;
        let n = g.node_mut(bulb).unwrap();
        n.insert_key(1, 0.0);
        n.set(1, &[3.0]);
        n.insert_key(1, 4.0);
        let mut gs = crate::graph::Graphs { material: g, ..Default::default() };
        let mut h = crate::graph::history::History::default();
        h.commit(&gs);
        for t in [0.5, 1.7, 3.2] {
            gs.animate(t);
            h.commit(&gs);
        }
        assert!(!h.can_undo(), "playback is not an edit");
        gs.material.node_mut(bulb).unwrap().delete_key(1, 4.0);
        h.commit(&gs);
        assert!(h.can_undo(), "deleting a key is");
        h.undo(&mut gs);
        assert_eq!(gs.material.node(bulb).unwrap().track(1).unwrap().keys.len(), 2);
    }

    /// Moving / deleting keys by time, vector inputs, save/load and paste.
    #[test]
    fn node_keys_edit_save_and_paste() {
        let (mut g, _) = preset("Psychedelic Juliabulb");
        let tr = g.find(NodeKind::Translate).unwrap().id;
        let bulb = g.nodes.iter().find(|n| n.kind == NodeKind::Mandelbulb).unwrap().id;
        g.disconnect(tr, 1);
        for (t, v) in [(0.0, [0.1, 0.2, 0.3]), (1.0, [0.5, -0.5, 0.0])] {
            let n = g.node_mut(tr).unwrap();
            n.set(1, &v);
            n.insert_key(1, t);
        }
        g.node_mut(bulb).unwrap().insert_key(1, 1.0);
        g.move_keys(&[1.0], 0.5);
        assert_eq!(g.key_times(), vec![0.0, 1.5]);
        g.animate(1.5);
        let o = g.node(tr).unwrap().value_offset(1);
        assert_eq!(&g.node(tr).unwrap().values[o..o + 3], &[0.5, -0.5, 0.0], "vector keys move with their time");

        let json = serde_json::to_string(&g).unwrap();
        let mut back: Graph = serde_json::from_str(&json).unwrap();
        back.repair();
        assert!(back.same_content(&g), "keys survive save / load");

        let clip = g.copy(&[tr]);
        let new = g.paste(&clip, [0.0, 0.0], crate::graph::GraphKind::Material)[0];
        assert_eq!(g.node(new).unwrap().anim, g.node(tr).unwrap().anim, "pasted nodes keep their keys");

        assert_eq!(g.delete_keys_at(&[1.5]), 3, "the bulb, the translate and its pasted copy each had one");
        assert_eq!(g.key_times(), vec![0.0]);
        validate(&g);
    }

    /// A muted formula node passes its state through, on the GPU and in the CPU eval.
    #[test]
    fn muted_node_is_bypassed() {
        let (mut g, iters) = preset("Bulb + Box hybrid");
        let bx = g.nodes.iter().find(|n| n.kind == NodeKind::Mandelbox).unwrap().id;
        assert!(NodeKind::Mandelbox.mutable() && !NodeKind::Math.mutable() && !NodeKind::IterOutput.mutable());
        let p = glam::Vec3::new(0.3, -0.9, 0.6);
        let d_on = crate::graph::eval::distance(&g, p, iters, 0.0, &[]);
        let before = g.clone();
        g.node_mut(bx).unwrap().muted = true;
        assert!(!g.same_structure(&before), "muting needs a recompile");
        let src = generate(&g).unwrap();
        assert!(!src.contains("f_mandelbox"), "{src}");
        validate(&g);
        let d_off = crate::graph::eval::distance(&g, p, iters, 0.0, &[]);
        assert!((d_on - d_off).abs() > 1e-4, "muting the box changes the shape ({d_on} vs {d_off})");
    }

    #[test]
    fn orbit_pass_only_when_needed() {
        let (mut g, _) = preset("Mandelbulb");
        assert!(generate(&g).unwrap().contains("NEEDS_ORBIT: bool = true"));
        // Replace the palette input with an inline color: no Orbit Data upstream.
        let co = g.find(NodeKind::ColorOutput).unwrap().id;
        g.disconnect(co, 0);
        let src = generate(&g).unwrap();
        assert!(src.contains("NEEDS_ORBIT: bool = false"), "{src}");
        validate(&g);
    }

    #[test]
    fn reflections_compiled_only_when_possible() {
        let src = |name: &str| generate(&preset(name).0).unwrap();
        assert!(src("Mandelbulb").contains("MAY_REFLECT: bool = false"));
        assert!(src("Chrome Mandelbox").contains("MAY_REFLECT: bool = true"));
        let (mut g, _) = preset("Mandelbulb");
        let m = g.find(NodeKind::ColorOutput).unwrap().id;
        let v = g.add(NodeKind::Value, [0.0; 2]);
        g.connect(v, 0, m, crate::graph::MAT_ROUGHNESS);
        assert!(g.may_reflect(), "driven roughness may go low");
    }

    #[test]
    fn wrong_context_is_an_error() {
        let (mut g, _) = preset("Mandelbulb");
        let bulb = g.nodes.iter().find(|n| n.kind == NodeKind::Mandelbulb).unwrap().id;
        let orbit = g.add(NodeKind::OrbitData, [0.0; 2]);
        assert!(g.connect(orbit, 0, bulb, 1));
        let err = generate(&g).unwrap_err();
        assert!(err.contains("color graph"), "{err}");
    }

    #[test]
    fn cut_outputs_leaves_a_trivial_graph() {
        for name in presets() {
            let mut g = preset(name).0;
            let nodes = g.nodes.len();
            assert!(g.cut_outputs() > 0, "{name}");
            assert!(g.links.iter().all(|l| g.node(l.to).unwrap().kind.category() != crate::graph::Category::Output));
            assert_eq!(g.nodes.len(), nodes, "{name}: nodes must survive");
            validate(&g);
            assert_eq!(g.cut_outputs(), 0);
        }
    }

    #[test]
    fn invalid_links_are_rejected() {
        use NodeKind::*;
        let mut g = Graph::new();
        let a = g.add(Scale, [0.0; 2]);
        let b = g.add(Rotate, [0.0; 2]);
        let m = g.add(Math, [0.0; 2]);
        let v = g.add(Value, [0.0; 2]);
        assert!(g.connect(a, 0, b, 0));
        assert!(!g.connect(b, 0, a, 0), "cycle");
        assert!(!g.connect(a, 0, a, 0), "self link");
        assert!(!g.connect(v, 0, b, 0), "float into state");
        assert!(!g.connect(a, 0, m, 1), "state into float");
        assert!(!g.connect(v, 0, m, 0), "enum inputs aren't linkable");
        assert!(g.connect(v, 0, b, 1), "float into vector converts");
    }
}
