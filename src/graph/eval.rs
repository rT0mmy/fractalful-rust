//! CPU evaluation of the iteration graph, mirroring nodes.wgsl.
//! Used for the camera's distance-to-surface (speed scaling), so it only needs to be
//! as accurate as the GPU version, not faster. Color-graph nodes are never reached
//! from Iteration Output in a valid graph and evaluate to zero.

use std::collections::HashMap;

use glam::{IVec3, Vec3, Vec3Swizzles};

use super::{Graph, NodeId, NodeKind, SocketType};

#[derive(Clone, Copy)]
struct State {
    z: Vec3,
    dr: f32,
}

#[derive(Clone, Copy)]
enum Val {
    S(State),
    F(f32),
    V(Vec3),
}

impl Val {
    fn f(self) -> f32 {
        match self {
            Val::F(f) => f,
            Val::V(v) => (v.x + v.y + v.z) / 3.0,
            Val::S(_) => 0.0,
        }
    }
    fn v(self) -> Vec3 {
        match self {
            Val::F(f) => Vec3::splat(f),
            Val::V(v) => v,
            Val::S(_) => Vec3::ZERO,
        }
    }
    fn s(self, fallback: State) -> State {
        match self {
            Val::S(s) => s,
            _ => fallback,
        }
    }
}

/// Distance estimate at `p` at animation time `t`, matching `de_n` in fractal.wgsl.
/// `live` holds the live (audio) node outputs, as uploaded to the shader.
pub fn distance(g: &Graph, p: Vec3, iters: u32, t: f32, live: &[f32]) -> f32 {
    let slots: HashMap<NodeId, usize> = g.live_slots().unwrap_or_default().into_iter().collect();
    let live = Live { values: live, slots: &slots };
    let Some(out) = g.find(NodeKind::IterOutput) else { return 1.0 };
    let (bailout, _, mode) = g.output_settings();
    let unit = State { z: p, dr: 1.0 };
    // Domain graph first (e.g. Repeat), like `graph_domain`.
    let q = match g.find(NodeKind::DomainOutput) {
        Some(d) => Ctx { g, s0: unit, c: p, p, i: 0, t, live }.input(d.id, 0, &mut HashMap::new()).v(),
        None => p,
    };
    let mut s = State { z: q, dr: 1.0 };
    for i in 0..iters {
        let ctx = Ctx { g, s0: s, c: q, p, i, t, live };
        s = ctx.input(out.id, 0, &mut HashMap::new()).s(s);
        if s.z.length_squared() > bailout * bailout {
            break;
        }
    }
    let r = s.z.length();
    let dr = s.dr.abs().max(1e-12);
    match mode {
        1 => r / dr,
        2 => {
            let rxy = s.z.truncate().length();
            (rxy - 0.92784).max((rxy * s.z.z).abs() / r.max(1e-12)) / dr
        }
        3 => 0.25 * s.z.y.abs() / dr,
        _ => 0.5 * r.max(1e-12).ln() * r / dr,
    }
}

/// Value of input `j` of node `id` at time `t`, outside any fractal iteration (position
/// inputs read zero; color-only nodes give zero). For settings evaluated on the CPU, such
/// as a processing filter's radius driven by an Oscillator or an Audio Level node.
pub fn input_value(g: &Graph, id: NodeId, j: usize, t: f32, live: &[f32]) -> f32 {
    let slots: HashMap<NodeId, usize> = g.live_slots().unwrap_or_default().into_iter().collect();
    let live = Live { values: live, slots: &slots };
    let s0 = State { z: Vec3::ZERO, dr: 1.0 };
    let ctx = Ctx { g, s0, c: Vec3::ZERO, p: Vec3::ZERO, i: 0, t, live };
    if g.node(id).is_none() {
        return 0.0;
    }
    ctx.input(id, j, &mut HashMap::new()).f()
}

struct Ctx<'a> {
    g: &'a Graph,
    s0: State,
    c: Vec3,
    /// Domain position (what unlinked position inputs read).
    p: Vec3,
    i: u32,
    /// Animation time in seconds.
    t: f32,
    live: Live<'a>,
}

#[derive(Clone, Copy)]
struct Live<'a> {
    values: &'a [f32],
    slots: &'a HashMap<NodeId, usize>,
}

type Cache = HashMap<NodeId, Vec<Val>>;

impl Ctx<'_> {
    fn input(&self, id: NodeId, j: usize, cache: &mut Cache) -> Val {
        let node = self.g.node(id).expect("link to missing node");
        let def = &node.kind.inputs()[j];
        if let Some((src, k)) = self.g.source(id, j) {
            let v = self.eval(src, cache)[k];
            return match def.ty {
                SocketType::Float => Val::F(v.f()),
                SocketType::Vec3 => Val::V(v.v()),
                SocketType::State => v,
            };
        }
        let o = node.value_offset(j);
        match def.ty {
            // The domain point in the domain graph (where s0.z = p), z when iterating.
            _ if def.implicit_position => Val::V(self.s0.z),
            _ if !def.linkable() => Val::F(node.values[o]),
            SocketType::State => Val::S(self.s0),
            SocketType::Float => Val::F(node.values[o]),
            SocketType::Vec3 => Val::V(Vec3::from_slice(&node.values[o..o + 3])),
        }
    }

    fn eval(&self, id: NodeId, cache: &mut Cache) -> Vec<Val> {
        if let Some(v) = cache.get(&id) {
            return v.clone();
        }
        use NodeKind::*;
        let node = self.g.node(id).expect("link to missing node");
        let kind = node.kind;
        let s0 = self.s0;
        // Only the active branch is evaluated, as on the GPU.
        let branch = |a: bool, cache: &mut Cache| self.input(id, if a { 0 } else { 1 }, cache);
        let out = match kind {
            IterStart => vec![Val::S(s0)],
            _ if node.muted && kind.mutable() => vec![self.input(id, 0, cache)],
            Alternate => {
                let a = self.input(id, 2, cache).f().round().max(0.0);
                let period = (a + self.input(id, 3, cache).f().round().max(0.0)).max(1.0);
                let k = self.i as f32 - (self.i as f32 / period).floor() * period;
                vec![branch(k < a, cache)]
            }
            IterSwitch => {
                let at = self.input(id, 2, cache).f().round();
                vec![branch((self.i as f32) < at, cache)]
            }
            _ => {
                let a: Vec<Val> = (0..kind.inputs().len()).map(|j| self.input(id, j, cache)).collect();
                let st = |j: usize| a[j].s(s0);
                let f = |j: usize| a[j].f();
                let v = |j: usize| a[j].v();
                let c = self.c;
                let one = |s: State| vec![Val::S(s)];
                match kind {
                    Mandelbulb => one(mandelbulb(st(0), c, f(1), f(2), node.choice(3))),
                    QuickDudley => {
                        let z = st(0).z;
                        let n = Vec3::new(z.x * z.x - 2.0 * z.y * z.z, z.z * z.z + 2.0 * z.x * z.y, z.y * z.y + 2.0 * z.x * z.z);
                        one(quadratic(st(0), n, c, f(1)))
                    }
                    Makin3d => {
                        let z = st(0).z;
                        let n = Vec3::new(z.x * z.x + 2.0 * z.y * z.z, -z.y * z.y - 2.0 * z.x * z.z, -z.z * z.z + 2.0 * z.x * z.y);
                        one(quadratic(st(0), n, c, f(1)))
                    }
                    PineTree => {
                        let z = st(0).z;
                        let yz2 = z.y * z.y + z.z * z.z;
                        let t = 2.0 * z.x / yz2.max(1e-12).sqrt();
                        let n = Vec3::new(z.x * z.x - yz2, 2.0 * t * z.y * z.z, t * (z.y * z.y - z.z * z.z));
                        one(quadratic(st(0), n, c, f(1)))
                    }
                    PolyFold => one(State { z: poly_fold(st(0).z, node.choice(1)), dr: st(0).dr }),
                    PseudoKleinian => {
                        let s = st(0);
                        let size = v(1);
                        let z = 2.0 * s.z.clamp(-size, size) - s.z;
                        let k = (f(2) / z.length_squared().max(1e-12)).max(1.0);
                        one(State { z: z * k, dr: s.dr * k })
                    }
                    FractFold => one(State { z: fract_fold(st(0).z, v(1)), dr: st(0).dr }),
                    Apollonian => {
                        let s = st(0);
                        let z = fract_fold(s.z, v(2));
                        let k = f(1) / z.length_squared().max(1e-12);
                        one(State { z: z * k, dr: s.dr * k })
                    }
                    PolyIfs => {
                        let kind = node.choice(1);
                        let n = 3 + kind;
                        let cospin = (std::f32::consts::PI / n as f32).cos();
                        let sc = (0.75 - cospin * cospin).sqrt();
                        let (pab, pbc, pca) =
                            (Vec3::Z, Vec3::new(sc, 0.0, 0.5).normalize(), Vec3::new(0.0, sc, cospin).normalize());
                        let w = v(3);
                        let d = w.x * pab + w.y * pbc + w.z * pca;
                        let center = d / d.length().max(1e-6);
                        let folded = State { z: poly_fold(st(0).z, kind), dr: st(0).dr };
                        one(ifs_scale(rotate(folded, Vec3::new(0.0, 0.0, f(4))), f(2), center))
                    }
                    Mandelbox => one(mandelbox(st(0), c, [f(1), f(2), f(3), f(4), f(5)], false)),
                    AmazingSurf => one(mandelbox(st(0), c, [f(1), f(2), f(3), f(4), f(5)], true)),
                    Bristorbrot => {
                        let (z, cw) = (st(0).z, f(1));
                        let n = Vec3::new(
                            z.x * z.x - z.y * z.y - z.z * z.z,
                            z.y * (2.0 * z.x - z.z),
                            z.z * (2.0 * z.x + z.y),
                        );
                        one(State { z: n + c * cw, dr: 2.0 * z.length() * st(0).dr + cw.abs() })
                    }
                    Kaliset => {
                        let s = st(0);
                        let k = f(1) / s.z.length_squared().max(1e-6);
                        one(State { z: s.z.abs() * k - v(2) + c * f(3), dr: s.dr * k.abs() + f(3).abs() })
                    }
                    Menger => {
                        let (scale, o) = (f(1), v(2) * (f(1) - 1.0));
                        let mut z = octa_fold(st(0).z) * scale - o;
                        if z.z < -0.5 * o.z {
                            z.z += o.z;
                        }
                        one(State { z, dr: st(0).dr * scale })
                    }
                    Sierpinski => {
                        let z = tetra_fold(st(0).z) * f(1) - Vec3::splat(f(2) * (f(1) - 1.0));
                        one(State { z, dr: st(0).dr * f(1) })
                    }
                    Kifs => {
                        let s = State { z: octa_fold(st(0).z), dr: st(0).dr };
                        one(ifs_scale(rotate(s, Vec3::new(0.0, 0.0, f(3))), f(1), v(2)))
                    }
                    BoxFold => one(box_fold(st(0), f(1))),
                    SphereFold => one(sphere_fold(st(0), f(1), f(2))),
                    AbsFold => {
                        let s = st(0);
                        let m = Vec3::new(f(1), f(2), f(3));
                        one(State { z: s.z + (s.z.abs() - s.z) * m, dr: s.dr })
                    }
                    OctaFold => one(State { z: octa_fold(st(0).z), dr: st(0).dr }),
                    TetraFold => one(State { z: tetra_fold(st(0).z), dr: st(0).dr }),
                    PlaneFold => {
                        let s = st(0);
                        let n = v(1) / v(1).length().max(1e-6);
                        let t = s.z.dot(n) - f(2);
                        one(State { z: s.z - 2.0 * t.min(0.0) * n, dr: s.dr })
                    }
                    Scale => one(scale(st(0), f(1))),
                    IfsScale => one(ifs_scale(st(0), f(1), v(2))),
                    Rotate => one(rotate(st(0), v(1))),
                    Translate => one(State { z: st(0).z + v(1), dr: st(0).dr }),
                    Twist => {
                        let s = st(0);
                        let (sn, cs) = (f(1) * s.z.y).sin_cos();
                        let z = Vec3::new(cs * s.z.x - sn * s.z.z, s.z.y, sn * s.z.x + cs * s.z.z);
                        one(State { z, dr: s.dr })
                    }
                    AddC => one(add_c(st(0), c, f(1))),
                    SphereInvert => {
                        let s = st(0);
                        let k = f(1) * f(1) / s.z.length_squared().max(1e-12);
                        one(State { z: s.z * k, dr: s.dr * k })
                    }
                    Mix => {
                        let (x, y, t) = (st(0), st(1), f(2));
                        one(State { z: x.z.lerp(y.z, t), dr: x.dr + (y.dr - x.dr) * t })
                    }
                    Xenodreambuie => one(xenodreambuie(st(0), c, f(1), f(2), f(3), f(4))),
                    PolarFold => {
                        let s = st(0);
                        one(State { z: polar_fold(s.z, f(1), node.choice(2), node.choice(3) == 1), dr: s.dr })
                    }
                    SmoothAbsFold => {
                        let s = st(0);
                        one(State { z: (s.z * s.z + Vec3::splat(f(1) * f(1))).sqrt(), dr: s.dr })
                    }
                    CylinderFold => {
                        let s = st(0);
                        let k = sphere_fold_k(to_axis(s.z, node.choice(3)).xy().length_squared(), f(1), f(2));
                        one(State { z: s.z * k, dr: s.dr * k })
                    }
                    NoiseTexture => {
                        let q = noise_point(v(0), f(1), f(4), f(5));
                        let ch = |k: u32| noise_channel(q, f(2), f(3), f(5), k);
                        let fac = ch(0);
                        vec![Val::F(fac), Val::V(Vec3::new(fac, ch(1), ch(2)))]
                    }
                    VoronoiTexture => {
                        let (f1, f2, cell) = voronoi(v(0) * f(1), f(2), tex_seed(f(3)));
                        let color = tex_unit3(tex_hash3(cell, tex_seed(f(3)) ^ 0x9e37_79b9));
                        vec![Val::F(f1), Val::F(f2 - f1), Val::V(color), Val::F(color.x)]
                    }
                    WaveTexture => vec![Val::F(wave_texture(v(0), node.choice(1), node.choice(2), f(3), f(4), f(5)))],
                    WhiteNoise => {
                        let h = white_noise_hash(v(0), f(1));
                        vec![Val::F((h >> 8) as f32 / 16_777_215.0), Val::V(tex_unit3(tex_hash(h)))]
                    }
                    PolarRepeat => {
                        let (p, n, axis) = (v(0), f(1), node.choice(2));
                        vec![Val::V(polar_fold(p, n, axis, node.choice(3) == 1)), Val::F(polar_sector(p, n, axis))]
                    }
                    IterInfo => vec![
                        Val::F(self.i as f32),
                        Val::F(s0.z.length()),
                        Val::F(s0.dr),
                        Val::V(s0.z),
                        Val::V(self.c),
                    ],
                    Value => vec![Val::F(f(0))],
                    Math => vec![Val::F(math(node.choice(0), f(1), f(2)))],
                    VectorMath => {
                        let (x, y) = (v(1), v(2));
                        let r = match node.choice(0) {
                            0 => x + y,
                            1 => x - y,
                            2 => x * y,
                            3 => x * f(3),
                            4 => x.cross(y),
                            5 => x.normalize_or_zero(),
                            _ => x,
                        };
                        let val = match node.choice(0) {
                            6 => x.length(),
                            7 => x.dot(y),
                            _ => r.length(),
                        };
                        vec![Val::V(r), Val::F(val)]
                    }
                    MapRange => {
                        let mut t = safe_div(f(0) - f(1), f(2) - f(1));
                        if node.choice(5) == 1 {
                            t = t.clamp(0.0, 1.0);
                        }
                        vec![Val::F(f(3) + (f(4) - f(3)) * t)]
                    }
                    Clamp => vec![Val::F(f(0).max(f(1)).min(f(2)))],
                    SeparateXyz => vec![Val::F(v(0).x), Val::F(v(0).y), Val::F(v(0).z)],
                    CombineXyz => vec![Val::V(Vec3::new(f(0), f(1), f(2)))],
                    RotateVector => vec![Val::V(rotate(State { z: v(0), dr: 1.0 }, v(1)).z)],
                    Time => vec![Val::F(self.t * f(0) + f(1))],
                    Oscillator => {
                        vec![Val::F(f(3) + f(2) * wave(node.choice(0), self.t * f(1) + f(4)))]
                    }
                    Animate => {
                        let mut x = (self.t - f(3)).max(0.0) / f(2).max(1e-4);
                        x = match node.choice(4) {
                            0 => x.min(1.0),
                            1 => x.fract(),
                            _ => 1.0 - (1.0 - 2.0 * (x * 0.5).fract()).abs(),
                        };
                        if node.choice(5) == 1 {
                            x = x * x * (3.0 - 2.0 * x);
                        }
                        vec![Val::F(f(0) + (f(1) - f(0)) * x)]
                    }
                    AudioLevel | Beat | Wiggle => {
                        let first = self.live.slots.get(&id).copied().unwrap_or(usize::MAX);
                        (0..kind.outputs().len())
                            .map(|k| Val::F(self.live.values.get(first.wrapping_add(k)).copied().unwrap_or(0.0)))
                            .collect()
                    }
                    Position => vec![Val::V(self.p)],
                    Repeat => {
                        let (pos, size, limit) = (v(0), v(1).abs().max(Vec3::splat(1e-4)), v(2).abs());
                        let lim = Vec3::select(limit.cmpge(Vec3::splat(0.5)), limit.round(), Vec3::splat(1e9));
                        let id = (pos / size).round().clamp(-lim, lim);
                        let mut q = pos - size * id;
                        if node.choice(3) == 1 {
                            let odd = id.abs() - 2.0 * (id.abs() * 0.5).floor();
                            q *= Vec3::ONE - 2.0 * odd;
                        }
                        vec![Val::V(q), Val::V(id)]
                    }
                    // Color-graph-only nodes: unreachable from Iteration Output when valid.
                    _ => kind
                        .outputs()
                        .iter()
                        .map(|o| if o.ty == SocketType::Float { Val::F(0.0) } else { Val::V(Vec3::ZERO) })
                        .collect(),
                }
            }
        };
        cache.insert(id, out.clone());
        out
    }
}

fn wave(kind: usize, x: f32) -> f32 {
    let f = x - x.floor();
    match kind {
        1 => 1.0 - 4.0 * (f - 0.5).abs(),
        2 => 2.0 * f - 1.0,
        3 => {
            if f < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        _ => (std::f32::consts::TAU * f).sin(),
    }
}

fn fract_fold(z: Vec3, half: Vec3) -> Vec3 {
    let h = half.abs().max(Vec3::splat(1e-4));
    let q = 0.5 * z / h + 0.5;
    h * (2.0 * (q - q.floor()) - 1.0)
}

fn safe_div(a: f32, b: f32) -> f32 {
    if b != 0.0 { a / b } else { 0.0 }
}

fn math(op: usize, a: f32, b: f32) -> f32 {
    match op {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => safe_div(a, b),
        4 => a.max(0.0).powf(b),
        5 => a.min(b),
        6 => a.max(b),
        7 => {
            if b != 0.0 {
                a - b * (a / b).floor()
            } else {
                0.0
            }
        }
        8 => a.sin(),
        9 => a.cos(),
        10 => a.abs(),
        11 => a - a.floor(),
        12 => a.floor(),
        13 => a.max(0.0).sqrt(),
        14 => a.max(1e-12).ln(),
        15 => a.exp(),
        16 => a.tan(),
        17 => a.atan2(b),
        // WGSL sign(0) is 0, unlike f32::signum.
        18 => {
            if a > 0.0 {
                1.0
            } else if a < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        19 => {
            if b != 0.0 {
                let x = (a - b) / (2.0 * b);
                ((x - x.floor()) * 2.0 * b - b).abs()
            } else {
                0.0
            }
        }
        20 => {
            if b != 0.0 {
                (a / b).floor() * b
            } else {
                0.0
            }
        }
        _ => round_up(a),
    }
}

// ---- Mirrors of the newer nodes.wgsl functions ------------------------------------

/// Wiggle node output at animation time `t`: smooth random motion, fBm along a line
/// through noise space, about center +- amplitude. Computed on the CPU once per frame
/// (a live node), so it costs the shader nothing.
pub fn wiggle(n: &super::Node, t: f32) -> f32 {
    let (freq, amp, center, detail, seed) = (n.value(0), n.value(1), n.value(2), n.value(3), n.value(4));
    center + amp * fbm(Vec3::new(t * freq, 0.37, 0.71), detail, 0.5, tex_seed(seed).wrapping_add(4099))
}

/// `round_up` in nodes.wgsl: half up, not half-to-even nor half-away-from-zero.
fn round_up(x: f32) -> f32 {
    (x + 0.5).floor()
}

fn cmul(a: glam::Vec2, b: glam::Vec2) -> glam::Vec2 {
    glam::Vec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
}

/// See `f_xenodreambuie` in nodes.wgsl for the power 8 / 2 fast path.
fn xenodreambuie(s: State, c: Vec3, power: f32, alpha: f32, beta: f32, cw: f32) -> State {
    if power != 8.0 && power != 2.0 {
        return xenodreambuie_trig(s, c, power, alpha, beta, cw);
    }
    let z = s.z;
    let w = if power == 8.0 { bulb_pow8(z) } else { bulb_pow2(z) };
    let rho2 = z.xy().length_squared();
    let mut u = if rho2 > 1e-24 { z.xy() / rho2.sqrt() } else { glam::Vec2::X };
    u = cmul(u, u);
    if power == 8.0 {
        u = cmul(u, u);
        u = cmul(u, u);
    }
    let (sa, ca) = alpha.to_radians().sin_cos();
    let sr = w.xy().dot(u);
    let m = cmul((sr * ca + w.z * sa) * u, glam::Vec2::from_angle(beta.to_radians()));
    let rp = z.length().powf(power - 1.0);
    State { z: m.extend(w.z * ca - sr * sa) + c * cw, dr: rp * power * s.dr + cw.abs() }
}

fn xenodreambuie_trig(s: State, c: Vec3, power: f32, alpha: f32, beta: f32, cw: f32) -> State {
    let r = s.z.length().max(1e-6);
    let th = (s.z.z / r).clamp(-1.0, 1.0).acos() * power + alpha.to_radians();
    let phi = s.z.y.atan2(s.z.x) * power + beta.to_radians();
    let rp = r.powf(power - 1.0);
    let dir = Vec3::new(th.sin() * phi.cos(), th.sin() * phi.sin(), th.cos());
    State { z: rp * r * dir + c * cw, dr: rp * power * s.dr + cw.abs() }
}

fn to_axis(p: Vec3, axis: usize) -> Vec3 {
    match axis {
        0 => p.yzx(),
        1 => p.zxy(),
        _ => p,
    }
}

fn from_axis(q: Vec3, axis: usize) -> Vec3 {
    match axis {
        0 => q.zxy(),
        1 => q.yzx(),
        _ => q,
    }
}

use std::f32::consts::TAU;

fn polar_fold(p: Vec3, n: f32, axis: usize, mirror: bool) -> Vec3 {
    let q = to_axis(p, axis);
    let sector = TAU / round_up(n).max(1.0);
    let mut a = q.y.atan2(q.x);
    a -= sector * round_up(a / sector);
    if mirror {
        a = a.abs();
    }
    let r = q.xy().length();
    from_axis(Vec3::new(r * a.cos(), r * a.sin(), q.z), axis)
}

fn polar_sector(p: Vec3, n: f32, axis: usize) -> f32 {
    let q = to_axis(p, axis);
    let k = round_up(n).max(1.0);
    let id = round_up(q.y.atan2(q.x) / (TAU / k));
    id - k * (id / k).floor()
}

fn tex_hash(x: u32) -> u32 {
    let s = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

fn tex_hash3(c: IVec3, seed: u32) -> u32 {
    tex_hash(c.x as u32 ^ tex_hash(c.y as u32 ^ tex_hash(c.z as u32 ^ seed)))
}

fn tex_unit3(h: u32) -> Vec3 {
    Vec3::new((h & 1023) as f32, ((h >> 10) & 1023) as f32, ((h >> 20) & 1023) as f32) / 1023.0
}

fn tex_seed(seed: f32) -> u32 {
    round_up(seed).max(0.0) as u32
}

const NOISE_GAIN: f32 = 1.6;

fn gradient_noise(p: Vec3, seed: u32) -> f32 {
    let cell = p.floor();
    let i = cell.as_ivec3();
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let mut n = 0.0;
    for k in 0..8 {
        let o = IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let of = o.as_vec3();
        let g = tex_unit3(tex_hash3(i.wrapping_add(o), seed)) * 2.0 - 1.0;
        let w = Vec3::select(o.cmpeq(IVec3::ONE), u, 1.0 - u);
        n += w.x * w.y * w.z * g.dot(f - of);
    }
    n * NOISE_GAIN
}

fn fbm(p: Vec3, detail: f32, roughness: f32, seed: u32) -> f32 {
    let octaves = detail.clamp(0.0, 8.0);
    let (mut sum, mut norm, mut amp, mut q) = (0.0, 0.0, 1.0, p);
    for k in 0..9u32 {
        let w = (octaves + 1.0 - k as f32).clamp(0.0, 1.0);
        if w <= 0.0 {
            break;
        }
        sum += amp * w * gradient_noise(q, seed.wrapping_add(k));
        norm += (amp * w) * (amp * w);
        amp *= roughness;
        q *= 2.0;
    }
    sum / norm.max(1e-12).sqrt()
}

fn noise_point(p: Vec3, scale: f32, distortion: f32, seed: f32) -> Vec3 {
    let q = p * scale;
    if distortion == 0.0 {
        return q;
    }
    let s = tex_seed(seed).wrapping_add(101);
    let warp = Vec3::new(gradient_noise(q, s), gradient_noise(q, s.wrapping_add(1)), gradient_noise(q, s.wrapping_add(2)));
    q + distortion * warp
}

fn noise_channel(q: Vec3, detail: f32, roughness: f32, seed: f32, channel: u32) -> f32 {
    0.5 + 0.5 * fbm(q, detail, roughness, tex_seed(seed).wrapping_add(16 * channel))
}

/// (nearest distance, second nearest, nearest point's cell).
fn voronoi(p: Vec3, randomness: f32, seed: u32) -> (f32, f32, IVec3) {
    let base = p.floor();
    let i = base.as_ivec3();
    let f = p - base;
    let (mut f1, mut f2, mut best) = (1e9f32, 1e9f32, i);
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let o = IVec3::new(x, y, z);
                let d = (o.as_vec3() + tex_unit3(tex_hash3(i.wrapping_add(o), seed)) * randomness - f).length();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                    best = i.wrapping_add(o);
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
    }
    (f1, f2, best)
}

fn white_noise_hash(v: Vec3, seed: f32) -> u32 {
    let bits = |x: f32| if x == 0.0 { 0 } else { x.to_bits() as i32 };
    tex_hash3(IVec3::new(bits(v.x), bits(v.y), bits(v.z)), tex_seed(seed))
}

fn wave_texture(p: Vec3, kind: usize, profile: usize, scale: f32, distortion: f32, phase: f32) -> f32 {
    let q = p * scale;
    let mut x = match kind {
        0 => q.x,
        1 => q.y,
        2 => q.z,
        3 => q.length(),
        _ => q.xy().length() + q.y.atan2(q.x) / TAU,
    };
    if distortion != 0.0 {
        x += distortion * fbm(q, 2.0, 0.5, 7);
    }
    0.5 + 0.5 * wave(profile, x + phase)
}

/// `bulb_pow2` in nodes.wgsl: doubles the spherical angles, squares the radius.
fn bulb_pow2(z: Vec3) -> Vec3 {
    let rho2 = z.x * z.x + z.y * z.y;
    let k = if rho2 > 1e-24 { 2.0 * z.z / rho2.sqrt() } else { 0.0 };
    Vec3::new(k * (z.x * z.x - z.y * z.y), k * 2.0 * z.x * z.y, z.z * z.z - rho2)
}

/// `bulb_pow8` in nodes.wgsl: iq's polynomial with axes remapped to ours.
fn bulb_pow8(p: Vec3) -> Vec3 {
    let (x, y, z) = (p.y, p.z, p.x);
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let (x4, y4, z4) = (x2 * x2, y2 * y2, z2 * z2);
    let k3 = (x2 + z2).max(1e-10);
    let k2 = k3.sqrt().recip().powi(7);
    let k1 = x4 + y4 + z4 - 6.0 * y2 * z2 - 6.0 * x2 * y2 + 2.0 * z2 * x2;
    let k4 = x2 - y2 + z2;
    let nx = 64.0 * x * y * z * (x2 - z2) * k4 * (x4 - 6.0 * x2 * z2 + z4) * k1 * k2;
    let ny = -16.0 * y2 * k3 * k4 * k4 + k1 * k1;
    let nz = -8.0 * y * k4 * (x4 * x4 - 28.0 * x4 * x2 * z2 + 70.0 * x4 * z4 - 28.0 * x2 * z2 * z4 + z4 * z4) * k1 * k2;
    Vec3::new(nz, nx, ny)
}

/// See `f_mandelbulb` in nodes.wgsl for the variants and the power-of-two fast path.
fn mandelbulb(s: State, c: Vec3, power: f32, cw: f32, variant: usize) -> State {
    let z = if variant == 2 { s.z.abs() } else { s.z };
    if variant != 1 && (power == 8.0 || power == 2.0) {
        let mut w = if power == 8.0 { bulb_pow8(z) } else { bulb_pow2(z) };
        if variant == 3 {
            w.y = -w.y;
        }
        let rp = z.length().powf(power - 1.0);
        return State { z: w + c * cw, dr: rp * power * s.dr + cw.abs() };
    }
    mandelbulb_trig(s, c, power, cw, variant)
}

fn mandelbulb_trig(s: State, c: Vec3, power: f32, cw: f32, variant: usize) -> State {
    let z = if variant == 2 { s.z.abs() } else { s.z };
    let r = z.length().max(1e-6);
    let mut phi = z.y.atan2(z.x) * power;
    if variant == 3 {
        phi = -phi;
    }
    let dir = if variant == 1 {
        let th = (z.z / r).clamp(-1.0, 1.0).asin() * power;
        Vec3::new(th.cos() * phi.cos(), th.cos() * phi.sin(), th.sin())
    } else {
        let th = (z.z / r).clamp(-1.0, 1.0).acos() * power;
        Vec3::new(th.sin() * phi.cos(), th.sin() * phi.sin(), th.cos())
    };
    let rp = r.powf(power - 1.0);
    State { z: rp * r * dir + c * cw, dr: rp * power * s.dr + cw.abs() }
}

fn quadratic(s: State, n: Vec3, c: Vec3, cw: f32) -> State {
    State { z: n + c * cw, dr: 2.0 * s.z.length() * s.dr + cw.abs() }
}

fn poly_fold(z: Vec3, kind: usize) -> Vec3 {
    let n = 3 + kind;
    let cospin = (std::f32::consts::PI / n as f32).cos();
    let nc = Vec3::new(-0.5, -cospin, (0.75 - cospin * cospin).sqrt());
    let mut z = z;
    for _ in 0..n {
        z = Vec3::new(z.x.abs(), z.y.abs(), z.z);
        z -= 2.0 * z.dot(nc).min(0.0) * nc;
    }
    z
}

/// Mandelbox, or Amazing Surf when `surf` (box fold on x and y only).
/// `p` = [scale, min radius, fixed radius, fold limit, c weight].
fn mandelbox(s: State, c: Vec3, p: [f32; 5], surf: bool) -> State {
    let [k, min_r, fixed_r, limit, cw] = p;
    let folded = if surf {
        let xy = s.z.truncate();
        let xy = xy.clamp(glam::Vec2::splat(-limit), glam::Vec2::splat(limit)) * 2.0 - xy;
        State { z: xy.extend(s.z.z), dr: s.dr }
    } else {
        box_fold(s, limit)
    };
    add_c(scale(sphere_fold(folded, min_r, fixed_r), k), c, cw)
}

fn box_fold(s: State, limit: f32) -> State {
    State { z: s.z.clamp(Vec3::splat(-limit), Vec3::splat(limit)) * 2.0 - s.z, dr: s.dr }
}

fn sphere_fold_k(r2: f32, min_r: f32, fixed_r: f32) -> f32 {
    let min2 = (min_r * min_r).max(1e-6);
    let fix2 = fixed_r * fixed_r;
    if r2 < min2 {
        fix2 / min2
    } else if r2 < fix2 {
        fix2 / r2.max(1e-12)
    } else {
        1.0
    }
}

fn sphere_fold(s: State, min_r: f32, fixed_r: f32) -> State {
    let k = sphere_fold_k(s.z.length_squared(), min_r, fixed_r);
    State { z: s.z * k, dr: s.dr * k }
}

fn scale(s: State, k: f32) -> State {
    State { z: s.z * k, dr: s.dr * k.abs() }
}

fn ifs_scale(s: State, k: f32, center: Vec3) -> State {
    State { z: s.z * k - center * (k - 1.0), dr: s.dr * k.abs() }
}

fn add_c(s: State, c: Vec3, w: f32) -> State {
    State { z: s.z + c * w, dr: s.dr + w.abs() }
}

fn rotate(s: State, angles: Vec3) -> State {
    let (sx, cx) = angles.x.to_radians().sin_cos();
    let (sy, cy) = angles.y.to_radians().sin_cos();
    let (sz, cz) = angles.z.to_radians().sin_cos();
    let mut z = s.z;
    z = Vec3::new(z.x, cx * z.y - sx * z.z, sx * z.y + cx * z.z);
    z = Vec3::new(cy * z.x + sy * z.z, z.y, -sy * z.x + cy * z.z);
    z = Vec3::new(cz * z.x - sz * z.y, sz * z.x + cz * z.y, z.z);
    State { z, dr: s.dr }
}

fn octa_fold(z: Vec3) -> Vec3 {
    let mut z = z.abs();
    if z.x < z.y {
        z = Vec3::new(z.y, z.x, z.z);
    }
    if z.x < z.z {
        z = Vec3::new(z.z, z.y, z.x);
    }
    if z.y < z.z {
        z = Vec3::new(z.x, z.z, z.y);
    }
    z
}

fn tetra_fold(z: Vec3) -> Vec3 {
    let mut z = z;
    if z.x + z.y < 0.0 {
        z = Vec3::new(-z.y, -z.x, z.z);
    }
    if z.x + z.z < 0.0 {
        z = Vec3::new(-z.z, z.y, -z.x);
    }
    if z.y + z.z < 0.0 {
        z = Vec3::new(z.x, -z.z, -z.y);
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{preset, presets};

    /// Each preset's surface must lie inside its bound radius, or the bounding-sphere
    /// clip in the raymarcher visibly cuts it off.
    #[test]
    fn presets_fit_their_bound() {
        for name in presets() {
            let (g, iters) = preset(name);
            if g.unbounded_view_distance().is_some() {
                continue; // repeats through space: no bound by design
            }
            let bound = g.output_settings().1;
            let (n, extent) = (48, bound * 2.0);
            let mut max_r: f32 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let f = |a: i32| (a as f32 / (n - 1) as f32 * 2.0 - 1.0) * extent;
                        let p = Vec3::new(f(i), f(j), f(k));
                        if distance(&g, p, iters, 0.0, &[]) < extent / n as f32 {
                            max_r = max_r.max(p.length());
                        }
                    }
                }
            }
            println!("{name}: surface reaches r = {max_r:.2}, bound = {bound}");
            assert!(max_r > 0.0, "{name}: no surface found");
            assert!(max_r < bound, "{name}: surface reaches {max_r:.2} but bound is {bound}");
        }
    }

    /// Presets that pick their own starting view must start it in open space, not
    /// inside (or grazing) the surface.
    #[test]
    fn camera_hints_start_in_open_space() {
        for name in presets() {
            let (g, iters) = preset(name);
            if let Some((p, _, _)) = g.camera_hint {
                let d = distance(&g, Vec3::from(p), iters, 0.0, &[]);
                assert!(d > 0.02, "{name}: camera starts {d} from the surface");
            }
        }
    }

    /// Landscapes go on forever: marched to a view distance, with a chosen viewpoint
    /// (the default one for repeating fields would put the camera at y = cell / 2).
    #[test]
    fn landscapes_are_endless() {
        let (_, names) = crate::graph::PRESET_CATEGORIES.iter().find(|(c, _)| *c == "Landscapes").unwrap();
        assert!(names.len() >= 10);
        for name in *names {
            let (g, _) = preset(name);
            assert!(g.unbounded_view_distance().is_some(), "{name} is bounded");
            assert!(g.camera_hint.is_some(), "{name} has no camera hint");
        }
    }

    /// Repeat must make the distance field periodic: the same distance one cell over.
    #[test]
    fn repeat_is_periodic() {
        let (g, iters) = preset("Infinite Menger field");
        for p in [Vec3::new(0.3, 1.1, -0.4), Vec3::new(1.7, -0.2, 0.9)] {
            let d0 = distance(&g, p, iters, 0.0, &[]);
            for shift in [Vec3::X * 4.0, Vec3::new(-8.0, 4.0, 12.0)] {
                let d1 = distance(&g, p + shift, iters, 0.0, &[]);
                assert!((d0 - d1).abs() < 1e-3, "{p} vs {}: {d0} != {d1}", p + shift);
            }
        }
    }

    /// The angle-doubling fast path must match the trig formula it replaces.
    #[test]
    fn fast_mandelbulb_matches_trig() {
        let mut seed = 0x2545_f491_u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 * 3.0 - 1.5
        };
        let c = Vec3::new(0.1, -0.2, 0.3);
        for power in [2.0, 8.0] {
            for variant in [0, 2, 3] {
                for _ in 0..2000 {
                    let s = State { z: Vec3::new(rnd(), rnd(), rnd()), dr: 1.7 };
                    let (a, b) = (mandelbulb(s, c, power, 1.0, variant), mandelbulb_trig(s, c, power, 1.0, variant));
                    let scale = 1.0 + b.z.length();
                    assert!((a.z - b.z).length() / scale < 1e-3, "p{power} v{variant} z {}: {} vs {}", s.z, a.z, b.z);
                    assert!((a.dr - b.dr).abs() / (1.0 + b.dr) < 1e-3, "p{power} v{variant} dr");
                }
            }
        }
        // Points on the z axis (rho = 0) are the special case.
        for zc in [-1.2f32, -0.5, 0.7] {
            let s = State { z: Vec3::new(0.0, 0.0, zc), dr: 1.0 };
            let (a, b) = (mandelbulb(s, c, 8.0, 1.0, 0), mandelbulb_trig(s, c, 8.0, 1.0, 0));
            assert!((a.z - b.z).length() < 1e-3, "axis {zc}: {} vs {}", a.z, b.z);
        }
    }

    /// The Xenodreambuie fast path (polynomial + rotations) must match the trig formula.
    #[test]
    fn fast_xenodreambuie_matches_trig() {
        let mut seed = 0x0bad_5eed_u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 * 3.0 - 1.5
        };
        let c = Vec3::new(0.1, -0.2, 0.3);
        for power in [2.0, 8.0] {
            for _ in 0..4000 {
                let s = State { z: Vec3::new(rnd(), rnd(), rnd()), dr: 1.3 };
                let (alpha, beta) = (rnd() * 120.0, rnd() * 120.0);
                let a = xenodreambuie(s, c, power, alpha, beta, 1.0);
                let b = xenodreambuie_trig(s, c, power, alpha, beta, 1.0);
                let scale = 1.0 + b.z.length();
                assert!((a.z - b.z).length() / scale < 1e-3, "p{power} a{alpha} b{beta} z {}: {} vs {}", s.z, a.z, b.z);
                assert!((a.dr - b.dr).abs() / (1.0 + b.dr) < 1e-3);
            }
            // On the z axis phi is undefined; both must take phi = 0.
            for zc in [-1.2f32, 0.7] {
                let s = State { z: Vec3::new(0.0, 0.0, zc), dr: 1.0 };
                let (a, b) = (xenodreambuie(s, c, power, 30.0, 50.0, 1.0), xenodreambuie_trig(s, c, power, 30.0, 50.0, 1.0));
                assert!((a.z - b.z).length() < 1e-3, "axis {zc}: {} vs {}", a.z, b.z);
            }
        }
    }

    /// NOISE_GAIN keeps gradient noise (and so fBm) about -1..1: rarely beyond, but
    /// using most of the range so Noise Texture's Fac has full contrast around 0.5.
    #[test]
    fn noise_range() {
        let mut seed = 0x1234_5677_u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 * 40.0 - 20.0
        };
        for detail in [0.0, 3.0] {
            let n = 100_000;
            let (mut lo, mut hi, mut sq, mut outside) = (f32::MAX, f32::MIN, 0.0, 0);
            for _ in 0..n {
                let v = fbm(Vec3::new(rnd(), rnd(), rnd()), detail, 0.5, 3);
                lo = lo.min(v);
                hi = hi.max(v);
                sq += v * v;
                outside += (v.abs() > 1.0) as u32;
            }
            let rms = (sq / n as f32).sqrt();
            println!("detail {detail}: {lo:.3}..{hi:.3}, rms {rms:.3}, outside ±1: {outside}");
            assert!(lo > -1.5 && hi < 1.5, "detail {detail}: {lo}..{hi}");
            assert!(rms > 0.2 && rms < 0.5, "detail {detail}: rms {rms}");
            assert!(outside < n / 100, "detail {detail}: {outside} samples beyond ±1");
        }
    }

    #[test]
    fn polar_fold_is_rigid_and_sectors_are_in_range() {
        for axis in 0..3 {
            for (i, n) in [1.0, 5.0, 6.0, 12.0].into_iter().enumerate() {
                let p = Vec3::new(0.3 + i as f32, -1.1, 0.7 * axis as f32 - 0.5);
                let q = polar_fold(p, n, axis, true);
                assert!((q.length() - p.length()).abs() < 1e-4, "length changed");
                assert!((to_axis(q, axis).z - to_axis(p, axis).z).abs() < 1e-5, "moved along the axis");
                let s = polar_sector(p, n, axis);
                assert!(s >= 0.0 && s < n && s == s.floor(), "sector {s} of {n}");
            }
        }
    }

    #[test]
    fn white_noise_is_stable_per_value() {
        let a = white_noise_hash(Vec3::new(1.0, 0.0, -2.0), 0.0);
        assert_eq!(a, white_noise_hash(Vec3::new(1.0, -0.0, -2.0), 0.0), "-0 and 0 must agree");
        assert_ne!(a, white_noise_hash(Vec3::new(1.0, 0.0, -2.0), 1.0), "seed must matter");
        assert_ne!(a, white_noise_hash(Vec3::new(2.0, 0.0, -2.0), 0.0));
    }

    #[test]
    fn oscillator_drives_the_fractal_over_time() {
        let (g, iters) = preset("Breathing bulb");
        let p = Vec3::new(0.0, 0.0, -1.3);
        let (a, b) = (distance(&g, p, iters, 0.0, &[]), distance(&g, p, iters, 2.5, &[]));
        assert!((a - b).abs() > 1e-3, "power should change between t=0 and t=2.5 ({a} vs {b})");
    }
}


