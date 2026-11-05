// Node library. Formula nodes in src/graph map to the `f_*` functions here, with
// arguments in input order (plus `c` after the state for formulas that add it).
// src/graph/eval.rs mirrors every iteration-graph function on the CPU; keep in sync.
//
// `State.dr` is the running scalar derivative used for the distance estimate:
// conformal maps (scale, sphere inversion) multiply it, rigid ones (rotations,
// reflections/folds, translation) leave it alone, and adding c adds the c weight.

struct State {
    z: vec3<f32>,
    dr: f32,
}

// Data available to the color graph at a surface hit.
struct Orbit {
    trap_r: f32,      // min |z| over the orbit
    trap: vec3<f32>,  // min |z.x|, |z.y|, |z.z| over the orbit
    iters: f32,       // iterations run before bailout
    pos: vec3<f32>,
    normal: vec3<f32>,
    dist: f32,        // distance from the camera
    view: vec3<f32>,  // unit direction towards the camera
}

// Surface material produced by the material graph (Material Output).
struct Material {
    base: vec3<f32>,
    metallic: f32,
    roughness: f32,
    specular: f32,
    emission: vec3<f32>, // color * strength (HDR, feeds bloom)
    subsurface: f32,
    sss_color: vec3<f32>,
    sss_radius: f32,
}

fn make_material(
    base: vec3<f32>, metallic: f32, roughness: f32, specular: f32,
    emission: vec3<f32>, strength: f32, subsurface: f32, sss_color: vec3<f32>, sss_radius: f32,
) -> Material {
    return Material(
        max(base, vec3<f32>(0.0)), clamp(metallic, 0.0, 1.0), clamp(roughness, 0.0, 1.0), clamp(specular, 0.0, 1.0),
        max(emission, vec3<f32>(0.0)) * max(strength, 0.0), clamp(subsurface, 0.0, 1.0),
        max(sss_color, vec3<f32>(0.0)), clamp(sss_radius, 0.0, 1.0),
    );
}

// Schlick Fresnel reflectance for index of refraction `ior` (dielectric).
fn fresnel_schlick(ior: f32, n: vec3<f32>, v: vec3<f32>) -> f32 {
    let r0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    return r0 + (1.0 - r0) * pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 5.0);
}

// ---- Fractal formulas ------------------------------------------------------------

// Trig-free Mandelbulb powers, exactly equal to the trig formula in f_mandelbulb
// (theta = acos(z/r), phi = atan2(y, x)). Only 2 and 8 have this property: repeated
// squaring is NOT the same as power 4/16, because 2*theta leaves [0, pi].

// Power 2: doubles both angles and squares the radius.
fn bulb_pow2(z: vec3<f32>) -> vec3<f32> {
    let rho2 = z.x * z.x + z.y * z.y;
    let k = select(0.0, 2.0 * z.z * inverseSqrt(rho2), rho2 > 1e-24);
    return vec3<f32>(k * (z.x * z.x - z.y * z.y), k * 2.0 * z.x * z.y, z.z * z.z - rho2);
}

// Power 8: iq's Chebyshev expansion, with his y-up axes (x, y, z) remapped to ours
// as (y, z, x).
fn bulb_pow8(p: vec3<f32>) -> vec3<f32> {
    let x = p.y;
    let y = p.z;
    let z = p.x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let y2 = y * y;
    let y4 = y2 * y2;
    let z2 = z * z;
    let z4 = z2 * z2;
    let k3 = max(x2 + z2, 1e-10);
    let ik = inverseSqrt(k3);
    let k2 = ik * ik * ik * ik * ik * ik * ik;
    let k1 = x4 + y4 + z4 - 6.0 * y2 * z2 - 6.0 * x2 * y2 + 2.0 * z2 * x2;
    let k4 = x2 - y2 + z2;
    let nx = 64.0 * x * y * z * (x2 - z2) * k4 * (x4 - 6.0 * x2 * z2 + z4) * k1 * k2;
    let ny = -16.0 * y2 * k3 * k4 * k4 + k1 * k1;
    let nz = -8.0 * y * k4 * (x4 * x4 - 28.0 * x4 * x2 * z2 + 70.0 * x4 * z4 - 28.0 * x2 * z2 * z4 + z4 * z4) * k1 * k2;
    return vec3<f32>(nz, nx, ny);
}

// variant: 0 standard, 1 cosine (White's original, theta from the equator),
// 2 Burning Ship (abs first), 3 Mandelbar (conjugate). Baked as a constant.
fn f_mandelbulb(s: State, c: vec3<f32>, power: f32, cw: f32, variant: f32) -> State {
    let v = i32(variant);
    var z = s.z;
    if (v == 2) { z = abs(z); }
    // Fast path for power 8 (the default) and 2: polynomial, identical to the trig
    // formula below but several times cheaper (no acos/atan2/pow/sin/cos). The branch
    // is uniform across the frame unless the power is driven per pixel.
    if (v != 1 && (power == 8.0 || power == 2.0)) {
        var w = select(bulb_pow2(z), bulb_pow8(z), power == 8.0);
        if (v == 3) { w.y = -w.y; }
        let r2 = dot(z, z);
        let rp = select(sqrt(r2), r2 * r2 * r2 * sqrt(r2), power == 8.0); // r^(power - 1)
        return State(w + c * cw, rp * power * s.dr + abs(cw));
    }
    let r = max(length(z), 1e-6);
    var phi = atan2(z.y, z.x) * power;
    if (v == 3) { phi = -phi; }
    var dir: vec3<f32>;
    if (v == 1) {
        let th = asin(clamp(z.z / r, -1.0, 1.0)) * power;
        dir = vec3<f32>(cos(th) * cos(phi), cos(th) * sin(phi), sin(th));
    } else {
        let th = acos(clamp(z.z / r, -1.0, 1.0)) * power;
        dir = vec3<f32>(sin(th) * cos(phi), sin(th) * sin(phi), cos(th));
    }
    let rp = pow(r, power - 1.0);
    return State(rp * r * dir + c * cw, rp * power * s.dr + abs(cw));
}

// Quadratic triplex formulas: dr uses the degree-2 estimate 2 |z| dr + |c weight|.
fn quadratic(s: State, n: vec3<f32>, c: vec3<f32>, cw: f32) -> State {
    return State(n + c * cw, 2.0 * length(s.z) * s.dr + abs(cw));
}

fn f_quick_dudley(s: State, c: vec3<f32>, cw: f32) -> State {
    let z = s.z;
    return quadratic(s, vec3<f32>(z.x * z.x - 2.0 * z.y * z.z, z.z * z.z + 2.0 * z.x * z.y, z.y * z.y + 2.0 * z.x * z.z), c, cw);
}

fn f_makin3d(s: State, c: vec3<f32>, cw: f32) -> State {
    let z = s.z;
    return quadratic(s, vec3<f32>(z.x * z.x + 2.0 * z.y * z.z, -z.y * z.y - 2.0 * z.x * z.z, -z.z * z.z + 2.0 * z.x * z.y), c, cw);
}

fn f_pine_tree(s: State, c: vec3<f32>, cw: f32) -> State {
    let z = s.z;
    let yz2 = z.y * z.y + z.z * z.z;
    let t = 2.0 * z.x / sqrt(max(yz2, 1e-12));
    let n = vec3<f32>(z.x * z.x - yz2, 2.0 * t * z.y * z.z, t * (z.y * z.y - z.z * z.z));
    return quadratic(s, n, c, cw);
}

fn f_mandelbox(s: State, c: vec3<f32>, scale: f32, min_r: f32, fixed_r: f32, limit: f32, cw: f32) -> State {
    let folded = f_sphere_fold(f_box_fold(s, limit), min_r, fixed_r);
    return f_add_c(f_scale(folded, scale), c, cw);
}

fn f_amazing_surf(s: State, c: vec3<f32>, scale: f32, min_r: f32, fixed_r: f32, limit: f32, cw: f32) -> State {
    var z = s.z;
    let xy = clamp(z.xy, vec2<f32>(-limit), vec2<f32>(limit)) * 2.0 - z.xy;
    z = vec3<f32>(xy, z.z);
    let folded = f_sphere_fold(State(z, s.dr), min_r, fixed_r);
    return f_add_c(f_scale(folded, scale), c, cw);
}

fn f_bristorbrot(s: State, c: vec3<f32>, cw: f32) -> State {
    let z = s.z;
    let n = vec3<f32>(z.x * z.x - z.y * z.y - z.z * z.z, z.y * (2.0 * z.x - z.z), z.z * (2.0 * z.x + z.y));
    return State(n + c * cw, 2.0 * length(z) * s.dr + abs(cw));
}

fn f_kaliset(s: State, c: vec3<f32>, scale: f32, offset: vec3<f32>, cw: f32) -> State {
    let r2 = max(dot(s.z, s.z), 1e-6);
    let k = scale / r2;
    return State(abs(s.z) * k - offset + c * cw, s.dr * abs(k) + abs(cw));
}

fn f_menger(s: State, scale: f32, offset: vec3<f32>) -> State {
    var z = f_octa_fold(s).z;
    let o = offset * (scale - 1.0);
    z = z * scale - o;
    if (z.z < -0.5 * o.z) { z.z += o.z; }
    return State(z, s.dr * scale);
}

fn f_sierpinski(s: State, scale: f32, offset: f32) -> State {
    let z = f_tetra_fold(s).z;
    return State(z * scale - vec3<f32>(offset * (scale - 1.0)), s.dr * scale);
}

fn f_kifs(s: State, scale: f32, offset: vec3<f32>, angle: f32) -> State {
    let folded = f_rotate(f_octa_fold(s), vec3<f32>(0.0, 0.0, angle));
    return f_ifs_scale(folded, scale, offset);
}

// ---- Folds -----------------------------------------------------------------------

fn f_box_fold(s: State, limit: f32) -> State {
    return State(clamp(s.z, vec3<f32>(-limit), vec3<f32>(limit)) * 2.0 - s.z, s.dr);
}

// Scale factor of a sphere fold at squared radius r2 (also used by Cylinder Fold).
fn sphere_fold_k(r2: f32, min_r: f32, fixed_r: f32) -> f32 {
    let min2 = max(min_r * min_r, 1e-6);
    let fix2 = fixed_r * fixed_r;
    var k = 1.0;
    if (r2 < min2) {
        k = fix2 / min2;
    } else if (r2 < fix2) {
        k = fix2 / max(r2, 1e-12);
    }
    return k;
}

fn f_sphere_fold(s: State, min_r: f32, fixed_r: f32) -> State {
    let k = sphere_fold_k(dot(s.z, s.z), min_r, fixed_r);
    return State(s.z * k, s.dr * k);
}

fn f_abs_fold(s: State, mx: f32, my: f32, mz: f32) -> State {
    return State(mix(s.z, abs(s.z), vec3<f32>(mx, my, mz)), s.dr);
}

// abs, then sort so x >= y >= z.
fn f_octa_fold(s: State) -> State {
    var z = abs(s.z);
    if (z.x < z.y) { z = z.yxz; }
    if (z.x < z.z) { z = z.zyx; }
    if (z.y < z.z) { z = z.xzy; }
    return State(z, s.dr);
}

fn f_tetra_fold(s: State) -> State {
    var z = s.z;
    if (z.x + z.y < 0.0) { z = vec3<f32>(-z.y, -z.x, z.z); }
    if (z.x + z.z < 0.0) { z = vec3<f32>(-z.z, z.y, -z.x); }
    if (z.y + z.z < 0.0) { z = vec3<f32>(z.x, -z.z, -z.y); }
    return State(z, s.dr);
}

// Knighty's polyhedral fold. kind: 0 tetrahedral, 1 octahedral, 2 icosahedral
// (symmetry order n = 3, 4, 5). Repeated mirrors across the xy planes and one
// slanted plane `nc` map space into the group's fundamental domain.
fn poly_n(kind: f32) -> i32 {
    return 3 + i32(kind);
}

fn f_poly_fold(s: State, kind: f32) -> State {
    let n = poly_n(kind);
    let cospin = cos(3.14159265 / f32(n));
    let nc = vec3<f32>(-0.5, -cospin, sqrt(0.75 - cospin * cospin));
    var z = s.z;
    for (var k = 0; k < n; k++) {
        z = vec3<f32>(abs(z.xy), z.z);
        z -= 2.0 * min(0.0, dot(z, nc)) * nc;
    }
    return State(z, s.dr);
}

// Fold, rotate about z, then scale towards a point of the fundamental domain chosen by
// `weights` over its vertex (x), edge midpoint (y) and face center (z) directions.
fn f_poly_ifs(s: State, kind: f32, scale: f32, weights: vec3<f32>, angle: f32) -> State {
    let n = poly_n(kind);
    let cospin = cos(3.14159265 / f32(n));
    let sc = sqrt(0.75 - cospin * cospin);
    let pab = vec3<f32>(0.0, 0.0, 1.0);
    let pbc = normalize(vec3<f32>(sc, 0.0, 0.5));
    let pca = normalize(vec3<f32>(0.0, sc, cospin));
    let d = weights.x * pab + weights.y * pbc + weights.z * pca;
    let center = d / max(length(d), 1e-6);
    let folded = f_rotate(f_poly_fold(s, kind), vec3<f32>(0.0, 0.0, angle));
    return f_ifs_scale(folded, scale, center);
}

// Knighty's pseudo-Kleinian step: box fold, then inversion clamped so it only ever
// expands (k >= 1). Pair with the 'kleinian' distance mode.
fn f_pseudo_kleinian(s: State, size: vec3<f32>, inversion: f32) -> State {
    let z = 2.0 * clamp(s.z, -size, size) - s.z;
    let k = max(inversion / max(dot(z, z), 1e-12), 1.0);
    return State(z * k, s.dr * k);
}

// Periodic fold into cells of `half` half-size: z -> half * (2 fract(z / 2half + 0.5) - 1).
fn f_fract_fold(s: State, half: vec3<f32>) -> State {
    let h = max(abs(half), vec3<f32>(1e-4));
    return State(h * (2.0 * fract(0.5 * s.z / h + 0.5) - 1.0), s.dr);
}

// iq's Apollonian gasket step: periodic fold, then sphere inversion (k = s / r^2).
// Pair with the 'planar' distance mode.
fn f_apollonian(s: State, inversion: f32, cell: vec3<f32>) -> State {
    let z = f_fract_fold(s, cell).z;
    let k = inversion / max(dot(z, z), 1e-12);
    return State(z * k, s.dr * k);
}

fn f_plane_fold(s: State, normal: vec3<f32>, offset: f32) -> State {
    let n = normal / max(length(normal), 1e-6);
    let t = dot(s.z, n) - offset;
    return State(s.z - 2.0 * min(t, 0.0) * n, s.dr);
}

// ---- Transforms ------------------------------------------------------------------

fn f_scale(s: State, k: f32) -> State {
    return State(s.z * k, s.dr * abs(k));
}

fn f_ifs_scale(s: State, k: f32, center: vec3<f32>) -> State {
    return State(s.z * k - center * (k - 1.0), s.dr * abs(k));
}

fn f_rotate(s: State, angles: vec3<f32>) -> State {
    let a = radians(angles);
    let cs = cos(a);
    let sn = sin(a);
    var z = s.z;
    z = vec3<f32>(z.x, cs.x * z.y - sn.x * z.z, sn.x * z.y + cs.x * z.z);
    z = vec3<f32>(cs.y * z.x + sn.y * z.z, z.y, -sn.y * z.x + cs.y * z.z);
    z = vec3<f32>(cs.z * z.x - sn.z * z.y, sn.z * z.x + cs.z * z.y, z.z);
    return State(z, s.dr);
}

fn f_translate(s: State, offset: vec3<f32>) -> State {
    return State(s.z + offset, s.dr);
}

fn f_twist(s: State, amount: f32) -> State {
    let a = amount * s.z.y;
    let cs = cos(a);
    let sn = sin(a);
    return State(vec3<f32>(cs * s.z.x - sn * s.z.z, s.z.y, sn * s.z.x + cs * s.z.z), s.dr);
}

fn f_add_c(s: State, c: vec3<f32>, w: f32) -> State {
    return State(s.z + c * w, s.dr + abs(w));
}

fn f_sphere_invert(s: State, radius: f32) -> State {
    let k = radius * radius / max(dot(s.z, s.z), 1e-12);
    return State(s.z * k, s.dr * k);
}

// ---- Combine ---------------------------------------------------------------------

fn f_mix(a: State, b: State, t: f32) -> State {
    return State(mix(a.z, b.z, t), mix(a.dr, b.dr, t));
}

// Alternate node: true while iteration `i` is in the A part of the (na + nb) cycle.
fn alt_is_a(i: i32, na: f32, nb: f32) -> bool {
    let a = max(round(na), 0.0);
    let period = max(a + max(round(nb), 0.0), 1.0);
    let k = f32(i) - floor(f32(i) / period) * period;
    return k < a;
}

// ---- Math & color ----------------------------------------------------------------

fn safe_div(a: f32, b: f32) -> f32 {
    return select(0.0, a / b, b != 0.0);
}

fn map_range(v: f32, a0: f32, a1: f32, b0: f32, b1: f32, clamped: bool) -> f32 {
    var t = safe_div(v - a0, a1 - a0);
    if (clamped) { t = clamp(t, 0.0, 1.0); }
    return mix(b0, b1, t);
}

// iq's cosine palette: brightness + contrast * cos(2 pi (frequency * t + phase)).
fn cosine_palette(t: f32, a: vec3<f32>, b: vec3<f32>, f: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.2831853 * (f * t + p));
}

fn color_ramp(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, mid: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    if (x < mid) { return mix(a, b, x / mid); }
    return mix(b, c, (x - mid) / (1.0 - mid));
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = clamp(abs(fract(h + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return v * mix(vec3<f32>(1.0), k, s);
}

// ---- Drivers (time is seconds of animation time, see `time_s`) --------------------

fn wave(kind: i32, x: f32) -> f32 {
    let f = fract(x);
    switch kind {
        case 1: { return 1.0 - 4.0 * abs(f - 0.5); }      // triangle
        case 2: { return 2.0 * f - 1.0; }                 // sawtooth
        case 3: { return select(-1.0, 1.0, f < 0.5); }    // square
        default: { return sin(6.2831853 * f); }           // sine
    }
}

fn oscillator(kind: i32, freq: f32, amp: f32, center: f32, phase: f32, t: f32) -> f32 {
    return center + amp * wave(kind, t * freq + phase);
}

// mode: 0 once, 1 loop, 2 ping-pong; easing: 0 linear, 1 smooth.
fn animate(a: f32, b: f32, duration: f32, start: f32, mode: i32, easing: i32, t: f32) -> f32 {
    var x = max(t - start, 0.0) / max(duration, 1e-4);
    if (mode == 0) {
        x = min(x, 1.0);
    } else if (mode == 1) {
        x = fract(x);
    } else {
        x = 1.0 - abs(1.0 - 2.0 * fract(x * 0.5));
    }
    if (easing == 1) { x = x * x * (3.0 - 2.0 * x); }
    return mix(a, b, x);
}

// ---- Domain ----------------------------------------------------------------------

// Cell index of `p` in a grid of `size` cells; `limit` > 0 caps cells per axis to
// -limit..limit (0 = infinite).
fn repeat_cell(p: vec3<f32>, size: vec3<f32>, limit: vec3<f32>) -> vec3<f32> {
    let s = max(abs(size), vec3<f32>(1e-4));
    let lim = select(vec3<f32>(1e9), round(abs(limit)), abs(limit) >= vec3<f32>(0.5));
    return clamp(round(p / s), -lim, lim);
}

// Position within the cell, centered on the cell's origin. Mirroring flips every other
// cell so neighbours meet seamlessly.
fn repeat_pos(p: vec3<f32>, size: vec3<f32>, limit: vec3<f32>, mirror: bool) -> vec3<f32> {
    let s = max(abs(size), vec3<f32>(1e-4));
    let id = repeat_cell(p, size, limit);
    var q = p - s * id;
    if (mirror) {
        let odd = abs(id) - 2.0 * floor(abs(id) * 0.5);
        q = q * (1.0 - 2.0 * odd);
    }
    return q;
}

fn rotate_vec(v: vec3<f32>, angles: vec3<f32>) -> vec3<f32> {
    return f_rotate(State(v, 1.0), angles).z;
}

// ---- More formulas & folds -------------------------------------------------------

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Mandelbulb3D's Xenodreambuie: the trig Mandelbulb with both spherical angles offset.
fn f_xenodreambuie(s: State, c: vec3<f32>, power: f32, alpha: f32, beta: f32, cw: f32) -> State {
    // Fast path for power 8 and 2, as in f_mandelbulb: w = r^p dir(p theta, p phi) from
    // the polynomial, then the offsets as rotations. alpha turns w within its meridian
    // plane, whose direction u = (cos p phi, sin p phi) is (x, y) normalized and
    // squared as a complex number; beta turns it about z.
    if (power == 8.0 || power == 2.0) {
        let z = s.z;
        let w = select(bulb_pow2(z), bulb_pow8(z), power == 8.0);
        let rho2 = dot(z.xy, z.xy);
        var u = select(vec2<f32>(1.0, 0.0), z.xy * inverseSqrt(rho2), rho2 > 1e-24);
        u = cmul(u, u);
        if (power == 8.0) {
            u = cmul(u, u);
            u = cmul(u, u);
        }
        let a = radians(alpha);
        let ca = cos(a);
        let sa = sin(a);
        let sr = dot(w.xy, u); // r^p sin(p theta)
        let m = cmul((sr * ca + w.z * sa) * u, vec2<f32>(cos(radians(beta)), sin(radians(beta))));
        let r2 = dot(z, z);
        let rp = select(sqrt(r2), r2 * r2 * r2 * sqrt(r2), power == 8.0); // r^(power - 1)
        return State(vec3<f32>(m, w.z * ca - sr * sa) + c * cw, rp * power * s.dr + abs(cw));
    }
    let r = max(length(s.z), 1e-6);
    let th = acos(clamp(s.z.z / r, -1.0, 1.0)) * power + radians(alpha);
    let phi = atan2(s.z.y, s.z.x) * power + radians(beta);
    let rp = pow(r, power - 1.0);
    let dir = vec3<f32>(sin(th) * cos(phi), sin(th) * sin(phi), cos(th));
    return State(rp * r * dir + c * cw, rp * power * s.dr + abs(cw));
}

// Rounds half up (WGSL's round() is half-to-even; the CPU mirror rounds the same way).
fn round_up(x: f32) -> f32 {
    return floor(x + 0.5);
}

// Rotates coordinates so `axis` (0 x, 1 y, 2 z) becomes the third one, and back.
fn to_axis(p: vec3<f32>, axis: i32) -> vec3<f32> {
    if (axis == 0) { return p.yzx; }
    if (axis == 1) { return p.zxy; }
    return p;
}

fn from_axis(q: vec3<f32>, axis: i32) -> vec3<f32> {
    if (axis == 0) { return q.zxy; }
    if (axis == 1) { return q.yzx; }
    return q;
}

// Folds the angle around `axis` into the sector centered on angle 0 (one of `n`);
// mirroring also reflects it into the sector's upper half. Distance-preserving.
fn polar_fold(p: vec3<f32>, n: f32, axis: f32, mirror: f32) -> vec3<f32> {
    let ax = i32(axis);
    let q = to_axis(p, ax);
    let sector = 6.2831853 / max(round_up(n), 1.0);
    var a = atan2(q.y, q.x);
    a -= sector * round_up(a / sector);
    if (mirror > 0.5) { a = abs(a); }
    let r = length(q.xy);
    return from_axis(vec3<f32>(r * cos(a), r * sin(a), q.z), ax);
}

// Which of the `n` sectors around `axis` the point is in, 0..n-1.
fn polar_sector(p: vec3<f32>, n: f32, axis: f32) -> f32 {
    let q = to_axis(p, i32(axis));
    let k = max(round_up(n), 1.0);
    let id = round_up(atan2(q.y, q.x) / (6.2831853 / k));
    return id - k * floor(id / k);
}

fn f_polar_fold(s: State, n: f32, axis: f32, mirror: f32) -> State {
    return State(polar_fold(s.z, n, axis, mirror), s.dr);
}

// sqrt(z² + k²) is a 1-Lipschitz abs with a rounded crease, so dr is unchanged.
fn f_smooth_abs_fold(s: State, k: f32) -> State {
    return State(sqrt(s.z * s.z + vec3<f32>(k * k)), s.dr);
}

fn f_cylinder_fold(s: State, min_r: f32, fixed_r: f32, axis: f32) -> State {
    let q = to_axis(s.z, i32(axis));
    let k = sphere_fold_k(dot(q.xy, q.xy), min_r, fixed_r);
    return State(s.z * k, s.dr * k);
}

// ---- Textures --------------------------------------------------------------------
// Integer hashing keeps them bit-identical with src/graph/eval.rs, so textures used in
// the domain or iteration graph give the camera's CPU distance the same shape.

// PCG output permutation (Jarzynski & Olano, "Hash Functions for GPU Rendering").
fn tex_hash(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn tex_hash3(c: vec3<i32>, seed: u32) -> u32 {
    return tex_hash(bitcast<u32>(c.x) ^ tex_hash(bitcast<u32>(c.y) ^ tex_hash(bitcast<u32>(c.z) ^ seed)));
}

// Three numbers in 0..1 from the hash's bits.
fn tex_unit3(h: u32) -> vec3<f32> {
    return vec3<f32>(vec3<u32>(h, h >> 10u, h >> 20u) & vec3<u32>(1023u)) / 1023.0;
}

fn tex_seed(seed: f32) -> u32 {
    return u32(max(round_up(seed), 0.0));
}

// Scales raw gradient noise to about -1..1 (measured; see the eval.rs tests).
const NOISE_GAIN: f32 = 1.6;

// Perlin-style gradient noise with quintic interpolation, about -1..1.
fn gradient_noise(p: vec3<f32>, seed: u32) -> f32 {
    let cell = floor(p);
    let i = vec3<i32>(cell);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var n = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let corner = vec3<f32>(o);
        let g = tex_unit3(tex_hash3(i + o, seed)) * 2.0 - 1.0;
        let w = mix(1.0 - u, u, corner);
        n += w.x * w.y * w.z * dot(g, f - corner);
    }
    return n * NOISE_GAIN;
}

// Fractal Brownian motion: `detail` octaves (the last one fractional) of gradient
// noise, each at twice the frequency and `roughness` times the amplitude. Octaves add
// in variance, so dividing by sqrt(sum amp²) keeps the contrast the same at any detail.
fn fbm(p: vec3<f32>, detail: f32, roughness: f32, seed: u32) -> f32 {
    let octaves = clamp(detail, 0.0, 8.0);
    var sum = 0.0;
    var norm = 0.0;
    var amp = 1.0;
    var q = p;
    for (var k = 0; k < 9; k++) {
        let w = clamp(octaves + 1.0 - f32(k), 0.0, 1.0);
        if (w <= 0.0) { break; }
        sum += amp * w * gradient_noise(q, seed + u32(k));
        norm += (amp * w) * (amp * w);
        amp *= roughness;
        q *= 2.0;
    }
    return sum / sqrt(max(norm, 1e-12));
}

// Noise Texture lookup point: scaled, then warped by noise when distorted.
fn noise_point(p: vec3<f32>, scale: f32, distortion: f32, seed: f32) -> vec3<f32> {
    let q = p * scale;
    if (distortion == 0.0) { return q; }
    let s = tex_seed(seed) + 101u;
    return q + distortion * vec3<f32>(gradient_noise(q, s), gradient_noise(q, s + 1u), gradient_noise(q, s + 2u));
}

// One Noise Texture channel (0 = Fac; 1, 2 = the extra color channels), about 0..1.
fn noise_channel(q: vec3<f32>, detail: f32, roughness: f32, seed: f32, channel: u32) -> f32 {
    return 0.5 + 0.5 * fbm(q, detail, roughness, tex_seed(seed) + 16u * channel);
}

struct Voronoi {
    f1: f32,         // distance to the nearest cell point
    f2: f32,         // distance to the second nearest
    cell: vec3<i32>, // the nearest point's cell
}

// Worley noise over the 3x3x3 neighbouring cells, one jittered point per cell.
fn voronoi(p: vec3<f32>, randomness: f32, seed: u32) -> Voronoi {
    let base = floor(p);
    let i = vec3<i32>(base);
    let f = p - base;
    var f1 = 1e9;
    var f2 = 1e9;
    var best = i;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec3<i32>(x, y, z);
                let d = length(vec3<f32>(o) + tex_unit3(tex_hash3(i + o, seed)) * randomness - f);
                if (d < f1) {
                    f2 = f1;
                    f1 = d;
                    best = i + o;
                } else if (d < f2) {
                    f2 = d;
                }
            }
        }
    }
    return Voronoi(f1, f2, best);
}

fn voronoi_color(v: Voronoi, seed: f32) -> vec3<f32> {
    return tex_unit3(tex_hash3(v.cell, tex_seed(seed) ^ 0x9e3779b9u));
}

// Hash of the exact float bits (-0 counts as 0), for White Noise.
fn white_noise_hash(v: vec3<f32>, seed: f32) -> u32 {
    let b = select(bitcast<vec3<u32>>(v), vec3<u32>(0u), v == vec3<f32>(0.0));
    return tex_hash3(bitcast<vec3<i32>>(b), tex_seed(seed));
}

fn white_noise_value(h: u32) -> f32 {
    return f32(h >> 8u) / 16777215.0;
}

// kind: bands along x / y / z, rings, or a spiral around z; profile as `wave`.
fn wave_texture(p: vec3<f32>, kind: f32, profile: f32, scale: f32, distortion: f32, phase: f32) -> f32 {
    let q = p * scale;
    var x: f32;
    switch i32(kind) {
        case 0: { x = q.x; }
        case 1: { x = q.y; }
        case 2: { x = q.z; }
        case 3: { x = length(q); }
        default: { x = length(q.xy) + atan2(q.y, q.x) / 6.2831853; }
    }
    if (distortion != 0.0) { x += distortion * fbm(q, 2.0, 0.5, 7u); }
    return 0.5 + 0.5 * wave(i32(profile), x + phase);
}

// ---- More color ------------------------------------------------------------------

// Tanner Helland's fit of the Planckian locus (sRGB), converted to linear.
fn blackbody(kelvin: f32) -> vec3<f32> {
    let t = clamp(kelvin, 800.0, 40000.0) / 100.0;
    var c: vec3<f32>;
    if (t <= 66.0) {
        let b = select(138.5177312231 * log(max(t - 10.0, 1e-3)) - 305.0447927307, 0.0, t <= 19.0);
        c = vec3<f32>(255.0, 99.4708025861 * log(t) - 161.1195681661, b);
    } else {
        c = vec3<f32>(329.698727446 * pow(t - 60.0, -0.1332047592), 288.1221695283 * pow(t - 60.0, -0.0755148492), 255.0);
    }
    return pow(clamp(c / 255.0, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

// Sam Hocevar's branchless RGB -> HSV (hue 0..1).
fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = select(vec4<f32>(c.gb, k.xy), vec4<f32>(c.bg, k.wz), c.g < c.b);
    let q = select(vec4<f32>(c.r, p.yzx), vec4<f32>(p.xyw, c.r), c.r < p.x);
    let d = q.x - min(q.w, q.y);
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}

fn hue_saturation(c: vec3<f32>, shift: f32, sat: f32, val: f32) -> vec3<f32> {
    let hsv = rgb_to_hsv(max(c, vec3<f32>(0.0)));
    return hsv_to_rgb(hsv.x + shift, clamp(hsv.y * sat, 0.0, 1.0), hsv.z * max(val, 0.0));
}

// Blend node: `mode` indexes graph::BLEND_MODES; `fac` fades from A to the result.
// Colors may be HDR (above 1): Screen becomes additive there, and the modes defined
// only for 0..1 (overlay, soft light, dodge) clamp the B layer.
fn blend(mode: i32, fac: f32, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let one = vec3<f32>(1.0);
    let bc = clamp(b, vec3<f32>(0.0), one);
    var r = b;
    switch mode {
        case 1: { r = a + b; }
        case 2: { r = a - b; }
        case 3: { r = a * b; }
        case 4: { r = a + b - min(max(a, vec3<f32>(0.0)), one) * bc; }
        case 5: { r = select(one - 2.0 * (one - a) * (one - bc), 2.0 * a * bc, a < vec3<f32>(0.5)); }
        case 6: { r = abs(a - b); }
        case 7: { r = max(a, b); }
        case 8: { r = min(a, b); }
        case 9: { r = (one - 2.0 * bc) * a * a + 2.0 * bc * a; }
        case 10: { r = a / max(b, vec3<f32>(1e-4)); }
        case 11: { r = a / max(one - bc, vec3<f32>(1e-3)); }
        default: {}
    }
    return mix(a, r, clamp(fac, 0.0, 1.0));
}
