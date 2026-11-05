// Distance-estimated raymarcher for node-graph fractals.
//
// This file is a template: `//@GRAPH` is replaced at runtime with nodes.wgsl,
// world.wgsl and the WGSL generated from the material and world graphs (see
// src/graph/codegen.rs).
//
// Output: linear scene color, alpha = distance to the hit (SKY_DEPTH for the sky, -1
// for debug views, which the compositor shows without tone mapping).
//
// A sample is rendered in passes (see renderer.rs):
// 1. fs_coarse: cone pre-pass at 1/4 resolution (how far each block's rays can skip);
// 2. fs_gbuffer: the march, writing the hit distance and normal (G-buffer). Kept free of
//    shading code: fewer registers make the march loop faster;
// 3. fs_light: soft shadows and ambient occlusion at half resolution, from one pixel of
//    each 2x2 block, a different one every frame (accumulation / TAA fill in the rest);
// 4. fs_shade: material, lighting and reflections at full resolution, with shadows and
//    AO upsampled by depth / normal similarity.
//
// LOD: the hit threshold grows linearly with ray distance (`t * pixel_angle * detail`),
// i.e. we stop as soon as the surface is closer than the pixel's cone footprint.
// Distant geometry therefore costs fewer steps and never aliases below pixel size.

struct Params {
    cam_pos: vec4<f32>,    // xyz, w = tan(fov / 2)
    cam_fwd: vec4<f32>,    // xyz, w = animation time (s)
    cam_right: vec4<f32>,  // xyz, w = distance mode (0 log, 1 linear, 2 kleinian, 3 planar)
    cam_up: vec4<f32>,     // xyz, w = detail multiplier (LOD)
    resolution: vec4<f32>, // xy = render size, z = max steps, w = fractal iterations
    base_color: vec4<f32>, // rgb, w = ambient occlusion strength
    misc: vec4<f32>,       // x = soft shadows on, y = bounding radius (< 0: unbounded, -view distance), z = debug view (see DebugView), w = bailout
    shadow: vec4<f32>,     // x = max steps, y = fractal iterations, z = softness (penumbra k), w = fractal size (bound radius)
    jitter: vec4<f32>,     // xy = sub-pixel offset in pixels (for accumulation AA), z = sample index, w = unused
    shade: vec4<f32>,      // x = ray-traced reflections on, y = reflection max steps, zw = unused
}

@group(0) @binding(0) var<uniform> u: Params;
// Formula graph parameters, addressed by generated code (see codegen::MAX_PARAM_VEC4S).
@group(0) @binding(1) var<uniform> gp: array<vec4<f32>, 256>;
// World graph parameters and image files (see world.rs).
@group(2) @binding(0) var<uniform> wp: array<vec4<f32>, 256>;
@group(2) @binding(1) var world_img0: texture_2d<f32>;
@group(2) @binding(2) var world_img1: texture_2d<f32>;
@group(2) @binding(3) var world_samp: sampler;
// Lighting from the world: sun, fog, ambient SH and the environment map.
@group(3) @binding(0) var<uniform> wl: WorldLight;
@group(3) @binding(1) var env_tex: texture_2d<f32>;

const SKY_DEPTH: f32 = 60000.0;
// Radiance of the visible sun disc relative to the sun's light (a real sun is ~10^5x
// brighter than its light suggests per pixel; this keeps it a bright, bloomable disc).
const SUN_DISC_GAIN: f32 = 40.0;

fn world_sun_dir() -> vec3<f32> {
    return wl.sun_dir.xyz;
}

fn world_sun_color() -> vec3<f32> {
    return wl.sun_color.rgb;
}

// Animation time in seconds, for driver nodes (0 when the graph has no drivers).
fn time_s() -> f32 {
    return u.cam_fwd.w;
}

//@GRAPH

const NORMAL_FOOTPRINT: f32 = 1.5;

// Cone pre-pass (fs_coarse): one cone per COARSE_BLOCK x COARSE_BLOCK pixel block
// records how far its rays can safely skip; fs_main starts marching from there.
const COARSE_BLOCK: f32 = 4.0;
@group(1) @binding(0) var coarse_t: texture_2d<f32>;
// G-buffer, 8 bytes a pixel (bandwidth matters here): x = hit distance (f32 bits, full
// precision for deep zooms), y = normal (octahedral, 2 x 16 bit). Misses: x = G_MISS
// (y = march steps as a fraction, for the near-miss glow), G_FOGGED, or G_STEPS (step
// heatmap view, y). See `load_g`.
@group(1) @binding(1) var gbuf: texture_2d<u32>;
// Half resolution: x = shadow, y = ambient occlusion (before its strength).
@group(1) @binding(2) var vis_tex: texture_2d<f32>;

const G_MISS: f32 = -1.0;
const G_FOGGED: f32 = -2.0;
const G_STEPS: f32 = -3.0;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle, no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return o;
}

fn de(p: vec3<f32>) -> f32 {
    return de_n(p, i32(u.resolution.w));
}

// Distance estimate. `iters` lets secondary rays use a cheaper estimate.
// The iteration body is the formula graph, generated into `graph_step`.
fn de_n(p: vec3<f32>, iters: i32) -> f32 {
    let bail2 = u.misc.w * u.misc.w;
    // Domain graph first (e.g. Repeat tiles space), then iterate on that point.
    let q = graph_domain(p);
    var s = State(q, 1.0);
    var r2 = dot(q, q);
    for (var i = 0; i < iters; i++) {
        s = graph_step(s, q, i);
        r2 = dot(s.z, s.z);
        if (r2 > bail2) { break; }
    }
    let r = sqrt(r2);
    let dr = max(abs(s.dr), 1e-12);
    // Linear DE suits fold/IFS formulas, log DE suits power (bulb) formulas; the
    // Kleinian (Knighty) and planar (iq's Apollonian) estimators suit space-filling
    // inversion fractals.
    switch i32(u.cam_right.w) {
        case 1: { return r / dr; }
        case 2: {
            let rxy = length(s.z.xy);
            return max(rxy - 0.92784, abs(rxy * s.z.z) / max(r, 1e-12)) / dr;
        }
        case 3: { return 0.25 * abs(s.z.y) / dr; }
        default: { return 0.5 * log(max(r, 1e-12)) * r / dr; }
    }
}

// Re-runs the orbit at a surface hit to collect traps for the color graph. Done once
// per pixel instead of inside every distance evaluation of the march.
fn orbit_data(p: vec3<f32>, n: vec3<f32>, dist: f32, view: vec3<f32>) -> Orbit {
    let bail2 = u.misc.w * u.misc.w;
    let iters = i32(u.resolution.w);
    let q = graph_domain(p);
    var s = State(q, 1.0);
    var trap_r2 = 1e10;
    var trap = vec3<f32>(1e10);
    var count = 0;
    for (var i = 0; i < iters; i++) {
        s = graph_step(s, q, i);
        let r2 = dot(s.z, s.z);
        trap_r2 = min(trap_r2, r2);
        trap = min(trap, abs(s.z));
        count = i + 1;
        if (r2 > bail2) { break; }
    }
    return Orbit(sqrt(trap_r2), trap, f32(count), p, n, dist, view);
}

// Ray interval to march: the bounding sphere, or for unbounded (repeating) scenes the
// view distance from the camera.
fn ray_range(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    var r: vec2<f32>;
    if (u.misc.y < 0.0) {
        r = vec2<f32>(0.0, -u.misc.y);
    } else {
        r = intersect_sphere(ro, rd, u.misc.y);
    }
    // Beyond the fog wall a hit would be indistinguishable from the sky: stop there.
    return vec2<f32>(r.x, min(r.y, fog_limit()));
}

// Distance fog in units of the fractal's size (bound radius), so it looks the same at
// any scale; 2.25 = 1.5^2 keeps the original Mandelbulb look.
fn fog_amount(t: f32) -> f32 {
    let tn = t / u.shadow.w;
    return 1.0 - exp(-tn * tn * 2.25 * wl.misc.x);
}

// Distance at which fog reaches 99.5% (-ln(0.005) = 5.3), i.e. under one 8-bit level.
fn fog_limit() -> f32 {
    if (wl.misc.x <= 0.0) { return 1e30; }
    return u.shadow.w * sqrt(5.3 / (2.25 * wl.misc.x));
}

fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let h = b * b - c;
    if (h < 0.0) { return vec2<f32>(-1.0, -1.0); }
    let s = sqrt(h);
    return vec2<f32>(-b - s, -b + s);
}

// Tetrahedral gradient: 4 DE evaluations instead of 6.
fn calc_normal(p: vec3<f32>, h: f32) -> vec3<f32> {
    let k = vec2<f32>(1.0, -1.0);
    return normalize(
        k.xyy * de(p + k.xyy * h) +
        k.yyx * de(p + k.yyx * h) +
        k.yxy * de(p + k.yxy * h) +
        k.xxx * de(p + k.xxx * h)
    );
}

// Soft shadow towards a directional light.
//
// Cost control:
// - the ray ends where it leaves the bounding sphere, not at a fixed distance;
// - the minimum step grows with distance (`cone`), so the ray never crawls along the
//   surface in tiny steps; the penumbra is blurry anyway, so the precision isn't needed;
// - it uses its own (lower) iteration count and step budget.
// Penumbra uses iq's improved estimator, which stays smooth with few steps.
fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, cone: f32) -> f32 {
    // Unbounded scenes: shadow rays only look a few fractal sizes far.
    let tmax = select(intersect_sphere(ro, rd, u.misc.y).y, 3.0 * u.shadow.w, u.misc.y < 0.0);
    if (tmax <= tmin) { return 1.0; }
    let max_steps = i32(u.shadow.x);
    let iters = i32(u.shadow.y);
    let k = u.shadow.z;

    var res = 1.0;
    var t = tmin;
    var ph = 1e10;
    for (var i = 0; i < max_steps; i++) {
        let h = de_n(ro + rd * t, iters);
        let y = h * h / (2.0 * ph);
        let d = sqrt(max(h * h - y * y, 0.0));
        res = min(res, k * d / max(t - y, 1e-6));
        ph = h;
        t += max(h, tmin + cone * t);
        if (res < 0.004 || t > tmax) { break; }
    }
    // Running out of steps before escaping the bounds almost always means the ray is
    // threading a crevice; treating that as lit causes visible light leaks.
    if (t <= tmax) { res = 0.0; }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

// Ambient occlusion from 5 distance samples along the normal. Unlike the step-count
// shortcut it is continuous, so it doesn't draw contour bands across smooth surfaces.
// `scale` sets the sampling radius; sampling relative to the pixel footprint keeps
// the effect stable at every zoom level.
// Uses full iterations: the cheaper shadow estimate is "fatter" than the real surface,
// so samples land inside it and over-darken everything.
// Compares each sample's distance to its offset as a ratio. The Mandelbulb DE
// underestimates by roughly 2x, so the ratio is rescaled to make open surfaces read
// as unoccluded instead of uniformly grey.
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>, scale: f32) -> f32 {
    let de_underestimate = 2.0;
    var vis = 0.0;
    var wsum = 0.0;
    var w = 1.0;
    for (var i = 1; i <= 5; i++) {
        let h = scale * f32(i * i);
        let d = de(p + n * h);
        vis += clamp(d * de_underestimate / h, 0.0, 1.0) * w;
        wsum += w;
        w *= 0.6;
    }
    return vis / wsum;
}

const PI: f32 = 3.14159265;

// Cook-Torrance GGX specular BRDF (Smith-Schlick geometry, Schlick Fresnel).
fn ggx_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let nl = max(dot(n, l), 1e-4);
    let nv = max(dot(n, v), 1e-4);
    let a = rough * rough;
    let a2 = a * a;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let g = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    return d * g * f / (4.0 * nv * nl);
}

// Schlick Fresnel with roughness, for image-based (sky / reflection) lighting.
fn fresnel_rough(f0: vec3<f32>, nv: f32, rough: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - nv, 5.0);
}

fn hash3(p: vec3<u32>) -> vec3<f32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3<f32>(v) * (1.0 / 4294967296.0);
}

struct Lit {
    color: vec3<f32>,
    // For environment lighting, applied by the caller.
    fe: vec3<f32>,
    rough: f32,
    ao: f32,
    // Debug output.
    shadow: f32,
}

// Shades a surface point with the material graph: direct light, subsurface, ambient
// and emission, plus the Fresnel weight for environment lighting (added by the
// caller). `vis` is the (shadow, AO) from the light pass for primary hits; reflected
// hits (`primary` false) compute AO here and skip shadows.
fn shade_direct(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, t: f32, eps: f32, primary: bool, vis: vec2<f32>) -> Lit {
    let l = wl.sun_dir.xyz;
    let light = wl.sun_color.rgb;

    var orbit = Orbit(0.0, vec3<f32>(0.0), 0.0, p, n, t, v);
    if (NEEDS_ORBIT) { orbit = orbit_data(p, n, t, v); }
    let m = graph_material(orbit);
    let rough = max(m.roughness, 0.03);
    let nv = max(dot(n, v), 1e-4);
    let ndl = dot(n, l);

    var ao_raw = vis.y;
    var shadow = vis.x;
    if (!primary) {
        ao_raw = ambient_occlusion(p, n, max(eps * 4.0, t * 0.01));
        shadow = 1.0;
    }
    let ao = pow(ao_raw, u.base_color.w);

    // Subsurface: light wraps past the terminator by up to `sss_radius`.
    let wrap = m.subsurface * m.sss_radius;
    let wrapped = max((ndl + wrap) / (1.0 + wrap), 0.0);

    let f0 = mix(vec3<f32>(0.08 * m.specular), m.base, m.metallic);
    let kd = 1.0 - m.metallic;

    // Diffuse is Lambert, or for subsurface materials the wrapped term tinted by the
    // subsurface color, with light bleeding into the shadow.
    let lambert = vec3<f32>(max(ndl, 0.0) * shadow);
    let sss_light = m.sss_color * wrapped * mix(shadow, 1.0, 0.6);
    let diffuse = mix(lambert, sss_light, m.subsurface);
    var spec = vec3<f32>(0.0);
    if (ndl > 0.0) {
        spec = ggx_specular(n, v, l, rough, f0) * PI * ndl * shadow;
    }
    // Transmission: thin, open parts glow when backlit (AO is the thickness proxy).
    let back = pow(clamp(dot(v, -l), 0.0, 1.0), 4.0) * m.subsurface * m.sss_radius * ao;
    let transmission = m.sss_color * back * 1.5;

    let ambient = sh_irradiance(wl.sh, n) * wl.misc.y * ao;
    let col = kd * m.base * ((diffuse + transmission) * light + ambient) + spec * light + m.emission;
    return Lit(col, fresnel_rough(f0, nv, rough), rough, ao, shadow * max(ndl, 0.0));
}

// The environment (world without the sun disc) in direction `r`, blurred for rough
// surfaces by sampling a smaller mip.
fn sky_env(r: vec3<f32>, rough: f32) -> vec3<f32> {
    return textureSampleLevel(env_tex, world_samp, dir_to_equirect(r), rough * wl.misc.w).rgb;
}

// What fog fades towards: the environment, slightly blurred.
fn fog_color(rd: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(env_tex, world_samp, dir_to_equirect(rd), 1.0).rgb;
}

// The sun disc (soft edge, limb darkening), only seen directly by camera rays: in
// reflections the specular highlight already is the sun.
fn sun_disc(rd: vec3<f32>) -> vec3<f32> {
    let radius = wl.sun_color.w;
    let c = dot(rd, wl.sun_dir.xyz);
    if (radius <= 0.0 || c <= 0.0) { return vec3<f32>(0.0); }
    let ang = asin(min(length(cross(rd, wl.sun_dir.xyz)), 1.0));
    if (ang > radius * 1.2) { return vec3<f32>(0.0); }
    let edge = 1.0 - smoothstep(radius * 0.85, radius * 1.15, ang);
    let mu = sqrt(max(1.0 - (ang * ang) / (radius * radius), 0.0));
    return wl.sun_color.rgb * SUN_DISC_GAIN * edge * (0.45 + 0.55 * mu);
}

// Background seen by a camera ray that misses: the world graph, plus the sun.
fn world_background(rd: vec3<f32>) -> vec3<f32> {
    return graph_world(rd) + sun_disc(rd);
}

// Full shading of a primary hit: shade_direct plus environment specular, which is a
// ray-traced reflection of the fractal itself for smooth enough surfaces.
fn shade_surface(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, t: f32, eps: f32, rnd: vec3<f32>, vis: vec2<f32>) -> Lit {
    var lit = shade_direct(p, n, v, t, eps, true, vis);
    var r = reflect(-v, n);
    var env = sky_env(r, lit.rough);
    let fe = lit.fe;
    // MAY_REFLECT (from codegen) compiles this out when no material can be smooth.
    if (MAY_REFLECT && u.shade.x > 0.5 && lit.rough < 0.4 && max(fe.r, max(fe.g, fe.b)) > 0.02) {
        // Glossy: jitter the mirror direction per accumulation sample, so rough
        // reflections converge to a blur while the camera is still.
        if (u.jitter.z > 0.0) {
            let jit = normalize(rnd * 2.0 - 1.0 + vec3<f32>(1e-4));
            r = normalize(r + jit * lit.rough * lit.rough * 1.5);
            if (dot(r, n) < 0.0) { r = reflect(r, n); }
        }
        env = trace_reflection(p + n * eps * 4.0, r, t, eps);
    }
    lit.color += fe * env * mix(lit.ao, 1.0, 0.5);
    return lit;
}

// World-space direction of the camera ray through render-target pixel position `pix`.
fn camera_ray(pix: vec2<f32>) -> vec3<f32> {
    let res = u.resolution.xy;
    let ndc = (pix / res * 2.0 - 1.0) * vec2<f32>(res.x / res.y, -1.0);
    return normalize(u.cam_fwd.xyz + (ndc.x * u.cam_right.xyz + ndc.y * u.cam_up.xyz) * u.cam_pos.w);
}

// Cone-marching pre-pass, rendered at 1/COARSE_BLOCK resolution. Marches one cone
// that contains every ray of the block (half its diagonal, plus a pixel for the
// accumulation jitter and a margin), stopping as soon as the cone may touch the
// surface. Outputs the last distance at which the whole cone was provably empty.
@fragment
fn fs_coarse(in: VsOut) -> @location(0) vec4<f32> {
    let ro = u.cam_pos.xyz;
    let rd = camera_ray(in.pos.xy * COARSE_BLOCK);
    let bounds = ray_range(ro, rd);
    if (bounds.y <= 0.0) { return vec4<f32>(0.0); } // no info: neighbours may still hit
    // Cone radius per unit distance; doubled so consecutive spheres overlap the cone.
    let k = 2.0 * (COARSE_BLOCK * 0.7072 + 1.5) * (2.0 * u.cam_pos.w / u.resolution.y);
    var t = max(bounds.x, 0.0);
    var safe = t;
    let max_steps = i32(u.resolution.z);
    for (var i = 0; i < max_steps; i++) {
        let d = de(ro + rd * t);
        if (d < k * t) { break; }
        safe = t;
        t += d;
        if (t > bounds.y) { safe = bounds.y; break; }
    }
    return vec4<f32>(safe, 0.0, 0.0, 0.0);
}

// One-bounce reflection: marches from a surface point and shades what it hits
// without shadows or further reflections. `t0` is the primary distance (for the
// pixel-footprint hit threshold and fog).
fn trace_reflection(ro: vec3<f32>, rd: vec3<f32>, t0: f32, eps0: f32) -> vec3<f32> {
    let range = ray_range(ro, rd);
    let pixel_angle = 2.0 * u.cam_pos.w / u.resolution.y * u.cam_up.w;
    var t = 0.0;
    let max_steps = i32(u.shade.y);
    for (var i = 0; i < max_steps; i++) {
        let d = de(ro + rd * t);
        let eps = max((t0 + t) * pixel_angle * 2.0, eps0);
        if (d < eps) {
            let p = ro + rd * t;
            let n = calc_normal(p, eps * NORMAL_FOOTPRINT);
            let lit = shade_direct(p, n, -rd, t0 + t, eps, false, vec2<f32>(1.0));
            let c = lit.color + lit.fe * sky_env(reflect(rd, n), lit.rough) * mix(lit.ao, 1.0, 0.5);
            return mix(c, fog_color(rd), fog_amount(t0 + t));
        }
        t += d;
        if (t > range.y) { break; }
    }
    return sky_env(rd, 0.0);
}

fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    return select(p, (1.0 - abs(p.yx)) * s, n.z < 0.0);
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}

fn pack_g(x: f32, n: vec3<f32>) -> vec4<u32> {
    return vec4<u32>(bitcast<u32>(x), pack2x16snorm(oct_encode(n)), 0u, 0u);
}

fn pack_g_miss(code: f32, value: f32) -> vec4<u32> {
    return vec4<u32>(bitcast<u32>(code), bitcast<u32>(value), 0u, 0u);
}

// The G-buffer at `p`: (distance, normal) for hits, (code, value, 0, 0) otherwise.
fn load_g(p: vec2<i32>) -> vec4<f32> {
    let raw = textureLoad(gbuf, p, 0);
    let x = bitcast<f32>(raw.x);
    if (x <= 0.0) { return vec4<f32>(x, bitcast<f32>(raw.y), 0.0, 0.0); }
    return vec4<f32>(x, oct_decode(unpack2x16snorm(raw.y)));
}

// Angular size of one pixel; scaled by `detail` to trade quality for speed.
fn pixel_angle() -> f32 {
    return 2.0 * u.cam_pos.w / u.resolution.y * u.cam_up.w;
}

// Which pixel of each 2x2 block the light pass shades this frame (jitter.w counts
// frames), so over four frames every pixel gets its own shadows and AO.
fn vis_offset() -> vec2<i32> {
    let k = u32(u.jitter.w) % 4u;
    return vec2<i32>(i32(k == 1u || k == 2u), i32(k == 1u || k == 3u));
}

@fragment
fn fs_gbuffer(in: VsOut) -> @location(0) vec4<u32> {
    let ro = u.cam_pos.xyz;
    let rd = camera_ray(in.pos.xy + u.jitter.xy);
    let pixel_angle = pixel_angle();
    let max_steps = i32(u.resolution.z);
    let bounds = ray_range(ro, rd);
    var steps = 0;
    var t = 0.0;
    var hit = false;
    var eps = 0.0;
    if (bounds.y > 0.0) {
        // Start where this pixel's coarse cone was still provably empty.
        let skip = textureLoad(coarse_t, vec2<i32>(in.pos.xy / COARSE_BLOCK), 0).r;
        t = max(bounds.x, skip);
        for (var i = 0; i < max_steps; i++) {
            let d = de(ro + rd * t);
            eps = max(t * pixel_angle, 1e-7);
            steps = i;
            if (d < eps) { hit = true; break; }
            t += d;
            if (t > bounds.y) { break; }
        }
    }
    let step_frac = f32(steps) / f32(max_steps);
    if (i32(u.misc.z) == 1) { return pack_g_miss(G_STEPS, step_frac); }
    if (!hit) { return pack_g_miss(G_MISS, step_frac); }
    // Hits deep in the fog look like sky: skip normals, AO, shadows and coloring.
    if (i32(u.misc.z) == 0 && fog_amount(t) > 0.99) { return pack_g_miss(G_FOGGED, 0.0); }
    // Sample the gradient over the pixel's footprint: smaller offsets resolve sub-pixel
    // geometry, which shows up as per-pixel normal noise.
    let n = calc_normal(ro + rd * t, eps * NORMAL_FOOTPRINT);
    return pack_g(t, n);
}

@fragment
fn fs_light(in: VsOut) -> @location(0) vec4<f32> {
    let full = vec2<i32>(u.resolution.xy) - 1;
    let src = clamp(vec2<i32>(in.pos.xy) * 2 + vis_offset(), vec2<i32>(0), full);
    let g = load_g(src);
    if (g.x <= 0.0) { return vec4<f32>(1.0, 1.0, 0.0, 0.0); }
    let t = g.x;
    let n = g.yzw;
    let p = u.cam_pos.xyz + camera_ray(vec2<f32>(src) + 0.5 + u.jitter.xy) * t;
    let eps = max(t * pixel_angle(), 1e-7);
    let ao = ambient_occlusion(p, n, max(eps * 4.0, t * 0.01));
    let l = wl.sun_dir.xyz;
    var shadow = 1.0;
    // Facing well away from the sun, nothing (not even subsurface wrap) is lit.
    if (u.misc.x > 0.5 && dot(n, l) > -0.5) {
        shadow = soft_shadow(p + n * eps * 2.0, l, eps * 2.0, 0.02);
    }
    return vec4<f32>(shadow, ao, 0.0, 0.0);
}

// Shadow and AO for full-resolution pixel `pix` (distance `t`, normal `n`) from the
// half-resolution light pass: its four nearest samples, weighted bilinearly and by how
// similar their surface is (depth, normal), so light doesn't bleed across edges.
fn upsample_vis(pix: vec2<i32>, t: f32, n: vec3<f32>) -> vec2<f32> {
    let o = vis_offset();
    let full = vec2<i32>(u.resolution.xy) - 1;
    let half_max = vec2<i32>(textureDimensions(vis_tex)) - 1;
    let hp = (vec2<f32>(pix - o)) * 0.5;
    let base = vec2<i32>(floor(hp));
    let f = hp - floor(hp);
    var sum = vec2<f32>(0.0);
    var wsum = 0.0;
    var best = vec2<f32>(1.0);
    var best_w = -1.0;
    for (var k = 0; k < 4; k++) {
        let d = vec2<i32>(k & 1, k >> 1);
        let h = clamp(base + d, vec2<i32>(0), half_max);
        let gs = load_g(clamp(h * 2 + o, vec2<i32>(0), full));
        let v = textureLoad(vis_tex, h, 0).xy;
        let bil = select(1.0 - f.x, f.x, d.x == 1) * select(1.0 - f.y, f.y, d.y == 1);
        var similar = 0.0;
        if (gs.x > 0.0) {
            let nd = max(dot(gs.yzw, n), 0.0);
            let nd2 = nd * nd;
            let nd4 = nd2 * nd2;
            similar = exp(-abs(gs.x - t) / (0.03 * t)) * nd4 * nd4;
        }
        let w = (bil + 0.01) * similar;
        sum += v * w;
        wsum += w;
        if (similar > best_w) {
            best_w = similar;
            best = v;
        }
    }
    if (wsum > 1e-4) { return sum / wsum; }
    return best;
}

@fragment
fn fs_shade(in: VsOut) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(in.pos.xy);
    let g = load_g(pix);
    let rd = camera_ray(in.pos.xy + u.jitter.xy);
    if (g.x < G_FOGGED - 0.5) {
        // Debug: step heatmap (blue = cheap, red = expensive).
        let s = g.y;
        return vec4<f32>(s, 0.2 * (1.0 - s), 1.0 - s, -1.0);
    }
    if (g.x <= 0.0) {
        var col = world_background(rd);
        if (g.x > G_FOGGED + 0.5) {
            // Near-miss glow from the march's step count.
            col += sh_irradiance(wl.sh, vec3<f32>(0.0, 1.0, 0.0)) * wl.misc.y * 1.25 * g.y * g.y;
        }
        return vec4<f32>(col, SKY_DEPTH);
    }
    let t = g.x;
    let n = g.yzw;
    let p = u.cam_pos.xyz + rd * t;
    let eps = max(t * pixel_angle(), 1e-7);
    let vis = upsample_vis(pix, t, n);
    let rnd = hash3(vec3<u32>(vec2<u32>(in.pos.xy), u32(u.jitter.z)));
    let lit = shade_surface(p, n, -rd, t, eps, rnd, vis);

    let view = i32(u.misc.z);
    // Debug views write alpha -1 so the compositor shows them without tone mapping.
    if (view == 2) { return vec4<f32>(n * 0.5 + 0.5, -1.0); }
    if (view == 3) { return vec4<f32>(vec3<f32>(lit.shadow), -1.0); }
    if (view == 4) { return vec4<f32>(vec3<f32>(lit.ao), -1.0); }
    // Linear scene light; tone mapping / HDR encoding happens in composite.wgsl.
    return vec4<f32>(mix(lit.color, fog_color(rd), fog_amount(t)), t);
}
