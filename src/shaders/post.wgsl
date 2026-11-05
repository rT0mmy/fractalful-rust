// Processing library: the per-pixel image nodes of the processing graph. Included in
// every processing pass after its header (bindings `pp`, `pu`, `post_samp`, the pass's
// `tex_*` inputs and `pp_at`), see codegen::generate_post.

// Distance written by the fractal pass for pixels that see the sky.
const SKY_DEPTH: f32 = 60000.0;
const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn to_display(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn from_display(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

// Pixel position centered on the frame: x in -aspect..aspect, y in -1..1 (up).
fn screen_centered(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((uv.x * 2.0 - 1.0) * pu.misc.z, 1.0 - uv.y * 2.0);
}

fn centered_to_uv(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.x / pu.misc.z * 0.5 + 0.5, 0.5 - p.y * 0.5);
}

// ---- RGB Curves ------------------------------------------------------------------
// Points are stored from `base` in `pp`: (x, y) x 5 points x 4 curves (C, R, G, B).
// Monotone cubic through the points, linear beyond the ends; mirrors
// graph::editor::curve_eval.

fn curve_eval(base: i32, ch: i32, x: f32) -> f32 {
    var p: array<vec2<f32>, 5>;
    for (var k = 0; k < 5; k++) {
        let o = base + (ch * 5 + k) * 2;
        p[k] = vec2<f32>(pp_at(o), pp_at(o + 1));
    }
    var d: array<f32, 4>;
    for (var k = 0; k < 4; k++) {
        d[k] = (p[k + 1].y - p[k].y) / max(p[k + 1].x - p[k].x, 1e-5);
    }
    var m: array<f32, 5>;
    m[0] = d[0];
    m[4] = d[3];
    for (var k = 1; k < 4; k++) {
        let s = d[k - 1] + d[k];
        m[k] = select(2.0 * d[k - 1] * d[k] / select(s, 1.0, s == 0.0), 0.0, d[k - 1] * d[k] <= 0.0);
    }
    if (x <= p[0].x) { return p[0].y + m[0] * (x - p[0].x); }
    if (x >= p[4].x) { return p[4].y + m[4] * (x - p[4].x); }
    var k = 3;
    for (var i = 0; i < 4; i++) {
        if (x <= p[i + 1].x) {
            k = i;
            break;
        }
    }
    let h = max(p[k + 1].x - p[k].x, 1e-5);
    let t = (x - p[k].x) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * p[k].y + (t3 - 2.0 * t2 + t) * h * m[k]
        + (-2.0 * t3 + 3.0 * t2) * p[k + 1].y + (t3 - t2) * h * m[k + 1];
}

// The combined curve, then each channel's, in display-like (gamma 2.2) space.
fn rgb_curves(c: vec3<f32>, fac: f32, base: i32) -> vec3<f32> {
    let g = to_display(c);
    let o = vec3<f32>(
        curve_eval(base, 1, curve_eval(base, 0, g.r)),
        curve_eval(base, 2, curve_eval(base, 0, g.g)),
        curve_eval(base, 3, curve_eval(base, 0, g.b)),
    );
    return mix(c, from_display(o), clamp(fac, 0.0, 1.0));
}

// ---- Adjustments -----------------------------------------------------------------

// Brightness offsets in display space; contrast stretches around middle grey (0.18).
fn bright_contrast(c: vec3<f32>, brightness: f32, contrast: f32) -> vec3<f32> {
    let x = from_display(to_display(c) + vec3<f32>(brightness));
    return 0.18 * pow(x / 0.18, vec3<f32>(max(1.0 + contrast, 0.0)));
}

// Lift / gamma / gain with grey (0.5) as neutral, in display space.
fn color_balance(c: vec3<f32>, fac: f32, lift: vec3<f32>, gamma: vec3<f32>, gain: vec3<f32>) -> vec3<f32> {
    let g = to_display(c);
    var o = g * gain * 2.0 + (lift - 0.5) * 0.5 * (1.0 - min(g, vec3<f32>(1.0)));
    o = pow(max(o, vec3<f32>(0.0)), 1.0 / exp2((gamma - 0.5) * 2.0));
    return mix(c, from_display(o), clamp(fac, 0.0, 1.0));
}

fn posterize(c: vec3<f32>, steps: f32) -> vec3<f32> {
    let n = max(steps - 1.0, 1.0);
    return from_display(round(to_display(c) * n) / n);
}

// Darkens (or tints with `color`) outside `size`, over `soft`. Roundness 1 is a circle,
// 0 follows the frame's shape.
fn vignette(c: vec3<f32>, uv: vec2<f32>, amount: f32, size: f32, soft: f32, roundness: f32, color: vec3<f32>) -> vec3<f32> {
    let p = screen_centered(uv);
    let q = vec2<f32>(mix(p.x / pu.misc.z, p.x, clamp(roundness, 0.0, 1.0)), p.y);
    let v = smoothstep(size, size + max(soft, 1e-3), length(q)) * clamp(amount, 0.0, 1.0);
    return mix(c, color, v);
}

// Noise added in display space (so it shows in shadows like film grain), new every
// 1/24 s of animation time when animated.
fn film_grain(c: vec3<f32>, uv: vec2<f32>, amount: f32, size: f32, animated: bool, colored: bool) -> vec3<f32> {
    let px = vec2<i32>(floor(uv * pu.size.xy / max(size, 0.5)));
    let frame = select(0, i32(floor(time_s() * 24.0)), animated);
    let n = tex_unit3(tex_hash3(vec3<i32>(px, frame), 0x51ed27u)) - 0.5;
    let grain = select(vec3<f32>(n.x), n, colored);
    return from_display(to_display(c) + grain * amount * 2.0);
}

fn image_texture(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, scale: f32, extension: i32) -> vec4<f32> {
    var q = (uv - 0.5) / max(scale, 1e-3) + 0.5;
    if (extension == 0) {
        q = fract(q);
    } else if (extension == 2 && (any(q < vec2<f32>(0.0)) || any(q > vec2<f32>(1.0)))) {
        return vec4<f32>(0.0);
    }
    return textureSampleLevel(t, s, clamp(q, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

// ---- Effects that sample their input around the pixel ------------------------------

fn lens_sample(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, k: f32) -> vec4<f32> {
    let p = screen_centered(uv);
    let r2 = dot(p, p) / (1.0 + pu.misc.z * pu.misc.z);
    return textureSampleLevel(t, s, centered_to_uv(p * (1.0 - k * r2)), 0.0);
}

// Barrel (+) / pincushion (-) distortion; dispersion spreads the channels apart.
fn lens_distortion(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, distortion: f32, dispersion: f32) -> vec3<f32> {
    let k = distortion * 0.5;
    let d = dispersion * 0.08;
    return vec3<f32>(lens_sample(t, s, uv, k - d).r, lens_sample(t, s, uv, k).g, lens_sample(t, s, uv, k + d).b);
}

fn pixelate(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, size: f32) -> vec3<f32> {
    let cell = max(size, 1.0) * pu.size.zw;
    return textureSampleLevel(t, s, (floor(uv / cell) + 0.5) * cell, 0.0).rgb;
}

fn sharpen(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, amount: f32, radius: f32) -> vec3<f32> {
    let o = pu.size.zw * radius;
    let c = textureSampleLevel(t, s, uv, 0.0).rgb;
    let blur = (textureSampleLevel(t, s, uv + vec2<f32>(o.x, 0.0), 0.0).rgb
        + textureSampleLevel(t, s, uv - vec2<f32>(o.x, 0.0), 0.0).rgb
        + textureSampleLevel(t, s, uv + vec2<f32>(0.0, o.y), 0.0).rgb
        + textureSampleLevel(t, s, uv - vec2<f32>(0.0, o.y), 0.0).rgb) * 0.25;
    return max(c + (c - blur) * amount, vec3<f32>(0.0));
}

// Mirrors the frame into `segments` wedges around the center.
fn kaleidoscope(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, segments: f32, rotation: f32, zoom: f32) -> vec3<f32> {
    let p = screen_centered(uv) / max(zoom, 1e-3);
    let rot = radians(rotation);
    let seg = 2.0 * WPI_POST / max(round(segments), 1.0);
    let a = atan2(p.y, p.x) - rot;
    var b = a - seg * floor(a / seg);
    b = min(b, seg - b);
    let q = vec2<f32>(cos(b + rot), sin(b + rot)) * length(p);
    return textureSampleLevel(t, s, centered_to_uv(q), 0.0).rgb;
}

const WPI_POST: f32 = 3.14159265;
