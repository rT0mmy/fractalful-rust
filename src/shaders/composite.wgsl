// Final pass: upscales the linear scene (the processing graph's output), overlays the UI
// layer, and encodes for the display.
//
// - SDR:    the view transform (ACES filmic, AgX, Reinhard or clip; chosen in the
//           processing graph's Image Output), then sRGB (by the swapchain format or
//           manually).
// - HDR10:  highlight shoulder up to the display's peak, BT.709 -> BT.2020, PQ (ST 2084)
//           encode of absolute nits into an Rgb10a2 swapchain (BT.2100 PQ color space).
// - scRGB:  same nits, written linear with 1.0 = 80 nits (Rgba16Float swapchain).
//
// In both HDR modes scene value 1.0 and UI white are shown at "paper white" nits, so
// SDR-range content keeps its brightness and only highlights go above it.

struct Config {
    mode: f32,        // 0 = SDR, 1 = HDR10 (PQ), 2 = scRGB
    paper_white: f32, // nits for scene 1.0 / UI white
    peak: f32,        // display peak nits
    srgb_encode: f32, // 1 = SDR swapchain isn't an *Srgb format, encode manually
    view: f32,        // view transform: 0 ACES, 1 AgX, 2 Reinhard, 3 standard (clip)
    no_ui: f32,       // 1 = skip the UI layer (screenshots)
    fit: vec2<f32>,   // scene size as a fraction of the output (letterboxing)
}

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var ui: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> cfg: Config;

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

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// AgX (Troy Sobotka), in the polynomial fit by Benjamin Wrensch: wide dynamic range,
// and saturated highlights roll off to white instead of skewing hue.
fn agx(c: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(0.842479, 0.0423282, 0.0423756, 0.0784336, 0.878468, 0.0784336, 0.0792237, 0.0791661, 0.879142);
    let outset = mat3x3<f32>(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116,
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = clamp(log2(max(inset * max(c, vec3<f32>(0.0)), vec3<f32>(1e-10))), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    let x2 = v * v;
    let x4 = x2 * x2;
    v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
    // AgX's output is display encoded (about gamma 2.2); return linear.
    return pow(clamp(outset * v, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn view_transform(c: vec3<f32>) -> vec3<f32> {
    let v = i32(cfg.view);
    if (v == 1) { return agx(c); }
    if (v == 2) { return c / (vec3<f32>(1.0) + max(c, vec3<f32>(0.0))); }
    if (v == 3) { return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)); }
    return aces(c);
}

fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// Leaves values below the knee untouched and rolls everything above it smoothly
// toward `limit` (both relative to paper white). Applied to the max channel and used
// as a scale, so hues don't shift as highlights compress.
fn hdr_shoulder(c: vec3<f32>, limit: f32) -> vec3<f32> {
    let knee = 0.8 * min(limit, 1.0);
    let m = max(max(c.r, c.g), c.b);
    if (m <= knee || limit <= knee) {
        return min(c, vec3<f32>(limit));
    }
    let span = limit - knee;
    let mapped = knee + span * (1.0 - exp(-(m - knee) / span));
    return c * (mapped / m);
}

fn bt709_to_bt2020(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.6274040, 0.3292820, 0.0433136)),
        dot(c, vec3<f32>(0.0690970, 0.9195400, 0.0113612)),
        dot(c, vec3<f32>(0.0163916, 0.0880132, 0.8955950)),
    );
}

// SMPTE ST 2084 inverse EOTF: absolute nits -> PQ signal in 0..1.
fn pq_encode(nits: vec3<f32>) -> vec3<f32> {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let y = pow(clamp(nits / 10000.0, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(m1));
    return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3<f32>(m2));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Letterbox a scene whose aspect differs from the output (e.g. the video preview).
    let suv = (in.uv - 0.5) / cfg.fit + 0.5;
    let inside = all(suv >= vec2<f32>(0.0)) && all(suv <= vec2<f32>(1.0));
    var s = textureSample(scene, samp, suv);
    if (!inside) {
        s = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // UI layer matches the swapchain size: load texels directly for crisp text.
    var u = textureLoad(ui, vec2<i32>(floor(in.pos.xy)), 0); // premultiplied, linear
    if (cfg.no_ui > 0.5) { u = vec4<f32>(0.0); }
    let raw = s.a < 0.0; // debug views: no tone mapping

    if (cfg.mode < 0.5) {
        let scene_c = select(view_transform(s.rgb), clamp(s.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), raw);
        let c = scene_c * (1.0 - u.a) + u.rgb;
        return vec4<f32>(select(c, srgb_encode(c), cfg.srgb_encode > 0.5), 1.0);
    }

    let limit = cfg.peak / cfg.paper_white;
    // HDR keeps highlights above paper white: a shoulder up to the display's peak, or a
    // hard clip there for the standard transform.
    var hdr = hdr_shoulder(max(s.rgb, vec3<f32>(0.0)), limit);
    if (i32(cfg.view) == 3) { hdr = clamp(s.rgb, vec3<f32>(0.0), vec3<f32>(limit)); }
    let scene_rel = select(hdr, clamp(s.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), raw);
    let nits = (scene_rel * (1.0 - u.a) + u.rgb) * cfg.paper_white;

    if (cfg.mode < 1.5) {
        return vec4<f32>(pq_encode(bt709_to_bt2020(nits)), 1.0);
    }
    return vec4<f32>(nits / 80.0, 1.0);
}
