// Physically-based bloom (Call of Duty: Advanced Warfare / Jimenez 2014 style):
// a bright-pass prefilter, a mip pyramid of 13-tap downsamples, then 3x3 tent
// upsamples blended additively back up the pyramid. The result (mip 0) is added to
// the scene in the compositor, before tone mapping / HDR encoding.

struct Pass {
    texel: vec2<f32>,   // 1 / source size
    threshold: f32,     // linear brightness where bloom starts (prefilter only)
    knee: f32,          // soft-knee width (prefilter only)
    radius: f32,        // upsample filter radius in source texels
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> pass_cfg: Pass;

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

fn tap(uv: vec2<f32>, dx: f32, dy: f32) -> vec3<f32> {
    return textureSampleLevel(src, samp, uv + vec2<f32>(dx, dy) * pass_cfg.texel, 0.0).rgb;
}

// 13-tap downsample: a 4x4 box made of five overlapping 2x2 boxes, which avoids the
// shimmering of a plain 2x2 average.
fn downsample13(uv: vec2<f32>) -> vec3<f32> {
    let a = tap(uv, -2.0, -2.0);
    let b = tap(uv, 0.0, -2.0);
    let c = tap(uv, 2.0, -2.0);
    let d = tap(uv, -2.0, 0.0);
    let e = tap(uv, 0.0, 0.0);
    let f = tap(uv, 2.0, 0.0);
    let g = tap(uv, -2.0, 2.0);
    let h = tap(uv, 0.0, 2.0);
    let i = tap(uv, 2.0, 2.0);
    let j = tap(uv, -1.0, -1.0);
    let k = tap(uv, 1.0, -1.0);
    let l = tap(uv, -1.0, 1.0);
    let m = tap(uv, 1.0, 1.0);
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

// Soft-knee bright pass, applied on the first downsample. Values are clamped first so
// a single extreme pixel (a sub-pixel specular highlight) can't flicker into a blob.
@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    // Debug views (alpha < 0) never bloom.
    if (textureSampleLevel(src, samp, in.uv, 0.0).a < 0.0) { return vec4<f32>(0.0); }
    let c = min(downsample13(in.uv), vec3<f32>(64.0));
    let br = max(c.r, max(c.g, c.b));
    let knee = max(pass_cfg.threshold * pass_cfg.knee, 1e-4);
    var soft = clamp(br - pass_cfg.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    let w = max(soft, br - pass_cfg.threshold) / max(br, 1e-4);
    return vec4<f32>(c * w, 1.0);
}

@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample13(in.uv), 1.0);
}

// 3x3 tent upsample of the smaller mip, blended additively onto the larger one.
@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    let r = pass_cfg.radius;
    let s = tap(in.uv, -r, -r) + tap(in.uv, r, -r) + tap(in.uv, -r, r) + tap(in.uv, r, r)
        + 2.0 * (tap(in.uv, 0.0, -r) + tap(in.uv, -r, 0.0) + tap(in.uv, r, 0.0) + tap(in.uv, 0.0, r))
        + 4.0 * tap(in.uv, 0.0, 0.0);
    return vec4<f32>(s / 16.0, 1.0);
}
