// One direction of a separable gaussian blur (the Blur node of the processing graph).
// Large radii are blurred at reduced resolution (see post.rs), so `taps` stays small.

struct Blur {
    step: vec2<f32>, // one texel along the blur direction, in UV
    sigma: f32,      // in texels
    taps: f32,       // taps on each side of the center
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> cfg: Blur;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = i32(cfg.taps);
    let k = -0.5 / max(cfg.sigma * cfg.sigma, 1e-4);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = -n; i <= n; i++) {
        let w = exp(f32(i * i) * k);
        sum += textureSampleLevel(src, samp, in.uv + cfg.step * f32(i), 0.0).rgb * w;
        wsum += w;
    }
    return vec4<f32>(sum / wsum, 1.0);
}
