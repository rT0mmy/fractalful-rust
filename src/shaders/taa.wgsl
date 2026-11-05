// Temporal anti-aliasing and upscaling: resolves the fractal render (at render scale,
// one jittered sample per pixel while moving) into a history at window resolution.
//
// Each output pixel:
// - gathers the 3x3 render samples around it, weighted by their (jittered) distance:
//   the current estimate, plus their mean / spread for clamping;
// - reprojects into the previous frame with the depth in the render's alpha (sky pixels
//   by direction only) and reads the history there (Catmull-Rom, so it doesn't blur);
// - rejects history that is off screen or at a clearly different depth (disocclusion),
//   clips the rest to the current neighbourhood's color box (no ghosting), and blends in
//   a fraction of the current samples (luminance weighted, so fireflies don't flicker).
// While the camera is still the renderer accumulates exact samples instead; the history
// then just fades over to that accumulation (`mode` 1).
//
// Output alpha keeps the depth (for the next frame and the Render Layers node).

struct Taa {
    cam_pos: vec4<f32>,    // xyz, w = tan(fov / 2)
    cam_fwd: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    prev_pos: vec4<f32>,   // xyz, w = tan(fov / 2)
    prev_fwd: vec4<f32>,
    prev_right: vec4<f32>,
    prev_up: vec4<f32>,
    sizes: vec4<f32>,      // render width, height, output width, height
    misc: vec4<f32>,       // xy = this frame's jitter (render pixels), z = mode (0 temporal, 1 still), w = still weight
    flags: vec4<f32>,      // x = history valid
}

@group(0) @binding(0) var cur: texture_2d<f32>;
@group(0) @binding(1) var hist: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var<uniform> t: Taa;

const SKY_DEPTH: f32 = 60000.0;

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

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Catmull-Rom history read in 9 bilinear taps (no blur from repeated resampling).
fn history_bicubic(uv: vec2<f32>) -> vec3<f32> {
    let size = t.sizes.zw;
    let pos = uv * size;
    let center = floor(pos - 0.5) + 0.5;
    let f = pos - center;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = -0.5 * f3 + f2 - 0.5 * f;
    let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    let w3 = 0.5 * f3 - 0.5 * f2;
    let w12 = w1 + w2;
    let tc12 = (center + w2 / w12) / size;
    let tc0 = (center - 1.0) / size;
    let tc3 = (center + 2.0) / size;
    var c = vec3<f32>(0.0);
    c += textureSampleLevel(hist, lin, vec2<f32>(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y);
    c += textureSampleLevel(hist, lin, vec2<f32>(tc0.x, tc12.y), 0.0).rgb * (w0.x * w12.y);
    c += textureSampleLevel(hist, lin, vec2<f32>(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y);
    c += textureSampleLevel(hist, lin, vec2<f32>(tc3.x, tc12.y), 0.0).rgb * (w3.x * w12.y);
    c += textureSampleLevel(hist, lin, vec2<f32>(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);
    let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(c / wsum, vec3<f32>(0.0));
}

// Clips `h` toward the box center until it's inside the box.
fn clip_to_box(h: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let center = 0.5 * (lo + hi);
    let extent = max(0.5 * (hi - lo), vec3<f32>(1e-5));
    let d = h - center;
    let a = abs(d / extent);
    let m = max(a.x, max(a.y, a.z));
    return select(h, center + d / m, m > 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let rsize = t.sizes.xy;
    let rmax = vec2<i32>(rsize) - 1;
    // This pixel in render pixels; render texel k holds the sample at k + 0.5 + jitter.
    let p = uv * rsize - t.misc.xy;
    let base = vec2<i32>(floor(p));

    if (t.misc.z > 0.5) {
        // Still: the render is the exact accumulation at full resolution.
        let a = textureLoad(cur, clamp(base, vec2<i32>(0), rmax), 0);
        if (a.a < 0.0 || t.flags.x < 0.5) { return a; }
        let h = textureLoad(hist, vec2<i32>(in.pos.xy), 0);
        return vec4<f32>(mix(h.rgb, a.rgb, t.misc.w), a.a);
    }

    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var closest = 1e9;
    var depth = SKY_DEPTH;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let k = clamp(base + vec2<i32>(x, y), vec2<i32>(0), rmax);
            let s = textureLoad(cur, k, 0);
            if (s.a < 0.0) {
                // Debug views aren't resolved.
                return textureLoad(cur, clamp(base, vec2<i32>(0), rmax), 0);
            }
            let c = s.rgb / (1.0 + luma(s.rgb));
            let d = vec2<f32>(k) + 0.5 - p;
            let d2 = dot(d, d);
            let w = exp(-2.2 * d2);
            sum += c * w;
            wsum += w;
            m1 += c;
            m2 += c * c;
            if (d2 < closest) {
                closest = d2;
                depth = s.a;
            }
        }
    }
    let current = sum / wsum;
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    // How well a sample covers this pixel: 1 at its center.
    let coverage = exp(-2.2 * closest);

    // The point seen here, in the previous frame.
    let aspect = t.sizes.z / t.sizes.w;
    let ndc = (uv * 2.0 - 1.0) * vec2<f32>(aspect, -1.0);
    let rd = normalize(t.cam_fwd.xyz + (ndc.x * t.cam_right.xyz + ndc.y * t.cam_up.xyz) * t.cam_pos.w);
    let sky = depth >= SKY_DEPTH * 0.99;
    var v = rd;
    if (!sky) { v = t.cam_pos.xyz + rd * depth - t.prev_pos.xyz; }
    let z = dot(v, t.prev_fwd.xyz);
    let q = vec2<f32>(dot(v, t.prev_right.xyz), dot(v, t.prev_up.xyz)) / max(z, 1e-9) / t.prev_pos.w;
    let prev_uv = vec2<f32>((q.x / aspect + 1.0) * 0.5, (1.0 - q.y) * 0.5);
    var valid = t.flags.x > 0.5 && z > 0.0 && all(prev_uv > vec2<f32>(0.0)) && all(prev_uv < vec2<f32>(1.0));
    if (valid) {
        let hd = textureLoad(hist, vec2<i32>(prev_uv * t.sizes.zw), 0).a;
        let hist_sky = hd >= SKY_DEPTH * 0.99;
        if (sky || hist_sky) {
            valid = sky == hist_sky;
        } else {
            let expected = length(v);
            valid = abs(hd - expected) < 0.1 * expected;
        }
    }

    var c = current;
    if (valid) {
        let h = history_bicubic(prev_uv);
        let hc = clip_to_box(h / (1.0 + luma(h)), mean - 1.25 * sigma, mean + 1.25 * sigma);
        // Lean on the history, more so where no sample lands near this pixel.
        let alpha = mix(0.05, 0.16, coverage);
        c = mix(hc, current, alpha);
    }
    c = c / max(1.0 - luma(c), 1e-4);
    return vec4<f32>(max(c, vec3<f32>(0.0)), depth);
}
