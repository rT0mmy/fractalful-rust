// Renders the world graph into the environment map (equirectangular, mip 0), which
// lights the fractal: rough reflections sample its mips, ambient light comes from its
// spherical harmonics (world_compute.wgsl). The sun disc is left out (the sun lights
// the scene directly). `//@WORLD` is replaced with nodes.wgsl, world.wgsl and the
// generated world graph.

@group(0) @binding(0) var<uniform> wp: array<vec4<f32>, 256>;
@group(0) @binding(1) var world_img0: texture_2d<f32>;
@group(0) @binding(2) var world_img1: texture_2d<f32>;
@group(0) @binding(3) var world_samp: sampler;
@group(0) @binding(4) var<uniform> world_time: vec4<f32>;
@group(1) @binding(0) var<uniform> wl: WorldLight;

fn time_s() -> f32 {
    return world_time.x;
}

fn world_sun_dir() -> vec3<f32> {
    return wl.sun_dir.xyz;
}

fn world_sun_color() -> vec3<f32> {
    return wl.sun_color.rgb;
}

//@WORLD

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
fn fs_env(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(min(graph_world(equirect_to_dir(in.uv)), vec3<f32>(60000.0)), 1.0);
}
