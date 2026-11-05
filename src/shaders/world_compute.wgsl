// World lighting, computed on the GPU so every node works in the world graph:
// - `cs_settings` evaluates the World Output's sun / fog / ambient values (run before
//   the environment map is rendered, which may need the sun direction);
// - `cs_sh` projects a small mip of the environment map onto order-2 spherical
//   harmonics, convolved with the cosine lobe: ambient light for every normal.
// `//@WORLD` is replaced with nodes.wgsl, world.wgsl and the generated world graph.

@group(0) @binding(0) var<uniform> wp: array<vec4<f32>, 256>;
@group(0) @binding(1) var world_img0: texture_2d<f32>;
@group(0) @binding(2) var world_img1: texture_2d<f32>;
@group(0) @binding(3) var world_samp: sampler;
@group(0) @binding(4) var<uniform> world_time: vec4<f32>;
@group(1) @binding(0) var<storage, read_write> wl: WorldLight;
@group(1) @binding(1) var env_src: texture_2d<f32>;

// Mip levels of the environment map (see world.rs), for rough reflections.
const ENV_MAX_LOD: f32 = 7.0;

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

@compute @workgroup_size(1)
fn cs_settings() {
    let s = graph_world_settings();
    wl.sun_dir = vec4<f32>(sun_direction(s.sun_azimuth, s.sun_elevation), 0.0);
    wl.sun_color = vec4<f32>(max(s.sun_color, vec3<f32>(0.0)) * max(s.sun_strength, 0.0), radians(max(s.sun_size, 0.0)) * 0.5);
    wl.misc = vec4<f32>(max(s.fog, 0.0), max(s.ambient, 0.0), max(s.strength, 0.0), ENV_MAX_LOD);
}

const SH_THREADS: u32 = 64u;
var<workgroup> partial: array<array<vec3<f32>, 9>, 64>;

@compute @workgroup_size(64)
fn cs_sh(@builtin(local_invocation_index) li: u32) {
    let size = textureDimensions(env_src);
    let count = size.x * size.y;
    var acc: array<vec3<f32>, 9>;
    for (var k = 0; k < 9; k++) {
        acc[k] = vec3<f32>(0.0);
    }
    for (var i = li; i < count; i += SH_THREADS) {
        let x = i % size.x;
        let y = i / size.x;
        let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(size);
        let d = equirect_to_dir(uv);
        // Solid angle of the texel.
        let w = (2.0 * WPI / f32(size.x)) * (WPI / f32(size.y)) * sin(uv.y * WPI);
        let l = min(textureLoad(env_src, vec2<i32>(i32(x), i32(y)), 0).rgb, vec3<f32>(1000.0)) * w;
        acc[0] += l * 0.282095;
        acc[1] += l * (0.488603 * d.y);
        acc[2] += l * (0.488603 * d.z);
        acc[3] += l * (0.488603 * d.x);
        acc[4] += l * (1.092548 * d.x * d.y);
        acc[5] += l * (1.092548 * d.y * d.z);
        acc[6] += l * (0.315392 * (3.0 * d.z * d.z - 1.0));
        acc[7] += l * (1.092548 * d.x * d.z);
        acc[8] += l * (0.546274 * (d.x * d.x - d.y * d.y));
    }
    partial[li] = acc;
    workgroupBarrier();
    if (li == 0u) {
        // Cosine-lobe convolution divided by pi: band weights 1, 2/3, 1/4.
        let band = array<f32, 9>(1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25);
        for (var k = 0; k < 9; k++) {
            var sum = vec3<f32>(0.0);
            for (var t = 0u; t < SH_THREADS; t++) {
                sum += partial[t][k];
            }
            wl.sh[k] = vec4<f32>(sum * band[k], 0.0);
        }
    }
}
