// World library: sky nodes and environment lighting helpers, shared by the fractal
// shader and the world passes (world_env.wgsl, world_compute.wgsl). Each includer
// declares `wl` and defines `world_sun_dir()` / `world_sun_color()` / `time_s()`.

const WPI: f32 = 3.14159265;

// World Output values, from `graph_world_settings()` (see codegen::generate_world).
struct WorldSettings {
    strength: f32,
    ambient: f32,
    sun_azimuth: f32,   // degrees
    sun_elevation: f32, // degrees
    sun_color: vec3<f32>,
    sun_strength: f32,
    sun_size: f32,      // angular diameter, degrees
    fog: f32,
}

// Lighting derived from the world, written by world_compute.wgsl, read when shading.
struct WorldLight {
    sun_dir: vec4<f32>,      // xyz (unit), w = unused
    sun_color: vec4<f32>,    // rgb = color * strength, w = angular radius (radians)
    misc: vec4<f32>,         // x = fog density, y = ambient, z = strength, w = env map max LOD
    sh: array<vec4<f32>, 9>, // irradiance / pi as order-2 spherical harmonics (rgb)
}

// Equirectangular mapping: u around the vertical axis (u = 0.5 looks down -z), v from
// straight up (0) to straight down (1).
fn dir_to_equirect(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(d.x, -d.z) / (2.0 * WPI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / WPI);
}

fn equirect_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let phi = (uv.x - 0.5) * 2.0 * WPI;
    let theta = uv.y * WPI;
    return vec3<f32>(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
}

fn rotate_y_deg(d: vec3<f32>, deg: f32) -> vec3<f32> {
    let a = radians(deg);
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

// Sun direction from azimuth / elevation in degrees (azimuth 0 = towards +z).
fn sun_direction(azimuth: f32, elevation: f32) -> vec3<f32> {
    let a = radians(azimuth);
    let e = radians(elevation);
    return vec3<f32>(sin(a) * cos(e), sin(e), cos(a) * cos(e));
}

// Irradiance / pi in direction n from the SH coefficients (already convolved with the
// cosine lobe, see cs_sh).
fn sh_irradiance(sh: array<vec4<f32>, 9>, n: vec3<f32>) -> vec3<f32> {
    let c = sh[0].rgb * 0.282095
        + sh[1].rgb * (0.488603 * n.y)
        + sh[2].rgb * (0.488603 * n.z)
        + sh[3].rgb * (0.488603 * n.x)
        + sh[4].rgb * (1.092548 * n.x * n.y)
        + sh[5].rgb * (1.092548 * n.y * n.z)
        + sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
        + sh[7].rgb * (1.092548 * n.x * n.z)
        + sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(c, vec3<f32>(0.0));
}

// ---- Sky nodes -------------------------------------------------------------------

// Ground -> horizon -> zenith by view height; `sharp` > 1 narrows the horizon band.
fn sky_gradient(dir: vec3<f32>, zenith: vec3<f32>, horizon: vec3<f32>, ground: vec3<f32>, sharp: f32) -> vec3<f32> {
    let y = clamp(normalize(dir).y, -1.0, 1.0);
    let w = pow(abs(y), 1.0 / max(sharp, 0.01));
    return select(mix(horizon, ground, w), mix(horizon, zenith, w), y >= 0.0);
}

fn atmo_sphere(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r * r;
    let h = b * b - c;
    if (h < 0.0) { return vec2<f32>(1e30, -1e30); }
    let s = sqrt(h);
    return vec2<f32>(-b - s, -b + s);
}

// Single-scattering atmosphere (Rayleigh + Mie, after Nishita / wwwtyro), lit by the
// sun. `air` scales Rayleigh (blue sky), `haze` scales Mie (white glow, sun halo).
fn physical_sky(dir: vec3<f32>, sun: vec3<f32>, air: f32, haze: f32, altitude_km: f32, ground: vec3<f32>, strength: f32) -> vec3<f32> {
    let rd = normalize(dir);
    let r_planet = 6371e3;
    let r_atmo = 6471e3;
    let k_rlh = vec3<f32>(5.5e-6, 13.0e-6, 22.4e-6) * max(air, 0.0);
    let k_mie = 21e-6 * max(haze, 0.0);
    let sh_rlh = 8e3;
    let sh_mie = 1.2e3;
    let g = 0.758;
    let ro = vec3<f32>(0.0, r_planet + clamp(altitude_km, 0.0, 90.0) * 1000.0, 0.0);

    let atmo = atmo_sphere(ro, rd, r_atmo);
    let planet = atmo_sphere(ro, rd, r_planet);
    let hits_ground = planet.x > 0.0 && planet.x < 1e29;
    let t_end = select(atmo.y, planet.x, hits_ground);
    let t_start = max(atmo.x, 0.0);
    if (t_end <= t_start) { return vec3<f32>(0.0); }

    let steps = 16;
    let light_steps = 6;
    let ds = (t_end - t_start) / f32(steps);
    let mu = dot(rd, sun);
    let mu2 = mu * mu;
    let p_rlh = 3.0 / (16.0 * WPI) * (1.0 + mu2);
    let g2 = g * g;
    let p_mie = 3.0 / (8.0 * WPI) * ((1.0 - g2) * (mu2 + 1.0)) / (pow(1.0 + g2 - 2.0 * mu * g, 1.5) * (2.0 + g2));

    var total_rlh = vec3<f32>(0.0);
    var total_mie = vec3<f32>(0.0);
    var od_rlh = 0.0;
    var od_mie = 0.0;
    for (var i = 0; i < steps; i++) {
        let p = ro + rd * (t_start + ds * (f32(i) + 0.5));
        let h = length(p) - r_planet;
        let hr = exp(-h / sh_rlh) * ds;
        let hm = exp(-h / sh_mie) * ds;
        od_rlh += hr;
        od_mie += hm;
        // Optical depth towards the sun (zero if the planet shadows this point).
        let lt = atmo_sphere(p, sun, r_atmo).y;
        let dsl = lt / f32(light_steps);
        var lod_rlh = 0.0;
        var lod_mie = 0.0;
        for (var j = 0; j < light_steps; j++) {
            let q = p + sun * (dsl * (f32(j) + 0.5));
            let hq = length(q) - r_planet;
            lod_rlh += exp(-hq / sh_rlh) * dsl;
            lod_mie += exp(-hq / sh_mie) * dsl;
        }
        let tau = k_mie * (od_mie + lod_mie) + k_rlh * (od_rlh + lod_rlh);
        let atten = exp(-tau);
        total_rlh += hr * atten;
        total_mie += hm * atten;
    }
    var c = (p_rlh * k_rlh * total_rlh + p_mie * k_mie * total_mie) * 22.0;
    if (hits_ground) {
        // Lit ground seen through the air in front of it.
        let trans = exp(-(k_rlh * od_rlh + k_mie * od_mie));
        c += ground * max(sun.y, 0.0) * trans * (1.0 / WPI) * 3.0;
    }
    return c * 0.3 * max(strength, 0.0);
}

// Points of light on a grid of cells around the viewer; `scale` cells per radian.
fn star_field(dir: vec3<f32>, density: f32, scale: f32, brightness: f32, twinkle: f32, seed: f32, t: f32) -> vec3<f32> {
    let p = normalize(dir) * max(scale, 1.0);
    let cell = vec3<i32>(floor(p));
    let h = tex_hash3(cell, tex_seed(seed));
    let r = tex_unit3(h);
    if (r.x > density) { return vec3<f32>(0.0); }
    // The star sits away from the cell's faces so it is never cut off.
    let star = vec3<f32>(cell) + 0.25 + 0.5 * tex_unit3(tex_hash(h));
    let d = length(p - star);
    let size = 0.04 + 0.05 * r.y;
    var b = exp(-d * d / (2.0 * size * size)) * brightness * (0.2 + 0.8 * r.z * r.z);
    b *= 1.0 + twinkle * sin(t * (2.0 + 6.0 * r.y) + 40.0 * r.x);
    let tint = blackbody(3000.0 + 9000.0 * r.z);
    return tint / max(max(tint.r, max(tint.g, tint.b)), 1e-3) * max(b, 0.0);
}

// An equirectangular panorama around the scene, rotated about the vertical axis.
fn env_texture(img: texture_2d<f32>, samp: sampler, dir: vec3<f32>, rotation: f32) -> vec3<f32> {
    let d = rotate_y_deg(normalize(dir), rotation);
    return textureSampleLevel(img, samp, dir_to_equirect(d), 0.0).rgb;
}
