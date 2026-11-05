//! Ready-made world and processing graphs (the node editor's Load preset menu in the
//! World and Processing modes). Material presets are in `preset`.

use super::*;

pub const WORLD_PRESETS: &[&str] = &["Twilight gradient", "Clear day", "Golden hour", "Sunset", "Starry night", "Alien dusk", "Studio"];

pub const POST_PRESETS: &[&str] = &["Bloom", "Cinematic", "Dreamy glow", "Lens", "Kaleidoscope", "Retro", "Clean"];

/// Sun settings on a World Output: azimuth / elevation (degrees), color, strength, size.
fn sun(g: &mut Graph, out: NodeId, az: f32, el: f32, color: [f32; 3], strength: f32, size: f32) {
    let n = g.node_mut(out).unwrap();
    n.set(WORLD_SUN_AZIMUTH, &[az]);
    n.set(WORLD_SUN_ELEVATION, &[el]);
    n.set(WORLD_SUN_COLOR, &color);
    n.set(WORLD_SUN_STRENGTH, &[strength]);
    n.set(WORLD_SUN_SIZE, &[size]);
}

fn gradient(g: &mut Graph, pos: [f32; 2], zenith: [f32; 3], horizon: [f32; 3], ground: [f32; 3], sharp: f32) -> NodeId {
    let n = g.add(NodeKind::SkyGradient, pos);
    let node = g.node_mut(n).unwrap();
    node.set(1, &zenith);
    node.set(2, &horizon);
    node.set(3, &ground);
    node.set(4, &[sharp]);
    n
}

/// `a` + `b` through a Blend node in Add mode.
fn add(g: &mut Graph, pos: [f32; 2], a: (NodeId, usize), b: (NodeId, usize)) -> NodeId {
    let n = g.add(NodeKind::Blend, pos);
    g.node_mut(n).unwrap().set(0, &[1.0]);
    g.connect(a.0, a.1, n, 2);
    g.connect(b.0, b.1, n, 3);
    n
}

pub fn world_preset(name: &str) -> Graph {
    let mut g = Graph::new();
    let out = g.add(NodeKind::WorldOutput, [620.0, 40.0]);
    let set = |g: &mut Graph, j: usize, v: f32| g.node_mut(out).unwrap().set(j, &[v]);
    match name {
        "Clear day" | "Golden hour" | "Sunset" => {
            let sky = g.add(NodeKind::PhysicalSky, [300.0, 40.0]);
            g.connect(sky, 0, out, WORLD_COLOR);
            // Azimuth 180 is behind the default camera (lighting the fractal's front); a
            // sunset is shown backlit, with its glow on the horizon.
            let (az, el, color, strength, haze, ambient) = match name {
                "Clear day" => (215.0, 38.0, [1.0, 0.95, 0.88], 2.2, 1.0, 1.2),
                "Golden hour" => (125.0, 9.0, [1.0, 0.72, 0.45], 2.4, 1.6, 1.4),
                _ => (40.0, 2.0, [1.0, 0.45, 0.22], 1.6, 2.5, 2.2),
            };
            g.node_mut(sky).unwrap().set(2, &[haze]);
            sun(&mut g, out, az, el, color, strength, 0.6);
            set(&mut g, WORLD_AMBIENT, ambient);
            set(&mut g, WORLD_FOG, 0.01);
        }
        "Starry night" => {
            let sky = gradient(&mut g, [40.0, 40.0], [0.004, 0.006, 0.016], [0.018, 0.022, 0.045], [0.003, 0.003, 0.006], 0.6);
            let stars = g.add(NodeKind::StarField, [40.0, 260.0]);
            g.node_mut(stars).unwrap().set(4, &[0.3]);
            let both = add(&mut g, [340.0, 120.0], (sky, 0), (stars, 0));
            g.connect(both, 0, out, WORLD_COLOR);
            // Moonlight.
            sun(&mut g, out, -60.0, 32.0, [0.75, 0.82, 1.0], 0.45, 0.55);
            set(&mut g, WORLD_AMBIENT, 4.0);
            set(&mut g, WORLD_FOG, 0.0);
        }
        "Alien dusk" => {
            let sky = gradient(&mut g, [40.0, 40.0], [0.05, 0.01, 0.12], [0.6, 0.12, 0.35], [0.01, 0.04, 0.05], 2.5);
            let stars = g.add(NodeKind::StarField, [40.0, 260.0]);
            let n = g.node_mut(stars).unwrap();
            n.set(1, &[0.15]);
            n.set(3, &[8.0]);
            let both = add(&mut g, [340.0, 120.0], (sky, 0), (stars, 0));
            g.connect(both, 0, out, WORLD_COLOR);
            sun(&mut g, out, 120.0, 6.0, [1.0, 0.35, 0.6], 1.8, 3.0);
            set(&mut g, WORLD_AMBIENT, 1.6);
            set(&mut g, WORLD_FOG, 0.04);
        }
        "Studio" => {
            let sky = gradient(&mut g, [300.0, 40.0], [0.16, 0.16, 0.17], [0.06, 0.06, 0.065], [0.015, 0.015, 0.017], 1.0);
            g.connect(sky, 0, out, WORLD_COLOR);
            sun(&mut g, out, 215.0, 45.0, [1.0, 0.97, 0.93], 2.4, 0.0);
            set(&mut g, WORLD_AMBIENT, 1.5);
            set(&mut g, WORLD_FOG, 0.0);
        }
        _ => return Graph::world_from(&LegacyLook::default()),
    }
    g
}

/// A chain of processing nodes from Render Layers to the Image Output, each fed by the
/// previous one's Image; `setup` sets each node's values. Returns the graph and its nodes.
fn chain(kinds: &[NodeKind], view: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new();
    let render = g.add(NodeKind::RenderLayers, [40.0, 40.0]);
    let mut prev = render;
    let mut nodes = Vec::new();
    for (i, &k) in kinds.iter().enumerate() {
        let n = g.add(k, [300.0 + 280.0 * i as f32, 40.0 + 30.0 * (i % 2) as f32]);
        g.connect(prev, 0, n, 0);
        nodes.push(n);
        prev = n;
    }
    let out = g.add(NodeKind::PostOutput, [300.0 + 280.0 * kinds.len() as f32, 40.0]);
    g.node_mut(out).unwrap().set(POST_VIEW_TRANSFORM, &[view as f32]);
    g.connect(prev, 0, out, 0);
    (g, nodes)
}

pub fn post_preset(name: &str) -> Graph {
    use NodeKind::*;
    match name {
        "Cinematic" => {
            let (mut g, n) = chain(&[Bloom, ColorBalance, Curves, Vignette, FilmGrain, MotionBlur], 1);
            g.node_mut(n[0]).unwrap().set(1, &[0.5]);
            let cb = g.node_mut(n[1]).unwrap();
            cb.set(1, &[0.6]);
            cb.set(2, &[0.42, 0.5, 0.58]); // teal shadows
            cb.set(4, &[0.54, 0.5, 0.45]); // warm highlights
            // A gentle S curve on the combined channel.
            let curves = g.node_mut(n[2]).unwrap();
            let o = curves.value_offset(2);
            for (k, y) in [0.0, 0.21, 0.5, 0.79, 1.0].into_iter().enumerate() {
                curves.values[o + k * 2 + 1] = y;
            }
            let v = g.node_mut(n[3]).unwrap();
            v.set(1, &[0.45]);
            v.set(2, &[0.75]);
            g.node_mut(n[4]).unwrap().set(1, &[0.025]);
            g
        }
        "Dreamy glow" => {
            let (mut g, n) = chain(&[Bloom, Blend, MotionBlur], 0);
            let b = g.node_mut(n[0]).unwrap();
            b.set(1, &[1.4]);
            b.set(2, &[0.6]);
            b.set(4, &[2.0]);
            // A blurred copy screened over the image.
            let blur = g.add(Blur, [300.0, 300.0]);
            g.node_mut(blur).unwrap().set(1, &[2.5]);
            g.connect(n[0], 0, blur, 0);
            let blend = g.node_mut(n[1]).unwrap();
            blend.set(0, &[4.0]);
            blend.set(1, &[0.45]);
            g.connect(n[0], 0, n[1], 2);
            g.connect(blur, 0, n[1], 3);
            g
        }
        "Lens" => {
            let (mut g, n) = chain(&[Bloom, LensDistortion, Vignette, MotionBlur], 0);
            let l = g.node_mut(n[1]).unwrap();
            l.set(1, &[0.12]);
            l.set(2, &[0.35]);
            g.node_mut(n[2]).unwrap().set(1, &[0.35]);
            g
        }
        "Kaleidoscope" => {
            let (mut g, n) = chain(&[Kaleidoscope, Bloom, MotionBlur], 0);
            // Slowly turning wedges.
            let osc = g.add(Oscillator, [300.0, 300.0]);
            let o = g.node_mut(osc).unwrap();
            o.set(0, &[2.0]); // sawtooth
            o.set(1, &[0.02]);
            o.set(2, &[180.0]);
            g.connect(osc, 0, n[0], 2);
            g
        }
        "Retro" => {
            let (mut g, n) = chain(&[Pixelate, Posterize, MotionBlur], 3);
            g.node_mut(n[0]).unwrap().set(1, &[4.0]);
            g.node_mut(n[1]).unwrap().set(1, &[6.0]);
            g
        }
        "Clean" => chain(&[MotionBlur], 0).0,
        _ => Graph::post_from(&LegacyLook::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_compile() {
        for name in WORLD_PRESETS {
            let g = world_preset(name);
            assert!(g.find(NodeKind::WorldOutput).is_some(), "{name}");
            codegen::generate_world(&g).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        for name in POST_PRESETS {
            let g = post_preset(name);
            codegen::generate_post(&g).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(g.motion_blur_shutter() > 0.0, "{name} keeps the default export motion blur");
        }
        assert!(world_preset("Twilight gradient").same_content(&Graph::world_from(&LegacyLook::default())));
        assert!(post_preset("Bloom").same_content(&Graph::post_from(&LegacyLook::default())));
    }
}
