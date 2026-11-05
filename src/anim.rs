//! Keyframe curves, shared by the camera path (`video`) and animated node inputs
//! (`graph::Track`).
//!
//! A curve is a list of keys sorted by time, each holding up to N channels. Each segment
//! uses its first key's interpolation. Smooth is a cubic Hermite whose tangents are the
//! time-aware slopes through the neighbouring keys, clamped flat where a channel peaks
//! (Blender's "auto clamped" handles): speed stays continuous even with unevenly spaced
//! keys, and the curve never overshoots a key.

/// How a curve moves from a key to the next one (Blender's interpolation).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum Interp {
    /// Smooth curve through the keys, keeping the speed continuous across them.
    #[default]
    Smooth,
    /// Slows to a stop at both keys.
    Ease,
    Linear,
    /// Holds this key's value until the next key, then jumps.
    Constant,
}

impl Interp {
    pub const ALL: [Interp; 4] = [Self::Smooth, Self::Ease, Self::Linear, Self::Constant];

    pub fn label(self) -> &'static str {
        match self {
            Self::Smooth => "Smooth",
            Self::Ease => "Ease in / out",
            Self::Linear => "Linear",
            Self::Constant => "Constant (hold)",
        }
    }
}

/// Keys closer than this in time are the same key.
pub const KEY_EPSILON: f32 = 1e-3;

/// One key of a curve: time, channel values, interpolation towards the next key.
pub type Key<const N: usize> = (f32, [f32; N], Interp);

/// The curve's value at `t`; holds the first / last key outside them. `None` if empty.
pub fn sample<const N: usize>(keys: &[Key<N>], t: f32) -> Option<[f32; N]> {
    let (first, last) = (keys.first()?, keys.last()?);
    if keys.len() == 1 || t <= first.0 {
        return Some(first.1);
    }
    if t >= last.0 {
        return Some(last.1);
    }
    let i = keys.windows(2).position(|w| t >= w[0].0 && t < w[1].0)?;
    let ((t1, p1, interp), (t2, p2, _)) = (keys[i], keys[i + 1]);
    let h = (t2 - t1).max(1e-6);
    let u = (t - t1) / h;
    let mut out = [0.0; N];
    match interp {
        Interp::Constant => return Some(p1),
        Interp::Linear => {
            for c in 0..N {
                out[c] = p1[c] + (p2[c] - p1[c]) * u;
            }
        }
        Interp::Smooth | Interp::Ease => {
            let (m1, m2) = if interp == Interp::Ease { ([0.0; N], [0.0; N]) } else { (slope(keys, i), slope(keys, i + 1)) };
            let (u2, u3) = (u * u, u * u * u);
            let (h00, h10, h01, h11) = (2.0 * u3 - 3.0 * u2 + 1.0, u3 - 2.0 * u2 + u, 3.0 * u2 - 2.0 * u3, u3 - u2);
            for c in 0..N {
                out[c] = h00 * p1[c] + h10 * h * m1[c] + h01 * p2[c] + h11 * h * m2[c];
            }
        }
    }
    Some(out)
}

/// Rate of change of each channel at key `j`: the slope between its neighbours (one
/// side at the ends), zero where the channel peaks there.
fn slope<const N: usize>(keys: &[Key<N>], j: usize) -> [f32; N] {
    let (a, b) = (j.saturating_sub(1), (j + 1).min(keys.len() - 1));
    let dt = (keys[b].0 - keys[a].0).max(1e-6);
    let (va, vj, vb) = (keys[a].1, keys[j].1, keys[b].1);
    let mut m = [0.0; N];
    for c in 0..N {
        let peak = a != j && b != j && (vj[c] - va[c]) * (vb[c] - vj[c]) <= 0.0;
        m[c] = if peak { 0.0 } else { (vb[c] - va[c]) / dt };
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(pts: &[(f32, f32)], interp: Interp) -> Vec<Key<1>> {
        pts.iter().map(|&(t, v)| (t, [v], interp)).collect()
    }

    #[test]
    fn passes_through_keys_and_holds_outside() {
        let k = keys(&[(0.0, 1.0), (1.0, 3.0), (2.5, -1.0)], Interp::Smooth);
        for &(t, [v], _) in &k {
            assert!((sample(&k, t).unwrap()[0] - v).abs() < 1e-5);
        }
        assert_eq!(sample(&k, -3.0).unwrap()[0], 1.0);
        assert_eq!(sample(&k, 9.0).unwrap()[0], -1.0);
        assert!(sample::<1>(&[], 0.0).is_none());
    }

    #[test]
    fn interpolation_modes() {
        let x = |interp| {
            let k = keys(&[(0.0, 0.0), (1.0, 2.0)], interp);
            move |t: f32| sample(&k, t).unwrap()[0]
        };
        assert_eq!(x(Interp::Constant)(0.9), 0.0);
        assert!((x(Interp::Linear)(0.25) - 0.5).abs() < 1e-6);
        let ease = x(Interp::Ease);
        assert!(ease(0.01) < 0.01 && (ease(0.5) - 1.0).abs() < 1e-5);
    }

    /// Unevenly spaced keys on a straight line keep a constant speed.
    #[test]
    fn smooth_speed_is_continuous() {
        let k = keys(&[(0.0, 0.0), (1.0, 1.0), (4.0, 4.0)], Interp::Smooth);
        for t in [0.2, 0.9, 1.0, 1.1, 2.5, 3.7] {
            let v = (sample(&k, t + 1e-3).unwrap()[0] - sample(&k, t - 1e-3).unwrap()[0]) / 2e-3;
            assert!((v - 1.0).abs() < 0.02, "speed {v} at t = {t}");
        }
    }

    #[test]
    fn no_overshoot_at_a_peak() {
        let k = keys(&[(0.0, 0.0), (1.0, 1.0), (1.2, 0.0)], Interp::Smooth);
        for i in 0..=120 {
            assert!(sample(&k, i as f32 * 0.01).unwrap()[0] <= 1.0 + 1e-4);
        }
    }
}
