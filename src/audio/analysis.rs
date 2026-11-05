//! Offline analysis of a whole song, run once when it's loaded (on a background
//! thread): loudness per frequency band over time, drum hits (kick / snare / hi-hat
//! onsets) and a beat grid (tempo estimate + dynamic-programming beat tracking, after
//! Ellis 2007). Everything is stored as time series, so audio nodes can be evaluated
//! at any time, e.g. per motion-blur sample while exporting video.

use std::f32::consts::PI;

/// Analysis window and hop, in samples.
const FFT: usize = 2048;
const HOP: usize = 512;

/// Frequency bands of the Audio Level node; order matches `graph::AUDIO_BANDS`.
const BANDS: [(f32, f32); 6] = [
    (20.0, 16000.0), // Full
    (20.0, 150.0),   // Bass
    (150.0, 500.0),  // Low mids
    (500.0, 2000.0), // Mids
    (2000.0, 6000.0),
    (6000.0, 16000.0), // Air
];
pub const BAND_COUNT: usize = BANDS.len();

/// Onset detection bands (Hz) for the drum triggers, and the shortest gap between
/// two hits (s).
const DRUMS: [(f32, f32, f32); 3] = [
    (30.0, 150.0, 0.12),    // kick
    (1000.0, 5000.0, 0.1),  // snare (wires; kicks carry little energy here)
    (7000.0, 16000.0, 0.06), // hi-hat
];
/// Onset band for tempo / beat tracking.
const ALL: (f32, f32) = (30.0, 16000.0);

/// Beat node triggers; order matches `graph::BEAT_TRIGGERS`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trigger {
    Beat,
    Bar,
    Kick,
    Snare,
    HiHat,
}

impl Trigger {
    pub fn from_index(i: usize) -> Self {
        [Self::Beat, Self::Bar, Self::Kick, Self::Snare, Self::HiHat][i.min(4)]
    }
}

pub struct Analysis {
    /// Seconds between analysis frames; frame i is centered at `i * hop_s + offset_s`.
    pub hop_s: f32,
    offset_s: f32,
    pub duration: f32,
    /// Loudness per frame and band, mapped so this song's quiet parts sit near 0 and
    /// its loudest near 1.
    pub levels: Vec<[f32; BAND_COUNT]>,
    /// Onset strength per frame (normalized, ~1 = a strong hit) for kick / snare / hat,
    /// and for all bands (the beat tracker's input).
    flux: [Vec<f32>; 3],
    onset_env: Vec<f32>,
    /// Hit times (s) for kick / snare / hat at the current sensitivity.
    pub hits: [Vec<f32>; 3],
    pub beats: Vec<f32>,
    pub bars: Vec<f32>,
    /// Tempo found in the song, and the one the beat grid uses (may be overridden).
    pub detected_bpm: f32,
    pub bpm: f32,
}

impl Analysis {
    /// Analyzes interleaved samples with `channels` channels at `rate` Hz.
    pub fn new(samples: &[f32], channels: usize, rate: u32) -> Self {
        let mono: Vec<f32> = samples.chunks_exact(channels.max(1)).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect();
        let peak = mono.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        let rate_f = rate as f32;
        let frames = if mono.len() >= FFT { (mono.len() - FFT) / HOP + 1 } else { 1 };
        let bin_hz = rate_f / FFT as f32;
        let bins = |(lo, hi): (f32, f32)| {
            let a = ((lo / bin_hz).floor() as usize).max(1);
            let b = ((hi / bin_hz).ceil() as usize).min(FFT / 2);
            a..b.max(a + 1)
        };

        let window: Vec<f32> = (0..FFT).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / FFT as f32).cos()).collect();
        let mut fft = Fft::new(FFT);
        let mut re = vec![0f32; FFT];
        let mut im = vec![0f32; FFT];
        // Two compressions: moderate for drum hits (loud hits stand out from quiet ones),
        // near-logarithmic for the beat tracker (every onset counts about equally, so
        // accented beats don't halve the tempo).
        let mut log_mag = vec![0f32; FFT / 2];
        let mut prev_log = vec![0f32; FFT / 2];
        let mut beat_mag = vec![0f32; FFT / 2];
        let mut prev_beat = vec![0f32; FFT / 2];

        let mut db = vec![[0f32; BAND_COUNT]; frames];
        let mut flux: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; frames]);
        let mut onset_env = vec![0f32; frames];
        let drum_bins = DRUMS.map(|(lo, hi, _)| bins((lo, hi)));
        let band_bins = BANDS.map(bins);
        let all_bins = bins(ALL);

        for f in 0..frames {
            let start = f * HOP;
            for i in 0..FFT {
                re[i] = mono.get(start + i).copied().unwrap_or(0.0) / peak * window[i];
                im[i] = 0.0;
            }
            fft.run(&mut re, &mut im);
            for k in 0..FFT / 2 {
                // Scaled so a full-scale sine peaks near 1 (a Hann window sums to N/2).
                let mag = (re[k] * re[k] + im[k] * im[k]).sqrt() / (FFT as f32 / 4.0);
                // Log compression makes the flux follow perceived (relative) changes.
                log_mag[k] = (1.0 + 100.0 * mag).ln();
                beat_mag[k] = (1.0 + 50000.0 * mag).ln();
            }
            for (b, range) in band_bins.iter().enumerate() {
                let power: f32 = range.clone().map(|k| (log_mag[k].exp() - 1.0).powi(2)).sum();
                db[f][b] = 10.0 * (power / range.len() as f32 + 1e-10).log10();
            }
            let rise = |range: &std::ops::Range<usize>, now: &[f32], before: &[f32]| -> f32 {
                range.clone().map(|k| (now[k] - before[k]).max(0.0)).sum::<f32>() / range.len() as f32
            };
            if f > 0 {
                for d in 0..3 {
                    flux[d][f] = rise(&drum_bins[d], &log_mag, &prev_log);
                }
                onset_env[f] = rise(&all_bins, &beat_mag, &prev_beat);
            }
            prev_log.copy_from_slice(&log_mag);
            prev_beat.copy_from_slice(&beat_mag);
        }

        // Loudness: map each band's dB range for this song to ~0..1.
        let mut levels = vec![[0f32; BAND_COUNT]; frames];
        for b in 0..BAND_COUNT {
            let col: Vec<f32> = db.iter().map(|r| r[b]).collect();
            let hi = percentile(&col, 0.995);
            let lo = percentile(&col, 0.15).min(hi - 6.0);
            for (f, row) in levels.iter_mut().enumerate() {
                row[b] = ((col[f] - lo) / (hi - lo)).clamp(0.0, 1.0);
            }
        }
        for curve in flux.iter_mut().chain([&mut onset_env]) {
            let p = percentile(curve, 0.99).max(1e-6);
            curve.iter_mut().for_each(|v| *v /= p);
        }

        let hop_s = HOP as f32 / rate_f;
        let mut a = Self {
            hop_s,
            offset_s: FFT as f32 * 0.5 / rate_f,
            duration: mono.len() as f32 / rate_f,
            levels,
            flux,
            onset_env,
            hits: Default::default(),
            beats: Vec::new(),
            bars: Vec::new(),
            detected_bpm: 120.0,
            bpm: 120.0,
        };
        a.detected_bpm = a.estimate_bpm();
        a.pick_hits(0.5);
        a.track_beats(None);
        a
    }

    fn time(&self, frame: usize) -> f32 {
        frame as f32 * self.hop_s + self.offset_s
    }

    /// Fractional frame index at time `t`.
    fn frame_at(&self, t: f32) -> f32 {
        (t - self.offset_s) / self.hop_s
    }

    /// Band loudness at time `t` (0 outside the song), linearly interpolated.
    pub fn level(&self, band: usize, t: f32) -> f32 {
        let x = self.frame_at(t);
        if x < 0.0 || x > (self.levels.len() - 1) as f32 || t > self.duration {
            return 0.0;
        }
        let i = x.floor() as usize;
        let j = (i + 1).min(self.levels.len() - 1);
        let f = x - i as f32;
        self.levels[i][band] * (1.0 - f) + self.levels[j][band] * f
    }

    /// Loudness envelope at `t` with attack / release smoothing (seconds), like a
    /// compressor's detector. Deterministic: it re-runs over the preceding window, so
    /// scrubbing and exporting give the same values as playback.
    pub fn envelope(&self, band: usize, attack: f32, release: f32, t: f32) -> f32 {
        let coef = |tau: f32| if tau <= 0.0 { 1.0 } else { 1.0 - (-self.hop_s / tau).exp() };
        let (ca, cr) = (coef(attack), coef(release));
        let span = (5.0 * attack.max(release)).clamp(0.05, 4.0);
        let end = self.frame_at(t);
        let mut i = (end - span / self.hop_s).floor();
        let mut env = 0.0;
        while i <= end {
            let x = self.level(band, self.time(0) + i * self.hop_s);
            env += (x - env) * if x > env { ca } else { cr };
            i += 1.0;
        }
        // Partial step to `t` itself, so the value moves smoothly between frames.
        let x = self.level(band, t);
        let frac = (end - (i - 1.0)).clamp(0.0, 1.0);
        env + (x - env) * if x > env { ca } else { cr } * frac
    }

    pub fn events(&self, trigger: Trigger) -> &[f32] {
        match trigger {
            Trigger::Beat => &self.beats,
            Trigger::Bar => &self.bars,
            Trigger::Kick => &self.hits[0],
            Trigger::Snare => &self.hits[1],
            Trigger::HiHat => &self.hits[2],
        }
    }

    /// Re-detects drum hits; `sensitivity` 0..1 (higher finds quieter hits).
    pub fn pick_hits(&mut self, sensitivity: f32) {
        let delta = 0.05 + (1.0 - sensitivity.clamp(0.0, 1.0)) * 0.4;
        for d in 0..3 {
            let gap = (DRUMS[d].2 / self.hop_s).round() as usize;
            let frames = pick_peaks(&self.flux[d], self.hop_s, delta, gap);
            self.hits[d] = frames.into_iter().map(|f| self.time(f)).collect();
        }
    }

    /// Builds the beat grid at the detected tempo, or at `bpm` when given; bars start on
    /// the beat whose phase (of four) lines up best with kick drums.
    pub fn track_beats(&mut self, bpm: Option<f32>) {
        self.bpm = bpm.filter(|b| (20.0..=400.0).contains(b)).unwrap_or(self.detected_bpm);
        let period = 60.0 / self.bpm / self.hop_s;
        let beats = track(&self.onset_env, period, 100.0);
        let downbeat = (0..4)
            .max_by(|&a, &b| {
                let weight = |k: usize| beats.iter().skip(k).step_by(4).map(|&f| self.flux[0][f]).sum::<f32>();
                weight(a).total_cmp(&weight(b))
            })
            .unwrap_or(0);
        self.beats = beats.iter().map(|&f| self.time(f)).collect();
        self.bars = self.beats.iter().skip(downbeat).step_by(4).copied().collect();
    }

    /// Tempo from the autocorrelation of the onset envelope, weighted toward ~120 BPM
    /// (the usual way to resolve half / double tempo ambiguity).
    fn estimate_bpm(&self) -> f32 {
        let env = &self.onset_env;
        let mean = env.iter().sum::<f32>() / env.len().max(1) as f32;
        let e: Vec<f32> = env.iter().map(|v| v - mean).collect();
        let lag_of = |bpm: f32| 60.0 / bpm / self.hop_s;
        let (lo, hi) = (lag_of(200.0).floor() as usize, lag_of(50.0).ceil() as usize);
        if e.len() <= hi + 1 {
            return 120.0;
        }
        let acf: Vec<f32> = (0..=hi + 1).map(|lag| e.iter().zip(&e[lag..]).map(|(a, b)| a * b).sum()).collect();
        let weighted = |lag: usize| {
            let bpm = 60.0 / (lag as f32 * self.hop_s);
            acf[lag] * (-0.5 * ((bpm / 120.0).log2() / 0.9).powi(2)).exp()
        };
        let best = (lo.max(1)..=hi).max_by(|&a, &b| weighted(a).total_cmp(&weighted(b))).unwrap();
        // Parabolic interpolation for a sub-frame period.
        let (y0, y1, y2) = (acf[best - 1], acf[best], acf[best + 1]);
        let denom = y0 - 2.0 * y1 + y2;
        let shift = if denom.abs() > 1e-9 { (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5) } else { 0.0 };
        60.0 / ((best as f32 + shift) * self.hop_s)
    }

    /// The peak level of `band` in each of `points` equal slices of the song (for drawing).
    pub fn overview(&self, band: usize, points: usize) -> Vec<f32> {
        (0..points)
            .map(|i| {
                let a = (i as f32 / points as f32 * self.levels.len() as f32) as usize;
                let b = (((i + 1) as f32 / points as f32 * self.levels.len() as f32) as usize).max(a + 1);
                self.levels[a.min(self.levels.len() - 1)..b.min(self.levels.len())]
                    .iter()
                    .map(|r| r[band])
                    .fold(0.0, f32::max)
            })
            .collect()
    }
}

/// Dynamic-programming beat tracker (Ellis 2007): the best chain of beats through
/// strong onsets, spaced close to `period` frames. Returns frame indices.
fn track(onset: &[f32], period: f32, tightness: f32) -> Vec<usize> {
    let n = onset.len();
    if n == 0 || period < 1.0 {
        return Vec::new();
    }
    let std = {
        let m = onset.iter().sum::<f32>() / n as f32;
        (onset.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n as f32).sqrt().max(1e-6)
    };
    let mut score = vec![0f32; n];
    let mut back = vec![usize::MAX; n];
    for i in 0..n {
        let lo = (i as f32 - 2.0 * period).round().max(0.0) as usize;
        let hi = (i as f32 - period * 0.5).round();
        let mut best = (f32::NEG_INFINITY, usize::MAX);
        if hi >= 0.0 {
            for j in lo..=(hi as usize).min(i.saturating_sub(1)) {
                let gap = (i - j) as f32 / period;
                let c = score[j] - tightness * gap.ln().powi(2);
                if c > best.0 {
                    best = (c, j);
                }
            }
        }
        let local = onset[i] / std;
        if best.0 > 0.0 {
            score[i] = local + best.0;
            back[i] = best.1;
        } else {
            score[i] = local;
        }
    }
    // End on the best-scoring frame within the last two periods, then walk back.
    let tail = n.saturating_sub((2.0 * period) as usize);
    let mut i = (tail..n).max_by(|&a, &b| score[a].total_cmp(&score[b])).unwrap();
    let mut beats = vec![i];
    while back[i] != usize::MAX {
        i = back[i];
        beats.push(i);
    }
    beats.reverse();
    beats
}

/// Onset peak picking (Böck et al. 2012): a local maximum within ±30 ms that exceeds
/// the local mean (−100 ms .. +70 ms) by `delta`, at least `gap` frames after the last.
fn pick_peaks(flux: &[f32], hop_s: f32, delta: f32, gap: usize) -> Vec<usize> {
    let w = |s: f32| (s / hop_s).round().max(1.0) as usize;
    let (max_w, pre, post) = (w(0.03), w(0.1), w(0.07));
    let mut out: Vec<usize> = Vec::new();
    for i in 0..flux.len() {
        let v = flux[i];
        let lo = i.saturating_sub(max_w);
        let hi = (i + max_w + 1).min(flux.len());
        if flux[lo..hi].iter().any(|&x| x > v) {
            continue;
        }
        let (a, b) = (i.saturating_sub(pre), (i + post + 1).min(flux.len()));
        let mean = flux[a..b].iter().sum::<f32>() / (b - a) as f32;
        if v >= mean + delta && out.last().is_none_or(|&l| i - l >= gap) {
            out.push(i);
        }
    }
    out
}

fn percentile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    let k = ((v.len() - 1) as f32 * q).round() as usize;
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

/// Iterative radix-2 complex FFT (in place) with precomputed twiddles.
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let (cos, sin) = (0..n / 2).map(|k| (-2.0 * PI * k as f32 / n as f32).sin_cos()).map(|(s, c)| (c, s)).unzip();
        Self { n, cos, sin }
    }

    fn run(&mut self, re: &mut [f32], im: &mut [f32]) {
        let n = self.n;
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + len / 2);
                    let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A drum loop at `bpm`: kick on every beat, snare on 2 and 4, hats on eighths,
    /// over a quiet synth pad, for `secs` seconds at 44.1 kHz mono.
    pub fn drum_loop(bpm: f32, secs: f32) -> Vec<f32> {
        let rate = 44100.0;
        let n = (secs * rate) as usize;
        let beat = 60.0 / bpm;
        let mut out = vec![0f32; n];
        let mut noise = 12345u32;
        let mut prev_noise = 0.0;
        let mut rnd = move || {
            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            noise as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        for (i, s) in out.iter_mut().enumerate() {
            let t = i as f32 / rate;
            let in_beat = t % beat;
            let beat_idx = (t / beat) as usize;
            let in_eighth = t % (beat / 2.0);
            // Kick: decaying 55 Hz sine with a pitch drop.
            let kick = (2.0 * PI * (55.0 + 80.0 * (-in_beat * 40.0).exp()) * in_beat).sin() * (-in_beat * 12.0).exp();
            // Snare on beats 2 and 4: noise burst.
            let snare = if beat_idx % 2 == 1 { rnd() * (-in_beat * 25.0).exp() * 0.5 } else { 0.0 };
            // Hats: high-passed noise (first difference of white noise), short decay.
            let n = rnd();
            let hat = (n - prev_noise) * (-in_eighth * 120.0).exp() * 0.15;
            prev_noise = n;
            let pad = (2.0 * PI * 220.0 * t).sin() * 0.05;
            *s = kick + snare + hat + pad;
        }
        out
    }

    #[test]
    fn fft_matches_dft() {
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; n]);
        Fft::new(n).run(&mut re, &mut im);
        for k in 0..n {
            let (mut r, mut i) = (0.0f32, 0.0f32);
            for (t, v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f32 / n as f32;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert!((re[k] - r).abs() < 1e-3 && (im[k] - i).abs() < 1e-3, "bin {k}");
        }
    }

    #[test]
    fn finds_tempo_and_beats_of_a_drum_loop() {
        for bpm in [96.0, 128.0, 140.0] {
            let a = Analysis::new(&drum_loop(bpm, 20.0), 1, 44100);
            assert!((a.bpm - bpm).abs() < 2.0, "detected {} BPM for a {bpm} BPM loop", a.bpm);
            let beat = 60.0 / bpm;
            // Beats land on kicks (multiples of the beat period), within ~2 frames.
            let inner: Vec<f32> = a.beats.iter().copied().filter(|&t| t > 2.0 && t < 18.0).collect();
            assert!(inner.len() as f32 >= 16.0 / beat - 2.0, "{bpm}: only {} beats", inner.len());
            for t in &inner {
                let err = (t / beat - (t / beat).round()).abs() * beat;
                assert!(err < 0.03, "{bpm}: beat at {t:.3} s is {err:.3} s off the grid");
            }
            // Bars start on every fourth beat.
            assert!((a.bars.len() as f32 - a.beats.len() as f32 / 4.0).abs() <= 1.0);
        }
    }

    #[test]
    fn detects_drum_hits() {
        let bpm = 120.0;
        let a = Analysis::new(&drum_loop(bpm, 10.0), 1, 44100);
        let near_beats = |hits: &[f32]| hits.iter().filter(|&&t| ((t / 0.5) - (t / 0.5).round()).abs() * 0.5 < 0.04).count();
        let kicks = &a.hits[0];
        // 20 beats; the first may be cut off by the analysis window.
        assert!((18..=21).contains(&kicks.len()), "{} kicks", kicks.len());
        assert!(near_beats(kicks) >= kicks.len() - 1);
        let snares = &a.hits[1];
        assert!((8..=12).contains(&snares.len()), "{} snares", snares.len());
        assert!(a.hits[2].len() >= 30, "{} hats", a.hits[2].len());
    }

    #[test]
    fn manual_tempo_overrides_the_grid() {
        let mut a = Analysis::new(&drum_loop(120.0, 12.0), 1, 44100);
        a.track_beats(Some(60.0));
        let gaps: Vec<f32> = a.beats.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
        assert!((mean - 1.0).abs() < 0.05, "mean beat gap {mean}");
    }

    #[test]
    fn levels_follow_loudness() {
        let rate = 44100;
        // 1 s of silence, then 1 s of a loud 80 Hz tone.
        let mut x = vec![0f32; rate];
        x.extend((0..rate).map(|i| (2.0 * PI * 80.0 * i as f32 / rate as f32).sin()));
        let a = Analysis::new(&x, 1, rate as u32);
        assert!(a.level(1, 0.5) < 0.1, "bass in silence: {}", a.level(1, 0.5));
        assert!(a.level(1, 1.5) > 0.8, "bass in the tone: {}", a.level(1, 1.5));
        // The envelope rises with the attack time constant.
        let fast = a.envelope(1, 0.01, 0.2, 1.1);
        let slow = a.envelope(1, 0.5, 0.2, 1.1);
        assert!(fast > 0.8 && slow < fast * 0.5, "fast {fast} slow {slow}");
    }
}
