//! Music: load a song (FLAC / MP3 / WAV), analyze it, play it in sync with the
//! animation clock, and evaluate the audio nodes (Audio Level, Beat) from it.
//!
//! Song time = animation time − `offset`, so the song can start later in the
//! animation. While the song plays, the animation clock follows the audio device's
//! clock (what you see matches what you hear); scrubbing the animation time seeks it.

pub mod analysis;
pub mod decode;
mod player;
pub mod ui;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use analysis::{Analysis, Trigger};
use decode::Song;
use player::Player;

use crate::graph::{Graph, LIVE_SLOTS, NodeKind};

/// Saved with projects.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    pub path: Option<PathBuf>,
    /// Animation time at which the song starts.
    pub offset: f32,
    pub volume: f32,
    pub muted: bool,
    /// Manual tempo for the beat grid (None = detected).
    pub bpm: Option<f32>,
    /// Drum-hit detection sensitivity, 0..1.
    pub sensitivity: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self { path: None, offset: 0.0, volume: 0.8, muted: false, bpm: None, sensitivity: 0.5 }
    }
}

pub struct Track {
    pub name: String,
    pub analysis: Analysis,
    /// None when there's no usable output device (the song still drives the nodes).
    player: Option<Player>,
}

type Loaded = Result<(Song, Analysis), String>;

#[derive(Default)]
pub struct Audio {
    pub settings: AudioSettings,
    pub track: Option<Track>,
    loading: Option<(PathBuf, Receiver<Loaded>)>,
    picking: Option<Receiver<Option<PathBuf>>>,
    /// A song just loaded: the UI opens the Audio panel once.
    pub reveal: bool,
}

impl Audio {
    /// Starts decoding + analyzing `path` on a background thread.
    pub fn load(&mut self, path: PathBuf) {
        let (tx, rx) = channel();
        let p = path.clone();
        std::thread::spawn(move || {
            let result = decode::decode(&p).map(|song| {
                let analysis = Analysis::new(&song.samples, song.channels, song.rate);
                (song, analysis)
            });
            let _ = tx.send(result);
        });
        self.loading = Some((path, rx));
    }

    pub fn loading(&self) -> Option<&Path> {
        self.loading.as_ref().map(|(p, _)| p.as_path())
    }

    pub fn unload(&mut self) {
        self.track = None;
        self.loading = None;
        self.settings.path = None;
    }

    /// Opens the system file dialog without blocking rendering.
    pub fn pick_file(&mut self) {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let file = rfd::FileDialog::new()
                .set_title("Open a song")
                .add_filter("Audio (FLAC, MP3, WAV)", &decode::EXTENSIONS)
                .pick_file();
            let _ = tx.send(file);
        });
        self.picking = Some(rx);
    }

    /// Applies saved settings, (re)loading the song if it changed.
    pub fn apply_settings(&mut self, settings: AudioSettings) {
        let same_song = settings.path == self.settings.path && (self.track.is_some() || self.loading.is_some());
        self.settings = settings;
        if same_song {
            self.reanalyze();
            return;
        }
        self.track = None;
        self.loading = None;
        if let Some(path) = self.settings.path.clone() {
            self.load(path);
        }
    }

    /// Re-detects hits and the beat grid with the current sensitivity / tempo.
    pub fn reanalyze(&mut self) {
        if let Some(t) = self.track.as_mut() {
            t.analysis.pick_hits(self.settings.sensitivity);
            t.analysis.track_beats(self.settings.bpm);
        }
    }

    /// Call once per frame: finishes loads and file picks. Returns a message to show.
    pub fn poll(&mut self) -> Option<Result<String, String>> {
        if let Some(rx) = &self.picking {
            if let Ok(file) = rx.try_recv() {
                self.picking = None;
                if let Some(path) = file {
                    self.load(path);
                }
            }
        }
        let (path, rx) = self.loading.as_ref()?;
        let result = rx.try_recv().ok()?;
        let path = path.clone();
        self.loading = None;
        Some(result.map(|(song, analysis)| {
            let song = Arc::new(song);
            let name = path.file_name().map_or("song".into(), |n| n.to_string_lossy().into_owned());
            let player = Player::new(song.clone()).map_err(|e| eprintln!("audio: {e}")).ok();
            self.track = Some(Track { name: name.clone(), analysis, player });
            self.settings.path = Some(path);
            self.reveal = true;
            self.reanalyze();
            let a = &self.track.as_ref().unwrap().analysis;
            let mut msg = format!("Loaded {name}: {:.0}:{:02}, {:.0} BPM", a.duration as u32 / 60, a.duration as u32 % 60, a.bpm);
            if self.track.as_ref().unwrap().player.is_none() {
                msg += " (no audio output device: silent)";
            }
            msg
        }))
    }

    fn player(&self) -> Option<&Player> {
        self.track.as_ref()?.player.as_ref()
    }

    /// Before the UI each frame: while the song plays, nudge the animation clock toward
    /// the song's clock. Big differences are fixed by seeking in `drive` instead.
    pub fn follow(&self, anim_time: &mut f32) {
        if let Some(p) = self.player().filter(|p| p.is_playing()) {
            let err = p.time() + self.settings.offset - *anim_time;
            if err.abs() < 0.25 {
                *anim_time += err * 0.15;
            }
        }
    }

    /// After the UI each frame: play or pause with the animation, and seek after the
    /// animation time jumped (scrubbing, rewind, keyframe "go to").
    pub fn drive(&self, anim_time: f32, play: bool) {
        let Some(p) = self.player() else { return };
        p.set_volume(if self.settings.muted { 0.0 } else { self.settings.volume });
        if !play {
            p.set_playing(false);
            return;
        }
        let song_t = anim_time - self.settings.offset;
        if !p.is_playing() || (p.time() - song_t).abs() > 0.25 {
            p.seek(song_t);
        }
        p.set_playing(true);
    }

    /// Outputs of every live node (audio, Wiggle) at animation time `t`, laid out by
    /// `Graph::live_slots`. Audio nodes give zeros without a song.
    pub fn live_values(&self, g: &Graph, t: f32) -> [f32; LIVE_SLOTS] {
        let mut out = [0.0; LIVE_SLOTS];
        let Ok(slots) = g.live_slots() else { return out };
        let song_t = t - self.settings.offset;
        for (id, first) in slots {
            let Some(n) = g.node(id) else { continue };
            if n.kind == NodeKind::Wiggle {
                out[first] = crate::graph::eval::wiggle(n, t);
                continue;
            }
            let Some(track) = &self.track else { continue };
            let a = &track.analysis;
            let v = |j: usize| n.value(j);
            match n.kind {
                NodeKind::AudioLevel => {
                    let level = a.envelope(n.choice(0).min(analysis::BAND_COUNT - 1), v(1), v(2), song_t);
                    out[first] = level * v(3);
                }
                NodeKind::Beat => {
                    let events = a.events(Trigger::from_index(n.choice(0)));
                    let every = v(1).round().max(1.0) as usize;
                    let o = beat_outputs(events, every, v(2), song_t - v(3));
                    out[first..first + 4].copy_from_slice(&o);
                }
                _ => {}
            }
        }
        out
    }

    /// For video export: the song file and the song time where the video starts.
    pub fn soundtrack(&self, video_start: f32) -> Option<(PathBuf, f32)> {
        self.track.as_ref()?;
        Some((self.settings.path.clone()?, video_start - self.settings.offset))
    }
}

/// Beat node outputs [pulse, count, phase, glide] at time `t`, from sorted event times
/// (every `every`-th one counts).
pub fn beat_outputs(events: &[f32], every: usize, decay: f32, t: f32) -> [f32; 4] {
    let passed = events.partition_point(|&e| e <= t);
    let count = passed.div_ceil(every);
    if count == 0 {
        return [0.0; 4];
    }
    let prev = events[(count - 1) * every];
    // After the last event, keep the average spacing going.
    let next = events.get(count * every).copied().unwrap_or_else(|| {
        let span = events.last().unwrap() - events[0];
        let gap = if events.len() > 1 { span / (events.len() - 1) as f32 } else { 1.0 };
        prev + gap * every as f32
    });
    let since = t - prev;
    let pulse = if decay > 0.0 { (-4.6 * since / decay).exp() } else { (since <= 0.0) as u32 as f32 };
    let phase = (since / (next - prev).max(1e-4)).clamp(0.0, 1.0);
    let x = if decay > 0.0 { (since / decay).clamp(0.0, 1.0) } else { 1.0 };
    let glide = (count - 1) as f32 + x * x * (3.0 - 2.0 * x);
    [pulse, count as f32, phase, glide]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_outputs_count_pulse_and_glide() {
        let beats = [1.0, 1.5, 2.0, 2.5, 3.0];
        assert_eq!(beat_outputs(&beats, 1, 0.2, 0.5), [0.0; 4]);
        let [pulse, count, phase, glide] = beat_outputs(&beats, 1, 0.2, 1.0);
        assert_eq!((count, glide), (1.0, 0.0));
        assert!((pulse - 1.0).abs() < 1e-6 && phase == 0.0);
        let [pulse, count, phase, glide] = beat_outputs(&beats, 1, 0.2, 1.25);
        assert_eq!(count, 1.0);
        assert!(pulse < 0.01, "pulse fades out within the decay");
        assert!((phase - 0.5).abs() < 1e-6 && glide == 1.0);
        // Every 2nd beat: counts 1.0, 2.0, 3.0 at 1.0, 2.0, 3.0.
        assert_eq!(beat_outputs(&beats, 2, 0.2, 1.7)[1], 1.0);
        assert_eq!(beat_outputs(&beats, 2, 0.2, 2.1)[1], 2.0);
        // Past the end the grid continues at the average spacing.
        let [_, count, phase, _] = beat_outputs(&beats, 1, 0.2, 3.25);
        assert_eq!(count, 5.0);
        assert!((phase - 0.5).abs() < 1e-6);
    }

    #[test]
    fn live_values_follow_the_graph_layout() {
        let mut g = Graph::new();
        let level = g.add(NodeKind::AudioLevel, [0.0; 2]);
        let beat = g.add(NodeKind::Beat, [0.0; 2]);
        let slots: std::collections::HashMap<_, _> = g.live_slots().unwrap().into_iter().collect();
        assert_eq!((slots[&level], slots[&beat]), (0, 1));

        let x = analysis::tests::drum_loop(120.0, 8.0);
        let analysis = Analysis::new(&x, 1, 44100);
        let mut audio = Audio::default();
        audio.track = Some(Track { name: "loop".into(), analysis, player: None });
        audio.settings.offset = 1.0;
        let a = audio.live_values(&g, 4.0);
        assert!(a[0] > 0.0, "a drum loop has bass");
        assert!(a[2] >= 5.0, "beats counted by 3 s of song: {}", a[2]);
        // Before the song starts, nothing has happened yet.
        assert_eq!(audio.live_values(&g, 0.5)[2], 0.0);
    }
}
