//! Offline video export.
//!
//! Frames are rendered at a fixed timestep (animation time = start + frame / fps), each
//! accumulating many jittered samples (anti-aliasing, glossy reflections, optional
//! motion blur), composited with bloom at the video resolution and streamed into
//! ffmpeg as raw frames (or written as a PNG sequence).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use glam::Vec3;

use crate::camera::Camera;

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum VideoFormat {
    /// 8-bit SDR H.264 in MP4: plays everywhere.
    H264,
    /// 10-bit HDR10 H.265 in MP4 (BT.2020, PQ, with HDR10 metadata).
    Hdr10,
    /// 8-bit SDR PNG files, one per frame (no ffmpeg needed).
    PngSequence,
}

impl VideoFormat {
    pub const ALL: [VideoFormat; 3] = [Self::H264, Self::Hdr10, Self::PngSequence];

    pub fn label(self) -> &'static str {
        match self {
            Self::H264 => "MP4 H.264 (SDR)",
            Self::Hdr10 => "MP4 H.265 HDR10",
            Self::PngSequence => "PNG sequence",
        }
    }

    pub fn is_hdr(self) -> bool {
        self == Self::Hdr10
    }

    fn extension(self) -> &'static str {
        match self {
            Self::PngSequence => "",
            _ => "mp4",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum CameraMode {
    /// The camera stays where it is.
    Current,
    /// Smooth path through the keyframes.
    Keyframes,
    /// Circle the origin over the video's duration, looking at the center.
    Orbit,
}

impl CameraMode {
    pub const ALL: [CameraMode; 3] = [Self::Current, Self::Keyframes, Self::Orbit];

    pub fn label(self) -> &'static str {
        match self {
            Self::Current => "Current view",
            Self::Keyframes => "Keyframes",
            Self::Orbit => "Orbit",
        }
    }
}

pub use crate::anim::{Interp, KEY_EPSILON};

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Keyframe {
    pub time: f32,
    pub camera: Camera,
    /// Interpolation towards the next key.
    #[serde(default)]
    pub interp: Interp,
    /// Selected in the timeline (travels with the key when keys are re-sorted).
    #[serde(skip)]
    pub selected: bool,
}

impl Keyframe {
    pub fn new(time: f32, camera: Camera) -> Self {
        Self { time, camera, interp: Interp::Smooth, selected: false }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VideoSettings {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration: f32,
    /// Animation time of the first frame.
    pub start: f32,
    /// Accumulated samples per frame (anti-aliasing, glossy reflections, motion blur).
    pub samples: u32,
    /// Fraction of a frame interval the shutter is open (0 = no motion blur). Set from
    /// the processing graph's Motion Blur node when an export starts; read from projects
    /// saved before that node existed (see `Project::graphs`), never written.
    #[serde(skip_serializing)]
    pub shutter: f32,
    pub format: VideoFormat,
    /// ffmpeg quality: lower = better / bigger (CRF).
    pub crf: u32,
    pub output: String,
    pub camera: CameraMode,
    pub orbit_turns: f32,
    /// Mux the loaded song into MP4 exports.
    pub include_audio: bool,
    /// Playback loops over the scene (export) range.
    pub loop_playback: bool,
}

impl Default for VideoSettings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 30,
            duration: 10.0,
            start: 0.0,
            samples: 32,
            shutter: 0.5,
            format: VideoFormat::H264,
            crf: 16,
            output: "fractalful_video".into(),
            camera: CameraMode::Current,
            orbit_turns: 1.0,
            include_audio: true,
            loop_playback: false,
        }
    }
}

impl VideoSettings {
    pub fn frame_count(&self) -> u32 {
        ((self.duration * self.fps as f32).round() as u32).max(1)
    }

    /// Output file (or directory for PNG sequences), with the right extension. A
    /// relative name goes into Documents\fractalful\videos.
    pub fn output_path(&self) -> PathBuf {
        let mut p = crate::paths::under(&crate::paths::videos_dir(), std::path::Path::new(self.output.trim()));
        let ext = self.format.extension();
        if !ext.is_empty() && p.extension().is_none_or(|e| e != ext) {
            p.set_extension(ext);
        }
        p
    }
}

/// Video settings plus the camera keyframes; lives across exports.
#[derive(Default)]
pub struct Studio {
    pub settings: VideoSettings,
    pub keyframes: Vec<Keyframe>,
    pub status: Option<Result<String, String>>,
}

impl Studio {
    /// Inserts a key, or updates the camera of the key already at `time` (keeping its
    /// interpolation). The new key becomes the only selected one.
    pub fn add_keyframe(&mut self, time: f32, camera: Camera) {
        for k in &mut self.keyframes {
            k.selected = false;
        }
        match self.keyframes.iter_mut().find(|k| (k.time - time).abs() < KEY_EPSILON) {
            Some(k) => {
                k.camera = camera;
                k.selected = true;
            }
            None => {
                self.keyframes.push(Keyframe { selected: true, ..Keyframe::new(time, camera) });
                sort_keys(&mut self.keyframes);
            }
        }
    }

    /// The camera follows the keyframe path (in the viewport and in exports).
    pub fn camera_animated(&self) -> bool {
        self.settings.camera == CameraMode::Keyframes && !self.keyframes.is_empty()
    }

    /// Camera at animation time `t` for the configured mode. `base` is the camera when
    /// the export started (used by Current and Orbit).
    pub fn camera_at(&self, t: f32, base: &Camera) -> Camera {
        match self.settings.camera {
            CameraMode::Current => *base,
            CameraMode::Keyframes => keyframe_camera(&self.keyframes, t).unwrap_or(*base),
            CameraMode::Orbit => {
                let s = &self.settings;
                let u = ((t - s.start) / s.duration.max(1e-3)).clamp(0.0, 1.0);
                orbit_camera(base, u * s.orbit_turns * std::f32::consts::TAU)
            }
        }
    }
}

/// Sorts keys by time (stable, so a moved key lands after one it now equals).
pub fn sort_keys(keys: &mut [Keyframe]) {
    keys.sort_by(|a, b| a.time.total_cmp(&b.time));
}

/// Camera on the keyframe path at time `t` (keys sorted by time); holds the first /
/// last key outside them. Position, yaw, pitch and field of view are interpolated as
/// one curve (see `anim::sample`), with yaw unwrapped along the whole path so the
/// camera always turns the short way round.
pub fn keyframe_camera(keys: &[Keyframe], t: f32) -> Option<Camera> {
    let mut yaw = keys.first()?.camera.yaw;
    let curve: Vec<crate::anim::Key<6>> = keys
        .iter()
        .map(|k| {
            let d = (k.camera.yaw - yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            yaw += d;
            let c = &k.camera;
            (k.time, [c.pos.x, c.pos.y, c.pos.z, yaw, c.pitch, c.fov_deg], k.interp)
        })
        .collect();
    let [x, y, z, yaw, pitch, fov_deg] = crate::anim::sample(&curve, t)?;
    Some(Camera { pos: Vec3::new(x, y, z), yaw, pitch: pitch.clamp(-1.55, 1.55), fov_deg })
}

/// `base` rotated `angle` radians around the vertical axis through the origin, aimed
/// at the origin.
fn orbit_camera(base: &Camera, angle: f32) -> Camera {
    let (s, c) = angle.sin_cos();
    let p = base.pos;
    let pos = Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
    let to = -pos.normalize_or_zero();
    Camera { pos, yaw: to.x.atan2(to.z), pitch: to.y.clamp(-1.0, 1.0).asin(), fov_deg: base.fov_deg }
}

enum Sink {
    Ffmpeg { child: Child, path: PathBuf },
    Png(PathBuf),
}

/// An export in progress: one frame is rendered per app frame.
pub struct VideoJob {
    pub settings: VideoSettings,
    pub keyframes: Vec<Keyframe>,
    pub base_camera: Camera,
    pub frame: u32,
    pub total: u32,
    sink: Sink,
    started: Instant,
}

impl VideoJob {
    /// `soundtrack`: song file and the song time at the video's first frame.
    pub fn start(studio: &Studio, base_camera: Camera, soundtrack: Option<(PathBuf, f32)>) -> Result<Self, String> {
        let s = studio.settings.clone();
        let soundtrack = soundtrack.filter(|_| s.include_audio);
        if s.camera == CameraMode::Keyframes && studio.keyframes.is_empty() {
            return Err("camera mode is Keyframes but there are no keyframes".into());
        }
        // yuv420p needs even dimensions.
        if s.format != VideoFormat::PngSequence && (s.width % 2 == 1 || s.height % 2 == 1) {
            return Err("width and height must be even for MP4 output".into());
        }
        let path = s.output_path();
        let sink = match s.format {
            VideoFormat::PngSequence => {
                std::fs::create_dir_all(&path).map_err(|e| format!("can't create {}: {e}", path.display()))?;
                Sink::Png(path)
            }
            fmt => {
                // ffmpeg doesn't create folders (e.g. Documents\fractalful\videos on first use).
                if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                    std::fs::create_dir_all(dir).map_err(|e| format!("can't create {}: {e}", dir.display()))?;
                }
                let child = ffmpeg_command(&s, fmt, &path, soundtrack.as_ref())
                    .spawn()
                    .map_err(|e| {
                        format!("couldn't start ffmpeg ({e}): put ffmpeg.exe next to fractalful.exe, install it on PATH, or use PNG sequence")
                    })?;
                Sink::Ffmpeg { child, path }
            }
        };
        Ok(Self {
            total: s.frame_count(),
            settings: s,
            keyframes: studio.keyframes.clone(),
            base_camera,
            frame: 0,
            sink,
            started: Instant::now(),
        })
    }

    /// Animation time for sample `sample` of `samples` of the current frame: spread
    /// over the open shutter for motion blur, otherwise the frame's start time.
    pub fn sample_time(&self, sample: u32, samples: u32) -> f32 {
        let s = &self.settings;
        let offset = if samples > 1 { (sample as f32 + 0.5) / samples as f32 * s.shutter } else { 0.0 };
        s.start + (self.frame as f32 + offset) / s.fps as f32
    }

    pub fn camera_at(&self, t: f32) -> Camera {
        let studio = Studio { settings: self.settings.clone(), keyframes: self.keyframes.clone(), ..Default::default() };
        studio.camera_at(t, &self.base_camera)
    }

    /// Whether the camera moves during the video (so per-sample camera blur matters).
    pub fn camera_moves(&self) -> bool {
        self.settings.camera != CameraMode::Current
    }

    /// `pixels` is RGBA8 (SDR) or RGBA16 little-endian (HDR10), top row first.
    pub fn push_frame(&mut self, pixels: &[u8]) -> Result<(), String> {
        let (w, h) = (self.settings.width, self.settings.height);
        match &mut self.sink {
            Sink::Ffmpeg { child, .. } => child
                .stdin
                .as_mut()
                .ok_or("ffmpeg stdin closed")?
                .write_all(pixels)
                .map_err(|e| format!("ffmpeg stopped accepting frames ({e}); see its output in the console"))?,
            Sink::Png(dir) => {
                let path = dir.join(format!("frame_{:05}.png", self.frame));
                crate::renderer::save_png(&path, [w, h], pixels).map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        self.frame += 1;
        Ok(())
    }

    pub fn done(&self) -> bool {
        self.frame >= self.total
    }

    /// Estimated seconds remaining, from the average time per frame so far.
    pub fn eta(&self) -> Option<f32> {
        (self.frame > 0).then(|| {
            let per = self.started.elapsed().as_secs_f32() / self.frame as f32;
            per * (self.total - self.frame) as f32
        })
    }

    /// Closes the stream and waits for ffmpeg; returns a status message.
    pub fn finish(self) -> Result<String, String> {
        let secs = self.started.elapsed().as_secs_f32();
        match self.sink {
            Sink::Png(dir) => Ok(format!("wrote {} frames to {} in {secs:.0} s", self.frame, dir.display())),
            Sink::Ffmpeg { mut child, path } => {
                drop(child.stdin.take()); // EOF: ffmpeg finalizes the file
                let status = child.wait().map_err(|e| e.to_string())?;
                if status.success() {
                    Ok(format!("wrote {} ({} frames) in {secs:.0} s", path.display(), self.frame))
                } else {
                    Err(format!("ffmpeg failed ({status}); see its output in the console"))
                }
            }
        }
    }

    /// Stops early. For MP4 the frames so far are still finalized into a playable file.
    pub fn cancel(self) -> String {
        let frames = self.frame;
        match self.finish() {
            Ok(_) | Err(_) => format!("cancelled after {frames} frames"),
        }
    }
}

/// ffmpeg next to our exe if it's shipped there, else whatever is on PATH.
fn ffmpeg_exe() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.join("ffmpeg.exe")))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("ffmpeg"))
}

fn ffmpeg_command(s: &VideoSettings, fmt: VideoFormat, path: &std::path::Path, soundtrack: Option<&(PathBuf, f32)>) -> Command {
    let mut cmd = Command::new(ffmpeg_exe());
    cmd.args(["-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo"])
        .args(["-pix_fmt", if fmt.is_hdr() { "rgba64le" } else { "rgba" }])
        .args(["-s", &format!("{}x{}", s.width, s.height)])
        .args(["-r", &s.fps.to_string()])
        .args(["-i", "-"]);
    if let Some((song, song_start)) = soundtrack {
        // Start reading the song where the video starts; if the video starts before
        // the song, delay the song instead.
        if *song_start >= 0.0 {
            cmd.args(["-ss", &format!("{song_start:.4}")]);
        } else {
            cmd.args(["-itsoffset", &format!("{:.4}", -song_start)]);
        }
        cmd.arg("-i").arg(song);
        // Pad with silence and stop with the video, whichever is longer.
        cmd.args(["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "320k", "-af", "apad", "-shortest"]);
    }
    let crf = s.crf.to_string();
    if fmt.is_hdr() {
        // Input is already BT.2020 + PQ encoded (the compositor's HDR10 path).
        cmd.args(["-vf", "scale=out_color_matrix=bt2020nc:out_range=tv"])
            .args(["-c:v", "libx265", "-preset", "medium", "-crf", &crf, "-pix_fmt", "yuv420p10le"])
            .args(["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"])
            .args([
                "-x265-params",
                "hdr10=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:\
                 master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400",
            ])
            .args(["-tag:v", "hvc1"]);
    } else {
        cmd.args(["-vf", "scale=out_color_matrix=bt709:out_range=tv"])
            .args(["-c:v", "libx264", "-preset", "slow", "-crf", &crf, "-pix_fmt", "yuv420p"])
            .args(["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"])
            .args(["-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709"]);
    }
    cmd.args(["-movflags", "+faststart"]).arg(path).stdin(Stdio::piped()).stdout(Stdio::null());
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(x: f32, yaw: f32) -> Camera {
        Camera { pos: Vec3::new(x, 0.0, -3.0), yaw, pitch: 0.0, fov_deg: 60.0 }
    }

    #[test]
    fn keyframes_interpolate_through_keys() {
        let keys = [
            Keyframe::new(0.0, cam(0.0, 0.0)),
            Keyframe::new(1.0, cam(1.0, 0.5)),
            Keyframe::new(3.0, cam(3.0, 1.0)),
        ];
        for k in &keys {
            let c = keyframe_camera(&keys, k.time).unwrap();
            assert!((c.pos - k.camera.pos).length() < 1e-4 && (c.yaw - k.camera.yaw).abs() < 1e-4);
        }
        let mid = keyframe_camera(&keys, 0.5).unwrap();
        assert!(mid.pos.x > 0.0 && mid.pos.x < 1.0);
        assert_eq!(keyframe_camera(&keys, -5.0).unwrap().pos, keys[0].camera.pos);
        assert_eq!(keyframe_camera(&keys, 9.0).unwrap().pos, keys[2].camera.pos);
    }

    #[test]
    fn yaw_takes_the_short_way() {
        let keys = [Keyframe::new(0.0, cam(0.0, 3.0)), Keyframe::new(1.0, cam(0.0, -3.0))];
        let mid = keyframe_camera(&keys, 0.5).unwrap();
        // 3.0 -> -3.0 is 0.28 rad across +-pi, not 6 rad the long way round.
        assert!(mid.yaw.cos() < -0.95, "yaw {} should be near pi", mid.yaw);
        // Also across several keys: the unwrap carries on along the whole path.
        let keys = [Keyframe::new(0.0, cam(0.0, 2.0)), Keyframe::new(1.0, cam(0.0, -2.5)), Keyframe::new(2.0, cam(0.0, -0.5))];
        for i in 0..=20 {
            let a = keyframe_camera(&keys, i as f32 * 0.05).unwrap().yaw;
            let b = keyframe_camera(&keys, (i + 1) as f32 * 0.05).unwrap().yaw;
            assert!((a - b).abs() < 0.5, "yaw jumps from {a} to {b}");
        }
    }

    /// Smooth keeps the speed continuous through a key even with uneven spacing, and
    /// a straight run at constant speed stays at constant speed.
    #[test]
    fn smooth_speed_is_continuous() {
        let keys = [Keyframe::new(0.0, cam(0.0, 0.0)), Keyframe::new(1.0, cam(1.0, 0.0)), Keyframe::new(4.0, cam(4.0, 0.0))];
        let x = |t: f32| keyframe_camera(&keys, t).unwrap().pos.x;
        for t in [0.2, 0.9, 1.0, 1.1, 2.5, 3.7] {
            let v = (x(t + 1e-3) - x(t - 1e-3)) / 2e-3;
            assert!((v - 1.0).abs() < 0.02, "speed {v} at t = {t}");
        }
    }

    #[test]
    fn interpolation_modes() {
        let mut keys = [Keyframe::new(0.0, cam(0.0, 0.0)), Keyframe::new(1.0, cam(2.0, 0.0))];
        keys[0].interp = Interp::Constant;
        assert_eq!(keyframe_camera(&keys, 0.9).unwrap().pos.x, 0.0);
        keys[0].interp = Interp::Linear;
        assert!((keyframe_camera(&keys, 0.25).unwrap().pos.x - 0.5).abs() < 1e-5);
        keys[0].interp = Interp::Ease;
        let x = |t: f32| keyframe_camera(&keys, t).unwrap().pos.x;
        assert!(x(0.01) < 0.01 && (x(0.5) - 1.0).abs() < 1e-5, "eases out of the first key");
    }

    /// Auto-clamped tangents: a key that is a peak isn't overshot.
    #[test]
    fn smooth_does_not_overshoot_a_peak() {
        let keys = [Keyframe::new(0.0, cam(0.0, 0.0)), Keyframe::new(1.0, cam(1.0, 0.0)), Keyframe::new(1.2, cam(0.0, 0.0))];
        for i in 0..=120 {
            let x = keyframe_camera(&keys, i as f32 * 0.01).unwrap().pos.x;
            assert!(x <= 1.0 + 1e-4, "overshoot {x}");
        }
    }

    #[test]
    fn orbit_keeps_distance_and_looks_at_origin() {
        let base = cam(1.0, 0.0);
        for a in [0.0f32, 1.0, 2.5] {
            let c = orbit_camera(&base, a);
            assert!((c.pos.length() - base.pos.length()).abs() < 1e-4);
            let (fwd, _, _) = c.basis();
            assert!(fwd.dot(-c.pos.normalize()) > 0.999);
        }
    }
}
