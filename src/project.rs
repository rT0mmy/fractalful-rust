//! Saving the scene to disk as JSON: named projects in `Documents\fractalful\projects`
//! (see `paths`), and the session (autosaved, restored at the next launch) in
//! `%APPDATA%\fractalful`.
//!
//! A project holds everything needed to get back to a shot: the node graphs (material,
//! world, processing), render settings, camera, animation time, and the video studio
//! (settings + keyframes).
//! Display-specific settings (output mode, nits) are left to the machine.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::Settings;
use crate::audio::AudioSettings;
use crate::camera::Camera;
use crate::graph::{Graph, Graphs};
use crate::video::{Keyframe, VideoSettings};

#[derive(Serialize, Deserialize)]
pub struct Project {
    /// The material graph (saved as `graph` from before there were three).
    pub graph: Graph,
    /// The world and processing graphs. Projects saved before they existed have none:
    /// `Project::graphs` builds them from the old render settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world: Option<Graph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post: Option<Graph>,
    #[serde(default)]
    pub settings: Settings,
    pub camera: Camera,
    #[serde(default)]
    pub anim_time: f32,
    #[serde(default)]
    pub video: VideoSettings,
    #[serde(default)]
    pub keyframes: Vec<Keyframe>,
    /// The song (by path) and its playback / analysis settings.
    #[serde(default)]
    pub audio: AudioSettings,
}

impl Project {
    pub fn new(graphs: &Graphs, settings: Settings, camera: Camera) -> Self {
        Self {
            graph: graphs.material.clone(),
            world: Some(graphs.world.clone()),
            post: Some(graphs.post.clone()),
            settings,
            camera,
            anim_time: 0.0,
            video: VideoSettings::default(),
            keyframes: Vec::new(),
            audio: AudioSettings::default(),
        }
    }

    /// The scene's graphs, repaired for the current node definitions. A project from
    /// before the world / processing graphs gets ones matching its old settings (sky,
    /// light, fog, bloom, motion blur), so it looks the same.
    pub fn graphs(&self) -> Graphs {
        let look = self.settings.legacy_look(self.video.shutter);
        let mut g = Graphs {
            material: self.graph.clone(),
            world: self.world.clone().unwrap_or_else(|| Graph::world_from(&look)),
            post: self.post.clone().unwrap_or_else(|| Graph::post_from(&look)),
        };
        g.iter_mut().for_each(Graph::repair);
        g
    }
}

/// The project plus window / editor state, restored on the next launch.
#[derive(Serialize, Deserialize)]
pub struct Session {
    pub project: Project,
    /// Project (or preset) name shown in the title bar and offered when saving.
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub window_size: Option<[u32; 2]>,
    #[serde(default)]
    pub show_editor: bool,
    #[serde(default)]
    pub editor_detached: bool,
    #[serde(default)]
    pub maximized: bool,
    /// The saved project this session was opened from / last saved to, if any.
    #[serde(default)]
    pub file: Option<String>,
}

pub fn session_path() -> PathBuf {
    let dir = std::env::var_os("APPDATA").map_or_else(|| PathBuf::from(".fractalful"), |d| PathBuf::from(d).join("fractalful"));
    dir.join("session.json")
}

/// Copies the session file aside (e.g. `session-crashed.json`) before recovery
/// changes it. Returns the copy's path.
pub fn back_up_session(suffix: &str) -> Option<PathBuf> {
    let to = session_path().with_file_name(format!("session-{suffix}.json"));
    std::fs::copy(session_path(), &to).ok().map(|_| to)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LastRun {
    Clean,
    /// Crashed, hung and was killed (End task), or the machine went down. Holds when
    /// that run started, if known.
    Unclean(Option<SystemTime>),
    /// Another instance is still running (and owns the session).
    StillRunning,
}

/// Marks a session in progress: a file created at launch and removed on a clean exit.
/// It's held open without sharing, so Windows frees it the moment the process dies
/// however that happens, and a leftover file that opens fine means the last run
/// didn't exit cleanly. It holds the run's start time (Unix seconds).
pub struct RunLock {
    path: PathBuf,
    file: Option<std::fs::File>,
}

impl RunLock {
    pub fn path() -> PathBuf {
        session_path().with_file_name("running.lock")
    }

    pub fn claim(path: PathBuf) -> (Option<RunLock>, LastRun) {
        use std::io::Write;
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let existed = path.exists();
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        // Share delete only, so Drop can remove the file while it's still open.
        #[cfg(windows)]
        std::os::windows::fs::OpenOptionsExt::share_mode(&mut options, 4 /* FILE_SHARE_DELETE */);
        match options.open(&path) {
            Ok(mut file) => {
                let last = if existed {
                    let mut text = String::new();
                    let _ = std::io::Read::read_to_string(&mut file, &mut text);
                    let secs = text.trim().parse::<u64>().ok();
                    LastRun::Unclean(secs.map(|s| SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(s)))
                } else {
                    LastRun::Clean
                };
                let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                let _ = file.set_len(0).and_then(|()| std::io::Seek::rewind(&mut file)).and_then(|()| write!(file, "{now}"));
                (Some(RunLock { path, file: Some(file) }), last)
            }
            // ERROR_SHARING_VIOLATION: another instance holds it.
            Err(e) if e.raw_os_error() == Some(32) => (None, LastRun::StillRunning),
            Err(e) => {
                eprintln!("run lock {}: {e}", path.display());
                (None, LastRun::Clean)
            }
        }
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn project_path(name: &str) -> PathBuf {
    crate::paths::projects_dir().join(format!("{}.json", file_stem(name)))
}

/// `name` made safe as a Windows file name.
pub fn file_stem(name: &str) -> String {
    let s: String = name.trim().chars().map(|c| if r#"<>:"/\|?*"#.contains(c) || c.is_control() { '_' } else { c }).collect();
    let s = s.trim_end_matches(['.', ' ']).to_string();
    if s.is_empty() { "untitled".into() } else { s }
}

pub fn to_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).expect("project types always serialize")
}

/// Writes via a temporary file, so a crash mid-write never leaves a truncated file.
pub fn write(path: &Path, json: &str) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("could not save {}: {e}", path.display());
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(fail)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).map_err(fail)?;
    std::fs::rename(&tmp, path).map_err(fail)
}

pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("could not open {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} is not a valid project: {e}", path.display()))
}

/// Saved project names with when they were last saved, most recent first.
pub fn list() -> Vec<(String, SystemTime)> {
    let Ok(dir) = std::fs::read_dir(crate::paths::projects_dir()) else { return Vec::new() };
    let mut found: Vec<(SystemTime, String)> = dir
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| {
            let modified = e.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            Some((modified, e.path().file_stem()?.to_string_lossy().into_owned()))
        })
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(t, name)| (name, t)).collect()
}

/// "just now", "5 min ago", "3 h ago", "2 days ago".
pub fn ago(t: SystemTime) -> String {
    let s = SystemTime::now().duration_since(t).map_or(0, |d| d.as_secs());
    match s {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", s / 60),
        3600..86400 => format!("{} h ago", s / 3600),
        86400..172800 => "yesterday".into(),
        _ => format!("{} days ago", s / 86400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_round_trips_every_preset() {
        for name in crate::graph::presets() {
            let graphs = Graphs { material: crate::graph::preset(name).0, ..Graphs::default() };
            let mut project = Project::new(&graphs, Settings::default(), Camera::new());
            project.anim_time = 1.5;
            project.keyframes = vec![Keyframe::new(2.0, Camera::new())];
            let json = to_json(&project);
            let back: Project = serde_json::from_str(&json).unwrap();
            assert!(back.graphs().same_content(&graphs), "{name} changed in a save/load round trip");
            assert_eq!(to_json(&back), json, "{name}");
        }
    }

    #[test]
    fn missing_fields_take_defaults() {
        let (graph, _) = crate::graph::preset("Mandelbox");
        let json = format!(r#"{{"graph": {}, "camera": {}}}"#, to_json(&graph), to_json(&Camera::new()));
        let p: Project = serde_json::from_str(&json).unwrap();
        assert_eq!(p.settings.iterations, Settings::default().iterations);
        assert!(p.keyframes.is_empty());
    }

    /// A project saved before the world / processing graphs keeps its look: its sky,
    /// light, fog, bloom and motion blur settings become nodes.
    #[test]
    fn old_projects_get_matching_graphs() {
        use crate::graph::{NodeKind, WORLD_FOG, WORLD_SUN_ELEVATION, WORLD_SUN_STRENGTH};
        let (graph, _) = crate::graph::preset("Mandelbox");
        let json = format!(
            r#"{{"graph": {}, "camera": {}, "settings": {{"sky_top": [0.5, 0.25, 0.125], "sky_solid": true, "fog": 0.1,
                "light_intensity": 2.0, "light_pitch": 0.5, "bloom": {{"enabled": false, "intensity": 2.0}}}},
                "video": {{"shutter": 0.25}}}}"#,
            to_json(&graph),
            to_json(&Camera::new())
        );
        let p: Project = serde_json::from_str(&json).unwrap();
        let g = p.graphs();
        let sky = g.world.find(NodeKind::SkyGradient).unwrap();
        let o = sky.value_offset(3);
        assert_eq!(&sky.values[o..o + 3], &[0.5, 0.25, 0.125], "solid sky: the ground is the top color");
        let out = g.world.find(NodeKind::WorldOutput).unwrap();
        assert_eq!(out.value(WORLD_FOG), 0.1);
        assert!((out.value(WORLD_SUN_STRENGTH) - 2.6).abs() < 1e-6);
        assert!((out.value(WORLD_SUN_ELEVATION) - 0.5f32.to_degrees()).abs() < 1e-4);
        let bloom = g.post.find(NodeKind::Bloom).unwrap();
        assert!(bloom.muted && bloom.value(1) == 2.0);
        assert_eq!(g.post.motion_blur_shutter(), 0.25);
        // Saved again, the graphs are stored and the old settings are not.
        let mut p = p;
        p.world = Some(g.world.clone());
        p.post = Some(g.post.clone());
        let json = to_json(&p);
        assert!(json.contains("\"world\"") && !json.contains("sky_top") && !json.contains("shutter"), "{json}");
    }

    #[test]
    fn run_lock_detects_unclean_exits() {
        let dir = std::env::temp_dir().join(format!("fractalful-runlock-{}", std::process::id()));
        let path = dir.join("running.lock");
        let _ = std::fs::remove_file(&path);

        let (lock, last) = RunLock::claim(path.clone());
        assert_eq!(last, LastRun::Clean);
        #[cfg(windows)]
        assert_eq!(RunLock::claim(path.clone()).1, LastRun::StillRunning);
        drop(lock);
        assert!(!path.exists(), "a clean exit removes the lock");

        // A killed process leaves the file behind, but its handle is gone.
        std::fs::write(&path, "1700000000").unwrap();
        let (lock, last) = RunLock::claim(path.clone());
        let started = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        assert_eq!(last, LastRun::Unclean(Some(started)), "the crashed run's start time is read back");
        drop(lock);

        // An older lock format (empty) still counts as unclean.
        std::fs::write(&path, "").unwrap();
        assert_eq!(RunLock::claim(path.clone()).1, LastRun::Unclean(None));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_stems_are_safe() {
        assert_eq!(file_stem("  my: shot?  "), "my_ shot_");
        assert_eq!(file_stem("..."), "untitled");
        assert_eq!(file_stem("a/b\\c"), "a_b_c");
    }
}
