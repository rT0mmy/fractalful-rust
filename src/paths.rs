//! Where the app keeps files, independent of the working directory (an installed copy
//! may be started from anywhere, and can't write next to its exe in Program Files).
//!
//! - `Documents\fractalful\` holds what the user makes: projects, screenshots, videos.
//!   `FRACTALFUL_DATA_DIR` overrides it (portable installs, test runs).
//! - The autosaved session lives in `%APPDATA%\fractalful` (see `project::session_path`).

use std::path::{Path, PathBuf};

/// The user's fractalful folder (not created until something is saved).
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FRACTALFUL_DATA_DIR") {
        return dir.into();
    }
    documents_dir().unwrap_or_else(|| PathBuf::from(".")).join("fractalful")
}

pub fn projects_dir() -> PathBuf {
    data_dir().join("projects")
}

pub fn screenshots_dir() -> PathBuf {
    data_dir().join("screenshots")
}

pub fn videos_dir() -> PathBuf {
    data_dir().join("videos")
}

/// `path` as given if absolute, else under `base`.
pub fn under(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { base.join(path) }
}

/// Older versions kept projects in `projects\` under the working directory. The first
/// time the new projects folder is missing, copies them over (originals are left in
/// place). Returns (copied, from) when anything was copied.
pub fn migrate_old_projects() -> Option<(usize, PathBuf)> {
    let new = projects_dir();
    let old = std::env::current_dir().ok()?.join("projects");
    if new.exists() || !old.is_dir() || same_dir(&old, &new) {
        return None;
    }
    let files: Vec<PathBuf> = std::fs::read_dir(&old)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    if files.is_empty() {
        return None;
    }
    std::fs::create_dir_all(&new).ok()?;
    let copied = files.iter().filter(|f| f.file_name().is_some_and(|n| std::fs::copy(f, new.join(n)).is_ok())).count();
    (copied > 0).then_some((copied, old))
}

fn same_dir(a: &Path, b: &Path) -> bool {
    matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

/// The Documents known folder (follows OneDrive / moved-folder redirection).
#[cfg(windows)]
fn documents_dir() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::Com::CoTaskMemFree;
    use windows_sys::Win32::UI::Shell::{FOLDERID_Documents, KF_FLAG_DEFAULT, SHGetKnownFolderPath};

    let mut raw: windows_sys::core::PWSTR = std::ptr::null_mut();
    // SAFETY: on success `raw` is a NUL-terminated wide string that we own and free
    // with CoTaskMemFree; on failure it must still be freed (it may be null).
    unsafe {
        let hr = SHGetKnownFolderPath(&FOLDERID_Documents, KF_FLAG_DEFAULT as u32, std::ptr::null_mut(), &mut raw);
        let path = (hr >= 0 && !raw.is_null()).then(|| {
            let len = (0..).take_while(|&i| *raw.add(i) != 0).count();
            PathBuf::from(std::ffi::OsString::from_wide(std::slice::from_raw_parts(raw, len)))
        });
        CoTaskMemFree(raw as *const _);
        path
    }
}

#[cfg(not(windows))]
fn documents_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Documents"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_go_under_the_base() {
        let base = Path::new(r"C:\Users\x\Documents\fractalful\videos");
        assert_eq!(under(base, Path::new("clip.mp4")), base.join("clip.mp4"));
        let abs = Path::new(r"D:\renders\clip.mp4");
        assert_eq!(under(base, abs), abs);
    }

    #[cfg(windows)]
    #[test]
    fn documents_folder_exists() {
        let d = documents_dir().expect("known folder");
        assert!(d.is_absolute() && d.is_dir(), "{}", d.display());
    }
}
