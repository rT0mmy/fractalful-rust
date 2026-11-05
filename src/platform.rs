//! Process setup for a windowed (no console) release build.
//!
//! Release builds use the Windows GUI subsystem, so double-clicking the exe opens no
//! terminal. Started from a terminal, the app attaches to it so `println!` output
//! (bench results, export progress) still shows up; redirected output (`> log.txt`,
//! pipes) works either way. Panics, which would otherwise vanish with no console, are
//! appended to `%APPDATA%\fractalful\crash.log`.

use std::io::Write;

/// Attaches to the parent process's console when stdout isn't already redirected.
#[cfg(windows)]
pub fn attach_parent_console() {
    use windows_sys::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole, GetStdHandle, STD_OUTPUT_HANDLE};
    // SAFETY: plain Win32 calls without pointers; failure (no parent console, as when
    // started from Explorer) just leaves output disabled.
    unsafe {
        let out = GetStdHandle(STD_OUTPUT_HANDLE);
        if out.is_null() || out as isize == -1 {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

#[cfg(not(windows))]
pub fn attach_parent_console() {}

/// Asks for the Windows 11 Mica backdrop. The client area is drawn opaque, so Mica
/// shows on the title bar, tinted by the wallpaper and dark / light with the app
/// theme. Earlier Windows versions ignore it.
pub fn with_mica(attrs: egui_winit::winit::window::WindowAttributes) -> egui_winit::winit::window::WindowAttributes {
    #[cfg(windows)]
    {
        use egui_winit::winit::platform::windows::{BackdropType, WindowAttributesExtWindows};
        attrs.with_system_backdrop(BackdropType::MainWindow)
    }
    #[cfg(not(windows))]
    attrs
}

/// Opens a file or folder with its default app.
pub fn open(path: &std::path::Path) -> Result<(), String> {
    let program = if cfg!(windows) { "explorer" } else { "xdg-open" };
    std::process::Command::new(program).arg(path).spawn().map(drop).map_err(|e| format!("couldn't open {}: {e}", path.display()))
}

/// Opens the file's folder in Explorer with the file selected.
#[cfg(windows)]
pub fn reveal(path: &std::path::Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    // Explorer wants the quotes around the path only, not around "/select,...".
    std::process::Command::new("explorer")
        .raw_arg(format!("/select,\"{}\"", path.display()))
        .spawn()
        .map(drop)
        .map_err(|e| format!("couldn't show {}: {e}", path.display()))
}

#[cfg(not(windows))]
pub fn reveal(path: &std::path::Path) -> Result<(), String> {
    open(path.parent().unwrap_or(path))
}

/// The last panic logged at or after `since` (the crash log entry), if any.
pub fn panic_since(since: std::time::SystemTime) -> Option<String> {
    let log = std::fs::read_to_string(crash_log_path()).ok()?;
    let entry = log.rsplit("--- fractalful ").next().filter(|e| !e.trim().is_empty())?;
    let when: u64 = entry.split("(unix time ").nth(1)?.split(')').next()?.trim().parse().ok()?;
    let since = since.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    (when >= since).then(|| format!("--- fractalful {}", entry.trim_end()))
}

/// Where panics are logged.
pub fn crash_log_path() -> std::path::PathBuf {
    crate::project::session_path().with_file_name("crash.log")
}

/// Keeps the default panic message (for terminals) and also appends it to the crash log.
pub fn install_crash_log() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        let path = crash_log_path();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let when = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let backtrace = std::backtrace::Backtrace::force_capture();
            let _ = writeln!(f, "--- fractalful {} crashed (unix time {when}) ---\n{info}\n{backtrace}\n", env!("CARGO_PKG_VERSION"));
        }
    }));
}
