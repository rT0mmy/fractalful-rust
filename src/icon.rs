//! Window icons that follow the Windows light / dark theme.
//!
//! The black glyph is for light backgrounds, the white one for dark. Windows themes
//! the title bar (app mode) and the taskbar (system mode) separately, so the small
//! icon follows the window's theme and the taskbar icon the system's. Both are
//! downscaled here for the current DPI: Windows' own scaling of a 200 px icon is
//! nearest-neighbour and shreds the thin Sierpinski gaps.
//!
//! The exe's file icon (Explorer, pinned shortcuts) is embedded by build.rs and
//! can't follow the theme.

use egui_winit::winit;
use image::imageops::FilterType;
use winit::window::{Icon, Theme, Window};

const BLACK: &[u8] = include_bytes!("assets/icon_black.png");
const WHITE: &[u8] = include_bytes!("assets/icon_white.png");

fn icon(dark: bool, size: u32) -> Option<Icon> {
    let img = image::load_from_memory(if dark { WHITE } else { BLACK }).ok()?;
    let rgba = img.resize_exact(size, size, FilterType::Lanczos3).into_rgba8();
    Icon::from_rgba(rgba.into_raw(), size, size).ok()
}

/// Sets the title bar and taskbar icons for the current theme and DPI. Call again
/// on `ThemeChanged` / `ScaleFactorChanged`.
pub fn apply(window: &Window) {
    let scale = window.scale_factor();
    let px = |logical: f64| (logical * scale).round().max(16.0) as u32;
    let app_dark = window.theme() != Some(Theme::Light);
    window.set_window_icon(icon(app_dark, px(16.0)));
    #[cfg(windows)]
    {
        use winit::platform::windows::WindowExtWindows;
        window.set_taskbar_icon(icon(system_dark().unwrap_or(app_dark), px(32.0)));
    }
}

/// Whether the taskbar / Start use dark mode (Settings > Personalization > Colors).
#[cfg(windows)]
fn system_dark() -> Option<bool> {
    use windows_sys::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_DWORD, RegGetValueW};
    let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
    let key = wide(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize");
    let value = wide("SystemUsesLightTheme");
    let mut light = 0u32;
    let mut len = size_of::<u32>() as u32;
    // SAFETY: NUL-terminated wide strings and a u32 out-buffer whose size is passed in.
    let status = unsafe {
        RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), value.as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(), (&raw mut light).cast(), &mut len)
    };
    (status == 0).then_some(light == 0)
}
