//! Image files read by nodes (Environment Texture in the world graph, Image Texture in
//! the processing graph). Each file is decoded once on a background thread to linear
//! RGBA (8-bit formats are sRGB-decoded; HDR / EXR are already linear), shrunk to at
//! most `MAX_WIDTH` wide, and uploaded as a half-float texture shared by every node
//! using it. Until then (or if it fails) nodes read a black placeholder.

use std::collections::HashMap;
use std::sync::mpsc;

use egui_wgpu::wgpu;

/// Wider images are scaled down (an 8K panorama would take 268 MB of video memory).
const MAX_WIDTH: u32 = 4096;

enum Entry {
    Loading,
    Ready(wgpu::TextureView),
    Failed,
}

type Decoded = Result<(u32, u32, Vec<u16>), String>;

pub struct Images {
    entries: HashMap<String, Entry>,
    tx: mpsc::Sender<(String, Decoded)>,
    rx: mpsc::Receiver<(String, Decoded)>,
    placeholder: wgpu::TextureView,
    /// Bumped whenever an image finishes loading (bind groups using it must be rebuilt).
    pub version: u64,
}

impl Images {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (tx, rx) = mpsc::channel();
        let black = [0u16, 0, 0, f32_to_f16(1.0)];
        let placeholder = upload(device, queue, "image placeholder", 1, 1, &black);
        Self { entries: HashMap::new(), tx, rx, placeholder, version: 0 }
    }

    /// Starts loading `path` unless it is loaded, loading or failed already.
    pub fn request(&mut self, path: &str) {
        if self.entries.contains_key(path) {
            return;
        }
        self.entries.insert(path.to_string(), Entry::Loading);
        let tx = self.tx.clone();
        let path = path.to_string();
        std::thread::spawn(move || {
            let result = decode(&path);
            let _ = tx.send((path, result));
        });
    }

    /// The texture for `path`, or the black placeholder while it loads / if it failed.
    pub fn view(&self, path: Option<&str>) -> &wgpu::TextureView {
        match path.and_then(|p| self.entries.get(p)) {
            Some(Entry::Ready(view)) => view,
            _ => &self.placeholder,
        }
    }

    /// Uploads images that finished decoding. Returns a message per finished file (for
    /// toasts): Ok("name (w×h)") or Err(reason).
    pub fn poll(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Result<String, String>> {
        let mut out = Vec::new();
        while let Ok((path, result)) = self.rx.try_recv() {
            let name = std::path::Path::new(&path).file_name().map_or(path.clone(), |n| n.to_string_lossy().into_owned());
            let entry = match result {
                Ok((w, h, pixels)) => {
                    out.push(Ok(format!("Loaded {name} ({w}×{h})")));
                    Entry::Ready(upload(device, queue, &name, w, h, &pixels))
                }
                Err(e) => {
                    out.push(Err(format!("Couldn't load {name}: {e}")));
                    Entry::Failed
                }
            };
            self.entries.insert(path, entry);
            self.version += 1;
        }
        out
    }

}

fn upload(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, w: u32, h: u32, rgba: &[u16]) -> wgpu::TextureView {
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(rgba),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
        size,
    );
    tex.create_view(&Default::default())
}

fn decode(path: &str) -> Decoded {
    let img = image::ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;
    let linear = matches!(img.color(), image::ColorType::Rgb32F | image::ColorType::Rgba32F);
    let img = if img.width() > MAX_WIDTH {
        let h = (img.height() as u64 * MAX_WIDTH as u64 / img.width() as u64).max(1) as u32;
        img.resize_exact(MAX_WIDTH, h, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgba = img.to_rgba32f();
    let (w, h) = rgba.dimensions();
    let pixels = rgba
        .pixels()
        .flat_map(|p| {
            let c = |v: f32| if linear { v.max(0.0) } else { srgb_to_linear(v) };
            [c(p[0]), c(p[1]), c(p[2]), p[3]].map(|v| f32_to_f16(v.min(65000.0)))
        })
        .collect();
    Ok((w, h, pixels))
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// IEEE half from f32 (round to nearest; overflow to infinity, tiny values to zero or
/// subnormals).
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let h = ((exp as u32) << 10) | (man >> 13);
    // Round to nearest (a carry into the exponent is still correct).
    (sign as u32 | (h + ((man >> 12) & 1))) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion() {
        for (v, h) in [(0.0, 0x0000), (1.0, 0x3c00), (-2.0, 0xc000), (0.5, 0x3800), (65504.0, 0x7bff), (1e9, 0x7c00)] {
            assert_eq!(f32_to_f16(v), h, "{v}");
        }
        // A subnormal half.
        assert_eq!(f32_to_f16(2f32.powi(-20)), 0x0010);
    }

    fn f16_to_f32(h: u16) -> f32 {
        let exp = ((h >> 10) & 0x1f) as i32;
        let man = (h & 0x3ff) as f32;
        match exp {
            0 => man * 2f32.powi(-24),
            _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
        }
    }

    /// HDR files stay linear (values above 1 kept); 8-bit images are sRGB-decoded.
    #[test]
    fn decodes_hdr_linear_and_png_srgb() {
        let dir = std::env::temp_dir().join(format!("fractalful-images-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hdr = dir.join("sky.hdr");
        let px = vec![image::Rgb([4.0f32, 0.5, 0.125]); 8 * 4];
        image::codecs::hdr::HdrEncoder::new(std::fs::File::create(&hdr).unwrap()).encode(&px, 8, 4).unwrap();
        let (w, h, pixels) = decode(hdr.to_str().unwrap()).unwrap();
        assert_eq!((w, h), (8, 4));
        let rgb: Vec<f32> = pixels[..3].iter().map(|&v| f16_to_f32(v)).collect();
        assert!((rgb[0] - 4.0).abs() < 0.05 && (rgb[1] - 0.5).abs() < 0.01 && (rgb[2] - 0.125).abs() < 0.005, "{rgb:?}");

        let png = dir.join("tint.png");
        image::save_buffer(&png, &[128u8, 128, 128, 255].repeat(4), 2, 2, image::ColorType::Rgba8).unwrap();
        let (_, _, pixels) = decode(png.to_str().unwrap()).unwrap();
        let v = f16_to_f32(pixels[0]);
        assert!((v - 0.2158).abs() < 0.002, "sRGB 128 is linear 0.2158, got {v}");
        assert!((f16_to_f32(pixels[3]) - 1.0).abs() < 1e-3, "alpha stays as is");
        assert!(decode(dir.join("missing.hdr").to_str().unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
