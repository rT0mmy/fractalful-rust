//! Embeds the exe icon and version info (Explorer, pinned shortcuts, Task Manager's
//! "Fractalful" name) as a compiled .res file, which MSVC's linker takes directly, so
//! no rc.exe / Windows SDK is needed.
//!
//! A file icon can't follow the light / dark theme, so it is the white glyph with a
//! thin dark outline, readable on both. The running window's icons are swapped per
//! theme at runtime (src/icon.rs).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/assets/icon_white.png");
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("fractalful.res");
        std::fs::write(&out, res::build()).expect("write fractalful.res");
        println!("cargo:rustc-link-arg-bins={}", out.display());
    }
}

#[cfg(windows)]
mod res {
    use image::imageops::FilterType;
    use image::{Rgba, RgbaImage};

    const SIZES: [u32; 8] = [16, 20, 24, 32, 40, 48, 64, 256];
    const RT_ICON: u16 = 3;
    const RT_GROUP_ICON: u16 = 14;
    const RT_VERSION: u16 = 16;

    /// The white glyph over its own silhouette grown by ~1 px in near-black.
    fn outlined(glyph: &RgbaImage, size: u32) -> RgbaImage {
        let white = image::imageops::resize(glyph, size, size, FilterType::Lanczos3);
        let r = (size as i32 / 32).max(1);
        let mut out = RgbaImage::new(size, size);
        for y in 0..size as i32 {
            for x in 0..size as i32 {
                let mut a = 0u8;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (sx, sy) = (x + dx, y + dy);
                        if dx * dx + dy * dy <= r * r && (0..size as i32).contains(&sx) && (0..size as i32).contains(&sy) {
                            a = a.max(white.get_pixel(sx as u32, sy as u32)[3]);
                        }
                    }
                }
                let top = white.get_pixel(x as u32, y as u32);
                // "over" compositing of the white glyph on the dark outline (straight alpha).
                let (ta, ba) = (top[3] as f32 / 255.0, a as f32 / 255.0 * 0.85);
                let oa = ta + ba * (1.0 - ta);
                let c = |tc: u8| if oa > 0.0 { ((tc as f32 * ta + 16.0 * ba * (1.0 - ta)) / oa).round() as u8 } else { 0 };
                out.put_pixel(x as u32, y as u32, Rgba([c(top[0]), c(top[1]), c(top[2]), (oa * 255.0).round() as u8]));
            }
        }
        out
    }

    fn png(img: &RgbaImage) -> Vec<u8> {
        let mut bytes = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).expect("encode icon");
        bytes
    }

    fn pad4(v: &mut Vec<u8>) {
        while v.len() % 4 != 0 {
            v.push(0);
        }
    }

    /// One .res entry: header with numeric type and name, then the data.
    fn entry(res: &mut Vec<u8>, kind: u16, id: u16, data: &[u8]) {
        res.extend((data.len() as u32).to_le_bytes());
        res.extend(32u32.to_le_bytes());
        for v in [0xFFFF, kind, 0xFFFF, id] {
            res.extend(u16::to_le_bytes(v));
        }
        res.extend(0u32.to_le_bytes()); // data version
        res.extend(0x1030u16.to_le_bytes()); // moveable | pure | discardable
        res.extend(0x0409u16.to_le_bytes()); // en-US
        res.extend([0u8; 8]); // version, characteristics
        res.extend(data);
        pad4(res);
    }

    fn wide(s: &str) -> Vec<u8> {
        s.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect()
    }

    enum Value<'a> {
        None,
        Text(&'a str),
        Binary(&'a [u8]),
    }

    /// A VS_VERSIONINFO node: wLength, wValueLength, wType, key, value, children.
    fn node(key: &str, value: Value, children: &[Vec<u8>]) -> Vec<u8> {
        let (value_bytes, value_len, kind) = match value {
            Value::None => (Vec::new(), 0, 1),
            Value::Text(t) => (wide(t), t.encode_utf16().count() as u16 + 1, 1),
            Value::Binary(b) => (b.to_vec(), b.len() as u16, 0),
        };
        let mut v = vec![0, 0];
        v.extend(value_len.to_le_bytes());
        v.extend(u16::to_le_bytes(kind));
        v.extend(wide(key));
        pad4(&mut v);
        v.extend(value_bytes);
        for c in children {
            pad4(&mut v);
            v.extend(c);
        }
        let len = v.len() as u16;
        v[..2].copy_from_slice(&len.to_le_bytes());
        v
    }

    fn version_info() -> Vec<u8> {
        let version = env!("CARGO_PKG_VERSION");
        let n: Vec<u16> = version.split(['.', '-']).filter_map(|p| p.parse().ok()).chain([0; 4]).take(4).collect();
        let (ms, ls) = ((n[0] as u32) << 16 | n[1] as u32, (n[2] as u32) << 16 | n[3] as u32);
        let mut fixed = Vec::new();
        // signature, struct version, file + product version, flags mask, flags,
        // OS (VOS_NT_WINDOWS32), type (VFT_APP), subtype, date.
        for v in [0xFEEF04BD, 0x0001_0000, ms, ls, ms, ls, 0x3F, 0, 0x0004_0004, 1, 0, 0, 0] {
            fixed.extend(u32::to_le_bytes(v));
        }
        let strings: Vec<Vec<u8>> = [
            ("FileDescription", "Fractalful"),
            ("ProductName", "Fractalful"),
            ("FileVersion", version),
            ("ProductVersion", version),
            ("InternalName", "fractalful"),
            ("OriginalFilename", "fractalful.exe"),
        ]
        .iter()
        .map(|(k, v)| node(k, Value::Text(v), &[]))
        .collect();
        let table = node("040904B0", Value::None, &strings);
        let string_info = node("StringFileInfo", Value::None, &[table]);
        let translation = node("Translation", Value::Binary(&[0x09, 0x04, 0xB0, 0x04]), &[]);
        let var_info = node("VarFileInfo", Value::None, &[translation]);
        node("VS_VERSION_INFO", Value::Binary(&fixed), &[string_info, var_info])
    }

    pub fn build() -> Vec<u8> {
        let glyph = image::open("src/assets/icon_white.png").expect("read src/assets/icon_white.png").into_rgba8();
        let mut res = Vec::new();
        entry(&mut res, 0, 0, &[]); // the empty entry every .res file starts with
        let mut group = Vec::new();
        for v in [0u16, 1, SIZES.len() as u16] {
            group.extend(v.to_le_bytes());
        }
        for (i, &size) in SIZES.iter().enumerate() {
            let data = png(&outlined(&glyph, size));
            let id = i as u16 + 1;
            entry(&mut res, RT_ICON, id, &data);
            let side = if size >= 256 { 0 } else { size as u8 };
            group.extend([side, side, 0, 0]);
            group.extend(1u16.to_le_bytes()); // planes
            group.extend(32u16.to_le_bytes()); // bits per pixel
            group.extend((data.len() as u32).to_le_bytes());
            group.extend(id.to_le_bytes());
        }
        entry(&mut res, RT_GROUP_ICON, 1, &group);
        entry(&mut res, RT_VERSION, 1, &version_info());
        res
    }
}
