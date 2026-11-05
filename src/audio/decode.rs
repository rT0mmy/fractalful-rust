//! Decodes a whole audio file (FLAC, MP3, WAV) into memory with symphonia.

use std::fs::File;
use std::path::Path;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Longest song accepted (decoded songs are kept in memory as f32).
const MAX_SECONDS: f32 = 30.0 * 60.0;

pub const EXTENSIONS: [&str; 3] = ["flac", "mp3", "wav"];

/// Decoded audio: interleaved f32 samples, mono or stereo.
pub struct Song {
    pub rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
}

pub fn is_audio(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

pub fn decode(path: &Path) -> Result<Song, String> {
    let file = File::open(path).map_err(|e| format!("can't open {}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let unsupported = |e: Error| format!("{} isn't a supported audio file ({e})", path.display());
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(unsupported)?;
    let track = format.default_track(TrackType::Audio).ok_or("the file has no audio track")?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or("the file has no audio track")?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(unsupported)?;
    let track_id = track.id;

    let mut rate = 0;
    let mut src_channels = 0;
    let mut samples: Vec<f32> = Vec::new();
    let mut buf: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("error reading {}: {e}", path.display())),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A corrupt frame: skip it, as players do.
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("error decoding {}: {e}", path.display())),
        };
        if rate == 0 {
            rate = decoded.spec().rate();
            src_channels = decoded.spec().channels().count().max(1);
        }
        buf.resize(decoded.samples_interleaved(), 0.0);
        decoded.copy_to_slice_interleaved(&mut buf);
        // Keep mono as is; fold everything else down to stereo (extra channels into both).
        match src_channels {
            1 | 2 => samples.extend_from_slice(&buf),
            n => {
                for f in buf.chunks_exact(n) {
                    let rest = f[2..].iter().sum::<f32>() / (n - 2) as f32 * 0.5;
                    samples.extend_from_slice(&[f[0] + rest, f[1] + rest]);
                }
            }
        }
        if samples.len() as f32 / (src_channels.min(2) * rate as usize) as f32 > MAX_SECONDS {
            return Err(format!("songs longer than {} minutes aren't supported", MAX_SECONDS / 60.0));
        }
    }
    if rate == 0 || samples.is_empty() {
        return Err(format!("{} contains no audio", path.display()));
    }
    Ok(Song { rate, channels: src_channels.min(2), samples })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a 16-bit PCM WAV.
    pub fn write_wav(path: &Path, rate: u32, channels: u16, samples: &[f32]) {
        let data: Vec<u8> = samples.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()).collect();
        let mut f = Vec::new();
        f.extend_from_slice(b"RIFF");
        f.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        f.extend_from_slice(b"WAVEfmt ");
        f.extend_from_slice(&16u32.to_le_bytes());
        f.extend_from_slice(&1u16.to_le_bytes());
        f.extend_from_slice(&channels.to_le_bytes());
        f.extend_from_slice(&rate.to_le_bytes());
        f.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        f.extend_from_slice(&(channels * 2).to_le_bytes());
        f.extend_from_slice(&16u16.to_le_bytes());
        f.extend_from_slice(b"data");
        f.extend_from_slice(&(data.len() as u32).to_le_bytes());
        f.extend_from_slice(&data);
        std::fs::write(path, f).unwrap();
    }

    #[test]
    fn decodes_a_wav() {
        let dir = std::env::temp_dir().join("fractalful-decode-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        let rate = 22050;
        let x: Vec<f32> = (0..rate * 2).map(|i| if i % 2 == 0 { 0.5 } else { -0.25 }).collect();
        write_wav(&path, rate as u32, 2, &x);
        let song = decode(&path).unwrap();
        assert_eq!((song.rate, song.channels), (rate as u32, 2));
        assert_eq!(song.samples.len(), x.len());
        assert!((song.samples[0] - 0.5).abs() < 1e-3 && (song.samples[1] + 0.25).abs() < 1e-3);
        assert!(is_audio(Path::new("a/B.FLAC")) && !is_audio(Path::new("x.json")));
        assert!(decode(&dir.join("missing.mp3")).is_err());
    }
}
