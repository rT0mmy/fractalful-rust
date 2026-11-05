//! Song playback on the default output device (cpal). The audio callback owns the play
//! position; the app reads it to keep the animation clock in step with what's heard,
//! and seeks it when the user scrubs.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

use super::decode::Song;

const NO_SEEK: u64 = u64::MAX;

#[derive(Default)]
struct Shared {
    /// Song time (s, f64 bits) of the next sample the callback writes; may be negative
    /// or past the end (silence), so the clock keeps running around the song.
    pos: AtomicU64,
    /// Pending seek target (f64 bits), applied by the callback.
    seek: AtomicU64,
    playing: AtomicBool,
    volume: AtomicU32,
    /// Seconds between writing a sample and hearing it (f32 bits).
    latency: AtomicU32,
}

pub struct Player {
    _stream: cpal::Stream,
    shared: Arc<Shared>,
}

impl Player {
    pub fn new(song: Arc<Song>) -> Result<Self, String> {
        let device = cpal::default_host().default_output_device().ok_or("no audio output device")?;
        let config = device.default_output_config().map_err(|e| e.to_string())?;
        let shared = Arc::new(Shared::default());
        shared.seek.store(NO_SEEK, Ordering::Relaxed);
        shared.volume.store(1f32.to_bits(), Ordering::Relaxed);
        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();
        let stream = match format {
            SampleFormat::F32 => build::<f32>(&device, config, song, shared.clone()),
            SampleFormat::I16 => build::<i16>(&device, config, song, shared.clone()),
            SampleFormat::I32 => build::<i32>(&device, config, song, shared.clone()),
            SampleFormat::U16 => build::<u16>(&device, config, song, shared.clone()),
            f => return Err(format!("unsupported output sample format {f}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self { _stream: stream, shared })
    }

    pub fn set_playing(&self, on: bool) {
        self.shared.playing.store(on, Ordering::Relaxed);
    }

    pub fn is_playing(&self) -> bool {
        self.shared.playing.load(Ordering::Relaxed)
    }

    pub fn seek(&self, t: f32) {
        self.shared.seek.store((t as f64).to_bits(), Ordering::Relaxed);
    }

    pub fn set_volume(&self, v: f32) {
        self.shared.volume.store(v.to_bits(), Ordering::Relaxed);
    }

    /// Song time being heard right now (a pending seek counts as done).
    pub fn time(&self) -> f32 {
        let seek = self.shared.seek.load(Ordering::Relaxed);
        if seek != NO_SEEK {
            return f64::from_bits(seek) as f32;
        }
        let pos = f64::from_bits(self.shared.pos.load(Ordering::Relaxed)) as f32;
        pos - f32::from_bits(self.shared.latency.load(Ordering::Relaxed))
    }
}

fn build<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    song: Arc<Song>,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, String> {
    let out_channels = config.channels as usize;
    let dt = 1.0 / config.sample_rate as f64;
    let frames = song.samples.len() / song.channels;
    let rate = song.rate as f64;
    // Linear interpolation between source frames (device and song rates may differ).
    let frame_at = move |t: f64| -> (f32, f32) {
        let x = t * rate;
        if x < 0.0 || x >= (frames - 1) as f64 {
            return (0.0, 0.0);
        }
        let i = x as usize;
        let f = (x - i as f64) as f32;
        let get = |i: usize| match song.channels {
            1 => (song.samples[i], song.samples[i]),
            _ => (song.samples[2 * i], song.samples[2 * i + 1]),
        };
        let ((l0, r0), (l1, r1)) = (get(i), get(i + 1));
        (l0 + (l1 - l0) * f, r0 + (r1 - r0) * f)
    };
    device
        .build_output_stream(
            config,
            move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
                let ts = info.timestamp();
                let latency = ts.playback.duration_since(ts.callback).as_secs_f32();
                shared.latency.store(latency.to_bits(), Ordering::Relaxed);
                let seek = shared.seek.swap(NO_SEEK, Ordering::Relaxed);
                let mut pos = if seek != NO_SEEK { f64::from_bits(seek) } else { f64::from_bits(shared.pos.load(Ordering::Relaxed)) };
                if !shared.playing.load(Ordering::Relaxed) {
                    data.fill(T::from_sample(0.0f32));
                    shared.pos.store(pos.to_bits(), Ordering::Relaxed);
                    return;
                }
                let volume = f32::from_bits(shared.volume.load(Ordering::Relaxed));
                for frame in data.chunks_mut(out_channels) {
                    let (l, r) = frame_at(pos);
                    let (l, r) = (l * volume, r * volume);
                    match frame {
                        [mono] => *mono = T::from_sample((l + r) * 0.5),
                        [a, b, rest @ ..] => {
                            *a = T::from_sample(l);
                            *b = T::from_sample(r);
                            rest.fill(T::from_sample(0.0f32));
                        }
                        [] => {}
                    }
                    pos += dt;
                }
                shared.pos.store(pos.to_bits(), Ordering::Relaxed);
            },
            |e| eprintln!("audio output: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}
