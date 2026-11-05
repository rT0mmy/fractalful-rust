//! GPU time per stage of a frame (world, march, temporal resolve, processing, display),
//! from timestamp queries written between them. Results are read back a few frames late
//! through a small ring of buffers, so measuring never stalls the GPU. Without timestamp
//! support (the adapter decides) it stays silent.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use egui_wgpu::wgpu;

/// Timestamps per frame: a start, then one after each stage.
const MAX_MARKS: usize = 16;
/// Readback buffers in flight.
const RING: usize = 3;

struct Slot {
    buf: wgpu::Buffer,
    /// Set by the map callback once the results can be read.
    ready: Arc<AtomicBool>,
    /// Waiting for a map (stage labels of the frame it measured).
    pending: Option<Vec<&'static str>>,
}

pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    /// Ticks to nanoseconds.
    period: f32,
    /// This frame's stage labels (a timestamp is written after each).
    labels: Vec<&'static str>,
    /// The slot this frame resolves into (None: all busy, skip measuring this frame).
    slot: Option<usize>,
    /// Smoothed milliseconds per stage, in frame order.
    pub stages: Vec<(&'static str, f32)>,
}

impl GpuTimer {
    /// The features to request for timing, when the adapter has them.
    pub fn features(adapter: &wgpu::Adapter) -> wgpu::Features {
        let want = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if adapter.features().contains(want) { want } else { wgpu::Features::empty() }
    }

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
            return None;
        }
        let size = (MAX_MARKS * 8) as u64;
        let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("gpu timer"), ty: wgpu::QueryType::Timestamp, count: MAX_MARKS as u32 });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu timer resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..RING)
            .map(|_| Slot {
                buf: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpu timer readback"),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                ready: Arc::new(AtomicBool::new(false)),
                pending: None,
            })
            .collect();
        Some(Self { set, resolve, slots, period: queue.get_timestamp_period(), labels: Vec::new(), slot: None, stages: Vec::new() })
    }

    /// Starts a frame: writes the first timestamp.
    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.labels.clear();
        self.slot = self.slots.iter().position(|s| s.pending.is_none());
        if self.slot.is_some() {
            encoder.write_timestamp(&self.set, 0);
        }
    }

    /// Ends the stage `label` (everything recorded since the previous mark).
    pub fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        if self.slot.is_none() || self.labels.len() + 1 >= MAX_MARKS {
            return;
        }
        self.labels.push(label);
        encoder.write_timestamp(&self.set, self.labels.len() as u32);
    }

    /// Copies this frame's timestamps out; call before submitting `encoder`.
    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(i) = self.slot.filter(|_| !self.labels.is_empty()) else {
            self.slot = None;
            return;
        };
        let n = self.labels.len() as u32 + 1;
        encoder.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.slots[i].buf, 0, n as u64 * 8);
        self.slots[i].pending = Some(self.labels.clone());
    }

    /// After submitting: starts reading this frame's results, and takes in any that
    /// arrived (`device.poll` drives the callbacks).
    pub fn collect(&mut self, device: &wgpu::Device) {
        if let Some(i) = self.slot.take() {
            if self.slots[i].pending.is_some() {
                let ready = self.slots[i].ready.clone();
                self.slots[i].buf.map_async(wgpu::MapMode::Read, .., move |r| ready.store(r.is_ok(), Ordering::Release));
            }
        }
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in &mut self.slots {
            if !slot.ready.swap(false, Ordering::Acquire) {
                continue;
            }
            let labels = slot.pending.take().unwrap_or_default();
            let ticks: Vec<u64> = {
                let Ok(data) = slot.buf.slice(..).get_mapped_range() else { continue };
                data.chunks_exact(8).take(labels.len() + 1).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
            };
            slot.buf.unmap();
            // A different set of stages (e.g. processing turned off) starts over.
            if self.stages.iter().map(|s| s.0).ne(labels.iter().copied()) {
                self.stages = labels.iter().map(|l| (*l, 0.0)).collect();
            }
            for (k, st) in self.stages.iter_mut().enumerate() {
                let ms = ticks[k + 1].saturating_sub(ticks[k]) as f32 * self.period * 1e-6;
                // Smoothed like the frame time readout.
                st.1 = if st.1 == 0.0 { ms } else { st.1 * 0.9 + ms * 0.1 };
            }
        }
    }

    pub fn total_ms(&self) -> f32 {
        self.stages.iter().map(|s| s.1).sum()
    }
}
