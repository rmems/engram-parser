// SPDX-License-Identifier: MIT OR Apache-2.0
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static RSS_PEAK: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

pub struct TrackingAllocator;

fn added(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
    LARGEST.fetch_max(size, Relaxed);
}

// SAFETY: all allocation operations delegate to System with the original
// layout/pointer. Accounting uses only atomics and cannot allocate or panic.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            added(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            added(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies a live pointer and its original layout.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller satisfies System's realloc contract.
        let next = unsafe { System.realloc(ptr, layout, size) };
        if !next.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            added(size);
        }
        next
    }
}

#[derive(Clone, Copy)]
pub struct Memory {
    pub rss: usize,
    pub rss_peak: usize,
    pub hwm: usize,
    pub virtual_bytes: usize,
    pub anonymous: usize,
    pub heap_peak: usize,
    pub largest: usize,
}

impl Memory {
    pub fn read() -> Result<Self, Box<dyn std::error::Error>> {
        // smaps_rollup is more accurate than the asynchronous VmRSS counters.
        let smaps = std::fs::read_to_string("/proc/self/smaps_rollup")?;
        let status = std::fs::read_to_string("/proc/self/status")?;
        let rss = field(&smaps, "Rss:")?;
        RSS_PEAK.fetch_max(rss, Relaxed);
        Ok(Self {
            rss,
            rss_peak: RSS_PEAK.load(Relaxed),
            anonymous: field(&smaps, "Anonymous:")?,
            hwm: field(&status, "VmHWM:")?,
            virtual_bytes: field(&status, "VmSize:")?,
            heap_peak: PEAK.load(Relaxed),
            largest: LARGEST.load(Relaxed),
        })
    }

    pub fn report(self, phase: &str) {
        println!(
            "memory phase={phase} rss={} hwm={} virtual={} anonymous={} heap_peak={} largest_allocation={}",
            self.rss, self.hwm, self.virtual_bytes, self.anonymous, self.heap_peak, self.largest
        );
    }
}

fn field(text: &str, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix(key)?.split_whitespace().next())
        .ok_or_else(|| format!("missing Linux memory counter {key}"))?;
    Ok(value
        .parse::<usize>()?
        .checked_mul(1024)
        .ok_or("memory counter overflow")?)
}
