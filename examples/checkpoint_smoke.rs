// SPDX-License-Identifier: MIT OR Apache-2.0
//! Opt-in Linux checkpoint validation. See docs/checkpoint-smoke.md.

#[cfg(target_os = "linux")]
#[path = "checkpoint_smoke/run.rs"]
mod run;

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOCATOR: run::memory::TrackingAllocator = run::memory::TrackingAllocator;

fn main() {
    #[cfg(target_os = "linux")]
    if let Err(error) = run::main() {
        eprintln!("checkpoint smoke failed: {error}");
        std::process::exit(1);
    }
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("checkpoint_smoke requires Linux /proc memory accounting");
        std::process::exit(1);
    }
}
