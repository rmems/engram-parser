// SPDX-License-Identifier: MIT OR Apache-2.0
//! Format-agnostic tensor inventory for GGUF or Safetensors checkpoints.
//!
//! After `open_checkpoint`, everything goes through `&dyn Checkpoint`;
//! there is no format branching in the report.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example inspect_checkpoint -- /path/to/model.gguf
//! cargo run --features safetensors --example inspect_checkpoint -- /path/to/hf-model-dir
//! cargo run --all-features --example inspect_checkpoint -- --mmap --all /path/to/model.safetensors
//! ```
//!
//! Flags: `--all` prints every tensor (default: first 32), `--mmap`
//! memory-maps instead of reading (requires the `mmap` feature).
//! CPU-only; payloads are read as raw bytes, never decoded.

use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;

use engram_parser::{AnyCheckpoint, Checkpoint, Result, open_checkpoint};

const DEFAULT_ROWS: usize = 32;

struct Args {
    path: String,
    all: bool,
    mmap: bool,
}

fn parse_args() -> std::result::Result<Args, String> {
    let (mut path, mut all, mut mmap) = (None, false, false);
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--all" => all = true,
            "--mmap" => mmap = true,
            "--" => {}
            "-h" | "--help" => return Err(String::new()),
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            other if path.is_none() => path = Some(other.to_owned()),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    let path = path.ok_or_else(|| "missing checkpoint path".to_string())?;
    Ok(Args { path, all, mmap })
}

fn open(args: &Args) -> Result<AnyCheckpoint> {
    if args.mmap {
        #[cfg(feature = "mmap")]
        return engram_parser::open_checkpoint_mmap(&args.path);
        #[cfg(not(feature = "mmap"))]
        return Err(engram_parser::ParserError::FeatureDisabled {
            path: args.path.clone(),
            feature: "mmap",
        });
    }
    open_checkpoint(&args.path)
}

/// The whole report sees only the shared contract.
fn report(ckpt: &dyn Checkpoint, rows: usize) -> Result<()> {
    let source = ckpt.source();
    let tensors = ckpt.tensors();
    let total_bytes: usize = tensors.iter().map(|t| t.byte_len).sum();
    println!("format:        {} ({})", ckpt.format(), source.kind);
    println!("path:          {}", source.path.display());
    println!("root:          {}", source.root.display());
    if let Some(index) = &source.index_file {
        println!("index:         {index}");
    }
    println!("files:         {}", source.files.len());
    println!("metadata keys: {}", ckpt.metadata().len());
    println!("tensors:       {}", tensors.len());
    println!("payload bytes: {total_bytes}");
    println!();

    println!(
        "{:<48} {:<10} {:<10} {:<22} {:>12}  location",
        "name", "dtype", "native", "shape (outer-first)", "bytes"
    );
    for t in tensors.iter().take(rows) {
        println!(
            "{:<48} {:<10} {:<10} {:<22} {:>12}  {}@{}",
            t.name,
            t.dtype.label(),
            t.native_dtype,
            format!("{:?}", t.shape.dims()),
            t.byte_len,
            t.location.source,
            t.location.data_offset,
        );
    }
    if tensors.len() > rows {
        println!("… {} more (use --all)", tensors.len() - rows);
    }
    println!();

    let mut histogram: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in tensors {
        let entry = histogram.entry(t.native_dtype.clone()).or_default();
        entry.0 += 1;
        entry.1 += t.byte_len;
    }
    println!("dtype histogram (native label: count, bytes):");
    for (dtype, (count, bytes)) in &histogram {
        println!("  {dtype:<12} {count:>6}  {bytes:>14}");
    }

    // Smallest non-empty tensor keeps the payload probe cheap on huge checkpoints.
    if let Some(probe) = tensors
        .iter()
        .filter(|t| t.byte_len > 0)
        .min_by_key(|t| t.byte_len)
    {
        let bytes = ckpt.tensor_bytes(&probe.name)?;
        assert_eq!(bytes.len(), probe.byte_len, "payload length mismatch");
        println!();
        println!(
            "payload probe: '{}' -> {} bytes (matches byte_len)",
            probe.name,
            bytes.len()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}");
            }
            eprintln!(
                "usage: cargo run --example inspect_checkpoint -- [--all] [--mmap] <checkpoint>"
            );
            return ExitCode::from(2);
        }
    };
    let rows = if args.all { usize::MAX } else { DEFAULT_ROWS };
    match open(&args).and_then(|ckpt| report(&ckpt, rows)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
