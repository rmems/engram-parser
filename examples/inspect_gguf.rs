// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inventory a real on-disk GGUF (xai-dissect-style pilot path).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example inspect_gguf -- /path/to/model.gguf
//! ENGRAM_GGUF=~/.models/gguf/foo.gguf cargo run --example inspect_gguf
//! ```
//!
//! CPU-only. No CUDA, no dequant, no generation.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use engram_parser::analysis::moe::{extract_expert, list_experts};
use engram_parser::{GgufLayout, ggml_type_label, load_gguf};

enum Resolved {
    Path(PathBuf),
    Help,
    Version,
}

fn print_usage() {
    eprintln!("usage: cargo run --example inspect_gguf -- [--] <model.gguf>");
    eprintln!("   or: ENGRAM_GGUF=<model.gguf> cargo run --example inspect_gguf");
}

fn run(path: &Path) -> ExitCode {
    let t0 = Instant::now();
    let layout = match load_gguf(path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("load_gguf failed: {e}");
            return ExitCode::from(1);
        }
    };
    // Includes full-file read + parse (not parse-only).
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    print_inventory(&layout, path, load_ms);
    print_dtype_histogram(&layout);
    print_moe_summary(&layout);
    print_tensor_sample(&layout);

    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    match resolve_path() {
        Ok(Resolved::Path(p)) => {
            if !p.is_file() {
                eprintln!("not a file: {}", p.display());
                return ExitCode::from(1);
            }
            run(&p)
        }
        Ok(Resolved::Help) => {
            print_usage();
            ExitCode::SUCCESS
        }
        Ok(Resolved::Version) => {
            eprintln!("engram-parser {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn print_inventory(layout: &GgufLayout, path: &Path, load_ms: f64) {
    println!("path:          {}", path.display());
    println!("load_ms:       {load_ms:.2}");
    println!("architecture:  {}", layout.metadata.architecture());
    println!("quantization:  {}", layout.metadata.quantization());
    println!("alignment:     {}", layout.alignment);
    println!("tensor_count:  {}", layout.tensors.len());
    println!("block_count:   {:?}", layout.metadata.block_count());
    println!("expert_count:  {:?}", layout.metadata.expert_count());
    println!("expert_used:   {:?}", layout.metadata.expert_used_count());
    println!("embed_len:     {:?}", layout.metadata.embedding_length());
}

fn print_dtype_histogram(layout: &GgufLayout) {
    let mut counts: Vec<(String, usize)> = {
        let mut m: HashMap<String, usize> = HashMap::new();
        for t in layout.tensors.values() {
            *m.entry(ggml_type_label(t.ggml_type).to_owned())
                .or_default() += 1;
        }
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    };
    if counts.len() > 16 {
        counts.truncate(16);
    }
    println!("dtype_hist:    {counts:?}");
}

fn print_moe_summary(layout: &GgufLayout) {
    let experts = list_experts(layout);
    println!("moe_pairs:     {} (block,expert)", experts.len());
    if experts.is_empty() {
        return;
    }

    let show = experts.len().min(8);
    println!("moe_pairs_hd:  {:?}", &experts[..show]);

    let (b, e) = experts[0];
    let t1 = Instant::now();
    match extract_expert(layout, b, e) {
        Ok(w) => {
            let extract_ms = t1.elapsed().as_secs_f64() * 1000.0;
            println!(
                "extract ({b},{e}): complete={} extract_ms={extract_ms:.2}",
                w.is_complete()
            );
            let roles = [
                ("gate", &w.gate, true),
                ("up", &w.up, false),
                ("down", &w.down, false),
            ];
            for (name, opt, show_stacked) in roles {
                if let Some(t) = opt.as_ref() {
                    let mut line = format!(
                        "  {name}: dims={:?} bytes={} dtype={:?}",
                        t.dims,
                        t.bytes.len(),
                        t.dtype
                    );
                    if show_stacked {
                        line.push_str(&format!(" stacked={}", t.stacked_slice));
                    }
                    println!("{line}");
                }
            }
        }
        Err(err) => println!("extract ({b},{e}) failed: {err}"),
    }
}

fn print_tensor_sample(layout: &GgufLayout) {
    let mut names: Vec<_> = layout.tensors.keys().cloned().collect();
    names.sort();
    let n = names.len().min(12);
    println!("tensor_names_hd ({n}/{}):", names.len());
    for name in &names[..n] {
        let t = &layout.tensors[name];
        println!(
            "  {name}: dims={:?} type={} byte_len={}",
            t.dims,
            ggml_type_label(t.ggml_type),
            t.byte_len
        );
    }
}

fn resolve_path() -> Result<Resolved, String> {
    // nosemgrep: argv is used only for CLI dispatch, never as a security trust anchor.
    let args = env::args_os().skip(1);
    let mut seen_dash_dash = false;
    for p in args {
        if let Some(s) = p.to_str() {
            if !seen_dash_dash {
                if s == "--" {
                    seen_dash_dash = true;
                    continue;
                }
                if s == "--help" || s == "-h" {
                    return Ok(Resolved::Help);
                }
                if s == "--version" || s == "-V" {
                    return Ok(Resolved::Version);
                }
                if s.starts_with('-') {
                    return Err(format!("unknown option {s}"));
                }
            }
            return Ok(Resolved::Path(PathBuf::from(p)));
        }
        // Non-UTF-8 path: treat as positional after a `--` or as the first argument.
        return Ok(Resolved::Path(PathBuf::from(p)));
    }
    env::var("ENGRAM_GGUF")
        .map(PathBuf::from)
        .map(Resolved::Path)
        .map_err(|_| "missing model path (arg or ENGRAM_GGUF)".into())
}
