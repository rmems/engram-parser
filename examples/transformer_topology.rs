// SPDX-License-Identifier: MIT OR Apache-2.0
//! Transformer topology inventory for a GGUF or Safetensors checkpoint.
//!
//! Classifies every tensor into a role (embedding, attention Q/K/V/O,
//! MLP gate/up/down, norms, output head, MoE router / experts) and
//! prints the per-layer structure. Dense and MoE checkpoints share the
//! same report. Nothing is executed or decoded.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example transformer_topology -- /path/to/model.gguf
//! cargo run --features safetensors --example transformer_topology -- /path/to/hf-model-dir
//! cargo run --example transformer_topology -- --json /path/to/model.gguf > topology.json
//! ```
//!
//! Flags: `--json` prints the deterministic JSON rendering instead of
//! the text inventory, `--summary` prints only the header and per-layer
//! lines (no per-tensor rows).

use std::env;
use std::process::ExitCode;

use engram_parser::analysis::transformer::{TransformerTopology, discover_topology};
use engram_parser::open_checkpoint;

enum Mode {
    Text,
    Summary,
    Json,
}

fn parse_args() -> std::result::Result<(String, Mode), String> {
    let (mut path, mut mode) = (None, Mode::Text);
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--json" => mode = Mode::Json,
            "--summary" => mode = Mode::Summary,
            "--" => {}
            "-h" | "--help" => return Err(String::new()),
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            other if path.is_none() => path = Some(other.to_owned()),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    let path = path.ok_or_else(|| "missing checkpoint path".to_string())?;
    Ok((path, mode))
}

fn print_summary(topo: &TransformerTopology) {
    let s = &topo.summary;
    println!(
        "architecture: {}",
        topo.architecture_hint.as_deref().unwrap_or("(none)")
    );
    println!(
        "layers: {} discovered, {} missing; moe: {}; hidden={:?} vocab={:?} tied={}",
        s.layer_count,
        s.missing_layers.len(),
        s.is_moe,
        s.hidden_size_hint,
        s.vocab_size_hint,
        s.tied_embeddings_hint
    );
    println!("roles:");
    for (role, n) in &s.role_counts {
        println!("  {role:<22} {n:>6}");
    }
    println!("confidence:");
    for (c, n) in &s.confidence_counts {
        println!("  {c:<22} {n:>6}");
    }
    for layer in &topo.layers {
        print!(
            "layer {:>3}: attention={:<10} feed_forward={:<13}",
            layer.index, layer.attention, layer.feed_forward
        );
        if let Some(n) = layer.expert_count {
            print!(" experts={n}");
        }
        if layer.has_shared_expert {
            print!(" shared_expert");
        }
        println!();
    }
    if !s.ambiguous_tensors.is_empty() {
        println!("ambiguous: {:?}", s.ambiguous_tensors);
    }
    if !s.unknown_tensors.is_empty() {
        println!("unknown:   {:?}", s.unknown_tensors);
    }
}

fn main() -> ExitCode {
    let (path, mode) = match parse_args() {
        Ok(parsed) => parsed,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}");
            }
            eprintln!(
                "usage: cargo run --example transformer_topology -- [--json|--summary] <checkpoint>"
            );
            return ExitCode::from(2);
        }
    };
    let ckpt = match open_checkpoint(&path) {
        Ok(ckpt) => ckpt,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let topo = discover_topology(&ckpt);
    match mode {
        Mode::Text => print!("{topo}"),
        Mode::Summary => print_summary(&topo),
        Mode::Json => println!("{}", topo.to_json()),
    }
    ExitCode::SUCCESS
}
