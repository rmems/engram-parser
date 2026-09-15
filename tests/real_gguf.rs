// SPDX-License-Identifier: MIT OR Apache-2.0

//! Path-gated real GGUF pilots (xai-dissect style).
//!
//! CI never runs these: they are `#[ignore]` and need multi-GB weights on disk.
//! Locally, point at a file or a tree under `~/.models/gguf`:
//!
//! ```bash
//! ENGRAM_GGUF=~/.models/gguf/.../model.gguf \
//!   cargo test --test real_gguf -- --ignored --nocapture
//!
//! ENGRAM_MODEL_DIR=~/.models/gguf \
//!   ENGRAM_GGUF_MAX=3 \
//!   cargo test --test real_gguf -- --ignored --nocapture
//!
//! # Hard-fail if no MoE experts are discovered; sample multiple expert pairs:
//! ENGRAM_GGUF=~/.models/gguf/.../moe.gguf \
//!   ENGRAM_EXPECT_MOE=1 ENGRAM_MOE_SAMPLES=3 \
//!   cargo test --test real_gguf real_gguf_moe -- --ignored --nocapture
//! ```
//!
//! GPU / kernel experiments on the same weights belong in
//! `~/rmems/blackwell-kernel-lab` (or myelin-accelerator), not this crate.

use std::env;
use std::fs;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ENV_GGUF: &str = "ENGRAM_GGUF";
const ENV_MODEL_DIR: &str = "ENGRAM_MODEL_DIR";
const ENV_MAX: &str = "ENGRAM_GGUF_MAX";

/// When set to `1`/`true`/`yes`, MoE extract must find at least one expert
/// pair and a successful `extract_expert` (hard fail on dense / unknown names).
const ENV_EXPECT_MOE: &str = "ENGRAM_EXPECT_MOE";

/// How many (block, expert) pairs to extract when MoE is present (default 1).
const ENV_MOE_SAMPLES: &str = "ENGRAM_MOE_SAMPLES";

fn expect_moe_from(raw: Option<&str>) -> bool {
    match raw {
        Some(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        None => false,
    }
}

fn expect_moe() -> bool {
    expect_moe_from(env::var(ENV_EXPECT_MOE).ok().as_deref())
}

fn moe_sample_count_from(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.parse().ok()).unwrap_or(1).max(1)
}

fn moe_sample_count() -> usize {
    moe_sample_count_from(env::var(ENV_MOE_SAMPLES).ok().as_deref())
}

#[cfg(feature = "mmap")]
use engram_parser::load_gguf_mmap;
use engram_parser::{GgufLayout, MoeExpertWeights, extract_expert, list_experts, load_gguf};

/// Assert that an extracted expert has at least one role tensor and that each
/// role tensor has a consistent payload size for its declared dtype.
fn assert_expert_weights_valid(path: &Path, b: usize, e: usize, w: &MoeExpertWeights) {
    assert!(
        w.gate.is_some() || w.up.is_some() || w.down.is_some(),
        "{}: empty extract for ({b},{e})",
        path.display()
    );

    for (role, opt) in [("gate", &w.gate), ("up", &w.up), ("down", &w.down)] {
        let Some(t) = opt else { continue };
        assert!(!t.bytes.is_empty(), "{role} empty bytes");
        assert!(!t.dims.is_empty(), "{role} empty dims");

        let n_elements = t.dims.iter().product::<usize>();
        if let Some(expected) = t.dtype.byte_len_for_elements(n_elements) {
            assert_eq!(
                t.bytes.len(),
                expected,
                "{role} byte length for {} ({b},{e})",
                path.display()
            );
        }
    }
}

/// Resolve pilot paths the way xai-dissect resolves checkpoint pilots:
/// explicit file, else scan a directory for `*.gguf` recursively, limited by
/// the configured depth so huge trees stay controllable.
fn load_and_scan(path: &Path) -> (GgufLayout, Vec<(usize, usize)>) {
    let t0 = Instant::now();
    let layout = load_gguf(path).expect("load");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let experts = list_experts(&layout);

    eprintln!(
        "moe_scan {}  arch={} quant={} expert_meta={:?} pairs={} load_ms={load_ms:.1}",
        path.display(),
        layout.metadata.architecture(),
        layout.metadata.quantization(),
        layout.metadata.expert_count(),
        experts.len(),
    );

    (layout, experts)
}

fn extract_and_report(layout: &GgufLayout, path: &Path, b: usize, e: usize) {
    let w = extract_expert(layout, b, e).unwrap_or_else(|err| {
        panic!("extract_expert({}, {b}, {e}): {err}", path.display());
    });

    assert_expert_weights_valid(path, b, e, &w);

    eprintln!(
        "OK moe {}  pair=({b},{e}) complete={} stacked_gate={}",
        path.display(),
        w.is_complete(),
        w.gate.as_ref().map(|g| g.stacked_slice).unwrap_or(false),
    );
}

/// Load one pilot and extract up to `samples` MoE expert pairs.
/// Returns `true` if the file contained at least one discoverable MoE pair.
fn scan_one_pilot(path: &Path, samples: usize) -> bool {
    let (layout, experts) = load_and_scan(path);

    if experts.is_empty() {
        eprintln!("skip MoE (none discovered): {}", path.display());
        return false;
    }

    let take = samples.min(experts.len());
    for &(b, e) in experts.iter().take(take) {
        extract_and_report(&layout, path, b, e);
    }

    if let Some(n) = layout.metadata.expert_count() {
        assert!(n > 0, "{}: expert_count metadata is 0", path.display());
    }
    true
}

fn pilot_gguf_paths_from(
    single: Option<&str>,
    model_dir: Option<&str>,
    max: Option<&str>,
) -> Vec<PathBuf> {
    if let Some(single) = single {
        let p = PathBuf::from(single);
        return if p.is_file() { vec![p] } else { Vec::new() };
    }

    let Some(root) = model_dir else {
        return Vec::new();
    };
    let root = PathBuf::from(root);
    if !root.is_dir() {
        return Vec::new();
    }

    let max = max.and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);

    // Scan the directory tree for `*.gguf` files, limited by depth and the
    // requested cap. Entries are sorted at each directory so the result is
    // deterministic even though `fs::read_dir` order is unspecified.
    let mut out = Vec::new();
    collect_gguf(&root, 0, 6, max, &mut out);
    out.sort();
    out.truncate(max);
    out
}

fn pilot_gguf_paths() -> Vec<PathBuf> {
    pilot_gguf_paths_from(
        env::var(ENV_GGUF).ok().as_deref(),
        env::var(ENV_MODEL_DIR).ok().as_deref(),
        env::var(ENV_MAX).ok().as_deref(),
    )
}

fn partition_entries(dir: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return (files, dirs);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            files.push(path);
        }
    }
    files.sort();
    dirs.sort();
    (files, dirs)
}

fn collect_gguf(dir: &Path, depth: usize, max_depth: usize, max: usize, out: &mut Vec<PathBuf>) {
    if depth > max_depth || out.len() >= max {
        return;
    }
    let (files, dirs) = partition_entries(dir);
    for f in files {
        if out.len() >= max {
            break;
        }
        out.push(f);
    }
    for d in dirs {
        if out.len() >= max {
            break;
        }
        collect_gguf(&d, depth + 1, max_depth, max, out);
    }
}

fn require_pilots() -> Vec<PathBuf> {
    let paths = pilot_gguf_paths();
    assert!(
        !paths.is_empty(),
        "no pilot GGUFs found — set {ENV_GGUF}=/path/to/model.gguf \
         or {ENV_MODEL_DIR}=~/.models/gguf (optional {ENV_MAX}=N)"
    );
    for p in &paths {
        assert!(p.is_file(), "not a file: {}", p.display());
    }
    paths
}

#[test]
#[ignore = "pilot: set ENGRAM_GGUF or ENGRAM_MODEL_DIR; not run in CI"]
fn real_gguf_parse_inventory() {
    let paths = require_pilots();
    for path in paths {
        let t0 = Instant::now();
        let layout = load_gguf(&path).unwrap_or_else(|e| {
            panic!("load_gguf({}) failed: {e}", path.display());
        });
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        assert!(
            !layout.tensors.is_empty(),
            "{}: expected tensors",
            path.display()
        );
        assert!(layout.alignment >= 1, "alignment");

        // Every directory entry must have a consistent byte_len for known dtypes.
        for (name, tensor) in &layout.tensors {
            assert!(
                tensor.byte_len > 0 || tensor.n_elements == 0,
                "{name}: zero byte_len with n_elements={}",
                tensor.n_elements
            );
            if let Some(expected) = tensor.dtype.byte_len_for_elements(tensor.n_elements) {
                assert_eq!(
                    tensor.byte_len, expected,
                    "{name}: byte_len mismatch for {:?}",
                    tensor.dtype
                );
            }
            // Payload must be in-range.
            let bytes = layout.tensor_bytes(tensor).unwrap_or_else(|e| {
                panic!("{name}: tensor_bytes: {e}");
            });
            assert_eq!(bytes.len(), tensor.byte_len, "{name}: payload len");
        }

        eprintln!(
            "OK inventory {}  tensors={} arch={} quant={} parse_ms={ms:.1}",
            path.display(),
            layout.tensors.len(),
            layout.metadata.architecture(),
            layout.metadata.quantization(),
        );
    }
}

#[test]
#[ignore = "pilot: set ENGRAM_GGUF or ENGRAM_MODEL_DIR; not run in CI"]
fn real_gguf_moe_extract_when_present() {
    let paths = require_pilots();
    let hard = expect_moe();
    let samples = moe_sample_count();
    let mut any_moe = false;

    for path in paths {
        if scan_one_pilot(&path, samples) {
            any_moe = true;
        }
    }

    if hard {
        assert!(
            any_moe,
            "ENGRAM_EXPECT_MOE set but no MoE expert tensors found in pilot set"
        );
    } else if !any_moe {
        eprintln!(
            "note: no MoE expert tensors in pilot set; inventory-only is fine for dense GGUFs"
        );
    }
}

/// Generate a unique temporary path under `std::env::temp_dir()`.
fn unique_tmp(prefix: &str) -> PathBuf {
    let pid = process::id();
    for _ in 0..10 {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!("{prefix}_{pid}_{n}"));
        if !path.exists() {
            return path;
        }
    }
    panic!("could not generate a unique temporary path");
}

struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn real_gguf_helpers_document_env() {
    // Always runs in CI: documents the pilot contract and exercises the
    // env-dependent helpers without mutating the process environment.
    let env_names = [
        (ENV_GGUF, "ENGRAM_GGUF"),
        (ENV_MODEL_DIR, "ENGRAM_MODEL_DIR"),
        (ENV_MAX, "ENGRAM_GGUF_MAX"),
        (ENV_EXPECT_MOE, "ENGRAM_EXPECT_MOE"),
        (ENV_MOE_SAMPLES, "ENGRAM_MOE_SAMPLES"),
    ];
    for (actual, expected) in env_names {
        assert_eq!(actual, expected);
    }

    let expect_cases = [
        (None, false),
        (Some("1"), true),
        (Some("YES"), true),
        (Some("no"), false),
    ];
    for (input, expected) in expect_cases {
        assert_eq!(
            expect_moe_from(input),
            expected,
            "expect_moe_from({input:?})"
        );
    }

    let sample_cases = [(Some("7"), 7usize), (Some("0"), 1), (None, 1)];
    for (input, expected) in sample_cases {
        assert_eq!(
            moe_sample_count_from(input),
            expected,
            "moe_sample_count_from({input:?})"
        );
    }

    // pilot_gguf_paths resolves a single file.
    let tmp = unique_tmp("engram_helpers_file");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .expect("create temp file");
    let _file_guard = TempFile(tmp.clone());
    assert_eq!(
        pilot_gguf_paths_from(Some(tmp.to_str().unwrap()), None, None),
        vec![tmp.clone()]
    );
    assert!(pilot_gguf_paths_from(Some("/does/not/exist"), None, None).is_empty());

    // pilot_gguf_paths scans a directory for *.gguf.
    let dir = unique_tmp("engram_helpers_dir");
    fs::create_dir(&dir).expect("create temp dir");
    let _dir_guard = TempDir(dir.clone());
    let gf = dir.join("model.gguf");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&gf)
        .expect("create temp gguf file");
    assert_eq!(
        pilot_gguf_paths_from(None, Some(dir.to_str().unwrap()), Some("5")),
        vec![gf]
    );
}

#[cfg(feature = "mmap")]
#[test]
#[ignore = "pilot: set ENGRAM_GGUF or ENGRAM_MODEL_DIR; not run in CI"]
fn real_gguf_mmap_parity() {
    let paths = require_pilots();
    for path in paths {
        let owned = load_gguf(&path).unwrap_or_else(|e| {
            panic!("load_gguf({}) failed: {e}", path.display());
        });
        let mapped = load_gguf_mmap(&path).unwrap_or_else(|e| {
            panic!("load_gguf_mmap({}) failed: {e}", path.display());
        });
        assert!(
            mapped.directory_matches(&owned),
            "{}: mmap directory mismatch vs owned load",
            path.display()
        );
        for (name, tensor) in &owned.tensors {
            let mapped_t = mapped.tensor(name).unwrap_or_else(|e| {
                panic!("{} missing mapped tensor {name}: {e}", path.display());
            });
            let owned_bytes = owned.tensor_bytes(tensor).expect("owned payload");
            let mapped_bytes = mapped.tensor_bytes(mapped_t).expect("mmap payload");
            assert_eq!(
                owned_bytes,
                mapped_bytes,
                "{} tensor {name} payload mismatch",
                path.display()
            );
        }
    }
}
