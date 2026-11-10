// SPDX-License-Identifier: MIT OR Apache-2.0

//! Real-model topology discovery over the pinned RM-2367 fixtures (RM-1800).
//!
//! Dense (`Qwen3-0.6B`, `Qwen2.5-0.5B-Instruct`, `Qwen3.5-0.8B`) and MoE
//! (`OLMoE-1B-7B-0125-Instruct`) checkpoints, each in GGUF and Safetensors,
//! are fetched once into a cache, verified against pinned revisions and
//! SHA-256s, and fed through [`discover_topology`]. Layer count, expert
//! count, and shared-expert presence are checked against each model's
//! `config.json` (Safetensors side) or GGUF metadata, and the two format
//! views of a model are checked against each other, allowing for the
//! documented naming differences listed under `parity_notes`.
//!
//! No binaries are committed and CI never runs these tests: every real-model
//! test is `#[ignore]`d and additionally gated by the fixture helper. When a
//! fixture is unavailable (offline, no `curl`, or
//! `ENGRAM_REAL_FIXTURES=skip|offline`) the test logs `SKIP ...` and passes;
//! `ENGRAM_REAL_FIXTURES=require` turns that into a failure. See
//! `tests/common/real_fixtures.rs` for the full contract.
//!
//! ```bash
//! # Pre-staged weights under <root>/<fixture id>/ (see docs/real-fixtures.md):
//! ENGRAM_FIXTURE_ROOT=C:\models \
//!   cargo test --all-features --test real_topology -- --ignored --nocapture
//!
//! # Or let the helper download the ~28 GB pin set into the cache:
//! ENGRAM_REAL_FIXTURES=require \
//!   cargo test --all-features --test real_topology -- --ignored
//! ```

#![cfg_attr(not(feature = "safetensors"), allow(dead_code))]

#[path = "common/real_fixtures.rs"]
mod real_fixtures;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use engram_parser::analysis::transformer::{
    AttentionKind, FeedForwardKind, RoleScope, TensorRole, TransformerTopology, discover_topology,
};
use engram_parser::{load_gguf, open_checkpoint};
use real_fixtures::{
    Config, FixtureFile, Mode, Outcome, RealFixture, Sha256, ensure, fixture_or_skip, sha256_hex,
    to_hex,
};
use real_fixtures::{
    OLMOE_GGUF, OLMOE_SAFETENSORS, QWEN3_GGUF, QWEN3_SAFETENSORS, QWEN25_GGUF, QWEN25_SAFETENSORS,
    QWEN35_GGUF, QWEN35_SAFETENSORS,
};

// ---------------------------------------------------------------------------
// Model expectations, from each fixture's config.json / GGUF metadata
// ---------------------------------------------------------------------------

/// What one model must look like. `*_gguf` / `*_st` fields cover the
/// documented naming differences between the formats (see `parity_notes`).
struct ModelSpec {
    label: &'static str,
    gguf: &'static RealFixture,
    st: &'static RealFixture,
    /// `general.architecture` of the GGUF copy.
    gguf_arch: &'static str,
    /// `num_hidden_layers` (text tower for multimodal models).
    layers: usize,
    hidden: usize,
    /// `intermediate_size` (per-expert for MoE).
    intermediate: usize,
    vocab: usize,
    /// Routed experts per layer (`num_experts`); `None` for dense.
    experts: Option<usize>,
    /// `num_experts_per_tok`, for the GGUF `expert_used_count` check.
    experts_used: Option<usize>,
    shared_expert: bool,
    /// Primary layer stack on the Safetensors side.
    st_stack: &'static str,
    /// Uniform attention kind when every primary layer shares one
    /// (`None` for hybrids like Qwen3.5, asserted per-layer instead).
    uniform_attention: Option<AttentionKind>,
    /// `tied_embeddings_hint` differs per format when one copy materializes
    /// the output head and the other does not.
    gguf_tied: bool,
    st_tied: bool,
    gguf_tensors: usize,
    st_tensors: usize,
}

/// Qwen3-0.6B: 28 layers, hidden 1024, FFN 3072, 16 heads / 8 KV heads,
/// vocab 151936. HF Safetensors materializes `lm_head.weight` even though
/// `tie_word_embeddings` is set; the GGUF copy does not carry `output.weight`.
const QWEN3: ModelSpec = ModelSpec {
    label: "qwen3-0.6b",
    gguf: &QWEN3_GGUF,
    st: &QWEN3_SAFETENSORS,
    gguf_arch: "qwen3",
    layers: 28,
    hidden: 1024,
    intermediate: 3072,
    vocab: 151_936,
    experts: None,
    experts_used: None,
    shared_expert: false,
    st_stack: "model.layers",
    uniform_attention: Some(AttentionKind::Separate),
    gguf_tied: true,
    st_tied: false,
    gguf_tensors: 310,
    st_tensors: 311,
};

/// Qwen2.5-0.5B-Instruct: 24 layers, hidden 896, FFN 4864, 14 heads / 2 KV
/// heads, QKV biases, vocab 151936. Here the GGUF copy materializes
/// `output.weight` and the Safetensors copy drops `lm_head` — the reverse of
/// Qwen3-0.6B.
const QWEN25: ModelSpec = ModelSpec {
    label: "qwen2.5-0.5b-instruct",
    gguf: &QWEN25_GGUF,
    st: &QWEN25_SAFETENSORS,
    gguf_arch: "qwen2",
    layers: 24,
    hidden: 896,
    intermediate: 4864,
    vocab: 151_936,
    experts: None,
    experts_used: None,
    shared_expert: false,
    st_stack: "model.layers",
    uniform_attention: Some(AttentionKind::Separate),
    gguf_tied: false,
    st_tied: true,
    gguf_tensors: 291,
    st_tensors: 290,
};

/// OLMoE-1B-7B-0125-Instruct: 16 layers, hidden 2048, 16 heads, 64 routed
/// experts of FFN 1024 with 8 active, no shared expert, vocab 50304.
const OLMOE: ModelSpec = ModelSpec {
    label: "olmoe-0125-instruct",
    gguf: &OLMOE_GGUF,
    st: &OLMOE_SAFETENSORS,
    gguf_arch: "olmoe",
    layers: 16,
    hidden: 2048,
    intermediate: 1024,
    vocab: 50_304,
    experts: Some(64),
    experts_used: Some(8),
    shared_expert: false,
    st_stack: "model.layers",
    uniform_attention: Some(AttentionKind::Separate),
    gguf_tied: false,
    st_tied: false,
    gguf_tensors: 195,
    st_tensors: 3219,
};

/// Qwen3.5-0.8B (stretch): a multimodal hybrid — 24 text layers alternating
/// 18 `linear_attention` and 6 `full_attention` (indices 3, 7, 11, 15, 19,
/// 23), a 12-block vision tower, and a 1-layer MTP stack. Linear-attention
/// and vision-only tensors must surface as Unknown, never guessed.
const QWEN35: ModelSpec = ModelSpec {
    label: "qwen3.5-0.8b",
    gguf: &QWEN35_GGUF,
    st: &QWEN35_SAFETENSORS,
    gguf_arch: "qwen35",
    layers: 24,
    hidden: 1024,
    intermediate: 3584,
    vocab: 248_320,
    experts: None,
    experts_used: None,
    shared_expert: false,
    st_stack: "model.language_model.layers",
    uniform_attention: None,
    gguf_tied: true,
    st_tied: true,
    gguf_tensors: 320,
    st_tensors: 488,
};

/// Layers with full (separate Q/K/V/O) attention in Qwen3.5's text tower,
/// from `config.json` `layer_types` (`full_attention_interval: 4`).
const QWEN35_FULL_ATTENTION: [usize; 6] = [3, 7, 11, 15, 19, 23];

fn topology_of(path: &Path) -> TransformerTopology {
    let ckpt = open_checkpoint(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    discover_topology(&ckpt)
}

fn report(label: &str, topo: &TransformerTopology) {
    let s = &topo.summary;
    eprintln!(
        "{label}: arch={:?} layers={} stacks={:?} tensors={} moe={} tied={} hidden={:?} vocab={:?} \
         unknown={} ambiguous={}",
        topo.architecture_hint,
        s.layer_count,
        s.layer_stacks,
        s.tensor_count,
        s.is_moe,
        s.tied_embeddings_hint,
        s.hidden_size_hint,
        s.vocab_size_hint,
        s.unknown_tensors.len(),
        s.ambiguous_tensors.len(),
    );
}

/// Summary-level facts every format test shares.
fn assert_summary(
    topo: &TransformerTopology,
    spec: &ModelSpec,
    tensors: usize,
    tied: bool,
    stack: &str,
    label: &str,
) {
    let s = &topo.summary;
    assert_eq!(s.tensor_count, tensors, "{label}: tensor count");
    assert_eq!(s.layer_count, spec.layers, "{label}: layer count");
    assert_eq!(s.primary_stack.as_deref(), Some(stack), "{label}: stack");
    assert!(s.missing_layers.is_empty(), "{label}: missing layers");
    assert_eq!(s.is_moe, spec.experts.is_some(), "{label}: is_moe");
    assert_eq!(
        s.hidden_size_hint,
        Some(spec.hidden),
        "{label}: hidden hint"
    );
    assert_eq!(s.vocab_size_hint, Some(spec.vocab), "{label}: vocab hint");
    assert_eq!(s.tied_embeddings_hint, tied, "{label}: tied hint");
    assert!(
        s.ambiguous_tensors.is_empty(),
        "{label}: ambiguous {:?}",
        s.ambiguous_tensors
    );
}

/// Per-layer structural facts shared by dense and MoE models. Hybrids
/// (`uniform_attention == None`) get the feed-forward side checked here and
/// the attention side in their dedicated tests.
fn assert_layers(topo: &TransformerTopology, spec: &ModelSpec, label: &str) {
    let layers: Vec<_> = topo.primary_layers().collect();
    assert_eq!(layers.len(), spec.layers, "{label}: primary layers");
    for (i, layer) in layers.iter().enumerate() {
        assert_eq!(layer.index, i, "{label}: contiguous indices");
        if let Some(kind) = spec.uniform_attention {
            assert_eq!(layer.attention, kind, "{label} L{i}");
        }
        assert_eq!(layer.has_shared_expert, spec.shared_expert, "{label} L{i}");
        // A dense MLP next to routed experts only exists in hybrid MoEs;
        // OLMoE is pure-MoE, the rest are all-dense.
        assert_eq!(layer.has_dense_mlp, spec.experts.is_none(), "{label} L{i}");
        match spec.experts {
            Some(n) => {
                assert_eq!(layer.feed_forward, FeedForwardKind::Moe, "{label} L{i}");
                assert!(layer.has_router, "{label} L{i}: router");
                assert_eq!(layer.expert_count, Some(n), "{label} L{i}");
            }
            None => {
                assert_eq!(layer.feed_forward, FeedForwardKind::Gated, "{label} L{i}");
                assert!(!layer.has_router, "{label} L{i}: router");
                assert_eq!(layer.expert_count, None, "{label} L{i}");
            }
        }
    }
}

/// The gate/up role tensors must be `intermediate x hidden` (dense or per
/// expert), evidence that real FFN dims drive the shapes. GGUF stacks
/// experts into a leading expert dimension.
fn assert_intermediate(topo: &TransformerTopology, spec: &ModelSpec, label: &str) {
    let mut seen = false;
    for t in topo.tensors() {
        if t.role != TensorRole::MlpUp {
            continue;
        }
        let want = if t.scope == RoleScope::StackedExperts {
            vec![spec.experts.unwrap_or(1), spec.intermediate, spec.hidden]
        } else {
            vec![spec.intermediate, spec.hidden]
        };
        assert_eq!(t.dims, want, "{label}: mlp_up dims of {}", t.name);
        seen = true;
    }
    assert!(seen, "{label}: no mlp_up tensors");
}

fn assert_no_unresolved(topo: &TransformerTopology, label: &str) {
    let s = &topo.summary;
    assert!(
        s.unknown_tensors.is_empty(),
        "{label}: unknown tensors {:?}",
        s.unknown_tensors
    );
    assert!(
        s.ambiguous_tensors.is_empty(),
        "{label}: ambiguous tensors {:?}",
        s.ambiguous_tensors
    );
}

// ---------------------------------------------------------------------------
// config.json cross-checks (minimal key scan; the files are small and pinned)
// ---------------------------------------------------------------------------

/// First `"key": <uint>` in the file. Quoting the key keeps
/// `"mtp_num_hidden_layers"` from matching `"num_hidden_layers"`, and file
/// order puts `text_config` before `vision_config` for Qwen3.5.
fn config_usize(text: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let pos = text.find(&needle)?;
    let rest = &text[pos + needle.len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| *c == ':' || c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn config_path(fx: &RealFixture, dir: &Path) -> PathBuf {
    fx.file_path(dir, "config.json")
}

/// Layer / expert / shape counts must match the model's own config.
fn assert_config(topo: &TransformerTopology, spec: &ModelSpec, dir: &Path, label: &str) {
    let text = fs::read_to_string(config_path(spec.st, dir))
        .unwrap_or_else(|e| panic!("{label}: read config.json: {e}"));
    for (key, want) in [
        ("num_hidden_layers", spec.layers),
        ("hidden_size", spec.hidden),
        ("vocab_size", spec.vocab),
        ("intermediate_size", spec.intermediate),
    ] {
        assert_eq!(
            config_usize(&text, key),
            Some(want),
            "{label}: config {key}"
        );
    }
    if let Some(n) = spec.experts {
        assert_eq!(
            config_usize(&text, "num_experts"),
            Some(n),
            "{label}: config num_experts"
        );
    }
    let s = &topo.summary;
    assert_eq!(s.layer_count, spec.layers, "{label}: layers vs config");
    assert_eq!(
        s.hidden_size_hint,
        Some(spec.hidden),
        "{label}: hidden vs config"
    );
    assert_eq!(
        s.vocab_size_hint,
        Some(spec.vocab),
        "{label}: vocab vs config"
    );
    assert_intermediate(topo, spec, label);
}

// ---------------------------------------------------------------------------
// GGUF tests (work without the safetensors feature)
// ---------------------------------------------------------------------------

fn check_gguf(spec: &ModelSpec) -> Option<TransformerTopology> {
    let dir = fixture_or_skip(spec.gguf)?;
    let topo = topology_of(&spec.gguf.checkpoint_path(&dir));
    let label = format!("{} gguf", spec.label);
    report(&label, &topo);

    let s = &topo.summary;
    assert_eq!(
        topo.architecture_hint.as_deref(),
        Some(spec.gguf_arch),
        "{label}: architecture"
    );
    assert_eq!(
        topo.declared_layer_count,
        Some(spec.layers),
        "{label}: declared layers"
    );
    assert_eq!(
        topo.declared_expert_count, spec.experts,
        "{label}: declared experts"
    );
    assert_summary(
        &topo,
        spec,
        spec.gguf_tensors,
        spec.gguf_tied,
        "blk",
        &label,
    );
    assert_layers(&topo, spec, &label);
    assert_intermediate(&topo, spec, &label);
    assert_eq!(s.layer_stacks.len(), 1, "{label}: single-tower checkpoint");
    Some(topo)
}

/// Shared-expert presence and expert counts come straight from GGUF
/// metadata (`<arch>.expert_count`, `<arch>.expert_used_count`).
fn check_gguf_metadata(spec: &ModelSpec, dir: &Path, label: &str) {
    let path = spec.gguf.checkpoint_path(dir);
    let layout = load_gguf(&path).unwrap_or_else(|e| panic!("{label}: load_gguf: {e}"));
    assert_eq!(
        layout.metadata.block_count(),
        Some(spec.layers),
        "{label}: block_count"
    );
    assert_eq!(
        layout.metadata.expert_count(),
        spec.experts,
        "{label}: expert_count"
    );
    assert_eq!(
        layout.metadata.expert_used_count(),
        spec.experts_used,
        "{label}: expert_used_count"
    );
}

#[test]
#[ignore = "real weights"]
fn real_qwen3_gguf_topology() {
    let Some(topo) = check_gguf(&QWEN3) else {
        return;
    };
    assert_no_unresolved(&topo, "qwen3 gguf");
}

#[test]
#[ignore = "real weights"]
fn real_qwen25_gguf_topology() {
    let Some(topo) = check_gguf(&QWEN25) else {
        return;
    };
    assert_no_unresolved(&topo, "qwen2.5 gguf");
}

#[test]
#[ignore = "real weights"]
fn real_olmoe_gguf_topology() {
    let Some(topo) = check_gguf(&OLMOE) else {
        return;
    };
    assert_no_unresolved(&topo, "olmoe gguf");
}

/// GGUF metadata counts agree with config.json for the MoE fixture.
#[test]
#[ignore = "real weights"]
fn real_olmoe_gguf_metadata() {
    let Some(dir) = fixture_or_skip(&OLMOE_GGUF) else {
        return;
    };
    check_gguf_metadata(&OLMOE, &dir, "olmoe gguf");
}

/// Qwen3.5 GGUF carries the text tower only. The 6 full-attention layers are
/// fully classified; the 18 linear-attention layers keep `attn_qkv` (the
/// llama.cpp name for the linear in-projection) plus Unknown `ssm_*` /
/// `attn_gate` tensors — Unknown, not guessed.
#[test]
#[ignore = "real weights"]
fn real_qwen35_gguf_topology() {
    let Some(topo) = check_gguf(&QWEN35) else {
        return;
    };
    let label = "qwen3.5 gguf";
    let s = &topo.summary;

    let kinds: BTreeMap<AttentionKind, usize> =
        topo.primary_layers().fold(BTreeMap::new(), |mut m, l| {
            *m.entry(l.attention).or_default() += 1;
            m
        });
    assert_eq!(
        kinds[&AttentionKind::Separate],
        6,
        "{label}: full attention"
    );
    assert_eq!(
        kinds[&AttentionKind::Partial],
        18,
        "{label}: linear attention"
    );
    let separate: Vec<usize> = topo
        .primary_layers()
        .filter(|l| l.attention == AttentionKind::Separate)
        .map(|l| l.index)
        .collect();
    assert_eq!(separate, QWEN35_FULL_ATTENTION, "{label}: attention split");

    // 8 Unknown tensors per linear-attention layer: attn_gate + 7 ssm_*.
    assert_eq!(s.unknown_tensors.len(), 144, "{label}: unknown count");
    for name in &s.unknown_tensors {
        assert!(
            name.contains("attn_gate") || name.contains("ssm"),
            "{label}: unexpected unknown {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// Safetensors tests
// ---------------------------------------------------------------------------

#[cfg(feature = "safetensors")]
fn check_safetensors(spec: &ModelSpec) -> Option<TransformerTopology> {
    let dir = fixture_or_skip(spec.st)?;
    let topo = topology_of(&spec.st.checkpoint_path(&dir));
    let label = format!("{} safetensors", spec.label);
    report(&label, &topo);

    assert_eq!(topo.architecture_hint, None, "{label}: no gguf metadata");
    assert_eq!(topo.declared_layer_count, None, "{label}");
    assert_eq!(topo.declared_expert_count, None, "{label}");
    assert_summary(
        &topo,
        spec,
        spec.st_tensors,
        spec.st_tied,
        spec.st_stack,
        &label,
    );
    assert_layers(&topo, spec, &label);
    assert_config(&topo, spec, &dir, &label);
    Some(topo)
}

#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_qwen3_safetensors_topology() {
    let Some(topo) = check_safetensors(&QWEN3) else {
        return;
    };
    assert_no_unresolved(&topo, "qwen3 safetensors");
}

#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_qwen25_safetensors_topology() {
    let Some(topo) = check_safetensors(&QWEN25) else {
        return;
    };
    assert_no_unresolved(&topo, "qwen2.5 safetensors");
}

#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_olmoe_safetensors_topology() {
    let Some(topo) = check_safetensors(&OLMOE) else {
        return;
    };
    assert_no_unresolved(&topo, "olmoe safetensors");
}

/// Qwen3.5 Safetensors: three stacks — 24-layer text tower (primary),
/// 12-block vision tower, 1-layer MTP. Unknowns are confined to the
/// linear-attention machinery, the vision tower's non-block tensors, and
/// MTP-side tensors; nothing gets a guessed role.
#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_qwen35_safetensors_topology() {
    let Some(topo) = check_safetensors(&QWEN35) else {
        return;
    };
    let label = "qwen3.5 safetensors";
    let s = &topo.summary;

    let mut stacks: BTreeMap<String, usize> = s.layer_stacks.clone();
    assert_eq!(
        stacks.remove("model.language_model.layers"),
        Some(24),
        "{label}: text tower"
    );
    assert_eq!(
        stacks.remove("model.visual.blocks"),
        Some(12),
        "{label}: vision tower"
    );
    assert_eq!(stacks.remove("mtp.layers"), Some(1), "{label}: mtp stack");
    assert!(stacks.is_empty(), "{label}: extra stacks {stacks:?}");

    let kinds: BTreeMap<AttentionKind, usize> =
        topo.primary_layers().fold(BTreeMap::new(), |mut m, l| {
            *m.entry(l.attention).or_default() += 1;
            m
        });
    assert_eq!(
        kinds[&AttentionKind::Separate],
        6,
        "{label}: full attention"
    );
    assert_eq!(
        kinds[&AttentionKind::Absent],
        18,
        "{label}: linear attention"
    );
    let separate: Vec<usize> = topo
        .primary_layers()
        .filter(|l| l.attention == AttentionKind::Separate)
        .map(|l| l.index)
        .collect();
    assert_eq!(separate, QWEN35_FULL_ATTENTION, "{label}: attention split");

    // Every Unknown belongs to the linear-attention machinery, the vision
    // tower, or the MTP stack — no guessed roles on real tensors.
    assert_eq!(s.unknown_tensors.len(), 268, "{label}: unknown count");
    for name in &s.unknown_tensors {
        assert!(
            name.contains("linear_attn") || name.contains("visual") || name.contains("mtp"),
            "{label}: unexpected unknown {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// Cross-format parity
// ---------------------------------------------------------------------------

/// Per layer: role -> set of shapes, with a stacked expert tensor's leading
/// expert dimension removed so it compares equal to one Safetensors expert.
#[cfg(feature = "safetensors")]
fn shape_profile(topo: &TransformerTopology) -> Vec<BTreeMap<TensorRole, BTreeSet<Vec<usize>>>> {
    topo.primary_layers()
        .map(|layer| {
            let mut by_role: BTreeMap<TensorRole, BTreeSet<Vec<usize>>> = BTreeMap::new();
            for t in &layer.tensors {
                let dims = if t.scope == RoleScope::StackedExperts {
                    t.dims[1..].to_vec()
                } else {
                    t.dims.clone()
                };
                by_role.entry(t.role).or_default().insert(dims);
            }
            by_role
        })
        .collect()
}

#[cfg(feature = "safetensors")]
fn layer_kinds(
    topo: &TransformerTopology,
) -> Vec<(
    AttentionKind,
    FeedForwardKind,
    bool,
    Option<usize>,
    bool,
    bool,
)> {
    topo.primary_layers()
        .map(|l| {
            (
                l.attention,
                l.feed_forward,
                l.has_router,
                l.expert_count,
                l.has_shared_expert,
                l.has_dense_mlp,
            )
        })
        .collect()
}

/// GGUF and Safetensors views of the same model must agree on structure:
/// same layer count, per-layer kinds, expert counts, global roles, and
/// per-layer role->shape profiles (dtypes and expert storage layout differ
/// by format design).
#[cfg(feature = "safetensors")]
fn assert_formats_agree(spec: &ModelSpec) {
    let (Some(g), Some(s)) = (check_gguf(spec), check_safetensors(spec)) else {
        return;
    };
    let label = format!("{} parity", spec.label);

    assert_eq!(
        g.summary.layer_count, s.summary.layer_count,
        "{label}: layers"
    );
    assert_eq!(g.summary.is_moe, s.summary.is_moe, "{label}: is_moe");
    assert_eq!(
        g.summary.hidden_size_hint, s.summary.hidden_size_hint,
        "{label}: hidden"
    );
    assert_eq!(
        g.summary.vocab_size_hint, s.summary.vocab_size_hint,
        "{label}: vocab"
    );
    assert_eq!(layer_kinds(&g), layer_kinds(&s), "{label}: layer kinds");
    assert_eq!(shape_profile(&g), shape_profile(&s), "{label}: shapes");

    // Global roles agree up to the documented output-head divergence:
    // one format may materialize the head while the other ties embeddings
    // (each side's `tied` flag was already asserted per format above).
    let global_roles = |t: &TransformerTopology| -> BTreeSet<TensorRole> {
        t.global
            .iter()
            .map(|a| a.role)
            .filter(|r| *r != TensorRole::OutputHead)
            .collect()
    };
    assert_eq!(global_roles(&g), global_roles(&s), "{label}: global roles");

    assert_no_unresolved(&g, &label);
    assert_no_unresolved(&s, &label);
}

#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_qwen3_gguf_safetensors_agree() {
    assert_formats_agree(&QWEN3);
}

#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_qwen25_gguf_safetensors_agree() {
    assert_formats_agree(&QWEN25);
}

#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_olmoe_gguf_safetensors_agree() {
    assert_formats_agree(&OLMOE);
}

/// Qwen3.5 parity is looser by contract: the formats differ in three
/// documented ways — the GGUF copy has no vision tower or MTP stack, names
/// the linear-attention in-projection `attn_qkv` (a real name match, not a
/// guess), and calls the pre-MLP norm `post_attention_norm`. What must
/// agree: the text-tower layer count, which layers are full attention, the
/// attention roles of those layers, embedding dims, and the MoE verdict.
#[cfg(feature = "safetensors")]
#[test]
#[ignore = "real weights"]
fn real_qwen35_gguf_safetensors_agree() {
    let (Some(g), Some(s)) = (check_gguf(&QWEN35), check_safetensors(&QWEN35)) else {
        return;
    };
    let label = "qwen3.5 parity";

    assert_eq!(
        (g.summary.layer_count, s.summary.layer_count),
        (QWEN35.layers, QWEN35.layers),
        "{label}: text layers"
    );
    assert_eq!(g.summary.is_moe, s.summary.is_moe, "{label}: is_moe");
    assert_eq!(
        (g.summary.hidden_size_hint, s.summary.hidden_size_hint),
        (Some(QWEN35.hidden), Some(QWEN35.hidden)),
        "{label}: hidden"
    );
    assert_eq!(
        (g.summary.vocab_size_hint, s.summary.vocab_size_hint),
        (Some(QWEN35.vocab), Some(QWEN35.vocab)),
        "{label}: vocab"
    );
    assert_eq!(
        (
            g.summary.tied_embeddings_hint,
            s.summary.tied_embeddings_hint
        ),
        (true, true),
        "{label}: tied"
    );

    // Same 6-of-24 full-attention split in both formats.
    let separate = |t: &TransformerTopology| -> Vec<usize> {
        t.primary_layers()
            .filter(|l| l.attention == AttentionKind::Separate)
            .map(|l| l.index)
            .collect()
    };
    assert_eq!(
        (separate(&g), separate(&s)),
        (
            QWEN35_FULL_ATTENTION.to_vec(),
            QWEN35_FULL_ATTENTION.to_vec()
        ),
        "{label}: full-attention indices"
    );

    // Full-attention layers carry the same attention roles in both formats.
    const ATTN_ROLES: [TensorRole; 6] = [
        TensorRole::AttentionQuery,
        TensorRole::AttentionKey,
        TensorRole::AttentionValue,
        TensorRole::AttentionOutput,
        TensorRole::AttentionQueryNorm,
        TensorRole::AttentionKeyNorm,
    ];
    for &i in &QWEN35_FULL_ATTENTION {
        for (t, fmt) in [(&g, "gguf"), (&s, "safetensors")] {
            let layer = t.layer(i).unwrap_or_else(|| panic!("{label} L{i} {fmt}"));
            for role in ATTN_ROLES {
                assert!(
                    layer.tensors.iter().any(|a| a.role == role),
                    "{label} L{i} {fmt}: missing {role}"
                );
            }
        }
    }

    // Unknowns stay confined to the documented hybrid machinery.
    for (t, fmt) in [(&g, "gguf"), (&s, "safetensors")] {
        for name in &t.summary.unknown_tensors {
            assert!(
                name.contains("attn_gate")
                    || name.contains("ssm")
                    || name.contains("linear_attn")
                    || name.contains("visual")
                    || name.contains("mtp"),
                "{label} {fmt}: unexpected unknown {name}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fetch helper: always runs, no network (uses a file:// mirror)
// ---------------------------------------------------------------------------

#[test]
fn pins_are_well_formed() {
    let is_hex = |s: &str, n: usize| {
        s.len() == n
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    let mut seen = BTreeSet::new();
    assert_eq!(real_fixtures::ALL.len(), 8, "the RM-2367 table has 8 rows");
    for fx in real_fixtures::ALL {
        assert!(
            is_hex(fx.revision, 40),
            "{}: revision must be a 40-hex commit",
            fx.id
        );
        assert!(seen.insert(fx.id), "duplicate id {}", fx.id);
        let mut weights = 0;
        for f in fx.files {
            assert!(
                is_hex(f.sha256, 64),
                "{} {}: sha256 must be 64 lowercase hex",
                fx.id,
                f.file
            );
            assert!(f.size > 0, "{} {}", fx.id, f.file);
            weights += usize::from(f.is_weights());
            assert_eq!(
                fx.url("https://huggingface.co/", f),
                format!(
                    "https://huggingface.co/{}/resolve/{}/{}",
                    fx.repo, fx.revision, f.file
                )
            );
        }
        assert!(
            weights >= 1 && !fx.license.is_empty(),
            "{}: needs a weight file and a license",
            fx.id
        );
    }
    // Nothing large may live in the repo itself.
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut stack = vec![fixtures];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir).expect("read fixtures").flatten() {
            let meta = entry.metadata().expect("metadata");
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                assert!(
                    meta.len() < 1 << 20,
                    "{} is too large to commit",
                    entry.path().display()
                );
            }
        }
    }
}

#[test]
fn sha256_matches_published_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // 1,000,000 x 'a', fed in awkward chunks that straddle block boundaries.
    let mut h = Sha256::new();
    let chunk = [b'a'; 997];
    let mut left = 1_000_000usize;
    while left > 0 {
        let n = left.min(chunk.len());
        h.update(&chunk[..n]);
        left -= n;
    }
    assert_eq!(
        to_hex(&h.finalize()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn mode_parsing_is_strict() {
    for (raw, want) in [
        (None, Mode::Auto),
        (Some(""), Mode::Auto),
        (Some("AUTO"), Mode::Auto),
        (Some("offline"), Mode::Offline),
        (Some("skip"), Mode::Skip),
        (Some("0"), Mode::Skip),
        (Some("require"), Mode::Require),
    ] {
        assert_eq!(Mode::parse(raw), Ok(want), "{raw:?}");
    }
    assert!(
        Mode::parse(Some("sometimes")).is_err(),
        "typos must not skip silently"
    );
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("engram-fx-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    /// `file://` URL of this directory.
    fn file_url(&self) -> String {
        let p = self.0.to_string_lossy().replace('\\', "/");
        if p.starts_with('/') {
            format!("file://{p}")
        } else {
            format!("file:///{p}")
        }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// A tiny pinned fixture served from `<mirror>/<repo>/resolve/<rev>/<file>`.
/// `id` must be unique across tests: resolution state is process-global.
fn mirrored(
    mirror: &TempDir,
    id: &'static str,
    body: &[u8],
    claimed_sha: Option<&str>,
) -> RealFixture {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let dir = mirror.0.join("org/model/resolve").join(revision);
    fs::create_dir_all(&dir).expect("mkdir origin");
    fs::write(dir.join("w.gguf"), body).expect("write origin");
    let f = FixtureFile {
        file: "w.gguf",
        sha256: leak(claimed_sha.map_or_else(|| sha256_hex(body), str::to_owned)),
        size: body.len() as u64,
    };
    RealFixture {
        id,
        repo: "org/model",
        revision,
        license: "mit",
        files: Box::leak(vec![f].into_boxed_slice()),
    }
}

fn cfg(mode: Mode, cache: &TempDir, mirror: &TempDir) -> Config {
    Config {
        mode,
        fixture_root: None,
        cache_dir: cache.0.clone(),
        base_url: mirror.file_url(),
    }
}

fn cfg_with_root(mode: Mode, cache: &TempDir, root: &TempDir) -> Config {
    Config {
        mode,
        fixture_root: Some(root.0.clone()),
        cache_dir: cache.0.clone(),
        base_url: "file:///nonexistent".to_owned(),
    }
}

fn curl_available() -> bool {
    std::process::Command::new("curl")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

#[test]
fn fetch_downloads_verifies_and_reuses_the_cache() {
    if !curl_available() {
        eprintln!("SKIP: curl not installed");
        return;
    }
    let (mirror, cache) = (TempDir::new("m1"), TempDir::new("c1"));
    let fx = Box::leak(Box::new(mirrored(
        &mirror,
        "synth-dl",
        b"pinned weights",
        None,
    )));

    let Outcome::Ready(dir) = ensure(fx, &cfg(Mode::Auto, &cache, &mirror)) else {
        panic!("first fetch should succeed");
    };
    assert_eq!(fs::read(dir.join("w.gguf")).unwrap(), b"pinned weights");
    assert!(dir.starts_with(&cache.0));

    // Origin gone: the verified cache entry alone must satisfy offline mode.
    fs::remove_dir_all(mirror.0.join("org")).unwrap();
    assert_eq!(
        ensure(fx, &cfg(Mode::Offline, &cache, &mirror)),
        Outcome::Ready(dir.clone())
    );

    // Corrupt the cache (same length): dropped, then offline skips.
    fs::write(dir.join("w.gguf"), b"pinned WEIGHTS").unwrap();
    assert!(matches!(
        ensure(fx, &cfg(Mode::Offline, &cache, &mirror)),
        Outcome::Skipped(_)
    ));
    assert!(
        !dir.join("w.gguf").exists(),
        "corrupt cache entry must be removed"
    );
}

#[test]
fn fetch_skips_when_offline_or_disabled() {
    let (mirror, cache) = (TempDir::new("m2"), TempDir::new("c2"));
    let fx = Box::leak(Box::new(mirrored(&mirror, "synth-skip", b"abc", None)));

    for mode in [Mode::Skip, Mode::Offline] {
        assert!(
            matches!(ensure(fx, &cfg(mode, &cache, &mirror)), Outcome::Skipped(_)),
            "{mode:?} with an empty cache must skip"
        );
    }
    assert!(
        !fx.fixture_dir(&cache.0).exists(),
        "skip modes never download"
    );

    // Unreachable origin in auto mode: skipped, not failed, and nothing left behind.
    fs::remove_dir_all(mirror.0.join("org")).unwrap();
    let outcome = ensure(fx, &cfg(Mode::Auto, &cache, &mirror));
    assert!(matches!(outcome, Outcome::Skipped(_)), "{outcome:?}");
    let parts = fs::read_dir(fx.fixture_dir(&cache.0))
        .map(|d| d.flatten().count())
        .unwrap_or(0);
    assert_eq!(parts, 0, "no partial download may remain");
}

#[test]
fn fixture_root_is_used_first_and_verified() {
    let (root, cache) = (TempDir::new("r3"), TempDir::new("c3"));
    let fx = Box::leak(Box::new(mirrored(
        &TempDir::new("unused-mirror"),
        "synth-root",
        b"abc",
        None,
    )));
    // Stage the file under <root>/<id>/.
    let dir = fx.fixture_dir(&root.0);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("w.gguf"), b"abc").unwrap();

    // No network origin at all: the root alone must satisfy every mode.
    for mode in [Mode::Auto, Mode::Offline, Mode::Require] {
        assert_eq!(
            ensure(fx, &cfg_with_root(mode, &cache, &root)),
            Outcome::Ready(dir.clone()),
            "{mode:?}"
        );
    }
    // Nothing landed in the download cache.
    assert!(!fx.fixture_dir(&cache.0).exists());

    // A staged file with wrong bytes must fail, never fall back silently.
    fs::write(dir.join("w.gguf"), b"xyz").unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        ensure(fx, &cfg_with_root(Mode::Auto, &cache, &root))
    }));
    assert!(result.is_err(), "a wrong staged digest must fail");
}

#[test]
fn fixture_root_incomplete_falls_back_to_cache() {
    let (mirror, root, cache) = (
        TempDir::new("m3b"),
        TempDir::new("r3b"),
        TempDir::new("c3b"),
    );
    if !curl_available() {
        eprintln!("SKIP: curl not installed");
        return;
    }
    let fx = Box::leak(Box::new(mirrored(
        &mirror,
        "synth-partial-root",
        b"abc",
        None,
    )));
    // Root dir exists but is empty: must not count as staged.
    fs::create_dir_all(fx.fixture_dir(&root.0)).unwrap();

    let mut c = cfg(Mode::Auto, &cache, &mirror);
    c.fixture_root = Some(root.0.clone());
    let Outcome::Ready(dir) = ensure(fx, &c) else {
        panic!("mirror download should succeed");
    };
    assert!(dir.starts_with(&cache.0), "fell back to the cache");
}

#[test]
fn require_mode_fails_instead_of_skipping() {
    let (mirror, cache) = (TempDir::new("m4"), TempDir::new("c4"));
    let fx = Box::leak(Box::new(mirrored(&mirror, "synth-require", b"abc", None)));
    fs::remove_dir_all(mirror.0.join("org")).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        ensure(fx, &cfg(Mode::Require, &cache, &mirror))
    }));
    assert!(result.is_err(), "require must not skip");
}

#[test]
fn digest_mismatch_always_fails_and_is_not_cached() {
    if !curl_available() {
        eprintln!("SKIP: curl not installed");
        return;
    }
    let (mirror, cache) = (TempDir::new("m5"), TempDir::new("c5"));
    let wrong = sha256_hex(b"something else entirely");
    let fx = Box::leak(Box::new(mirrored(
        &mirror,
        "synth-digest",
        b"served bytes",
        Some(&wrong),
    )));

    let result = catch_unwind(AssertUnwindSafe(|| {
        ensure(fx, &cfg(Mode::Auto, &cache, &mirror))
    }));
    assert!(result.is_err(), "a wrong digest must fail, not skip");
    assert!(
        !fx.fixture_dir(&cache.0).join("w.gguf").exists(),
        "unverified bytes must not be cached"
    );
}
