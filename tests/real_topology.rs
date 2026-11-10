// SPDX-License-Identifier: MIT OR Apache-2.0

//! Real-model topology discovery over pinned fixtures (RM-2367, RM-1800).
//!
//! Dense (`SmolLM2-135M`) and MoE (`stories15M_MOE`) checkpoints, each in GGUF
//! and Safetensors, are fetched once into a cache, verified against a pinned
//! revision and SHA-256, and fed through [`discover_topology`]. Layer count,
//! expert count, shapes, and dtypes are asserted exactly, and the GGUF and
//! Safetensors views of the same model must agree.
//!
//! No binaries are committed. When a fixture is unavailable (offline, no
//! `curl`, or `ENGRAM_REAL_FIXTURES=skip|offline`) the affected test logs
//! `SKIP ...` and passes; `ENGRAM_REAL_FIXTURES=require` turns that into a
//! failure. See `tests/common/real_fixtures.rs` for the full contract.
//!
//! ```bash
//! cargo test --all-features --test real_topology -- --nocapture
//! ENGRAM_REAL_FIXTURES=skip    cargo test --test real_topology   # never touch the network
//! ENGRAM_REAL_FIXTURES=require cargo test --all-features --test real_topology
//! ```

// The Safetensors expectations are only used when that feature is on.
#![cfg_attr(not(feature = "safetensors"), allow(dead_code))]

#[path = "common/real_fixtures.rs"]
mod real_fixtures;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use engram_parser::analysis::transformer::{
    AttentionKind, FeedForwardKind, RoleAssignment, RoleConfidence, TensorRole,
    TransformerTopology, discover_topology,
};
use engram_parser::{Checkpoint, load_gguf, open_checkpoint};
use real_fixtures::{
    Config, DENSE_GGUF, MOE_GGUF, Mode, Outcome, RealFixture, Sha256, ensure, fixture_or_skip,
    sha256_hex, to_hex,
};
#[cfg(feature = "safetensors")]
use real_fixtures::{DENSE_SAFETENSORS, MOE_SAFETENSORS};

// ---------------------------------------------------------------------------
// Expected structure
// ---------------------------------------------------------------------------

/// `(role, scope, outermost-first dims, dtype label)`.
type Row = (TensorRole, String, Vec<usize>, String);

fn row(role: TensorRole, scope: &str, dims: &[usize], dtype: &str) -> Row {
    (role, scope.to_owned(), dims.to_vec(), dtype.to_owned())
}

fn rows_of(tensors: &[RoleAssignment]) -> Vec<Row> {
    let mut rows: Vec<Row> = tensors
        .iter()
        .map(|t| (t.role, t.scope.to_string(), t.dims.clone(), t.dtype.clone()))
        .collect();
    sort_rows(&mut rows);
    rows
}

fn sorted(mut rows: Vec<Row>) -> Vec<Row> {
    sort_rows(&mut rows);
    rows
}

fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|a, b| (a.0, &a.1, &a.2).cmp(&(b.0, &b.1, &b.2)));
}

/// What the format-specific loader must report, besides per-tensor rows.
struct Expect {
    layers: usize,
    tensor_count: usize,
    hidden: usize,
    vocab: usize,
    tied: bool,
    moe: bool,
    stack: &'static str,
    architecture_hint: Option<&'static str>,
    declared_layers: Option<usize>,
    declared_experts: Option<usize>,
    global: Vec<Row>,
    layer: Vec<Row>,
}

fn assert_topology(topo: &TransformerTopology, want: &Expect, label: &str) {
    let s = &topo.summary;
    assert_eq!(s.layer_count, want.layers, "{label}: layer count");
    assert_eq!(s.tensor_count, want.tensor_count, "{label}: tensor count");
    assert_eq!(
        s.primary_stack.as_deref(),
        Some(want.stack),
        "{label}: stack"
    );
    assert_eq!(s.layer_stacks.len(), 1, "{label}: single tower");
    assert!(s.missing_layers.is_empty(), "{label}: missing layers");
    assert_eq!(s.is_moe, want.moe, "{label}: is_moe");
    assert_eq!(s.hidden_size_hint, Some(want.hidden), "{label}: hidden");
    assert_eq!(s.vocab_size_hint, Some(want.vocab), "{label}: vocab");
    assert_eq!(
        s.tied_embeddings_hint, want.tied,
        "{label}: tied embeddings"
    );
    assert!(
        s.unknown_tensors.is_empty() && s.ambiguous_tensors.is_empty(),
        "{label}: unknown {:?} / ambiguous {:?}",
        s.unknown_tensors,
        s.ambiguous_tensors
    );
    assert_eq!(
        topo.architecture_hint.as_deref(),
        want.architecture_hint,
        "{label}: architecture"
    );
    assert_eq!(topo.declared_layer_count, want.declared_layers, "{label}");
    assert_eq!(topo.declared_expert_count, want.declared_experts, "{label}");
    assert_eq!(
        rows_of(&topo.global),
        sorted(want.global.clone()),
        "{label}: global"
    );

    let layers: Vec<_> = topo.primary_layers().collect();
    assert_eq!(layers.len(), want.layers, "{label}: primary layers");
    let want_layer = sorted(want.layer.clone());
    for (i, layer) in layers.iter().enumerate() {
        assert_eq!(layer.index, i, "{label}: layer indices are contiguous");
        assert_eq!(layer.attention, AttentionKind::Separate, "{label} L{i}");
        assert!(!layer.has_shared_expert, "{label} L{i}");
        assert!(!layer.has_dense_mlp, "{label} L{i}");
        assert_eq!(
            rows_of(&layer.tensors),
            want_layer,
            "{label}: layer {i} rows"
        );
        if want.moe {
            assert_eq!(layer.feed_forward, FeedForwardKind::Moe, "{label} L{i}");
            assert!(layer.has_router, "{label} L{i}: router");
            assert_eq!(layer.expert_count, Some(M_EXPERTS), "{label} L{i}");
        } else {
            assert_eq!(layer.feed_forward, FeedForwardKind::Gated, "{label} L{i}");
            assert!(!layer.has_router, "{label} L{i}: router");
            assert_eq!(layer.expert_count, None, "{label} L{i}");
        }
    }
    assert!(
        topo.tensors().iter().all(|t| matches!(
            t.confidence,
            RoleConfidence::Exact | RoleConfidence::Heuristic
        )),
        "{label}: every tensor has a resolved role"
    );
}

// SmolLM2-135M: hidden 576, 9 heads / 3 KV heads of 64, FFN 1536, vocab 49152, 30 layers.
const D_HIDDEN: usize = 576;
const D_KV: usize = 192;
const D_FFN: usize = 1536;
const D_VOCAB: usize = 49152;
const D_LAYERS: usize = 30;

/// Per-role dtype labels of one dense checkpoint.
struct DenseDtypes {
    embed: &'static str,
    norm: &'static str,
    q: &'static str,
    k: &'static str,
    v: &'static str,
    o: &'static str,
    gate: &'static str,
    up: &'static str,
    down: &'static str,
}

fn dense_expect(
    d: &DenseDtypes,
    stack: &'static str,
    architecture_hint: Option<&'static str>,
    declared_layers: Option<usize>,
) -> Expect {
    let h = D_HIDDEN;
    Expect {
        layers: D_LAYERS,
        tensor_count: 272,
        hidden: h,
        vocab: D_VOCAB,
        tied: true,
        moe: false,
        stack,
        architecture_hint,
        declared_layers,
        declared_experts: None,
        global: vec![
            row(TensorRole::TokenEmbedding, "global", &[D_VOCAB, h], d.embed),
            row(TensorRole::OutputNorm, "global", &[h], d.norm),
        ],
        layer: vec![
            row(TensorRole::AttentionNorm, "layer", &[h], d.norm),
            row(TensorRole::AttentionQuery, "layer", &[h, h], d.q),
            row(TensorRole::AttentionKey, "layer", &[D_KV, h], d.k),
            row(TensorRole::AttentionValue, "layer", &[D_KV, h], d.v),
            row(TensorRole::AttentionOutput, "layer", &[h, h], d.o),
            row(TensorRole::MlpNorm, "layer", &[h], d.norm),
            row(TensorRole::MlpGate, "layer", &[D_FFN, h], d.gate),
            row(TensorRole::MlpUp, "layer", &[D_FFN, h], d.up),
            row(TensorRole::MlpDown, "layer", &[h, D_FFN], d.down),
        ],
    }
}

// stories15M_MOE: hidden 288, 4 experts of FFN 768, vocab 32000, 6 layers.
const M_HIDDEN: usize = 288;
const M_FFN: usize = 768;
const M_VOCAB: usize = 32000;
const M_LAYERS: usize = 6;
const M_EXPERTS: usize = 4;

/// Dtype labels of one MoE checkpoint (router is F32 in both formats).
struct MoeDtypes {
    embed: &'static str,
    norm: &'static str,
    mat: &'static str,
    head: &'static str,
}

fn moe_common_layer(d: &MoeDtypes) -> Vec<Row> {
    let h = M_HIDDEN;
    vec![
        row(TensorRole::AttentionNorm, "layer", &[h], d.norm),
        row(TensorRole::AttentionQuery, "layer", &[h, h], d.mat),
        row(TensorRole::AttentionKey, "layer", &[h, h], d.mat),
        row(TensorRole::AttentionValue, "layer", &[h, h], d.mat),
        row(TensorRole::AttentionOutput, "layer", &[h, h], d.mat),
        row(TensorRole::MlpNorm, "layer", &[h], d.norm),
        row(TensorRole::MoeRouter, "layer", &[M_EXPERTS, h], "F32"),
    ]
}

fn moe_expect(d: &MoeDtypes, stack: &'static str, tensor_count: usize, gguf: bool) -> Expect {
    let h = M_HIDDEN;
    let mut layer = moe_common_layer(d);
    if gguf {
        // llama.cpp stacks the experts: [experts, out, in] outermost-first.
        layer.push(row(
            TensorRole::MlpGate,
            "experts[stacked]",
            &[M_EXPERTS, M_FFN, h],
            d.mat,
        ));
        layer.push(row(
            TensorRole::MlpUp,
            "experts[stacked]",
            &[M_EXPERTS, M_FFN, h],
            d.mat,
        ));
        layer.push(row(
            TensorRole::MlpDown,
            "experts[stacked]",
            &[M_EXPERTS, h, M_FFN],
            d.mat,
        ));
    } else {
        for e in 0..M_EXPERTS {
            let scope = format!("expert[{e}]");
            layer.push(row(TensorRole::MlpGate, &scope, &[M_FFN, h], d.mat));
            layer.push(row(TensorRole::MlpUp, &scope, &[M_FFN, h], d.mat));
            layer.push(row(TensorRole::MlpDown, &scope, &[h, M_FFN], d.mat));
        }
    }
    Expect {
        layers: M_LAYERS,
        tensor_count,
        hidden: h,
        vocab: M_VOCAB,
        tied: false,
        moe: true,
        stack,
        architecture_hint: gguf.then_some("llama"),
        declared_layers: gguf.then_some(M_LAYERS),
        declared_experts: gguf.then_some(M_EXPERTS),
        global: vec![
            row(TensorRole::TokenEmbedding, "global", &[M_VOCAB, h], d.embed),
            row(TensorRole::OutputHead, "global", &[M_VOCAB, h], d.head),
            row(TensorRole::OutputNorm, "global", &[h], d.norm),
        ],
        layer,
    }
}

const DENSE_GGUF_DTYPES: DenseDtypes = DenseDtypes {
    embed: "Q5_0",
    norm: "F32",
    q: "Q8_0",
    k: "Q8_0",
    v: "Q8_0",
    o: "Q8_0",
    gate: "Q5_0",
    up: "Q5_0",
    down: "Q4_K",
};

const DENSE_ST_DTYPES: DenseDtypes = DenseDtypes {
    embed: "BF16",
    norm: "BF16",
    q: "BF16",
    k: "BF16",
    v: "BF16",
    o: "BF16",
    gate: "BF16",
    up: "BF16",
    down: "BF16",
};

const MOE_GGUF_DTYPES: MoeDtypes = MoeDtypes {
    embed: "Q8_0",
    norm: "F32",
    mat: "Q8_0",
    head: "Q8_0",
};

const MOE_ST_DTYPES: MoeDtypes = MoeDtypes {
    embed: "F16",
    norm: "F16",
    mat: "F16",
    head: "F16",
};

fn dense_gguf_expect() -> Expect {
    dense_expect(&DENSE_GGUF_DTYPES, "blk", Some("llama"), Some(D_LAYERS))
}

fn dense_st_expect() -> Expect {
    dense_expect(&DENSE_ST_DTYPES, "model.layers", None, None)
}

fn moe_gguf_expect() -> Expect {
    moe_expect(&MOE_GGUF_DTYPES, "blk", 63, true)
}

fn moe_st_expect() -> Expect {
    moe_expect(&MOE_ST_DTYPES, "model.layers", 117, false)
}

fn topology_of(path: &Path) -> TransformerTopology {
    let ckpt = open_checkpoint(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    discover_topology(&ckpt)
}

// ---------------------------------------------------------------------------
// Real-model tests (skip gracefully when the fixture is unavailable)
// ---------------------------------------------------------------------------

#[test]
fn real_dense_gguf_topology() {
    let Some(path) = fixture_or_skip(&DENSE_GGUF) else {
        return;
    };
    assert_topology(&topology_of(&path), &dense_gguf_expect(), "dense gguf");
}

#[test]
fn real_moe_gguf_topology() {
    let Some(path) = fixture_or_skip(&MOE_GGUF) else {
        return;
    };
    assert_topology(&topology_of(&path), &moe_gguf_expect(), "moe gguf");
}

#[cfg(feature = "safetensors")]
#[test]
fn real_dense_safetensors_topology() {
    let Some(path) = fixture_or_skip(&DENSE_SAFETENSORS) else {
        return;
    };
    assert_topology(&topology_of(&path), &dense_st_expect(), "dense safetensors");
}

#[cfg(feature = "safetensors")]
#[test]
fn real_moe_safetensors_topology() {
    let Some(path) = fixture_or_skip(&MOE_SAFETENSORS) else {
        return;
    };
    assert_topology(&topology_of(&path), &moe_st_expect(), "moe safetensors");
}

/// Per layer: role -> the set of shapes, with a stacked expert tensor's
/// leading expert dimension removed so it compares equal to one expert.
fn shape_profile(topo: &TransformerTopology) -> Vec<BTreeMap<TensorRole, BTreeSet<Vec<usize>>>> {
    topo.primary_layers()
        .map(|layer| {
            let mut by_role: BTreeMap<TensorRole, BTreeSet<Vec<usize>>> = BTreeMap::new();
            for t in &layer.tensors {
                let dims =
                    if t.scope == engram_parser::analysis::transformer::RoleScope::StackedExperts {
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
fn assert_formats_agree(gguf: &TransformerTopology, st: &TransformerTopology, label: &str) {
    let (g, s) = (&gguf.summary, &st.summary);
    assert_eq!(g.layer_count, s.layer_count, "{label}: layers");
    assert_eq!(g.is_moe, s.is_moe, "{label}: is_moe");
    assert_eq!(g.hidden_size_hint, s.hidden_size_hint, "{label}: hidden");
    assert_eq!(g.vocab_size_hint, s.vocab_size_hint, "{label}: vocab");
    assert_eq!(
        g.tied_embeddings_hint, s.tied_embeddings_hint,
        "{label}: tied"
    );
    assert_eq!(g.tensor_count, s.tensor_count, "{label}: tensor count");
    assert_eq!(
        g.role_counts.len(),
        s.role_counts.len(),
        "{label}: role kinds"
    );
    let kinds = |t: &TransformerTopology| -> Vec<_> {
        t.primary_layers()
            .map(|l| (l.attention, l.feed_forward, l.has_router, l.expert_count))
            .collect()
    };
    assert_eq!(kinds(gguf), kinds(st), "{label}: per-layer kinds");
    assert_eq!(
        shape_profile(gguf),
        shape_profile(st),
        "{label}: per-layer shapes"
    );
}

#[cfg(feature = "safetensors")]
#[test]
fn real_dense_gguf_and_safetensors_agree() {
    let (Some(g), Some(s)) = (
        fixture_or_skip(&DENSE_GGUF),
        fixture_or_skip(&DENSE_SAFETENSORS),
    ) else {
        return;
    };
    assert_formats_agree(&topology_of(&g), &topology_of(&s), "dense");
}

#[cfg(feature = "safetensors")]
#[test]
fn real_moe_gguf_and_safetensors_agree() {
    let (Some(g), Some(s)) = (
        fixture_or_skip(&MOE_GGUF),
        fixture_or_skip(&MOE_SAFETENSORS),
    ) else {
        return;
    };
    let (gt, st) = (topology_of(&g), topology_of(&s));
    assert_formats_agree(&gt, &st, "moe");
    // Same expert count even though one format stacks and the other lists them.
    assert_eq!(gt.declared_expert_count, Some(M_EXPERTS));
    assert!(
        gt.primary_layers()
            .all(|l| l.expert_count == Some(M_EXPERTS))
    );
    assert!(
        st.primary_layers()
            .all(|l| l.expert_count == Some(M_EXPERTS))
    );
}

/// Payloads of a real checkpoint decode to finite floats (BF16 from
/// Safetensors; Q8_0 token rows from GGUF).
#[cfg(feature = "safetensors")]
#[test]
fn real_payloads_decode_finite() {
    use engram_parser::dequantize_bf16_bytes;

    if let Some(path) = fixture_or_skip(&DENSE_SAFETENSORS) {
        let ck = open_checkpoint(&path).expect("open dense safetensors");
        let info = ck.tensor("model.norm.weight").expect("final norm");
        assert_eq!(info.dtype.label(), "BF16");
        let out = dequantize_bf16_bytes(&ck.tensor_bytes(&info.name).expect("bytes"))
            .expect("bf16 decode");
        assert_eq!(out.len(), D_HIDDEN);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite BF16 weight");
        assert!(out.iter().any(|v| *v != 0.0), "all-zero norm");
    }
    if let Some(path) = fixture_or_skip(&MOE_GGUF) {
        let layout = load_gguf(&path).expect("load moe gguf");
        let t = layout.tensors.get("token_embd.weight").expect("token_embd");
        let (width, rows) = (t.dims[0], 8usize);
        let row_bytes = engram_parser::packed_row_size(t.ggml_type, width).expect("row size");
        let bytes = layout.tensor_bytes(t).expect("payload");
        let out =
            engram_parser::dequantize_packed(t.dtype, &bytes[..row_bytes * rows], &[width, rows])
                .expect("Q8_0 decode");
        assert_eq!(out.len(), width * rows);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite Q8_0 value");
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
    for fx in real_fixtures::ALL {
        assert!(
            is_hex(fx.revision, 40),
            "{}: revision must be a 40-hex commit",
            fx.id
        );
        assert!(
            is_hex(fx.sha256, 64),
            "{}: sha256 must be 64 lowercase hex",
            fx.id
        );
        assert!(
            fx.size > 0 && !fx.license.is_empty() && !fx.file.is_empty(),
            "{}",
            fx.id
        );
        assert!(seen.insert(fx.id), "duplicate id {}", fx.id);
        assert_eq!(
            fx.url("https://huggingface.co/"),
            format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                fx.repo, fx.revision, fx.file
            )
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
fn mirrored(mirror: &TempDir, body: &[u8], claimed_sha: Option<&str>) -> RealFixture {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let dir = mirror.0.join("org/model/resolve").join(revision);
    fs::create_dir_all(&dir).expect("mkdir origin");
    fs::write(dir.join("w.bin"), body).expect("write origin");
    RealFixture {
        id: "synthetic",
        repo: "org/model",
        revision,
        file: "w.bin",
        sha256: leak(claimed_sha.map_or_else(|| sha256_hex(body), str::to_owned)),
        size: body.len() as u64,
        license: "mit",
    }
}

fn cfg(mode: Mode, cache: &TempDir, mirror: &TempDir) -> Config {
    Config {
        mode,
        cache_dir: cache.0.clone(),
        base_url: mirror.file_url(),
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
    let fx = mirrored(&mirror, b"pinned weights", None);

    let Outcome::Ready(path) = ensure(&fx, &cfg(Mode::Auto, &cache, &mirror)) else {
        panic!("first fetch should succeed");
    };
    assert_eq!(fs::read(&path).unwrap(), b"pinned weights");
    assert!(path.starts_with(&cache.0) && path.to_string_lossy().contains(fx.sha256));

    // Origin gone: the verified cache entry alone must satisfy offline mode.
    fs::remove_dir_all(mirror.0.join("org")).unwrap();
    assert_eq!(
        ensure(&fx, &cfg(Mode::Offline, &cache, &mirror)),
        Outcome::Ready(path.clone())
    );

    // Corrupt the cache (same length): dropped, and offline then skips.
    fs::write(&path, b"pinned WEIGHTS").unwrap();
    assert!(matches!(
        ensure(&fx, &cfg(Mode::Offline, &cache, &mirror)),
        Outcome::Skipped(_)
    ));
    assert!(!path.exists(), "corrupt cache entry must be removed");
}

#[test]
fn fetch_skips_when_offline_or_disabled() {
    let (mirror, cache) = (TempDir::new("m2"), TempDir::new("c2"));
    let fx = mirrored(&mirror, b"abc", None);

    for mode in [Mode::Skip, Mode::Offline] {
        assert!(
            matches!(
                ensure(&fx, &cfg(mode, &cache, &mirror)),
                Outcome::Skipped(_)
            ),
            "{mode:?} with an empty cache must skip"
        );
    }
    assert!(
        !fx.cache_path(&cache.0).exists(),
        "skip modes never download"
    );

    // Unreachable origin in auto mode: skipped, not failed, and nothing left behind.
    fs::remove_dir_all(mirror.0.join("org")).unwrap();
    let outcome = ensure(&fx, &cfg(Mode::Auto, &cache, &mirror));
    assert!(matches!(outcome, Outcome::Skipped(_)), "{outcome:?}");
    let parts = fs::read_dir(fx.cache_path(&cache.0).parent().unwrap())
        .map(|d| d.flatten().count())
        .unwrap_or(0);
    assert_eq!(parts, 0, "no partial download may remain");
}

#[test]
fn require_mode_fails_instead_of_skipping() {
    let (mirror, cache) = (TempDir::new("m3"), TempDir::new("c3"));
    let fx = mirrored(&mirror, b"abc", None);
    fs::remove_dir_all(mirror.0.join("org")).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        ensure(&fx, &cfg(Mode::Require, &cache, &mirror))
    }));
    assert!(result.is_err(), "require must not skip");
}

#[test]
fn digest_mismatch_always_fails_and_is_not_cached() {
    if !curl_available() {
        eprintln!("SKIP: curl not installed");
        return;
    }
    let (mirror, cache) = (TempDir::new("m4"), TempDir::new("c4"));
    let wrong = sha256_hex(b"something else entirely");
    let fx = mirrored(&mirror, b"served bytes", Some(&wrong));
    // Same length so only the digest differs from the pin.
    let fx = RealFixture {
        size: b"served bytes".len() as u64,
        ..fx
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        ensure(&fx, &cfg(Mode::Auto, &cache, &mirror))
    }));
    assert!(result.is_err(), "a wrong digest must fail, not skip");
    assert!(
        !fx.cache_path(&cache.0).exists(),
        "unverified bytes must not be cached"
    );
}
