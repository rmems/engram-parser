// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transformer topology discovery over GGUF and Safetensors fixtures (RM-1785).

mod common;
use common::*;

use std::collections::BTreeSet;

use engram_parser::analysis::transformer::{
    AttentionKind, FeedForwardKind, RoleConfidence, RoleScope, TensorRole, TransformerTopology,
    discover_topology, discover_topology_from_parts,
};
use engram_parser::{
    Checkpoint, CheckpointMetadata, GgufBackend, MetadataValue, TensorDType, TensorInfo,
    TensorLocation, TensorShape,
};

const HIDDEN: usize = 8;
const FFN: usize = 16;
const VOCAB: usize = 32;

fn f32_tensor(name: &'static str, dims_innermost_first: Vec<usize>) -> TensorSpec {
    let n: usize = dims_innermost_first.iter().product();
    TensorSpec {
        name,
        dims: dims_innermost_first,
        ggml_type: GGML_F32,
        payload: f32_vec_to_le_bytes(&vec![0.0; n]),
    }
}

/// Two-layer dense llama.cpp-style GGUF (dims innermost-first).
fn dense_gguf() -> GgufBackend {
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("llama")),
        ("llama.block_count", KvValue::U32(2)),
    ];
    let mut tensors = vec![
        f32_tensor("token_embd.weight", vec![HIDDEN, VOCAB]),
        f32_tensor("output_norm.weight", vec![HIDDEN]),
        f32_tensor("output.weight", vec![HIDDEN, VOCAB]),
        f32_tensor("rope_freqs.weight", vec![4]),
    ];
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.1.attn_norm.weight",
        "blk.1.ffn_norm.weight",
    ] {
        tensors.push(f32_tensor(name, vec![HIDDEN]));
    }
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
    ] {
        tensors.push(f32_tensor(name, vec![HIDDEN, HIDDEN]));
    }
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
    ] {
        tensors.push(f32_tensor(name, vec![HIDDEN, FFN]));
    }
    tensors.push(f32_tensor("blk.0.ffn_down.weight", vec![FFN, HIDDEN]));
    tensors.push(f32_tensor("blk.1.ffn_down.weight", vec![FFN, HIDDEN]));
    tensors.push(f32_tensor("blk.1.custom_probe.weight", vec![3]));
    GgufBackend::from_bytes(build_gguf(&kv, &tensors), "mem://dense").expect("parse")
}

/// One-layer OLMoE-style GGUF with stacked experts and a router.
fn moe_gguf() -> GgufBackend {
    let n_experts = 4;
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
        ("olmoe.block_count", KvValue::U32(1)),
        ("olmoe.expert_count", KvValue::U32(n_experts as u32)),
    ];
    let tensors = vec![
        f32_tensor("token_embd.weight", vec![HIDDEN, VOCAB]),
        f32_tensor("output_norm.weight", vec![HIDDEN]),
        f32_tensor("blk.0.attn_norm.weight", vec![HIDDEN]),
        f32_tensor("blk.0.attn_qkv.weight", vec![HIDDEN, 3 * HIDDEN]),
        f32_tensor("blk.0.attn_output.weight", vec![HIDDEN, HIDDEN]),
        f32_tensor("blk.0.ffn_norm.weight", vec![HIDDEN]),
        f32_tensor("blk.0.ffn_gate_inp.weight", vec![HIDDEN, n_experts]),
        f32_tensor("blk.0.ffn_gate_exps.weight", vec![HIDDEN, FFN, n_experts]),
        f32_tensor("blk.0.ffn_up_exps.weight", vec![HIDDEN, FFN, n_experts]),
        f32_tensor("blk.0.ffn_down_exps.weight", vec![FFN, HIDDEN, n_experts]),
    ];
    GgufBackend::from_bytes(build_gguf(&kv, &tensors), "mem://moe").expect("parse")
}

fn mock(name: &str, dims: &[usize]) -> TensorInfo {
    TensorInfo::new(
        name,
        TensorDType::BF16,
        TensorShape::outermost_first(dims.to_vec()),
        dims.iter().product::<usize>() * 2,
        TensorLocation::new("model.safetensors", 0, None),
    )
}

/// Hugging Face Llama-style names as a bare inventory (what the
/// Safetensors backend yields, minus file I/O).
fn hf_dense_inventory() -> Vec<TensorInfo> {
    let mut v = vec![
        mock("model.embed_tokens.weight", &[VOCAB, HIDDEN]),
        mock("model.norm.weight", &[HIDDEN]),
    ];
    for l in 0..2 {
        for (suffix, dims) in [
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("self_attn.q_proj.weight", vec![HIDDEN, HIDDEN]),
            ("self_attn.k_proj.weight", vec![HIDDEN, HIDDEN]),
            ("self_attn.v_proj.weight", vec![HIDDEN, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, HIDDEN]),
            ("mlp.gate_proj.weight", vec![FFN, HIDDEN]),
            ("mlp.up_proj.weight", vec![FFN, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, FFN]),
        ] {
            v.push(mock(&format!("model.layers.{l}.{suffix}"), &dims));
        }
    }
    v
}

/// Mixtral-style per-expert tensors plus a Qwen-style shared expert.
fn hf_moe_inventory() -> Vec<TensorInfo> {
    let mut v = vec![
        mock("model.embed_tokens.weight", &[VOCAB, HIDDEN]),
        mock("lm_head.weight", &[VOCAB, HIDDEN]),
        mock("model.norm.weight", &[HIDDEN]),
        mock("model.layers.0.input_layernorm.weight", &[HIDDEN]),
        mock("model.layers.0.self_attn.q_proj.weight", &[HIDDEN, HIDDEN]),
        mock("model.layers.0.self_attn.k_proj.weight", &[HIDDEN, HIDDEN]),
        mock("model.layers.0.self_attn.v_proj.weight", &[HIDDEN, HIDDEN]),
        mock("model.layers.0.self_attn.o_proj.weight", &[HIDDEN, HIDDEN]),
        mock("model.layers.0.post_attention_layernorm.weight", &[HIDDEN]),
        mock("model.layers.0.block_sparse_moe.gate.weight", &[3, HIDDEN]),
        mock(
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            &[FFN, HIDDEN],
        ),
        mock(
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            &[HIDDEN, FFN],
        ),
        mock(
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            &[FFN, HIDDEN],
        ),
    ];
    for e in 0..3 {
        for w in ["w1", "w2", "w3"] {
            v.push(mock(
                &format!("model.layers.0.block_sparse_moe.experts.{e}.{w}.weight"),
                &[FFN, HIDDEN],
            ));
        }
    }
    v
}

fn roles_of(topo: &TransformerTopology, layer: usize) -> BTreeSet<TensorRole> {
    topo.layer(layer).expect("layer").roles()
}

#[test]
fn dense_gguf_classifies_core_roles() {
    use TensorRole::*;
    let ckpt = dense_gguf();
    let topo = discover_topology(&ckpt);

    assert_eq!(topo.architecture_hint.as_deref(), Some("llama"));
    assert_eq!(topo.declared_layer_count, Some(2));
    assert_eq!(topo.summary.layer_count, 2);
    assert!(topo.summary.missing_layers.is_empty());
    assert!(!topo.summary.is_moe);
    assert_eq!(topo.summary.hidden_size_hint, Some(HIDDEN));
    assert_eq!(topo.summary.vocab_size_hint, Some(VOCAB));
    assert!(!topo.summary.tied_embeddings_hint);

    let global: BTreeSet<TensorRole> = topo.global.iter().map(|t| t.role).collect();
    assert_eq!(
        global,
        [TokenEmbedding, OutputNorm, OutputHead, RotaryFrequencies]
            .into_iter()
            .collect()
    );

    let expected: BTreeSet<TensorRole> = [
        AttentionNorm,
        AttentionQuery,
        AttentionKey,
        AttentionValue,
        AttentionOutput,
        MlpNorm,
        MlpGate,
        MlpUp,
        MlpDown,
    ]
    .into_iter()
    .collect();
    assert_eq!(roles_of(&topo, 0), expected);
    let mut with_unknown = expected.clone();
    with_unknown.insert(Unknown);
    assert_eq!(roles_of(&topo, 1), with_unknown);

    for layer in &topo.layers {
        assert_eq!(layer.attention, AttentionKind::Separate);
        assert_eq!(layer.feed_forward, FeedForwardKind::Gated);
        assert!(!layer.has_router);
        assert_eq!(layer.expert_count, None);
    }

    // Unknown tensor is represented, not dropped or guessed.
    let probe = topo.tensor("blk.1.custom_probe.weight").expect("probe");
    assert_eq!(probe.role, Unknown);
    assert_eq!(probe.confidence, RoleConfidence::Unknown);
    assert_eq!(probe.scope, RoleScope::Layer);
    assert_eq!(probe.dims, vec![3]);
    assert_eq!(
        topo.summary.unknown_tensors,
        vec!["blk.1.custom_probe.weight".to_string()]
    );
    assert_eq!(topo.summary.tensor_count, ckpt.tensors().len());
    let counted: usize = topo.summary.role_counts.values().sum();
    assert_eq!(counted, topo.summary.tensor_count);
}

#[test]
fn moe_gguf_shares_the_topology_layer() {
    use TensorRole::*;
    let ckpt = moe_gguf();
    let topo = discover_topology(&ckpt);

    assert_eq!(topo.architecture_hint.as_deref(), Some("olmoe"));
    assert_eq!(topo.declared_expert_count, Some(4));
    assert!(topo.summary.is_moe);
    assert!(topo.summary.tied_embeddings_hint);

    let layer = topo.layer(0).expect("layer 0");
    assert_eq!(layer.attention, AttentionKind::FusedQkv);
    assert_eq!(layer.feed_forward, FeedForwardKind::Moe);
    assert!(layer.has_router);
    assert_eq!(layer.expert_count, Some(4));
    assert!(!layer.has_shared_expert);
    assert!(!layer.has_dense_mlp);

    // Dense roles and MoE roles coexist in the same layer record.
    assert!(layer.roles().contains(&AttentionQkv));
    assert!(layer.roles().contains(&MoeRouter));
    assert_eq!(
        layer.expert_roles(),
        [MlpGate, MlpUp, MlpDown].into_iter().collect()
    );
    let gate = topo.tensor("blk.0.ffn_gate_exps.weight").expect("exps");
    assert_eq!(gate.scope, RoleScope::StackedExperts);
    assert_eq!(gate.dims, vec![4, FFN, HIDDEN]);
    assert_eq!(gate.confidence, RoleConfidence::Exact);
}

#[test]
fn hf_dense_inventory_matches_gguf_roles() {
    let gguf = discover_topology(&dense_gguf());
    let hf = discover_topology_from_parts(&hf_dense_inventory(), &CheckpointMetadata::new());

    assert_eq!(hf.architecture_hint, None);
    assert_eq!(hf.declared_layer_count, None);
    assert_eq!(hf.summary.layer_count, 2);
    assert!(hf.summary.tied_embeddings_hint);
    assert_eq!(hf.summary.hidden_size_hint, Some(HIDDEN));

    // Same structural reading of both naming conventions.
    assert_eq!(roles_of(&hf, 0), roles_of(&gguf, 0));
    for (a, b) in hf.layers.iter().zip(gguf.layers.iter()) {
        assert_eq!(a.index, b.index);
        assert_eq!(a.attention, b.attention);
        assert_eq!(a.feed_forward, b.feed_forward);
    }
    let norm = hf
        .tensor("model.layers.0.post_attention_layernorm.weight")
        .expect("norm");
    assert_eq!(norm.role, TensorRole::MlpNorm);
    assert_eq!(norm.confidence, RoleConfidence::Heuristic);
}

#[test]
fn gemma2_sibling_norm_resolves_post_attention_layernorm() {
    let inv = vec![
        mock("model.layers.0.input_layernorm.weight", &[HIDDEN]),
        mock("model.layers.0.post_attention_layernorm.weight", &[HIDDEN]),
        mock("model.layers.0.pre_feedforward_layernorm.weight", &[HIDDEN]),
        mock(
            "model.layers.0.post_feedforward_layernorm.weight",
            &[HIDDEN],
        ),
    ];
    let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
    let post_attn = topo
        .tensor("model.layers.0.post_attention_layernorm.weight")
        .unwrap();
    assert_eq!(post_attn.role, TensorRole::PostAttentionNorm);
    assert_eq!(post_attn.confidence, RoleConfidence::Heuristic);
    assert!(post_attn.evidence.contains(&"resolved_by_sibling_mlp_norm"));
    assert_eq!(
        topo.tensor("model.layers.0.post_feedforward_layernorm.weight")
            .unwrap()
            .role,
        TensorRole::PostMlpNorm
    );
}

#[test]
fn hf_moe_inventory_counts_experts_and_shared_expert() {
    use TensorRole::*;
    let topo = discover_topology_from_parts(&hf_moe_inventory(), &CheckpointMetadata::new());
    assert!(topo.summary.is_moe);
    assert!(!topo.summary.tied_embeddings_hint);

    let layer = topo.layer(0).unwrap();
    assert_eq!(layer.feed_forward, FeedForwardKind::Moe);
    assert_eq!(layer.attention, AttentionKind::Separate);
    assert!(layer.has_router);
    assert_eq!(layer.expert_count, Some(3));
    assert!(layer.has_shared_expert);

    let w1 = topo
        .tensor("model.layers.0.block_sparse_moe.experts.2.w1.weight")
        .unwrap();
    assert_eq!((w1.role, w1.scope), (MlpGate, RoleScope::Expert(2)));
    assert_eq!(w1.confidence, RoleConfidence::Heuristic);
    assert!(w1.evidence.contains(&"w1_llama_convention_gate"));

    let shared = topo
        .tensor("model.layers.0.mlp.shared_expert.gate_proj.weight")
        .unwrap();
    assert_eq!(
        (shared.role, shared.scope),
        (MlpGate, RoleScope::SharedExpert)
    );

    let router = topo
        .tensor("model.layers.0.block_sparse_moe.gate.weight")
        .unwrap();
    assert_eq!(router.role, MoeRouter);
    assert_eq!(router.confidence, RoleConfidence::Exact);
}

#[test]
fn ambiguous_gate_is_preserved_and_resolved_only_with_evidence() {
    let dense_with_gate = vec![
        mock("model.layers.0.mlp.gate.weight", &[FFN, HIDDEN]),
        mock("model.layers.0.mlp.up_proj.weight", &[FFN, HIDDEN]),
        mock("model.layers.0.mlp.down_proj.weight", &[HIDDEN, FFN]),
    ];
    let topo = discover_topology_from_parts(&dense_with_gate, &CheckpointMetadata::new());
    let gate = topo.tensor("model.layers.0.mlp.gate.weight").unwrap();
    assert_eq!(gate.role, TensorRole::Unknown);
    assert_eq!(gate.confidence, RoleConfidence::Ambiguous);
    assert_eq!(
        gate.alternatives,
        vec![TensorRole::MlpGate, TensorRole::MoeRouter]
    );
    assert_eq!(
        topo.summary.ambiguous_tensors,
        vec!["model.layers.0.mlp.gate.weight".to_string()]
    );
    // Without a resolved gate the dense MLP is only Plain (up + down).
    assert_eq!(topo.layer(0).unwrap().feed_forward, FeedForwardKind::Plain);
    assert!(!topo.summary.is_moe);

    // The same name next to expert tensors becomes a router (heuristic).
    let mut with_experts = dense_with_gate.clone();
    with_experts.push(mock(
        "model.layers.0.mlp.experts.0.up_proj.weight",
        &[FFN, HIDDEN],
    ));
    let topo = discover_topology_from_parts(&with_experts, &CheckpointMetadata::new());
    let gate = topo.tensor("model.layers.0.mlp.gate.weight").unwrap();
    assert_eq!(gate.role, TensorRole::MoeRouter);
    assert_eq!(gate.confidence, RoleConfidence::Heuristic);
    assert!(topo.summary.ambiguous_tensors.is_empty());
    let layer = topo.layer(0).unwrap();
    assert_eq!(layer.feed_forward, FeedForwardKind::Moe);
    assert!(layer.has_dense_mlp);
}

#[test]
fn gpt2_fused_qkv_and_plain_mlp() {
    let inv = vec![
        mock("wte.weight", &[VOCAB, HIDDEN]),
        mock("wpe.weight", &[64, HIDDEN]),
        mock("h.0.ln_1.weight", &[HIDDEN]),
        mock("h.0.attn.c_attn.weight", &[HIDDEN, 3 * HIDDEN]),
        mock("h.0.attn.c_proj.weight", &[HIDDEN, HIDDEN]),
        mock("h.0.ln_2.weight", &[HIDDEN]),
        mock("h.0.mlp.c_fc.weight", &[HIDDEN, FFN]),
        mock("h.0.mlp.c_proj.weight", &[FFN, HIDDEN]),
        mock("ln_f.weight", &[HIDDEN]),
    ];
    let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
    let layer = topo.layer(0).unwrap();
    assert_eq!(layer.attention, AttentionKind::FusedQkv);
    assert_eq!(layer.feed_forward, FeedForwardKind::Plain);
    assert_eq!(
        topo.tensor("wpe.weight").unwrap().role,
        TensorRole::PositionEmbedding
    );
    assert!(topo.summary.unknown_tensors.is_empty());
}

#[test]
fn partial_layers_and_gaps_are_reported() {
    let inv = vec![
        mock("model.layers.0.self_attn.q_proj.weight", &[HIDDEN, HIDDEN]),
        mock("model.layers.2.mlp.up_proj.weight", &[FFN, HIDDEN]),
    ];
    let mut meta = CheckpointMetadata::new();
    meta.insert(
        "general.architecture",
        MetadataValue::String("llama".into()),
    );
    meta.insert("llama.block_count", MetadataValue::UInt(4));
    let topo = discover_topology_from_parts(&inv, &meta);
    assert_eq!(topo.declared_layer_count, Some(4));
    assert_eq!(topo.summary.layer_count, 2);
    assert_eq!(topo.summary.missing_layers, vec![1, 3]);
    assert_eq!(topo.layer(0).unwrap().attention, AttentionKind::Partial);
    assert_eq!(topo.layer(0).unwrap().feed_forward, FeedForwardKind::Absent);
    assert_eq!(topo.layer(2).unwrap().attention, AttentionKind::Absent);
    assert_eq!(
        topo.layer(2).unwrap().feed_forward,
        FeedForwardKind::Partial
    );
}

#[test]
fn output_is_deterministic_and_order_independent() {
    let forward = hf_moe_inventory();
    let mut reversed = forward.clone();
    reversed.reverse();
    let mut interleaved = forward.clone();
    interleaved.rotate_left(5);

    let a = discover_topology_from_parts(&forward, &CheckpointMetadata::new());
    let b = discover_topology_from_parts(&reversed, &CheckpointMetadata::new());
    let c = discover_topology_from_parts(&interleaved, &CheckpointMetadata::new());
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_eq!(a.to_string(), b.to_string());
    assert_eq!(a.to_json(), c.to_json());

    // tensors() is sorted and complete.
    let names: Vec<&str> = a.tensors().iter().map(|t| t.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
    assert_eq!(names.len(), forward.len());
}

#[test]
fn json_is_well_formed_and_carries_roles() {
    let topo = discover_topology(&moe_gguf());
    let json = topo.to_json();
    assert!(json.starts_with('{') && json.ends_with('}'));
    let opens = json.matches(['{', '[']).count();
    let closes = json.matches(['}', ']']).count();
    assert_eq!(opens, closes);
    assert!(json.contains(r#""architecture_hint":"olmoe""#));
    assert!(json.contains(r#""is_moe":true"#));
    assert!(json.contains(r#""role":"moe_router""#));
    assert!(json.contains(r#""scope":"experts[stacked]""#));
    assert!(json.contains(r#""expert_count":4"#));

    let text = topo.to_string();
    assert!(text.contains("architecture: olmoe"));
    assert!(text.contains("layer 0: attention=fused_qkv feed_forward=moe router experts=4"));
}

#[test]
fn empty_inventory_is_valid() {
    let topo = discover_topology_from_parts(&[], &CheckpointMetadata::new());
    assert!(topo.global.is_empty());
    assert!(topo.layers.is_empty());
    assert_eq!(topo.summary.tensor_count, 0);
    assert!(topo.summary.missing_layers.is_empty());
    assert!(!topo.summary.tied_embeddings_hint);
    assert_eq!(
        topo.to_json().matches('{').count(),
        topo.to_json().matches('}').count()
    );
}

#[cfg(feature = "safetensors")]
mod safetensors_backend {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir(PathBuf);

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(name: &str) -> TestDir {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("engram-topo-{name}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        TestDir(path)
    }

    /// Write a real Safetensors file holding the HF dense inventory (BF16 zeros).
    fn write_hf_dense(dir: &std::path::Path) -> PathBuf {
        let inv = hf_dense_inventory();
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (i, t) in inv.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let end = offset + t.byte_len;
            header.push_str(&format!(
                r#""{}":{{"dtype":"BF16","shape":{:?},"data_offsets":[{offset},{end}]}}"#,
                t.name,
                t.shape.dims()
            ));
            offset = end;
        }
        header.push('}');
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend(std::iter::repeat_n(0u8, offset));
        let path = dir.join("model.safetensors");
        fs::write(&path, out).unwrap();
        path
    }

    #[test]
    fn safetensors_checkpoint_matches_gguf_topology_shape() {
        let dir = temp_dir("hf-dense");
        let path = write_hf_dense(&dir.0);
        let st = engram_parser::open_checkpoint(&path).unwrap();
        let st_topo = discover_topology(&st);
        let gguf_topo = discover_topology(&dense_gguf());

        assert_eq!(st.format(), engram_parser::CheckpointFormat::Safetensors);
        assert_eq!(st_topo.summary.layer_count, gguf_topo.summary.layer_count);
        assert_eq!(roles_of(&st_topo, 0), roles_of(&gguf_topo, 0));
        for (a, b) in st_topo.layers.iter().zip(gguf_topo.layers.iter()) {
            assert_eq!((a.attention, a.feed_forward), (b.attention, b.feed_forward));
        }
        assert_eq!(st_topo.summary.hidden_size_hint, Some(HIDDEN));
        assert_eq!(
            st_topo.tensor("model.embed_tokens.weight").unwrap().dtype,
            "BF16"
        );
        // Same inventory through the backend and as bare parts.
        let parts = discover_topology_from_parts(st.tensors(), st.metadata());
        assert_eq!(parts, st_topo);
    }
}
