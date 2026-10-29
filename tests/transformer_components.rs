// SPDX-License-Identifier: MIT OR Apache-2.0

//! Component view golden snapshots and integration checks (RM-1799).

mod common;
use common::*;

use engram_parser::analysis::transformer::{
    ComponentKind, RoleConfidence, TensorRole, discover_topology,
};
use engram_parser::{GgufBackend, TensorDType, TensorInfo, TensorLocation, TensorShape};
use std::fs;
use std::path::Path;

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
    GgufBackend::from_bytes(build_gguf(&kv, &tensors), "mem://dense-components").expect("parse")
}

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
    GgufBackend::from_bytes(build_gguf(&kv, &tensors), "mem://moe-components").expect("parse")
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

fn golden_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden")
        .join(name)
}

#[test]
fn dense_gguf_components_match_golden() {
    let topo = discover_topology(&dense_gguf());
    let json = topo.components().to_json();
    let expected =
        fs::read_to_string(golden_path("transformer_components_dense.json")).expect("golden file");
    assert_eq!(json, expected.trim_end());
}

#[test]
fn moe_gguf_components_match_golden() {
    let topo = discover_topology(&moe_gguf());
    let json = topo.components().to_json();
    let expected =
        fs::read_to_string(golden_path("transformer_components_moe.json")).expect("golden file");
    assert_eq!(json, expected.trim_end());
}

#[test]
fn hf_moe_inventory_builds_expert_and_shared_components() {
    let mut inv = vec![
        mock("model.layers.0.block_sparse_moe.gate.weight", &[3, HIDDEN]),
        mock(
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            &[FFN, HIDDEN],
        ),
        mock(
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            &[FFN, HIDDEN],
        ),
        mock(
            "model.layers.0.block_sparse_moe.experts.1.w2.weight",
            &[HIDDEN, FFN],
        ),
    ];
    inv.push(mock(
        "model.layers.0.self_attn.q_proj.weight",
        &[HIDDEN, HIDDEN],
    ));
    let topo = engram_parser::analysis::transformer::discover_topology_from_parts(
        &inv,
        &engram_parser::CheckpointMetadata::new(),
    );
    let view = topo.components();
    let layer = view.layers.first().expect("layer");
    assert!(
        layer
            .components
            .iter()
            .any(|c| c.kind == ComponentKind::MoeRouter)
    );
    assert!(
        layer
            .components
            .iter()
            .any(|c| c.kind == ComponentKind::SharedExpert)
    );
    assert_eq!(
        layer
            .components
            .iter()
            .filter(|c| c.kind == ComponentKind::RoutedExpert)
            .count(),
        2
    );
}

#[test]
fn unclassified_unknown_and_ambiguous_are_sorted() {
    let inv = vec![
        mock("blk.1.custom_probe.weight", &[3]),
        mock("model.layers.0.mlp.gate.weight", &[FFN, HIDDEN]),
    ];
    let topo = engram_parser::analysis::transformer::discover_topology_from_parts(
        &inv,
        &engram_parser::CheckpointMetadata::new(),
    );
    let view = topo.components();
    assert_eq!(view.unclassified.len(), 2);
    assert_eq!(view.unclassified[0].name, "blk.1.custom_probe.weight");
    assert_eq!(view.unclassified[0].confidence, RoleConfidence::Unknown);
    assert_eq!(view.unclassified[1].name, "model.layers.0.mlp.gate.weight");
    assert_eq!(view.unclassified[1].confidence, RoleConfidence::Ambiguous);
    assert_eq!(view.unclassified[1].role, TensorRole::Unknown);
}

#[test]
fn components_json_is_deterministic() {
    let topo = discover_topology(&moe_gguf());
    let a = topo.components().to_json();
    let b = topo.components().to_json();
    assert_eq!(a, b);
}

/// Regenerate `tests/fixtures/golden/transformer_components_*.json`:
/// `UPDATE_GOLDEN=1 cargo test --test transformer_components update_component_goldens -- --ignored`
#[test]
#[ignore = "run manually to refresh golden JSON"]
fn update_component_goldens() {
    if std::env::var("UPDATE_GOLDEN").ok().as_deref() != Some("1") {
        panic!("set UPDATE_GOLDEN=1 to write golden files");
    }
    let dense = discover_topology(&dense_gguf()).components().to_json();
    let moe = discover_topology(&moe_gguf()).components().to_json();
    fs::write(golden_path("transformer_components_dense.json"), dense).unwrap();
    fs::write(golden_path("transformer_components_moe.json"), moe).unwrap();
}
