// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public API surface stability tests (RM-1792).
//!
//! Proves that the canonical `engram_parser::analysis::moe` path and the
//! legacy compatibility paths (`engram_parser::moe::*` and the
//! crate-root re-exports) resolve to the same single implementation:
//! identical successful results and identical errors.

mod common;
use common::*;

use engram_parser::{ParserError, parse_bytes};

/// Stacked-convention fixture: 3 experts, each a 4x2 f32 matrix.
/// Gate/up/down slices are filled with distinct per-expert constants.
fn stacked_layout() -> engram_parser::GgufLayout {
    let inner = 4usize;
    let outer = 2usize;
    let n_experts = 3usize;
    let per_expert = inner * outer;

    let mut gate = vec![0.0f32; n_experts * per_expert];
    let mut up = vec![0.0f32; n_experts * per_expert];
    let mut down = vec![0.0f32; n_experts * per_expert];
    for e in 0..n_experts {
        let base = e * per_expert;
        for slot in base..base + per_expert {
            gate[slot] = (e as f32) + 0.1;
            up[slot] = (e as f32) + 0.2;
            down[slot] = (e as f32) + 0.3;
        }
    }

    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
        ("olmoe.expert_count", KvValue::U32(3)),
        ("olmoe.block_count", KvValue::U32(1)),
    ];
    let dims = vec![inner, outer, n_experts];
    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate_exps.weight",
            dims: dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&gate),
        },
        TensorSpec {
            name: "blk.0.ffn_up_exps.weight",
            dims: dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&up),
        },
        TensorSpec {
            name: "blk.0.ffn_down_exps.weight",
            dims: dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&down),
        },
    ];
    parse_bytes(build_gguf(&kv, &tensors), "mem://stacked".into()).expect("parse")
}

/// Packed stacked-convention fixture: Q8_0 blocks (32 elements / 34
/// bytes per block), 2 experts. Exercises packed byte slicing, not just
/// f32 layouts.
fn packed_stacked_layout() -> engram_parser::GgufLayout {
    let n_experts = 2usize;
    // dims innermost-first: [32, 1, n_experts] -> 32 Q8_0 values per expert.
    let dims = vec![32, 1, n_experts];
    // 34 bytes per expert; tag each expert's slice with a distinct byte.
    let payload: Vec<u8> = (0..n_experts * 34)
        .map(|i| ((i / 34) as u8) * 10 + (i % 34) as u8)
        .collect();

    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
    ];
    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims,
        ggml_type: GGML_Q8_0,
        payload,
    }];
    parse_bytes(build_gguf(&kv, &tensors), "mem://packed".into()).expect("parse")
}

/// A whole expert tensor is byte-identical across canonical and
/// compatibility paths.
fn weights_equal(a: &engram_parser::MoeExpertWeights, b: &engram_parser::MoeExpertWeights) {
    assert_eq!(a.block, b.block);
    assert_eq!(a.expert, b.expert);
    for (x, y) in [(&a.gate, &b.gate), (&a.up, &b.up), (&a.down, &b.down)] {
        match (x, y) {
            (Some(x), Some(y)) => {
                assert_eq!(x.source_name, y.source_name);
                assert_eq!(x.dims, y.dims);
                assert_eq!(x.dtype, y.dtype);
                assert_eq!(x.ggml_type, y.ggml_type);
                assert_eq!(x.bytes, y.bytes);
                assert_eq!(x.stacked_slice, y.stacked_slice);
            }
            (None, None) => {}
            _ => panic!("projection presence differs"),
        }
    }
}

#[test]
fn canonical_analysis_moe_path_is_public() {
    use engram_parser::analysis::moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};

    let layout = stacked_layout();
    let pairs = list_experts(&layout);
    assert_eq!(pairs, vec![(0, 0), (0, 1), (0, 2)]);
    let w: MoeExpertWeights = extract_expert(&layout, 0, 1).expect("extract");
    let gate: &RawTensor = w.gate.as_ref().expect("gate");
    assert_eq!(gate.dims, vec![4, 2]);
    assert_eq!(gate.bytes.len(), 4 * 2 * 4);
    assert!(gate.stacked_slice);
    assert!(w.is_complete());
}

#[test]
fn legacy_moe_module_path_still_compiles() {
    use engram_parser::moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};

    let layout = stacked_layout();
    let pairs = list_experts(&layout);
    assert_eq!(pairs, vec![(0, 0), (0, 1), (0, 2)]);
    let w: MoeExpertWeights = extract_expert(&layout, 0, 0).expect("extract");
    let gate: &RawTensor = w.gate.as_ref().expect("gate");
    assert_eq!(gate.dims, vec![4, 2]);
}

#[test]
fn legacy_crate_root_path_still_compiles() {
    use engram_parser::{MoeExpertWeights, RawTensor, extract_expert, list_experts};

    let layout = stacked_layout();
    let pairs = list_experts(&layout);
    assert_eq!(pairs, vec![(0, 0), (0, 1), (0, 2)]);
    let w: MoeExpertWeights = extract_expert(&layout, 0, 2).expect("extract");
    let gate: &RawTensor = w.gate.as_ref().expect("gate");
    assert_eq!(gate.dims, vec![4, 2]);
}

#[test]
fn canonical_checkpoint_path_is_primary() {
    use engram_parser::checkpoint::{Checkpoint, TensorInfo};
    use engram_parser::{CheckpointFormat, TensorDType};

    let layout = stacked_layout();
    let backend = engram_parser::GgufBackend::from_layout(layout);
    let ckpt: &dyn Checkpoint = &backend;
    assert_eq!(ckpt.format(), CheckpointFormat::Gguf);
    let t: &TensorInfo = ckpt.tensor("blk.0.ffn_gate_exps.weight").expect("tensor");
    assert!(matches!(t.dtype, TensorDType::F32));
}

#[test]
fn canonical_and_legacy_paths_give_identical_results() {
    use engram_parser::analysis::moe as canonical;
    use engram_parser::moe as legacy;

    let layout = stacked_layout();

    // list_experts identical.
    assert_eq!(
        canonical::list_experts(&layout),
        legacy::list_experts(&layout)
    );
    assert_eq!(
        canonical::list_experts(&layout),
        engram_parser::list_experts(&layout)
    );

    // extract_expert identical across all experts.
    for expert in 0..3 {
        let canon = canonical::extract_expert(&layout, 0, expert).expect("canonical");
        let leg = legacy::extract_expert(&layout, 0, expert).expect("legacy");
        let root = engram_parser::extract_expert(&layout, 0, expert).expect("root");
        weights_equal(&canon, &leg);
        weights_equal(&canon, &root);
    }
}

#[test]
fn canonical_and_legacy_paths_give_identical_packed_slices() {
    use engram_parser::analysis::moe as canonical;
    use engram_parser::moe as legacy;

    let layout = packed_stacked_layout();
    assert_eq!(canonical::list_experts(&layout), vec![(0, 0), (0, 1)]);

    for expert in 0..2 {
        let canon = canonical::extract_expert(&layout, 0, expert).expect("canonical");
        let leg = legacy::extract_expert(&layout, 0, expert).expect("legacy");
        weights_equal(&canon, &leg);

        let gate = canon.gate.expect("gate");
        // Q8_0: 32 values -> one 34-byte block per expert.
        assert_eq!(gate.bytes.len(), 34);
        assert_eq!(gate.ggml_type, GGML_Q8_0);
        assert!(gate.stacked_slice);
        // First byte of each expert slice carries the per-expert tag.
        assert_eq!(gate.bytes[0], (expert as u8) * 10);
    }
}

#[test]
fn canonical_and_legacy_paths_give_identical_errors() {
    use engram_parser::analysis::moe as canonical;
    use engram_parser::moe as legacy;

    let layout = stacked_layout();

    // Expert index out of range on a stacked tensor.
    let canon_err = canonical::extract_expert(&layout, 0, 5).unwrap_err();
    let legacy_err = legacy::extract_expert(&layout, 0, 5).unwrap_err();
    let root_err = engram_parser::extract_expert(&layout, 0, 5).unwrap_err();
    match canon_err {
        ParserError::ExpertOutOfRange {
            block,
            expert,
            available,
        } => {
            assert_eq!((block, expert, available), (0, 5, 3));
        }
        other => panic!("expected ExpertOutOfRange, got {other:?}"),
    }
    assert_eq!(format!("{canon_err:?}"), format!("{legacy_err:?}"));
    assert_eq!(format!("{canon_err:?}"), format!("{root_err:?}"));

    // Block with no expert tensors at all -> MissingTensor.
    let canon_err = canonical::extract_expert(&layout, 7, 0).unwrap_err();
    let legacy_err = legacy::extract_expert(&layout, 7, 0).unwrap_err();
    match canon_err {
        ParserError::MissingTensor { .. } => {}
        other => panic!("expected MissingTensor, got {other:?}"),
    }
    assert_eq!(format!("{canon_err:?}"), format!("{legacy_err:?}"));
}

#[test]
fn legacy_module_is_a_reexport_not_a_copy() {
    // Same function items through both paths: re-exporting the same
    // implementation rather than duplicated logic.
    use engram_parser::analysis::moe as canonical;
    use engram_parser::moe as legacy;

    let canon_fn: fn(&engram_parser::GgufLayout) -> Vec<(usize, usize)> = canonical::list_experts;
    let legacy_fn: fn(&engram_parser::GgufLayout) -> Vec<(usize, usize)> = legacy::list_experts;
    assert_eq!(canon_fn as usize, legacy_fn as usize);
}
