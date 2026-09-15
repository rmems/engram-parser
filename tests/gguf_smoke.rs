// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end smoke test: build a synthetic GGUF in memory, parse it,
//! and verify that expert extraction round-trips both the stacked and
//! per-expert storage conventions.

mod common;
use common::*;

use engram_parser::{DType, extract_expert, list_experts, parse_bytes};

#[test]
fn parses_magic_and_metadata() {
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
        ("olmoe.expert_count", KvValue::U32(4)),
        ("olmoe.block_count", KvValue::U32(1)),
    ];
    let tensors = [TensorSpec {
        name: "token_embd.weight",
        dims: vec![4, 2],
        ggml_type: GGML_F32,
        payload: f32_vec_to_le_bytes(&[0.0; 4 * 2]),
    }];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://test".into()).expect("parse");
    let t = &layout.tensors["token_embd.weight"];
    assert_all(&[
        (layout.metadata.architecture() == "olmoe", "architecture"),
        (
            layout.metadata.numeric("olmoe.expert_count") == Some(4),
            "expert_count",
        ),
        (layout.alignment == ALIGNMENT as usize, "alignment"),
        (t.dtype == DType::F32, "token dtype"),
        (t.dims == vec![4, 2], "token dims"),
    ]);
}

#[test]
fn extracts_stacked_expert_slices() {
    // 3 experts, each a 4x2 matrix of f32.
    // Expert e fills its slice with the value (e + 1).
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

    // GGML dims are innermost-first, so the expert axis is the last dim.
    let stacked_dims = vec![inner, outer, n_experts];

    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate_exps.weight",
            dims: stacked_dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&gate),
        },
        TensorSpec {
            name: "blk.0.ffn_up_exps.weight",
            dims: stacked_dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&up),
        },
        TensorSpec {
            name: "blk.0.ffn_down_exps.weight",
            dims: stacked_dims.clone(),
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&down),
        },
    ];
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
    ];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://stacked".into()).expect("parse");

    let pairs = list_experts(&layout);
    assert_eq!(pairs, vec![(0, 0), (0, 1), (0, 2)]);

    for e in 0..n_experts {
        let out = extract_expert(&layout, 0, e).expect("extract");
        let gate_first = out
            .gate
            .as_ref()
            .map(|g| f32::from_le_bytes(g.bytes[0..4].try_into().unwrap()));
        let up_first = out
            .up
            .as_ref()
            .map(|u| f32::from_le_bytes(u.bytes[0..4].try_into().unwrap()));
        let down_first = out
            .down
            .as_ref()
            .map(|d| f32::from_le_bytes(d.bytes[0..4].try_into().unwrap()));
        assert_all(&[
            (out.block == 0 && out.expert == e, "expert ids"),
            (out.is_complete(), "complete"),
            (
                out.gate.as_ref().is_some_and(|g| g.stacked_slice),
                "gate stacked",
            ),
            (
                out.gate.as_ref().is_some_and(|g| {
                    g.dims == vec![inner, outer] && g.bytes.len() == per_expert * 4
                }),
                "gate shape",
            ),
            (
                gate_first.is_some_and(|f| (f - ((e as f32) + 0.1)).abs() < 1e-6),
                "gate first",
            ),
            (
                up_first.is_some_and(|f| (f - ((e as f32) + 0.2)).abs() < 1e-6),
                "up first",
            ),
            (
                down_first.is_some_and(|f| (f - ((e as f32) + 0.3)).abs() < 1e-6),
                "down first",
            ),
        ]);
    }
}

#[test]
fn extracts_per_expert_tensors() {
    let inner = 2usize;
    let outer = 2usize;

    let gate0 = f32_vec_to_le_bytes(&[10.0; 4]);
    let gate1 = f32_vec_to_le_bytes(&[11.0; 4]);
    let up0 = f32_vec_to_le_bytes(&[20.0; 4]);
    let up1 = f32_vec_to_le_bytes(&[21.0; 4]);
    let down0 = f32_vec_to_le_bytes(&[30.0; 4]);
    let down1 = f32_vec_to_le_bytes(&[31.0; 4]);

    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate.0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: gate0,
        },
        TensorSpec {
            name: "blk.0.ffn_gate.1.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: gate1,
        },
        TensorSpec {
            name: "blk.0.ffn_up.0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: up0,
        },
        TensorSpec {
            name: "blk.0.ffn_up.1.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: up1,
        },
        TensorSpec {
            name: "blk.0.ffn_down.0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: down0,
        },
        TensorSpec {
            name: "blk.0.ffn_down.1.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: down1,
        },
    ];
    let kv = [("general.architecture", KvValue::Str("qwen3moe"))];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://peri".into()).expect("parse");

    let pairs = list_experts(&layout);
    assert_eq!(pairs, vec![(0, 0), (0, 1)]);

    let e0 = extract_expert(&layout, 0, 0).unwrap();
    let gate0_first = e0
        .gate
        .as_ref()
        .map(|g| f32::from_le_bytes(g.bytes[0..4].try_into().unwrap()));

    let e1 = extract_expert(&layout, 0, 1).unwrap();
    let up1_first = e1
        .up
        .as_ref()
        .map(|u| f32::from_le_bytes(u.bytes[0..4].try_into().unwrap()));

    assert_all(&[
        (
            e0.gate.as_ref().is_some_and(|g| !g.stacked_slice),
            "e0 gate not stacked",
        ),
        (
            e0.gate
                .as_ref()
                .is_some_and(|g| g.source_name == "blk.0.ffn_gate.0.weight"),
            "e0 gate source",
        ),
        (
            gate0_first.is_some_and(|f| (f - 10.0).abs() < 1e-6),
            "e0 gate first",
        ),
        (
            up1_first.is_some_and(|f| (f - 21.0).abs() < 1e-6),
            "e1 up first",
        ),
    ]);
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = vec![b'X', b'Y', b'Z', b'!'];
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let err = parse_bytes(bytes, "mem://bad-magic".into()).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("unsupported format"), "got: {msg}");
}

#[test]
fn expert_out_of_range() {
    let inner = 2usize;
    let outer = 2usize;
    let n_experts = 2usize;
    let data = f32_vec_to_le_bytes(&vec![0.0f32; inner * outer * n_experts]);
    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims: vec![inner, outer, n_experts],
        ggml_type: GGML_F32,
        payload: data,
    }];
    let kv = [("general.architecture", KvValue::Str("olmoe"))];
    let bytes = build_gguf(&kv, &tensors);
    let layout = parse_bytes(bytes, "mem://oor".into()).unwrap();

    let err = extract_expert(&layout, 0, 5).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("expert index out of range"), "got: {msg}");
}

fn metadata_layout() -> engram_parser::GgufLayout {
    let kv = [
        ("general.architecture", KvValue::Str("qwen2moe")),
        ("general.quantization_type", KvValue::Str("Q4_K_M")),
        ("qwen2moe.block_count", KvValue::U32(28)),
        ("qwen2moe.expert_count", KvValue::U32(64)),
        ("qwen2moe.expert_used_count", KvValue::U32(8)),
        ("qwen2moe.embedding_length", KvValue::U32(2048)),
        ("qwen2moe.attention.head_count", KvValue::U32(16)),
        ("qwen2moe.rope_freq_base", KvValue::F32(10_000.0)),
        ("general.name", KvValue::Str("Qwen2-MoE-A2.7B")),
    ];
    parse_bytes(build_gguf(&kv, &[]), "mem://metadata".into()).expect("parse")
}

#[test]
fn metadata_helpers_basic() {
    let layout = metadata_layout();
    let rope_freq = layout.metadata.float32("qwen2moe.rope_freq_base").unwrap();
    assert_all(&[
        (layout.metadata.architecture() == "qwen2moe", "architecture"),
        (layout.metadata.quantization() == "Q4_K_M", "quantization"),
        (
            layout.metadata.string("general.name") == Some("Qwen2-MoE-A2.7B"),
            "name",
        ),
        ((rope_freq - 10_000.0).abs() < 1e-6, "rope_freq_base"),
    ]);
}

#[test]
fn metadata_helpers_counts() {
    let layout = metadata_layout();
    assert_eq!(layout.metadata.block_count(), Some(28));
    assert_eq!(layout.metadata.expert_count(), Some(64));
    assert_eq!(layout.metadata.expert_used_count(), Some(8));
}

#[test]
fn metadata_helpers_geometry() {
    let layout = metadata_layout();
    assert_eq!(layout.metadata.embedding_length(), Some(2048));
    assert_eq!(layout.metadata.head_count(), Some(16));
}

#[test]
fn metadata_helpers_missing() {
    let layout = metadata_layout();
    assert_eq!(layout.metadata.string("nonexistent"), None);
    assert_eq!(layout.metadata.float32("nonexistent"), None);
}

#[test]
fn metadata_helpers_with_alternative_keys() {
    let kv = [
        ("general.architecture", KvValue::Str("mixtral")),
        ("mixtral.num_experts", KvValue::U32(8)),
        ("mixtral.num_experts_per_tok", KvValue::U32(2)),
    ];
    let layout = parse_bytes(build_gguf(&kv, &[]), "mem://alt-keys".into()).expect("parse");

    assert_eq!(layout.metadata.expert_count(), Some(8));
    assert_eq!(layout.metadata.expert_used_count(), Some(2));
}

#[test]
fn metadata_helpers_with_unknown_architecture() {
    let kv = [("some.block_count", KvValue::U32(10))];
    let layout = parse_bytes(build_gguf(&kv, &[]), "mem://no-arch".into()).expect("parse");

    assert_all(&[
        (layout.metadata.architecture() == "unknown", "architecture"),
        (layout.metadata.quantization() == "unknown", "quantization"),
        (layout.metadata.block_count().is_none(), "block_count"),
        (layout.metadata.expert_count().is_none(), "expert_count"),
    ]);
}

#[test]
fn ggml_type_label_function() {
    use engram_parser::ggml_type_label;

    let cases = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (8, "Q8_0"),
        (10, "Q2_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (21, "IQ3_S"),
        (30, "BF16"),
        (31, "Q4_0_4_4"),
    ];
    for (code, expected) in cases {
        assert_eq!(ggml_type_label(code), expected, "label for code {code}");
    }

    assert_eq!(ggml_type_label(999), "unknown");
}

#[test]
fn dtype_f32_roundtrip() {
    use engram_parser::DType;

    let dt = DType::from_ggml_type(0);
    assert_all(&[
        (dt == DType::F32, "variant"),
        (dt.ggml_type() == 0, "ggml_type"),
        (dt.byte_len_for_elements(100) == Some(400), "byte_len"),
        (dt.has_known_byte_layout(), "layout"),
    ]);
}

#[test]
fn dtype_q4k_layout() {
    use engram_parser::DType;

    let dt = DType::from_ggml_type(12);
    assert_all(&[
        (dt == DType::Q4_K, "variant"),
        (dt.ggml_type() == 12, "ggml_type"),
        (dt.byte_len_for_elements(256) == Some(144), "byte_len_256"),
        (dt.byte_len_for_elements(512) == Some(288), "byte_len_512"),
        (
            dt.byte_len_for_elements(100).is_none(),
            "byte_len_misaligned",
        ),
    ]);
}

#[test]
fn dtype_wire_31_is_other() {
    use engram_parser::DType;

    let dt = DType::from_ggml_type(31);
    assert_all(&[
        (dt == DType::Other(31), "variant"),
        (dt.ggml_type() == 31, "ggml_type"),
        (dt.byte_len_for_elements(256).is_none(), "byte_len"),
        (!dt.has_known_byte_layout(), "layout"),
        (dt.label() == "Q4_0_4_4", "label"),
    ]);
}

#[test]
fn dtype_unknown_is_other() {
    use engram_parser::DType;

    let dt = DType::from_ggml_type(999);
    assert_all(&[
        (dt == DType::Other(999), "variant"),
        (dt.ggml_type() == 999, "ggml_type"),
        (dt.byte_len_for_elements(100).is_none(), "byte_len"),
        (!dt.has_known_byte_layout(), "layout"),
    ]);
}

#[test]
fn dtype_label_method() {
    use engram_parser::DType;

    let cases = [
        (DType::F32, "F32"),
        (DType::F16, "F16"),
        (DType::Q4_0, "Q4_0"),
        (DType::Q8_0, "Q8_0"),
        (DType::Q4_K, "Q4_K"),
        (DType::IQ3_S, "IQ3_S"),
        (DType::Other(31), "Q4_0_4_4"),
        (DType::BF16, "BF16"),
        (DType::Other(999), "unknown"),
    ];
    for (dt, expected) in cases {
        assert_eq!(dt.label(), expected, "label for {dt:?}");
    }
}

#[test]
fn tensor_with_wire_type_31_fails_closed() {
    // Wire type 31 (Q4_0_4_4) has no modeled byte layout — parse must fail closed.
    let inner = 256;
    let outer = 2;
    let n_experts = 2;
    let dummy_bytes_per_expert = 512;
    let mut payload = Vec::new();
    for i in 0..n_experts {
        for _ in 0..dummy_bytes_per_expert {
            payload.push(i as u8);
        }
    }

    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims: vec![inner, outer, n_experts],
        ggml_type: 31, // Q4_0_4_4 — not IQ3_M
        payload,
    }];

    let kv = [("general.architecture", KvValue::Str("testmoe"))];
    let bytes = build_gguf(&kv, &tensors);
    let err = parse_bytes(bytes, "mem://t31".into())
        .expect_err("type 31 must not parse with known layout");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown byte-length")
            || msg.contains("InvalidLayout")
            || msg.contains("ggml_type=31"),
        "unexpected error: {msg}"
    );
}

#[test]
fn rejects_unsupported_gguf_version() {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, 2); // version 2 — not supported
    push_u64(&mut out, 0);
    push_u64(&mut out, 0);
    let err = parse_bytes(out, "mem://v2".into()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported GGUF version") || msg.contains("unsupported format"),
        "got: {msg}"
    );
}

#[test]
fn rejects_truncated_file() {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    // Claim one KV but provide no body → EOF while parsing.
    push_u64(&mut out, 0); // tensor_count
    push_u64(&mut out, 1); // kv_count
    let err = parse_bytes(out, "mem://trunc".into()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("EOF")
            || msg.contains("overflow")
            || msg.contains("unsupported")
            || msg.contains("InvalidLayout")
            || msg.contains("invalid"),
        "got: {msg}"
    );
}

#[test]
fn quantization_falls_back_to_file_type() {
    let kv = [
        ("general.architecture", KvValue::Str("olmoe")),
        ("general.file_type", KvValue::U32(15)),
    ];
    let bytes = build_gguf(&kv, &[]);
    let layout = parse_bytes(bytes, "mem://file-type".into()).expect("parse");
    assert_eq!(layout.metadata.quantization(), "GGUF(15)");

    let kv_f32 = [
        ("general.architecture", KvValue::Str("olmoe")),
        ("general.file_type", KvValue::U32(0)),
    ];
    let layout_f32 = parse_bytes(build_gguf(&kv_f32, &[]), "mem://ft0".into()).unwrap();
    assert_eq!(layout_f32.metadata.quantization(), "F32");

    // Explicit quantization_type wins over file_type.
    let kv_pref = [
        ("general.quantization_type", KvValue::Str("Q4_K_M")),
        ("general.file_type", KvValue::U32(15)),
    ];
    let layout_pref = parse_bytes(build_gguf(&kv_pref, &[]), "mem://pref".into()).unwrap();
    assert_eq!(layout_pref.metadata.quantization(), "Q4_K_M");
}

#[test]
fn quantization_from_default_metadata_reads_file_type() {
    use engram_parser::GgufMetadata;
    let mut meta = GgufMetadata::default();
    meta.numerics.insert("general.file_type".into(), 0);
    assert_eq!(meta.quantization(), "F32");
    meta.numerics.insert("general.file_type".into(), 15);
    assert_eq!(meta.quantization(), "GGUF(15)");
    meta.strings
        .insert("general.quantization_type".into(), "Q8_0".into());
    assert_eq!(meta.quantization(), "Q8_0");
}

#[test]
fn rejects_non_row_aligned_blocked_quant() {
    // Total elems = 32 (block-aligned) but dims[0]=16 is not divisible by 32.
    let payload = vec![0u8; 18]; // would be one Q4_0 block if shape were valid
    let tensors = [TensorSpec {
        name: "bad.q4_0",
        dims: vec![16, 2],
        ggml_type: 2, // Q4_0
        payload,
    }];
    let kv = [("general.architecture", KvValue::Str("test"))];
    let err = parse_bytes(build_gguf(&kv, &tensors), "mem://bad-row".into()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("innermost dim") || msg.contains("block size"),
        "got: {msg}"
    );
}

#[test]
fn rejects_negative_alignment_metadata() {
    // Hand-built: INT32 general.alignment = -1 must not wrap to huge usize.
    const VT_INT32: u32 = 5;
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, 0); // tensors
    push_u64(&mut out, 1); // one KV
    push_string(&mut out, "general.alignment");
    push_u32(&mut out, VT_INT32);
    out.extend_from_slice(&(-1i32).to_le_bytes());
    let err = parse_bytes(out, "mem://neg-align".into()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("negative") || msg.contains("InvalidLayout"),
        "got: {msg}"
    );
}

#[test]
fn accepts_negative_vendor_signed_metadata() {
    // Non-layout signed KVs may be negative; must not fail the whole file.
    const VT_INT32: u32 = 5;
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, 0);
    push_u64(&mut out, 1);
    push_string(&mut out, "vendor.custom_signed");
    push_u32(&mut out, VT_INT32);
    out.extend_from_slice(&(-7i32).to_le_bytes());
    let layout = parse_bytes(out, "mem://neg-vendor".into()).expect("parse");
    // Bit-preserving cast of -7 as i32 → u64.
    let expected = (-7i32) as i64 as u64;
    assert_eq!(
        layout.metadata.numerics.get("vendor.custom_signed"),
        Some(&expected)
    );
}

#[test]
fn parses_iq3_s_tensor_layout() {
    // IQ3_S: 256 elements per block, 110 bytes/block (GGUF wire layout).
    let n = 256usize;
    let payload = vec![0xABu8; 110];
    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate.weight",
        dims: vec![n],
        ggml_type: GGML_IQ3_S,
        payload,
    }];
    let kv = [("general.architecture", KvValue::Str("testmoe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://iq3s".into()).expect("parse");
    let t = layout.tensor("blk.0.ffn_gate.weight").unwrap();
    assert_eq!(t.dtype, DType::IQ3_S);
    assert_eq!(t.byte_len, 110);
    assert_eq!(layout.tensor_bytes(t).unwrap().len(), 110);
}

#[test]
fn extracts_stacked_q8_0_expert_slices() {
    // Q8_0: block size 32, 34 bytes/block. Per-expert: 32 elems → 34 bytes.
    let inner = 32usize;
    let outer = 1usize;
    let n_experts = 3usize;
    let per_expert_bytes = 34usize;
    let mut payload = Vec::with_capacity(n_experts * per_expert_bytes);
    for e in 0..n_experts {
        payload.extend(std::iter::repeat_n(e as u8, per_expert_bytes));
    }
    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate_exps.weight",
            dims: vec![inner, outer, n_experts],
            ggml_type: GGML_Q8_0,
            payload: payload.clone(),
        },
        TensorSpec {
            name: "blk.0.ffn_up_exps.weight",
            dims: vec![inner, outer, n_experts],
            ggml_type: GGML_Q8_0,
            payload: payload.clone(),
        },
        TensorSpec {
            name: "blk.0.ffn_down_exps.weight",
            dims: vec![inner, outer, n_experts],
            ggml_type: GGML_Q8_0,
            payload,
        },
    ];
    let kv = [("general.architecture", KvValue::Str("olmoe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://q8-stacked".into()).expect("parse");
    assert_eq!(list_experts(&layout), vec![(0, 0), (0, 1), (0, 2)]);

    for e in 0..n_experts {
        let out = extract_expert(&layout, 0, e).expect("extract");
        let gate = out.gate.as_ref();
        assert_all(&[
            (gate.is_some_and(|g| g.stacked_slice), "gate stacked"),
            (
                gate.is_some_and(|g| g.bytes.len() == per_expert_bytes),
                "gate bytes len",
            ),
            (
                gate.is_some_and(|g| g.bytes.iter().all(|&b| b == e as u8)),
                "gate bytes",
            ),
            (gate.is_some_and(|g| g.dtype == DType::Q8_0), "gate dtype"),
            (out.is_complete(), "complete"),
        ]);
    }
}

#[test]
fn extracts_stacked_q4_k_expert_slices() {
    // Q4_K: 256 elems/block, 144 bytes/block. dims [256, 1, 2] experts.
    let inner = 256usize;
    let outer = 1usize;
    let n_experts = 2usize;
    let per_expert_bytes = 144usize;
    let mut payload = Vec::with_capacity(n_experts * per_expert_bytes);
    for e in 0..n_experts {
        payload.extend(std::iter::repeat_n((0x10 + e) as u8, per_expert_bytes));
    }
    let tensors = [TensorSpec {
        name: "blk.0.ffn_gate_exps.weight",
        dims: vec![inner, outer, n_experts],
        ggml_type: GGML_Q4_K,
        payload,
    }];
    let kv = [("general.architecture", KvValue::Str("olmoe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://q4k-stacked".into()).expect("parse");
    let e0 = extract_expert(&layout, 0, 0).unwrap();
    let gate0 = e0.gate.as_ref();
    assert_all(&[
        (gate0.is_some_and(|g| g.bytes.len() == 144), "e0 bytes len"),
        (
            gate0.is_some_and(|g| g.bytes.iter().all(|&b| b == 0x10)),
            "e0 bytes",
        ),
        (gate0.is_some_and(|g| g.dtype == DType::Q4_K), "e0 dtype"),
    ]);

    let e1 = extract_expert(&layout, 0, 1).unwrap();
    let gate1 = e1.gate.as_ref();
    assert!(
        gate1.is_some_and(|g| g.bytes.iter().all(|&b| b == 0x11)),
        "e1 bytes"
    );
}

#[test]
fn extracts_underscore_per_expert_tensors() {
    // Alternate naming: ffn_gate_0.weight instead of ffn_gate.0.weight
    let inner = 2usize;
    let outer = 2usize;
    let tensors = [
        TensorSpec {
            name: "blk.0.ffn_gate_0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&[1.0; 4]),
        },
        TensorSpec {
            name: "blk.0.ffn_up_0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&[2.0; 4]),
        },
        TensorSpec {
            name: "blk.0.ffn_down_0.weight",
            dims: vec![inner, outer],
            ggml_type: GGML_F32,
            payload: f32_vec_to_le_bytes(&[3.0; 4]),
        },
    ];
    let kv = [("general.architecture", KvValue::Str("qwen3moe"))];
    let layout = parse_bytes(build_gguf(&kv, &tensors), "mem://uscore".into()).expect("parse");
    let pairs = list_experts(&layout);
    assert!(
        pairs.contains(&(0, 0)),
        "expected (0,0) in list_experts, got {pairs:?}"
    );
    let e0 = extract_expert(&layout, 0, 0).expect("extract underscore expert");
    assert!(e0.is_complete());
    assert_eq!(
        e0.gate.as_ref().unwrap().source_name,
        "blk.0.ffn_gate_0.weight"
    );
}
