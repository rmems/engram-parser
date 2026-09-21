// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parser resource budgets, host-size conversions, and mmap error parity.

mod common;
use common::*;

use engram_parser::{
    HostSizeField, ParseLimitKind, ParseLimits, ParserError, parse_bytes, parse_bytes_with_limits,
};

fn expect_limit(err: ParserError, kind: ParseLimitKind, declared: u64, budget: u64) {
    let msg = err.to_string();
    match err {
        ParserError::LimitExceeded {
            limit,
            declared: got_declared,
            budget: got_budget,
            ..
        } => {
            assert_eq!(limit, kind, "limit kind");
            assert_eq!(got_declared, declared, "declared");
            assert_eq!(got_budget, budget, "budget");
            assert!(
                msg.contains(kind.as_str()),
                "Display must name the limit: {msg}"
            );
        }
        other => panic!("expected LimitExceeded({kind}), got {other}"),
    }
}

fn error_payload(err: &ParserError) -> String {
    match err {
        ParserError::Io { source, .. } => format!("io:{source}"),
        ParserError::UnsupportedFormat { reason, .. } => format!("unsupported:{reason}"),
        ParserError::MissingTensor { name, .. } => format!("missing:{name}"),
        ParserError::InvalidLayout { reason, .. } => format!("invalid:{reason}"),
        ParserError::ExpertOutOfRange {
            block,
            expert,
            available,
        } => format!("oor:{block}:{expert}:{available}"),
        ParserError::LimitExceeded {
            limit,
            declared,
            budget,
            ..
        } => format!("limit:{limit}:{declared}:{budget}"),
        ParserError::HostSizeOverflow { field, value, .. } => {
            format!("host:{field}:{value}")
        }
        ParserError::DuplicateTensorOwnership { name, shards, .. } => {
            format!("dup:{name}:{}", shards.join(","))
        }
        ParserError::MissingShard { shard, .. } => format!("missing-shard:{shard}"),
    }
}

fn parse_ok(bytes: Vec<u8>, limits: ParseLimits) {
    parse_bytes_with_limits(bytes, "mem://ok".into(), limits).expect("parse should succeed");
}

fn parse_err(bytes: Vec<u8>, limits: ParseLimits) -> ParserError {
    parse_bytes_with_limits(bytes, "mem://err".into(), limits).expect_err("parse should fail")
}

fn kv_only(kv: &[(&str, KvValue)]) -> Vec<u8> {
    build_gguf(kv, &[])
}

fn one_tensor(name: &'static str, dims: Vec<usize>) -> Vec<u8> {
    let n: usize = dims.iter().product();
    let tensors = [TensorSpec {
        name,
        dims,
        ggml_type: GGML_F32,
        payload: vec![0u8; n * 4],
    }];
    build_gguf(&[("general.architecture", KvValue::Str("test"))], &tensors)
}

#[test]
fn default_limits_accept_tiny_fixture() {
    let bytes = one_tensor("token_embd.weight", vec![4, 2]);
    parse_bytes(bytes, "mem://tiny".into()).expect("default limits must accept tiny fixtures");
}

#[test]
fn kv_count_boundary_and_plus_one() {
    let two = kv_only(&[
        ("general.architecture", KvValue::Str("a")),
        ("k2", KvValue::U32(1)),
    ]);
    let three = kv_only(&[
        ("general.architecture", KvValue::Str("a")),
        ("k2", KvValue::U32(1)),
        ("k3", KvValue::U32(2)),
    ]);
    let limits = ParseLimits {
        max_kv_count: 2,
        ..ParseLimits::default()
    };
    parse_ok(two, limits);
    expect_limit(parse_err(three, limits), ParseLimitKind::KvCount, 3, 2);
}

#[test]
fn tensor_count_boundary_and_plus_one() {
    let one = one_tensor("a", vec![2, 2]);
    let two = {
        let payload = vec![0u8; 16];
        build_gguf(
            &[("general.architecture", KvValue::Str("test"))],
            &[
                TensorSpec {
                    name: "a",
                    dims: vec![2, 2],
                    ggml_type: GGML_F32,
                    payload: payload.clone(),
                },
                TensorSpec {
                    name: "b",
                    dims: vec![2, 2],
                    ggml_type: GGML_F32,
                    payload,
                },
            ],
        )
    };
    let limits = ParseLimits {
        max_tensor_count: 1,
        ..ParseLimits::default()
    };
    parse_ok(one, limits);
    expect_limit(parse_err(two, limits), ParseLimitKind::TensorCount, 2, 1);
}

#[test]
fn string_bytes_boundary_and_plus_one() {
    let exact = kv_only(&[("abcd", KvValue::U32(1))]);
    let over = kv_only(&[("abcde", KvValue::U32(1))]);
    let limits = ParseLimits {
        max_string_bytes: 4,
        ..ParseLimits::default()
    };
    parse_ok(exact, limits);
    expect_limit(parse_err(over, limits), ParseLimitKind::StringBytes, 5, 4);
}

#[test]
fn tensor_rank_boundary_and_plus_one() {
    let rank2 = one_tensor("t", vec![2, 2]);
    let rank3 = one_tensor("t", vec![2, 2, 1]);
    let limits = ParseLimits {
        max_tensor_rank: 2,
        ..ParseLimits::default()
    };
    parse_ok(rank2, limits);
    expect_limit(parse_err(rank3, limits), ParseLimitKind::TensorRank, 3, 2);
}

#[test]
fn array_work_items_boundary_and_plus_one() {
    fn file_with_uint8_array(n: u64) -> Vec<u8> {
        let mut out = gguf_header(0, 1);
        push_array_uint8_header(&mut out, "arr", n);
        for i in 0..n {
            out.push(i as u8);
        }
        out
    }
    let limits = ParseLimits {
        max_array_work_items: 4,
        ..ParseLimits::default()
    };
    parse_ok(file_with_uint8_array(4), limits);
    expect_limit(
        parse_err(file_with_uint8_array(5), limits),
        ParseLimitKind::ArrayWorkItems,
        5,
        4,
    );
}

#[test]
fn metadata_bytes_boundary_and_plus_one() {
    let mut bytes = gguf_header(0, 1);
    push_kv_u32(&mut bytes, "a", 1);
    let used = (bytes.len() - 24) as u64;
    let ok_limits = ParseLimits {
        max_metadata_bytes: used,
        ..ParseLimits::default()
    };
    parse_ok(bytes.clone(), ok_limits);
    let tight = ParseLimits {
        max_metadata_bytes: used - 1,
        ..ParseLimits::default()
    };
    expect_limit(
        parse_err(bytes, tight),
        ParseLimitKind::MetadataBytes,
        used,
        used - 1,
    );
}

#[test]
fn u64_max_counts_and_strings_name_the_limit() {
    let tensor_max = gguf_header(u64::MAX, 0);
    expect_limit(
        parse_err(tensor_max, ParseLimits::default()),
        ParseLimitKind::TensorCount,
        u64::MAX,
        ParseLimits::DEFAULT.max_tensor_count,
    );

    let kv_max = gguf_header(0, u64::MAX);
    expect_limit(
        parse_err(kv_max, ParseLimits::default()),
        ParseLimitKind::KvCount,
        u64::MAX,
        ParseLimits::DEFAULT.max_kv_count,
    );

    let mut huge_string = gguf_header(0, 1);
    push_u64(&mut huge_string, u64::MAX);
    expect_limit(
        parse_err(huge_string, ParseLimits::default()),
        ParseLimitKind::StringBytes,
        u64::MAX,
        ParseLimits::DEFAULT.max_string_bytes,
    );

    let mut huge_array = gguf_header(0, 1);
    push_array_uint8_header(&mut huge_array, "arr", u64::MAX);
    expect_limit(
        parse_err(huge_array, ParseLimits::default()),
        ParseLimitKind::ArrayWorkItems,
        u64::MAX,
        ParseLimits::DEFAULT.max_array_work_items,
    );
}

fn two_f32_tensors_at_relative_offset(offset: u64) -> Vec<u8> {
    let mut out = gguf_header(2, 0);
    for name in ["left", "right"] {
        push_string(&mut out, name);
        push_u32(&mut out, 1);
        push_u64(&mut out, 1);
        push_u32(&mut out, GGML_F32);
        push_u64(&mut out, offset);
    }
    while !out.len().is_multiple_of(ALIGNMENT as usize) {
        out.push(0);
    }
    out.extend_from_slice(&1.0f32.to_le_bytes());
    out
}

#[test]
fn relative_zero_cannot_expose_header_bytes() {
    let bytes = one_tensor("w", vec![2]);
    let layout = parse_bytes(bytes, "mem://rel0".into()).expect("parse");
    let tensor = layout.tensor("w").expect("tensor");
    let payload = layout.tensor_bytes(tensor).expect("payload");
    assert_all(&[
        (tensor.relative_offset == 0, "relative-zero"),
        (
            tensor.absolute_offset == layout.tensor_data_offset,
            "absolute-at-data-section",
        ),
        (
            tensor.absolute_offset >= layout.tensor_data_offset,
            "not-before-data-section",
        ),
        (layout.bytes[..4] == GGUF_MAGIC, "header-magic-present"),
        (!payload.starts_with(&GGUF_MAGIC), "payload-not-header"),
        (
            layout.bytes[..layout.tensor_data_offset].starts_with(&GGUF_MAGIC),
            "metadata-keeps-header",
        ),
    ]);
}

#[test]
fn overlapping_relative_offsets_stay_in_tensor_data() {
    let bytes = two_f32_tensors_at_relative_offset(0);
    let layout = parse_bytes(bytes, "mem://overlap".into()).expect("parse overlap");
    let left = layout.tensor("left").expect("left");
    let right = layout.tensor("right").expect("right");
    let left_bytes = layout.tensor_bytes(left).expect("left payload");
    let right_bytes = layout.tensor_bytes(right).expect("right payload");
    assert_all(&[
        (left.relative_offset == 0, "left-rel"),
        (right.relative_offset == 0, "right-rel"),
        (
            left.absolute_offset == layout.tensor_data_offset,
            "left-abs",
        ),
        (
            right.absolute_offset == layout.tensor_data_offset,
            "right-abs",
        ),
        (!left_bytes.starts_with(&GGUF_MAGIC), "left-not-header"),
        (!right_bytes.starts_with(&GGUF_MAGIC), "right-not-header"),
        (left_bytes == right_bytes, "shared-data-section-bytes"),
    ]);
}

#[test]
fn u64_max_relative_offset_cannot_wrap() {
    let mut out = gguf_header(1, 0);
    push_string(&mut out, "t");
    push_u32(&mut out, 1);
    push_u64(&mut out, 1);
    push_u32(&mut out, GGML_F32);
    push_u64(&mut out, u64::MAX);
    let err = parse_err(out, ParseLimits::default());
    match err {
        ParserError::InvalidLayout { reason, .. } => {
            assert!(
                reason.contains("absolute offset overflow") || reason.contains("overflow"),
                "got: {reason}"
            );
        }
        ParserError::HostSizeOverflow { field, value, .. } => {
            assert_eq!(field, HostSizeField::RelativeOffset);
            assert_eq!(value, u64::MAX);
        }
        other => panic!("expected wrap/host overflow, got {other}"),
    }
}

#[test]
fn deep_nested_arrays_are_bounded_by_total_work() {
    const DEPTH: usize = 32;
    let mut ok = gguf_header(0, 1);
    push_string(&mut ok, "nested");
    push_u32(&mut ok, VT_ARRAY);
    push_nested_array_payload(&mut ok, DEPTH);

    parse_ok(ok.clone(), ParseLimits::default());

    // 32-level nest of length-1 arrays charges 31 work items (see cursor tests).
    let exact = ParseLimits {
        max_array_work_items: 31,
        ..ParseLimits::default()
    };
    parse_ok(ok.clone(), exact);

    let too_small = ParseLimits {
        max_array_work_items: 30,
        ..ParseLimits::default()
    };
    let err = parse_err(ok, too_small);
    match err {
        ParserError::LimitExceeded { limit, .. } => {
            assert_eq!(limit, ParseLimitKind::ArrayWorkItems);
        }
        other => panic!("expected array work limit, got {other}"),
    }
}

#[test]
fn pathological_rank_is_rejected_before_dim_iteration() {
    let mut out = gguf_header(1, 0);
    push_string(&mut out, "t");
    push_u32(&mut out, u32::MAX);
    expect_limit(
        parse_err(out, ParseLimits::default()),
        ParseLimitKind::TensorRank,
        u64::from(u32::MAX),
        ParseLimits::DEFAULT.max_tensor_rank,
    );
}

#[test]
fn element_count_multiplication_overflow() {
    let half = (usize::MAX / 2) as u64 + 1;
    let mut out = gguf_header(1, 0);
    push_string(&mut out, "t");
    push_u32(&mut out, 2);
    push_u64(&mut out, half);
    push_u64(&mut out, half);
    push_u32(&mut out, GGML_F32);
    push_u64(&mut out, 0);
    let err = parse_err(out, ParseLimits::default());
    match err {
        ParserError::InvalidLayout { reason, .. } => {
            assert!(reason.contains("element count overflow"), "got: {reason}");
        }
        ParserError::HostSizeOverflow { field, .. } => {
            assert_eq!(field, HostSizeField::TensorDim);
        }
        other => panic!("expected overflow, got {other}"),
    }
}

#[test]
fn truncated_buffers_at_each_header_directory_stage() {
    let mut kv_and_tensor = gguf_header(1, 1);
    push_kv_string(&mut kv_and_tensor, "general.architecture", "t");
    push_string(&mut kv_and_tensor, "weight");
    push_u32(&mut kv_and_tensor, 1);
    push_u64(&mut kv_and_tensor, 1);
    push_u32(&mut kv_and_tensor, GGML_F32);
    push_u64(&mut kv_and_tensor, 0);

    let stages: [(&[u8], &str); 8] = [
        (&[][..], "empty"),
        (b"GGU", "partial-magic"),
        (b"GGUF", "magic-only"),
        (&kv_and_tensor[..8], "magic-version"),
        (&kv_and_tensor[..16], "tensor-count-only"),
        (&kv_and_tensor[..24], "counts-no-kv"),
        (&kv_and_tensor[..28], "partial-kv-key-len"),
        (
            &kv_and_tensor[..kv_and_tensor.len() - 4],
            "partial-tensor-offset",
        ),
    ];
    for (bytes, label) in stages {
        let err = parse_bytes(bytes.to_vec(), format!("mem://trunc-{label}")).expect_err(label);
        let msg = err.to_string();
        assert!(
            msg.contains("EOF")
                || msg.contains("overflow")
                || msg.contains("unsupported")
                || msg.contains("invalid")
                || msg.contains("parse limit")
                || msg.contains("host-size"),
            "{label}: {msg}"
        );
    }
}

#[test]
fn trusted_override_does_not_change_default_safety() {
    let three = kv_only(&[
        ("a", KvValue::U32(1)),
        ("b", KvValue::U32(2)),
        ("c", KvValue::U32(3)),
    ]);
    let tight = ParseLimits {
        max_kv_count: 2,
        ..ParseLimits::default()
    };
    expect_limit(
        parse_err(three.clone(), tight),
        ParseLimitKind::KvCount,
        3,
        2,
    );
    parse_ok(three, ParseLimits::trusted());
    assert_eq!(ParseLimits::default().max_kv_count, 1_000_000);
}

#[cfg(feature = "mmap")]
fn write_temp_gguf(bytes: &[u8]) -> (std::path::PathBuf, TempGguf) {
    use std::fs;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "engram-parser-limits-{}-{nanos}.gguf",
        process::id()
    ));
    fs::write(&path, bytes).expect("write temp gguf");
    (path.clone(), TempGguf(path))
}

#[cfg(feature = "mmap")]
struct TempGguf(std::path::PathBuf);
#[cfg(feature = "mmap")]
impl Drop for TempGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(feature = "mmap")]
#[test]
fn default_and_mmap_error_parity_on_limit_failures() {
    use engram_parser::load_gguf_mmap_with_limits;

    let cases: Vec<(Vec<u8>, ParseLimits)> = vec![
        (gguf_header(u64::MAX, 0), ParseLimits::default()),
        (gguf_header(0, u64::MAX), ParseLimits::default()),
        (
            {
                let mut b = gguf_header(0, 1);
                push_u64(&mut b, u64::MAX);
                b
            },
            ParseLimits::default(),
        ),
        (
            kv_only(&[
                ("a", KvValue::U32(1)),
                ("b", KvValue::U32(2)),
                ("c", KvValue::U32(3)),
            ]),
            ParseLimits {
                max_kv_count: 2,
                ..ParseLimits::default()
            },
        ),
        (
            one_tensor("t", vec![2, 2, 1]),
            ParseLimits {
                max_tensor_rank: 2,
                ..ParseLimits::default()
            },
        ),
    ];

    for (i, (bytes, limits)) in cases.into_iter().enumerate() {
        let owned = parse_bytes_with_limits(bytes.clone(), format!("mem://mmap-{i}"), limits)
            .expect_err("owned");
        let (path, _guard) = write_temp_gguf(&bytes);
        let mapped = load_gguf_mmap_with_limits(&path, limits).expect_err("mmap");
        assert_eq!(
            error_payload(&owned),
            error_payload(&mapped),
            "default vs mmap error payload mismatch\n owned={owned}\n mmap={mapped}"
        );
    }
}

#[cfg(feature = "mmap")]
#[test]
fn default_mmap_success_parity_on_tiny_file() {
    use engram_parser::load_gguf_mmap;

    let bytes = one_tensor("token_embd.weight", vec![4, 2]);
    let (path, _guard) = write_temp_gguf(&bytes);
    let owned = parse_bytes(bytes, "mem://mmap-ok".into()).expect("owned");
    let mapped = load_gguf_mmap(&path).expect("mmap");
    assert!(mapped.directory_matches(&owned));
}
