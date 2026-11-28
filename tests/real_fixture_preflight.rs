// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared pin manifest and opt-in preflight for GGUF/Safetensors baseline pairs.
#![cfg(all(feature = "mmap", feature = "safetensors"))]

#[path = "common/real_fixtures.rs"]
mod real_fixtures;

use engram_parser::{Checkpoint, DType, TensorDType, open_checkpoint_mmap};
use real_fixtures::{
    ALL, Config, Mode, OLMOE_GGUF, OLMOE_SAFETENSORS, Outcome, QWEN3_GGUF, QWEN3_SAFETENSORS,
    QWEN25_GGUF, QWEN25_SAFETENSORS, RealFixture, ensure,
};

const GGUF_TENSOR: &str = "blk.0.attn_v.weight";
const SAFETENSORS_TENSOR: &str = "model.layers.0.self_attn.v_proj.weight";

fn check_pair(gguf: &'static RealFixture, safetensors: &'static RealFixture) {
    let mut cfg = Config::from_env();
    if cfg.mode == Mode::Auto {
        cfg.mode = Mode::Offline;
    }
    let load = |fixture: &'static RealFixture| match ensure(fixture, &cfg) {
        Outcome::Ready(dir) => Some(fixture.checkpoint_path(&dir)),
        Outcome::Skipped(reason) => {
            eprintln!("SKIP {}: {reason}", fixture.id);
            None
        }
    };
    let Some(gguf_path) = load(gguf) else {
        return;
    };
    let Some(st_path) = load(safetensors) else {
        return;
    };
    let g = open_checkpoint_mmap(&gguf_path)
        .unwrap_or_else(|err| panic!("open {}: {err}", gguf_path.display()));
    let s = open_checkpoint_mmap(&st_path)
        .unwrap_or_else(|err| panic!("open {}: {err}", st_path.display()));

    // Match the same semantic parameter explicitly. Shape and dtype
    // checks are structural, and do not imply that the values agree.
    let gt = g.tensor(GGUF_TENSOR).expect("GGUF V projection");
    let st = s
        .tensor(SAFETENSORS_TENSOR)
        .expect("Safetensors V projection");
    assert_eq!(
        gt.dtype,
        TensorDType::GgufPacked {
            ggml_type: DType::Q8_0.ggml_type()
        }
    );
    assert_eq!(st.dtype, TensorDType::BF16);
    assert_eq!(gt.shape.dims(), st.shape.dims());
    eprintln!(
        "PREFLIGHT {}: {GGUF_TENSOR} ↔ {SAFETENSORS_TENSOR}, shape {:?}, pins verified",
        safetensors.id,
        gt.shape.dims(),
    );
}

#[test]
#[ignore = "requires pinned local checkpoints; see docs/real-fixtures.md"]
fn qwen3_fixture_preflight() {
    check_pair(&QWEN3_GGUF, &QWEN3_SAFETENSORS);
}

#[test]
#[ignore = "requires pinned local checkpoints; see docs/real-fixtures.md"]
fn qwen25_fixture_preflight() {
    check_pair(&QWEN25_GGUF, &QWEN25_SAFETENSORS);
}

#[test]
#[ignore = "requires pinned local checkpoints; see docs/real-fixtures.md"]
fn olmoe_fixture_preflight() {
    check_pair(&OLMOE_GGUF, &OLMOE_SAFETENSORS);
}

#[test]
fn documented_manifest_matches_fetch_pins() {
    use std::fmt::Write;
    let mut actual =
        String::from("id\ttier\trepo\trevision\tfile\tdtype\tbytes\tsha256\tlicense\n");
    for fixture in ALL {
        let tier = if fixture.id.starts_with("qwen3.5-") {
            "stretch"
        } else {
            "baseline"
        };
        for file in fixture.files {
            let dtype = if file.file.ends_with(".gguf") {
                "Q8_0"
            } else if file.file.ends_with(".safetensors") {
                "BF16"
            } else {
                "metadata"
            };
            writeln!(
                actual,
                "{}\t{tier}\t{}\t{}\t{}\t{dtype}\t{}\t{}\t{}",
                fixture.id,
                fixture.repo,
                fixture.revision,
                file.file,
                file.size,
                file.sha256,
                fixture.license
            )
            .unwrap();
        }
    }
    let published = include_str!("../docs/smoke/cross-format-fixtures.tsv").replace("\r\n", "\n");
    assert_eq!(
        published, actual,
        "update the shared TSV when fixture pins change"
    );
}
