// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared pin manifest and opt-in preflight for GGUF/Safetensors baseline pairs.
#![cfg(all(feature = "mmap", feature = "safetensors"))]

#[path = "common/real_fixtures.rs"]
mod real_fixtures;

use std::fs;
use std::path::PathBuf;

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
    // Resolve both pins so an absent fixture cannot hide corruption in its peer.
    let (Some(gguf_path), Some(st_path)) = (load(gguf), load(safetensors)) else {
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

fn fixture_manifest(fixtures: &[RealFixture]) -> String {
    use std::fmt::Write;
    let mut actual =
        String::from("id\ttier\trepo\trevision\tfile\tdtype\tbytes\tsha256\tlicense\n");
    for fixture in fixtures {
        let tier = fixture.tier.label();
        for file in fixture.files {
            let dtype = file.dtype_label();
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
    actual
}

#[test]
fn documented_manifest_matches_fetch_pins() {
    let published = include_str!("../docs/smoke/cross-format-fixtures.tsv").replace("\r\n", "\n");
    assert_eq!(
        published,
        fixture_manifest(ALL),
        "update the shared TSV when fixture pins change"
    );
}

#[test]
fn manifest_uses_pinned_dtype_and_tier() {
    use real_fixtures::{FixtureFile, FixtureFileKind, FixtureTier};

    const FILES: &[FixtureFile] = &[
        FixtureFile {
            kind: FixtureFileKind::Weights("Q4_K_M"),
            ..QWEN3_GGUF.files[0]
        },
        FixtureFile {
            kind: FixtureFileKind::Weights("F32"),
            ..QWEN3_SAFETENSORS.files[0]
        },
    ];
    let fixture = RealFixture {
        id: "synthetic-no-prefix",
        tier: FixtureTier::Stretch,
        files: FILES,
        ..QWEN3_GGUF
    };
    let manifest = fixture_manifest(&[fixture]);
    let rows: Vec<_> = manifest
        .lines()
        .skip(1)
        .map(|row| {
            let cols: Vec<_> = row.split('\t').collect();
            (cols[1], cols[5])
        })
        .collect();
    assert_eq!(rows, [("stretch", "Q4_K_M"), ("stretch", "F32")]);
}

#[test]
fn sharded_preflight_opens_the_pinned_index() {
    use real_fixtures::{FixtureFile, FixtureFileKind, sha256_hex};

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let root = Scratch(
        std::env::temp_dir().join(format!("engram-preflight-index-{}", std::process::id())),
    );
    let config = Config {
        mode: Mode::Offline,
        fixture_root: None,
        cache_dir: root.0.clone(),
        base_url: "file:///unused".to_owned(),
    };
    let dir = OLMOE_SAFETENSORS.fixture_dir(&config.cache_dir);
    fs::create_dir_all(&dir).unwrap();

    let entries: &[(&str, FixtureFileKind, &[u8])] = &[
        (
            "model-00001-of-00002.safetensors",
            FixtureFileKind::Weights("F32"),
            include_bytes!("fixtures/safetensors/sharded/model-00001-of-00002.safetensors"),
        ),
        (
            "model-00002-of-00002.safetensors",
            FixtureFileKind::Weights("F32"),
            include_bytes!("fixtures/safetensors/sharded/model-00002-of-00002.safetensors"),
        ),
        (
            "model.safetensors.index.json",
            FixtureFileKind::ShardIndex,
            include_bytes!("fixtures/safetensors/sharded/model.safetensors.index.json"),
        ),
    ];
    let files: Vec<_> = entries
        .iter()
        .map(|&(file, kind, bytes)| {
            fs::write(dir.join(file), bytes).unwrap();
            FixtureFile {
                file,
                kind,
                sha256: Box::leak(sha256_hex(bytes).into_boxed_str()),
                size: bytes.len() as u64,
            }
        })
        .collect();
    fs::write(dir.join("unrelated.safetensors.index.json"), "{}").unwrap();

    let fixture = Box::leak(Box::new(RealFixture {
        files: Box::leak(files.into_boxed_slice()),
        ..OLMOE_SAFETENSORS
    }));
    let Outcome::Ready(verified_dir) = ensure(fixture, &config) else {
        panic!("expected the tiny fixture to verify");
    };
    let path = fixture.checkpoint_path(&verified_dir);
    assert_eq!(path, dir.join("model.safetensors.index.json"));
    assert_eq!(OLMOE_SAFETENSORS.checkpoint_path(&verified_dir), path);
    assert!(open_checkpoint_mmap(&verified_dir).is_err());
    let checkpoint = open_checkpoint_mmap(&path).expect("open the pinned shard index");
    assert_eq!(checkpoint.tensors().len(), 2);
}
