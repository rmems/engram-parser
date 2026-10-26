// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "safetensors")]

//! BF16 → F32 decode from Safetensors payloads.
//!
//! The synthetic test runs in CI. The real-checkpoint smoke test is
//! `#[ignore]` and path-gated until the shared fixtures (RM-2367) are pinned:
//!
//! ```bash
//! ENGRAM_BF16_SAFETENSORS=/path/to/model.safetensors \
//!   cargo test --features safetensors --test bf16_decode -- --ignored --nocapture
//! ```
//!
//! Optionally set `ENGRAM_BF16_TENSOR` to pick a tensor; otherwise the
//! smallest BF16 tensor is decoded.

use engram_parser::dequantize_bf16_bytes;
use engram_parser::safetensors::open_safetensors_checkpoint;
use std::fs;

#[test]
fn decodes_bf16_safetensors_payload() {
    let dir = std::env::temp_dir().join(format!("engram-bf16-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let header = r#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
    let payload: Vec<u8> = [0x3F80u16, 0xBF80, 0x0000, 0x7F80]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    fs::write(&path, file).unwrap();

    let ck = open_safetensors_checkpoint(&path).unwrap();
    assert_eq!(ck.tensor("w").unwrap().dtype, "BF16");
    let out = dequantize_bf16_bytes(&ck.tensor_bytes("w").unwrap()).unwrap();
    assert_eq!(out, vec![1.0, -1.0, 0.0, f32::INFINITY]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs a real BF16 Safetensors checkpoint (ENGRAM_BF16_SAFETENSORS); becomes a required CI test once the RM-2367 fixture is pinned"]
fn real_checkpoint_bf16_tensor_decodes() {
    // Ignored by default; when explicitly requested it must not pass vacuously.
    let path = std::env::var("ENGRAM_BF16_SAFETENSORS")
        .expect("set ENGRAM_BF16_SAFETENSORS to a BF16 .safetensors file to run this test");
    let ck = open_safetensors_checkpoint(&path).expect("open checkpoint");
    let record = match std::env::var("ENGRAM_BF16_TENSOR") {
        Ok(name) => ck.tensor(&name).expect("named tensor").clone(),
        Err(_) => ck
            .manifest()
            .tensors
            .iter()
            .filter(|t| t.dtype == "BF16")
            .min_by_key(|t| t.byte_size)
            .expect("checkpoint has no BF16 tensor")
            .clone(),
    };
    assert_eq!(record.dtype, "BF16");
    let out = dequantize_bf16_bytes(&ck.tensor_bytes(&record.name).unwrap()).unwrap();
    let n: usize = record.shape.iter().product();
    assert_eq!(out.len(), n);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite weight");
    assert!(out.iter().any(|v| *v != 0.0), "all-zero tensor");
    eprintln!("decoded {} ({n} elements)", record.name);
}
