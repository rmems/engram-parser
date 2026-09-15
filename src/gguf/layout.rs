// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parsed GGUF file layout: metadata KV store + tensor directory.
//!
//! This module is file-agnostic: it operates on an already-loaded
//! `&[u8]` slice (typically the full contents of a `.gguf` file). It
//! does **not** perform any neural-network math — only header parsing,
//! bounds-checked tensor-directory extraction, and metadata coercion.

use std::collections::HashMap;

use super::cursor::{
    GGUF_MAGIC, GGUF_VALUE_TYPE_STRING, GGUF_VERSION, GgufCursor, invalid_layout,
    is_signed_layout_type, unsupported,
};
use super::tensor::{DType, Tensor};
use crate::error::{ParserError, Result};

/// Upper bounds to prevent OOM from malformed inputs.
const MAX_TENSOR_COUNT: u64 = 1_000_000;
const MAX_KV_COUNT: u64 = 1_000_000;
const MAX_TENSOR_DIMS: usize = 8;

/// Parsed GGUF metadata key-value store.
///
/// GGUF stores arbitrary scalar KV pairs. We keep strings and numeric
/// values in typed maps; signed integers are additionally kept in
/// `signed_numerics` so callers can distinguish bit-preserved negatives
/// from large unsigned `UINT64` values. Array values are skipped (their
/// byte contents remain available via the original file buffer if needed).
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    /// String-typed KV pairs (e.g. `general.architecture = "olmoe"`).
    pub strings: HashMap<String, String>,
    /// Numeric-typed KV pairs coerced to `u64`
    /// (e.g. `olmoe.expert_count = 64`).
    pub numerics: HashMap<String, u64>,
    /// Signed integer KV pairs coerced to `i64`, keyed separately so
    /// `numeric()` can reject negatives without rejecting large unsigned
    /// `UINT64` metadata.
    pub(crate) signed_numerics: HashMap<String, i64>,
    /// `f32`-typed KV pairs.
    pub floats_32: HashMap<String, f32>,
    /// `f64`-typed KV pairs.
    pub floats_64: HashMap<String, f64>,
}

impl GgufMetadata {
    /// Convenience: architecture string (`general.architecture`) or `"unknown"`.
    pub fn architecture(&self) -> &str {
        self.strings
            .get("general.architecture")
            .map(String::as_str)
            .unwrap_or("unknown")
    }

    /// Convenience: numeric KV coerced to `usize`.
    ///
    /// Signed integer KVs are checked first: a negative value returns
    /// `None`. Unsigned `UINT64` values larger than `i64::MAX` are still
    /// accepted as long as they fit in `usize`.
    pub fn numeric(&self, key: &str) -> Option<usize> {
        if let Some(&s) = self.signed_numerics.get(key) {
            return usize::try_from(s).ok();
        }
        let &v = self.numerics.get(key)?;
        usize::try_from(v).ok()
    }

    /// Convenience: quantization label.
    ///
    /// Prefers the string KV `general.quantization_type`. When that is
    /// missing, falls back to the **current** `general.file_type` numeric
    /// (GGUF enum): `0 → "F32"`, `1 → "F16"`, otherwise `"GGUF(n)"`.
    /// Returns `"unknown"` when neither is present. Derived at call time
    /// so `Default` + public map edits stay consistent.
    pub fn quantization(&self) -> String {
        if let Some(s) = self.strings.get("general.quantization_type") {
            return s.clone();
        }
        // If a signed value was stored, reject negatives rather than
        // rendering them as huge unsigned labels.
        if let Some(&s) = self.signed_numerics.get("general.file_type") {
            return match s {
                0 => "F32".into(),
                1 => "F16".into(),
                n if n >= 0 => format!("GGUF({n})"),
                _ => "unknown".into(),
            };
        }
        match self.numerics.get("general.file_type").copied() {
            Some(0) => "F32".into(),
            Some(1) => "F16".into(),
            Some(n) => format!("GGUF({n})"),
            None => "unknown".into(),
        }
    }

    /// Convenience: numeric KV coerced to `usize`, looking up
    /// `{architecture}.{key}` (e.g. `olmoe.block_count`).
    ///
    /// Returns `None` if the architecture is unknown or the key is missing.
    pub fn arch_numeric(&self, key: &str) -> Option<usize> {
        let arch = self.architecture();
        if arch == "unknown" {
            return None;
        }
        let full_key = format!("{arch}.{key}");
        self.numeric(&full_key)
    }

    /// Convenience: block count from `{architecture}.block_count`.
    pub fn block_count(&self) -> Option<usize> {
        self.arch_numeric("block_count")
    }

    /// Convenience: expert count from `{architecture}.expert_count`
    /// (some models use `num_experts` instead).
    pub fn expert_count(&self) -> Option<usize> {
        self.arch_numeric("expert_count")
            .or_else(|| self.arch_numeric("num_experts"))
    }

    /// Convenience: number of experts used per token from
    /// `{architecture}.expert_used_count` (some models use
    /// `num_experts_per_tok`).
    pub fn expert_used_count(&self) -> Option<usize> {
        self.arch_numeric("expert_used_count")
            .or_else(|| self.arch_numeric("num_experts_per_tok"))
    }

    /// Convenience: embedding length from `{architecture}.embedding_length`.
    pub fn embedding_length(&self) -> Option<usize> {
        self.arch_numeric("embedding_length")
    }

    /// Convenience: attention head count from
    /// `{architecture}.attention.head_count`.
    pub fn head_count(&self) -> Option<usize> {
        self.arch_numeric("attention.head_count")
    }

    /// Generic string metadata lookup.
    pub fn string(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(String::as_str)
    }

    /// Generic f32 metadata lookup.
    pub fn float32(&self, key: &str) -> Option<f32> {
        self.floats_32.get(key).copied()
    }

    /// Generic f64 metadata lookup.
    pub fn float64(&self, key: &str) -> Option<f64> {
        self.floats_64.get(key).copied()
    }
}

/// Fully-parsed GGUF checkpoint layout.
///
/// Owns the raw file bytes (loaded into a `Vec<u8>` by [`load_gguf`])
/// and all derived tensor directory + metadata information. Callers use
/// the [`Tensor`] handles + [`GgufLayout::tensor_bytes`] to obtain raw
/// byte slices for each weight.
///
/// [`load_gguf`]: super::load_gguf
#[derive(Debug)]
pub struct GgufLayout {
    /// Path the checkpoint was loaded from (kept for error messages).
    pub path: String,
    /// Parsed KV metadata.
    pub metadata: GgufMetadata,
    /// Name -> tensor directory entry.
    pub tensors: HashMap<String, Tensor>,
    /// Byte alignment specified by the file (default 32).
    pub alignment: usize,
    /// Absolute byte offset within `bytes` where tensor payloads begin.
    pub tensor_data_offset: usize,
    /// Full file contents. Tensor payloads live at `bytes[tensor.absolute_offset ..]`.
    pub bytes: Vec<u8>,
}

impl GgufLayout {
    /// Return a borrowed slice of the raw tensor payload bytes.
    pub fn tensor_bytes<'a>(&'a self, tensor: &Tensor) -> Result<&'a [u8]> {
        tensor_payload_bytes(&self.bytes, &self.path, tensor)
    }

    /// Lookup a tensor by exact name.
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| ParserError::MissingTensor {
                name: name.to_owned(),
                path: self.path.clone(),
            })
    }

    /// Find all tensors whose name ends with the given suffix. Useful
    /// for locating per-block MoE tensors like `ffn_gate_exps.weight`.
    pub fn find_tensors_with_suffix<'a>(&'a self, suffix: &str) -> Vec<&'a Tensor> {
        let mut matches: Vec<&Tensor> = self
            .tensors
            .values()
            .filter(|t| t.name.ends_with(suffix))
            .collect();
        matches.sort_unstable_by_key(|t| tensor_block_sort_key(&t.name));
        matches
    }
}

/// Bounds-checked tensor payload slice from any byte source (owned buffer or mmap).
pub(crate) fn tensor_payload_bytes<'a>(
    bytes: &'a [u8],
    path: &str,
    tensor: &Tensor,
) -> Result<&'a [u8]> {
    let start = tensor.absolute_offset;
    let end = start.checked_add(tensor.byte_len).ok_or_else(|| {
        invalid_layout(
            path,
            format!("tensor '{}' byte-length overflow", tensor.name),
        )
    })?;
    if end > bytes.len() {
        return Err(invalid_layout(
            path,
            format!(
                "tensor '{}' extends beyond mapped file ({end} > {})",
                tensor.name,
                bytes.len()
            ),
        ));
    }
    Ok(&bytes[start..end])
}

struct LayoutHeader {
    tensor_count: usize,
    kv_count: usize,
}

/// Parse the GGUF header + KV metadata + tensor directory out of a
/// byte slice. Does not validate payload bytes, only directory offsets.
pub(crate) fn parse_layout(
    bytes: &[u8],
    path: &str,
) -> Result<(GgufMetadata, HashMap<String, Tensor>, usize, usize)> {
    let mut cursor = GgufCursor::new(bytes, path);
    let header = read_layout_header(&mut cursor, path)?;
    let (alignment, metadata) = read_metadata_section(&mut cursor, path, header.kv_count)?;
    let mut tensors = read_tensor_directory(&mut cursor, path, header.tensor_count)?;
    let tensor_data_offset = finalize_tensor_offsets(&mut tensors, cursor.offset(), alignment);
    Ok((metadata, tensors, alignment, tensor_data_offset))
}

fn read_layout_header(cursor: &mut GgufCursor<'_>, path: &str) -> Result<LayoutHeader> {
    validate_gguf_header(cursor, path)?;
    let tensor_count = bounded_count(cursor.read_u64()?, MAX_TENSOR_COUNT, "tensor_count", path)?;
    let kv_count = bounded_count(cursor.read_u64()?, MAX_KV_COUNT, "kv_count", path)?;
    Ok(LayoutHeader {
        tensor_count,
        kv_count,
    })
}

fn validate_gguf_header(cursor: &mut GgufCursor<'_>, path: &str) -> Result<()> {
    let magic = cursor.read_exact(4)?;
    if magic != GGUF_MAGIC {
        return Err(unsupported(
            path,
            format!("unrecognised GGUF magic bytes: {magic:?}"),
        ));
    }

    let version = cursor.read_u32()?;
    if version != GGUF_VERSION {
        return Err(unsupported(
            path,
            format!("unsupported GGUF version {version}; expected {GGUF_VERSION}"),
        ));
    }

    Ok(())
}

fn bounded_count(raw: u64, limit: u64, label: &str, path: &str) -> Result<usize> {
    if raw > limit {
        return Err(unsupported(
            path,
            format!("{label} {raw} exceeds sanity limit {limit}"),
        ));
    }
    Ok(raw as usize)
}

fn read_metadata_section(
    cursor: &mut GgufCursor<'_>,
    path: &str,
    kv_count: usize,
) -> Result<(usize, GgufMetadata)> {
    let mut alignment: usize = 32;
    let mut metadata = GgufMetadata::default();

    for _ in 0..kv_count {
        let key = cursor.read_string()?;
        let value_type = cursor.read_u32()?;
        if key == "general.alignment" {
            // Layout-critical: reject signed negatives (do not wrap to huge usize)
            // and require a positive power of two.
            alignment = cursor.read_nonneg_layout_usize(value_type)?;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(invalid_layout(
                    path,
                    format!("general.alignment must be a positive power of 2, got {alignment}"),
                ));
            }
        } else {
            capture_kv(cursor, &mut metadata, key, value_type)?;
        }
    }

    Ok((alignment, metadata))
}

fn read_tensor_directory(
    cursor: &mut GgufCursor<'_>,
    path: &str,
    tensor_count: usize,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let tensor = read_tensor_entry(cursor, path)?;
        tensors.insert(tensor.name.clone(), tensor);
    }
    Ok(tensors)
}

fn read_tensor_entry(cursor: &mut GgufCursor<'_>, path: &str) -> Result<Tensor> {
    let name = cursor.read_string()?;
    let dims = read_tensor_dims(cursor, path, &name)?;
    let ggml_type = cursor.read_u32()?;
    let relative_offset = cursor.read_u64()? as usize;
    let dtype = DType::from_ggml_type(ggml_type);
    let n_elements = tensor_element_count(&dims, &name, path)?;
    validate_blocked_inner_dim(dtype, &dims, &name, path)?;
    let byte_len = tensor_byte_len(dtype, ggml_type, n_elements, &name, path)?;

    Ok(Tensor {
        name,
        dims,
        dtype,
        ggml_type,
        n_elements,
        byte_len,
        relative_offset,
        absolute_offset: 0,
    })
}

/// Blocked quant layouts pack along the **innermost** GGUF dim (`dims[0]`).
/// Total element count alone can accept shapes that cannot form valid blocks
/// per row (e.g. Q4_0 with dims `[16, 2]` → 32 elems but row len 16).
fn validate_blocked_inner_dim(dtype: DType, dims: &[usize], name: &str, path: &str) -> Result<()> {
    let Some(block) = dtype.quant_block_size() else {
        return Ok(());
    };
    let Some(&inner) = dims.first() else {
        return Ok(());
    };
    if inner.is_multiple_of(block) {
        return Ok(());
    }
    Err(invalid_layout(
        path,
        format!(
            "tensor '{name}' innermost dim {inner} is not divisible by \
             quant block size {block} for dtype {}",
            dtype.label()
        ),
    ))
}

fn read_tensor_dims(cursor: &mut GgufCursor<'_>, path: &str, name: &str) -> Result<Vec<usize>> {
    let n_dims_raw = cursor.read_u32()? as usize;
    if n_dims_raw > MAX_TENSOR_DIMS {
        return Err(unsupported(
            path,
            format!("tensor '{name}' has {n_dims_raw} dims; max {MAX_TENSOR_DIMS}"),
        ));
    }

    let mut dims = Vec::with_capacity(n_dims_raw);
    for _ in 0..n_dims_raw {
        dims.push(cursor.read_u64()? as usize);
    }
    Ok(dims)
}

fn tensor_element_count(dims: &[usize], name: &str, path: &str) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| invalid_layout(path, format!("tensor '{name}' element count overflow")))
}

fn tensor_byte_len(
    dtype: DType,
    ggml_type: u32,
    n_elements: usize,
    name: &str,
    path: &str,
) -> Result<usize> {
    dtype.byte_len_for_elements(n_elements).ok_or_else(|| {
        invalid_layout(
            path,
            format!(
                "tensor '{name}' (ggml_type={ggml_type}) has unknown byte-length for {n_elements} elements",
            ),
        )
    })
}

fn finalize_tensor_offsets(
    tensors: &mut HashMap<String, Tensor>,
    cursor_offset: usize,
    alignment: usize,
) -> usize {
    let tensor_data_offset = align_up(cursor_offset, alignment);
    for tensor in tensors.values_mut() {
        tensor.absolute_offset = tensor_data_offset + tensor.relative_offset;
    }
    tensor_data_offset
}

fn capture_kv(
    cursor: &mut GgufCursor<'_>,
    metadata: &mut GgufMetadata,
    key: String,
    value_type: u32,
) -> Result<()> {
    use super::cursor::{
        GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64,
        GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32, GGUF_VALUE_TYPE_INT64,
        GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16, GGUF_VALUE_TYPE_UINT32,
        GGUF_VALUE_TYPE_UINT64,
    };
    match value_type {
        GGUF_VALUE_TYPE_UINT8
        | GGUF_VALUE_TYPE_INT8
        | GGUF_VALUE_TYPE_UINT16
        | GGUF_VALUE_TYPE_INT16
        | GGUF_VALUE_TYPE_UINT32
        | GGUF_VALUE_TYPE_INT32
        | GGUF_VALUE_TYPE_UINT64
        | GGUF_VALUE_TYPE_INT64
        | GGUF_VALUE_TYPE_BOOL => capture_numeric_kv(cursor, metadata, key, value_type),
        GGUF_VALUE_TYPE_FLOAT32 => capture_f32_kv(cursor, metadata, key),
        GGUF_VALUE_TYPE_FLOAT64 => capture_f64_kv(cursor, metadata, key),
        GGUF_VALUE_TYPE_STRING => capture_string_kv(cursor, metadata, key),
        _ => capture_skipped_kv(cursor, value_type),
    }
}

fn capture_numeric_kv(
    cursor: &mut GgufCursor<'_>,
    metadata: &mut GgufMetadata,
    key: String,
    value_type: u32,
) -> Result<()> {
    let v = cursor.read_numeric_as_u64(value_type)?;
    if is_signed_layout_type(value_type) {
        metadata.signed_numerics.insert(key.clone(), v as i64);
    }
    metadata.numerics.insert(key, v);
    Ok(())
}

fn capture_f32_kv(
    cursor: &mut GgufCursor<'_>,
    metadata: &mut GgufMetadata,
    key: String,
) -> Result<()> {
    let v = cursor.read_f32()?;
    metadata.floats_32.insert(key, v);
    Ok(())
}

fn capture_f64_kv(
    cursor: &mut GgufCursor<'_>,
    metadata: &mut GgufMetadata,
    key: String,
) -> Result<()> {
    let v = cursor.read_f64()?;
    metadata.floats_64.insert(key, v);
    Ok(())
}

fn capture_string_kv(
    cursor: &mut GgufCursor<'_>,
    metadata: &mut GgufMetadata,
    key: String,
) -> Result<()> {
    let v = cursor.read_string()?;
    metadata.strings.insert(key, v);
    Ok(())
}

fn capture_skipped_kv(cursor: &mut GgufCursor<'_>, value_type: u32) -> Result<()> {
    cursor.skip_value(value_type)
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

fn tensor_block_sort_key(name: &str) -> (usize, String) {
    let block = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(idx, _)| idx.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    (block, name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_keeps_large_unsigned_and_rejects_negative_signed() {
        let mut meta = GgufMetadata::default();

        // A UINT64 larger than i64::MAX is still accepted when it fits in usize.
        meta.numerics.insert("big".into(), u64::MAX);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(meta.numeric("big"), Some(usize::MAX));

        // A signed negative value must not wrap into a positive usize.
        meta.numerics.insert("neg".into(), (-7i64) as u64);
        meta.signed_numerics.insert("neg".into(), -7);
        assert_eq!(meta.numeric("neg"), None);
    }

    #[test]
    fn quantization_suppresses_negative_signed_file_type() {
        let mut meta = GgufMetadata::default();

        // A negative signed file_type must not render as an unsigned label.
        meta.numerics
            .insert("general.file_type".into(), (-7i64) as u64);
        meta.signed_numerics.insert("general.file_type".into(), -7);
        assert_eq!(meta.quantization(), "unknown");

        // A positive signed file_type still resolves normally.
        meta.signed_numerics.insert("general.file_type".into(), 15);
        assert_eq!(meta.quantization(), "GGUF(15)");

        // An unsigned file_type falls back to the unsigned map.
        meta.signed_numerics.remove("general.file_type");
        meta.numerics.insert("general.file_type".into(), 15);
        assert_eq!(meta.quantization(), "GGUF(15)");
    }
}
