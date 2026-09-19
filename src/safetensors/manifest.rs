// SPDX-License-Identifier: MIT OR Apache-2.0
//! Header-only Safetensors inspection and deterministic manifest generation.

use super::discovery::{SafetensorsCandidateSummary, classify_tensor, discover_candidates};
use super::json::{
    JsonNumber, JsonValue, encode_compact, encode_pretty, parse_json_rejecting_duplicate_keys,
    parse_metadata_object, parse_string_array, stringify_metadata,
};
use super::paths::{
    index_shard_path, is_safetensors_index, parent_or_current, validate_path_stays_under_root,
};
use super::relative_path;
use super::validate::{
    expected_tensor_byte_size, reject_output_checkpoint_conflict, reject_tensor_data_ranges,
};
use super::{duplicate_tensor_ownership, io_error, model_load, unsupported};
use crate::error::Result;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

pub(super) const SAFETENSORS_EXTENSION: &str = "safetensors";
pub(super) const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
pub(super) const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
/// Reserved index metadata key for shards present on disk but not in `weight_map`.
/// Unreferenced shards are reported here rather than rejected. Missing referenced
/// shards fail closed with [`crate::ParserError::MissingShard`] before a manifest
/// is returned.
pub const INDEX_UNREFERENCED_SHARDS_KEY: &str = "index:unreferenced_shards";
/// Unambiguous boundary between shard relative path and logical metadata key in
/// `shard:*` manifest keys (avoids ambiguity when the logical key contains `:`).
pub(super) const SHARD_METADATA_KEY_SEP: char = '\u{001e}';

/// Deterministic Safetensors checkpoint inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsManifest {
    /// Manifest schema version (currently 2).
    pub manifest_version: u32,
    /// Always `"safetensors"`.
    pub format: &'static str,
    /// Input kind, shard count, and flattened metadata.
    pub checkpoint: SafetensorsCheckpointSource,
    /// Tensor records sorted by `(name, source_shard)`.
    pub tensors: Vec<SafetensorsTensorRecord>,
    /// MoE router/expert candidates inferred from names and shapes.
    pub candidates: SafetensorsCandidateSummary,
}

/// Where a checkpoint came from and the metadata collected from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsCheckpointSource {
    /// `single_file`, `hf_index`, or `directory`.
    pub input_kind: String,
    /// Relative index path when `input_kind` is `hf_index`.
    pub index_file: Option<String>,
    /// Number of shards inspected.
    pub shard_count: usize,
    /// Number of tensor records in the manifest.
    pub tensor_count: usize,
    /// Flattened string metadata (index-prefixed, shard-namespaced, …).
    pub metadata: BTreeMap<String, String>,
}

/// One tensor from a Safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsTensorRecord {
    /// Tensor name.
    pub name: String,
    /// Safetensors dtype label (`F32`, `F16`, …).
    pub dtype: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Payload byte length implied by `data_offsets`.
    pub byte_size: usize,
    /// Relative shard path.
    pub source_shard: String,
    /// Inclusive-exclusive payload offsets within the shard data section.
    pub data_offsets: [usize; 2],
    /// Discovery labels (`moe_router_candidate`, …).
    pub labels: Vec<&'static str>,
}

#[derive(Debug)]
pub(super) struct RawIndex {
    pub(super) metadata: BTreeMap<String, JsonValue>,
    pub(super) weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
pub(super) struct ShardInspection {
    pub(super) metadata: BTreeMap<String, String>,
    pub(super) tensors: Vec<SafetensorsTensorRecord>,
}

/// Inspect a Safetensors file, sharded Safetensors index, or directory of
/// Safetensors shards and return a deterministic manifest.
pub fn inspect_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsManifest> {
    let input = path.as_ref();
    let metadata = fs::metadata(input).map_err(|e| io_error(input, e))?;

    if metadata.is_dir() {
        return inspect_directory(input);
    }

    if is_safetensors_index(input) {
        let root = parent_or_current(input);
        return inspect_index(root, input);
    }

    if input.extension().and_then(|ext| ext.to_str()) == Some(SAFETENSORS_EXTENSION) {
        return inspect_single_file(input);
    }

    Err(unsupported(
        input,
        format!(
            "expected .safetensors file, .safetensors.index.json file, or directory, got '{}'",
            input.display()
        ),
    ))
}

/// Write a deterministic pretty-printed Safetensors manifest to `output_path`.
pub fn write_safetensors_manifest(
    checkpoint_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<SafetensorsManifest> {
    let checkpoint_path = checkpoint_path.as_ref();
    let output_path = output_path.as_ref();
    let manifest = inspect_safetensors_checkpoint(checkpoint_path)?;
    reject_output_checkpoint_conflict(checkpoint_path, output_path, &manifest)?;
    let json = manifest.to_pretty_json();
    fs::write(output_path, json).map_err(|e| io_error(output_path, e))?;
    Ok(manifest)
}

impl SafetensorsManifest {
    /// Pretty-print this manifest as deterministic JSON.
    #[must_use]
    pub fn to_pretty_json(&self) -> String {
        encode_pretty(&self.to_json_value())
    }

    fn to_json_value(&self) -> JsonValue {
        let mut checkpoint = BTreeMap::new();
        checkpoint.insert(
            "input_kind".into(),
            JsonValue::String(self.checkpoint.input_kind.clone()),
        );
        checkpoint.insert(
            "index_file".into(),
            match &self.checkpoint.index_file {
                Some(path) => JsonValue::String(path.clone()),
                None => JsonValue::Null,
            },
        );
        checkpoint.insert(
            "shard_count".into(),
            JsonValue::Number(JsonNumber::U64(self.checkpoint.shard_count as u64)),
        );
        checkpoint.insert(
            "tensor_count".into(),
            JsonValue::Number(JsonNumber::U64(self.checkpoint.tensor_count as u64)),
        );
        checkpoint.insert(
            "metadata".into(),
            JsonValue::Object(
                self.checkpoint
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), JsonValue::String(v.clone())))
                    .collect(),
            ),
        );

        let tensors = self
            .tensors
            .iter()
            .map(tensor_record_json)
            .collect::<Vec<_>>();
        let candidates = candidates_json(&self.candidates);

        let mut root = BTreeMap::new();
        root.insert(
            "manifest_version".into(),
            JsonValue::Number(JsonNumber::U64(u64::from(self.manifest_version))),
        );
        root.insert("format".into(), JsonValue::String(self.format.to_string()));
        root.insert("checkpoint".into(), JsonValue::Object(checkpoint));
        root.insert("tensors".into(), JsonValue::Array(tensors));
        root.insert("candidates".into(), candidates);
        JsonValue::Object(root)
    }
}

fn tensor_record_json(tensor: &SafetensorsTensorRecord) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert("name".into(), JsonValue::String(tensor.name.clone()));
    object.insert("dtype".into(), JsonValue::String(tensor.dtype.clone()));
    object.insert(
        "shape".into(),
        JsonValue::Array(
            tensor
                .shape
                .iter()
                .map(|dim| JsonValue::Number(JsonNumber::U64(*dim as u64)))
                .collect(),
        ),
    );
    object.insert(
        "byte_size".into(),
        JsonValue::Number(JsonNumber::U64(tensor.byte_size as u64)),
    );
    object.insert(
        "source_shard".into(),
        JsonValue::String(tensor.source_shard.clone()),
    );
    object.insert(
        "data_offsets".into(),
        JsonValue::Array(vec![
            JsonValue::Number(JsonNumber::U64(tensor.data_offsets[0] as u64)),
            JsonValue::Number(JsonNumber::U64(tensor.data_offsets[1] as u64)),
        ]),
    );
    object.insert(
        "labels".into(),
        JsonValue::Array(
            tensor
                .labels
                .iter()
                .map(|label| JsonValue::String((*label).to_string()))
                .collect(),
        ),
    );
    JsonValue::Object(object)
}

fn candidates_json(summary: &SafetensorsCandidateSummary) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert(
        "detected_layout_family".into(),
        match summary.detected_layout_family {
            Some(family) => JsonValue::String(family.to_string()),
            None => JsonValue::Null,
        },
    );
    object.insert(
        "router_tensors".into(),
        JsonValue::Array(
            summary
                .router_tensors
                .iter()
                .map(|name| JsonValue::String(name.clone()))
                .collect(),
        ),
    );
    object.insert(
        "expert_tensors".into(),
        JsonValue::Array(
            summary
                .expert_tensors
                .iter()
                .map(|name| JsonValue::String(name.clone()))
                .collect(),
        ),
    );
    object.insert(
        "router_candidates".into(),
        JsonValue::Array(
            summary
                .router_candidates
                .iter()
                .map(router_candidate_json)
                .collect(),
        ),
    );
    object.insert(
        "expert_groups".into(),
        JsonValue::Array(
            summary
                .expert_groups
                .iter()
                .map(expert_group_json)
                .collect(),
        ),
    );
    JsonValue::Object(object)
}

fn router_candidate_json(candidate: &super::SafetensorsRouterCandidate) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert("name".into(), JsonValue::String(candidate.name.clone()));
    object.insert(
        "layer_hint".into(),
        match candidate.layer_hint {
            Some(hint) => JsonValue::Number(JsonNumber::U64(hint as u64)),
            None => JsonValue::Null,
        },
    );
    object.insert(
        "source_shard".into(),
        JsonValue::String(candidate.source_shard.clone()),
    );
    object.insert(
        "shape".into(),
        JsonValue::Array(
            candidate
                .shape
                .iter()
                .map(|dim| JsonValue::Number(JsonNumber::U64(*dim as u64)))
                .collect(),
        ),
    );
    object.insert(
        "score".into(),
        JsonValue::Number(JsonNumber::U64(u64::from(candidate.score))),
    );
    object.insert(
        "reasons".into(),
        JsonValue::Array(
            candidate
                .reasons
                .iter()
                .map(|reason| JsonValue::String((*reason).to_string()))
                .collect(),
        ),
    );
    JsonValue::Object(object)
}

fn expert_group_json(group: &super::SafetensorsExpertGroup) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert(
        "group_key".into(),
        JsonValue::String(group.group_key.clone()),
    );
    object.insert(
        "layer_hint".into(),
        match group.layer_hint {
            Some(hint) => JsonValue::Number(JsonNumber::U64(hint as u64)),
            None => JsonValue::Null,
        },
    );
    object.insert(
        "expert_indices".into(),
        JsonValue::Array(
            group
                .expert_indices
                .iter()
                .map(|idx| JsonValue::Number(JsonNumber::U64(*idx as u64)))
                .collect(),
        ),
    );
    object.insert(
        "tensor_names".into(),
        JsonValue::Array(
            group
                .tensor_names
                .iter()
                .map(|name| JsonValue::String(name.clone()))
                .collect(),
        ),
    );
    object.insert(
        "source_shards".into(),
        JsonValue::Array(
            group
                .source_shards
                .iter()
                .map(|shard| JsonValue::String(shard.clone()))
                .collect(),
        ),
    );
    object.insert(
        "weight_kinds".into(),
        JsonValue::Array(
            group
                .weight_kinds
                .iter()
                .map(|kind| JsonValue::String((*kind).to_string()))
                .collect(),
        ),
    );
    JsonValue::Object(object)
}

pub(super) fn inspect_single_file(path: &Path) -> Result<SafetensorsManifest> {
    let root = parent_or_current(path);
    let shard = inspect_shard(path, root)?;
    build_manifest(
        "single_file",
        None,
        vec![path.to_path_buf()],
        shard.metadata,
        shard.tensors,
        path,
    )
}

pub(super) fn inspect_directory(root: &Path) -> Result<SafetensorsManifest> {
    if let Some(index_path) = find_index_file(root)? {
        return inspect_index(root, &index_path);
    }

    let shards = list_safetensors_files(root)?;
    if shards.is_empty() {
        return Err(unsupported(
            root,
            format!("no .safetensors files found in '{}'", root.display()),
        ));
    }
    inspect_shards("directory", root, None, shards)
}

pub(super) fn inspect_index(root: &Path, index_path: &Path) -> Result<SafetensorsManifest> {
    let raw = read_index(index_path)?;
    let index_tensor_count = raw.weight_map.len();
    let mut expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for (tensor_name, relative) in raw.weight_map {
        let shard_path = index_shard_path(root, index_path, &relative)?;
        expected_by_shard
            .entry(shard_path)
            .or_default()
            .insert(tensor_name);
    }
    let shards = expected_by_shard.keys().cloned().collect::<Vec<_>>();

    let mut metadata = stringify_metadata("index", raw.metadata);
    metadata.remove(INDEX_UNREFERENCED_SHARDS_KEY);
    let indexed_shards = shards.iter().cloned().collect::<BTreeSet<_>>();
    let unreferenced_shards = list_safetensors_files(root)?
        .into_iter()
        .filter(|path| !indexed_shards.contains(path))
        .map(|path| relative_path(&path, root))
        .collect::<Vec<_>>();
    let unreferenced_shards_json = if unreferenced_shards.is_empty() {
        None
    } else {
        let encoded = encode_compact(&JsonValue::Array(
            unreferenced_shards
                .into_iter()
                .map(JsonValue::String)
                .collect(),
        ));
        Some(encoded)
    };
    let index_file = Some(relative_path(index_path, root));
    inspect_index_shards(
        root,
        index_file,
        shards,
        expected_by_shard,
        metadata,
        index_tensor_count,
        unreferenced_shards_json,
    )
}

pub(super) fn inspect_index_shards(
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>>,
    mut metadata: BTreeMap<String, String>,
    index_tensor_count: usize,
    unreferenced_shards_json: Option<String>,
) -> Result<SafetensorsManifest> {
    let tensor_owners = expected_tensor_owners(&expected_by_shard);
    let mut inspections = Vec::new();
    for shard_path in &shards {
        if !expected_by_shard.contains_key(shard_path) {
            return Err(model_load(
                shard_path,
                "internal error: index shard has no expected tensor set".into(),
            ));
        }
        inspections.push((shard_path.clone(), inspect_shard(shard_path, root)?));
    }

    reject_indexed_tensor_ownership(root, &tensor_owners, &inspections)?;

    let mut tensors = Vec::new();
    for (shard_path, shard) in inspections {
        let expected = expected_by_shard.get(&shard_path).ok_or_else(|| {
            model_load(
                &shard_path,
                "internal error: index shard has no expected tensor set".into(),
            )
        })?;
        merge_shard_metadata(
            &mut metadata,
            &relative_path(&shard_path, root),
            shard.metadata,
        );

        let found = shard
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        if let Some(missing) = expected.difference(&found).next() {
            return Err(model_load(
                &shard_path,
                format!(
                    "index maps tensor '{missing}' to this shard, but the shard header does not contain it"
                ),
            ));
        }

        tensors.extend(
            shard
                .tensors
                .into_iter()
                .filter(|tensor| expected.contains(&tensor.name)),
        );
    }

    metadata.insert("index_tensor_count".into(), index_tensor_count.to_string());
    metadata.remove(INDEX_UNREFERENCED_SHARDS_KEY);
    if let Some(encoded) = unreferenced_shards_json {
        metadata.insert(INDEX_UNREFERENCED_SHARDS_KEY.into(), encoded);
    }

    build_manifest("hf_index", index_file, shards, metadata, tensors, root)
}

pub(super) fn inspect_shards(
    input_kind: &str,
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
) -> Result<SafetensorsManifest> {
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for shard_path in &shards {
        let shard = inspect_shard(shard_path, root)?;
        merge_shard_metadata(
            &mut metadata,
            &relative_path(shard_path, root),
            shard.metadata,
        );
        tensors.extend(shard.tensors);
    }

    build_manifest(input_kind, index_file, shards, metadata, tensors, root)
}

fn expected_tensor_owners(
    expected_by_shard: &BTreeMap<PathBuf, BTreeSet<String>>,
) -> BTreeMap<String, PathBuf> {
    let mut owners = BTreeMap::new();
    for (shard_path, names) in expected_by_shard {
        for name in names {
            owners.insert(name.clone(), shard_path.clone());
        }
    }
    owners
}

fn reject_indexed_tensor_ownership(
    root: &Path,
    tensor_owners: &BTreeMap<String, PathBuf>,
    inspections: &[(PathBuf, ShardInspection)],
) -> Result<()> {
    let mut found_in: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (shard_path, shard) in inspections {
        let relative = relative_path(shard_path, root);
        for tensor in &shard.tensors {
            if tensor_owners.contains_key(&tensor.name) {
                found_in
                    .entry(tensor.name.clone())
                    .or_default()
                    .insert(relative.clone());
            }
        }
    }

    for (name, expected_path) in tensor_owners {
        let Some(owners) = found_in.get(name) else {
            continue;
        };
        if owners.len() > 1 {
            return Err(duplicate_tensor_ownership(
                root,
                name.clone(),
                owners.iter().cloned().collect(),
            ));
        }
        let expected_rel = relative_path(expected_path, root);
        if !owners.contains(&expected_rel) {
            return Err(duplicate_tensor_ownership(
                root,
                name.clone(),
                owners
                    .iter()
                    .cloned()
                    .chain(std::iter::once(expected_rel))
                    .collect(),
            ));
        }
    }
    Ok(())
}

pub(super) fn build_manifest(
    input_kind: &str,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    metadata: BTreeMap<String, String>,
    mut tensors: Vec<SafetensorsTensorRecord>,
    error_path: &Path,
) -> Result<SafetensorsManifest> {
    tensors.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.source_shard.cmp(&right.source_shard))
    });
    reject_duplicate_tensor_ownership(error_path, &tensors)?;
    let candidates = discover_candidates(&tensors);

    Ok(SafetensorsManifest {
        manifest_version: 2,
        format: "safetensors",
        checkpoint: SafetensorsCheckpointSource {
            input_kind: input_kind.to_string(),
            index_file,
            shard_count: shards.len(),
            tensor_count: tensors.len(),
            metadata,
        },
        tensors,
        candidates,
    })
}

fn reject_duplicate_tensor_ownership(
    path: &Path,
    tensors: &[SafetensorsTensorRecord],
) -> Result<()> {
    let mut owners: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for tensor in tensors {
        owners
            .entry(tensor.name.as_str())
            .or_default()
            .insert(tensor.source_shard.as_str());
    }
    if let Some((name, shards)) = owners.into_iter().find(|(_, shards)| shards.len() > 1) {
        return Err(duplicate_tensor_ownership(
            path,
            name,
            shards.into_iter().map(str::to_string).collect(),
        ));
    }
    Ok(())
}

pub(super) fn inspect_shard(path: &Path, root: &Path) -> Result<ShardInspection> {
    let mut file = File::open(path).map_err(|e| io_error(path, e))?;
    let file_len = file.metadata().map_err(|e| io_error(path, e))?.len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header length: {e}")))?;
    let header_len_u64 = u64::from_le_bytes(len_bytes);
    let header_len = usize::try_from(header_len_u64).map_err(|_| {
        model_load(
            path,
            format!("Safetensors header length {header_len_u64} does not fit in usize"),
        )
    })?;
    if header_len > MAX_HEADER_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
        ));
    }
    if 8u64
        .checked_add(header_len_u64)
        .is_none_or(|end| end > file_len)
    {
        return Err(model_load(
            path,
            "Safetensors header extends beyond file".into(),
        ));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header: {e}")))?;
    let header = parse_json_rejecting_duplicate_keys(&header_bytes, path, "Safetensors header")?;
    parse_header(path, root, file_len, header_len_u64, header)
}

pub(super) fn parse_header(
    path: &Path,
    root: &Path,
    file_len: u64,
    header_len: u64,
    header: JsonValue,
) -> Result<ShardInspection> {
    let object = header
        .as_object()
        .ok_or_else(|| model_load(path, "Safetensors header must be a JSON object".to_string()))?;
    let data_len = file_len
        .checked_sub(8)
        .and_then(|len| len.checked_sub(header_len))
        .ok_or_else(|| model_load(path, "Safetensors data range underflow".into()))?;
    let source_shard = relative_path(path, root);
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for (name, value) in object {
        if name == "__metadata__" {
            metadata.extend(parse_metadata_object(path, value)?);
            continue;
        }

        let tensor = value.as_object().ok_or_else(|| {
            model_load(path, format!("tensor '{name}' metadata must be an object"))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing dtype")))?
            .to_string();
        let shape = tensor
            .get("shape")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing shape")))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets must have length 2"),
            ));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid start offset")))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid end offset")))?;
        if start > end {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets are reversed"),
            ));
        }
        if (end as u64) > data_len {
            return Err(model_load(
                path,
                format!("tensor '{name}' extends beyond Safetensors data section"),
            ));
        }
        let byte_size = end - start;
        let expected = expected_tensor_byte_size(&dtype, &shape, path, name)?;
        if expected != byte_size {
            return Err(model_load(
                path,
                format!(
                    "tensor '{name}' byte size mismatch: shape/dtype imply {expected} bytes, data_offsets span {byte_size} bytes"
                ),
            ));
        }

        tensors.push(SafetensorsTensorRecord {
            name: name.clone(),
            dtype,
            shape: shape.clone(),
            byte_size,
            source_shard: source_shard.clone(),
            data_offsets: [start, end],
            labels: classify_tensor(name, &shape),
        });
    }

    reject_tensor_data_ranges(path, &tensors, data_len)?;

    Ok(ShardInspection { metadata, tensors })
}

pub(super) fn read_index(path: &Path) -> Result<RawIndex> {
    let len = fs::metadata(path).map_err(|e| io_error(path, e))?.len();
    if len > MAX_INDEX_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors index is {len} bytes, exceeding limit {MAX_INDEX_BYTES}"),
        ));
    }
    let bytes = fs::read(path).map_err(|e| io_error(path, e))?;
    let value = parse_json_rejecting_duplicate_keys(&bytes, path, "Safetensors index")?;
    parse_raw_index(path, &value)
}

fn parse_raw_index(path: &Path, value: &JsonValue) -> Result<RawIndex> {
    let object = value
        .as_object()
        .ok_or_else(|| model_load(path, "Safetensors index must be a JSON object".into()))?;
    let metadata = match object.get("metadata") {
        None => BTreeMap::new(),
        Some(JsonValue::Object(map)) => map.clone(),
        Some(_) => {
            return Err(model_load(
                path,
                "Safetensors index metadata must be a JSON object".into(),
            ));
        }
    };
    let weight_map_value = object
        .get("weight_map")
        .ok_or_else(|| model_load(path, "Safetensors index is missing weight_map".into()))?;
    let weight_object = weight_map_value.as_object().ok_or_else(|| {
        model_load(
            path,
            "Safetensors index weight_map must be an object".into(),
        )
    })?;
    let mut weight_map = BTreeMap::new();
    for (tensor, shard) in weight_object {
        let shard = shard.as_str().ok_or_else(|| {
            model_load(
                path,
                format!("index weight_map entry '{tensor}' must be a string shard path"),
            )
        })?;
        weight_map.insert(tensor.clone(), shard.to_string());
    }
    Ok(RawIndex {
        metadata,
        weight_map,
    })
}

pub(super) fn find_index_file(root: &Path) -> Result<Option<PathBuf>> {
    let mut candidates = Vec::new();
    for path in read_dir_paths(root)? {
        if is_safetensors_index(&path) && is_regular_file(&path)? {
            validate_path_stays_under_root(root, &path)?;
            candidates.push(path);
        }
    }
    candidates.sort();
    if candidates.len() > 1 {
        let names = candidates
            .iter()
            .map(|path| relative_path(path, root))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(unsupported(
            root,
            format!(
                "multiple Safetensors index files found in '{}': {names}; pass the intended index file explicitly",
                root.display()
            ),
        ));
    }
    Ok(candidates.pop())
}

pub(super) fn list_safetensors_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = Vec::new();
    for path in read_dir_paths(root)? {
        if path.extension().and_then(|ext| ext.to_str()) != Some(SAFETENSORS_EXTENSION) {
            continue;
        }
        if !is_regular_file(&path)? {
            continue;
        }
        validate_path_stays_under_root(root, &path)?;
        shards.push(path);
    }
    shards.sort();
    Ok(shards)
}

pub(super) fn is_regular_file(path: &Path) -> Result<bool> {
    fs::metadata(path)
        .map(|metadata| metadata.is_file())
        .map_err(|e| io_error(path, e))
}

pub(super) fn read_dir_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(root).map_err(|e| io_error(root, e))? {
        let entry = entry.map_err(|e| model_load(root, format!("read directory entry: {e}")))?;
        paths.push(entry.path());
    }
    paths.sort();
    Ok(paths)
}

pub(super) fn shard_metadata_namespaced_key(shard_path: &str, key: &str) -> String {
    format!("shard:{shard_path}{SHARD_METADATA_KEY_SEP}{key}")
}

pub(super) fn namespaced_shard_metadata_logical_key(namespaced: &str) -> Option<&str> {
    let rest = namespaced.strip_prefix("shard:")?;
    if let Some((_, logical_key)) = rest.split_once(SHARD_METADATA_KEY_SEP) {
        return Some(logical_key);
    }
    rest.rsplit_once(':').map(|(_, logical_key)| logical_key)
}

pub(super) fn merge_shard_metadata(
    metadata: &mut BTreeMap<String, String>,
    shard_path: &str,
    shard_metadata: BTreeMap<String, String>,
) {
    for (key, value) in shard_metadata {
        let namespaced_key = shard_metadata_namespaced_key(shard_path, &key);
        match metadata.get(&key) {
            None => {
                if !has_shard_metadata_key(metadata, &key) {
                    metadata.insert(key.clone(), value.clone());
                }
                metadata.insert(namespaced_key, value);
            }
            Some(existing) if existing == &value => {
                metadata.insert(namespaced_key, value);
            }
            Some(_) => {
                metadata.remove(&key);
                metadata.insert(namespaced_key, value);
            }
        }
    }
}

pub(super) fn has_shard_metadata_key(metadata: &BTreeMap<String, String>, key: &str) -> bool {
    metadata
        .keys()
        .any(|existing| namespaced_shard_metadata_logical_key(existing).is_some_and(|k| k == key))
}

pub(super) fn parse_unreferenced_shards_metadata(encoded: &str) -> Vec<String> {
    if let Some(paths) = parse_string_array(encoded) {
        return paths;
    }
    encoded
        .split(',')
        .filter(|shard| !shard.is_empty())
        .map(str::to_string)
        .collect()
}
