// SPDX-License-Identifier: MIT OR Apache-2.0
//! Name/shape heuristics for MoE router and expert candidate discovery.

use super::SafetensorsTensorRecord;
use std::collections::{BTreeMap, BTreeSet};

/// Layout-family guess plus scored router/expert candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsCandidateSummary {
    /// Inferred family when both router and expert evidence is present.
    pub detected_layout_family: Option<&'static str>,
    /// Router tensor names, score-then-name ordered.
    pub router_tensors: Vec<String>,
    /// Expert tensor names in discovery order.
    pub expert_tensors: Vec<String>,
    /// Scored router candidates.
    pub router_candidates: Vec<SafetensorsRouterCandidate>,
    /// Expert groups keyed by the path prefix before `experts`/`expert`.
    pub expert_groups: Vec<SafetensorsExpertGroup>,
}

/// A tensor whose name/shape looks like a MoE router / gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsRouterCandidate {
    /// Tensor name from the Safetensors header.
    pub name: String,
    /// Layer index parsed from the name, if present.
    pub layer_hint: Option<usize>,
    /// Relative shard path that owns this tensor.
    pub source_shard: String,
    /// Tensor shape from the header.
    pub shape: Vec<usize>,
    /// Heuristic score (higher is more router-like).
    pub score: u32,
    /// Human-readable score reasons.
    pub reasons: Vec<&'static str>,
}

/// A group of expert tensors that share a path prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsExpertGroup {
    /// Path prefix up to (but not including) `experts`/`expert`.
    pub group_key: String,
    /// Layer index parsed from the name, if present.
    pub layer_hint: Option<usize>,
    /// Distinct expert indices observed in this group.
    pub expert_indices: Vec<usize>,
    /// Tensor names in this group, sorted.
    pub tensor_names: Vec<String>,
    /// Shards contributing tensors to this group, sorted.
    pub source_shards: Vec<String>,
    /// Weight-kind labels (`w1`, `gate_proj`, `unknown`, …), sorted.
    pub weight_kinds: Vec<&'static str>,
}

#[derive(Debug, Clone)]
struct TensorNameParts {
    lower: String,
    path_segments: Vec<String>,
    all_segments: Vec<String>,
}

#[derive(Debug, Clone)]
struct ExpertTensorParts {
    layer_hint: Option<usize>,
    expert_index: Option<usize>,
    group_key: String,
    weight_kind: &'static str,
}

#[derive(Debug)]
struct ExpertGroupBuilder {
    group_key: String,
    layer_hint: Option<usize>,
    expert_indices: BTreeSet<usize>,
    tensor_names: Vec<String>,
    source_shards: BTreeSet<String>,
    weight_kinds: BTreeSet<&'static str>,
}

/// Discover MoE router/expert candidates from header metadata only.
#[must_use]
pub fn discover_candidates(tensors: &[SafetensorsTensorRecord]) -> SafetensorsCandidateSummary {
    let mut router_candidates = Vec::new();
    let mut expert_records = Vec::new();
    let mut family_evidence = LayoutFamilyEvidence::default();

    for tensor in tensors {
        let parts = TensorNameParts::new(&tensor.name);
        let expert_parts = expert_tensor_parts(&parts);
        let router_scoring = expert_parts
            .is_none()
            .then(|| router_candidate_score(&parts, &tensor.shape))
            .flatten();

        family_evidence.observe(&parts, router_scoring.is_some(), expert_parts.is_some());

        if let Some((score, reasons)) = router_scoring {
            router_candidates.push(SafetensorsRouterCandidate {
                name: tensor.name.clone(),
                layer_hint: extract_layer_hint(&parts),
                source_shard: tensor.source_shard.clone(),
                shape: tensor.shape.clone(),
                score,
                reasons,
            });
        }

        if let Some(expert_parts) = expert_parts {
            expert_records.push((tensor, expert_parts));
        }
    }

    router_candidates.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then(left.name.cmp(&right.name))
            .then(left.source_shard.cmp(&right.source_shard))
    });

    let expert_groups = discover_expert_groups(expert_records.iter().cloned());
    let expert_tensors = expert_records
        .iter()
        .map(|(tensor, _)| tensor.name.clone())
        .collect::<Vec<_>>();

    SafetensorsCandidateSummary {
        detected_layout_family: family_evidence.detected_layout_family(),
        router_tensors: router_candidates
            .iter()
            .map(|candidate| candidate.name.clone())
            .collect(),
        expert_tensors,
        router_candidates,
        expert_groups,
    }
}

/// Classify one tensor name/shape as expert, router, or possible router shape.
#[must_use]
pub fn classify_tensor(name: &str, shape: &[usize]) -> Vec<&'static str> {
    let parts = TensorNameParts::new(name);
    let mut labels = Vec::new();

    if expert_tensor_parts(&parts).is_some() {
        labels.push("moe_expert_candidate");
    } else if router_candidate_score(&parts, shape).is_some() {
        labels.push("moe_router_candidate");
    } else if router_shape_hint(shape) {
        labels.push("possible_moe_router_shape");
    }

    labels
}

impl TensorNameParts {
    fn new(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        let path_segments = split_name(&lower, &['.', '/']);
        let all_segments = split_name(&lower, &['.', '/', '_']);
        Self {
            lower,
            path_segments,
            all_segments,
        }
    }

    fn has_path_segment(&self, needle: &str) -> bool {
        self.path_segments.iter().any(|segment| segment == needle)
    }

    fn has_any_segment(&self, needle: &str) -> bool {
        self.all_segments.iter().any(|segment| segment == needle)
    }
}

#[derive(Debug, Default)]
struct LayoutFamilyEvidence {
    seen_phimoe: bool,
    seen_granitemoe: bool,
    seen_afmoe: bool,
    seen_lfm2_moe: bool,
    seen_deepseek_v3_family: bool,
    has_router: bool,
    has_expert: bool,
}

impl LayoutFamilyEvidence {
    fn observe(&mut self, parts: &TensorNameParts, is_router: bool, is_expert: bool) {
        self.seen_phimoe |= parts.lower.contains("phimoe") || parts.lower.contains("phi_moe");
        self.seen_granitemoe |=
            parts.lower.contains("granitemoe") || parts.lower.contains("granite_moe");
        self.seen_afmoe |= parts.lower.contains("afmoe") || parts.lower.contains("af_moe");
        self.seen_lfm2_moe |= parts.lower.contains("lfm2_moe") || parts.lower.contains("lfm2moe");
        self.seen_deepseek_v3_family |=
            parts.lower.contains("moonlight") || parts.lower.contains("block_sparse_moe");
        self.has_router |= is_router;
        self.has_expert |= is_expert;
    }

    fn detected_layout_family(&self) -> Option<&'static str> {
        if !(self.has_router && self.has_expert) {
            return None;
        }
        if self.seen_phimoe {
            return Some("phimoe");
        }
        if self.seen_granitemoe {
            return Some("granitemoe");
        }
        if self.seen_afmoe {
            return Some("afmoe");
        }
        if self.seen_lfm2_moe {
            return Some("lfm2_moe");
        }
        if self.seen_deepseek_v3_family {
            return Some("deepseek_v3_family");
        }
        Some("generic_moe")
    }
}

fn router_candidate_score(
    parts: &TensorNameParts,
    shape: &[usize],
) -> Option<(u32, Vec<&'static str>)> {
    let lower = parts.lower.as_str();
    let mut score = 0u32;
    let mut reasons = Vec::new();
    let has_moe = has_moe_context(parts);
    let has_router_shape = router_shape_hint(shape);

    if lower.contains("router") {
        score += 80;
        reasons.push("router_name");
    }
    if lower.contains("gate_inp") {
        score += 80;
        reasons.push("gate_inp_name");
    }
    if lower.contains("block_sparse_moe.gate.weight")
        || lower.contains("block_sparse_moe/gate/weight")
    {
        score += 90;
        reasons.push("deepseek_v3_family_block_sparse_moe_gate");
    }
    if lower.contains("moe.gate.weight")
        || lower.contains("moe/gate/weight")
        || lower.contains("feed_forward.gate.weight")
        || lower.contains("feed_forward/gate/weight")
        || lower.contains("ffn.gate.weight")
        || lower.contains("ffn/gate/weight")
    {
        score += 70;
        reasons.push("moe_gate_name");
    }
    if (lower.contains("gating.weight") || lower.contains("gating/weight"))
        && (has_moe || has_router_shape)
    {
        score += 65;
        reasons.push("gating_name");
    }

    let gate_weight_suffix = lower.ends_with(".gate.weight") || lower.ends_with("/gate/weight");
    if gate_weight_suffix && has_moe {
        score += 60;
        reasons.push("moe_context_gate_weight");
    } else if gate_weight_suffix && has_router_shape {
        score += 45;
        reasons.push("shape_guarded_gate_weight");
    }

    if has_router_shape {
        score += 30;
        reasons.push("router_shape");
    }

    (!reasons.is_empty() && score >= 60).then_some((score, reasons))
}

fn router_shape_hint(shape: &[usize]) -> bool {
    shape.len() == 2
        && shape
            .iter()
            .copied()
            .min()
            .is_some_and(|v| (2..=512).contains(&v))
        && shape.iter().copied().max().is_some_and(|v| v >= 512)
}

fn has_moe_context(parts: &TensorNameParts) -> bool {
    parts.lower.contains("moe")
        || parts.has_path_segment("experts")
        || parts.has_path_segment("expert")
}

fn discover_expert_groups<'a>(
    expert_records: impl Iterator<Item = (&'a SafetensorsTensorRecord, ExpertTensorParts)>,
) -> Vec<SafetensorsExpertGroup> {
    let mut groups: BTreeMap<String, ExpertGroupBuilder> = BTreeMap::new();
    for (tensor, parts) in expert_records {
        groups
            .entry(parts.group_key.clone())
            .or_insert_with(|| ExpertGroupBuilder::new(parts.group_key.clone(), parts.layer_hint))
            .push(tensor, parts.expert_index, parts.weight_kind);
    }

    groups
        .into_values()
        .map(ExpertGroupBuilder::build)
        .collect()
}

impl ExpertGroupBuilder {
    fn new(group_key: String, layer_hint: Option<usize>) -> Self {
        Self {
            group_key,
            layer_hint,
            expert_indices: BTreeSet::new(),
            tensor_names: Vec::new(),
            source_shards: BTreeSet::new(),
            weight_kinds: BTreeSet::new(),
        }
    }

    fn push(
        &mut self,
        tensor: &SafetensorsTensorRecord,
        expert_index: Option<usize>,
        weight_kind: &'static str,
    ) {
        if let Some(expert_index) = expert_index {
            self.expert_indices.insert(expert_index);
        }
        self.tensor_names.push(tensor.name.clone());
        self.source_shards.insert(tensor.source_shard.clone());
        self.weight_kinds.insert(weight_kind);
    }

    fn build(mut self) -> SafetensorsExpertGroup {
        self.tensor_names.sort();
        SafetensorsExpertGroup {
            group_key: self.group_key,
            layer_hint: self.layer_hint,
            expert_indices: self.expert_indices.into_iter().collect(),
            tensor_names: self.tensor_names,
            source_shards: self.source_shards.into_iter().collect(),
            weight_kinds: self.weight_kinds.into_iter().collect(),
        }
    }
}

fn expert_tensor_parts(parts: &TensorNameParts) -> Option<ExpertTensorParts> {
    if !has_moe_context(parts) {
        return None;
    }
    let group_prefix = expert_group_prefix(parts)?;
    Some(ExpertTensorParts {
        layer_hint: extract_layer_hint(parts),
        expert_index: extract_expert_index(parts),
        group_key: group_prefix,
        weight_kind: expert_weight_kind(parts).unwrap_or("unknown"),
    })
}

fn expert_weight_kind(parts: &TensorNameParts) -> Option<&'static str> {
    [
        ("gate_proj", "gate_proj"),
        ("up_proj", "up_proj"),
        ("down_proj", "down_proj"),
        ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"),
        ("linear_1", "linear_1"),
        ("linear_2", "linear_2"),
        ("linear_3", "linear_3"),
        ("fc1", "fc1"),
        ("fc2", "fc2"),
        ("fc3", "fc3"),
        ("w1", "w1"),
        ("w2", "w2"),
        ("w3", "w3"),
    ]
    .into_iter()
    .find_map(|(needle, kind)| has_weight_kind_segment(parts, needle).then_some(kind))
}

fn has_weight_kind_segment(parts: &TensorNameParts, needle: &str) -> bool {
    if parts.has_path_segment(needle) {
        return true;
    }
    needle.contains('_') && parts.has_any_segment(needle)
}

fn extract_layer_hint(parts: &TensorNameParts) -> Option<usize> {
    number_after_any_segment(parts, &["layers", "layer", "blocks", "block", "h"])
}

fn extract_expert_index(parts: &TensorNameParts) -> Option<usize> {
    number_after_any_segment(parts, &["experts", "expert"])
}

fn expert_group_prefix(parts: &TensorNameParts) -> Option<String> {
    for marker in ["experts", "expert"] {
        if let Some(index) = parts
            .path_segments
            .iter()
            .position(|segment| segment == marker)
        {
            if index == 0 {
                return Some(marker.to_string());
            }
            return Some(parts.path_segments[..index].join("."));
        }
    }
    None
}

fn number_after_any_segment(parts: &TensorNameParts, markers: &[&str]) -> Option<usize> {
    parts.all_segments.windows(2).find_map(|window| {
        markers
            .contains(&window[0].as_str())
            .then(|| window[1].parse::<usize>().ok())
            .flatten()
    })
}

fn split_name(name: &str, separators: &[char]) -> Vec<String> {
    name.split(separators)
        .filter(|segment| !segment.is_empty())
        .map(str::to_owned)
        .collect()
}
