// SPDX-License-Identifier: MIT OR Apache-2.0
//! Assemble per-tensor roles into a layer-indexed Transformer topology.
//!
//! The output is plain data: sorted `Vec`s and `BTreeMap`s only, so two
//! runs over the same inventory produce identical structures, identical
//! [`fmt::Display`] text, and identical [`TransformerTopology::to_json`]
//! strings regardless of input order.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};

use crate::checkpoint::{CheckpointMetadata, TensorInfo};

use super::role::{
    ClassifyContext, RoleAssignment, RoleConfidence, RoleScope, TensorRole, classify_tensor,
    names_expert_tensor,
};

/// How a layer's attention projections are stored.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttentionKind {
    /// Separate Q, K, V, and output projections.
    Separate,
    /// One fused QKV projection plus an output projection.
    FusedQkv,
    /// Some attention projections are present but the set is incomplete
    /// (for example Q/K/V without an output projection).
    Partial,
    /// No attention projection found in the layer.
    Absent,
}

impl AttentionKind {
    /// Stable snake_case name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::FusedQkv => "fused_qkv",
            Self::Partial => "partial",
            Self::Absent => "absent",
        }
    }
}

impl fmt::Display for AttentionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a layer's feed-forward block is stored.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FeedForwardKind {
    /// Dense gate + up + down (SwiGLU-style).
    Gated,
    /// Dense fused gate/up + down.
    FusedGateUp,
    /// Dense up + down (GELU-style two-matrix MLP).
    Plain,
    /// Routed experts (any expert-scoped tensor or a router).
    Moe,
    /// Some dense projections are present but the set is incomplete.
    Partial,
    /// No feed-forward projection found in the layer.
    Absent,
}

impl FeedForwardKind {
    /// Stable snake_case name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gated => "gated",
            Self::FusedGateUp => "fused_gate_up",
            Self::Plain => "plain",
            Self::Moe => "moe",
            Self::Partial => "partial",
            Self::Absent => "absent",
        }
    }
}

impl fmt::Display for FeedForwardKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One Transformer layer as seen through tensor names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTopology {
    /// Layer index parsed from tensor names.
    pub index: usize,
    /// Attention storage pattern.
    pub attention: AttentionKind,
    /// Feed-forward storage pattern.
    pub feed_forward: FeedForwardKind,
    /// `true` when a [`TensorRole::MoeRouter`] tensor is present.
    pub has_router: bool,
    /// Routed expert count: distinct `experts.N` indices, or the leading
    /// dimension of a stacked `*_exps` tensor. `None` when no expert
    /// tensor is present or the stacked tensors disagree.
    pub expert_count: Option<usize>,
    /// `true` when a shared-expert tensor is present.
    pub has_shared_expert: bool,
    /// `true` when the layer also stores a dense MLP next to its experts.
    pub has_dense_mlp: bool,
    /// Every tensor of this layer, sorted by name.
    pub tensors: Vec<RoleAssignment>,
}

impl LayerTopology {
    /// Dense-scoped roles present in this layer, sorted.
    pub fn roles(&self) -> BTreeSet<TensorRole> {
        self.tensors
            .iter()
            .filter(|t| t.scope == RoleScope::Layer)
            .map(|t| t.role)
            .collect()
    }

    /// Roles present in any MoE scope of this layer, sorted.
    pub fn expert_roles(&self) -> BTreeSet<TensorRole> {
        self.tensors
            .iter()
            .filter(|t| t.scope.is_moe())
            .map(|t| t.role)
            .collect()
    }
}

/// Checkpoint-wide counts and hints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TopologySummary {
    /// Number of tensors inspected.
    pub tensor_count: usize,
    /// Number of layers discovered from names.
    pub layer_count: usize,
    /// Layer indices that are below the highest discovered (or declared)
    /// index but have no tensors.
    pub missing_layers: Vec<usize>,
    /// `true` when any layer is [`FeedForwardKind::Moe`].
    pub is_moe: bool,
    /// Hidden size read from the token-embedding shape, when rank 2.
    pub hidden_size_hint: Option<usize>,
    /// Vocabulary size read from the token-embedding shape, when rank 2.
    pub vocab_size_hint: Option<usize>,
    /// `true` when a token embedding is present but no output head is.
    pub tied_embeddings_hint: bool,
    /// Tensor count per role.
    pub role_counts: BTreeMap<TensorRole, usize>,
    /// Tensor count per confidence level.
    pub confidence_counts: BTreeMap<RoleConfidence, usize>,
    /// Names with [`RoleConfidence::Ambiguous`], sorted.
    pub ambiguous_tensors: Vec<String>,
    /// Names with [`RoleConfidence::Unknown`], sorted.
    pub unknown_tensors: Vec<String>,
}

/// Descriptive Transformer topology discovered from normalized metadata.
///
/// Built by [`super::discover_topology`] /
/// [`super::discover_topology_from_parts`]. Nothing here is executed or
/// validated numerically; `*_hint` fields and heuristic roles are
/// evidence, not claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerTopology {
    /// `general.architecture` (GGUF) when present; Safetensors has no
    /// equivalent and leaves this `None`.
    pub architecture_hint: Option<String>,
    /// `<arch>.block_count` (GGUF) when present.
    pub declared_layer_count: Option<usize>,
    /// `<arch>.expert_count` (GGUF) when present.
    pub declared_expert_count: Option<usize>,
    /// Tensors with no layer index, sorted by name.
    pub global: Vec<RoleAssignment>,
    /// Discovered layers, sorted by index.
    pub layers: Vec<LayerTopology>,
    /// Counts and hints.
    pub summary: TopologySummary,
}

impl TransformerTopology {
    /// Every classified tensor (global then per layer), sorted by name.
    pub fn tensors(&self) -> Vec<&RoleAssignment> {
        let mut all: Vec<&RoleAssignment> = self
            .global
            .iter()
            .chain(self.layers.iter().flat_map(|l| l.tensors.iter()))
            .collect();
        all.sort_by(|a, b| a.name.cmp(&b.name));
        all
    }

    /// Look up one tensor's assignment by exact name.
    pub fn tensor(&self, name: &str) -> Option<&RoleAssignment> {
        self.global
            .iter()
            .chain(self.layers.iter().flat_map(|l| l.tensors.iter()))
            .find(|t| t.name == name)
    }

    /// Look up a layer by index.
    pub fn layer(&self, index: usize) -> Option<&LayerTopology> {
        self.layers
            .binary_search_by_key(&index, |l| l.index)
            .ok()
            .map(|i| &self.layers[i])
    }

    /// Deterministic JSON rendering (no external dependency; keys are
    /// emitted in a fixed order, arrays in sorted order).
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        let mut w = JsonWriter::new(&mut out);
        w.begin_object();
        w.key("architecture_hint");
        w.opt_string(self.architecture_hint.as_deref());
        w.key("declared_layer_count");
        w.opt_usize(self.declared_layer_count);
        w.key("declared_expert_count");
        w.opt_usize(self.declared_expert_count);

        w.key("summary");
        w.begin_object();
        let s = &self.summary;
        w.key("tensor_count");
        w.usize(s.tensor_count);
        w.key("layer_count");
        w.usize(s.layer_count);
        w.key("missing_layers");
        w.usize_array(&s.missing_layers);
        w.key("is_moe");
        w.bool(s.is_moe);
        w.key("hidden_size_hint");
        w.opt_usize(s.hidden_size_hint);
        w.key("vocab_size_hint");
        w.opt_usize(s.vocab_size_hint);
        w.key("tied_embeddings_hint");
        w.bool(s.tied_embeddings_hint);
        w.key("role_counts");
        w.begin_object();
        for (role, n) in &s.role_counts {
            w.key(role.as_str());
            w.usize(*n);
        }
        w.end_object();
        w.key("confidence_counts");
        w.begin_object();
        for (c, n) in &s.confidence_counts {
            w.key(c.as_str());
            w.usize(*n);
        }
        w.end_object();
        w.key("ambiguous_tensors");
        w.string_array(s.ambiguous_tensors.iter().map(String::as_str));
        w.key("unknown_tensors");
        w.string_array(s.unknown_tensors.iter().map(String::as_str));
        w.end_object();

        w.key("global");
        w.begin_array();
        for t in &self.global {
            write_assignment(&mut w, t);
        }
        w.end_array();

        w.key("layers");
        w.begin_array();
        for layer in &self.layers {
            w.begin_object();
            w.key("index");
            w.usize(layer.index);
            w.key("attention");
            w.string(layer.attention.as_str());
            w.key("feed_forward");
            w.string(layer.feed_forward.as_str());
            w.key("has_router");
            w.bool(layer.has_router);
            w.key("expert_count");
            w.opt_usize(layer.expert_count);
            w.key("has_shared_expert");
            w.bool(layer.has_shared_expert);
            w.key("has_dense_mlp");
            w.bool(layer.has_dense_mlp);
            w.key("tensors");
            w.begin_array();
            for t in &layer.tensors {
                write_assignment(&mut w, t);
            }
            w.end_array();
            w.end_object();
        }
        w.end_array();
        w.end_object();
        out
    }
}

fn write_assignment(w: &mut JsonWriter<'_>, t: &RoleAssignment) {
    w.begin_object();
    w.key("name");
    w.string(&t.name);
    w.key("role");
    w.string(t.role.as_str());
    w.key("scope");
    w.string(&t.scope.to_string());
    w.key("param");
    w.string(t.param.as_str());
    w.key("confidence");
    w.string(t.confidence.as_str());
    w.key("alternatives");
    w.string_array(t.alternatives.iter().map(|r| r.as_str()));
    w.key("layer");
    w.opt_usize(t.layer);
    w.key("dims");
    w.usize_array(&t.dims);
    w.key("dtype");
    w.string(&t.dtype);
    w.key("evidence");
    w.string_array(t.evidence.iter().copied());
    w.end_object();
}

impl fmt::Display for TransformerTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.summary;
        writeln!(
            f,
            "architecture: {}",
            self.architecture_hint.as_deref().unwrap_or("(none)")
        )?;
        writeln!(
            f,
            "layers:       {} discovered{}{}",
            s.layer_count,
            opt_suffix(", declared ", self.declared_layer_count),
            if s.missing_layers.is_empty() {
                String::new()
            } else {
                format!(", missing {:?}", s.missing_layers)
            }
        )?;
        writeln!(
            f,
            "moe:          {}{}",
            if s.is_moe { "yes" } else { "no" },
            opt_suffix(", declared experts ", self.declared_expert_count)
        )?;
        writeln!(
            f,
            "hints:        hidden={} vocab={} tied_embeddings={}",
            opt_str(s.hidden_size_hint),
            opt_str(s.vocab_size_hint),
            s.tied_embeddings_hint
        )?;
        writeln!(
            f,
            "tensors:      {} (ambiguous {}, unknown {})",
            s.tensor_count,
            s.ambiguous_tensors.len(),
            s.unknown_tensors.len()
        )?;
        writeln!(f)?;
        writeln!(f, "global:")?;
        for t in &self.global {
            write_assignment_line(f, t)?;
        }
        for layer in &self.layers {
            writeln!(f)?;
            write!(
                f,
                "layer {}: attention={} feed_forward={}",
                layer.index, layer.attention, layer.feed_forward
            )?;
            if layer.has_router {
                f.write_str(" router")?;
            }
            if let Some(n) = layer.expert_count {
                write!(f, " experts={n}")?;
            }
            if layer.has_shared_expert {
                f.write_str(" shared_expert")?;
            }
            if layer.has_dense_mlp && layer.feed_forward == FeedForwardKind::Moe {
                f.write_str(" dense_mlp")?;
            }
            writeln!(f)?;
            for t in &layer.tensors {
                write_assignment_line(f, t)?;
            }
        }
        Ok(())
    }
}

fn write_assignment_line(f: &mut fmt::Formatter<'_>, t: &RoleAssignment) -> fmt::Result {
    write!(
        f,
        "  {:<56} {:<22} {:<16} {:<7} {:<10} {:?} {}",
        t.name, t.role, t.scope, t.param, t.confidence, t.dims, t.dtype
    )?;
    if !t.alternatives.is_empty() {
        let alts: Vec<&str> = t.alternatives.iter().map(|r| r.as_str()).collect();
        write!(f, " alternatives={}", alts.join("|"))?;
    }
    writeln!(f)
}

fn opt_suffix(prefix: &str, v: Option<usize>) -> String {
    v.map(|n| format!("{prefix}{n}")).unwrap_or_default()
}

fn opt_str(v: Option<usize>) -> String {
    v.map_or_else(|| "?".to_owned(), |n| n.to_string())
}

/// Build the topology from an inventory and metadata.
pub(super) fn assemble(
    tensors: &[TensorInfo],
    metadata: &CheckpointMetadata,
) -> TransformerTopology {
    let mut sorted: Vec<&TensorInfo> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    sorted.dedup_by(|a, b| a.name == b.name);

    let ctx = ClassifyContext {
        has_expert_tensors: sorted.iter().any(|t| names_expert_tensor(&t.name)),
    };

    let mut global = Vec::new();
    let mut by_layer: BTreeMap<usize, Vec<RoleAssignment>> = BTreeMap::new();
    for info in &sorted {
        let assignment = classify_tensor(info, ctx);
        match assignment.layer {
            Some(layer) => by_layer.entry(layer).or_default().push(assignment),
            None => global.push(assignment),
        }
    }

    let layers: Vec<LayerTopology> = by_layer
        .into_iter()
        .map(|(index, mut tensors)| {
            refine_layer_norms(&mut tensors);
            build_layer(index, tensors)
        })
        .collect();

    let architecture_hint = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let declared = |suffix: &str| {
        architecture_hint
            .as_deref()
            .and_then(|arch| metadata.get(&format!("{arch}.{suffix}")))
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
    };
    let declared_layer_count = declared("block_count");
    let declared_expert_count = declared("expert_count");

    let summary = summarize(&global, &layers, declared_layer_count);

    TransformerTopology {
        architecture_hint,
        declared_layer_count,
        declared_expert_count,
        global,
        layers,
        summary,
    }
}

/// Gemma-2 pairs `post_attention_layernorm` with `pre_feedforward_layernorm`;
/// in that layout the former is a true post-attention norm.
fn refine_layer_norms(tensors: &mut [RoleAssignment]) {
    let has_pre_ffn = tensors.iter().any(|t| {
        t.role == TensorRole::MlpNorm
            && t.confidence == RoleConfidence::Exact
            && t.scope == RoleScope::Layer
    });
    if !has_pre_ffn {
        return;
    }
    for t in tensors.iter_mut() {
        if t.evidence
            .contains(&"post_attention_layernorm_llama_convention")
        {
            t.role = TensorRole::PostAttentionNorm;
            t.confidence = RoleConfidence::Heuristic;
            t.evidence.push("resolved_by_sibling_mlp_norm");
        }
    }
}

fn build_layer(index: usize, tensors: Vec<RoleAssignment>) -> LayerTopology {
    use TensorRole::*;

    let dense: BTreeSet<TensorRole> = tensors
        .iter()
        .filter(|t| t.scope == RoleScope::Layer)
        .map(|t| t.role)
        .collect();
    let moe_tensors: Vec<&RoleAssignment> = tensors.iter().filter(|t| t.scope.is_moe()).collect();

    let has_qkv = dense.contains(&AttentionQkv);
    let q = dense.contains(&AttentionQuery);
    let k = dense.contains(&AttentionKey);
    let v = dense.contains(&AttentionValue);
    let o = dense.contains(&AttentionOutput);
    let attention = if has_qkv && o {
        AttentionKind::FusedQkv
    } else if q && k && v && o {
        AttentionKind::Separate
    } else if has_qkv || q || k || v || o {
        AttentionKind::Partial
    } else {
        AttentionKind::Absent
    };

    let gate = dense.contains(&MlpGate);
    let up = dense.contains(&MlpUp);
    let down = dense.contains(&MlpDown);
    let fused = dense.contains(&MlpGateUp);
    let has_router = tensors.iter().any(|t| t.role == MoeRouter);
    let has_dense_mlp = gate || up || down || fused;
    let dense_kind = if fused && down {
        FeedForwardKind::FusedGateUp
    } else if gate && up && down {
        FeedForwardKind::Gated
    } else if up && down {
        FeedForwardKind::Plain
    } else if has_dense_mlp {
        FeedForwardKind::Partial
    } else {
        FeedForwardKind::Absent
    };
    let feed_forward = if !moe_tensors.is_empty() || has_router {
        FeedForwardKind::Moe
    } else {
        dense_kind
    };

    let has_shared_expert = moe_tensors
        .iter()
        .any(|t| t.scope == RoleScope::SharedExpert);
    let expert_count = expert_count(&moe_tensors);

    LayerTopology {
        index,
        attention,
        feed_forward,
        has_router,
        expert_count,
        has_shared_expert,
        has_dense_mlp,
        tensors,
    }
}

fn expert_count(moe_tensors: &[&RoleAssignment]) -> Option<usize> {
    let indexed: BTreeSet<usize> = moe_tensors
        .iter()
        .filter_map(|t| match t.scope {
            RoleScope::Expert(i) => Some(i),
            _ => None,
        })
        .collect();
    if !indexed.is_empty() {
        return Some(indexed.len());
    }
    let stacked: BTreeSet<usize> = moe_tensors
        .iter()
        .filter(|t| t.scope == RoleScope::StackedExperts && t.dims.len() == 3)
        .map(|t| t.dims[0])
        .collect();
    (stacked.len() == 1).then(|| stacked.into_iter().next().unwrap_or_default())
}

fn summarize(
    global: &[RoleAssignment],
    layers: &[LayerTopology],
    declared_layer_count: Option<usize>,
) -> TopologySummary {
    let all = global
        .iter()
        .chain(layers.iter().flat_map(|l| l.tensors.iter()));

    let mut summary = TopologySummary {
        layer_count: layers.len(),
        is_moe: layers
            .iter()
            .any(|l| l.feed_forward == FeedForwardKind::Moe),
        ..TopologySummary::default()
    };
    for t in all {
        summary.tensor_count += 1;
        *summary.role_counts.entry(t.role).or_default() += 1;
        *summary.confidence_counts.entry(t.confidence).or_default() += 1;
        match t.confidence {
            RoleConfidence::Ambiguous => summary.ambiguous_tensors.push(t.name.clone()),
            RoleConfidence::Unknown => summary.unknown_tensors.push(t.name.clone()),
            _ => {}
        }
    }
    summary.ambiguous_tensors.sort();
    summary.unknown_tensors.sort();

    let embedding = global.iter().find(|t| {
        t.role == TensorRole::TokenEmbedding && t.param == super::role::ParamKind::Weight
    });
    if let Some(e) = embedding {
        if e.dims.len() == 2 {
            summary.vocab_size_hint = Some(e.dims[0]);
            summary.hidden_size_hint = Some(e.dims[1]);
        }
        summary.tied_embeddings_hint = !global.iter().any(|t| t.role == TensorRole::OutputHead);
    }

    let present: BTreeSet<usize> = layers.iter().map(|l| l.index).collect();
    let expected_end = match (declared_layer_count, present.iter().next_back()) {
        (Some(n), Some(&max)) => n.max(max + 1),
        (Some(n), None) => n,
        (None, Some(&max)) => max + 1,
        (None, None) => 0,
    };
    summary.missing_layers = (0..expected_end).filter(|i| !present.contains(i)).collect();
    summary
}

/// Minimal JSON emitter: objects, arrays, strings, integers, booleans, null.
struct JsonWriter<'a> {
    out: &'a mut String,
    /// Whether the current container already has one item (per depth).
    first: Vec<bool>,
}

impl<'a> JsonWriter<'a> {
    fn new(out: &'a mut String) -> Self {
        Self {
            out,
            first: Vec::new(),
        }
    }

    fn separator(&mut self) {
        if let Some(first) = self.first.last_mut() {
            if *first {
                *first = false;
            } else {
                self.out.push(',');
            }
        }
    }

    fn begin_object(&mut self) {
        self.separator();
        self.out.push('{');
        self.first.push(true);
    }

    fn end_object(&mut self) {
        self.first.pop();
        self.out.push('}');
    }

    fn begin_array(&mut self) {
        self.separator();
        self.out.push('[');
        self.first.push(true);
    }

    fn end_array(&mut self) {
        self.first.pop();
        self.out.push(']');
    }

    fn key(&mut self, key: &str) {
        self.separator();
        escape_into(self.out, key);
        self.out.push(':');
        // The value that follows must not emit a separator.
        if let Some(first) = self.first.last_mut() {
            *first = true;
        }
    }

    fn string(&mut self, s: &str) {
        self.separator();
        escape_into(self.out, s);
    }

    fn opt_string(&mut self, s: Option<&str>) {
        match s {
            Some(s) => self.string(s),
            None => {
                self.separator();
                self.out.push_str("null");
            }
        }
    }

    fn usize(&mut self, v: usize) {
        self.separator();
        let _ = write!(self.out, "{v}");
    }

    fn opt_usize(&mut self, v: Option<usize>) {
        match v {
            Some(v) => self.usize(v),
            None => {
                self.separator();
                self.out.push_str("null");
            }
        }
    }

    fn bool(&mut self, v: bool) {
        self.separator();
        self.out.push_str(if v { "true" } else { "false" });
    }

    fn usize_array(&mut self, values: &[usize]) {
        self.begin_array();
        for &v in values {
            self.usize(v);
        }
        self.end_array();
    }

    fn string_array<'s>(&mut self, values: impl Iterator<Item = &'s str>) {
        self.begin_array();
        for v in values {
            self.string(v);
        }
        self.end_array();
    }
}

fn escape_into(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_writer_handles_nesting_and_escapes() {
        let mut out = String::new();
        let mut w = JsonWriter::new(&mut out);
        w.begin_object();
        w.key("a");
        w.usize(1);
        w.key("b");
        w.begin_array();
        w.string("x\"y");
        w.bool(false);
        w.begin_object();
        w.end_object();
        w.end_array();
        w.key("c");
        w.opt_string(None);
        w.end_object();
        assert_eq!(out, r#"{"a":1,"b":["x\"y",false,{}],"c":null}"#);
    }
}
