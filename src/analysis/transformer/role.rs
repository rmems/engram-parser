// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-tensor role classification from normalized tensor names and shapes.
//!
//! Everything here is descriptive: a [`TensorRole`] says what a tensor
//! *looks like* under common Transformer naming conventions (llama.cpp
//! GGUF, Hugging Face Llama / Mixtral / Qwen-MoE, GPT-2, GPT-NeoX, the
//! original Llama release, BERT). Nothing is executed, and nothing is
//! inferred from payload bytes.
//!
//! A name that matches no convention becomes [`TensorRole::Unknown`]. A
//! name whose meaning depends on the model family is reported with
//! [`RoleConfidence::Ambiguous`] plus the candidate roles, never
//! silently resolved.

use std::fmt;

use crate::checkpoint::TensorInfo;

/// Functional role of a tensor in a Transformer.
///
/// MoE expert projections reuse the `Mlp*` roles; [`RoleScope`] says
/// whether the tensor is a dense block, one routed expert, a stacked
/// expert tensor, or a shared expert.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorRole {
    /// Input token embedding table.
    TokenEmbedding,
    /// Learned absolute position embedding table.
    PositionEmbedding,
    /// Final projection to vocabulary logits (`lm_head`, `output`).
    OutputHead,
    /// Final normalization before the output head.
    OutputNorm,
    /// Normalization applied before attention.
    AttentionNorm,
    /// Normalization applied to the attention output (Gemma-2 style).
    PostAttentionNorm,
    /// Query projection.
    AttentionQuery,
    /// Key projection.
    AttentionKey,
    /// Value projection.
    AttentionValue,
    /// Fused query/key/value projection.
    AttentionQkv,
    /// Attention output projection.
    AttentionOutput,
    /// Per-head query normalization.
    AttentionQueryNorm,
    /// Per-head key normalization.
    AttentionKeyNorm,
    /// Normalization applied before the feed-forward block.
    MlpNorm,
    /// Normalization applied to the feed-forward output (Gemma-2 style).
    PostMlpNorm,
    /// Feed-forward gate projection (SwiGLU-style).
    MlpGate,
    /// Feed-forward up projection.
    MlpUp,
    /// Feed-forward down projection.
    MlpDown,
    /// Fused gate+up projection.
    MlpGateUp,
    /// MoE router / expert-selection gate.
    MoeRouter,
    /// Rotary position frequency table.
    RotaryFrequencies,
    /// No convention matched (or only ambiguous ones; see
    /// [`RoleAssignment::alternatives`]).
    Unknown,
}

impl TensorRole {
    /// Stable snake_case name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::PositionEmbedding => "position_embedding",
            Self::OutputHead => "output_head",
            Self::OutputNorm => "output_norm",
            Self::AttentionNorm => "attention_norm",
            Self::PostAttentionNorm => "post_attention_norm",
            Self::AttentionQuery => "attention_query",
            Self::AttentionKey => "attention_key",
            Self::AttentionValue => "attention_value",
            Self::AttentionQkv => "attention_qkv",
            Self::AttentionOutput => "attention_output",
            Self::AttentionQueryNorm => "attention_query_norm",
            Self::AttentionKeyNorm => "attention_key_norm",
            Self::MlpNorm => "mlp_norm",
            Self::PostMlpNorm => "post_mlp_norm",
            Self::MlpGate => "mlp_gate",
            Self::MlpUp => "mlp_up",
            Self::MlpDown => "mlp_down",
            Self::MlpGateUp => "mlp_gate_up",
            Self::MoeRouter => "moe_router",
            Self::RotaryFrequencies => "rotary_frequencies",
            Self::Unknown => "unknown",
        }
    }

    /// `true` for the attention-side roles (norms included).
    pub const fn is_attention(self) -> bool {
        matches!(
            self,
            Self::AttentionNorm
                | Self::PostAttentionNorm
                | Self::AttentionQuery
                | Self::AttentionKey
                | Self::AttentionValue
                | Self::AttentionQkv
                | Self::AttentionOutput
                | Self::AttentionQueryNorm
                | Self::AttentionKeyNorm
        )
    }

    /// `true` for the feed-forward-side roles (norms and router included).
    pub const fn is_feed_forward(self) -> bool {
        matches!(
            self,
            Self::MlpNorm
                | Self::PostMlpNorm
                | Self::MlpGate
                | Self::MlpUp
                | Self::MlpDown
                | Self::MlpGateUp
                | Self::MoeRouter
        )
    }
}

impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which structural unit a tensor belongs to.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RoleScope {
    /// Not tied to a layer (embeddings, output head, final norm, …).
    Global,
    /// Dense part of one Transformer layer.
    Layer,
    /// One routed MoE expert stored as its own tensor.
    Expert(usize),
    /// All routed experts stacked into one tensor (llama.cpp `*_exps`).
    StackedExperts,
    /// An always-on shared expert (Qwen-MoE / DeepSeek style).
    SharedExpert,
}

impl RoleScope {
    /// `true` for every MoE-specific scope.
    pub const fn is_moe(self) -> bool {
        matches!(
            self,
            Self::Expert(_) | Self::StackedExperts | Self::SharedExpert
        )
    }
}

impl fmt::Display for RoleScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => f.write_str("global"),
            Self::Layer => f.write_str("layer"),
            Self::Expert(i) => write!(f, "expert[{i}]"),
            Self::StackedExperts => f.write_str("experts[stacked]"),
            Self::SharedExpert => f.write_str("shared_expert"),
        }
    }
}

/// Parameter kind within a role (`.weight`, `.bias`, other suffix).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ParamKind {
    /// Main weight tensor (or a name with no recognized suffix).
    Weight,
    /// Bias vector.
    Bias,
    /// Quantization sidecar or other auxiliary tensor (`scales`, `qzeros`, …).
    Other,
}

impl ParamKind {
    /// Stable snake_case name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Weight => "weight",
            Self::Bias => "bias",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for ParamKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How much the classifier trusts a role.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RoleConfidence {
    /// The name is a direct, family-independent convention match.
    Exact,
    /// The mapping depends on a convention assumption (for example the
    /// `w1`/`w3`/`w2` = gate/up/down ordering of the original Llama release).
    Heuristic,
    /// Several roles are plausible and the checkpoint gives no evidence to
    /// pick one. `role` is [`TensorRole::Unknown`]; see `alternatives`.
    Ambiguous,
    /// No convention matched.
    Unknown,
}

impl RoleConfidence {
    /// Stable snake_case name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Heuristic => "heuristic",
            Self::Ambiguous => "ambiguous",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for RoleConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classification of one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleAssignment {
    /// Exact tensor name.
    pub name: String,
    /// Assigned role ([`TensorRole::Unknown`] when none or ambiguous).
    pub role: TensorRole,
    /// Structural scope.
    pub scope: RoleScope,
    /// Weight / bias / other.
    pub param: ParamKind,
    /// Trust level.
    pub confidence: RoleConfidence,
    /// Candidate roles when `confidence` is [`RoleConfidence::Ambiguous`];
    /// sorted, otherwise empty.
    pub alternatives: Vec<TensorRole>,
    /// Layer index parsed from the name, if any.
    pub layer: Option<usize>,
    /// Stable, human-readable reasons behind the assignment.
    pub evidence: Vec<&'static str>,
    /// Outermost-first dims, copied from the inventory.
    pub dims: Vec<usize>,
    /// Normalized dtype label, copied from the inventory.
    pub dtype: String,
}

/// Checkpoint-wide facts the per-tensor classifier may consult.
///
/// Built by [`super::discover_topology`]; tests can construct it directly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassifyContext {
    /// Some tensor in the checkpoint is an expert tensor (`experts.N.*`,
    /// `*_exps`, `shared_expert*`). Lets a bare `mlp.gate.weight` be read
    /// as a router instead of staying ambiguous.
    pub has_expert_tensors: bool,
}

/// `true` when the name alone marks an MoE expert tensor.
pub(super) fn names_expert_tensor(name: &str) -> bool {
    let parts = NameParts::new(name);
    parts.in_experts || parts.in_shared_expert
}

/// Classify one tensor by name and shape.
///
/// Pure and deterministic. Uses `ctx` only to break the documented
/// `gate` ambiguity; a default context never claims MoE.
pub fn classify_tensor(info: &TensorInfo, ctx: ClassifyContext) -> RoleAssignment {
    let parts = NameParts::new(&info.name);
    let dims = info.shape.dims().to_vec();
    let mut assignment = RoleAssignment {
        name: info.name.clone(),
        role: TensorRole::Unknown,
        scope: parts.scope(),
        param: parts.param,
        confidence: RoleConfidence::Unknown,
        alternatives: Vec::new(),
        layer: parts.layer,
        evidence: Vec::new(),
        dims,
        dtype: info.dtype.label().to_owned(),
    };

    match match_component(&parts, ctx, &assignment.dims) {
        Outcome::Role(role, confidence, reason) => {
            assignment.role = role;
            assignment.confidence = confidence;
            assignment.evidence.push(reason);
        }
        Outcome::Ambiguous(mut candidates, reason) => {
            candidates.sort();
            candidates.dedup();
            assignment.confidence = RoleConfidence::Ambiguous;
            assignment.alternatives = candidates;
            assignment.evidence.push(reason);
        }
        Outcome::Unknown => {
            assignment.evidence.push("no_convention_match");
        }
    }

    if parts.in_attention {
        assignment.evidence.push("attention_context");
    }
    if parts.in_feed_forward {
        assignment.evidence.push("feed_forward_context");
    }
    assignment
}

/// Marker segments that identify a layer index (`blk.3`, `layers.3`, `h.3`).
const LAYER_MARKERS: &[&str] = &["blk", "layers", "layer", "blocks", "block", "h"];
/// Marker segments that identify a per-expert index.
const EXPERT_MARKERS: &[&str] = &["experts", "expert"];
const ATTENTION_CONTEXT: &[&str] = &["self_attn", "attention", "attn", "self_attention"];
const FEED_FORWARD_CONTEXT: &[&str] = &[
    "mlp",
    "feed_forward",
    "ffn",
    "block_sparse_moe",
    "moe",
    "experts",
    "expert",
    "shared_expert",
    "shared_experts",
    "intermediate",
];
const PARAM_SUFFIXES: &[(&str, ParamKind)] = &[
    ("weight", ParamKind::Weight),
    ("bias", ParamKind::Bias),
    ("scale", ParamKind::Other),
    ("scales", ParamKind::Other),
    ("qweight", ParamKind::Other),
    ("qzeros", ParamKind::Other),
    ("g_idx", ParamKind::Other),
    ("weight_scale", ParamKind::Other),
    ("input_scale", ParamKind::Other),
    ("zeros", ParamKind::Other),
];

#[derive(Debug)]
struct NameParts {
    /// Lowercased `.`/`/`-separated segments without the parameter suffix.
    segments: Vec<String>,
    /// Last segment of `segments`: the component name (`attn_q`, `q_proj`).
    component: String,
    param: ParamKind,
    layer: Option<usize>,
    expert_index: Option<usize>,
    in_attention: bool,
    in_feed_forward: bool,
    in_experts: bool,
    in_shared_expert: bool,
}

impl NameParts {
    fn new(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        let mut segments: Vec<String> = lower
            .split(['.', '/'])
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();

        let mut param = ParamKind::Weight;
        if segments.len() > 1 {
            let last = segments.last().map(String::as_str).unwrap_or_default();
            if let Some((_, kind)) = PARAM_SUFFIXES.iter().find(|(s, _)| *s == last) {
                param = *kind;
                segments.pop();
            }
        }
        let component = segments.last().cloned().unwrap_or_default();

        let layer = number_after(&segments, LAYER_MARKERS);
        let expert_index = number_after(&segments, EXPERT_MARKERS);
        let in_attention = segments
            .iter()
            .any(|s| ATTENTION_CONTEXT.contains(&s.as_str()));
        let in_shared_expert = segments.iter().any(|s| s.starts_with("shared_expert"))
            || component.ends_with("_shexp");
        let in_experts = segments
            .iter()
            .any(|s| EXPERT_MARKERS.contains(&s.as_str()))
            || component.ends_with("_exps");
        let in_feed_forward = segments
            .iter()
            .any(|s| FEED_FORWARD_CONTEXT.contains(&s.as_str()))
            || component.starts_with("ffn_")
            || in_experts
            || in_shared_expert;

        Self {
            segments,
            component,
            param,
            layer,
            expert_index,
            in_attention,
            in_feed_forward,
            in_experts,
            in_shared_expert,
        }
    }

    fn scope(&self) -> RoleScope {
        if self.in_shared_expert {
            RoleScope::SharedExpert
        } else if self.component.ends_with("_exps") {
            RoleScope::StackedExperts
        } else if let Some(i) = self.expert_index {
            RoleScope::Expert(i)
        } else if self.layer.is_some() {
            RoleScope::Layer
        } else {
            RoleScope::Global
        }
    }

    /// Segment immediately before the component, if any.
    fn parent(&self) -> &str {
        self.segments
            .len()
            .checked_sub(2)
            .map(|i| self.segments[i].as_str())
            .unwrap_or_default()
    }

    fn has_segment(&self, needle: &str) -> bool {
        self.segments.iter().any(|s| s == needle)
    }
}

fn number_after(segments: &[String], markers: &[&str]) -> Option<usize> {
    segments.windows(2).find_map(|w| {
        markers
            .contains(&w[0].as_str())
            .then(|| w[1].parse::<usize>().ok())
            .flatten()
    })
}

enum Outcome {
    Role(TensorRole, RoleConfidence, &'static str),
    Ambiguous(Vec<TensorRole>, &'static str),
    Unknown,
}

fn exact(role: TensorRole, reason: &'static str) -> Outcome {
    Outcome::Role(role, RoleConfidence::Exact, reason)
}

fn heuristic(role: TensorRole, reason: &'static str) -> Outcome {
    Outcome::Role(role, RoleConfidence::Heuristic, reason)
}

/// Router-like 2-D shape: a small expert axis against a large hidden axis.
fn router_shape_hint(dims: &[usize]) -> bool {
    dims.len() == 2
        && dims
            .iter()
            .copied()
            .min()
            .is_some_and(|v| (2..=512).contains(&v))
        && dims.iter().copied().max().is_some_and(|v| v >= 512)
}

fn match_component(parts: &NameParts, ctx: ClassifyContext, dims: &[usize]) -> Outcome {
    use TensorRole::*;

    let c = parts.component.as_str();

    // --- Global tensors -------------------------------------------------
    if parts.layer.is_none() {
        return match c {
            "token_embd" | "embed_tokens" | "tok_embeddings" | "wte" | "embed_in"
            | "word_embeddings" | "embedding" | "embeddings" | "word_embedding" => {
                exact(TokenEmbedding, "token_embedding_name")
            }
            "position_embd"
            | "wpe"
            | "embed_positions"
            | "position_embeddings"
            | "pos_embd"
            | "pos_embedding" => exact(PositionEmbedding, "position_embedding_name"),
            "output" | "lm_head" | "embed_out" => exact(OutputHead, "output_head_name"),
            "output_norm" | "ln_f" | "final_layer_norm" | "final_layernorm" | "final_norm"
            | "norm" => exact(OutputNorm, "output_norm_name"),
            "rope_freqs" | "inv_freq" => exact(RotaryFrequencies, "rotary_frequencies_name"),
            _ => Outcome::Unknown,
        };
    }

    // --- Layer-scoped: rotary tables stored per layer -------------------
    if c == "inv_freq" || c == "rope_freqs" {
        return exact(RotaryFrequencies, "rotary_frequencies_name");
    }

    // --- Norms ----------------------------------------------------------
    match c {
        "attn_norm"
        | "input_layernorm"
        | "ln_1"
        | "attention_norm"
        | "ln_attn"
        | "pre_attention_layernorm"
        | "attention_layernorm" => {
            return exact(AttentionNorm, "attention_norm_name");
        }
        "post_attention_norm" | "post_attn_norm" => {
            return exact(PostAttentionNorm, "post_attention_norm_name");
        }
        "post_attention_layernorm" => {
            // Llama: the pre-MLP norm. Gemma-2: a true post-attention norm,
            // paired with `pre_feedforward_layernorm`. The topology pass
            // resolves this per layer; alone it stays a heuristic.
            return heuristic(MlpNorm, "post_attention_layernorm_llama_convention");
        }
        "ffn_norm"
        | "ln_2"
        | "pre_feedforward_layernorm"
        | "ln_mlp"
        | "mlp_norm"
        | "pre_mlp_layernorm" => return exact(MlpNorm, "mlp_norm_name"),
        "post_ffw_norm" | "post_feedforward_layernorm" | "post_mlp_norm" => {
            return exact(PostMlpNorm, "post_mlp_norm_name");
        }
        "attn_q_norm" | "q_norm" | "query_norm" | "q_layernorm" => {
            return exact(AttentionQueryNorm, "attention_query_norm_name");
        }
        "attn_k_norm" | "k_norm" | "key_norm" | "k_layernorm" => {
            return exact(AttentionKeyNorm, "attention_key_norm_name");
        }
        _ => {}
    }

    // --- Attention projections ----------------------------------------
    match c {
        "attn_q" | "q_proj" | "wq" | "query" => return exact(AttentionQuery, "query_name"),
        "attn_k" | "k_proj" | "wk" | "key" => return exact(AttentionKey, "key_name"),
        "attn_v" | "v_proj" | "wv" | "value" => return exact(AttentionValue, "value_name"),
        "attn_qkv" | "c_attn" | "query_key_value" | "qkv_proj" | "wqkv" | "qkv" | "in_proj" => {
            return exact(AttentionQkv, "fused_qkv_name");
        }
        "attn_output" | "o_proj" | "wo" | "out_proj" => {
            return exact(AttentionOutput, "attention_output_name");
        }
        _ => {}
    }
    if parts.in_attention && !parts.in_feed_forward {
        match c {
            "q" => return exact(AttentionQuery, "query_name"),
            "k" => return exact(AttentionKey, "key_name"),
            "v" => return exact(AttentionValue, "value_name"),
            "dense" | "c_proj" | "proj" | "o" => {
                return exact(AttentionOutput, "attention_output_name");
            }
            _ => {}
        }
    }

    // --- MoE router -----------------------------------------------------
    match c {
        "ffn_gate_inp" | "router" | "gate_inp" => return exact(MoeRouter, "router_name"),
        "shared_expert_gate" => return exact(MoeRouter, "shared_expert_gate_name"),
        "gate" => {
            let parent = parts.parent();
            if parent == "block_sparse_moe" || parent == "moe" || parent == "router" {
                return exact(MoeRouter, "moe_parent_gate_name");
            }
            if parts.in_experts {
                // `experts.N.gate`: a per-expert gate projection.
                return heuristic(MlpGate, "expert_scoped_gate_name");
            }
            if ctx.has_expert_tensors {
                return heuristic(MoeRouter, "gate_name_with_expert_tensors_present");
            }
            if router_shape_hint(dims) {
                return heuristic(MoeRouter, "gate_name_with_router_shape");
            }
            return Outcome::Ambiguous(vec![MoeRouter, MlpGate], "bare_gate_name");
        }
        _ => {}
    }

    // --- Feed-forward projections --------------------------------------
    match c {
        "ffn_gate" | "ffn_gate_exps" | "ffn_gate_shexp" | "gate_proj" => {
            return exact(MlpGate, "gate_proj_name");
        }
        "ffn_up" | "ffn_up_exps" | "ffn_up_shexp" | "up_proj" | "c_fc" | "dense_h_to_4h"
        | "fc1" | "fc_in" | "wi" | "linear_1" => return exact(MlpUp, "up_proj_name"),
        "ffn_down" | "ffn_down_exps" | "ffn_down_shexp" | "down_proj" | "dense_4h_to_h" | "fc2"
        | "fc_out" | "linear_2" => return exact(MlpDown, "down_proj_name"),
        "gate_up_proj" | "ffn_gate_up" | "ffn_up_gate" | "w13" | "fused_gate_up" => {
            return exact(MlpGateUp, "fused_gate_up_name");
        }
        "w1" => return heuristic(MlpGate, "w1_llama_convention_gate"),
        "w3" => return heuristic(MlpUp, "w3_llama_convention_up"),
        "w2" => return heuristic(MlpDown, "w2_llama_convention_down"),
        _ => {}
    }
    if parts.in_feed_forward && !parts.in_attention {
        match c {
            "c_proj" => return exact(MlpDown, "down_proj_name"),
            "dense" if parts.parent() == "intermediate" => {
                return heuristic(MlpUp, "bert_intermediate_dense");
            }
            "dense" if parts.has_segment("output") => {
                return heuristic(MlpDown, "bert_output_dense");
            }
            _ => {}
        }
    }
    // BERT-style `layer.N.output.dense` has no mlp marker at all.
    if c == "dense" && parts.parent() == "output" && !parts.in_attention {
        return heuristic(MlpDown, "bert_output_dense");
    }

    Outcome::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{TensorDType, TensorLocation, TensorShape};

    fn info(name: &str, dims: &[usize]) -> TensorInfo {
        TensorInfo::new(
            name,
            TensorDType::F32,
            TensorShape::outermost_first(dims.to_vec()),
            dims.iter().product::<usize>() * 4,
            TensorLocation::new("model", 0, None),
        )
    }

    fn role_of(name: &str) -> (TensorRole, RoleScope, RoleConfidence) {
        let a = classify_tensor(&info(name, &[8, 8]), ClassifyContext::default());
        (a.role, a.scope, a.confidence)
    }

    #[test]
    fn gguf_dense_names() {
        use RoleConfidence::Exact;
        use RoleScope::*;
        use TensorRole::*;
        assert_eq!(
            role_of("token_embd.weight"),
            (TokenEmbedding, Global, Exact)
        );
        assert_eq!(role_of("output.weight"), (OutputHead, Global, Exact));
        assert_eq!(role_of("output_norm.weight"), (OutputNorm, Global, Exact));
        assert_eq!(
            role_of("blk.0.attn_q.weight"),
            (AttentionQuery, Layer, Exact)
        );
        assert_eq!(
            role_of("blk.0.attn_output.weight"),
            (AttentionOutput, Layer, Exact)
        );
        assert_eq!(role_of("blk.0.ffn_gate.weight"), (MlpGate, Layer, Exact));
        assert_eq!(role_of("blk.0.ffn_norm.weight"), (MlpNorm, Layer, Exact));
        assert_eq!(
            role_of("rope_freqs.weight"),
            (RotaryFrequencies, Global, Exact)
        );
    }

    #[test]
    fn moe_scopes() {
        use RoleScope::*;
        use TensorRole::*;
        assert_eq!(role_of("blk.0.ffn_gate_exps.weight").0, MlpGate);
        assert_eq!(role_of("blk.0.ffn_gate_exps.weight").1, StackedExperts);
        assert_eq!(role_of("blk.0.ffn_gate_inp.weight").0, MoeRouter);
        assert_eq!(
            role_of("blk.0.ffn_up_shexp.weight"),
            (MlpUp, SharedExpert, RoleConfidence::Exact)
        );
        assert_eq!(
            role_of("model.layers.2.block_sparse_moe.experts.5.w2.weight"),
            (MlpDown, Expert(5), RoleConfidence::Heuristic)
        );
        assert_eq!(
            role_of("model.layers.2.block_sparse_moe.gate.weight").0,
            MoeRouter
        );
        assert_eq!(
            role_of("model.layers.0.mlp.shared_expert.down_proj.weight"),
            (MlpDown, SharedExpert, RoleConfidence::Exact)
        );
        assert!(names_expert_tensor("blk.0.ffn_gate_exps.weight"));
        assert!(names_expert_tensor(
            "model.layers.0.mlp.experts.3.up_proj.weight"
        ));
        assert!(!names_expert_tensor("model.layers.0.mlp.up_proj.weight"));
    }

    #[test]
    fn bare_gate_is_ambiguous_without_evidence() {
        let a = classify_tensor(
            &info("model.layers.0.mlp.gate.weight", &[8, 8]),
            ClassifyContext::default(),
        );
        assert_eq!(a.role, TensorRole::Unknown);
        assert_eq!(a.confidence, RoleConfidence::Ambiguous);
        assert_eq!(
            a.alternatives,
            vec![TensorRole::MlpGate, TensorRole::MoeRouter]
        );

        let with_experts = classify_tensor(
            &info("model.layers.0.mlp.gate.weight", &[8, 8]),
            ClassifyContext {
                has_expert_tensors: true,
            },
        );
        assert_eq!(with_experts.role, TensorRole::MoeRouter);
        assert_eq!(with_experts.confidence, RoleConfidence::Heuristic);

        let by_shape = classify_tensor(
            &info("model.layers.0.mlp.gate.weight", &[8, 4096]),
            ClassifyContext::default(),
        );
        assert_eq!(by_shape.role, TensorRole::MoeRouter);
    }

    #[test]
    fn gpt2_and_neox_names() {
        use TensorRole::*;
        assert_eq!(role_of("wte.weight").0, TokenEmbedding);
        assert_eq!(role_of("wpe.weight").0, PositionEmbedding);
        assert_eq!(role_of("h.0.ln_1.weight").0, AttentionNorm);
        assert_eq!(role_of("h.0.attn.c_attn.weight").0, AttentionQkv);
        assert_eq!(role_of("h.0.attn.c_proj.weight").0, AttentionOutput);
        assert_eq!(role_of("h.0.mlp.c_fc.weight").0, MlpUp);
        assert_eq!(role_of("h.0.mlp.c_proj.weight").0, MlpDown);
        assert_eq!(role_of("ln_f.weight").0, OutputNorm);
        assert_eq!(
            role_of("gpt_neox.layers.3.attention.query_key_value.weight").0,
            AttentionQkv
        );
        assert_eq!(
            role_of("gpt_neox.layers.3.attention.dense.weight").0,
            AttentionOutput
        );
        assert_eq!(
            role_of("gpt_neox.layers.3.mlp.dense_4h_to_h.weight").0,
            MlpDown
        );
        assert_eq!(role_of("embed_out.weight").0, OutputHead);
    }

    #[test]
    fn bert_and_original_llama_names() {
        use TensorRole::*;
        assert_eq!(
            role_of("bert.encoder.layer.1.attention.self.query.weight").0,
            AttentionQuery
        );
        assert_eq!(
            role_of("bert.encoder.layer.1.attention.output.dense.weight").0,
            AttentionOutput
        );
        assert_eq!(
            role_of("bert.encoder.layer.1.intermediate.dense.weight").0,
            MlpUp
        );
        assert_eq!(
            role_of("bert.encoder.layer.1.output.dense.weight").0,
            MlpDown
        );
        assert_eq!(
            role_of("bert.embeddings.word_embeddings.weight").0,
            TokenEmbedding
        );
        assert_eq!(role_of("layers.0.attention.wq.weight").0, AttentionQuery);
        assert_eq!(role_of("layers.0.feed_forward.w3.weight").0, MlpUp);
        assert_eq!(role_of("tok_embeddings.weight").0, TokenEmbedding);
        assert_eq!(role_of("norm.weight").0, OutputNorm);
    }

    #[test]
    fn unknown_names_stay_unknown() {
        let a = classify_tensor(
            &info("blk.0.mystery.weight", &[4]),
            ClassifyContext::default(),
        );
        assert_eq!(a.role, TensorRole::Unknown);
        assert_eq!(a.confidence, RoleConfidence::Unknown);
        assert!(a.alternatives.is_empty());
        assert_eq!(a.layer, Some(0));
        assert_eq!(a.scope, RoleScope::Layer);
        let b = classify_tensor(
            &info("vision_tower.patch_embed", &[4]),
            ClassifyContext::default(),
        );
        assert_eq!(b.role, TensorRole::Unknown);
        assert_eq!(b.scope, RoleScope::Global);
    }

    #[test]
    fn param_kinds() {
        let a = classify_tensor(&info("blk.0.attn_q.bias", &[4]), ClassifyContext::default());
        assert_eq!(
            (a.role, a.param),
            (TensorRole::AttentionQuery, ParamKind::Bias)
        );
        let b = classify_tensor(
            &info("model.layers.0.self_attn.q_proj.qzeros", &[4]),
            ClassifyContext::default(),
        );
        assert_eq!(
            (b.role, b.param),
            (TensorRole::AttentionQuery, ParamKind::Other)
        );
        // A one-segment name is never stripped to nothing.
        let c = classify_tensor(&info("weight", &[4]), ClassifyContext::default());
        assert_eq!(c.role, TensorRole::Unknown);
    }
}
