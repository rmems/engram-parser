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
        f.pad(self.as_str())
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
            Self::Global => f.pad("global"),
            Self::Layer => f.pad("layer"),
            Self::Expert(i) => f.pad(&format!("expert[{i}]")),
            Self::StackedExperts => f.pad("experts[stacked]"),
            Self::SharedExpert => f.pad("shared_expert"),
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
        f.pad(self.as_str())
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
        f.pad(self.as_str())
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
    /// Lowercased name prefix up to and including the layer marker
    /// (`"blk"`, `"model.layers"`, `"visual.blocks"`), when the name carries
    /// a layer index. Distinguishes layer stacks of multi-tower checkpoints.
    pub stack: Option<String>,
    /// Layer index parsed from the name, if any.
    pub layer: Option<usize>,
    /// Stable, human-readable reasons behind the assignment.
    pub evidence: Vec<&'static str>,
    /// Outermost-first dims, copied from the inventory.
    pub dims: Vec<usize>,
    /// Normalized dtype label, copied from the inventory.
    pub dtype: String,
}

/// Layer-local facts the per-tensor classifier may consult.
///
/// Built by [`super::discover_topology`] for each layer; tests can
/// construct it directly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassifyContext {
    /// Some tensor **in the same layer** is an expert tensor
    /// (`experts.N.*`, `*_exps`, `shared_expert*`, `ffn_<role>.<E>`). Lets
    /// a bare `mlp.gate.weight` in that layer be read as a router instead
    /// of staying ambiguous. Must not be set from other layers.
    pub has_expert_tensors: bool,
}

/// Name-only facts used by the topology pass to group tensors before
/// classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NameFacts {
    /// `(stack, layer index)` when the name carries a layer marker.
    pub(super) layer_key: Option<(String, usize)>,
    /// `true` when the name alone marks an MoE expert tensor.
    pub(super) names_expert_tensor: bool,
}

/// Extract grouping facts from a tensor name.
pub(super) fn name_facts(name: &str) -> NameFacts {
    let parts = NameParts::new(name);
    NameFacts {
        layer_key: parts.layer_key(),
        names_expert_tensor: parts.in_experts || parts.in_shared_expert,
    }
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
        stack: parts.layer_key().map(|(stack, _)| stack),
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
/// GGUF per-expert components: `blk.N.ffn_gate.E` or `blk.N.ffn_gate_E`.
const GGUF_EXPERT_ROLES: &[&str] = &["ffn_gate", "ffn_up", "ffn_down"];
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
    /// Lowercased `.`/`/`-separated segments without the parameter suffix
    /// or a trailing GGUF per-expert index.
    segments: Vec<String>,
    /// Last segment of `segments`: the component name (`attn_q`, `q_proj`).
    component: String,
    param: ParamKind,
    /// Position of the layer marker segment, when present.
    layer_marker: Option<usize>,
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

        let param = strip_param_suffix(&mut segments);
        let gguf_expert = split_gguf_expert_index(&mut segments);
        let component = segments.last().cloned().unwrap_or_default();

        let (layer_marker, layer) = match position_and_number_after(&segments, LAYER_MARKERS) {
            Some((pos, n)) => (Some(pos), Some(n)),
            None => (None, None),
        };
        let expert_index = gguf_expert
            .or_else(|| position_and_number_after(&segments, EXPERT_MARKERS).map(|(_, n)| n));
        let in_attention = segments
            .iter()
            .any(|s| ATTENTION_CONTEXT.contains(&s.as_str()));
        let in_shared_expert = segments.iter().any(|s| s.starts_with("shared_expert"))
            || component.ends_with("_shexp");
        let in_experts = gguf_expert.is_some()
            || segments
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
            layer_marker,
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

    /// `(stack prefix, layer index)`; the stack is every segment up to and
    /// including the layer marker.
    fn layer_key(&self) -> Option<(String, usize)> {
        let (marker, layer) = (self.layer_marker?, self.layer?);
        Some((self.segments[..=marker].join("."), layer))
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

/// Pop a recognized parameter suffix (`weight`, `bias`, …) and return its
/// kind; a one-segment name is never stripped.
fn strip_param_suffix(segments: &mut Vec<String>) -> ParamKind {
    if segments.len() < 2 {
        return ParamKind::Weight;
    }
    let last = segments.last().map(String::as_str).unwrap_or_default();
    match PARAM_SUFFIXES.iter().find(|(s, _)| *s == last) {
        Some((_, kind)) => {
            let kind = *kind;
            segments.pop();
            kind
        }
        None => ParamKind::Weight,
    }
}

/// Recognize the GGUF per-expert conventions `ffn_<role>.<E>` and
/// `ffn_<role>_<E>`; normalize the component to `ffn_<role>` and return `E`.
fn split_gguf_expert_index(segments: &mut Vec<String>) -> Option<usize> {
    let last = segments.last()?.clone();
    if let Ok(index) = last.parse::<usize>() {
        let prev = segments.len().checked_sub(2)?;
        if GGUF_EXPERT_ROLES.contains(&segments[prev].as_str()) {
            segments.pop();
            return Some(index);
        }
        return None;
    }
    for role in GGUF_EXPERT_ROLES {
        if let Some(tail) = last.strip_prefix(role).and_then(|t| t.strip_prefix('_'))
            && let Ok(index) = tail.parse::<usize>()
        {
            *segments.last_mut()? = (*role).to_owned();
            return Some(index);
        }
    }
    None
}

/// First `marker.<number>` pair: `(marker position, number)`.
fn position_and_number_after(segments: &[String], markers: &[&str]) -> Option<(usize, usize)> {
    segments.windows(2).enumerate().find_map(|(pos, w)| {
        markers
            .contains(&w[0].as_str())
            .then(|| w[1].parse::<usize>().ok().map(|n| (pos, n)))
            .flatten()
    })
}

enum Outcome {
    Role(TensorRole, RoleConfidence, &'static str),
    Ambiguous(Vec<TensorRole>, &'static str),
    Unknown,
}

fn exact(role: TensorRole, reason: &'static str) -> Option<Outcome> {
    Some(Outcome::Role(role, RoleConfidence::Exact, reason))
}

fn heuristic(role: TensorRole, reason: &'static str) -> Option<Outcome> {
    Some(Outcome::Role(role, RoleConfidence::Heuristic, reason))
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

/// Run the component matchers in precedence order.
fn match_component(parts: &NameParts, ctx: ClassifyContext, dims: &[usize]) -> Outcome {
    if parts.layer.is_none() {
        return match_global(parts).unwrap_or(Outcome::Unknown);
    }
    match_norm(parts)
        .or_else(|| match_attention(parts))
        .or_else(|| match_router(parts, ctx, dims))
        .or_else(|| match_feed_forward(parts))
        .unwrap_or(Outcome::Unknown)
}

/// Tensors with no layer index.
fn match_global(parts: &NameParts) -> Option<Outcome> {
    use TensorRole::*;
    match parts.component.as_str() {
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
        "output_norm" | "ln_f" | "final_layer_norm" | "final_layernorm" | "final_norm" | "norm" => {
            exact(OutputNorm, "output_norm_name")
        }
        "rope_freqs" | "inv_freq" => exact(RotaryFrequencies, "rotary_frequencies_name"),
        _ => None,
    }
}

/// Layer-scoped normalization tensors and rotary tables.
fn match_norm(parts: &NameParts) -> Option<Outcome> {
    use TensorRole::*;
    match parts.component.as_str() {
        "inv_freq" | "rope_freqs" => exact(RotaryFrequencies, "rotary_frequencies_name"),
        "attn_norm"
        | "input_layernorm"
        | "ln_1"
        | "attention_norm"
        | "ln_attn"
        | "pre_attention_layernorm"
        | "attention_layernorm" => exact(AttentionNorm, "attention_norm_name"),
        "post_attention_norm" | "post_attn_norm" => {
            exact(PostAttentionNorm, "post_attention_norm_name")
        }
        // Llama: the pre-MLP norm. Gemma-2: a true post-attention norm,
        // paired with `pre_feedforward_layernorm`. The topology pass
        // resolves this per layer; alone it stays a heuristic.
        "post_attention_layernorm" => {
            heuristic(MlpNorm, "post_attention_layernorm_llama_convention")
        }
        "pre_feedforward_layernorm" => exact(MlpNorm, "pre_feedforward_layernorm_name"),
        "ffn_norm" | "ln_2" | "ln_mlp" | "mlp_norm" | "pre_mlp_layernorm" => {
            exact(MlpNorm, "mlp_norm_name")
        }
        "post_ffw_norm" | "post_feedforward_layernorm" | "post_mlp_norm" => {
            exact(PostMlpNorm, "post_mlp_norm_name")
        }
        "attn_q_norm" | "q_norm" | "query_norm" | "q_layernorm" => {
            exact(AttentionQueryNorm, "attention_query_norm_name")
        }
        "attn_k_norm" | "k_norm" | "key_norm" | "k_layernorm" => {
            exact(AttentionKeyNorm, "attention_key_norm_name")
        }
        _ => None,
    }
}

/// Attention projections. Names that other block types reuse (`in_proj`,
/// `out_proj`, `dense`, `c_proj`, …) require an attention segment in the
/// path; Mamba's `mixer.in_proj` therefore stays unknown.
fn match_attention(parts: &NameParts) -> Option<Outcome> {
    use TensorRole::*;
    let c = parts.component.as_str();
    let unconditional = match c {
        "attn_q" | "q_proj" | "wq" | "query" => exact(AttentionQuery, "query_name"),
        "attn_k" | "k_proj" | "wk" | "key" => exact(AttentionKey, "key_name"),
        "attn_v" | "v_proj" | "wv" | "value" => exact(AttentionValue, "value_name"),
        "attn_qkv" | "c_attn" | "query_key_value" | "qkv_proj" | "wqkv" | "qkv" => {
            exact(AttentionQkv, "fused_qkv_name")
        }
        "attn_output" | "o_proj" | "wo" => exact(AttentionOutput, "attention_output_name"),
        _ => None,
    };
    if unconditional.is_some() || !parts.in_attention || parts.in_feed_forward {
        return unconditional;
    }
    match c {
        "q" => exact(AttentionQuery, "query_name"),
        "k" => exact(AttentionKey, "key_name"),
        "v" => exact(AttentionValue, "value_name"),
        "in_proj" => exact(AttentionQkv, "fused_qkv_name"),
        "dense" | "c_proj" | "proj" | "o" | "out_proj" => {
            exact(AttentionOutput, "attention_output_name")
        }
        _ => None,
    }
}

/// MoE routers, including the documented bare-`gate` ambiguity.
fn match_router(parts: &NameParts, ctx: ClassifyContext, dims: &[usize]) -> Option<Outcome> {
    use TensorRole::*;
    match parts.component.as_str() {
        "ffn_gate_inp" | "router" | "gate_inp" => exact(MoeRouter, "router_name"),
        "shared_expert_gate" => exact(MoeRouter, "shared_expert_gate_name"),
        "gate" => {
            let parent = parts.parent();
            if parent == "block_sparse_moe" || parent == "moe" || parent == "router" {
                exact(MoeRouter, "moe_parent_gate_name")
            } else if parts.in_experts {
                // `experts.N.gate`: a per-expert gate projection.
                heuristic(MlpGate, "expert_scoped_gate_name")
            } else if ctx.has_expert_tensors {
                heuristic(MoeRouter, "gate_name_with_expert_tensors_present")
            } else if router_shape_hint(dims) {
                heuristic(MoeRouter, "gate_name_with_router_shape")
            } else {
                Some(Outcome::Ambiguous(
                    vec![MoeRouter, MlpGate],
                    "bare_gate_name",
                ))
            }
        }
        _ => None,
    }
}

/// Dense and expert feed-forward projections.
fn match_feed_forward(parts: &NameParts) -> Option<Outcome> {
    use TensorRole::*;
    let c = parts.component.as_str();
    let unconditional = match c {
        "ffn_gate" | "ffn_gate_exps" | "ffn_gate_shexp" | "gate_proj" => {
            exact(MlpGate, "gate_proj_name")
        }
        "ffn_up" | "ffn_up_exps" | "ffn_up_shexp" | "up_proj" | "c_fc" | "dense_h_to_4h"
        | "fc1" | "fc_in" | "wi" | "linear_1" => exact(MlpUp, "up_proj_name"),
        "ffn_down" | "ffn_down_exps" | "ffn_down_shexp" | "down_proj" | "dense_4h_to_h" | "fc2"
        | "fc_out" | "linear_2" => exact(MlpDown, "down_proj_name"),
        "gate_up_proj" | "ffn_gate_up" | "ffn_up_gate" | "w13" | "fused_gate_up" => {
            exact(MlpGateUp, "fused_gate_up_name")
        }
        "w1" => heuristic(MlpGate, "w1_llama_convention_gate"),
        "w3" => heuristic(MlpUp, "w3_llama_convention_up"),
        "w2" => heuristic(MlpDown, "w2_llama_convention_down"),
        _ => None,
    };
    if unconditional.is_some() || parts.in_attention {
        return unconditional;
    }
    match c {
        "c_proj" if parts.in_feed_forward => exact(MlpDown, "down_proj_name"),
        "dense" if parts.parent() == "intermediate" => heuristic(MlpUp, "bert_intermediate_dense"),
        // BERT-style `layer.N.output.dense` has no mlp marker at all.
        "dense" if parts.parent() == "output" || parts.has_segment("output") => {
            heuristic(MlpDown, "bert_output_dense")
        }
        _ => None,
    }
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
        assert!(name_facts("blk.0.ffn_gate_exps.weight").names_expert_tensor);
        assert!(name_facts("model.layers.0.mlp.experts.3.up_proj.weight").names_expert_tensor);
        assert!(!name_facts("model.layers.0.mlp.up_proj.weight").names_expert_tensor);
    }

    #[test]
    fn gguf_per_expert_conventions() {
        use RoleScope::*;
        use TensorRole::*;
        // Dot-delimited: blk.B.ffn_ROLE.E.weight
        let a = classify_tensor(
            &info("blk.1.ffn_gate.3.weight", &[8, 8]),
            ClassifyContext::default(),
        );
        assert_eq!(
            (a.role, a.scope, a.confidence),
            (MlpGate, Expert(3), RoleConfidence::Exact)
        );
        assert_eq!(a.layer, Some(1));
        assert_eq!(a.param, ParamKind::Weight);
        // Underscore-delimited: blk.B.ffn_ROLE_E.weight
        let b = classify_tensor(
            &info("blk.1.ffn_down_12.weight", &[8, 8]),
            ClassifyContext::default(),
        );
        assert_eq!((b.role, b.scope), (MlpDown, Expert(12)));
        assert!(name_facts("blk.1.ffn_up.0.weight").names_expert_tensor);
        assert!(name_facts("blk.1.ffn_up_0.weight").names_expert_tensor);
        // Stacked and router names are untouched by the split.
        assert_eq!(role_of("blk.1.ffn_gate_inp.weight").0, MoeRouter);
        assert_eq!(role_of("blk.1.ffn_gate_exps.weight").1, StackedExperts);
        // A numeric tail after a non-ffn component is not an expert index.
        let c = classify_tensor(
            &info("blk.1.attn_q.3.weight", &[8]),
            ClassifyContext::default(),
        );
        assert_eq!(c.role, Unknown);
        assert_eq!(c.scope, Layer);
    }

    #[test]
    fn layer_stack_prefix_is_recorded() {
        let a = classify_tensor(
            &info("model.layers.4.self_attn.q_proj.weight", &[8, 8]),
            ClassifyContext::default(),
        );
        assert_eq!(a.stack.as_deref(), Some("model.layers"));
        assert_eq!(a.layer, Some(4));
        let b = classify_tensor(
            &info("visual.blocks.4.attn.qkv.weight", &[8, 8]),
            ClassifyContext::default(),
        );
        assert_eq!(b.stack.as_deref(), Some("visual.blocks"));
        let c = classify_tensor(
            &info("blk.0.attn_q.weight", &[8, 8]),
            ClassifyContext::default(),
        );
        assert_eq!(c.stack.as_deref(), Some("blk"));
        let d = classify_tensor(
            &info("token_embd.weight", &[8, 8]),
            ClassifyContext::default(),
        );
        assert_eq!(d.stack, None);
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
    fn reused_projection_names_need_attention_context() {
        use TensorRole::*;
        // Mamba / SSM mixers reuse `in_proj` / `out_proj`: never attention.
        assert_eq!(role_of("backbone.layers.0.mixer.in_proj.weight").0, Unknown);
        assert_eq!(
            role_of("backbone.layers.0.mixer.out_proj.weight").0,
            Unknown
        );
        // Under an attention segment they are attention projections.
        assert_eq!(
            role_of("model.layers.0.self_attn.in_proj.weight").0,
            AttentionQkv
        );
        assert_eq!(
            role_of("model.layers.0.attn.out_proj.weight").0,
            AttentionOutput
        );
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

    #[test]
    fn display_honors_width_and_alignment() {
        assert_eq!(format!("{:<12}|", TensorRole::MlpUp), "mlp_up      |");
        assert_eq!(format!("{:>8}|", ParamKind::Bias), "    bias|");
        assert_eq!(format!("{:<10}|", RoleConfidence::Exact), "exact     |");
        assert_eq!(format!("{:<12}|", RoleScope::Expert(7)), "expert[7]   |");
    }
}
