# Real checkpoint fixtures

`tests/real_topology.rs` runs `discover_topology` over real dense and MoE
checkpoints in both GGUF and Safetensors (RM-1800, on the RM-2367 pin set).
No weights are committed and CI never downloads them: every real-model test
is `#[ignore]`d and gated by `tests/common/real_fixtures.rs`.

## Running

```bash
# Pre-staged weights: point the helper at a root holding <fixture id>/ dirs
# (e.g. C:\models\qwen3-0.6b\model.safetensors):
ENGRAM_FIXTURE_ROOT=C:\models cargo test --all-features \
    --test real_topology -- --ignored --nocapture

# Or let it download ~28 GB once into a verified cache:
ENGRAM_REAL_FIXTURES=require cargo test --all-features \
    --test real_topology -- --ignored
```

`ENGRAM_REAL_FIXTURES` modes: `auto` (default; use a verified local root or
cache, download if missing, skip if unreachable), `offline` (local only),
`skip`, `require` (fail instead of skipping). `ENGRAM_FIXTURE_ROOT` points at
a directory of `<fixture id>/` folders; `ENGRAM_FIXTURE_CACHE` overrides the
download cache (default `%TEMP%/engram-fixtures`); `ENGRAM_FIXTURE_BASE_URL`
swaps the `https://huggingface.co/` origin for a mirror. Pinned bytes are
verified by size + SHA-256; a checksum mismatch always fails, and verified
files carry a stamp so repeat runs skip re-hashing.

## Pin set

| Model | Repo | Revision | Files |
|---|---|---|---|
| Qwen3-0.6B (dense) | `Qwen/Qwen3-0.6B` | `c1899de2` | `model.safetensors` + `config.json` |
| | `Qwen/Qwen3-0.6B-GGUF` | `23749fef` | `Qwen3-0.6B-Q8_0.gguf` |
| Qwen2.5-0.5B-Instruct (dense) | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae55760` | `model.safetensors` + `config.json` |
| | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | `9217f5db` | `qwen2.5-0.5b-instruct-q8_0.gguf` |
| OLMoE-1B-7B-0125-Instruct (MoE) | `allenai/OLMoE-1B-7B-0125-Instruct` | `b89a7c4b` | 3 shards + index + `config.json` |
| | `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` | `2ac1d273` | `OLMoE-1B-7B-0125-Instruct-Q8_0.gguf` |
| Qwen3.5-0.8B (hybrid, stretch) | `Qwen/Qwen3.5-0.8B` | `2fc06364` | `model.safetensors-00001-of-00001` + index + `config.json` |
| | `unsloth/Qwen3.5-0.8B-GGUF` | `6ab46149` | `Qwen3.5-0.8B-Q8_0.gguf` |

Full pinned revisions, sizes, and SHA-256s live in
`tests/common/real_fixtures.rs`.

## Observed results (run against the pins, Windows, 2026-11)

| Model | Format | Layers | Experts | Unknowns |
|---|---|---|---|---|
| Qwen3-0.6B | GGUF / ST | 28 / 28 | — | 0 / 0 |
| Qwen2.5-0.5B | GGUF / ST | 24 / 24 | — | 0 / 0 |
| OLMoE | GGUF / ST | 16 / 16 | 64 / 64 (8 active, no shared) | 0 / 0 |
| Qwen3.5-0.8B | GGUF | 24 text | — | 144 (`ssm_*`, `attn_gate`) |
| | ST | 24 text + 12 vision + 1 MTP | — | 268 (`linear_attn`, vision, `mtp`) |

Documented format differences the parity checks allow:

- **Output head**: Qwen3-0.6B GGUF ties embeddings but its Safetensors copy
  materializes `lm_head.weight`; Qwen2.5-0.5B is the reverse (GGUF carries
  `output.weight`, Safetensors ties). `tied_embeddings_hint` therefore
  differs per format by design.
- **Expert storage**: OLMoE GGUF packs experts into stacked
  `ffn_{gate,up,down}_exps` tensors (195 tensors total) while Safetensors
  keeps one tensor per expert (3219); expert dims compare after dropping the
  leading expert axis.
- **Hybrid layout**: Qwen3.5's GGUF carries only the 24-layer text tower and
  names the linear-attention in-projection `attn_qkv` (Partial attention);
  the Safetensors copy also carries `model.visual.blocks` and `mtp.layers`
  stacks whose tensors stay Unknown. Six `full_attention` layers
  (indices 3, 7, 11, 15, 19, 23) are fully classified in both formats.
