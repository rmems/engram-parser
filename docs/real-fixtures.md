# Real checkpoint fixtures

`tests/real_topology.rs` runs `discover_topology` over real dense and MoE
checkpoints in both GGUF and Safetensors (RM-1800, on the RM-2367 pin set).
No weights are committed and CI never downloads them: every real-model test
is `#[ignore]`d and gated by `tests/common/real_fixtures.rs`.

## Running

```bash
# Pre-staged weights: point the helper at a root holding <fixture id>/ dirs
# (e.g. C:/models/qwen3-0.6b/model.safetensors):
ENGRAM_FIXTURE_ROOT=C:/models cargo test --all-features \
    --test real_topology -- --ignored --nocapture

# Or let it download ~28 GB once into a verified cache:
ENGRAM_REAL_FIXTURES=require cargo test --all-features \
    --test real_topology -- --ignored
```

`ENGRAM_REAL_FIXTURES` modes: `auto` (default; use a verified local root or
cache, download if missing, skip if unreachable), `offline` (local only),
`skip`, `require` (fail instead of skipping). `ENGRAM_FIXTURE_ROOT` points at
a directory of `<fixture id>/` folders; `ENGRAM_FIXTURE_CACHE` overrides the
download cache (default `<target>/tmp/engram-real-fixtures`);
`ENGRAM_FIXTURE_BASE_URL`
swaps the `https://huggingface.co/` origin for a mirror. Pinned bytes are
verified by size and SHA-256, then stamped so repeat runs can skip re-hashing.
A mismatch in pre-staged or newly downloaded files fails. A corrupt cache
entry fails in offline mode without removing the file. In auto or
require mode the helper tries to replace it with verified bytes and fails
if the repair cannot finish.

## Pin set

| Model | Repo | Revision | Files |
|---|---|---|---|
| Qwen3-0.6B (dense) | `Qwen/Qwen3-0.6B` | `c1899de2` | `model.safetensors` + `config.json` |
| | `Qwen/Qwen3-0.6B-GGUF` | `23749fef` | `Qwen3-0.6B-Q8_0.gguf` |
| Qwen2.5-0.5B-Instruct (dense) | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae55760` | `model.safetensors` + `config.json` |
| | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | `9217f5db` | `qwen2.5-0.5b-instruct-q8_0.gguf` |
| OLMoE-1B-7B-0125-Instruct (MoE) | `allenai/OLMoE-1B-7B-0125-Instruct` | `b89a7c4b` | 3 shards + index + `config.json` |
| | `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` | `2ac1d273` | `OLMoE-1B-7B-0125-Instruct-Q8_0.gguf` |
| Qwen3.5-0.8B (hybrid, stretch) | `Qwen/Qwen3.5-0.8B` | `2fc06364` | `model.safetensors-00001-of-00001.safetensors` + index + `config.json` |
| | `unsloth/Qwen3.5-0.8B-GGUF` | `6ab46149` | `Qwen3.5-0.8B-Q8_0.gguf` |

Full pinned revisions, sizes, and SHA-256s live in
`tests/common/real_fixtures.rs`.

## Shared manifest and preflight (RM-2367)

[Cross-format fixture manifest](smoke/cross-format-fixtures.tsv) lists the
exact revision, path, bytes, SHA-256, and license for each weight and sidecar.
The dtype column describes the artifact's weight format; GGUF can mix dtypes.
The parity candidates are the three baseline pairs; the staged Qwen3.5 Q8_0
stretch pair is included for topology only. The remaining optional quant ladder
and LFM2.5 fixtures are outside this manifest. An integration test verifies
that the manifest matches the fetch pins, so other repositories can consume
the TSV without duplicating model lists.

The ignored CPU preflight verifies downloaded bytes through the shared helper,
opens both formats with mmap, then checks the name, dtype and normalized shape
for `blk.0.attn_v.weight` (GGUF) and
`model.layers.0.self_attn.v_proj.weight` (Safetensors). It does not assert that
their values match.

```bash
# Cache-only (default for preflight); missing files print SKIP.
ENGRAM_FIXTURE_ROOT=/models cargo test --all-features --locked \
  --test real_fixture_preflight -- --ignored --nocapture

# Require one pair and permit its pinned download if missing (~1.66 GB).
ENGRAM_FIXTURE_ROOT=/models ENGRAM_REAL_FIXTURES=require \
  cargo test --release --all-features --locked --test real_fixture_preflight \
    qwen25_fixture_preflight -- --ignored --nocapture
```

The other baseline filters are `qwen3_fixture_preflight` (~2.14 GB) and
`olmoe_fixture_preflight` (~21.20 GB). Cache-only preflight skips missing pins
but fails on a detected size or SHA-256 mismatch in pre-staged files or
the download cache. `require` permits repairing a cache mismatch by
downloading a verified replacement; it never skips.

### Numeric parity is not established

The pinned Qwen2.5-0.5B-Instruct pair passes the structural preflight but
does not pass a numeric parity check. The Q8_0 GGUF
(`9217f5db79a29953eb74d5343926648285ec7e67`) and BF16 Safetensors
(`7ae557604adf67be50417f59c2c2f167def9a775`) produce, respectively,
`0.0037572384` and `0.006652832` at V projection element 0 after decoding.
The absolute difference (`0.0028955936`) exceeds the expected Q8_0 rounding
bound for the source block (`0.0000751873`), including the quantization
half-step and rounding the scale to F16. Independent `gguf-py==0.19.0`
and direct Safetensors BF16 byte reads reproduced the discrepancy. The
corresponding F32 GGUF and BF16 Safetensors input norm also differed, so this
should not be dismissed as Q8_0 noise. The observation does not establish the
cause. Confirm conversion provenance or repin a verified common weight source
before treating the pair as a numeric parity gate. No numeric result is
claimed for Qwen3 or OLMoE.

## Observed topology results (run against the pins, Windows, 2026-11)

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
