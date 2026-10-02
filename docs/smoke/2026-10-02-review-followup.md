# PR #104 feedback validation — 2026-10-02

Follow-up to [the initial report](2026-10-02.md), on base
`f8776151303aff3152c5e2e06f8b2f2d2f167a2d` plus this review-fix patch.
The original report remains evidence for the original implementation.

Changes: normalize native INT4 to I4 for the normalized metadata comparison;
retain exact native dtype matching; measure HWM growth relative to baseline
HWM; clarify that external metadata checks cover TSV samples. Three bounded
payload windows remain intentional, as required by RM-1794.

Fixtures, TSVs, file hashes, budgets and copy audit are unchanged from the
initial report. Linux 6.18.49 x86_64, 4096-byte pages; release executable built
with Rust 1.99.0/Cargo 1.99.0. Cache remains uncontrolled from earlier runs;
no cold-cache claim. Each warm run follows the first run in a fresh process.
The first/warm distinction below denotes invocation order only.

Resident growth is now `max(peak_RSS - baseline_RSS, HWM - baseline_HWM)`
with saturating subtraction. A pre-baseline high-water mark cannot by itself
fail the growth budget. The guide documents the remaining snapshot limitation.

Stable and Rust 1.97.1 formatting, all-feature Clippy with warnings denied,
builds and tests passed: 223 tests including five smoke-harness tests.
Three existing real-file pilots remain ignored. Default/single-feature and
RustSec results from the initial report are reused; their inputs are unchanged.

## Real fixture runs

All memory figures below are bytes; all runs use 128 MiB resident-growth and
64 MiB heap-peak budgets. Metadata, bytes, pointers, lookup orders and typed
missing-resource errors passed.

### gguf / first

```bash
./target/release/examples/checkpoint_smoke gguf /home/vercel-sandbox/workspace/inputs/checkpoint-smoke/qwen.gguf docs/smoke/qwen-q8.tsv 128 64
```

| Phase | RSS | HWM | Virtual | Anonymous | Heap peak | Largest allocation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2232320 | 2236416 | 4005888 | 139264 | 6068 | 2048 |
| file_open | 2236416 | 2236416 | 4005888 | 143360 | 6068 | 2048 |
| mmap_open_and_metadata_index | 8126464 | 8126464 | 1898704896 | 401408 | 217080 | 61968 |
| inventory | 8130560 | 8130560 | 1898704896 | 401408 | 217080 | 61968 |
| first_access | 8208384 | 8208384 | 1898704896 | 401408 | 217080 | 61968 |
| repeated_access | 8208384 | 8208384 | 1898704896 | 401408 | 217080 | 61968 |
| errors_checked | 8220672 | 8220672 | 1898704896 | 401408 | 217080 | 61968 |

`observed rss_or_hwm_growth=5988352 heap_peak=217080 largest_allocation=61968`; exit **0 / PASS**.

### gguf / warm

```bash
./target/release/examples/checkpoint_smoke gguf /home/vercel-sandbox/workspace/inputs/checkpoint-smoke/qwen.gguf docs/smoke/qwen-q8.tsv 128 64
```

| Phase | RSS | HWM | Virtual | Anonymous | Heap peak | Largest allocation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2338816 | 2342912 | 4005888 | 135168 | 6068 | 2048 |
| file_open | 2342912 | 2342912 | 4005888 | 139264 | 6068 | 2048 |
| mmap_open_and_metadata_index | 8187904 | 8187904 | 1898704896 | 401408 | 217080 | 61968 |
| inventory | 8187904 | 8187904 | 1898704896 | 401408 | 217080 | 61968 |
| first_access | 8265728 | 8265728 | 1898704896 | 401408 | 217080 | 61968 |
| repeated_access | 8265728 | 8265728 | 1898704896 | 401408 | 217080 | 61968 |
| errors_checked | 8265728 | 8265728 | 1898704896 | 401408 | 217080 | 61968 |

`observed rss_or_hwm_growth=5926912 heap_peak=217080 largest_allocation=61968`; exit **0 / PASS**.

### safetensors / first

```bash
./target/release/examples/checkpoint_smoke safetensors /home/vercel-sandbox/workspace/inputs/checkpoint-smoke/phi/model.safetensors.index.json docs/smoke/phi3.tsv 128 64
```

| Phase | RSS | HWM | Virtual | Anonymous | Heap peak | Largest allocation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2256896 | 2256896 | 4005888 | 139264 | 6380 | 2048 |
| file_open | 2256896 | 2256896 | 4005888 | 139264 | 6380 | 2048 |
| mmap_open_and_metadata_index | 3239936 | 3239936 | 7646564352 | 512000 | 256206 | 36864 |
| inventory | 3244032 | 3244032 | 7646564352 | 512000 | 256206 | 36864 |
| first_access | 3334144 | 3334144 | 7646564352 | 512000 | 256206 | 36864 |
| repeated_access | 3334144 | 3334144 | 7646564352 | 512000 | 256206 | 36864 |
| errors_checked | 3338240 | 3338240 | 7646564352 | 516096 | 256206 | 36864 |

`observed rss_or_hwm_growth=1081344 heap_peak=256206 largest_allocation=36864`; exit **0 / PASS**.

### safetensors / warm

```bash
./target/release/examples/checkpoint_smoke safetensors /home/vercel-sandbox/workspace/inputs/checkpoint-smoke/phi/model.safetensors.index.json docs/smoke/phi3.tsv 128 64
```

| Phase | RSS | HWM | Virtual | Anonymous | Heap peak | Largest allocation |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2244608 | 2244608 | 4005888 | 143360 | 6380 | 2048 |
| file_open | 2244608 | 2244608 | 4005888 | 143360 | 6380 | 2048 |
| mmap_open_and_metadata_index | 3244032 | 3244032 | 7646564352 | 516096 | 256206 | 36864 |
| inventory | 3244032 | 3244032 | 7646564352 | 516096 | 256206 | 36864 |
| first_access | 3330048 | 3330048 | 7646564352 | 516096 | 256206 | 36864 |
| repeated_access | 3330048 | 3330048 | 7646564352 | 516096 | 256206 | 36864 |
| errors_checked | 3334144 | 3334144 | 7646564352 | 520192 | 256206 | 36864 |

`observed rss_or_hwm_growth=1089536 heap_peak=256206 largest_allocation=36864`; exit **0 / PASS**.

## Negative check and source identity

The GGUF run with a 1 MiB RSS budget still fails with exit 1 and the expected
resident/high-water growth diagnostic. Regression tests cover startup HWM,
new HWM growth, sampled peaks followed by reclamation, saturating subtraction,
and matching/mismatching native and normalized INT4 metadata.

SHA256 of executed source:

```text
9666c468a945de97929fa5cd1d9d7c84de6984138aa46b6e3ab81b439bd8ba9b  examples/checkpoint_smoke.rs
01811e09a5e32d13584750b6475061cd51c488f0680be9c719bec556de689526  examples/checkpoint_smoke/memory.rs
c66f5bbecb776bc39280fd7b5e00e96278cbd6ef470933e54b59d4cff6f9d79a  examples/checkpoint_smoke/run.rs
```
