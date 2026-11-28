# Real GGUF dequantization conformance (RM-2013)

This ignored CPU-only test compares engram-parser against the independent
`gguf-py` decoder used for the synthetic oracle (RM-2209). Both readers open the
**same GGUF** and resolve the tensor name independently. This checks GGUF
directory/payload decoding and dequantization; it does not establish numeric
parity with independently distributed Safetensors files. See the
[fixture preflight](real-fixtures.md#shared-manifest-and-preflight-rm-2367)
for cross-format checks.

## Reproduce

Install or run with `gguf==0.19.0` (the oracle rejects any other version).
Download only the desired artifacts below from the
[revision-pinned Qwen repository](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67),
outside the checkout, and verify SHA-256. Both are Apache-2.0.

| File | Exact bytes | SHA-256 |
|---|---:|---|
| `qwen2.5-0.5b-instruct-q8_0.gguf` | 675710816 | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `qwen2.5-0.5b-instruct-q5_k_m.gguf` | 522186592 | `041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55` |

```bash
uv run --with 'gguf==0.19.0' env \
  ENGRAM_GGUF=/models/qwen2.5-0.5b-instruct-q8_0.gguf \
  ENGRAM_GGUF_ORACLE_PYTHON=python ENGRAM_GGUF_ORACLE_DTYPES=Q8_0 \
  cargo test --locked --features mmap --test real_gguf \
  real_gguf_gguf_py_conformance -- --ignored --nocapture

uv run --with 'gguf==0.19.0' env \
  ENGRAM_GGUF=/models/qwen2.5-0.5b-instruct-q5_k_m.gguf \
  ENGRAM_GGUF_ORACLE_PYTHON=python ENGRAM_GGUF_ORACLE_DTYPES=Q5_K,Q6_K \
  cargo test --locked --features mmap --test real_gguf \
  real_gguf_gguf_py_conformance -- --ignored --nocapture
```

The test mmaps the checkpoint and selects the smallest nonempty tensor of
each requested wire dtype (ties by name), up to 8,388,608 elements. It
compares the full selected tensor, failing on missing dtypes, shape-count
mismatch, non-finite data, or errors outside
`abs(actual - reference) <= 1e-7 + 1e-6 * abs(reference)`.
It prints per-dtype maximum absolute error and maximum relative error
(the denominator is floored at `1e-12` to handle zero reference values).
Normal tests neither invoke Python nor download weights.

## Observed results

CPU, Linux x86_64, Rust 1.99.0, gguf-py 0.19.0, NumPy 2.5.5, using the
files above. These are sampled tensors, not full-model comparisons.

| Wire dtype | Tensor | Values | Max abs | Max rel |
|---|---|---:|---:|---:|
| Q8_0 | `blk.0.attn_k.weight` | 114688 | 0 | 0 |
| Q5_K | `blk.11.ffn_down.weight` | 4358144 | 0 | 0 |
| Q6_K | `blk.0.ffn_down.weight` | 4358144 | 0 | 0 |
