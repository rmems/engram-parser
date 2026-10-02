# Local Ollama Granite GGUF smoke — 2026-10-02

Two fresh-process GGUF mmap smoke runs passed on Fedora 44, Linux
`7.1.5-200.fc44.x86_64`, with 4096-byte pages and Rust/Cargo 1.99.0.
This is pre-merge evidence; repeat the publication gate on the exact merged
release commit. No model was downloaded, copied, modified, or packaged.

The local Ollama manifest for `granite4.2:8b` points to model blob
`sha256-16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311`.
The 5,347,917,952-byte file's full SHA256 matches that digest. An independent
local `llama.cpp` Python GGUF reader found GGUF v3, 363 tensors, and 36
metadata keys. Five independently selected records in
[ollama-granite4.2-8b.tsv](ollama-granite4.2-8b.tsv) span early, middle, and
late offsets and cover Q6_K, Q4_K, and F32 layouts. The TSV was generated
from the independent reader's directory, reversing native GGUF dimensions
into the checkpoint API's outermost-first order; it was not copied from
`engram-parser` output.

```bash
cargo build --locked --release --all-features --example checkpoint_smoke
./target/release/examples/checkpoint_smoke gguf \
  ~/.ollama/models/blobs/sha256-16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311 \
  docs/smoke/ollama-granite4.2-8b.tsv 128 64
```

The command exited 0 twice in separate processes:

| Run | Mapped bytes | Tensors | RSS/HWM growth | Rust heap peak | Largest allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 5,347,917,952 | 363 | 4,046,848 | 267,793 | 61,968 |
| Warm | 5,347,917,952 | 363 | 4,083,712 | 268,337 | 61,968 |

Both runs used 128 MiB resident-growth and 64 MiB Rust heap-peak budgets.
The five records matched name, dtype, shape, byte length, source file, and
absolute offset. Three bounded byte windows per tensor matched independent
positional file reads; returned payloads were borrowed from mmap. Eight
reordered lookup rounds preserved pointer and sampled-byte identity, and
missing-tensor handling passed. The full-file SHA256 scan and header
inspection make cache state uncontrolled; “first” and “warm” mean invocation
order, not cold-cache measurement.

The complete local logs are under `/tmp/engram-release-v030/ollama-granite/`
and are not shared release attachments. The bounded sample checks do not
prove numerical dequantization or inference correctness.
