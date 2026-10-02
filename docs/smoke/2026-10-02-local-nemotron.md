# Local Nemotron single-file Safetensors smoke — 2026-10-02

Two fresh-process runs passed with Rust/Cargo 1.99.0 on Fedora 44, Linux
`7.1.5-200.fc44.x86_64`, with 4096-byte pages. This is pre-merge evidence
for PR #107 plus the single-file smoke change; repeat the gate on the exact
merged release commit before publication. The fixture was already present at
`~/.models/safetensors/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/model.safetensors`.
No model weights were downloaded, modified, or added to this repository.

The local Hugging Face metadata names revision
`dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`. The 7,947,142,640-byte
file has SHA256
`55d4e2519456c4a9bddf596b0748d630e3b2ce6ff6f4c2b7ed3e07e2b00dad42`,
verified against the local download metadata. The model card identifies the
NVIDIA Nemotron Open Model License; this report redistributes no weights.

An independent Python `struct`/`json` read of the 28,968-byte header found
263 BF16 tensors. Five records in [nemotron-nano-4b.tsv](nemotron-nano-4b.tsv)
were selected across early, middle, and late payload offsets. Their byte
spans were checked against `2 × product(shape)`. No tensor payload was read
to generate the TSV.

Before the smoke change, the executable exited 1 with
`Safetensors smoke requires an explicit shard index` for this single-file
model. After the change, the exact invocation below exited 0 twice in fresh
processes:

```bash
cargo build --locked --release --all-features --example checkpoint_smoke
./target/release/examples/checkpoint_smoke safetensors \
  ~/.models/safetensors/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/model.safetensors \
  docs/smoke/nemotron-nano-4b.tsv 128 64
```

| Run | Mapped bytes | Tensors | RSS/HWM growth | Rust heap peak | Largest allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 7,947,142,640 | 263 | 1,953,792 | 523,668 | 73,728 |
| Warm | 7,947,142,640 | 263 | 1,892,352 | 523,668 | 73,728 |

Both runs used 128 MiB resident-growth and 64 MiB Rust heap-peak budgets.
All five sampled records matched name, dtype, shape, byte length, source file,
and absolute offset. The returned payloads were borrowed from mmap; three
bounded byte windows per tensor matched independent positional file reads.
Eight reordered lookup rounds retained pointer and sampled-byte identity;
missing-tensor and isolated missing-shard errors were typed as expected.
The 7.95 GB virtual mapping is not a 7.95 GB resident allocation. The runs
followed header inspection and a file-hash scan, so cache state is
uncontrolled; “first” and “warm” denote invocation order only.

The complete local logs are under `/tmp/engram-release-v030/nemotron/` and
are not shared release attachments. Sampled byte comparisons do not prove
whole-file integrity beyond the separate SHA256 scan or numerical model
execution.
