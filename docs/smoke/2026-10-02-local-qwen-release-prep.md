# Local Qwen GGUF release-preparation smoke — 2026-10-02

Both fresh-process runs passed on Fedora 44, Linux
`7.1.5-200.fc44.x86_64`, with 4096-byte pages and Rust/Cargo 1.99.0.
This is **pre-commit** evidence from the release-preparation working copy,
not qualification of a final merged or published commit. The GGUF parser and
smoke executable matched the source audited at `70da04b35040e2a0865dfd0ef42d1ba5996ab06d`;
the local edits at run time were docs.rs metadata in `Cargo.toml` and release
instructions in `RELEASE.md`. This report is being added afterward.

The fixture was the pinned
[Qwen2.5-1.5B-Instruct Q8_0 GGUF](../checkpoint-smoke.md#run-with-local-fixtures)
at revision `91cad51170dc346986eccefdc2dd33a9da36ead9` (Apache-2.0),
1,894,532,128 bytes, SHA256
`d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8`.
The checked-in independent expectations in [qwen-q8.tsv](qwen-q8.tsv) select
five F32/Q8_0 tensors across the file. The fixture was downloaded to `/tmp`
for this run and removed afterward; it is not in the repository or package.
The host cache was uncontrolled after the download. “First” and “warm” below
mean invocation order, not a claim of cold-cache measurement.

```bash
cargo build --locked --release --all-features --example checkpoint_smoke
./target/release/examples/checkpoint_smoke gguf /tmp/engram-release-v030/qwen.gguf docs/smoke/qwen-q8.tsv 128 64
./target/release/examples/checkpoint_smoke gguf /tmp/engram-release-v030/qwen.gguf docs/smoke/qwen-q8.tsv 128 64
```

| Fresh process | Exit | Mapped bytes | Tensors | RSS/HWM growth | Rust heap peak | Largest allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First | 0 | 1,894,532,128 | 339 | 6,590,464 | 216,960 | 61,968 |
| Warm | 0 | 1,894,532,128 | 339 | 6,717,440 | 216,960 | 61,968 |

Both runs used 128 MiB RSS-growth and 64 MiB heap-peak budgets. All five
sampled records matched name, dtype, shape, byte length, file offset, and
bounded file-read comparisons. Payloads were borrowed from the mapping;
eight reordered lookup rounds retained pointer and sampled-byte identity.
Missing-resource error checks passed. The complete local logs are
`/tmp/engram-release-v030/gguf-first.log` and `gguf-warm.log`; those paths
are temporary and are not release attachments. The sampled windows do not
prove whole-file integrity or numerical dequantization correctness.

For final publication, repeat the release gate on the exact merged commit
and link that evidence from the release issue. Earlier Qwen/Phi-3 results and
the smoke method are in [the fixture guide](../checkpoint-smoke.md).
