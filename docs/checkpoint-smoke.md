# Large checkpoint mmap smoke (RM-1794)

This is an **opt-in Linux, CPU-only** release smoke, not a throughput benchmark.
Normal tests do not download models, run the smoke executable, or need GPUs.
The example requires both `mmap` and `safetensors`; it adds no dependencies.
Existing GGUF mmap, sparse-file, K-quant and cross-format parity tests remain in
place. Track the work in [RM-1794](https://linear.app/rpd-34/issue/RM-1794/bench-add-large-checkpoint-mmap-and-sharded-safetensors-smoke-coverage).

## Run with local fixtures

Use a 64-bit Linux host with readable `/proc/self/status` and
`/proc/self/smaps_rollup`, enough virtual address space for the checkpoint, and
immutable local checkpoint files. Do not truncate or rewrite mapped files.
Build before measuring, then run each format in a fresh process:

```bash
cargo build --locked --release --all-features --example checkpoint_smoke
set -o pipefail
./target/release/examples/checkpoint_smoke gguf \
  /models/qwen.gguf docs/smoke/qwen-q8.tsv 128 64 | tee /tmp/gguf-smoke.txt
./target/release/examples/checkpoint_smoke safetensors \
  /models/phi/model.safetensors.index.json docs/smoke/phi3.tsv 128 64 \
  | tee /tmp/safetensors-smoke.txt
```

The last two required arguments are **maximum resident growth in MiB** and
**maximum absolute Rust live-heap peak in MiB**. These are fixture-specific
budgets, not universal performance promises. Missing arguments, missing local
files, malformed expectations, wrong metadata/bytes, unexpected owned payloads,
and exceeded budgets exit nonzero. There is no download or owned-read fallback.
Only a completed successful run prints `PASS`. Preserve stderr and exit status
as well as stdout when recording failures.

The sample TSV files match these exact public upstream artifacts:

| Format | Source and immutable revision | Artifact | License |
|---|---|---|---|
| GGUF | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9) | `qwen2.5-1.5b-instruct-q8_0.gguf`, locally named `qwen.gguf` | Apache-2.0 |
| Safetensors | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/f39ac1d28e925b323eae81227eaba4464caced4e) | `model.safetensors.index.json` and both referenced shards | MIT |

Download only when explicitly running the smoke, outside the checkout. Review
upstream licenses and model cards before redistributing model artifacts. This
repository contains only sample metadata and reports, not weights. For example:

```bash
mkdir -p /models/phi
curl --fail --location --retry 2 \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q8_0.gguf \
  -o /models/qwen.gguf
for file in model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  curl --fail --location --retry 2 \
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/f39ac1d28e925b323eae81227eaba4464caced4e/$file" \
    -o "/models/phi/$file" || exit 1
done
```

### Other fixtures and expected metadata

Supply a UTF-8 TSV containing six tab-separated fields per row:

```text
# name  native dtype  outermost-first shape  byte length  root-relative file  absolute file offset
```

Dimensions are comma-separated decimal integers, or `scalar` for rank zero.
Comments start with `#`. Names must be unique. Supply at least three tensors,
spread over early/middle/late payload offsets, including packed types for a
quantized GGUF. For Safetensors select early/middle/late tensors in at least two
shards. The executable requires multiple actual shards and samples from at
least two of them. The checked-in selections cover both ends of each fixture.

Obtain expectations from an independent header inspection or a trusted fixture
manifest, not by copying this executable's inventory output. For GGUF, reverse
native dimensions; compute packed lengths from the GGML block layout and add
the aligned data-section start to the directory offset. For Safetensors, verify
each name against the index's `weight_map`, then read its shard header: absolute
offset is `8 + header_length + data_offsets[0]` and byte length is the offset
span (also check shape × dtype size). The checked-in TSVs were derived with
Python `struct`/`json` header reads independently of the Rust checkpoint API.
The TSV records the native dtype label; normalized labels are checked after
resolving aliases such as native `INT4` to normalized `I4`.

No tensor payload is read when generating the expectations. Header inspection
can warm filesystem cache; record that fact. File hashes, if computed, scan the
entire file and should be taken separately from smoke memory measurements.

## What is measured and checked

Output records UTC Unix time, OS/architecture/kernel, page size, exact arguments,
file sizes, tensor/shard counts, metadata-key count, expected sample metadata,
and FNV-1a fingerprints of the sampled windows (not cryptographic file hashes).
For every TSV-sampled tensor the common `Checkpoint` interface must return
matching name, shape, dtype, byte length, source shard and absolute file offset.
Raw bytes must
be borrowed. Unsampled inventory entries are checked for sorted, unique names;
their metadata is not compared with external expectations. Three windows of at
most 4 KiB each (start, middle, end) are compared with independent positional
file reads. Whole tensors are never scanned or
hashed. Tiny windows can overlap. Eight subsequent rounds alternate reverse
and rotated lookup orders and require identical pointers, metadata and sampled
bytes. Pointer equality alone is not proof of correct bytes or absence of copies.

Memory snapshots are separate for baseline, file open, mmap open plus
metadata/index loading, inventory enumeration, first access, repeated access,
and negative cases. Public `open_checkpoint_mmap` combines mapping and metadata
loading; the report does **not** pretend to time or measure those internals
independently. `file_open` measures a plain descriptor open; the next phase
includes format detection, all shard opens, parsing and catalog construction.

All reported memory quantities are bytes:

- `mapped_file_bytes`: sum of payload file lengths (page rounding excluded).
- `virtual`: total process `VmSize`, including file mappings, heap and runtime.
- `rss`: resident process memory from `smaps_rollup`; `anonymous` separates its
  anonymous pages. File-backed pages contribute to RSS without being heap copies.
- `hwm`: kernel `VmHWM`; together with the largest observed precise RSS snapshot,
  catches new process peaks even if later pages are reclaimed. The bound uses
  the larger of observed RSS growth above baseline RSS and HWM growth above
  baseline HWM (both saturating at zero). Startup peaks are excluded. A transient
  peak below startup HWM between snapshots can still be missed; the Rust heap
  tracker and copy audit provide additional evidence. Report/harness overhead
  after baseline is included.
- `heap_peak` and `largest_allocation`: atomically tracked Rust `System` allocator
  requests, including realloc and zeroed allocations, from process start. The
  peak catches transient Rust copies released before an RSS snapshot. It excludes
  allocator bookkeeping, native allocations outside Rust and direct mmap calls.

The checked-in fixtures use a 128 MiB resident-growth allowance and 64 MiB Rust
heap-peak allowance. Their headers are approximately 5.7 MiB (GGUF) and 23 KiB
(Safetensors combined), and samples touch at most 60/72 KiB per round before
window overlap. The allowances leave room for tokenizer/header parsing,
catalogs, allocator overhead, page rounding and kernel fault-around/readahead.
They remain far below a full checkpoint/shard allocation. The program rejects
heap budgets at least as large as the smallest payload file, and RSS budgets at
least half the total checkpoint size. For a different fixture, justify budgets
from header/index sizes, tensor count, sample volume, page size and platform;
do not raise them merely to turn a failing run green.

An already cached page still counts toward this process's RSS once faulted into
its mapping; unreferenced filesystem-cache pages do not. A multi-GiB virtual
mapping is expected and is not a multi-GiB resident allocation. Repeat each
invocation in a new process and label the second run warm. A first run after a
download/header inspection is **cache uncontrolled**, not cold. For cold-cache
experiments use a dedicated host and an operator-controlled cache procedure;
this executable never drops shared host caches. These checks do not measure
system-wide page-cache growth or claim whole-file integrity.

Missing tensor lookup must yield `MissingTensor`. Missing-shard validation uses
an isolated temporary index pointing at an absent shard and requires
`MissingShard`; the user's checkpoint is never mutated. This checks an absent
shard at open time, not removal of a file already mapped into a running process.

### Access-path copy audit

Memory measurements alone cannot establish absence of copies. For this path:

1. `checkpoint::open::open_checkpoint_mmap` dispatches to mmap backends.
2. GGUF `load_gguf_mmap_with_limits` maps the file and parses its directory;
   `GgufMmapBackend` constructs a metadata-only catalog. Tensor access slices the
   mapping and returns `Cow::Borrowed`.
3. Safetensors manifest inspection reads the bounded index and each header;
   `open_safetensors_checkpoint_mmap` maps each shard, validates upstream views,
   and retains mappings plus offsets. `SafetensorsMmapBackend` constructs a
   metadata-only catalog. Tensor access slices the owning shard mapping and
   returns `Cow::Borrowed`.
4. The smoke's only payload copies are fixed 4 KiB comparison buffers. No call
   to `load_gguf`, `open_checkpoint`, `into_owned`, full-slice hashing, or
   whole-checkpoint `fs::read` occurs. The negative index/TSV reads are small
   harness metadata. Upstream view validation must be re-audited on upgrades.

## Release evidence

Keep a dated report under `docs/smoke/` (link it from the release checklist),
including base revision/patch, compiler versions, fixture revisions, sizes and
hashes, license/provenance, TSVs, exact commands, cache conditions, phase memory,
pass/fail and limitations. Include both first and warm fresh-process runs.
Do not commit checkpoints or generated large artifacts. See
[the initial report](smoke/2026-10-02.md) and
[PR review follow-up](smoke/2026-10-02-review-followup.md).

Lightweight harness checks can be run explicitly with:

```bash
cargo test --locked --all-features --example checkpoint_smoke
```
