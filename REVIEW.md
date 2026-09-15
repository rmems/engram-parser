# Review: quality gate (engram-parser)

Local commands that must pass before merge or PR for this crate.
Aligned with `.github/workflows/ci.yml` and the README Development section.

**Charter:** pure-Rust, **zero-dependency by default** checkpoint parse + MoE
raw expert extract — GGUF v3 plus packed K-quant dequant (Q8_0 / Q5_K / Q6_K /
IQ3_M block). Optional `mmap` feature (#45) adds `memmap2` (`load_gguf_mmap`).
Safetensors **headers** ship behind an off-by-default `safetensors` feature
(#10: manifest + candidate discovery only, no payload mmap, no extra crates).
**No CUDA or GGML compute** in this repo. GGUF on-wire `ggml_type` codes are
labels + packed sizes; wire 31 is `Q4_0_4_4`, not IQ3_M.

| Repo | Role |
|------|------|
| **engram-parser** (this) | GGUF parse + inventory + raw expert bytes + optional mmap/K-quant dequant; safetensors header/manifest/discovery (feature-gated — #10) |
| **myelin-accelerator** | Production CUDA kernels / FFI (`~/Limen-Neural/myelin-accelerator`) |
| **blackwell-kernel-lab** | Scratch GPU experiments / real-model pipelines (`~/rmems/blackwell-kernel-lab`) |

Do **not** add myelin (or CUDA) as a dependency of engram-parser — optional or not.

---

## 1. Full quality gate (copy-paste)

Run from the **repo root** (`engram-parser` checkout, e.g. branch
`feat/gguf-parser-7`). There must be a `Cargo.toml` in the current directory
(`cargo fmt` fails with `could not find Cargo.toml` if you run it elsewhere).

```bash
# From any checkout of this repo (requires Cargo.toml in the tree):
cd "$(git rev-parse --show-toplevel)"

# 0) Ensure rustfmt is installed for this toolchain (once per toolchain)
rustup component add rustfmt
rustup component add clippy   # needed for the next step

# 1a) Apply formatting (rewrites sources; often prints nothing if already clean)
cargo fmt

# 1b) CI-style check (fails with exit 1 + a diff if anything needs format)
cargo fmt --check

# 2) Lint (fail on warnings)
cargo clippy --all-targets --all-features -- -D warnings

# 3) Build
cargo build --all-features

# 4) Tests (unit + integration + doctests)
cargo test --all-features

# 5) Clean-tree guard (matches CI after build/test)
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree dirty after gate — unexpected artifacts or uncommitted edits:"
  git status --short
  exit 1
fi
echo "Working tree clean"
```

**Pass criteria:** all steps exit 0; `cargo test` reports 0 failed; tree clean
after you commit any files that `cargo fmt` rewrote.

Optional one-liner (check-only; does not rewrite):

```bash
cd ~/rmems/engram-parser && \
  cargo fmt --check && \
  cargo clippy --all-targets --all-features -- -D warnings && \
  cargo build --all-features && \
  cargo test --all-features && \
  test -z "$(git status --porcelain)" && echo "QUALITY GATE PASS"
```

### `cargo fmt` notes (common “doesn’t work” cases)

| What you run | Expected behavior |
|--------------|-------------------|
| `cargo fmt` | **Applies** rustfmt. Exit 0 and **no stdout** when sources are already formatted — that is success, not a no-op bug. |
| `cargo fmt --check` | **Does not write**. Exit 0 if clean; exit 1 and prints a diff if not. This is what CI runs. |
| `cargo fmt -v` | Verbose: shows which crate roots rustfmt visits (`src/lib.rs`, `tests/*.rs`). Nested modules under `src/gguf/`, `src/moe/` are formatted via the module tree. |

**Install / toolchain fixes:**

```bash
# Active toolchain
rustc --version
rustup show

# Install rustfmt if cargo fmt says it is missing
rustup component add rustfmt
# or pin explicitly:
rustup component add rustfmt --toolchain stable
rustup component add rustfmt --toolchain 1.97.1   # for MSRV checks

# Confirm the binary cargo will call
cargo fmt --version
# → rustfmt x.y.z-stable (...)
```

This repo ships [`rust-toolchain.toml`](rust-toolchain.toml) (`channel = "stable"`),
so `rustup` / `cargo` in this directory use latest stable automatically.

**Typical errors:**

| Symptom | Fix |
|---------|-----|
| `could not find Cargo.toml` | `cd` into `engram-parser` first |
| `'cargo-fmt' is not installed` / missing rustfmt | `rustup component add rustfmt` |
| Wrong toolchain (old rustfmt, edition 2024 issues) | Use stable ≥ MSRV **1.97.1**: `rustup update stable` or `cargo +stable fmt` |
| “Nothing happened” after `cargo fmt` | Tree was already formatted; use `cargo fmt --check` (expect exit 0) or `cargo fmt -v` |
| `--check` prints diffs | Run `cargo fmt` (no `--check`) once, then commit |

There is no `rustfmt.toml` in this repo; defaults are fine.

## 2. Gate table

| Step | Command | What it proves |
|------|---------|----------------|
| `fmt` (apply) | `cargo fmt` | Rewrites sources to rustfmt style (silent if already clean) |
| `fmt` (CI) | `cargo fmt --check` | Style matches rustfmt; fails with a diff if not |
| `clippy` | `cargo clippy --all-targets --all-features -- -D warnings` | No Clippy warnings on lib + tests |
| `build` (all features) | `cargo build --all-features` | Crate builds with every declared feature; `[features]` is still empty, so today this equals the default build. Becomes meaningful when `safetensors` lands (#10) |
| `build` (default) | `cargo build` | **Default build stays GGUF-only and dep-free** — will matter once `safetensors` exists; not yet a CI job, see #10 |
| `test` | `cargo test --all-features` | Unit (`src/gguf/tensor.rs`), smoke (`tests/gguf_smoke.rs`), doctests |
| `clean-tree` | `git status --porcelain` empty | No stray outputs after build/test |
| `coverage` (opt) | `cargo llvm-cov --all-targets --all-features --locked --lcov --output-path lcov.info` | LCOV for Codecov (CI installs `cargo-llvm-cov`) |
| `msrv` (opt) | toolchain **1.97.1** + same fmt/clippy/build/test | Matches `rust-version` / CI `msrv` job |
| `docker` (opt) | `docker build -t engram-parser .` then `docker run --rm engram-parser` | Image uses `RUST_VERSION=1.97.1` |

### Coverage (local)

`llvm-cov` is a **cargo subcommand**, not a cargo flag. The space is required.

```bash
# WRONG — cargo parses "-llvm-cov" as options → unexpected argument '-l'
#   cargo -llvm-cov
#   $HOME/.cargo/bin/cargo -llvm-cov

# once per machine (installs ~/.cargo/bin/cargo-llvm-cov)
cargo install cargo-llvm-cov --locked

# also need llvm-tools on the active toolchain (rust-toolchain.toml already lists it)
rustup component add llvm-tools-preview

# correct: subcommand after cargo (same as CI)
cd ~/rmems/engram-parser
cargo llvm-cov --all-targets --all-features --locked --lcov --output-path lcov.info

# human-readable summary only (no lcov file)
cargo llvm-cov --all-targets --all-features --locked
```

If you get `no such command: llvm-cov`, the binary is missing or not on `PATH`:

```bash
which cargo-llvm-cov || cargo install cargo-llvm-cov --locked
export PATH="$HOME/.cargo/bin:$PATH"
cargo llvm-cov --version
```

`lcov.info` is a local artifact; do not commit it.

### MSRV (local)

**You must install the MSRV toolchain first.** If you skip this, you get:

```text
error: toolchain '1.97.1-x86_64-unknown-linux-gnu' is not installed
help: run `rustup toolchain install 1.97.1` ...
```

**One-time setup:**

```bash
# Install the exact MSRV pin (matches Cargo.toml rust-version / CI msrv job)
rustup toolchain install 1.97.1 --component rustfmt,clippy

# Confirm cargo can see it (must print 1.97.1)
cargo +1.97.1 -V
rustc +1.97.1 -V
```

**Then build/test on MSRV** (from repo root):

```bash
cd ~/rmems/engram-parser

# Preferred: explicit +toolchain (overrides rust-toolchain.toml for this command)
cargo +1.97.1 fmt --check
cargo +1.97.1 clippy --all-targets --all-features -- -D warnings
cargo +1.97.1 build --all-features
cargo +1.97.1 test --all-features
```

**Alternatives if `+1.97.1` is awkward in an IDE/script:**

```bash
# Same effect via env (also overrides directory rust-toolchain.toml)
RUSTUP_TOOLCHAIN=1.97.1 cargo build --all-features
RUSTUP_TOOLCHAIN=1.97.1 cargo test --all-features

# Or rustup run
rustup run 1.97.1 cargo test --all-features
```

**When MSRV == current stable (today: both 1.97.x):** plain `cargo build` /
`cargo test` already use stable via `rust-toolchain.toml` and are enough for
day-to-day work. Use `+1.97.1` only when you want an explicit MSRV gate matching
the CI `msrv` job.

| Symptom | Fix |
|---------|-----|
| `toolchain '1.97.1' is not installed` | `rustup toolchain install 1.97.1 --component rustfmt,clippy` |
| `+1.97.1` ignored / still wrong version | Prefer `cargo +1.97.1 -V` to verify; or `RUSTUP_TOOLCHAIN=1.97.1` |
| `clippy-driver` / rustfmt missing on 1.97.1 | `rustup component add clippy rustfmt --toolchain 1.97.1` |
| IDE “Cargo” has no `+1.97.1` | Set env `RUSTUP_TOOLCHAIN=1.97.1` in the run config, or use the terminal |

---

## 3. What `cargo test` covers (this branch)

| Surface | Location | Focus |
|---------|----------|--------|
| Unit | `src/gguf/tensor.rs` | `DType`, IQ/Q block `byte_len`, wire **31 = Q4_0_4_4** (not IQ3_M), labels |
| Integration | `tests/gguf_smoke.rs` | Synthetic GGUF parse, stacked/per-expert MoE extract, Q8_0/Q4_K slices, `file_type` quant fallback, bad magic/version/truncation |
| Pilot (ignored) | `tests/real_gguf.rs` | Real weights via `ENGRAM_GGUF` / `ENGRAM_MODEL_DIR` (xai-dissect pilots) |
| Example | `examples/inspect_gguf.rs` | Human inventory of one real GGUF |
| Always-on contract | `real_gguf_helpers_document_env` | Env names + empty pilot list when unset |
| Doctests | `src/lib.rs`, `ggml_type_label` | Public API examples compile |

There is **no** `benches/` or Criterion target. Do **not** use `cargo bench`
as a quality gate for this crate.

---

## 4. Test tiers (xai-dissect pattern)

Same split as `~/rmems/xai-dissect`: **always-on fixtures in CI**, **path-gated
pilots on real weights locally**.

| Tier | What | Command | CI? |
|------|------|---------|-----|
| **T0** | Synthetic GGUF builders | `cargo test --all-features` | Yes |
| **T1** | Real `.gguf` inventory + MoE extract | `ENGRAM_GGUF=… cargo test --test real_gguf -- --ignored` | No |
| **T2** | GPU kernels / Nsight / experiments | **blackwell-kernel-lab** or myelin | No |

### T1 — real GGUF pilots (this repo, CPU only)

```bash
cd ~/rmems/engram-parser

# Single file (any dense or MoE GGUF under ~/.models, ollama export, etc.)
ENGRAM_GGUF=~/.models/gguf/Abiray/ZAYA1-8B-GGUF/ZAYA1-8B-Q8_0.gguf \
  cargo test --test real_gguf -- --ignored --nocapture

# Scan a tree (depth-limited; cap with ENGRAM_GGUF_MAX, default 1)
ENGRAM_MODEL_DIR=~/.models/gguf ENGRAM_GGUF_MAX=3 \
  cargo test --test real_gguf -- --ignored --nocapture

# Human-readable inventory (not a test)
cargo run --example inspect_gguf -- ~/.models/gguf/.../model.gguf
# or: ENGRAM_GGUF=... cargo run --example inspect_gguf
```

Env vars:

| Var | Meaning |
|-----|---------|
| `ENGRAM_GGUF` | One `.gguf` path (wins over dir scan) |
| `ENGRAM_MODEL_DIR` | Root to walk for `*.gguf` |
| `ENGRAM_GGUF_MAX` | Max files when scanning (default 1) |
| `ENGRAM_EXPECT_MOE` | `1`/`true` → fail if no expert pairs discovered |
| `ENGRAM_MOE_SAMPLES` | Number of `(block,expert)` pairs to extract (default 1) |

**Pass criteria (T1):** `load_gguf` ok; tensors non-empty; each `tensor_bytes`
in-range; when MoE names exist, `list_experts` + `extract_expert` return
non-empty projections. Dense GGUFs may skip MoE (inventory-only is fine).

Do **not** commit multi-GB weights. Prefer `~/.models/gguf/…` over scraping
`~/.ollama` blobs (export / copy to a real `.gguf` path first).

### T1 large MoE (local only — RAM-bound)

`load_gguf` reads the **entire** file into memory (no mmap). Run **one**
`ENGRAM_GGUF` path per process. Do not scan a tree of multi-GB files (each
ignored test loads the file again — peak RSS ≈ 2× file size for inventory +
MoE tests in one `cargo test` invocation).

| Model | Path (this machine) | Size | Min free RAM |
|-------|---------------------|------|--------------|
| jina F16 (smoke) | `~/.models/jinaai/…/v5-nano-text-matching-F16.gguf` | ~0.4 GiB | ≥ 2 GiB |
| ZAYA1-8B Q8_0 | `~/.models/gguf/Abiray/ZAYA1-8B-GGUF/ZAYA1-8B-Q8_0.gguf` | ~8.83 GiB | ≥ 12 GiB available |
| OLMoE-1B-7B F16 | `~/.models/gguf/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/OLMoE-1B-7B-0125-Instruct-F16.gguf` (symlink → Downloads) | ~12.89 GiB | ≥ 18 GiB available |

More GGUFs exist under `~/.models/gguf/` and `~/Downloads/SNN_Quantization/`
(Qwen3-MoE, DeepSeek-Coder-V2 Lite, Gemma-4 A4B, Kimi-VL, …). Optional P1
pilots — still one path per process.

```bash
free -h

ENGRAM_GGUF=~/.models/gguf/Abiray/ZAYA1-8B-GGUF/ZAYA1-8B-Q8_0.gguf \
  ENGRAM_EXPECT_MOE=1 ENGRAM_MOE_SAMPLES=3 \
  cargo test --release --test real_gguf -- --ignored --nocapture

ENGRAM_GGUF=~/.models/gguf/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/OLMoE-1B-7B-0125-Instruct-F16.gguf \
  ENGRAM_EXPECT_MOE=1 ENGRAM_MOE_SAMPLES=5 \
  cargo test --release --test real_gguf -- --ignored --nocapture
```

**Proven on this host (2026-08-02):**

| Pilot | tensors | moe pairs | samples | max RSS (test) |
|-------|---------|-----------|---------|----------------|
| ZAYA1 Q8 | 1283 | 640 | 3 OK (`complete=false` partial roles) | ~17.7 GiB |
| OLMoE F16 | 195 | 1024 | 5 OK (`complete=true`, stacked) | ~25.8 GiB |

### T2 — GPU experiments: use blackwell-kernel-lab

Yes — **run real-model GPU tests and experiments in
`~/rmems/blackwell-kernel-lab`**, not inside engram-parser.

Suggested ownership:

| Concern | Where |
|---------|--------|
| Parse / MoE raw extract correctness | engram-parser T0 + T1 |
| Scratch CUDA kernels, pipeline prototypes, Nsight | **blackwell-kernel-lab** |
| Stable Blackwell kernels / FFI | myelin-accelerator |

blackwell-kernel-lab can depend on engram-parser **and** myelin (path deps).
engram-parser never depends on either.

Production-ish GPU gate (still not engram CI):

```bash
cd ~/Limen-Neural/myelin-accelerator
export CUDA_NVCC=/usr/local/cuda/bin/nvcc
cargo test --locked --features cuda -- --ignored --nocapture
cargo run --locked --example benchmark --profile bench --features bench,cuda
```

---

## 5. Out of scope for engram quality gate

| Item | Where it belongs |
|------|------------------|
| CUDA / PTX / Nsight / real-model GPU benches | **blackwell-kernel-lab** (experiments) or myelin-accelerator |
| Row dequant / mmap host load | corinth-canal (reference) or downstream |
| Safetensors **payload** mmap load + HF `config.json` | corinth-canal (reference) — only the header/manifest/discovery half is ported here, see #10 |
| Routing / MoE matmul / generation | cortex-tensor / hybrid stack |
| Optional myelin dep on this crate | **Never** — keeps zero-dep charter |

---

## 6. CI mapping

| Local step | Workflow job |
|------------|----------------|
| fmt, clippy, build, test (T0 only), clean-tree, llvm-cov | `validate` in `.github/workflows/ci.yml` (**stable** = latest) |
| MSRV 1.97.1 fmt/clippy/build/test | `msrv` in `.github/workflows/ci.yml` (pinned `toolchain: "1.97.1"`) |
| Security audit / Snyk | `.github/workflows/security.yml` (not required for every local edit) |
| Docker image | `Dockerfile` (`ARG RUST_VERSION=1.97.1`) + `.github/workflows/docker-build.yml` |
| T1 real GGUF / T2 GPU | **Not in CI** — local pilots only |
| Default-feature build (`cargo build`, no features) | **No workflow job yet** — every feature-sensitive CI step (`clippy`, `build`, `test`, `llvm-cov`) passes `--all-features`; only `cargo fmt --check` is feature-agnostic. Once `safetensors` lands (#10), a `#[cfg(feature = "safetensors")]` mistake that breaks the default build would ship green |

**Yes, the GitHub workflow is part of a Rust version bump:** keep `validate` on
`stable` (auto-tracks latest), and update the `msrv` job + `Cargo.toml`
`rust-version` + Docker tag together whenever you raise the floor.

---

## 7. `.gitignore` note

Local tool dirs (`.claude/`, `.opencode/`, `docs/superpowers/`, etc.),
`/target/`, and env files are ignored. Quality-gate commands should not
create tracked files; if `git status` is dirty after the gate, fix the
cause before merge (or ensure only intentional source edits are staged).
Do not commit `lcov.info`, large GGUFs, or GPU profile dumps into this repo.
