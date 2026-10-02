# Coverage and quality gates (RM-1951)

The Linux coverage job runs `cargo llvm-cov --all-targets --all-features --locked
--lcov --output-path coverage.lcov`. It includes default-path GGUF code and the
optional `mmap` and `safetensors` features. No production source path is
excluded to raise the reported percentage. Opt-in real-checkpoint tests remain
ignored in normal CI because model weights are not committed; Windows and macOS
runtime branches are tested in their own CI jobs but are not covered by this
Linux LCOV report.
The two invalid-UTF-8 filename fixtures run on Linux only: macOS rejects
those filenames at file creation, before the parser can exercise the case.

Before the RM-1951 tests, on release tree
`5d5a4d32f453e81cc86b49ce65a532a3570024c5`, the local Linux all-target,
all-feature baseline was **4,451 / 5,337 lines (83.40%)** and **506 / 648
functions (78.09%)**. After adding a malformed second-shard/recovery test,
owned-versus-mmap truncated-GGUF test, shape-overflow contract test, and
upstream-error-category test, the same command covered **4,470 / 5,337 lines
(83.75%)**, 19 more covered lines. The tests also exercise error behavior that
was only weakly asserted before. These are local measurements; Codecov's
hosted values must be verified against the PR and default branch separately.

Codecov is the only coverage-reporting service. The job uploads the LCOV file
as a downloadable GitHub Actions artifact before attempting Codecov upload.
For an internal PR or `main` push, a missing `CODECOV_TOKEN` fails the job;
an uploader error also fails it. A fork PR without that secret records an
explicit warning and retains the artifact without claiming a successful
upload. The coverage job checks out the exact PR head (or push SHA) and passes
that SHA to Codecov. Codecov comments are disabled to avoid duplicate PR
messages; project and patch checks remain visible.

The initial Codecov checks are informational while a reliable `main` baseline
is established. Project coverage compares with the base commit (`target:
auto`) and allows a 2-point regression margin; this leaves room for small
platform/test variance around the measured 83.40% baseline. Patch coverage
targets 80%, rounded below the baseline rather than imposing 100% on new
parser/error paths. Revisit these values after hosted project and patch
reports exist on both a PR and `main`; informational status is not a claim that
coverage is a blocking gate.

Qlty uses `.qlty/qlty.toml` for GitHub Actions `actionlint` and built-in
duplication/complexity smells in comment mode. The GitHub Actions Qlty job
runs those checks without uploading coverage. It excludes generated `target`
and worktree directories; tests and production source remain in scope. Rust
fmt/Clippy stay authoritative for Rust style/lints, and the separate RustSec
workflow owns dependency advisories. The Qlty maintainability badge is shown
in the README; the Qlty coverage badge is intentionally omitted to keep one
coverage dashboard.

The former root `Dockerfile`, `.dockerignore`, and Docker image workflow are
removed. Cursor Cloud Agent files under `.cursor/` remain because
`.cursor/environment.json` explicitly selects that Dockerfile to prepare the
agent's coding environment. Its image and install script use Rust 1.99.0 and
the local install command passes. Removing that setup would cause Cursor to
fall back to an unverified saved environment. This is distinct from building
or publishing a Docker image for the crate in CI.
