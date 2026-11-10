// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pinned real-model fixtures with a cached, verified fetch helper (RM-2367).
//!
//! Nothing here is committed as a binary. Each [`RealFixture`] names one model
//! in one format: a Hugging Face repo at a **commit revision**, plus every
//! file of that model the tests need, each pinned by **SHA-256** and exact
//! size. The helper downloads the files once into a cache directory, verifies
//! the digests, and reuses them on later runs. Hugging Face is the only
//! default origin and only `curl` is required, so the crate stays
//! dependency-free.
//!
//! # Environment
//!
//! | Variable | Meaning |
//! |---|---|
//! | `ENGRAM_REAL_FIXTURES` | `auto` (default): use the root/cache, download when missing, skip if the network is unavailable. `offline`: local files only, skip when missing. `skip`: skip every real-fixture test. `require`: succeed or fail — never skip. |
//! | `ENGRAM_FIXTURE_ROOT` | Directory of pre-downloaded models laid out as `<root>/<fixture id>/<files>` (for example `C:\models` from the RM-2367 download script). Checked before the cache; files are verified against the same pins. |
//! | `ENGRAM_FIXTURE_CACHE` | Download cache directory. Default: `<target>/tmp/engram-real-fixtures`. |
//! | `ENGRAM_FIXTURE_BASE_URL` | Origin replacing `https://huggingface.co` (mirror, or a `file://` tree). |
//!
//! A digest mismatch is never skipped: it means the pin or the bytes are
//! wrong. A root directory that is only partially populated falls back to the
//! normal cache/download path rather than using an incomplete model.

#![allow(dead_code)]

use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

pub const ENV_MODE: &str = "ENGRAM_REAL_FIXTURES";
pub const ENV_ROOT: &str = "ENGRAM_FIXTURE_ROOT";
pub const ENV_CACHE: &str = "ENGRAM_FIXTURE_CACHE";
pub const ENV_BASE_URL: &str = "ENGRAM_FIXTURE_BASE_URL";
pub const DEFAULT_BASE_URL: &str = "https://huggingface.co";

/// One file of a pinned fixture.
#[derive(Debug, Clone, Copy)]
pub struct FixtureFile {
    /// Path of the file inside the repo (flat names only).
    pub file: &'static str,
    /// Lowercase hex SHA-256 of the file.
    pub sha256: &'static str,
    /// Exact size in bytes.
    pub size: u64,
}

impl FixtureFile {
    pub fn file_name(&self) -> &str {
        self.file.rsplit('/').next().unwrap_or(self.file)
    }

    pub fn is_weights(&self) -> bool {
        self.file.ends_with(".gguf") || self.file.ends_with(".safetensors")
    }
}

/// One model in one format, pinned to a Hugging Face commit.
#[derive(Debug, Clone, Copy)]
pub struct RealFixture {
    /// Short stable label used in logs and `<root>/<id>` lookups.
    pub id: &'static str,
    /// Repo id, e.g. `Qwen/Qwen3-0.6B-GGUF`.
    pub repo: &'static str,
    /// Full 40-hex commit the files are pinned to (never a branch name).
    pub revision: &'static str,
    /// License declared by the repo's model card.
    pub license: &'static str,
    /// Weight file(s) plus the sidecars the tests read (`config.json`,
    /// shard index). Downloaded and verified together.
    pub files: &'static [FixtureFile],
}

const fn file(file: &'static str, sha256: &'static str, size: u64) -> FixtureFile {
    FixtureFile { file, sha256, size }
}

/// Qwen3-0.6B dense baseline, GGUF Q8_0 (llama.cpp `blk.N` naming).
pub const QWEN3_GGUF: RealFixture = RealFixture {
    id: "qwen3-0.6b-gguf",
    repo: "Qwen/Qwen3-0.6B-GGUF",
    revision: "23749fefcc72300e3a2ad315e1317431b06b590a",
    license: "apache-2.0",
    files: &[file(
        "Qwen3-0.6B-Q8_0.gguf",
        "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
        639_446_688,
    )],
};

/// The same dense model as Safetensors (BF16) plus its `config.json`.
pub const QWEN3_SAFETENSORS: RealFixture = RealFixture {
    id: "qwen3-0.6b",
    repo: "Qwen/Qwen3-0.6B",
    revision: "c1899de289a04d12100db370d81485cdf75e47ca",
    license: "apache-2.0",
    files: &[
        file(
            "model.safetensors",
            "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
            1_503_300_328,
        ),
        file(
            "config.json",
            "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
            726,
        ),
    ],
};

/// Qwen2.5-0.5B-Instruct dense baseline (older layout, QKV biases), GGUF Q8_0.
pub const QWEN25_GGUF: RealFixture = RealFixture {
    id: "qwen2.5-0.5b-instruct-gguf",
    repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    revision: "9217f5db79a29953eb74d5343926648285ec7e67",
    license: "apache-2.0",
    files: &[file(
        "qwen2.5-0.5b-instruct-q8_0.gguf",
        "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        675_710_816,
    )],
};

/// The same dense model as Safetensors (BF16) plus its `config.json`.
pub const QWEN25_SAFETENSORS: RealFixture = RealFixture {
    id: "qwen2.5-0.5b-instruct",
    repo: "Qwen/Qwen2.5-0.5B-Instruct",
    revision: "7ae557604adf67be50417f59c2c2f167def9a775",
    license: "apache-2.0",
    files: &[
        file(
            "model.safetensors",
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
            988_097_824,
        ),
        file(
            "config.json",
            "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
            659,
        ),
    ],
};

/// OLMoE-1B-7B-0125-Instruct MoE baseline, GGUF Q8_0 (stacked `*_exps`).
pub const OLMOE_GGUF: RealFixture = RealFixture {
    id: "olmoe-0125-instruct-gguf",
    repo: "allenai/OLMoE-1B-7B-0125-Instruct-GGUF",
    revision: "2ac1d27317927518c9ef7bd99f91f3f1e4ee288d",
    license: "apache-2.0",
    files: &[file(
        "OLMoE-1B-7B-0125-Instruct-Q8_0.gguf",
        "d4e5c4f4925fc3b685b77f6f04a2647d964b768127d329a7cd421ca8e8259f39",
        7_359_942_656,
    )],
};

/// The same MoE as three Safetensors shards, the shard index, and
/// `config.json` (per-expert `gate_proj`/`up_proj`/`down_proj`, BF16).
pub const OLMOE_SAFETENSORS: RealFixture = RealFixture {
    id: "olmoe-0125-instruct",
    repo: "allenai/OLMoE-1B-7B-0125-Instruct",
    revision: "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e",
    license: "apache-2.0",
    files: &[
        file(
            "model-00001-of-00003.safetensors",
            "61874210ca7c360f43f8c622cecc12441083d40190eae3b56bc9d6e1c0a30c1e",
            4_997_744_872,
        ),
        file(
            "model-00002-of-00003.safetensors",
            "c523a43b8a17269d5fab33395048a83633f4d1d89c1958570cea738e2bbe80c9",
            4_997_235_176,
        ),
        file(
            "model-00003-of-00003.safetensors",
            "97ae01e3519c52e63a018bca96ab17a89c4cd5cab1c6d742efed0fa5c0e2bb17",
            3_843_741_912,
        ),
        file(
            "model.safetensors.index.json",
            "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
            287_214,
        ),
        file(
            "config.json",
            "272998dd7ba4846dcc682f0b5a46144f4bcd9dde8e94d2f17bd8e5cf2f23d6ce",
            828,
        ),
    ],
};

/// Qwen3.5-0.8B stretch fixture (linear-attention hybrid + vision tower),
/// GGUF Q8_0 — the text tower only; llama.cpp ships `mmproj` separately.
pub const QWEN35_GGUF: RealFixture = RealFixture {
    id: "qwen3.5-0.8b-gguf",
    repo: "unsloth/Qwen3.5-0.8B-GGUF",
    revision: "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
    license: "apache-2.0",
    files: &[file(
        "Qwen3.5-0.8B-Q8_0.gguf",
        "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c",
        811_843_840,
    )],
};

/// The same model as Safetensors (BF16): one shard, the index, and the
/// multimodal `config.json` (`text_config` / `vision_config`).
pub const QWEN35_SAFETENSORS: RealFixture = RealFixture {
    id: "qwen3.5-0.8b",
    repo: "Qwen/Qwen3.5-0.8B",
    revision: "2fc06364715b967f1860aea9cf38778875588b17",
    license: "apache-2.0",
    files: &[
        file(
            "model.safetensors-00001-of-00001.safetensors",
            "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
            1_746_942_600,
        ),
        file(
            "model.safetensors.index.json",
            "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
            50_900,
        ),
        file(
            "config.json",
            "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
            2_907,
        ),
    ],
};

/// Every pinned fixture (RM-2367 required + Qwen3.5 stretch).
pub const ALL: &[RealFixture] = &[
    QWEN3_GGUF,
    QWEN3_SAFETENSORS,
    QWEN25_GGUF,
    QWEN25_SAFETENSORS,
    OLMOE_GGUF,
    OLMOE_SAFETENSORS,
    QWEN35_GGUF,
    QWEN35_SAFETENSORS,
];

impl RealFixture {
    /// `<base>/<repo>/resolve/<revision>/<file>`.
    pub fn url(&self, base: &str, file: &FixtureFile) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            base.trim_end_matches('/'),
            self.repo,
            self.revision,
            file.file
        )
    }

    /// Directory holding all of this fixture's files under `root`.
    pub fn fixture_dir(&self, root: &Path) -> PathBuf {
        root.join(self.id)
    }

    /// What to hand to `open_checkpoint`: the shard directory when the model
    /// is multi-file or carries an index, otherwise the single weight file.
    pub fn checkpoint_path(&self, dir: &Path) -> PathBuf {
        let weights: Vec<_> = self.files.iter().filter(|f| f.is_weights()).collect();
        let has_index = self.files.iter().any(|f| f.file.ends_with(".index.json"));
        if weights.len() == 1 && !has_index {
            dir.join(weights[0].file_name())
        } else {
            dir.to_path_buf()
        }
    }

    /// One pinned file's path inside a fixture directory.
    pub fn file_path(&self, dir: &Path, file: &str) -> PathBuf {
        let name = file.rsplit('/').next().unwrap_or(file);
        dir.join(name)
    }
}

/// How the helper treats local files and the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Auto,
    Offline,
    Skip,
    Require,
}

impl Mode {
    /// Parse `ENGRAM_REAL_FIXTURES`. Unknown values are an error so a typo
    /// cannot silently turn the tests into no-ops.
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        let Some(raw) = raw else {
            return Ok(Self::Auto);
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "offline" | "cached" => Ok(Self::Offline),
            "skip" | "off" | "0" | "false" | "no" => Ok(Self::Skip),
            "require" | "required" => Ok(Self::Require),
            other => Err(format!(
                "{ENV_MODE}={other:?}: expected auto, offline, skip, or require"
            )),
        }
    }
}

/// Resolved helper configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub mode: Mode,
    /// Pre-downloaded `<root>/<id>/` directories, checked first.
    pub fixture_root: Option<PathBuf>,
    pub cache_dir: PathBuf,
    pub base_url: String,
}

impl Config {
    pub fn from_env() -> Self {
        let mode =
            Mode::parse(env::var(ENV_MODE).ok().as_deref()).unwrap_or_else(|e| panic!("{e}"));
        let fixture_root = env::var_os(ENV_ROOT)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let cache_dir = env::var_os(ENV_CACHE)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("engram-real-fixtures"));
        let base_url = env::var(ENV_BASE_URL)
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        Self {
            mode,
            fixture_root,
            cache_dir,
            base_url,
        }
    }
}

/// Result of [`ensure`].
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Verified fixture directory on disk (all files of the fixture inside).
    Ready(PathBuf),
    /// Not available, with the reason. Never returned in [`Mode::Require`].
    Skipped(String),
}

/// Serializes fixture resolution: tests in one binary run on parallel
/// threads and would otherwise download or re-hash the same files twice.
/// Repeat calls are cheap even without memoizing: the stamp files make
/// re-verification a size/mtime/stat check rather than another full hash.
static FETCH_LOCK: Mutex<()> = Mutex::new(());

/// Return the verified fixture directory of `fx`, downloading it if the
/// mode allows.
///
/// # Panics
///
/// On a SHA-256 or size mismatch (always), and on any failure in
/// [`Mode::Require`].
pub fn ensure(fx: &'static RealFixture, cfg: &Config) -> Outcome {
    let _guard = FETCH_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    match ensure_locked(fx, cfg) {
        Outcome::Skipped(why) if cfg.mode == Mode::Require => {
            panic!("{} unavailable but {ENV_MODE}=require: {why}", fx.id)
        }
        other => other,
    }
}

fn ensure_locked(fx: &RealFixture, cfg: &Config) -> Outcome {
    if cfg.mode == Mode::Skip {
        return Outcome::Skipped(format!("{ENV_MODE}=skip"));
    }

    // 1. A pre-staged `<root>/<id>` directory wins when it is complete.
    if let Some(root) = &cfg.fixture_root {
        let dir = fx.fixture_dir(root);
        if dir.is_dir() {
            match verify_fixture_dir(fx, &dir, &stamps_dir(cfg)) {
                Ok(()) => return Outcome::Ready(dir),
                Err(DirState::Incomplete(missing)) => {
                    eprintln!(
                        "{}: {ENV_ROOT} entry misses {missing:?}; trying cache",
                        fx.id
                    );
                }
                // A staged file that fails the pin must not be silently
                // replaced: it is the user's tree, not a disposable cache.
                Err(DirState::Corrupt { why, .. }) => panic!("{}: {why}", fx.id),
            }
        }
    }

    // 2. The download cache. A corrupted entry is disposable: drop the bad
    // file and let the download path refill it.
    let dir = fx.fixture_dir(&cfg.cache_dir);
    match verify_fixture_dir(fx, &dir, &stamps_dir(cfg)) {
        Ok(()) => return Outcome::Ready(dir),
        Err(DirState::Incomplete(_)) => {}
        Err(DirState::Corrupt { path, why }) => {
            eprintln!("{}: dropping bad cache entry ({why})", fx.id);
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(stamp_path(&stamps_dir(cfg), &path));
        }
    }

    if cfg.mode == Mode::Offline {
        return Outcome::Skipped(format!("not cached and {ENV_MODE}=offline"));
    }
    match download(fx, cfg, &dir) {
        Ok(()) => Outcome::Ready(dir),
        Err(DownloadError::Unavailable(why)) => Outcome::Skipped(why),
        Err(DownloadError::Corrupt(why)) => panic!("{}: {why}", fx.id),
    }
}

/// Stamps for already-hashed files live under the cache dir even when the
/// files themselves come from `ENGRAM_FIXTURE_ROOT` (never write into the
/// user's model tree).
fn stamps_dir(cfg: &Config) -> PathBuf {
    cfg.cache_dir.join(".stamps")
}

enum DirState {
    /// Some pinned files are absent: fall back to another source.
    Incomplete(Vec<&'static str>),
    /// A file exists but fails the pin: never silently tolerated.
    Corrupt { path: PathBuf, why: String },
}

fn verify_fixture_dir(fx: &RealFixture, dir: &Path, stamps: &Path) -> Result<(), DirState> {
    let mut missing = Vec::new();
    for f in fx.files {
        let path = dir.join(f.file_name());
        if !path.is_file() {
            missing.push(f.file);
            continue;
        }
        verify_file(f, &path, stamps).map_err(|why| DirState::Corrupt { path, why })?;
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(DirState::Incomplete(missing))
    }
}

enum DownloadError {
    /// No network, no `curl`, or the origin refused: tests may skip.
    Unavailable(String),
    /// Bytes arrived but do not match the pin: never skipped.
    Corrupt(String),
}

fn stamp_path(stamps: &Path, path: &Path) -> PathBuf {
    // Keyed by the file's absolute path so one stamp store serves both the
    // cache and an ENGRAM_FIXTURE_ROOT tree.
    stamps.join(sha256_hex(
        path.to_string_lossy().replace('\\', "/").as_bytes(),
    ))
}

/// `<sha256> <len> <mtime_ns>` of the file as it was when last hashed.
fn stamp_for(f: &FixtureFile, path: &Path) -> io::Result<String> {
    let meta = fs::metadata(path)?;
    let mtime = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(format!("{} {} {mtime}", f.sha256, meta.len()))
}

/// Trust a file when its size, mtime, and a stamp written after a full hash
/// all agree; otherwise re-hash it.
fn verify_file(f: &FixtureFile, path: &Path, stamps: &Path) -> Result<(), String> {
    let len = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len != f.size {
        return Err(format!("{}: size {len} != pinned {}", f.file, f.size));
    }
    let want = stamp_for(f, path).map_err(|e| e.to_string())?;
    let stamp = stamp_path(stamps, path);
    if fs::read_to_string(&stamp).is_ok_and(|s| s.trim() == want) {
        return Ok(());
    }
    let got = sha256_file(path).map_err(|e| e.to_string())?;
    if got != f.sha256 {
        return Err(format!("{}: sha256 {got} != pinned {}", f.file, f.sha256));
    }
    let _ = fs::create_dir_all(stamps);
    let _ = fs::write(&stamp, want);
    Ok(())
}

fn download(fx: &RealFixture, cfg: &Config, dir: &Path) -> Result<(), DownloadError> {
    let unavailable = |why: String| DownloadError::Unavailable(why);
    fs::create_dir_all(dir).map_err(|e| unavailable(format!("create {}: {e}", dir.display())))?;

    for f in fx.files {
        let dest = dir.join(f.file_name());
        let part = dir.join(format!("{}.{}.part", f.file_name(), std::process::id()));
        let url = fx.url(&cfg.base_url, f);
        eprintln!("{}: downloading {url} ({} bytes)", fx.id, f.size);

        let status = Command::new("curl")
            .args(["--fail", "--location", "--silent", "--show-error"])
            .args([
                "--retry",
                "2",
                "--connect-timeout",
                "20",
                "--max-time",
                "1800",
            ])
            .arg("--output")
            .arg(&part)
            .arg(&url)
            .status();
        let result = match status {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(unavailable(format!("curl exited with {s} for {url}"))),
            Err(e) => Err(unavailable(format!("cannot run curl: {e}"))),
        };
        if let Err(e) = result {
            let _ = fs::remove_file(&part);
            return Err(e);
        }

        if let Err(why) = check_bytes(f, &part) {
            let _ = fs::remove_file(&part);
            return Err(DownloadError::Corrupt(why));
        }
        fs::rename(&part, &dest).map_err(|e| {
            let _ = fs::remove_file(&part);
            unavailable(format!("move into cache: {e}"))
        })?;
    }

    // Everything downloaded: hash the finished set once (fills stamps).
    verify_fixture_dir(fx, dir, &stamps_dir(cfg)).map_err(|e| match e {
        DirState::Incomplete(missing) => {
            DownloadError::Corrupt(format!("download incomplete: {missing:?}"))
        }
        DirState::Corrupt { why, .. } => DownloadError::Corrupt(why),
    })
}

fn check_bytes(f: &FixtureFile, path: &Path) -> Result<(), String> {
    let len = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len != f.size {
        return Err(format!("downloaded {len} bytes, pinned size is {}", f.size));
    }
    let got = sha256_file(path).map_err(|e| e.to_string())?;
    if got != f.sha256 {
        return Err(format!(
            "downloaded sha256 {got} does not match pinned {}",
            f.sha256
        ));
    }
    Ok(())
}

/// Verified fixture directory for `fx`, or `None` (after logging why) when
/// it must be skipped. Call at the top of a real-fixture test and `return`
/// on `None`.
pub fn fixture_or_skip(fx: &'static RealFixture) -> Option<PathBuf> {
    match ensure(fx, &Config::from_env()) {
        Outcome::Ready(dir) => Some(dir),
        Outcome::Skipped(why) => {
            eprintln!("SKIP {} ({}): {why}", fx.id, fx.repo);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Local so the test tree needs no hashing crate.
// ---------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            filled: 0,
            total: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.filled > 0 {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < 64 {
                return;
            }
            let block = self.block;
            self.compress(&block);
            self.filled = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for block in blocks {
            self.compress(block);
        }
        self.block[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        let used = (self.filled + 1) % 64;
        pad.resize(1 + (64 + 56 - used) % 64, 0);
        pad.extend_from_slice(&bit_len.to_be_bytes());
        self.update(&pad);
        debug_assert_eq!(self.filled, 0);
        let mut out = [0u8; 32];
        for (dst, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *dst = word.to_be_bytes();
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (slot, word) in w.iter_mut().zip(block.as_chunks::<4>().0) {
            *slot = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    to_hex(&h.finalize())
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex(&hasher.finalize()))
}
