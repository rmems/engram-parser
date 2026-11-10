// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pinned real-model fixtures with a cached, verified fetch helper (RM-2367).
//!
//! Nothing here is committed as a binary. Each [`RealFixture`] names a file in
//! a Hugging Face model repo by **commit revision** and **SHA-256**; the helper
//! downloads it once into a cache directory, verifies the digest, and reuses it
//! on later runs. Hugging Face is the only default origin and only `curl` is
//! required, so the crate stays dependency-free.
//!
//! # Environment
//!
//! | Variable | Meaning |
//! |---|---|
//! | `ENGRAM_REAL_FIXTURES` | `auto` (default): use the cache, download when missing, skip if the network is unavailable. `offline`: cache only, skip when missing. `skip`: skip every real-fixture test. `require`: download, and fail instead of skipping on any problem. |
//! | `ENGRAM_FIXTURE_CACHE` | Cache directory. Default: `<target>/tmp/engram-real-fixtures`. |
//! | `ENGRAM_FIXTURE_BASE_URL` | Origin replacing `https://huggingface.co` (mirror, or a `file://` tree). |
//!
//! A digest mismatch is never skipped: it means the pin or the bytes are wrong.

#![allow(dead_code)]

use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

pub const ENV_MODE: &str = "ENGRAM_REAL_FIXTURES";
pub const ENV_CACHE: &str = "ENGRAM_FIXTURE_CACHE";
pub const ENV_BASE_URL: &str = "ENGRAM_FIXTURE_BASE_URL";
pub const DEFAULT_BASE_URL: &str = "https://huggingface.co";

/// One pinned file of a Hugging Face model repo.
#[derive(Debug, Clone, Copy)]
pub struct RealFixture {
    /// Short stable label used in logs and cache messages.
    pub id: &'static str,
    /// Repo id, e.g. `ggml-org/stories15M_MOE`.
    pub repo: &'static str,
    /// Full 40-hex commit the file is pinned to (never a branch name).
    pub revision: &'static str,
    /// Path of the file inside the repo.
    pub file: &'static str,
    /// Lowercase hex SHA-256 of the file.
    pub sha256: &'static str,
    /// Exact size in bytes.
    pub size: u64,
    /// License declared by the repo's model card.
    pub license: &'static str,
}

/// Dense Llama-architecture model, GGUF (mixed Q8_0 / Q5_0 / Q4_K + F32).
pub const DENSE_GGUF: RealFixture = RealFixture {
    id: "smollm2-135m-gguf",
    repo: "ggml-org/SmolLM2-135M-GGUF",
    revision: "1076b85d135017971845b19b8d07df1479155988",
    file: "SmolLM2-135M-Q4_K_M.gguf",
    sha256: "e3131339bf4e8065265593d4fd8f7bb7ff2d3abff1edb5618aa1197b89cad9f5",
    size: 101_016_128,
    license: "apache-2.0",
};

/// The same dense model as Safetensors (all BF16).
pub const DENSE_SAFETENSORS: RealFixture = RealFixture {
    id: "smollm2-135m-safetensors",
    repo: "HuggingFaceTB/SmolLM2-135M",
    revision: "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    file: "model.safetensors",
    sha256: "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    size: 269_060_552,
    license: "apache-2.0",
};

/// 4-expert Mixtral-architecture MoE, GGUF (stacked `*_exps`, Q8_0 + F32).
pub const MOE_GGUF: RealFixture = RealFixture {
    id: "stories15m-moe-gguf",
    repo: "ggml-org/stories15M_MOE",
    revision: "b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
    file: "stories15M_MOE-Q8_0.gguf",
    sha256: "c7aa6863f9a4b3cdf19716e2c95622dcbd3bd06989324bf1ac8e60486ef8e881",
    size: 39_390_272,
    license: "mit",
};

/// The same MoE as Safetensors (per-expert `w1`/`w2`/`w3`, F16 + F32 router).
pub const MOE_SAFETENSORS: RealFixture = RealFixture {
    id: "stories15m-moe-safetensors",
    repo: "ggml-org/stories15M_MOE",
    revision: "b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
    file: "model.safetensors",
    sha256: "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082",
    size: 72_744_704,
    license: "mit",
};

/// Every pinned fixture.
pub const ALL: [RealFixture; 4] = [DENSE_GGUF, DENSE_SAFETENSORS, MOE_GGUF, MOE_SAFETENSORS];

impl RealFixture {
    /// `<base>/<repo>/resolve/<revision>/<file>`.
    pub fn url(&self, base: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            base.trim_end_matches('/'),
            self.repo,
            self.revision,
            self.file
        )
    }

    fn file_name(&self) -> &str {
        self.file.rsplit('/').next().unwrap_or(self.file)
    }

    /// Cache path: `<cache>/<sha256>/<file name>`, so a new pin never reuses
    /// bytes cached for an older one.
    pub fn cache_path(&self, cache: &Path) -> PathBuf {
        cache.join(self.sha256).join(self.file_name())
    }
}

/// How the helper treats the cache and the network.
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
    pub cache_dir: PathBuf,
    pub base_url: String,
}

impl Config {
    pub fn from_env() -> Self {
        let mode =
            Mode::parse(env::var(ENV_MODE).ok().as_deref()).unwrap_or_else(|e| panic!("{e}"));
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
            cache_dir,
            base_url,
        }
    }
}

/// Result of [`ensure`].
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Verified file on disk.
    Ready(PathBuf),
    /// Not available, with the reason. Never returned in [`Mode::Require`].
    Skipped(String),
}

/// Serializes cache access: tests in one binary run on parallel threads and
/// would otherwise download the same file twice.
static FETCH_LOCK: Mutex<()> = Mutex::new(());

/// Return the verified local path of `fx`, downloading it if the mode allows.
///
/// # Panics
///
/// On a SHA-256 or size mismatch (always), and on any failure in
/// [`Mode::Require`].
pub fn ensure(fx: &RealFixture, cfg: &Config) -> Outcome {
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
    let path = fx.cache_path(&cfg.cache_dir);

    if path.is_file() {
        match verify_cached(fx, &path) {
            Ok(()) => return Outcome::Ready(path),
            Err(why) => {
                eprintln!("{}: dropping bad cache entry ({why})", fx.id);
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(stamp_path(&path));
            }
        }
    }

    if cfg.mode == Mode::Offline {
        return Outcome::Skipped(format!("not cached and {ENV_MODE}=offline"));
    }
    match download(fx, cfg, &path) {
        Ok(()) => Outcome::Ready(path),
        Err(DownloadError::Unavailable(why)) => Outcome::Skipped(why),
        Err(DownloadError::Corrupt(why)) => panic!("{}: {why}", fx.id),
    }
}

enum DownloadError {
    /// No network, no `curl`, or the origin refused: tests may skip.
    Unavailable(String),
    /// Bytes arrived but do not match the pin: never skipped.
    Corrupt(String),
}

fn stamp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".verified");
    PathBuf::from(s)
}

/// `<sha256> <len> <mtime_ns>` of the file as it was when last hashed.
fn stamp_for(fx: &RealFixture, path: &Path) -> io::Result<String> {
    let meta = fs::metadata(path)?;
    let mtime = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(format!("{} {} {mtime}", fx.sha256, meta.len()))
}

/// Trust a cached file when its size, mtime, and a stamp written after a full
/// hash all agree; otherwise re-hash it.
fn verify_cached(fx: &RealFixture, path: &Path) -> Result<(), String> {
    let len = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len != fx.size {
        return Err(format!("size {len} != pinned {}", fx.size));
    }
    let want = stamp_for(fx, path).map_err(|e| e.to_string())?;
    if fs::read_to_string(stamp_path(path)).is_ok_and(|s| s.trim() == want) {
        return Ok(());
    }
    let got = sha256_file(path).map_err(|e| e.to_string())?;
    if got != fx.sha256 {
        return Err(format!("sha256 {got} != pinned {}", fx.sha256));
    }
    let _ = fs::write(stamp_path(path), want);
    Ok(())
}

fn download(fx: &RealFixture, cfg: &Config, dest: &Path) -> Result<(), DownloadError> {
    let unavailable = |why: String| DownloadError::Unavailable(why);
    let dir = dest.parent().expect("cache path has a parent");
    fs::create_dir_all(dir).map_err(|e| unavailable(format!("create {}: {e}", dir.display())))?;

    let part = dir.join(format!("{}.{}.part", fx.file_name(), std::process::id()));
    let url = fx.url(&cfg.base_url);
    eprintln!("{}: downloading {url} ({} bytes)", fx.id, fx.size);

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

    let verdict = check_bytes(fx, &part);
    if let Err(why) = verdict {
        let _ = fs::remove_file(&part);
        return Err(DownloadError::Corrupt(why));
    }
    fs::rename(&part, dest).map_err(|e| {
        let _ = fs::remove_file(&part);
        unavailable(format!("move into cache: {e}"))
    })?;
    if let Ok(stamp) = stamp_for(fx, dest) {
        let _ = fs::write(stamp_path(dest), stamp);
    }
    Ok(())
}

fn check_bytes(fx: &RealFixture, path: &Path) -> Result<(), String> {
    let len = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len != fx.size {
        return Err(format!(
            "downloaded {len} bytes, pinned size is {}",
            fx.size
        ));
    }
    let got = sha256_file(path).map_err(|e| e.to_string())?;
    if got != fx.sha256 {
        return Err(format!(
            "downloaded sha256 {got} does not match pinned {}",
            fx.sha256
        ));
    }
    Ok(())
}

/// Verified path for `fx`, or `None` (after logging why) when it must be
/// skipped. Call at the top of a real-fixture test and `return` on `None`.
pub fn fixture_or_skip(fx: &RealFixture) -> Option<PathBuf> {
    match ensure(fx, &Config::from_env()) {
        Outcome::Ready(path) => Some(path),
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
