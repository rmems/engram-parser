// SPDX-License-Identifier: MIT OR Apache-2.0
pub mod memory;

use engram_parser::{Checkpoint, CheckpointFormat, ParserError, SourceKind, open_checkpoint_mmap};
use memory::Memory;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, File};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const WINDOW: usize = 4096;
const REPEATS: usize = 8;

#[derive(Debug)]
struct Expected {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: usize,
    shard: String,
    offset: usize,
}

fn expectations(path: &Path) -> Result<Vec<Expected>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    let mut names = BTreeSet::new();
    for (i, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 6 || fields[0].is_empty() || !names.insert(fields[0]) {
            return Err(format!(
                "invalid/duplicate expected TSV row {} (need six columns)",
                i + 1
            )
            .into());
        }
        let shape = if fields[2] == "scalar" {
            Vec::new()
        } else {
            fields[2]
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        out.push(Expected {
            name: fields[0].into(),
            dtype: fields[1].into(),
            shape,
            bytes: fields[3].parse()?,
            shard: fields[4].into(),
            offset: fields[5].parse()?,
        });
    }
    if out.len() < 3 {
        return Err(
            "expected TSV must select at least three tensors across file offsets/shards".into(),
        );
    }
    Ok(out)
}

fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

fn positive_mib(value: &str) -> Result<usize> {
    let bytes = value
        .parse::<usize>()?
        .checked_mul(1024 * 1024)
        .ok_or("MiB overflow")?;
    require(bytes > 0, "memory budgets must be positive MiB")?;
    Ok(bytes)
}

pub fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: checkpoint_smoke <gguf|safetensors> <checkpoint> <expected.tsv> <rss-growth-MiB> <heap-peak-MiB>; supply local immutable fixtures, see docs/checkpoint-smoke.md".into());
    }
    let format = match args[0].to_str() {
        Some("gguf") => CheckpointFormat::Gguf,
        Some("safetensors") => CheckpointFormat::Safetensors,
        _ => return Err("format must be gguf or safetensors".into()),
    };
    let rss_budget = positive_mib(args[3].to_str().ok_or("invalid RSS budget")?)?;
    let heap_budget = positive_mib(args[4].to_str().ok_or("invalid heap budget")?)?;
    let path = Path::new(&args[1]);
    let expected = expectations(Path::new(&args[2]))?;
    println!(
        "checkpoint smoke v1 unix_seconds={} platform={}-{} page_size={} invocation={:?}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        engram_parser::os_page_size(),
        std::env::args_os().collect::<Vec<_>>()
    );
    println!(
        "kernel={}",
        fs::read_to_string("/proc/sys/kernel/osrelease")?.trim()
    );
    println!(
        "budgets rss_growth={rss_budget} heap_peak={heap_budget} repeats={REPEATS} window={WINDOW}"
    );
    let baseline = Memory::read()?;
    baseline.report("baseline");
    // File-open cost is observed separately; the public mmap open then performs
    // metadata/index parsing, shard mapping and catalog construction atomically.
    let input = File::open(path)?;
    require(
        input.metadata()?.is_file(),
        "supply a GGUF file, Safetensors file, or explicit Safetensors index",
    )?;
    Memory::read()?.report("file_open");
    let checkpoint = open_checkpoint_mmap(path)?;
    Memory::read()?.report("mmap_open_and_metadata_index");
    require(
        checkpoint.format() == format,
        "unexpected checkpoint format",
    )?;
    if format == CheckpointFormat::Safetensors {
        validate_safetensors_source(&checkpoint, &expected)?;
    }
    let mut mapped_bytes = 0;
    let mut smallest_file = u64::MAX;
    for file in &checkpoint.source().files {
        let size = fs::metadata(checkpoint.source().root.join(file))?.len();
        mapped_bytes += size;
        smallest_file = smallest_file.min(size);
        println!("file={file:?} bytes={size}");
    }
    // A budget that could hide one full shard allocation is not useful evidence.
    require(
        (heap_budget as u64) < smallest_file,
        "heap budget must be smaller than every payload file",
    )?;
    require(
        (rss_budget as u64) < mapped_bytes / 2,
        "RSS budget must be less than half the checkpoint size",
    )?;
    println!(
        "mapped_file_bytes={mapped_bytes} tensors={} metadata_keys={} samples={}",
        checkpoint.tensors().len(),
        checkpoint.metadata().len(),
        expected.len()
    );
    require(
        checkpoint
            .tensors()
            .windows(2)
            .all(|w| w[0].name < w[1].name),
        "inventory is not sorted/unique",
    )?;
    for (key, value) in checkpoint.metadata().iter() {
        // Enumerate without printing potentially large tokenizer metadata.
        std::hint::black_box((key, value));
    }
    for e in &expected {
        verify_metadata(&checkpoint, e)?;
        println!(
            "sample name={:?} dtype={} shape={:?} bytes={} shard={:?} file_offset={}",
            e.name, e.dtype, e.shape, e.bytes, e.shard, e.offset
        );
    }
    Memory::read()?.report("inventory");
    let mut pointers = Vec::new();
    let mut fingerprints = Vec::new();
    for e in &expected {
        let (ptr, fingerprint) = probe(&checkpoint, e)?;
        pointers.push(ptr);
        fingerprints.push(fingerprint);
        println!(
            "sample_fingerprint name={:?} fnv1a64={fingerprint:016x}",
            e.name
        );
    }
    Memory::read()?.report("first_access");
    for round in 0..REPEATS {
        for step in 0..expected.len() {
            let i = if round % 2 == 0 {
                expected.len() - 1 - step
            } else {
                (step + round) % expected.len()
            };
            verify_metadata(&checkpoint, &expected[i])?;
            let (ptr, fingerprint) = probe(&checkpoint, &expected[i])?;
            require(
                ptr == pointers[i] && fingerprint == fingerprints[i],
                "lookup changed pointer or sampled bytes across orders",
            )?;
        }
    }
    Memory::read()?.report("repeated_access");
    let mut missing = "__engram_smoke_missing__".to_owned();
    while checkpoint.tensor(&missing).is_some() {
        missing.push('_');
    }
    require(
        matches!(
            checkpoint.tensor_bytes(&missing),
            Err(ParserError::MissingTensor { .. })
        ),
        "missing tensor must return MissingTensor",
    )?;
    if format == CheckpointFormat::Safetensors {
        missing_shard()?;
    }
    let final_memory = Memory::read()?;
    final_memory.report("errors_checked");
    let rss_growth = final_memory.resident_growth_since(baseline);
    println!(
        "observed rss_or_hwm_growth={rss_growth} heap_peak={} largest_allocation={}",
        final_memory.heap_peak, final_memory.largest
    );
    require(
        rss_growth <= rss_budget,
        "resident/high-water growth exceeded fixture budget",
    )?;
    require(
        final_memory.heap_peak <= heap_budget,
        "Rust heap peak exceeded fixture budget",
    )?;
    println!(
        "PASS borrowed payloads; metadata and bounded file reads agree; deterministic lookup; missing-resource errors; memory bounds"
    );
    Ok(())
}

fn validate_safetensors_source(checkpoint: &dyn Checkpoint, expected: &[Expected]) -> Result<()> {
    match checkpoint.source().kind {
        SourceKind::SingleFile => {
            require(
                checkpoint.source().files.len() == 1,
                "single-file Safetensors smoke requires one payload file",
            )?;
            require(
                expected
                    .iter()
                    .all(|e| e.shard == checkpoint.source().files[0]),
                "expected samples must name the single payload file",
            )?;
        }
        SourceKind::ShardIndex => {
            require(
                checkpoint.source().files.len() >= 2,
                "Safetensors shard-index smoke requires multiple shards",
            )?;
            require(
                expected
                    .iter()
                    .map(|e| &e.shard)
                    .collect::<BTreeSet<_>>()
                    .len()
                    >= 2,
                "expected samples must span multiple shards",
            )?;
        }
        _ => {
            return Err("Safetensors smoke requires a single file or explicit shard index".into());
        }
    }
    Ok(())
}

fn expected_dtype_label(native: &str) -> &str {
    match native {
        "INT4" => "I4",
        other => other,
    }
}

fn verify_metadata(checkpoint: &dyn Checkpoint, e: &Expected) -> Result<()> {
    let t = checkpoint
        .tensor(&e.name)
        .ok_or_else(|| format!("expected tensor {:?} missing", e.name))?;
    verify_tensor_metadata(t, e)
}

fn verify_tensor_metadata(t: &engram_parser::TensorInfo, e: &Expected) -> Result<()> {
    require(
        t.name == e.name
            && t.native_dtype == e.dtype
            && t.dtype.label() == expected_dtype_label(&e.dtype)
            && t.shape.dims() == e.shape
            && t.byte_len == e.bytes
            && t.location.source == e.shard
            && t.location.file_offset == Some(e.offset),
        format!(
            "metadata mismatch for {:?}: actual={t:?}, expected={e:?}",
            e.name
        ),
    )
}

fn probe(checkpoint: &dyn Checkpoint, e: &Expected) -> Result<(usize, u64)> {
    let raw = checkpoint.tensor_bytes(&e.name)?;
    require(
        matches!(raw, Cow::Borrowed(_)),
        "mmap access unexpectedly returned owned bytes",
    )?;
    require(raw.len() == e.bytes, "raw byte length mismatch")?;
    let file = File::open(checkpoint.source().root.join(&e.shard))?;
    let width = WINDOW.min(raw.len());
    let mut fingerprint = 0xcbf29ce484222325u64;
    // Read only three bounded windows, even when the returned view spans GiBs.
    for offset in [0, (raw.len() - width) / 2, raw.len() - width] {
        let mut expected = [0u8; WINDOW];
        file.read_exact_at(
            &mut expected[..width],
            e.offset.checked_add(offset).ok_or("offset overflow")? as u64,
        )?;
        require(
            raw[offset..offset + width] == expected[..width],
            format!("payload mismatch for {:?} at {offset}", e.name),
        )?;
        for byte in &raw[offset..offset + width] {
            fingerprint = (fingerprint ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }
    Ok((raw.as_ptr() as usize, fingerprint))
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn missing_shard() -> Result<()> {
    let dir = std::env::temp_dir().join(format!(
        "engram-smoke-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&dir)?;
    let guard = TempDir(dir);
    let index = guard.0.join("model.safetensors.index.json");
    // Isolated negative fixture: never rename, truncate or delete user shards.
    fs::write(
        &index,
        r#"{"weight_map":{"missing.weight":"absent.safetensors"}}"#,
    )?;
    match open_checkpoint_mmap(&index) {
        Err(error @ ParserError::MissingShard { .. }) => {
            println!("missing_shard={error}");
            Ok(())
        }
        other => Err(format!("expected MissingShard, got {other:?}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> TempDir {
        let path = std::env::temp_dir().join(format!(
            "engram-smoke-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        TempDir(path)
    }

    #[test]
    fn rejects_incomplete_and_duplicate_expectations() {
        let dir = temp_dir();
        let path = dir.0.join("expected.tsv");
        fs::write(&path, "# no rows\n").unwrap();
        assert!(expectations(&path).is_err());
        fs::write(
            &path,
            "a\tF32\t1\t4\tx\t0\na\tF32\t1\t4\tx\t4\nb\tF32\t1\t4\tx\t8\n",
        )
        .unwrap();
        assert!(expectations(&path).is_err());
        fs::write(
            &path,
            "a\tF32\t1\t4\tx\t0\nb\tF32\t1\t4\tx\t4\nc\tF32\tscalar\t4\tx\t8\n",
        )
        .unwrap();
        assert_eq!(expectations(&path).unwrap()[2].shape, Vec::<usize>::new());
        assert!(positive_mib("0").is_err());
        assert!(positive_mib("18446744073709551615").is_err());
    }

    #[test]
    fn bounded_probe_detects_bad_metadata_and_wrong_bytes() {
        let dir = temp_dir();
        let path = dir.0.join("model.safetensors");
        let header = r#"{"a":{"dtype":"U8","shape":[16384],"data_offsets":[0,16384]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend((0..16384).map(|i| (i % 251) as u8));
        fs::write(&path, bytes).unwrap();
        let checkpoint = open_checkpoint_mmap(&path).unwrap();
        let mut e = Expected {
            name: "a".into(),
            dtype: "U8".into(),
            shape: vec![16384],
            bytes: 16384,
            shard: "model.safetensors".into(),
            offset: 8 + header.len(),
        };
        verify_metadata(&checkpoint, &e).unwrap();
        let first = probe(&checkpoint, &e).unwrap();
        assert_eq!(first, probe(&checkpoint, &e).unwrap());
        e.dtype = "I8".into();
        assert!(verify_metadata(&checkpoint, &e).is_err());
        e.offset += 1;
        assert!(probe(&checkpoint, &e).is_err());
        e.shard = "wrong.safetensors".into();
        assert!(verify_metadata(&checkpoint, &e).is_err());
    }

    #[test]
    fn native_dtype_alias_matches_normalized_metadata() {
        use engram_parser::{TensorDType, TensorInfo, TensorLocation, TensorShape};
        let mut tensor = TensorInfo::new(
            "packed",
            TensorDType::I4,
            TensorShape::outermost_first(vec![2]),
            1,
            TensorLocation::new("a.safetensors", 0, Some(64)),
        )
        .with_native_dtype("INT4");
        let expected = Expected {
            name: "packed".into(),
            dtype: "INT4".into(),
            shape: vec![2],
            bytes: 1,
            shard: "a.safetensors".into(),
            offset: 64,
        };
        verify_tensor_metadata(&tensor, &expected).unwrap();
        tensor.dtype = TensorDType::U4;
        assert!(verify_tensor_metadata(&tensor, &expected).is_err());
        tensor.dtype = TensorDType::I4;
        tensor.native_dtype = "I4".into();
        assert!(verify_tensor_metadata(&tensor, &expected).is_err());
    }

    #[test]
    fn unavailable_shard_is_a_typed_error() {
        missing_shard().unwrap();
    }
}
