// SPDX-License-Identifier: MIT OR Apache-2.0
//! Non-regular checkpoint inputs (FIFOs, sockets, devices) must fail with a
//! clean error instead of blocking forever on `File::open` or streaming
//! unbounded data. Unix-only: `mkfifo`, Unix sockets, and `/dev/null` are
//! POSIX concepts.
#![cfg(unix)]

use std::ffi::CString;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};

use engram_parser::{ParserError, load_gguf, open_checkpoint};

fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("engram-nonregular-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn mkfifo(path: &Path) {
    unsafe extern "C" {
        fn mkfifo(path: *const i8, mode: u32) -> i32;
    }
    let c = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { mkfifo(c.as_ptr(), 0o644) }, 0, "mkfifo {path:?}");
}

fn assert_not_regular(err: ParserError) {
    match err {
        ParserError::UnsupportedFormat { reason, .. } => {
            assert_eq!(reason, "not a regular file");
        }
        ParserError::InvalidLayout { reason, .. } => {
            assert!(reason.contains("not a regular file"), "got: {reason}");
        }
        other => panic!("expected a not-a-regular-file error, got {other}"),
    }
}

/// A FIFO would previously block `detect()` forever in `File::open`.
#[test]
fn open_checkpoint_fifo_gguf_errors_instead_of_hanging() {
    let dir = tmpdir("fifo-gguf");
    let fifo = dir.join("model.gguf");
    mkfifo(&fifo);
    assert_not_regular(open_checkpoint(&fifo).unwrap_err());
}

#[test]
fn load_gguf_fifo_errors_instead_of_hanging() {
    let dir = tmpdir("fifo-load");
    let fifo = dir.join("model.gguf");
    mkfifo(&fifo);
    assert_not_regular(load_gguf(&fifo).unwrap_err());
}

#[cfg(feature = "mmap")]
#[test]
fn load_gguf_mmap_fifo_errors_instead_of_hanging() {
    let dir = tmpdir("fifo-mmap");
    let fifo = dir.join("model.gguf");
    mkfifo(&fifo);
    assert_not_regular(engram_parser::load_gguf_mmap(&fifo).unwrap_err());
}

#[test]
fn open_checkpoint_fifo_safetensors_errors_instead_of_hanging() {
    let dir = tmpdir("fifo-st");
    let fifo = dir.join("model.safetensors");
    mkfifo(&fifo);
    // With the `safetensors` feature this is InvalidLayout; without it the
    // backend is disabled — either way it must not hang or panic.
    let err = open_checkpoint(&fifo).unwrap_err();
    match err {
        ParserError::InvalidLayout { reason, .. } => {
            assert!(reason.contains("not a regular file"), "got: {reason}");
        }
        ParserError::FeatureDisabled { .. } | ParserError::UnsupportedFormat { .. } => {}
        other => panic!("unexpected error for FIFO safetensors input: {other}"),
    }
}

#[cfg(feature = "safetensors")]
#[test]
fn open_safetensors_checkpoint_fifo_errors_instead_of_hanging() {
    let dir = tmpdir("fifo-st-open");
    let fifo = dir.join("model.safetensors");
    mkfifo(&fifo);
    assert_not_regular(engram_parser::safetensors::open_safetensors_checkpoint(&fifo).unwrap_err());
}

/// `open_checkpoint` rejects a socket path before trying to open it.
#[test]
fn open_checkpoint_socket_errors() {
    let dir = tmpdir("sock");
    let sock = dir.join("model.gguf");
    let _listener = UnixListener::bind(&sock).unwrap();
    assert_not_regular(open_checkpoint(&sock).unwrap_err());
}

/// Character devices can stream unbounded data; they must be rejected
/// before `fs::read`.
#[test]
fn load_gguf_character_device_errors() {
    assert_not_regular(load_gguf("/dev/null").unwrap_err());
    assert_not_regular(load_gguf("/dev/zero").unwrap_err());
}

/// Missing files still surface as `Io` errors, not the new variant.
#[test]
fn missing_file_is_still_io_error() {
    let dir = tmpdir("missing");
    let err = open_checkpoint(dir.join("nope.gguf")).unwrap_err();
    assert!(matches!(err, ParserError::Io { .. }), "got: {err}");
}

/// A regular file must not hit the new guard (it fails later, at parse).
#[test]
fn regular_file_reaches_the_parser() {
    let dir = tmpdir("regular");
    let path = dir.join("model.gguf");
    fs::write(&path, b"not a gguf file").unwrap();
    let err = load_gguf(&path).unwrap_err();
    match err {
        ParserError::UnsupportedFormat { reason, .. } => {
            assert_ne!(reason, "not a regular file");
        }
        other => panic!("expected a parse error, got {other}"),
    }
}
