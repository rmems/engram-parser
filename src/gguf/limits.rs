// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit resource budgets for untrusted GGUF inputs.
//!
//! GGUF headers declare counts, string lengths, array lengths, tensor ranks,
//! and offsets as `u64`. Those values must not drive unbounded allocation or
//! iteration. [`ParseLimits`] is the single policy object for those checks.
//! Default values are generous enough for production checkpoints and every
//! committed fixture; trusted callers may raise or disable budgets without
//! changing the default path.

use crate::error::{HostSizeField, ParseLimitKind, ParserError, Result};

/// Resource budgets applied before allocation or loops proportional to
/// attacker-declared GGUF values.
///
/// Defaults accept large real checkpoints. Override individual fields for
/// tighter sandboxes, or use [`ParseLimits::trusted`] when the input is
/// already authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLimits {
    /// Maximum number of KV metadata pairs.
    pub max_kv_count: u64,
    /// Maximum number of tensor directory entries.
    pub max_tensor_count: u64,
    /// Maximum UTF-8 bytes in a single GGUF string.
    pub max_string_bytes: u64,
    /// Maximum nested-array elements visited while skipping metadata arrays.
    pub max_array_work_items: u64,
    /// Maximum tensor rank (`n_dims`).
    pub max_tensor_rank: u64,
    /// Maximum bytes consumed by the KV metadata section.
    pub max_metadata_bytes: u64,
}

impl ParseLimits {
    /// Generous production defaults. Same values as [`Default`].
    pub const DEFAULT: Self = Self {
        max_kv_count: 1_000_000,
        max_tensor_count: 1_000_000,
        max_string_bytes: 16 * 1024 * 1024,
        max_array_work_items: 16_000_000,
        max_tensor_rank: 8,
        max_metadata_bytes: 256 * 1024 * 1024,
    };

    /// Disable every budget (`u64::MAX`). Host-size conversions, alignment
    /// overflow, and EOF checks still apply.
    pub const TRUSTED: Self = Self {
        max_kv_count: u64::MAX,
        max_tensor_count: u64::MAX,
        max_string_bytes: u64::MAX,
        max_array_work_items: u64::MAX,
        max_tensor_rank: u64::MAX,
        max_metadata_bytes: u64::MAX,
    };

    /// Limits for already-authenticated inputs. Does not change [`Default`].
    pub const fn trusted() -> Self {
        Self::TRUSTED
    }

    /// Budget associated with `kind`.
    pub const fn budget(self, kind: ParseLimitKind) -> u64 {
        match kind {
            ParseLimitKind::KvCount => self.max_kv_count,
            ParseLimitKind::TensorCount => self.max_tensor_count,
            ParseLimitKind::StringBytes => self.max_string_bytes,
            ParseLimitKind::ArrayWorkItems => self.max_array_work_items,
            ParseLimitKind::TensorRank => self.max_tensor_rank,
            ParseLimitKind::MetadataBytes => self.max_metadata_bytes,
        }
    }

    /// Fail closed when `declared` is strictly greater than the named budget.
    pub(crate) fn reject(self, path: &str, kind: ParseLimitKind, declared: u64) -> Result<()> {
        let budget = self.budget(kind);
        if declared > budget {
            Err(ParserError::limit_exceeded(path, kind, declared, budget))
        } else {
            Ok(())
        }
    }

    /// Reject `declared` against `kind`, then convert it to a host `usize`.
    pub(crate) fn bounded_usize(
        self,
        path: &str,
        kind: ParseLimitKind,
        field: HostSizeField,
        declared: u64,
    ) -> Result<usize> {
        self.reject(path, kind, declared)?;
        u64_to_usize(declared, field, path)
    }
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Convert a file-declared `u64` to `usize`, naming the field on failure.
pub(crate) fn u64_to_usize(value: u64, field: HostSizeField, path: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| ParserError::host_size(path, field, value))
}

/// Round `value` up to a multiple of `alignment` without wrapping.
pub(crate) fn align_up_checked(value: usize, alignment: usize, path: &str) -> Result<usize> {
    if alignment <= 1 {
        return Ok(value);
    }
    let padding = alignment - 1;
    let padded = value
        .checked_add(padding)
        .ok_or_else(|| ParserError::InvalidLayout {
            path: path.to_owned(),
            reason: "aligned tensor data offset overflow".into(),
        })?;
    Ok(padded - (padded % alignment))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_are_generous_and_named() {
        let limits = ParseLimits::default();
        assert_eq!(limits, ParseLimits::DEFAULT);
        assert!(limits.max_kv_count >= 1_000_000);
        assert!(limits.max_tensor_count >= 1_000_000);
        assert!(limits.max_string_bytes >= 1024);
        assert!(limits.max_array_work_items >= 32);
        assert!(limits.max_tensor_rank >= 8);
        assert!(limits.max_metadata_bytes >= 1024);
    }

    #[test]
    fn trusted_does_not_change_default() {
        assert_ne!(ParseLimits::trusted(), ParseLimits::default());
        assert_eq!(ParseLimits::trusted().max_kv_count, u64::MAX);
        assert_eq!(ParseLimits::default().max_kv_count, 1_000_000);
    }

    #[test]
    fn reject_is_inclusive_at_the_budget() {
        let limits = ParseLimits {
            max_kv_count: 4,
            ..ParseLimits::default()
        };
        limits
            .reject("mem://lim", ParseLimitKind::KvCount, 4)
            .expect("exact budget must pass");
        let err = limits
            .reject("mem://lim", ParseLimitKind::KvCount, 5)
            .unwrap_err();
        match err {
            ParserError::LimitExceeded {
                limit,
                declared,
                budget,
                ..
            } => {
                assert_eq!(limit, ParseLimitKind::KvCount);
                assert_eq!(declared, 5);
                assert_eq!(budget, 4);
            }
            other => panic!("expected LimitExceeded, got {other}"),
        }
    }

    #[test]
    fn u64_to_usize_reports_field_when_unrepresentable() {
        #[cfg(target_pointer_width = "32")]
        {
            let err = u64_to_usize(u64::MAX, HostSizeField::StringLen, "mem://host").unwrap_err();
            match err {
                ParserError::HostSizeOverflow { field, value, .. } => {
                    assert_eq!(field, HostSizeField::StringLen);
                    assert_eq!(value, u64::MAX);
                }
                other => panic!("expected HostSizeOverflow, got {other}"),
            }
        }
        #[cfg(not(target_pointer_width = "32"))]
        {
            assert_eq!(
                u64_to_usize(u64::MAX, HostSizeField::StringLen, "mem://host").unwrap(),
                usize::MAX
            );
        }
    }

    #[test]
    fn align_up_checked_rejects_overflow() {
        let err = align_up_checked(usize::MAX, 32, "mem://align").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("aligned tensor data offset overflow"),
            "got: {msg}"
        );
    }

    #[test]
    fn align_up_checked_rounds_up() {
        assert_eq!(align_up_checked(1, 32, "mem://align").unwrap(), 32);
        assert_eq!(align_up_checked(32, 32, "mem://align").unwrap(), 32);
        assert_eq!(align_up_checked(7, 1, "mem://align").unwrap(), 7);
    }
}
