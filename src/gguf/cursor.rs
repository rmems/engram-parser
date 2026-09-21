// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streaming cursor over the raw GGUF byte stream.
//!
//! Only what the parser needs: little-endian scalar reads, length-prefixed
//! strings, and the ability to skip (or stringify) unknown KV values.

use super::limits::{ParseLimits, u64_to_usize};
use crate::error::{HostSizeField, ParseLimitKind, ParserError, Result};

pub(crate) const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub(crate) const GGUF_VERSION: u32 = 3;

pub(crate) fn unsupported(path: &str, reason: impl Into<String>) -> ParserError {
    ParserError::UnsupportedFormat {
        path: path.to_owned(),
        reason: reason.into(),
    }
}

pub(crate) fn invalid_layout(path: &str, reason: impl Into<String>) -> ParserError {
    ParserError::InvalidLayout {
        path: path.to_owned(),
        reason: reason.into(),
    }
}

/// GGUF value type: 8-bit unsigned integer.
pub const GGUF_VALUE_TYPE_UINT8: u32 = 0;
/// GGUF value type: 8-bit signed integer.
pub const GGUF_VALUE_TYPE_INT8: u32 = 1;
/// GGUF value type: 16-bit unsigned integer.
pub const GGUF_VALUE_TYPE_UINT16: u32 = 2;
/// GGUF value type: 16-bit signed integer.
pub const GGUF_VALUE_TYPE_INT16: u32 = 3;
/// GGUF value type: 32-bit unsigned integer.
pub const GGUF_VALUE_TYPE_UINT32: u32 = 4;
/// GGUF value type: 32-bit signed integer.
pub const GGUF_VALUE_TYPE_INT32: u32 = 5;
/// GGUF value type: 32-bit IEEE 754 float.
pub const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
/// GGUF value type: boolean (1 byte, 0 or 1).
pub const GGUF_VALUE_TYPE_BOOL: u32 = 7;
/// GGUF value type: length-prefixed UTF-8 string.
pub const GGUF_VALUE_TYPE_STRING: u32 = 8;
/// GGUF value type: length-prefixed array of nested values.
pub const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
/// GGUF value type: 64-bit unsigned integer.
pub const GGUF_VALUE_TYPE_UINT64: u32 = 10;
/// GGUF value type: 64-bit signed integer.
pub const GGUF_VALUE_TYPE_INT64: u32 = 11;
/// GGUF value type: 64-bit IEEE 754 float.
pub const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

pub(crate) fn is_signed_layout_type(value_type: u32) -> bool {
    matches!(
        value_type,
        GGUF_VALUE_TYPE_INT8
            | GGUF_VALUE_TYPE_INT16
            | GGUF_VALUE_TYPE_INT32
            | GGUF_VALUE_TYPE_INT64
    )
}

fn is_unsigned_layout_type(value_type: u32) -> bool {
    matches!(
        value_type,
        GGUF_VALUE_TYPE_UINT8
            | GGUF_VALUE_TYPE_UINT16
            | GGUF_VALUE_TYPE_UINT32
            | GGUF_VALUE_TYPE_UINT64
            | GGUF_VALUE_TYPE_BOOL
    )
}

fn nonneg_signed(path: &str, v: i64) -> Result<u64> {
    u64::try_from(v).map_err(|_| {
        invalid_layout(
            path,
            format!("signed layout value {v} is negative; expected non-negative"),
        )
    })
}

fn fixed_scalar_size(value_type: u32) -> Option<u64> {
    match value_type {
        GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => Some(1),
        GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => Some(2),
        GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => Some(4),
        GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

pub(crate) struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    path: &'a str,
    limits: ParseLimits,
    array_work_used: u64,
    metadata_origin: Option<usize>,
}

impl<'a> GgufCursor<'a> {
    #[cfg(test)]
    pub(crate) fn new(bytes: &'a [u8], path: &'a str) -> Self {
        Self::with_limits(bytes, path, ParseLimits::default())
    }

    pub(crate) fn with_limits(bytes: &'a [u8], path: &'a str, limits: ParseLimits) -> Self {
        Self {
            bytes,
            offset: 0,
            path,
            limits,
            array_work_used: 0,
            metadata_origin: None,
        }
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn limits(&self) -> ParseLimits {
        self.limits
    }

    pub(crate) fn begin_metadata_section(&mut self) {
        self.metadata_origin = Some(self.offset);
    }

    pub(crate) fn end_metadata_section(&mut self) {
        self.metadata_origin = None;
    }

    fn unsupported(&self, reason: String) -> ParserError {
        ParserError::UnsupportedFormat {
            path: self.path.to_owned(),
            reason,
        }
    }

    fn remaining_bytes(&self) -> u64 {
        self.bytes.len().saturating_sub(self.offset) as u64
    }

    fn metadata_used(&self) -> Option<u64> {
        let origin = self.metadata_origin?;
        Some(self.offset.saturating_sub(origin) as u64)
    }

    fn ensure_metadata_room(&self, extra: u64) -> Result<()> {
        let Some(used) = self.metadata_used() else {
            return Ok(());
        };
        let Some(projected) = used.checked_add(extra) else {
            return Err(ParserError::limit_exceeded(
                self.path,
                ParseLimitKind::MetadataBytes,
                u64::MAX,
                self.limits.max_metadata_bytes,
            ));
        };
        self.limits
            .reject(self.path, ParseLimitKind::MetadataBytes, projected)
    }

    fn charge_metadata_used(&self) -> Result<()> {
        let Some(used) = self.metadata_used() else {
            return Ok(());
        };
        self.limits
            .reject(self.path, ParseLimitKind::MetadataBytes, used)
    }

    fn consume_array_work(&mut self, items: u64) -> Result<()> {
        let Some(declared) = self.array_work_used.checked_add(items) else {
            return Err(ParserError::limit_exceeded(
                self.path,
                ParseLimitKind::ArrayWorkItems,
                u64::MAX,
                self.limits.max_array_work_items,
            ));
        };
        self.limits
            .reject(self.path, ParseLimitKind::ArrayWorkItems, declared)?;
        self.array_work_used = declared;
        Ok(())
    }

    fn prepare_array_elements(&mut self, element_type: u32, len: u64) -> Result<()> {
        self.consume_array_work(len)?;
        if let Some(size) = fixed_scalar_size(element_type) {
            let need = len.saturating_mul(size);
            self.ensure_metadata_room(need)?;
            if need > self.remaining_bytes() {
                return Err(
                    self.unsupported("GGUF array length exceeds remaining metadata bytes".into())
                );
            }
        } else if len > self.remaining_bytes() {
            return Err(
                self.unsupported("GGUF array length exceeds remaining metadata bytes".into())
            );
        }
        Ok(())
    }

    pub(crate) fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let extra = u64::try_from(len).unwrap_or(u64::MAX);
        self.ensure_metadata_room(extra)?;
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| self.unsupported("cursor overflow".into()))?;
        if end > self.bytes.len() {
            return Err(self.unsupported("unexpected EOF while parsing GGUF".into()));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        self.charge_metadata_used()?;
        Ok(slice)
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        let arr: [u8; 4] = bytes
            .try_into()
            .map_err(|_| self.unsupported("expected 4-byte u32 payload".into()))?;
        Ok(u32::from_le_bytes(arr))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        let arr: [u8; 8] = bytes
            .try_into()
            .map_err(|_| self.unsupported("expected 8-byte u64 payload".into()))?;
        Ok(u64::from_le_bytes(arr))
    }

    pub(crate) fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    pub(crate) fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    pub(crate) fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub(crate) fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    pub(crate) fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        self.limits
            .reject(self.path, ParseLimitKind::StringBytes, len)?;
        self.ensure_metadata_room(len)?;
        let len = u64_to_usize(len, HostSizeField::StringLen, self.path)?;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| self.unsupported(format!("invalid UTF-8 in GGUF string: {e}")))
    }

    /// Read an unsigned numeric GGUF value and coerce it to `u64`.
    fn read_unsigned_as_u64(&mut self, value_type: u32) -> Result<u64> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_BOOL => self.read_u8_as_u64(),
            GGUF_VALUE_TYPE_UINT16 => self.read_u16_as_u64(),
            GGUF_VALUE_TYPE_UINT32 => self.read_u32_as_u64(),
            GGUF_VALUE_TYPE_UINT64 => self.read_u64(),
            other => Err(self.unsupported(format!(
                "expected unsigned numeric GGUF value, got type {other}"
            ))),
        }
    }

    /// Read a numeric-typed GGUF value and coerce it to `u64`.
    pub(crate) fn read_numeric_as_u64(&mut self, value_type: u32) -> Result<u64> {
        if is_signed_layout_type(value_type) {
            self.read_signed_as_u64(value_type)
        } else if is_unsigned_layout_type(value_type) {
            self.read_unsigned_as_u64(value_type)
        } else {
            Err(self.unsupported(format!(
                "expected numeric GGUF value, got type {value_type}"
            )))
        }
    }

    fn read_u8_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_u8()? as u64)
    }

    fn read_u16_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_u16()? as u64)
    }

    fn read_u32_as_u64(&mut self) -> Result<u64> {
        Ok(self.read_u32()? as u64)
    }

    /// Read a signed GGUF value and return its bit-preserving `u64`
    /// representation. Negative values are not rejected here so that vendor
    /// metadata can store signed quantities without loss.
    fn read_signed_as_u64(&mut self, value_type: u32) -> Result<u64> {
        Ok(self.read_signed_as_i64(value_type)? as u64)
    }

    fn read_signed_as_i64(&mut self, value_type: u32) -> Result<i64> {
        match value_type {
            GGUF_VALUE_TYPE_INT8 => Ok(self.read_u8()? as i8 as i64),
            GGUF_VALUE_TYPE_INT16 => Ok(self.read_i16()? as i64),
            GGUF_VALUE_TYPE_INT32 => Ok(self.read_i32()? as i64),
            GGUF_VALUE_TYPE_INT64 => Ok(self.read_i64()?),
            other => unreachable!("caller filters signed types, got {other}"),
        }
    }

    /// Read a non-negative layout value (e.g. `general.alignment`).
    ///
    /// Rejects signed negatives so they do not wrap into huge alignments.
    /// Other signed KV pairs should use [`Self::read_numeric_as_u64`] instead.
    pub(crate) fn read_nonneg_layout_usize(&mut self, value_type: u32) -> Result<usize> {
        let v = if is_signed_layout_type(value_type) {
            let s = self.read_signed_as_i64(value_type)?;
            nonneg_signed(self.path, s)?
        } else if is_unsigned_layout_type(value_type) {
            self.read_numeric_as_u64(value_type)?
        } else {
            return Err(self.unsupported(format!(
                "expected integer GGUF value for layout field, got type {value_type}"
            )));
        };
        u64_to_usize(v, HostSizeField::Alignment, self.path)
    }

    /// Render a scalar GGUF value as a string (used for metadata KV).
    #[allow(dead_code)]
    pub(crate) fn read_scalar_as_string(&mut self, value_type: u32) -> Result<String> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8
            | GGUF_VALUE_TYPE_INT8
            | GGUF_VALUE_TYPE_UINT16
            | GGUF_VALUE_TYPE_INT16
            | GGUF_VALUE_TYPE_UINT32
            | GGUF_VALUE_TYPE_INT32
            | GGUF_VALUE_TYPE_UINT64
            | GGUF_VALUE_TYPE_INT64
            | GGUF_VALUE_TYPE_BOOL => Ok(self.read_numeric_as_u64(value_type)?.to_string()),
            GGUF_VALUE_TYPE_FLOAT32 => Ok(self.read_f32()?.to_string()),
            GGUF_VALUE_TYPE_FLOAT64 => Ok(self.read_f64()?.to_string()),
            GGUF_VALUE_TYPE_STRING => self.read_string(),
            other => Err(self.unsupported(format!("unexpected scalar GGUF value type {other}"))),
        }
    }

    /// Skip an arbitrary GGUF value without materialising it.
    pub(crate) fn skip_value(&mut self, value_type: u32) -> Result<()> {
        if value_type == GGUF_VALUE_TYPE_ARRAY {
            self.skip_array_value()
        } else {
            self.skip_scalar_value(value_type)
        }
    }

    fn skip_scalar_value(&mut self, value_type: u32) -> Result<()> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => {
                self.read_exact(1)?;
            }
            GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => {
                self.read_exact(2)?;
            }
            GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => {
                self.read_exact(4)?;
            }
            GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => {
                self.read_exact(8)?;
            }
            GGUF_VALUE_TYPE_STRING => {
                let _ = self.read_string()?;
            }
            other => {
                return Err(self.unsupported(format!("unsupported GGUF value type {other}")));
            }
        }
        Ok(())
    }

    /// Skip a GGUF array value using an explicit stack instead of recursion,
    /// so deep-but-valid metadata arrays are not rejected by an arbitrary
    /// depth limit. Total work is bounded by [`ParseLimits::max_array_work_items`]
    /// and, when parsing KV metadata, [`ParseLimits::max_metadata_bytes`].
    fn skip_array_value(&mut self) -> Result<()> {
        let nested = self.read_u32()?;
        let len = self.read_u64()?;
        self.prepare_array_elements(nested, len)?;

        // Stack of (element_type, elements_remaining) pairs. Depth is bounded
        // only by nesting of arrays, not by a hard-coded recursion limit.
        let mut stack: Vec<(u32, u64)> = Vec::new();
        stack.push((nested, len));

        while let Some((ty, mut count)) = stack.pop() {
            if ty == GGUF_VALUE_TYPE_ARRAY {
                if count == 0 {
                    continue;
                }
                let sub_ty = self.read_u32()?;
                let sub_len = self.read_u64()?;
                self.prepare_array_elements(sub_ty, sub_len)?;
                count -= 1;
                if count > 0 {
                    stack.push((ty, count));
                }
                stack.push((sub_ty, sub_len));
            } else {
                let n = u64_to_usize(count, HostSizeField::ArrayLen, self.path)?;
                for _ in 0..n {
                    self.skip_scalar_value(ty)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ParseLimitKind;

    fn push_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn nested_array_bytes(depth: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(depth * 12);
        for level in 0..depth {
            if level == depth - 1 {
                push_u32(&mut bytes, GGUF_VALUE_TYPE_UINT8);
            } else {
                push_u32(&mut bytes, GGUF_VALUE_TYPE_ARRAY);
            }
            push_u64(&mut bytes, if level == depth - 1 { 0 } else { 1 });
        }
        bytes
    }

    #[test]
    fn skip_array_value_handles_deep_nesting() {
        const DEPTH: usize = 32;
        let bytes = nested_array_bytes(DEPTH);
        let mut cursor = GgufCursor::new(&bytes, "mem://deep-array");
        cursor
            .skip_value(GGUF_VALUE_TYPE_ARRAY)
            .expect("skip deep array");
        assert_eq!(
            cursor.offset,
            bytes.len(),
            "did not consume entire nested array"
        );
    }

    #[test]
    fn skip_array_value_rejects_work_budget_before_loop() {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, GGUF_VALUE_TYPE_UINT8);
        push_u64(&mut bytes, 4);
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let limits = ParseLimits {
            max_array_work_items: 3,
            ..ParseLimits::default()
        };
        let mut cursor = GgufCursor::with_limits(&bytes, "mem://array-work", limits);
        let err = cursor.skip_value(GGUF_VALUE_TYPE_ARRAY).unwrap_err();
        match err {
            ParserError::LimitExceeded {
                limit,
                declared,
                budget,
                ..
            } => {
                assert_eq!(limit, ParseLimitKind::ArrayWorkItems);
                assert_eq!(declared, 4);
                assert_eq!(budget, 3);
            }
            other => panic!("expected LimitExceeded, got {other}"),
        }
        assert_eq!(cursor.offset, 12, "must reject before iterating elements");
    }

    #[test]
    fn skip_array_value_accepts_exact_work_budget() {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, GGUF_VALUE_TYPE_UINT8);
        push_u64(&mut bytes, 3);
        bytes.extend_from_slice(&[1, 2, 3]);
        let limits = ParseLimits {
            max_array_work_items: 3,
            ..ParseLimits::default()
        };
        let mut cursor = GgufCursor::with_limits(&bytes, "mem://array-work-ok", limits);
        cursor
            .skip_value(GGUF_VALUE_TYPE_ARRAY)
            .expect("exact work budget");
    }

    #[test]
    fn skip_array_value_rejects_u64_max_count() {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, GGUF_VALUE_TYPE_UINT8);
        push_u64(&mut bytes, u64::MAX);
        let mut cursor = GgufCursor::new(&bytes, "mem://array-max");
        let err = cursor.skip_value(GGUF_VALUE_TYPE_ARRAY).unwrap_err();
        match err {
            ParserError::LimitExceeded {
                limit, declared, ..
            } => {
                assert_eq!(limit, ParseLimitKind::ArrayWorkItems);
                assert_eq!(declared, u64::MAX);
            }
            other => panic!("expected LimitExceeded, got {other}"),
        }
    }

    #[test]
    fn read_string_rejects_oversize_before_allocating() {
        let mut bytes = Vec::new();
        push_u64(&mut bytes, 5);
        bytes.extend_from_slice(b"hello");
        let limits = ParseLimits {
            max_string_bytes: 4,
            ..ParseLimits::default()
        };
        let mut cursor = GgufCursor::with_limits(&bytes, "mem://str", limits);
        let err = cursor.read_string().unwrap_err();
        match err {
            ParserError::LimitExceeded {
                limit,
                declared,
                budget,
                ..
            } => {
                assert_eq!(limit, ParseLimitKind::StringBytes);
                assert_eq!(declared, 5);
                assert_eq!(budget, 4);
            }
            other => panic!("expected LimitExceeded, got {other}"),
        }
    }

    #[test]
    fn read_string_accepts_exact_budget() {
        let mut bytes = Vec::new();
        push_u64(&mut bytes, 4);
        bytes.extend_from_slice(b"abcd");
        let limits = ParseLimits {
            max_string_bytes: 4,
            ..ParseLimits::default()
        };
        let mut cursor = GgufCursor::with_limits(&bytes, "mem://str-ok", limits);
        assert_eq!(cursor.read_string().unwrap(), "abcd");
    }
}
