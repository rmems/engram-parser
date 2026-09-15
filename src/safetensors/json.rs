// SPDX-License-Identifier: MIT OR Apache-2.0
//! Zero-dependency JSON parser for Safetensors headers and HF index files.
//!
//! Rejects duplicate object keys. Does not depend on `serde_json`.

use super::model_load;
use crate::error::{ParserError, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// Parsed JSON value. Objects use [`BTreeMap`] so iteration is deterministic.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum JsonValue {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

/// Numeric JSON token preserved as an integer when possible.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum JsonNumber {
    I64(i64),
    U64(u64),
    F64(f64),
}

impl JsonValue {
    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub(super) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(JsonNumber::U64(n)) => Some(*n),
            Self::Number(JsonNumber::I64(n)) if *n >= 0 => Some(*n as u64),
            _ => None,
        }
    }

    pub(super) fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub(super) fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(obj) => Some(obj),
            _ => None,
        }
    }
}

pub(super) fn stringify_metadata(
    prefix: &str,
    metadata: BTreeMap<String, JsonValue>,
) -> BTreeMap<String, String> {
    metadata
        .into_iter()
        .map(|(key, value)| (format!("{prefix}:{key}"), stringify_json_value(&value)))
        .collect()
}

pub(super) fn parse_json_rejecting_duplicate_keys(
    bytes: &[u8],
    path: &Path,
    context: &str,
) -> Result<JsonValue> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| model_load(path, format!("parse {context} JSON: invalid UTF-8 ({e})")))?;
    parse_json(text, &path.display().to_string()).map_err(|err| match err {
        ParserError::InvalidLayout { reason, .. } => {
            model_load(path, format!("parse {context} JSON: {reason}"))
        }
        other => other,
    })
}

pub(super) fn parse_metadata_object(
    path: &Path,
    value: &JsonValue,
) -> Result<BTreeMap<String, String>> {
    let object = value.as_object().ok_or_else(|| {
        model_load(
            path,
            "Safetensors __metadata__ must be a JSON object of strings".into(),
        )
    })?;
    object
        .iter()
        .map(|(key, value)| {
            let value = value.as_str().ok_or_else(|| {
                model_load(
                    path,
                    format!("Safetensors __metadata__ key '{key}' must be a string"),
                )
            })?;
            Ok((key.clone(), value.to_string()))
        })
        .collect()
}

pub(super) fn stringify_json_value(value: &JsonValue) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| encode_compact(value))
}

pub(super) fn encode_compact(value: &JsonValue) -> String {
    let mut out = String::new();
    write_compact(&mut out, value);
    out
}

pub(super) fn encode_pretty(value: &JsonValue) -> String {
    let mut out = String::new();
    write_pretty(&mut out, value, 0);
    out.push('\n');
    out
}

pub(super) fn parse_string_array(encoded: &str) -> Option<Vec<String>> {
    let value = parse_json(encoded, "<metadata>").ok()?;
    let array = value.as_array()?;
    array
        .iter()
        .map(|item| item.as_str().map(ToOwned::to_owned))
        .collect()
}

fn parse_json(input: &str, path: &str) -> Result<JsonValue> {
    let mut parser = JsonParser {
        input,
        pos: 0,
        path,
    };
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.pos < parser.input.len() {
        return Err(parser.error("trailing content after JSON value"));
    }
    Ok(value)
}

struct JsonParser<'a> {
    input: &'a str,
    pos: usize,
    path: &'a str,
}

impl<'a> JsonParser<'a> {
    fn error(&self, reason: &str) -> ParserError {
        ParserError::InvalidLayout {
            path: self.path.to_owned(),
            reason: format!("JSON parse error at offset {}: {reason}", self.pos),
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.bump();
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn next_char(&mut self) -> Result<char> {
        self.bump().ok_or_else(|| self.error("unexpected EOF"))
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        let c = self.next_char()?;
        if c != expected {
            return Err(self.error(&format!("expected '{expected}', got '{c}'")));
        }
        Ok(())
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_whitespace();
        let c = self.peek().ok_or_else(|| self.error("unexpected EOF"))?;
        match c {
            '"' => self.parse_string().map(JsonValue::String),
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            't' | 'f' => self.parse_bool(),
            'n' => self.parse_null(),
            '-' | '0'..='9' => self.parse_number(),
            _ => Err(self.error(&format!("unexpected character: '{c}'"))),
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect_char('"')?;
        let mut s = String::new();
        loop {
            let c = self.next_char()?;
            match c {
                '"' => return Ok(s),
                '\\' => {
                    let escaped = self.next_char()?;
                    match escaped {
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        '/' => s.push('/'),
                        'b' => s.push('\u{0008}'),
                        'f' => s.push('\u{000C}'),
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        'u' => {
                            let code = self.parse_unicode_escape()?;
                            let ch = char::from_u32(code)
                                .ok_or_else(|| self.error("invalid unicode escape"))?;
                            s.push(ch);
                        }
                        _ => {
                            return Err(
                                self.error(&format!("invalid escape sequence: \\{escaped}"))
                            );
                        }
                    }
                }
                c if c < '\u{0020}' => {
                    return Err(self.error("control character in string"));
                }
                _ => s.push(c),
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<u32> {
        let mut code = 0u32;
        for _ in 0..4 {
            let c = self.next_char()?;
            let digit = c
                .to_digit(16)
                .ok_or_else(|| self.error("invalid hex digit in unicode escape"))?;
            code = code * 16 + digit;
        }
        Ok(code)
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.expect_char('{')?;
        let mut map = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.bump();
            return Ok(JsonValue::Object(map));
        }

        loop {
            self.skip_whitespace();
            if self.peek() != Some('"') {
                return Err(self.error("expected string key in object"));
            }
            let key = self.parse_string()?;
            if map.contains_key(&key) {
                return Err(self.error(&format!("duplicate JSON key '{key}'")));
            }
            self.skip_whitespace();
            self.expect_char(':')?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.bump();
                    self.skip_whitespace();
                    if self.peek() == Some('}') {
                        return Err(self.error("trailing comma in object"));
                    }
                }
                Some('}') => {
                    self.bump();
                    return Ok(JsonValue::Object(map));
                }
                Some(c) => {
                    return Err(self.error(&format!("expected ',' or '}}' in object, got '{c}'")));
                }
                None => return Err(self.error("unterminated object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.expect_char('[')?;
        let mut arr = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(']') {
            self.bump();
            return Ok(JsonValue::Array(arr));
        }

        loop {
            let value = self.parse_value()?;
            arr.push(value);
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.bump();
                    self.skip_whitespace();
                    if self.peek() == Some(']') {
                        return Err(self.error("trailing comma in array"));
                    }
                }
                Some(']') => {
                    self.bump();
                    return Ok(JsonValue::Array(arr));
                }
                Some(c) => {
                    return Err(self.error(&format!("expected ',' or ']' in array, got '{c}'")));
                }
                None => return Err(self.error("unterminated array")),
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.bump();
        }

        let first = self.next_char()?;
        if !first.is_ascii_digit() {
            return Err(self.error("invalid number"));
        }
        if first == '0' {
            if matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.error("leading zeros are not allowed"));
            }
        } else {
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }

        let mut is_float = false;
        if self.peek() == Some('.') {
            is_float = true;
            self.bump();
            let frac = self.next_char()?;
            if !frac.is_ascii_digit() {
                return Err(self.error("invalid fraction in number"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            let exp = self.next_char()?;
            if !exp.is_ascii_digit() {
                return Err(self.error("invalid exponent in number"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }

        let token = &self.input[start..self.pos];
        let number = if is_float {
            let n: f64 = token
                .parse()
                .map_err(|_| self.error("number out of range"))?;
            JsonNumber::F64(n)
        } else if token.starts_with('-') {
            let n: i64 = token
                .parse()
                .map_err(|_| self.error("number out of range"))?;
            JsonNumber::I64(n)
        } else {
            let n: u64 = token
                .parse()
                .map_err(|_| self.error("number out of range"))?;
            JsonNumber::U64(n)
        };
        Ok(JsonValue::Number(number))
    }

    fn parse_bool(&mut self) -> Result<JsonValue> {
        if self.input[self.pos..].starts_with("true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.input[self.pos..].starts_with("false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(self.error("invalid boolean"))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue> {
        if self.input[self.pos..].starts_with("null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(self.error("invalid null"))
        }
    }
}

fn write_compact(out: &mut String, value: &JsonValue) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(true) => out.push_str("true"),
        JsonValue::Bool(false) => out.push_str("false"),
        JsonValue::Number(n) => write_number(out, n),
        JsonValue::String(s) => write_string(out, s),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_compact(out, item);
            }
            out.push(']');
        }
        JsonValue::Object(map) => {
            out.push('{');
            for (i, (key, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, key);
                out.push(':');
                write_compact(out, val);
            }
            out.push('}');
        }
    }
}

fn write_pretty(out: &mut String, value: &JsonValue, indent: usize) {
    match value {
        JsonValue::Array(items) if items.is_empty() => out.push_str("[]"),
        JsonValue::Object(map) if map.is_empty() => out.push_str("{}"),
        JsonValue::Array(items) => {
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                push_indent(out, indent + 1);
                write_pretty(out, item, indent + 1);
                if i + 1 != items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(out, indent);
            out.push(']');
        }
        JsonValue::Object(map) => {
            out.push_str("{\n");
            for (i, (key, val)) in map.iter().enumerate() {
                push_indent(out, indent + 1);
                write_string(out, key);
                out.push_str(": ");
                write_pretty(out, val, indent + 1);
                if i + 1 != map.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            push_indent(out, indent);
            out.push('}');
        }
        other => write_compact(out, other),
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

fn write_number(out: &mut String, number: &JsonNumber) {
    match number {
        JsonNumber::I64(n) => {
            let _ = write!(out, "{n}");
        }
        JsonNumber::U64(n) => {
            let _ = write!(out, "{n}");
        }
        JsonNumber::F64(n) => {
            if n.is_finite() && n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                let _ = write!(out, "{}", *n as i64);
            } else {
                let _ = write!(out, "{n}");
            }
        }
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{0020}' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            _ => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_object() {
        let json = r#"{"key": "value", "num": 42}"#;
        let value = parse_json(json, "test").unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj.get("key").unwrap().as_str().unwrap(), "value");
        assert_eq!(obj.get("num").unwrap().as_u64().unwrap(), 42);
    }

    #[test]
    fn parses_arrays() {
        let json = r#"{"shape": [64, 2048], "offsets": [0, 524288]}"#;
        let value = parse_json(json, "test").unwrap();
        let obj = value.as_object().unwrap();
        let shape = obj.get("shape").unwrap().as_array().unwrap();
        assert_eq!(shape[0].as_u64(), Some(64));
        assert_eq!(shape[1].as_u64(), Some(2048));
    }

    #[test]
    fn rejects_duplicate_keys() {
        let json = r#"{"dup": 1, "dup": 2}"#;
        let err = parse_json(json, "test").unwrap_err();
        assert!(err.to_string().contains("duplicate JSON key"));
    }

    #[test]
    fn parses_escaped_strings() {
        let json = r#"{"key": "value with \"quotes\" and \\backslash"}"#;
        let value = parse_json(json, "test").unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(
            obj.get("key").unwrap().as_str().unwrap(),
            "value with \"quotes\" and \\backslash"
        );
    }

    #[test]
    fn compact_string_array_matches_corinth_encoding() {
        let value = JsonValue::Array(vec![JsonValue::String("unused.safetensors".into())]);
        assert_eq!(encode_compact(&value), r#"["unused.safetensors"]"#);
    }
}
