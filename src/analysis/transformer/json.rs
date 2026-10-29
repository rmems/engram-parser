// SPDX-License-Identifier: MIT OR Apache-2.0
//! Minimal JSON emitter shared by topology and component views.

use std::fmt::Write as _;

/// JSON emitter: objects, arrays, strings, integers, booleans, null.
pub(super) struct JsonWriter<'a> {
    out: &'a mut String,
    /// Whether the current container already has one item (per depth).
    first: Vec<bool>,
}

impl<'a> JsonWriter<'a> {
    pub(super) fn new(out: &'a mut String) -> Self {
        Self {
            out,
            first: Vec::new(),
        }
    }

    fn separator(&mut self) {
        if let Some(first) = self.first.last_mut() {
            if *first {
                *first = false;
            } else {
                self.out.push(',');
            }
        }
    }

    pub(super) fn begin_object(&mut self) {
        self.separator();
        self.out.push('{');
        self.first.push(true);
    }

    pub(super) fn end_object(&mut self) {
        self.first.pop();
        self.out.push('}');
    }

    pub(super) fn begin_array(&mut self) {
        self.separator();
        self.out.push('[');
        self.first.push(true);
    }

    pub(super) fn end_array(&mut self) {
        self.first.pop();
        self.out.push(']');
    }

    pub(super) fn key(&mut self, key: &str) {
        self.separator();
        escape_into(self.out, key);
        self.out.push(':');
        if let Some(first) = self.first.last_mut() {
            *first = true;
        }
    }

    pub(super) fn string(&mut self, s: &str) {
        self.separator();
        escape_into(self.out, s);
    }

    pub(super) fn opt_string(&mut self, s: Option<&str>) {
        match s {
            Some(s) => self.string(s),
            None => {
                self.separator();
                self.out.push_str("null");
            }
        }
    }

    pub(super) fn usize(&mut self, v: usize) {
        self.separator();
        let _ = write!(self.out, "{v}");
    }

    pub(super) fn opt_usize(&mut self, v: Option<usize>) {
        match v {
            Some(v) => self.usize(v),
            None => {
                self.separator();
                self.out.push_str("null");
            }
        }
    }

    pub(super) fn bool(&mut self, v: bool) {
        self.separator();
        self.out.push_str(if v { "true" } else { "false" });
    }

    pub(super) fn usize_array(&mut self, values: &[usize]) {
        self.begin_array();
        for &v in values {
            self.usize(v);
        }
        self.end_array();
    }

    pub(super) fn string_array<'s>(&mut self, values: impl Iterator<Item = &'s str>) {
        self.begin_array();
        for v in values {
            self.string(v);
        }
        self.end_array();
    }
}

fn escape_into(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_writer_handles_nesting_and_escapes() {
        let mut out = String::new();
        let mut w = JsonWriter::new(&mut out);
        w.begin_object();
        w.key("a");
        w.usize(1);
        w.key("b");
        w.begin_array();
        w.string("x\"y");
        w.bool(false);
        w.begin_object();
        w.end_object();
        w.end_array();
        w.key("c");
        w.opt_string(None);
        w.end_object();
        assert_eq!(out, r#"{"a":1,"b":["x\"y",false,{}],"c":null}"#);
    }
}
