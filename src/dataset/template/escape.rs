//! [`Escape`]: how a template writes a string value for each kind of location, and which
//! values it refuses.

/// How string values are written into a location; integers are always written in decimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Escape {
    /// A URL: percent-encoded as a URL component, every byte but `A-Z a-z 0-9 - . _ ~`
    /// becoming `%XX`, so that a value cannot add a path segment, a query parameter or a
    /// fragment.
    #[cfg_attr(not(feature = "http"), allow(dead_code))]
    Url,
    /// A file path: written as it is, but a value cannot hold `/`, `\` or NUL, so that it
    /// cannot leave the template's directory.
    Path,
}

impl Escape {
    /// The bytes a path component splits on: a value never writes one.
    pub(super) fn separators(self) -> &'static [char] {
        // URL parsers read `\` as `/` in http(s) URLs.
        &['/', '\\']
    }

    /// Why `value` cannot be written, if it cannot.
    pub(super) fn refuse(self, value: &str) -> Option<&'static str> {
        if let Some(reason) = self.refuse_component(value) {
            return Some(reason);
        }
        match self {
            Self::Url => None,
            Self::Path => value
                .contains(['/', '\\', '\0'])
                .then_some("holds '/', '\\' or NUL"),
        }
    }

    /// Why a path component holding a value is refused, if it is: `.` and `..` would move
    /// out of the template's directory.
    pub(super) fn refuse_component(self, component: &str) -> Option<&'static str> {
        matches!(component, "." | "..").then_some("would be resolved as a path segment")
    }

    /// The length of `value` once written.
    pub(super) fn len(self, value: &str) -> usize {
        match self {
            Self::Url => value.bytes().fold(0usize, |len, byte| {
                len.saturating_add(if is_unreserved(byte) { 1 } else { 3 })
            }),
            Self::Path => value.len(),
        }
    }

    /// Writes `value` onto `out`.
    pub(super) fn write(self, value: &str, out: &mut String) {
        match self {
            Self::Url => percent_encode(value, out),
            Self::Path => out.push_str(value),
        }
    }
}

/// Bytes kept as they are in an encoded value (RFC 3986 unreserved).
fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

/// Percent-encodes `value` as a URL component onto `out`.
fn percent_encode(value: &str, out: &mut String) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in value.bytes() {
        if is_unreserved(byte) {
            out.push(char::from(byte));
        } else {
            out.push('%');
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 0xf)]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Escape;

    fn written(escape: Escape, value: &str) -> String {
        let mut out = String::new();
        escape.write(value, &mut out);
        assert_eq!(out.len(), escape.len(value));
        out
    }

    #[test]
    fn urls_percent_encode_everything_but_unreserved_bytes() {
        assert_eq!(
            written(Escape::Url, "a b/c?d#e&f=g\\"),
            "a%20b%2Fc%3Fd%23e%26f%3Dg%5C"
        );
        assert_eq!(written(Escape::Url, "é~-._"), "%C3%A9~-._");
        assert_eq!(Escape::Url.refuse("a/b"), None);
        for value in [".", ".."] {
            assert!(Escape::Url.refuse(value).is_some(), "{value}");
        }
    }

    #[test]
    fn paths_are_raw_and_cannot_leave_the_directory() {
        assert_eq!(written(Escape::Path, "s 1.svs%é"), "s 1.svs%é");
        for value in [".", "..", "../etc", "a/b", "a\\b", "a\0b", "/abs"] {
            assert!(Escape::Path.refuse(value).is_some(), "{value:?}");
        }
        for value in ["", "...", "a..b", ".hidden", "s 1.svs"] {
            assert_eq!(Escape::Path.refuse(value), None, "{value:?}");
        }
    }
}
