//! [`Escape`]: how a template writes a string value for each kind of location, and which
//! values it refuses.

use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};

/// Bytes percent-encoded in a URL value: all but the RFC 3986 unreserved ones,
/// `A-Z a-z 0-9 - . _ ~`.
const URL_COMPONENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

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
    /// An `object_store` key: written as it is, but a value must be a valid key part: not
    /// empty, no `/`, no ASCII control character.
    #[cfg_attr(not(feature = "object-store"), allow(dead_code))]
    Key,
}

impl Escape {
    /// The bytes a path component splits on: a value never writes one.
    pub(super) fn separators(self) -> &'static [char] {
        match self {
            // URL parsers read `\` as `/` in http(s) URLs.
            Self::Url | Self::Path => &['/', '\\'],
            Self::Key => &['/'],
        }
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
            Self::Key => value
                .chars()
                .any(|c| c == '/' || c.is_ascii_control())
                .then_some("holds '/' or a control character"),
        }
    }

    /// Why a path component holding a value is refused, if it is: `.` and `..` would move
    /// out of the template's directory, an empty key part is invalid.
    pub(super) fn refuse_component(self, component: &str) -> Option<&'static str> {
        match component {
            "." | ".." => Some("would be resolved as a path segment"),
            "" if self == Self::Key => Some("is an empty key part"),
            _ => None,
        }
    }

    /// Why a path component of a location, holding a value, is refused, if it is: as
    /// [`refuse_component`](Self::refuse_component), and in a URL also the spellings of `.`
    /// and `..` with `%2e` (any case), which URL parsers resolve as dot segments too. A value
    /// written next to the template's `%` or `%2e` can make one.
    pub(super) fn refuse_location_component(self, component: &str) -> Option<&'static str> {
        if self == Self::Url && is_url_dot_segment(component) {
            return Some("would be resolved as a path segment");
        }
        self.refuse_component(component)
    }

    /// The length of `value` once written.
    pub(super) fn len(self, value: &str) -> usize {
        match self {
            Self::Url => utf8_percent_encode(value, URL_COMPONENT)
                .fold(0usize, |len, piece| len.saturating_add(piece.len())),
            Self::Path | Self::Key => value.len(),
        }
    }

    /// Writes `value` onto `out`.
    pub(super) fn write(self, value: &str, out: &mut String) {
        match self {
            Self::Url => out.extend(utf8_percent_encode(value, URL_COMPONENT)),
            Self::Path | Self::Key => out.push_str(value),
        }
    }
}

/// Whether a URL parser reads `component` as a `.` or `..` segment (URL Standard, single-dot
/// and double-dot URL path segments).
fn is_url_dot_segment(component: &str) -> bool {
    [".", "%2e", "..", ".%2e", "%2e.", "%2e%2e"]
        .iter()
        .any(|dots| component.eq_ignore_ascii_case(dots))
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
        // A value is encoded, so "%2e" is a harmless value; as a component, it is a dot.
        assert_eq!(Escape::Url.refuse("%2e"), None);
        for component in ["%2e", "%2E", ".%2e", "%2E.", "%2e%2E"] {
            assert!(
                Escape::Url.refuse_location_component(component).is_some(),
                "{component}"
            );
        }
        for component in ["%2e%2e%2e", "...", "%2f", "x%2e"] {
            assert_eq!(Escape::Url.refuse_location_component(component), None);
        }
        assert_eq!(Escape::Path.refuse_location_component("%2e%2e"), None);
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

    #[test]
    fn keys_follow_object_store_part_rules() {
        assert_eq!(written(Escape::Key, "s 1%é"), "s 1%é");
        for value in ["", ".", "..", "a/b", "a\nb", "a\u{7f}b"] {
            assert!(Escape::Key.refuse(value).is_some(), "{value:?}");
        }
        for value in ["...", "a..b", "a\\b", "s 1.svs"] {
            assert_eq!(Escape::Key.refuse(value), None, "{value:?}");
        }
    }
}
