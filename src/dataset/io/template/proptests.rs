//! Property tests of the templates: hostile values stay inside their URL component, path
//! component or key part; malformed templates are errors, never panics; locations stay within
//! the capacity computed when the template is parsed.

use std::collections::HashSet;

use proptest::prelude::*;

use super::render::MAX_LOCATION_BYTES;
use super::syntax::{Piece, parse};
use super::{Column, Escape, Template};

/// Pieces of text that mean something to URLs, paths, keys or templates.
const SPECIAL: &[&str] = &[
    "/", "\\", "?", "#", "&", "=", "%", "%2e", "%2E", "%2F", "%00", "{", "}", "{{", "}}", " ", ".",
    "..", ":", "@", ";", "+", "\0", "\n", "\t", "\u{7f}", "é", "😀", "\u{202e}", "",
];

/// A string made of special pieces and any characters, any string, or a long one.
fn value() -> impl Strategy<Value = String> {
    let piece = prop_oneof![
        prop::sample::select(SPECIAL).prop_map(str::to_owned),
        any::<char>().prop_map(String::from),
    ];
    prop_oneof![
        4 => prop::collection::vec(piece, 0..12).prop_map(|pieces| pieces.concat()),
        2 => any::<String>(),
        1 => (prop::sample::select(&["a", "é", "/", "%"][..]), 1usize..30_000)
            .prop_map(|(unit, n)| unit.repeat(n)),
    ]
}

/// One string column.
fn strings(values: &[&String]) -> Column {
    Column::from(values.iter().map(|v| (*v).clone()).collect::<Vec<String>>())
}

/// Whether `error` is the location-length cap.
fn too_long(error: &crate::error::Error) -> bool {
    error
        .to_string()
        .starts_with("the template makes locations over")
}

/// Template text from pieces: literal braces doubled, names in braces.
fn unparse(pieces: &[Piece]) -> String {
    pieces
        .iter()
        .map(|piece| match piece {
            Piece::Literal(text) => text.replace('{', "{{").replace('}', "}}"),
            Piece::Name(name) => format!("{{{name}}}"),
        })
        .collect()
}

/// Pieces with adjacent literals merged, as `parse` returns them.
fn merged(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    for piece in pieces {
        match (out.last_mut(), piece) {
            (_, Piece::Literal(text)) if text.is_empty() => {}
            (Some(Piece::Literal(last)), Piece::Literal(text)) => last.push_str(&text),
            (_, piece) => out.push(piece),
        }
    }
    out
}

fn piece() -> impl Strategy<Value = Piece> {
    prop_oneof![
        prop::collection::vec(
            prop_oneof![
                prop::sample::select(SPECIAL).prop_map(str::to_owned),
                any::<char>().prop_map(String::from),
            ],
            0..4
        )
        .prop_map(|parts| Piece::Literal(parts.concat())),
        "[A-Za-z0-9_]{1,8}".prop_map(Piece::Name),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Any text: a template or an error, never a panic; a template's locations fit in its
    /// capacity.
    #[test]
    fn any_template_text_is_parsed_or_refused(
        text in prop_oneof![
            any::<String>(),
            prop::collection::vec(
                prop::sample::select(&["{", "}", "{{", "}}", "{index}", "{x}", "{a b}", "x", "/", "é"][..]),
                0..16,
            )
            .prop_map(|parts| parts.concat()),
        ],
        escape in prop::sample::select(&[Escape::Url, Escape::Path, Escape::Key][..]),
    ) {
        let parsed = parse(&text);
        let template = Template::new(&text, vec![], Some(3), escape);
        if parsed.is_err() || text.is_empty() {
            prop_assert!(template.is_err());
        }
        if let Ok(template) = template {
            for index in 0..template.len() {
                if let Ok(location) = template.render(index) {
                    prop_assert!(location.len() <= template.capacity);
                }
            }
        }
    }

    /// Parsing the text of any pieces gives the pieces back.
    #[test]
    fn parse_inverts_unparse(pieces in prop::collection::vec(piece(), 0..8)) {
        let text = unparse(&pieces);
        prop_assert_eq!(parse(&text).unwrap(), merged(pieces));
    }

    /// Unknown, unused, reserved and duplicate names are errors, and only they are.
    #[test]
    fn names_are_checked_against_the_columns(
        placeholders in prop::collection::vec(
            prop::sample::select(&["a", "b", "c", "index"][..]), 0..5),
        names in prop::collection::vec(prop::sample::select(&["a", "b", "c", "index"][..]), 0..5),
    ) {
        let text: String = std::iter::once("http://h/").chain(
            placeholders.iter().flat_map(|name| ["{", name, "}/"])
        ).collect();
        let columns: Vec<(&str, Column)> =
            names.iter().map(|name| (*name, Column::from(vec![1u64, 2]))).collect();
        let used: HashSet<&str> = placeholders.iter().copied().collect();
        let given: HashSet<&str> = names.iter().copied().collect();
        let valid = given.len() == names.len()
            && !given.contains("index")
            && used.iter().all(|name| *name == "index" || given.contains(name))
            && given.iter().all(|name| used.contains(name))
            && (!names.is_empty());
        let template = Template::new(&text, columns, None, Escape::Url);
        prop_assert_eq!(template.is_ok(), valid, "{:?}", template.err().map(|e| e.to_string()));
    }

    /// A path value is refused if it holds a separator or NUL, or is `.` or `..`; any other
    /// is written as it is into its own component.
    #[test]
    fn path_values_stay_in_their_component(a in value(), b in value()) {
        let columns = vec![("a", strings(&[&a])), ("b", strings(&[&b]))];
        let template = Template::new("/data/{a}/x-{b}.bin", columns, None, Escape::Path);
        let refused = |v: &str| v.contains(['/', '\\', '\0']) || v == "." || v == "..";
        let template = match template {
            Err(error) => {
                prop_assert!(refused(&a) || refused(&b) || too_long(&error), "{}", error);
                return Ok(());
            }
            Ok(template) => template,
        };
        prop_assert!(!refused(&a) && !refused(&b));
        let location = template.render(0).unwrap();
        prop_assert!(location.len() <= template.capacity);
        prop_assert!(template.capacity <= MAX_LOCATION_BYTES);
        let parts: Vec<&str> = location.split('/').collect();
        let expected = ["", "data", a.as_str(), &format!("x-{b}.bin")];
        prop_assert_eq!(parts, expected);
    }

    /// An object key value is refused if it is empty, `.` or `..`, or holds `/` or a control
    /// character; any other is one valid key part.
    #[test]
    fn key_values_stay_in_their_part(a in value()) {
        let template = Template::new("tiles/{a}/0", vec![("a", strings(&[&a]))], None, Escape::Key);
        let refused = a.is_empty()
            || a == "."
            || a == ".."
            || a.chars().any(|c| c == '/' || c.is_ascii_control());
        let template = match template {
            Err(error) => {
                prop_assert!(refused || too_long(&error), "{}", error);
                return Ok(());
            }
            Ok(template) => template,
        };
        prop_assert!(!refused);
        let location = template.render(0).unwrap();
        prop_assert_eq!(location.split('/').collect::<Vec<_>>(), ["tiles", a.as_str(), "0"]);
        #[cfg(feature = "object-store")]
        {
            let key = object_store::path::Path::parse(&location).unwrap();
            let parts: Vec<String> = key.parts().map(|p| p.as_ref().to_owned()).collect();
            prop_assert_eq!(parts, ["tiles", a.as_str(), "0"]);
        }
    }
}

#[cfg(feature = "http")]
mod url {
    use percent_encoding::percent_decode_str;
    use proptest::prelude::*;
    use reqwest::Url;

    use super::{MAX_LOCATION_BYTES, strings, too_long, value};
    use crate::dataset::io::template::{Column, Escape, Template};

    /// Text the template itself may put around a placeholder in a path segment.
    const AROUND: &[&str] = &["", ".", "..", "%2e", "%2E", "%", "%2", "x"];

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// Values cannot add a path segment, a query parameter or a fragment, nor change the
        /// host: the parsed URL has the template's shape and decodes back to the values.
        #[test]
        fn url_values_stay_in_their_component(a in value(), b in value(), i in any::<i64>()) {
            let columns = vec![
                ("a", strings(&[&a])),
                ("b", strings(&[&b])),
                ("i", Column::from(vec![i])),
            ];
            let text = "https://h.example/v1/{a}/tile?x={b}&n={i}#top";
            let template = match Template::new(text, columns, None, Escape::Url) {
                Err(error) => {
                    let dots = |v: &str| v == "." || v == "..";
                    prop_assert!(dots(&a) || dots(&b) || too_long(&error), "{}", error);
                    return Ok(());
                }
                Ok(template) => template,
            };
            let location = template.render(0).unwrap();
            prop_assert!(location.len() <= template.capacity);
            prop_assert!(template.capacity <= MAX_LOCATION_BYTES);
            let url = Url::parse(&location).unwrap();
            // The parser keeps the text as it is: nothing in it is reinterpreted.
            prop_assert_eq!(url.as_str(), location.as_str());
            prop_assert_eq!(url.host_str(), Some("h.example"));
            let segments: Vec<&str> = url.path_segments().unwrap().collect();
            prop_assert_eq!(segments.len(), 3);
            prop_assert_eq!(segments[0], "v1");
            prop_assert_eq!(percent_decode_str(segments[1]).decode_utf8().unwrap(), a.as_str());
            prop_assert_eq!(segments[2], "tile");
            let pairs: Vec<(String, String)> = url.query_pairs().into_owned().collect();
            prop_assert_eq!(pairs, [("x".to_owned(), b), ("n".to_owned(), i.to_string())]);
            prop_assert_eq!(url.fragment(), Some("top"));
        }

        /// Values next to the template's own dots, `%` or `%2e` cannot make a `.` or `..`
        /// segment, however the URL parser spells it.
        #[test]
        fn url_values_cannot_make_dot_segments(
            before in prop::sample::select(AROUND),
            after in prop::sample::select(AROUND),
            a in prop::sample::select(&["", ".", "..", "2e", "2E", "e", "x", "%2e"][..]),
        ) {
            let text = format!("https://h.example/v1/{before}{{a}}{after}/tile");
            let template = Template::new(&text, vec![("a", strings(&[&a.to_owned()]))], None, Escape::Url);
            let Ok(template) = template else { return Ok(()) };
            let Ok(location) = template.render(0) else { return Ok(()) };
            let url = Url::parse(&location).unwrap();
            let segments: Vec<&str> = url.path_segments().unwrap().collect();
            prop_assert_eq!(segments.len(), 3, "{} -> {}", location, url);
            prop_assert_eq!(segments[0], "v1");
            prop_assert_eq!(segments[2], "tile");
        }

        /// A value in the host can make the parse fail, but cannot move the host out of the
        /// template's domain or add a path.
        #[test]
        fn url_values_cannot_change_the_domain(a in value()) {
            let template = Template::new(
                "https://{a}.h.example/x", vec![("a", strings(&[&a]))], None, Escape::Url,
            );
            let Ok(template) = template else { return Ok(()) };
            let location = template.render(0).unwrap();
            if let Ok(url) = Url::parse(&location) {
                prop_assert!(url.host_str().unwrap().ends_with(".h.example"), "{}", url);
                prop_assert_eq!(url.path(), "/x");
                prop_assert_eq!(url.query(), None);
                prop_assert_eq!(url.fragment(), None);
            }
        }
    }
}
