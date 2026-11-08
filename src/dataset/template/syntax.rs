//! The template syntax: literal text, `{name}` placeholders, `{{` and `}}` for literal braces.

use std::mem;

use crate::error::Result;

/// A piece of a parsed template.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Piece {
    Literal(String),
    Name(String),
}

/// Splits `template` into literal text and `{name}` placeholders; `{{` and `}}` are literal
/// braces.
pub(super) fn parse(template: &str) -> Result<Vec<Piece>> {
    let mut pieces = Vec::new();
    let mut literal = String::new();
    let mut rest = template;
    while let Some(at) = rest.find(['{', '}']) {
        let offset = template.len() - rest.len() + at;
        literal.push_str(&rest[..at]);
        let tail = &rest[at..];
        if let Some(after) = tail.strip_prefix("{{") {
            literal.push('{');
            rest = after;
        } else if let Some(after) = tail.strip_prefix("}}") {
            literal.push('}');
            rest = after;
        } else if tail.starts_with('}') {
            return Err(format!(
                "unbalanced '}}' at byte {offset} of the template, write '}}}}' for a literal one"
            )
            .into());
        } else {
            let Some(close) = tail.find('}') else {
                return Err(format!(
                    "unclosed '{{' at byte {offset} of the template, write '{{{{' for a literal one"
                )
                .into());
            };
            let name = &tail[1..close];
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                return Err(format!(
                    "invalid placeholder {:?} at byte {offset} of the template: names are \
                     letters, digits and '_'; write '{{{{' for a literal '{{'",
                    &tail[..=close]
                )
                .into());
            }
            if !literal.is_empty() {
                pieces.push(Piece::Literal(mem::take(&mut literal)));
            }
            pieces.push(Piece::Name(name.to_owned()));
            rest = &tail[close + 1..];
        }
    }
    literal.push_str(rest);
    if !literal.is_empty() {
        pieces.push(Piece::Literal(literal));
    }
    Ok(pieces)
}

#[cfg(test)]
mod tests {
    use super::{Piece, parse};

    #[test]
    fn parses_placeholders_and_escaped_braces() {
        assert_eq!(
            parse("a{x}b{{c}}{index}").unwrap(),
            [
                Piece::Literal("a".into()),
                Piece::Name("x".into()),
                Piece::Literal("b{c}".into()),
                Piece::Name("index".into()),
            ]
        );
        assert_eq!(parse("{{{x}}}").unwrap()[1], Piece::Name("x".into()));
        assert_eq!(parse("{{}}").unwrap(), [Piece::Literal("{}".into())]);
    }

    #[test]
    fn refuses_unbalanced_braces_and_bad_names() {
        for (text, message) in [
            ("http://h/{x", "unclosed '{' at byte 9"),
            ("http://h/x}", "unbalanced '}' at byte 10"),
            ("http://h/{}", "invalid placeholder \"{}\""),
            ("http://h/{a b}", "invalid placeholder \"{a b}\""),
            ("http://h/{a{b}", "invalid placeholder \"{a{b}\""),
        ] {
            let error = parse(text).unwrap_err().to_string();
            assert!(error.contains(message), "{text}: {error}");
        }
    }
}
