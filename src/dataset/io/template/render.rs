//! [`Template`]: the location of an item made on demand from a template parsed once and
//! per-item values.

use std::collections::HashSet;
use std::fmt::Write;
use std::iter::repeat_n;

use super::{
    column::{Column, MAX_INT_BYTES},
    escape::Escape,
    syntax::{Piece, parse},
};
use crate::error::Result;

/// Most columns a template may use.
const MAX_COLUMNS: usize = 64;

/// Longest template, in bytes.
const MAX_TEMPLATE_BYTES: usize = 8 << 10;

/// Longest location a template may make, in bytes, values written.
const MAX_LOCATION_BYTES: usize = 64 << 10;

/// Most items of a template dataset (2³²).
const MAX_ITEMS: u64 = 1 << 32;

/// The built-in placeholder: the index of the item.
const INDEX: &str = "index";

/// A part of the location of an item.
#[derive(Debug)]
enum Segment {
    Literal(String),
    Index,
    Column(usize),
}

/// Locations made on demand from a template parsed once, `{index}` and per-item values.
#[derive(Debug)]
pub(crate) struct Template {
    segments: Vec<Segment>,
    columns: Vec<Column>,
    escape: Escape,
    /// For each path component of a location, whether a value is written in it.
    with_values: Vec<bool>,
    len: usize,
    /// The longest location, in bytes: one allocation per location.
    capacity: usize,
}

impl Template {
    /// Parses `template` and checks it against `columns`, whose strings are written as
    /// `escape` says. The number of items is `len`, or that of the columns; given both, they
    /// must agree.
    pub(crate) fn new(
        template: &str,
        columns: Vec<(&str, Column)>,
        len: Option<usize>,
        escape: Escape,
    ) -> Result<Self> {
        if template.is_empty() {
            return Err("the template is empty".into());
        }
        if template.len() > MAX_TEMPLATE_BYTES {
            return Err(format!(
                "the template has {} bytes, at most {MAX_TEMPLATE_BYTES}",
                template.len()
            )
            .into());
        }
        if columns.len() > MAX_COLUMNS {
            return Err(format!("{} columns, at most {MAX_COLUMNS}", columns.len()).into());
        }
        let mut names: Vec<&str> = Vec::with_capacity(columns.len());
        for (name, _) in &columns {
            if *name == INDEX {
                return Err("column \"index\": the name is reserved for the item index".into());
            }
            if names.contains(name) {
                return Err(format!("column {name:?} is given twice").into());
            }
            names.push(name);
        }
        let len = items(&columns, len)?;

        let mut used = HashSet::new();
        let mut segments = Vec::new();
        for piece in parse(template)? {
            let segment = match piece {
                Piece::Literal(text) => Segment::Literal(text),
                Piece::Name(name) if name == INDEX => Segment::Index,
                Piece::Name(name) => {
                    let Some(column) = names.iter().position(|n| *n == name) else {
                        return Err(format!(
                            "unknown placeholder {{{name}}} in the template; columns: {}",
                            known(&names)
                        )
                        .into());
                    };
                    used.insert(column);
                    Segment::Column(column)
                }
            };
            segments.push(segment);
        }
        if let Some(unused) = (0..names.len()).find(|column| !used.contains(column)) {
            return Err(format!("column {:?} is not used by the template", names[unused]).into());
        }

        let mut widths = Vec::with_capacity(columns.len());
        for (name, column) in &columns {
            widths.push(column.max_bytes(name, escape)?);
        }
        let capacity = segments.iter().try_fold(0usize, |total, segment| {
            total.checked_add(match segment {
                Segment::Literal(text) => text.len(),
                Segment::Index => MAX_INT_BYTES,
                Segment::Column(column) => widths[*column],
            })
        });
        let capacity = capacity
            .filter(|&bytes| bytes <= MAX_LOCATION_BYTES)
            .ok_or_else(|| {
                format!("the template makes locations over {MAX_LOCATION_BYTES} bytes")
            })?;
        Ok(Self {
            with_values: with_values(&segments, escape),
            segments,
            columns: columns.into_iter().map(|(_, column)| column).collect(),
            escape,
            len,
            capacity,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The location of item `index`, which must be below [`len`](Self::len). Fails if values
    /// next to each other or to the template's text make a `.` or `..` path component (or an
    /// empty key part).
    pub(crate) fn render(&self, index: usize) -> Result<String> {
        let mut location = String::with_capacity(self.capacity);
        for segment in &self.segments {
            match segment {
                Segment::Literal(text) => location.push_str(text),
                // Writing to a String cannot fail.
                Segment::Index => _ = write!(location, "{index}"),
                Segment::Column(column) => {
                    self.columns[*column].write(index, self.escape, &mut location)
                }
            }
        }
        // Values hold no separator: the components are those of the template.
        let components = location.split(self.escape.separators());
        for (component, &with_value) in components.zip(&self.with_values) {
            if !with_value {
                continue;
            }
            if let Some(reason) = self.escape.refuse_component(component) {
                return Err(format!("item {index}: path component {component:?} {reason}").into());
            }
        }
        Ok(location)
    }
}

/// For each path component of the template, whether a placeholder is in it.
fn with_values(segments: &[Segment], escape: Escape) -> Vec<bool> {
    let mut components = vec![false];
    for segment in segments {
        match segment {
            Segment::Literal(text) => {
                let separators = text.matches(escape.separators()).count();
                components.extend(repeat_n(false, separators));
            }
            Segment::Index | Segment::Column(_) => {
                if let Some(last) = components.last_mut() {
                    *last = true;
                }
            }
        }
    }
    components
}

/// The number of items: `len`, or that of the columns, which must all have the same.
fn items(columns: &[(&str, Column)], len: Option<usize>) -> Result<usize> {
    let mut items = len;
    for (name, column) in columns {
        match items {
            None => items = Some(column.len()),
            Some(n) if n != column.len() => {
                let expected = match len {
                    Some(_) => "the length".to_owned(),
                    None => format!("column {:?}", columns[0].0),
                };
                return Err(format!(
                    "column {name:?} has {} values, {expected} has {n}",
                    column.len()
                )
                .into());
            }
            Some(_) => {}
        }
    }
    let items = items.ok_or("a template without columns needs a length")?;
    if items as u64 > MAX_ITEMS {
        return Err(format!("{items} items, at most {MAX_ITEMS}").into());
    }
    Ok(items)
}

fn known(names: &[&str]) -> String {
    if names.is_empty() {
        return "none (only {index} is built in)".to_owned();
    }
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::{Column, Escape, Template};

    fn template(text: &str, columns: Vec<(&str, Column)>) -> String {
        match Template::new(text, columns, None, Escape::Url) {
            Ok(_) => "ok".to_owned(),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn checks_columns_against_the_template() {
        let xs = || Column::from(vec![1i64, 2]);
        assert_eq!(template("", vec![]), "the template is empty");
        assert!(template("http://h/{y}", vec![("x", xs())]).contains("unknown placeholder {y}"));
        assert!(
            template("http://h/{x}", vec![("x", xs()), ("y", xs())]).contains("\"y\" is not used")
        );
        assert!(template("http://h/{x}", vec![("x", xs()), ("x", xs())]).contains("given twice"));
        assert!(template("http://h/{index}", vec![("index", xs())]).contains("reserved"));
        let short = Column::from(vec![1u64]);
        assert_eq!(
            template("http://h/{x}/{y}", vec![("x", xs()), ("y", short)]),
            "column \"y\" has 1 values, column \"x\" has 2"
        );
        assert!(template("http://h/{index}", vec![]).contains("needs a length"));
        let dots = Column::from(vec!["a", ".."]);
        assert!(template("http://h/{s}", vec![("s", dots)]).contains("item 1: \"..\""));
        let long = Column::from(vec!["é".repeat(20_000)]);
        assert!(template("http://h/{s}", vec![("s", long)]).contains("over 65536 bytes"));
        let many: Vec<(&str, Column)> = (0..65).map(|_| ("x", xs())).collect();
        assert!(template("{x}", many).contains("at most 64"));
        assert!(template(&"a".repeat(8193), vec![]).contains("at most 8192"));
    }

    #[test]
    fn writes_integers_index_and_encoded_strings() {
        let t = Template::new(
            "http://h/{name}/tile?x={x}&y={y}&i={index}&b={{}}",
            vec![
                ("x", Column::from(vec![i64::MIN, 7])),
                ("y", Column::from(vec![u64::MAX, 0])),
                ("name", Column::from(vec!["a b/c?d#e&f=g", "é~-._"])),
            ],
            None,
            Escape::Url,
        )
        .unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(
            t.render(0).unwrap(),
            "http://h/a%20b%2Fc%3Fd%23e%26f%3Dg/tile?x=-9223372036854775808\
             &y=18446744073709551615&i=0&b={}"
        );
        assert_eq!(
            t.render(1).unwrap(),
            "http://h/%C3%A9~-._/tile?x=7&y=0&i=1&b={}"
        );
        for index in 0..2 {
            assert!(t.render(index).unwrap().len() <= t.capacity);
        }
    }

    #[test]
    fn an_explicit_length_with_index_only_or_matching_columns() {
        let t = Template::new("http://h/{index}.png", vec![], Some(3), Escape::Url).unwrap();
        assert_eq!(
            (t.len(), t.render(2).unwrap().as_str()),
            (3, "http://h/2.png")
        );
        let xs = Column::from(vec![1i64, 2]);
        let error = Template::new("http://h/{x}", vec![("x", xs)], Some(3), Escape::Url);
        assert_eq!(
            error.unwrap_err().to_string(),
            "column \"x\" has 2 values, the length has 3"
        );
    }

    #[test]
    fn paths_are_written_raw_and_cannot_traverse() {
        let names = Column::from(vec!["s 1.svs", "%41é"]);
        let t = Template::new(
            "/data/{s}/{index}.png",
            vec![("s", names)],
            None,
            Escape::Path,
        );
        let t = t.unwrap();
        assert_eq!(t.render(0).unwrap(), "/data/s 1.svs/0.png");
        assert_eq!(t.render(1).unwrap(), "/data/%41é/1.png");
        for value in ["..", ".", "../etc", "a/b", "a\\b", "a\0b"] {
            let column = Column::from(vec![value]);
            let error = Template::new("/data/{s}", vec![("s", column)], None, Escape::Path);
            assert!(error.is_err(), "{value:?}");
        }
        // Empty values next to the template's dots cannot make `..` either.
        let empty = || Column::from(vec!["", "x"]);
        let t = Template::new("/data/..{a}/f", vec![("a", empty())], None, Escape::Path).unwrap();
        let error = t.render(0).unwrap_err().to_string();
        assert!(error.contains("item 0: path component \"..\""), "{error}");
        assert_eq!(t.render(1).unwrap(), "/data/..x/f");
        let columns = vec![("a", empty()), ("b", empty())];
        let t = Template::new("/data/.{a}.{b}", columns, None, Escape::Path).unwrap();
        assert!(t.render(0).is_err());
        // The template's own text is trusted.
        let t = Template::new("../data/{index}", vec![], Some(1), Escape::Path).unwrap();
        assert_eq!(t.render(0).unwrap(), "../data/0");
    }

    #[test]
    fn keys_are_written_raw_as_valid_parts() {
        let names = Column::from(vec!["s 1.svs", "a\\b"]);
        let t = Template::new("tiles/{s}/{index}", vec![("s", names)], None, Escape::Key);
        let t = t.unwrap();
        assert_eq!(t.render(0).unwrap(), "tiles/s 1.svs/0");
        assert_eq!(t.render(1).unwrap(), "tiles/a\\b/1");
        for value in ["", ".", "..", "a/b", "a\nb"] {
            let column = Column::from(vec![value]);
            let error = Template::new("tiles/{s}", vec![("s", column)], None, Escape::Key);
            assert!(error.is_err(), "{value:?}");
        }
        // A key part is never empty: values cannot make an empty or a dot part.
        let dot = Column::from(vec!["."]);
        assert!(Template::new("tiles/.{s}", vec![("s", dot)], None, Escape::Key).is_err());
    }

    #[test]
    fn urls_refuse_dot_segments_made_by_values() {
        let empty = Column::from(vec!["", "x"]);
        let t = Template::new("http://h/a/..{s}/b", vec![("s", empty)], None, Escape::Url);
        let t = t.unwrap();
        assert!(t.render(0).is_err());
        assert_eq!(t.render(1).unwrap(), "http://h/a/..x/b");
    }
}
