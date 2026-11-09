//! [`Column`]: the values of one template placeholder, one per item.

use std::fmt::Write;

use super::escape::Escape;
use crate::error::Result;

/// Longest decimal integer, `-9223372036854775808` or `18446744073709551615`.
pub(super) const MAX_INT_BYTES: usize = 20;

/// The values of one template placeholder, one per item.
///
/// Integers are written in decimal. Strings are written as the dataset's location needs:
/// percent-encoded in a URL, as they are in a file path or object key, where a value cannot
/// hold `/` (nor `\` or NUL in a path) and cannot be `.` or `..`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Column {
    /// Signed integers.
    I64(Vec<i64>),
    /// Unsigned integers.
    U64(Vec<u64>),
    /// Strings.
    Str(Vec<String>),
}

impl From<Vec<i64>> for Column {
    fn from(values: Vec<i64>) -> Self {
        Self::I64(values)
    }
}

impl From<Vec<u64>> for Column {
    fn from(values: Vec<u64>) -> Self {
        Self::U64(values)
    }
}

impl From<Vec<String>> for Column {
    fn from(values: Vec<String>) -> Self {
        Self::Str(values)
    }
}

impl From<Vec<&str>> for Column {
    fn from(values: Vec<&str>) -> Self {
        Self::Str(values.into_iter().map(str::to_owned).collect())
    }
}

impl Column {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::Str(values) => values.len(),
        }
    }

    /// The longest value once written; refuses the values `escape` refuses.
    pub(super) fn max_bytes(&self, name: &str, escape: Escape) -> Result<usize> {
        let Self::Str(values) = self else {
            return Ok(MAX_INT_BYTES);
        };
        let mut longest = 0;
        for (row, value) in values.iter().enumerate() {
            if let Some(reason) = escape.refuse(value) {
                return Err(format!("column {name:?}, item {row}: {value:?} {reason}").into());
            }
            longest = longest.max(escape.len(value));
        }
        Ok(longest)
    }

    pub(super) fn write(&self, row: usize, escape: Escape, out: &mut String) {
        // Writing to a String cannot fail.
        match self {
            Self::I64(values) => _ = write!(out, "{}", values[row]),
            Self::U64(values) => _ = write!(out, "{}", values[row]),
            Self::Str(values) => escape.write(&values[row], out),
        }
    }
}
