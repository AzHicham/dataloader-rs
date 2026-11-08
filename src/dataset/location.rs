//! [`Locations`]: where item `i` of a built-in dataset lives (a URL, a path): a list, a
//! template, or a function of the index.

use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::sync::Arc;

use super::template::Template;
use crate::{dataset::FetchError, error::Result};

/// The function of a dataset made with `url_fn` or `path_fn`.
pub(crate) type LocationFn = Arc<dyn Fn(usize) -> String + Send + Sync>;

/// The locations of the items of a dataset, of type `L` once parsed.
#[derive(Clone)]
pub(crate) enum Locations<L> {
    /// Parsed when the dataset is built.
    List(Vec<L>),
    /// Made and parsed when an item is fetched.
    Template(Arc<Template>),
    /// Made by the function and parsed when an item is fetched.
    Fn { len: usize, location: LocationFn },
}

impl<L: Clone> Locations<L> {
    /// Locations made from `template`, of which the first is checked with `parse`.
    pub(crate) fn template(template: Template, parse: impl Fn(&str) -> Result<L>) -> Result<Self> {
        if template.len() > 0 {
            parse(&template.render(0)?)?;
        }
        Ok(Self::Template(Arc::new(template)))
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::List(locations) => locations.len(),
            Self::Template(template) => template.len(),
            Self::Fn { len, .. } => *len,
        }
    }

    /// The location of item `index`: the listed one, or the one made for it, parsed with
    /// `parse`.
    pub(crate) fn get(
        &self,
        index: usize,
        parse: impl FnOnce(&str) -> Result<L>,
    ) -> Result<Cow<'_, L>> {
        let len = self.len();
        if index >= len {
            return Err(FetchError::OutOfRange { index, len }.into());
        }
        match self {
            Self::List(locations) => Ok(Cow::Borrowed(&locations[index])),
            Self::Template(template) => parse(&template.render(index)?).map(Cow::Owned),
            Self::Fn { location, .. } => parse(&location(index)).map(Cow::Owned),
        }
    }
}

impl<L> Debug for Locations<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::List(locations) => write!(f, "List({} items)", locations.len()),
            Self::Template(template) => write!(f, "Template({} items)", template.len()),
            Self::Fn { len, .. } => write!(f, "Fn({len} items)"),
        }
    }
}
