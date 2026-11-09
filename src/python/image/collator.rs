//! `ImageCollator` in Python: images into one `ImageBatch`, in Rust. As the loader's
//! `collate_fn` it collates the Rust transforms' images without the interpreter.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::batch::{Batch, PyImage, PyImageBatch};
use super::ops::Value;
use crate::collator::{BatchLayout, ImageCollator};
use crate::error::Result;
use crate::image::{Image, Layout, Sample, invalid};
use crate::python::error::into_py_err;

/// An `ImageCollator`'s settings, as the loader runs them.
#[derive(Clone)]
pub(crate) struct ImageCollate {
    collator: ImageCollator,
    /// The sample type items must have, if set.
    dtype: Option<&'static str>,
}

/// The images of `values`, which must all be of the kind `pick` takes.
fn same<'a, T: Sample, L: Layout>(
    values: &[&'a Value],
    pick: impl Fn(&'a Value) -> Option<&'a Image<T, L>>,
) -> Result<Vec<&'a Image<T, L>>> {
    let first = values[0].kind();
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            pick(value).ok_or_else(|| {
                invalid(format!(
                    "item {i} of the batch is {}, item 0 is {first}: the images of a batch \
                     must have the same type and layout",
                    value.kind()
                ))
            })
        })
        .collect()
}

impl ImageCollate {
    /// Collate the images of `values`.
    pub(crate) fn collate(&self, values: &[&Value]) -> Result<Batch> {
        let Some(first) = values.first() else {
            return Err(invalid("an empty batch has no shape".into()));
        };
        let dtype = match first {
            Value::U8(_) | Value::U8Chw(_) => u8::NAME,
            _ => f32::NAME,
        };
        if let Some(expected) = self.dtype.filter(|&expected| expected != dtype) {
            return Err(invalid(format!(
                "ImageCollator(dtype={expected:?}) got {dtype} images (ToFloat makes float32)"
            )));
        }
        let c = &self.collator;
        Ok(match first {
            Value::U8(_) => Batch::U8(c.collate_refs(&same(values, |v| match v {
                Value::U8(i) => Some(i),
                _ => None,
            })?)?),
            Value::U8Chw(_) => Batch::U8(c.collate_refs(&same(values, |v| match v {
                Value::U8Chw(i) => Some(i),
                _ => None,
            })?)?),
            Value::F32(_) => Batch::F32(c.collate_refs(&same(values, |v| match v {
                Value::F32(i) => Some(i),
                _ => None,
            })?)?),
            Value::F32Chw(_) => Batch::F32(c.collate_refs(&same(values, |v| match v {
                Value::F32Chw(i) => Some(i),
                _ => None,
            })?)?),
            Value::Bytes(_) => {
                return Err(invalid(
                    "ImageCollator takes images, not bytes (add Decode to the transform)".into(),
                ));
            }
        })
    }

    /// Collate Python items: `Image`s into an `ImageBatch`, or tuples `(Image, a, b, ...)`
    /// into `(ImageBatch, [a...], [b...], ...)`. The copy runs with the interpreter released.
    pub(crate) fn collate_objects(
        &self,
        py: Python<'_>,
        items: &[Bound<'_, PyAny>],
    ) -> PyResult<Py<PyAny>> {
        let tuples = items
            .first()
            .is_some_and(|item| item.is_instance_of::<PyTuple>());
        let mut images = Vec::with_capacity(items.len());
        let mut columns: Vec<Vec<Bound<'_, PyAny>>> = Vec::new();
        for item in items {
            let image = match tuples {
                false => item.clone(),
                true => {
                    let tuple = item.cast::<PyTuple>().map_err(|_| {
                        PyTypeError::new_err("ImageCollator: either every item is a tuple, or none")
                    })?;
                    let mut fields = tuple.iter();
                    let image = fields.next().ok_or_else(|| {
                        PyValueError::new_err("ImageCollator: an empty tuple has no image")
                    })?;
                    let rest: Vec<_> = fields.collect();
                    if columns.is_empty() {
                        columns = vec![Vec::with_capacity(items.len()); rest.len()];
                    }
                    if rest.len() != columns.len() {
                        return Err(PyValueError::new_err(
                            "ImageCollator: every tuple of a batch must have the same length",
                        ));
                    }
                    for (column, field) in columns.iter_mut().zip(rest) {
                        column.push(field);
                    }
                    image
                }
            };
            let image = image.cast_into::<PyImage>().map_err(|e| {
                PyTypeError::new_err(format!(
                    "ImageCollator collates dataloader_rs Images (from dataloader_rs.transforms) \
                     or tuples starting with one, not {}",
                    e.into_inner()
                        .get_type()
                        .name()
                        .map_or_else(|_| "?".into(), |n| n.to_string())
                ))
            })?;
            images.push(image);
        }
        let values: Vec<&Value> = images.iter().map(|image| image.get().value()).collect();
        let batch = py.detach(|| self.collate(&values)).map_err(into_py_err)?;
        let batch = Py::new(py, PyImageBatch::new(batch))?.into_any();
        if !tuples {
            return Ok(batch);
        }
        let mut fields = vec![batch];
        for column in columns {
            fields.push(PyList::new(py, column)?.into_any().unbind());
        }
        Ok(PyTuple::new(py, fields)?.into_any().unbind())
    }
}

/// Collates images from `dataloader_rs.transforms` into one `ImageBatch`: `layout` `"NCHW"`
/// (default) or `"NHWC"`; `dtype` `"uint8"` or `"float32"` checks the items' type (default:
/// theirs); `buffers` keeps that many dropped batches' buffers for the next ones.
#[pyclass(frozen, name = "ImageCollator", module = "dataloader_rs")]
pub struct PyImageCollator {
    collate: ImageCollate,
}

impl PyImageCollator {
    /// The settings, for the loader.
    pub(crate) fn collate(&self) -> ImageCollate {
        self.collate.clone()
    }
}

#[pymethods]
impl PyImageCollator {
    #[new]
    #[pyo3(signature = (layout="NCHW", dtype=None, buffers=0))]
    fn new(layout: &str, dtype: Option<&str>, buffers: usize) -> PyResult<Self> {
        let layout = match layout {
            "NCHW" => BatchLayout::Nchw,
            "NHWC" => BatchLayout::Nhwc,
            other => {
                return Err(PyValueError::new_err(format!(
                    "layout must be \"NCHW\" or \"NHWC\", not {other:?}"
                )));
            }
        };
        let dtype = match dtype {
            None => None,
            Some("uint8") => Some(u8::NAME),
            Some("float32") => Some(f32::NAME),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "dtype must be \"uint8\" or \"float32\", not {other:?}"
                )));
            }
        };
        Ok(Self {
            collate: ImageCollate {
                collator: ImageCollator::new(layout).reuse_buffers(buffers),
                dtype,
            },
        })
    }

    /// Collate `items`, a list of `Image`s or of tuples starting with one.
    fn __call__(&self, py: Python<'_>, items: Vec<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        self.collate.collate_objects(py, &items)
    }

    fn __repr__(&self) -> String {
        let dtype = self
            .collate
            .dtype
            .map_or("None".into(), |d| format!("'{d}'"));
        let layout = self.collate.collator.layout().name();
        format!("ImageCollator(layout='{layout}', dtype={dtype})")
    }
}
