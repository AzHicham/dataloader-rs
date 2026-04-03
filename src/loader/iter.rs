use std::marker::PhantomData;

use crossbeam_channel::Receiver;

use crate::error::Result;

/// Iterator over one epoch of batches, produced by `DataLoader::iter`.
pub struct DataLoaderIter<'a, B> {
    pub(super) rx: Option<Receiver<Result<B>>>,
    pub(super) remaining: usize,
    pub(super) handle: Option<std::thread::JoinHandle<()>>,
    pub(super) _borrow: PhantomData<&'a mut ()>,
}

impl<'a, B: Send + 'static> Iterator for DataLoaderIter<'a, B> {
    type Item = Result<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let rx = self.rx.as_ref()?;
        match rx.recv() {
            Ok(batch) => {
                self.remaining -= 1;
                Some(batch)
            }
            Err(_) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, B: Send + 'static> ExactSizeIterator for DataLoaderIter<'a, B> {}

impl<'a, B> Drop for DataLoaderIter<'a, B> {
    fn drop(&mut self) {
        drop(self.rx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
