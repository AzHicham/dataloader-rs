use crate::{collator::default_collator::DefaultCollate, error::Result};

macro_rules! primitive_impl {
    ($($t:ty),* $(,)?) => {
        $(
            impl DefaultCollate for $t {
                type Batch = Vec<$t>;
                fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
                    Ok(items)
                }
            }
        )*
    };
}

primitive_impl!(
    bool, char,
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
    String
);

#[cfg(test)]
mod tests {
    use crate::collator::{Collator, DefaultCollator};

    #[test]
    fn collates_primitives_to_vec() {
        let c = DefaultCollator;
        let out = c.collate(vec![1u32, 2, 3]).unwrap();
        assert_eq!(out, vec![1, 2, 3]);
    }
}
