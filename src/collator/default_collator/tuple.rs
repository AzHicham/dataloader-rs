use crate::{collator::default_collator::DefaultCollate, error::Result};

macro_rules! tuple_impl {
    ($($name:ident),+ ; $($idx:tt),+) => {
        impl<$($name),+> DefaultCollate for ($($name,)+)
        where
            $($name: DefaultCollate),+
        {
            type Batch = ($($name::Batch,)+);

            fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
                let mut cols: ($(Vec<$name>,)+) = (
                    $(Vec::<$name>::with_capacity(items.len()),)+
                );

                for item in items {
                    $(
                        cols.$idx.push(item.$idx);
                    )+
                }

                Ok((
                    $($name::collate_items(cols.$idx)?,)+
                ))
            }
        }
    };
}

tuple_impl!(A ; 0);
tuple_impl!(A, B ; 0, 1);
tuple_impl!(A, B, C ; 0, 1, 2);
tuple_impl!(A, B, C, D ; 0, 1, 2, 3);
tuple_impl!(A, B, C, D, E ; 0, 1, 2, 3, 4);
tuple_impl!(A, B, C, D, E, F ; 0, 1, 2, 3, 4, 5);
