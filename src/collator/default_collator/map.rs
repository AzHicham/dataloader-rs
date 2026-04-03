use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hash};

use crate::{collator::default_collator::DefaultCollate, error::Result};

impl<K, V, H> DefaultCollate for HashMap<K, V, H>
where
    K: Eq + Hash + Send + 'static,
    V: DefaultCollate,
    H: BuildHasher + Default + Send + 'static,
{
    type Batch = HashMap<K, V::Batch, H>;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let mut iter = items.into_iter();
        let Some(first) = iter.next() else {
            return Ok(HashMap::with_hasher(H::default()));
        };

        let mut buckets: HashMap<K, Vec<V>, H> = HashMap::with_hasher(H::default());
        for (k, v) in first {
            buckets.insert(k, vec![v]);
        }

        for map in iter {
            if map.len() != buckets.len() {
                return Err("all maps in batch must have the same keys".into());
            }
            for (k, v) in map {
                match buckets.get_mut(&k) {
                    Some(bucket) => bucket.push(v),
                    None => return Err("all maps in batch must have the same keys".into()),
                }
            }
        }

        let mut out = HashMap::with_hasher(H::default());
        for (k, vals) in buckets {
            out.insert(k, V::collate_items(vals)?);
        }
        Ok(out)
    }
}

impl<K, V> DefaultCollate for BTreeMap<K, V>
where
    K: Ord + Send + 'static,
    V: DefaultCollate,
{
    type Batch = BTreeMap<K, V::Batch>;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let mut iter = items.into_iter();
        let Some(first) = iter.next() else {
            return Ok(BTreeMap::new());
        };

        let mut buckets: BTreeMap<K, Vec<V>> = BTreeMap::new();
        for (k, v) in first {
            buckets.insert(k, vec![v]);
        }

        for map in iter {
            if map.len() != buckets.len() {
                return Err("all maps in batch must have the same keys".into());
            }
            for (k, v) in map {
                match buckets.get_mut(&k) {
                    Some(bucket) => bucket.push(v),
                    None => return Err("all maps in batch must have the same keys".into()),
                }
            }
        }

        let mut out = BTreeMap::new();
        for (k, vals) in buckets {
            out.insert(k, V::collate_items(vals)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::collator::{Collator, DefaultCollator};

    #[test]
    fn collates_hashmaps_by_key() {
        let c = DefaultCollator;
        let m1 = HashMap::from([("x", 1u32), ("y", 2u32)]);
        let m2 = HashMap::from([("x", 10u32), ("y", 20u32)]);
        let out = c.collate(vec![m1, m2]).unwrap();
        assert_eq!(out["x"], vec![1, 10]);
        assert_eq!(out["y"], vec![2, 20]);
    }

    #[test]
    fn hashmaps_with_different_keys_fail() {
        let c = DefaultCollator;
        let m1 = HashMap::from([("x", 1u32)]);
        let m2 = HashMap::from([("y", 2u32)]);
        assert!(c.collate(vec![m1, m2]).is_err());
    }
}
