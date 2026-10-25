use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hash};

use crate::{collator::default_collator::DefaultCollate, error::Result};

const DIFFERENT_KEYS: &str = "all maps in batch must have the same keys";

// Both impls transpose the batch into one column of values per key, then
// collate each column; they differ only in the map type.

impl<K, V, H> DefaultCollate for HashMap<K, V, H>
where
    K: Eq + Hash + Send + 'static,
    V: DefaultCollate,
    H: BuildHasher + Default + Send + 'static,
{
    type Batch = HashMap<K, V::Batch, H>;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let rows = items.len();
        let mut maps = items.into_iter();
        let mut columns: HashMap<K, Vec<V>, H> = match maps.next() {
            Some(first) => first
                .into_iter()
                .map(|(k, v)| (k, column(rows, v)))
                .collect(),
            None => HashMap::default(),
        };
        for map in maps {
            if map.len() != columns.len() {
                return Err(DIFFERENT_KEYS.into());
            }
            for (k, v) in map {
                columns.get_mut(&k).ok_or(DIFFERENT_KEYS)?.push(v);
            }
        }
        columns
            .into_iter()
            .map(|(k, values)| Ok((k, V::collate_items(values)?)))
            .collect()
    }
}

impl<K, V> DefaultCollate for BTreeMap<K, V>
where
    K: Ord + Send + 'static,
    V: DefaultCollate,
{
    type Batch = BTreeMap<K, V::Batch>;

    fn collate_items(items: Vec<Self>) -> Result<Self::Batch> {
        let rows = items.len();
        let mut maps = items.into_iter();
        let mut columns: BTreeMap<K, Vec<V>> = match maps.next() {
            Some(first) => first
                .into_iter()
                .map(|(k, v)| (k, column(rows, v)))
                .collect(),
            None => BTreeMap::new(),
        };
        for map in maps {
            if map.len() != columns.len() {
                return Err(DIFFERENT_KEYS.into());
            }
            for (k, v) in map {
                columns.get_mut(&k).ok_or(DIFFERENT_KEYS)?.push(v);
            }
        }
        columns
            .into_iter()
            .map(|(k, values)| Ok((k, V::collate_items(values)?)))
            .collect()
    }
}

/// A key's column, sized for the whole batch, starting with `first`.
fn column<V>(rows: usize, first: V) -> Vec<V> {
    let mut column = Vec::with_capacity(rows);
    column.push(first);
    column
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

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

    #[test]
    fn collates_btreemaps_by_key() {
        let c = DefaultCollator;
        let m1 = BTreeMap::from([("x", 1u32), ("y", 2u32)]);
        let m2 = BTreeMap::from([("x", 10u32), ("y", 20u32)]);
        let out = c.collate(vec![m1, m2]).unwrap();
        assert_eq!(out["x"], vec![1, 10]);
        assert_eq!(out["y"], vec![2, 20]);
    }

    #[test]
    fn btreemaps_with_other_keys_fail() {
        let c = DefaultCollator;
        let m1 = BTreeMap::from([("x", 1u32), ("y", 2u32)]);
        let m2 = BTreeMap::from([("x", 10u32), ("z", 20u32)]);
        assert!(c.collate(vec![m1, m2]).is_err());
        let m3 = BTreeMap::from([("x", 1u32)]);
        let m4 = BTreeMap::from([("x", 10u32), ("y", 20u32)]);
        assert!(c.collate(vec![m3, m4]).is_err());
    }
}
