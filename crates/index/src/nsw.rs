//! NSW (Navigable Small World) — a single-layer proximity graph index.

use proxima_core::Vector;

struct Node {
    id: u64,
    vector: Vector,
    #[allow(dead_code)]
    neighbors: Vec<usize>,
}

pub struct NswIndex {
    nodes: Vec<Node>,
    entry: Option<usize>,
    next_id: u64,
    #[allow(dead_code)]
    m: usize,
    #[allow(dead_code)]
    ef_construction: usize,
}

impl NswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry: None,
            next_id: 0,
            m,
            ef_construction,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get(&self, id: u64) -> Option<&Vector> {
        self.nodes.iter().find(|n| n.id == id).map(|n| &n.vector)
    }

    pub fn insert(&mut self, vector: Vector) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let index = self.nodes.len();
        self.nodes.push(Node {
            id,
            vector,
            neighbors: Vec::new(),
        });

        if self.entry.is_none() {
            self.entry = Some(index);
        }

        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index_has_no_entry_and_zero_len() {
        let idx = NswIndex::new(4, 16);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn insert_assigns_increasing_ids_and_grows_len() {
        let mut idx = NswIndex::new(4, 16);
        let a = idx.insert(Vector::from([0.0, 0.0]));
        let b = idx.insert(Vector::from([1.0, 0.0]));
        assert_eq!((a, b), (0, 1));
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn get_returns_the_stored_vector() {
        let mut idx = NswIndex::new(4, 16);
        let id = idx.insert(Vector::from([3.0, 4.0]));
        assert_eq!(idx.get(id), Some(&Vector::from([3.0, 4.0])));
        assert_eq!(idx.get(999), None);
    }
}
