//! NSW (Navigable Small World) — a single-layer proximity graph index.

use proxima_core::Vector;

struct Node {
    id: u64,
    vector: Vector,
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
        let neighbors: Vec<usize> = (0..index).collect();
        for existing in &mut self.nodes {
            existing.neighbors.push(index);
        }
        self.nodes.push(Node {
            id,
            vector,
            neighbors,
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

    #[test]
    fn first_node_has_no_neighbors() {
        let mut idx = NswIndex::new(4, 16);
        idx.insert(Vector::from([0.0, 0.0]));
        assert!(idx.nodes[0].neighbors.is_empty());
    }

    #[test]
    fn insert_builds_a_complete_graph() {
        let mut idx = NswIndex::new(4, 16);
        idx.insert(Vector::from([0.0, 0.0]));
        idx.insert(Vector::from([1.0, 0.0]));
        idx.insert(Vector::from([2.0, 0.0]));

        // K3: every node is linked to the other two.
        for node in &idx.nodes {
            assert_eq!(node.neighbors.len(), 2);
        }
        // Edges are bidirectional.
        assert!(idx.nodes[0].neighbors.contains(&2));
        assert!(idx.nodes[2].neighbors.contains(&0));
    }
}
