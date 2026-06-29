//! Brute-force (flat) kNN index — the exact baseline.
//!
//! It compares a query against every stored vector and returns the `k` closest.
//! O(N·d) per query: too slow at scale, but exact by construction (recall = 1.0),
//! so it is the oracle the approximate HNSW index will be measured against.
//! (FAISS calls this a "Flat" index.)

use proxima_core::{Metric, Vector};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub mod nsw;
pub use nsw::NswIndex;

/// A search result: the id of a stored vector and its distance to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u64,
    pub distance: f32,
}

/// A flat collection of vectors searchable by exact kNN.
///
/// Stores `(id, vector)` pairs and assigns monotonically increasing ids on
/// insert. The metric is supplied per [`search`](BruteForceIndex::search) rather
/// than owned, so the same data can be queried under different metrics.
#[derive(Default)]
pub struct BruteForceIndex {
    entries: Vec<(u64, Vector)>,
    next_id: u64,
}

impl BruteForceIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a vector and return its assigned id.
    pub fn insert(&mut self, vector: Vector) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push((id, vector));
        id
    }

    /// Remove the vector with `id`. Returns whether it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.entries.iter().position(|(i, _)| *i == id) {
            self.entries.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Replace the vector stored under `id`. Returns whether it existed.
    pub fn update(&mut self, id: u64, vector: Vector) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|(i, _)| *i == id) {
            entry.1 = vector;
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, id: u64) -> Option<&Vector> {
        self.entries.iter().find(|(i, _)| *i == id).map(|(_, v)| v)
    }

    /// Iterate over `(id, vector)` pairs in storage order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &Vector)> {
        self.entries.iter().map(|(id, v)| (*id, v))
    }

    /// Exact kNN: the `k` stored vectors closest to `query` under `metric`,
    /// sorted nearest first.
    ///
    /// Keeps a bounded max-heap of size `k` so memory is O(k), not O(N): a
    /// candidate enters only if it beats the current worst, and the worst is
    /// evicted. This is the "don't do unnecessary work" idea HNSW takes further.
    pub fn search<M: Metric>(&self, metric: &M, query: &[f32], k: usize) -> Vec<Neighbor> {
        if k == 0 {
            return Vec::new();
        }
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k);
        for (id, vector) in &self.entries {
            let distance = metric.distance(query, vector);
            let cand = Candidate { id: *id, distance };
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(worst) = heap.peek() {
                // The heap's top is the largest distance kept so far.
                if cand < *worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }
        let mut out: Vec<Neighbor> = heap
            .into_iter()
            .map(|c| Neighbor {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        out.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        out
    }
}

/// Heap entry ordered by distance. `Ord` makes [`BinaryHeap`] a max-heap on
/// distance, so its top is the worst (farthest) candidate currently kept —
/// the one to evict when a closer one arrives.
struct Candidate {
    id: u64,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proxima_core::L2;

    fn build() -> (BruteForceIndex, [u64; 4]) {
        let mut idx = BruteForceIndex::new();
        let ids = [
            idx.insert(Vector::from([0.0, 0.0])),
            idx.insert(Vector::from([1.0, 0.0])),
            idx.insert(Vector::from([5.0, 0.0])),
            idx.insert(Vector::from([0.0, 10.0])),
        ];
        (idx, ids)
    }

    #[test]
    fn returns_k_nearest_sorted() {
        let (idx, ids) = build();
        let res = idx.search(&L2, &[0.0, 0.0], 2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, ids[0]); // (0,0) itself, distance 0
        assert_eq!(res[1].id, ids[1]); // (1,0), distance 1
        assert!(res[0].distance <= res[1].distance);
    }

    #[test]
    fn k_larger_than_len_returns_all() {
        let (idx, _) = build();
        assert_eq!(idx.search(&L2, &[0.0, 0.0], 99).len(), 4);
    }

    #[test]
    fn k_zero_returns_empty() {
        let (idx, _) = build();
        assert!(idx.search(&L2, &[0.0, 0.0], 0).is_empty());
    }

    #[test]
    fn update_moves_the_vector() {
        let (mut idx, ids) = build();
        assert!(idx.update(ids[2], Vector::from([0.5, 0.0]))); // move (5,0) -> (0.5,0)
        let res = idx.search(&L2, &[0.0, 0.0], 2);
        assert_eq!(res[0].id, ids[0]); // (0,0)
        assert_eq!(res[1].id, ids[2]); // now (0.5,0), closer than (1,0)
        assert!(!idx.update(99, Vector::from([0.0, 0.0]))); // unknown id
    }

    #[test]
    fn remove_then_absent_from_results() {
        let (mut idx, ids) = build();
        assert!(idx.remove(ids[1])); // drop (1,0)
        let res = idx.search(&L2, &[0.0, 0.0], 2);
        assert_eq!(res[0].id, ids[0]);
        assert_eq!(res[1].id, ids[2]); // (5,0) now the runner-up
        assert!(!idx.remove(ids[1])); // already gone
    }
}
