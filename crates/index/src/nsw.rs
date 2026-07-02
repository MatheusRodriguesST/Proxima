//! NSW (Navigable Small World) — a single-layer proximity graph index.

use crate::Neighbor;
use proxima_core::{Metric, Vector};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

struct Node {
    id: u64,
    vector: Vector,
    neighbors: Vec<usize>,
}

/// A node reached during traversal, scored by distance to the query.
/// `Ord` by distance makes a [`BinaryHeap`] of these a max-heap (farthest on
/// top) — exactly what `SEARCH-LAYER`'s `result` beam needs to evict the
/// worst candidate. Wrap in [`Reverse`] to get a min-heap for `candidates`.
struct Scored {
    index: usize,
    distance: f32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
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

    /// Greedy beam search within the graph (Alg. 2, `SEARCH-LAYER`).
    ///
    /// Starting from `entry_points`, repeatedly expands the closest
    /// unvisited candidate's neighbors, keeping a beam of the `ef` best
    /// nodes seen (`result`). Stops as soon as the closest remaining
    /// candidate is farther than the worst node currently kept — nothing
    /// better can come from expanding it further.
    fn search_layer<M: Metric>(
        &self,
        metric: &M,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
    ) -> Vec<Scored> {
        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();
        let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        let mut result: BinaryHeap<Scored> = BinaryHeap::new();

        for &ep in entry_points {
            let distance = metric.distance(query, &self.nodes[ep].vector);
            candidates.push(Reverse(Scored {
                index: ep,
                distance,
            }));
            result.push(Scored {
                index: ep,
                distance,
            });
        }

        while let Some(Reverse(closest)) = candidates.pop() {
            if let Some(worst) = result.peek() {
                if closest.distance > worst.distance {
                    break;
                }
            }

            for &neighbor in &self.nodes[closest.index].neighbors {
                if !visited.insert(neighbor) {
                    continue;
                }
                let distance = metric.distance(query, &self.nodes[neighbor].vector);
                let better_than_worst = result.peek().is_none_or(|worst| distance < worst.distance);
                if result.len() < ef || better_than_worst {
                    candidates.push(Reverse(Scored {
                        index: neighbor,
                        distance,
                    }));
                    result.push(Scored {
                        index: neighbor,
                        distance,
                    });
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        let mut out: Vec<Scored> = result.into_vec();
        out.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        out
    }

    /// Approximate kNN: greedy beam search from the entry point, widened by
    /// `ef_search` (the recall×latency knob), truncated to the `k` best.
    pub fn search<M: Metric>(
        &self,
        metric: &M,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Vec<Neighbor> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        if k == 0 {
            return Vec::new();
        }
        let ef = ef_search.max(k);
        let mut results = self.search_layer(metric, query, &[entry], ef);
        results.truncate(k);
        results
            .into_iter()
            .map(|c| Neighbor {
                id: self.nodes[c.index].id,
                distance: c.distance,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BruteForceIndex;
    use proxima_core::L2;

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

    #[test]
    fn search_on_empty_index_returns_empty() {
        let idx = NswIndex::new(4, 16);
        assert!(idx.search(&L2, &[0.0, 0.0], 3, 10).is_empty());
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let mut idx = NswIndex::new(4, 16);
        idx.insert(Vector::from([0.0, 0.0]));
        assert!(idx.search(&L2, &[0.0, 0.0], 0, 10).is_empty());
    }

    #[test]
    fn search_finds_the_exact_nearest_in_a_complete_graph() {
        let mut idx = NswIndex::new(4, 16);
        idx.insert(Vector::from([0.0, 0.0]));
        idx.insert(Vector::from([10.0, 0.0]));
        let closest = idx.insert(Vector::from([1.0, 0.0]));

        let res = idx.search(&L2, &[1.5, 0.0], 1, 10);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, closest);
    }

    // In a complete graph every node is one hop from every other, so greedy
    // beam search is exact: it must agree with the brute-force oracle
    // (etapa 3's whole point — validate SEARCH-LAYER before etapa 4
    // sparsifies the graph and recall can drop for a different reason).
    #[test]
    fn search_matches_brute_force_recall_1_on_complete_graph() {
        let mut nsw = NswIndex::new(4, 16);
        let mut brute = BruteForceIndex::new();
        // Colinear, distinct-spacing points: every distance to a
        // non-integer query is unique, so there are no tie-breaks to
        // worry about between the two indexes.
        for i in 0..15 {
            let v = Vector::from([i as f32, 0.0]);
            nsw.insert(v.clone());
            brute.insert(v);
        }

        for query in [[4.3, 0.0], [0.2, 0.0], [13.9, 0.0]] {
            for k in [1, 3, 5] {
                let expected = brute.search(&L2, &query, k);
                let got = nsw.search(&L2, &query, k, 15);
                assert_eq!(
                    got.iter().map(|n| n.id).collect::<Vec<_>>(),
                    expected.iter().map(|n| n.id).collect::<Vec<_>>(),
                    "query {query:?}, k={k}"
                );
            }
        }
    }

    #[test]
    fn search_result_is_sorted_nearest_first() {
        let mut idx = NswIndex::new(4, 16);
        for i in 0..6 {
            idx.insert(Vector::from([i as f32, 0.0]));
        }
        let res = idx.search(&L2, &[0.0, 0.0], 4, 10);
        for pair in res.windows(2) {
            assert!(pair[0].distance <= pair[1].distance);
        }
    }
}
