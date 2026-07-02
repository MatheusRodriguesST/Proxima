# Proxima

A **vector database / approximate-nearest-neighbor (ANN) search engine** written in Rust.
Proxima stores high-dimensional embeddings and answers *k*-nearest-neighbor queries — "give me
the K vectors most similar to this one" — the backbone of RAG, semantic search, recommendation,
and agent memory.

This is the third project in my systems portfolio and the successor to
**[Bedrock](https://github.com/MatheusRodriguesST/Bedrock)**, a durable storage engine. I build
these to sharpen my craft the only way that sticks: **learning systems by implementing them** — and
this time in the domain backend work values most, AI infrastructure. A vector database *is* a
database underneath, so Proxima will reuse Bedrock as its durable persistence layer and builds the
new, hard part on top: the ANN index. That continuity is deliberate — first I built the storage, now I
build the engine that searches it.

## The name

**Proxima**, from the Latin *proximus*, "the nearest" — which is exactly what nearest-neighbor
search computes: of all the vectors I hold, the ones closest to yours. The word also points at
*Proxima Centauri*, the nearest star to our own. Same idea at two scales — the closest point in a
space, whether that space has three dimensions or three hundred.

## The plan, in one paragraph

Start with **exact brute-force kNN** (cosine and L2 distance) — slow and O(N), but recall = 100% by
definition, which makes it the honest baseline to measure against. Then build the real index:
**HNSW** (Hierarchical Navigable Small World), the industry-standard ANN graph. The whole point is
the trade-off every vector database lives on — **recall vs latency vs memory** — measured, plotted,
and compared against a reference (FAISS / hnswlib) on a standard dataset, never just asserted.

## Status

Early, but no longer empty. Honest snapshot of what actually runs today:

- ✅ **`crates/core`** — `Vector` type and distance metrics (L2, cosine), with closed-form tests.
- ✅ **`crates/index`** — `BruteForceIndex`: exact kNN over a `BinaryHeap`. This is the recall = 1.0
  oracle the approximate index will be measured against (FAISS calls this a "Flat" index).
- ✅ **`playground/viz`** — an interactive, animated graph visualizer that drives the real engine:
  add points, run a search, watch the O(N) scan and the *k* nearest light up with real distances.
- 🔨 **In progress** — the NSW graph (the single-layer step toward HNSW), built in verifiable
  stages against the brute-force oracle. Persistence via Bedrock comes after the graph works.

No recall-vs-latency numbers exist yet, because there is no approximate index yet. When there is,
this README will state real measured numbers with their conditions — never "it's fast" on its own.

## Planned architecture

- **`crates/core`** — `Vector`, distance metrics (L2, cosine), and the vector store.
- **`crates/index`** — the ANN index: brute-force first, then HNSW.
- **`crates/server`** — an HTTP API, hand-rolled over `std::net` (same no-framework stance as Bedrock).
- **`crates/bench`** — a recall×latency harness and dataset loaders (SIFT / GloVe).
- **`playground/viz`** — an interactive graph visualizer (will grow to 3D in a late step).

Vectors are persisted durably via Bedrock; the HNSW index lives in memory for fast search and is
rebuilt/recovered from the durable store — disk is for durability and cold start, not the hot path.

## Definition of done (what makes it credible)

The same four deliverables that finished Bedrock, adapted to a vector database:

1. **Published benchmarks** — a recall×latency curve (sweeping `ef_search`), QPS, and memory, compared
   honestly against FAISS or hnswlib on the same dataset. Plotted, not just tabulated.
2. **Correctness demonstrated** — a reproducible test that measures recall@k against ground truth and
   asserts a floor (e.g. "recall@10 ≥ 0.95"). A vector DB without a measured recall number is just a claim.
3. **README explaining decisions and trade-offs** — why HNSW over IVF/LSH/kd-tree, why the layered
   graph works, the recall×latency×memory trade-off, and the engine's honest limits.
4. **CI** — build + tests + clippy + fmt on every push; ideally the recall benchmark as a regression gate.

The detailed step-by-step plan lives in [`ROADMAP.md`](ROADMAP.md).

## Reference

- HNSW: Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs"* (2016/2018).
- *Designing Data-Intensive Applications* (Kleppmann), ch. 3 — Storage and Retrieval.
- ANN-Benchmarks (Bernhardsson) — how recall×QPS is measured honestly.

## License

MIT — see [LICENSE](LICENSE).
