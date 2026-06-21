# Proxima

A **vector database / approximate-nearest-neighbor (ANN) search engine** written in Rust.
Proxima stores high-dimensional embeddings and answers *k*-nearest-neighbor queries — "give me
the K vectors most similar to this one" — the backbone of RAG, semantic search, recommendation,
and agent memory.

It is the successor to **[Bedrock](https://github.com/MatheusRodriguesST/Bedrock)** (a durable
storage engine), built in the same spirit: **learning systems by implementing them**, now in the
domain that backend work values most — AI infrastructure. Under the hood a vector database *is* a
database, so Proxima reuses Bedrock as its durable persistence layer and adds the new, hard part on
top: the ANN index.

> **Status: early development.** This repository is currently scaffolding only — no engine code yet.
> The plan, the design decisions, and the step-by-step roadmap live in [`ROADMAP.md`](ROADMAP.md)
> and `CLAUDE.md`. Nothing below is implemented; this README states the *goal*, and will grow into
> honest, measured documentation (with real recall/latency numbers) as each step lands.

## The plan, in one paragraph

Start with **exact brute-force kNN** (cosine and L2 distance) — slow and O(N), but recall = 100% by
definition, which makes it the honest baseline to measure against. Then build the real index:
**HNSW** (Hierarchical Navigable Small World), the industry-standard ANN graph. The whole point is
the trade-off every vector database lives on — **recall vs latency vs memory** — measured, plotted,
and compared against a reference (FAISS / hnswlib) on a standard dataset, never just asserted.

## Planned architecture

- **`crates/core`** — `Vector`, distance metrics (L2, cosine), and the vector store.
- **`crates/index`** — the ANN index: brute-force first, then HNSW.
- **`crates/server`** — an HTTP API, hand-rolled over `std::net` (same no-framework stance as Bedrock).
- **`crates/bench`** — a recall×latency harness and dataset loaders (SIFT / GloVe).
- **`playground/viz`** — a future 3D graph visualizer (late step; isolated from the core).

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

## Reference

- HNSW: Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs"* (2016/2018).
- *Designing Data-Intensive Applications* (Kleppmann), ch. 3 — Storage and Retrieval.
- ANN-Benchmarks (Bernhardsson) — how recall×QPS is measured honestly.

## License

TBD.
