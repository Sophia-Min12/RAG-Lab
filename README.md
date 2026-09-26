# RAG-Lab

![tests](https://github.com/Sophia-Min12/RAG-Lab/actions/workflows/tests.yml/badge.svg)

**One day, one concept, one commit — retrieval-augmented generation, built from the retrieval up.**

> Chunking, embedding indexes, approximate nearest neighbours, hybrid reranking, and the evaluation that decides whether any of it helped. The generator comes last, because **most RAG failures are retrieval failures wearing a costume.**

**Environment**: Python 3.10+ · NumPy · `sentence-transformers` from Level 2 · `pytest` as the test runner.

---

## 🧭 The rule this repo is built on

**Nothing is claimed to help until it is measured on held-out queries.** Every retrieval change is scored with `recall@k` and `MRR` against a fixed evaluation set, and every number is reported with the size of the set it came from. A retrieval system that is not evaluated is a retrieval system with an opinion.

The second rule follows from the first: **the baseline is never skipped.** BM25 is a strong, cheap, decades-old baseline, and a dense retriever that cannot beat it has not earned its GPU.

---

## 🗺️ Curriculum Roadmap

**Level 0 · Repo Setup**
- [x] **Day 0** — Repo scaffold, CI, and curriculum roadmap

**Level 1 · Chunking — what actually goes in the index**
- [x] **Day 1** — Chunking strategies: fixed, sentence-aware, and overlap
- [x] **Day 2** — Metadata and citation spans: pointing back at the source
- [x] **Day 3** — Chunk size versus answer quality, measured
- [x] **Day 4** — Documents that break chunkers: tables, code, Korean

**Level 2 · Retrieval**
- [x] **Day 5** — BM25, the baseline that must be beaten
- [x] **Day 6** — Real sentence embeddings (`sentence-transformers` arrives)
- [x] **Day 7** — A vector store from scratch: exact search, and why it stops scaling
- [x] **Day 8** — Approximate nearest neighbours: IVF and HNSW, and what they give up
- [x] **Day 9** — Hybrid retrieval: score fusion versus rank fusion
- [x] **Day 10** — Cross-encoder reranking: slow, accurate, and worth it for the top 50

**Level 3 · Evaluation**
- [x] **Day 11** — `recall@k`, `MRR`, `nDCG`, and which one answers your question
- [x] **Day 12** — Building an evaluation set without hand-labelling everything
- [x] **Day 13** — Ablations: turning each component off and reading the damage

**Level 4 · Generation**
- [ ] **Day 14** — Context packing: order, budget, and the lost-in-the-middle effect
- [ ] **Day 15** — Grounding and citation: making the answer point at its evidence
- [ ] **Day 16** — Measuring hallucination: faithfulness against the retrieved context
- [ ] **Day 17** — Capstone: the full loop, a CLI, and the honest writeup

---

## 📐 Conventions

- Every day folder `NN_topic/dayNN_name/` is **self-contained**: helpers from earlier days are copied forward with a `# reused from dayNN` comment.
- Every code day ships a runnable test; `pytest` from the repo root runs everything (this is what CI runs).
- Tests that need a downloaded model **skip** rather than fail, so CI stays green and fast without pretending the model was tested.

## 🔗 Sibling labs

This repo is the **fusion capstone** of the three before it.

- [NLP-Lab](https://github.com/Sophia-Min12/NLP-Lab) — its **Day 8 inverted index** and **Day 16 hybrid scorer** are the direct ancestors of Level 2 here. Its Day 17 writeup already states what lexical retrieval cannot find; this repo starts from that list.
- [Transformer-Lab](https://github.com/Sophia-Min12/Transformer-Lab) — the attention that a cross-encoder reranker runs on, built from scalars.
- [ArgMin-Lab](https://github.com/Sophia-Min12/ArgMin-Lab) — ANN index construction is a search problem; retrieval tuning is an optimization problem.

## License

[MIT](LICENSE) © Sophia Min
