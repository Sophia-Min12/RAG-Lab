# Day 1 · Chunking — What Actually Goes in the Index

> A retriever does not return documents, it returns **chunks**. That choice is made before any embedding model is loaded, it is almost never revisited, and it caps everything downstream: **a fact split across two chunks can be retrieved by neither**, and no reranker recovers it.

## Three strategies

| Strategy | Knows about | Fails by |
|---|---|---|
| `chunk_fixed(text, size)` | nothing | cutting words, and facts, in half |
| `chunk_fixed(text, size, overlap)` | nothing | costing index size |
| `chunk_sentences(text, max_chars)` | sentence boundaries | uneven chunk sizes |

```
strategy                  chunks  mean len  coverage  duplication
fixed(120)                     4     105.2      100%         1.00
fixed(120, overlap=30)         5     108.2      100%         1.29
fixed(120, overlap=60)         7     111.6      100%         1.86
sentence(120)                  6      69.3       99%         0.99
```

## The trade, measured

The claim "overlap improves recall" is usually asserted. Here both halves of it are numbers. The phrase `"the part that usually fails"` straddles the cut a `fixed(120)` chunker makes at character 120 — nothing unusual about it, which is the point. *Any* boundary lands somewhere.

```
strategy                  survives  duplication
fixed(120)                      NO         1.00
fixed(120, overlap=30)         yes         1.29
fixed(120, overlap=60)         yes         1.86
sentence(120)                  yes         0.99
```

**Overlap buys boundary recall and is paid for in duplication** — in storage, in embedding API calls, and in money. At 25% overlap the index is 29% larger. That is the entire decision.

`sentence(120)` gets it for free, because it never cuts mid-sentence. That is the argument for sentence-aware chunking in one row, and it holds exactly as long as your facts live inside single sentences — which Day 3 will test rather than assume.

One test guards against the demonstration passing for the wrong reason: it asserts that **no** `fixed(120)` chunk contains the phrase, so the "NO" is a real miss rather than a typo.

## Offsets are not optional

Every `Chunk` carries `start` and `end` into the original text, and a test checks `text[chunk.start:chunk.end] == chunk.text` for **every strategy**:

```
[0] chars 0..79   exact match: True
[1] chars 80..125 exact match: True
```

A chunker that loses offsets cannot produce a citation, and a RAG answer that cannot point at its evidence is an answer you have to take on faith. Day 2 turns these into citation spans; keeping them from Day 1 costs nothing and is impossible to retrofit cheaply.

## Two decisions that raise instead of guessing

**`overlap >= size` is refused.** The stride is `size - overlap`; at zero it advances nowhere and loops forever. Clamping silently is how a config typo produces a 40 GB index — so it raises and says why.

**An oversized sentence becomes its own chunk** rather than being split. The entire premise of sentence chunking is that sentence boundaries are the ones worth keeping; breaking one to satisfy a size budget defeats it. The oversized chunk is a real consequence, a test pins it, and the demo reports it rather than hiding it.

## Coverage

`coverage` is the share of original characters appearing in at least one chunk. Anything below 1.0 means the index **cannot** answer questions about the missing span, whatever the embedding model does.

Fixed chunking covers everything. Sentence chunking reaches 99% — and the missing 1% is inter-sentence whitespace, not content. A test asserts that distinction rather than accepting the number.

## Run it

```bash
# demo (strategy comparison, the boundary trade, offsets)
python chunking.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_chunking/day01_chunking

# tests — from inside this folder
python -m unittest
python test_chunking.py

# the docstring examples are runnable too
python -m doctest chunking.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

**Day 2** turns the offsets into citations that survive retrieval and reach the generated answer. **Day 3** stops reasoning about chunk size and measures it against answer quality — because everything above is about what *can* be retrieved, and nothing yet about what *is*.
