# Day 7 · A Vector Store From Scratch: Exact Search, and Why It Stops Scaling

> Day 6 searched with a matrix it happened to have lying around. This separates the store from the model and asks what exact search actually costs.

## It is one matrix multiply

```
10,000 vectors x 384 dimensions, 15.4 MB as float32

matrix multiply + argpartition      0.334 ms per query
the same thing as a Python loop    12.745 ms per query  (38x)
identical answers: True
```

**"Brute force is slow" is mostly a statement about the loop.** The same arithmetic through BLAS is not slow at this size — and it is still exact, which is the guarantee Day 8 spends.

`argpartition` finds the top `k` without sorting the rest: `O(n)` against `O(n log n)`. At a million vectors that is the difference between the sort and the multiply dominating.

## Batching, once there is something to batch

```
  queries   one at a time     batched   speedup
        1         2.313ms     2.121ms     1.09x
        8         1.988ms     0.798ms     2.49x
       32         2.108ms     0.645ms     3.27x
      128         1.881ms     0.989ms     1.90x
```

A matrix–matrix multiply keeps the hardware busy in a way a matrix–vector multiply cannot. At one query there is nothing to batch and it buys nothing.

My first version of this table showed batching *losing* (0.7×) — because the single-query path had been warmed up and the batch path had not, so the batch paid for the first BLAS call. Warming both is the fix.

## Where it stops

```
   vectors   ms/query      memory
    10,000       0.34        15 MB
    25,000       1.09        38 MB
    50,000       2.00        77 MB
   100,000       3.73       154 MB
   200,000       8.04       307 MB
   400,000      17.05       614 MB
log-log slope 1.03  (1.0 = linear in N)
```

Linear, as the multiply is — measured over a range wide enough to say so. A narrower sweep starting at a thousand vectors gave **0.83**, because at that size the BLAS call overhead has not amortized and the curve is still flattening.

So the extrapolation is honest:

```
       vectors   ms/query      memory
     1,000,000         43      1.5 GB
    10,000,000        426     15.4 GB
   100,000,000       4264    153.6 GB
```

And there is the real answer to *why exact search stops scaling*. At ten million vectors the latency is bad and the memory is impossible: 15 GB of float32 that must be resident, contiguous, and touched **in full on every single query**.

**Memory is the wall, and it arrives before latency does.** A test asserts both halves — that ten million vectors exceed a typical machine, and that one million still fits comfortably.

## float64 buys nothing

```
     dtype  bytes/number   1M vectors   ms/query
   float32             4        1.5 GB       0.93
   float64             8        3.1 GB       2.01
```

Double the memory, double the time. The embeddings come out of the model as float32, and cosine similarity between unit vectors has nowhere near that much signal in it.

## Exact means recall 1.0, by definition

```
recall@1   1.00
recall@3   1.00
recall@5   1.00
recall@10  1.00
```

Not a triumph — the definition. It is what makes exact search the reference every approximate index is measured against, and `recall_at_k` is written here so Day 8 can be scored with it.

## What this day settles

Exact search is not the naive option. It is one BLAS call, correct by construction, and up to roughly a million vectors it is fast enough that reaching for an approximate index first is premature.

Past that the binding constraint is RAM rather than time, and no cleverness in the scoring loop touches it. **Day 8's** IVF and HNSW attack exactly that — and what they spend to do it is the `recall@k` that was 1.0 above.

## Run it

```bash
# demo (the multiply, batching, the scaling wall, the handbook)
python vector_store.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_retrieval/day07_vector_store

# tests — from inside this folder
python -m unittest
```

Only the last demo section needs `sentence-transformers`; everything about scaling needs numpy alone, because the story is about vectors rather than what produced them.

> ⚠ On Windows, NumPy and PyTorch may both link OpenMP and abort with `OMP: Error #15`. `KMP_DUPLICATE_LIB_OK=TRUE` works around it.
