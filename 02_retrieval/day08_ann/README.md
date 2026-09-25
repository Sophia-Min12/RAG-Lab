# Day 8 · Approximate Nearest Neighbours: IVF and HNSW, and What They Give Up

> Day 7 ended at a wall made of memory. These are the two standard ways around it — and the currency they pay in is the `recall@k` that was 1.0 by construction.

## The dial: recall against work

```
IVF over 50,000 vectors, 128 lists (341-440 each)
  n_probe  recall@10   scanned  of total
        1       0.06       395        1%
        2       0.11       787        2%
        4       0.20      1572        3%
        8       0.35      3144        6%
       16       0.52      6280       13%
       32       0.72     12534       25%
       64       0.93     25019       50%
      128       1.00     50000      100%
```

Cluster once with k-means; at query time score the query against the few hundred centroids, take the `n_probe` nearest, scan only those lists.

At `n_probe = n_lists` it scans everything and recall reaches 1.00 — exact search with extra steps, which is the right sanity check: the approximation has a setting where it is not approximate. A test asserts it.

Everywhere else the missing recall is **silent**. The true neighbour was in a cluster nobody probed, so it is simply not in the results. No error, no warning, a slightly worse answer.

## The graph, and what dropping the hierarchy costs

```
NSW over 8,000 vectors, m=8, built in 2.4s
       ef  recall@10   scanned       ms
        8       0.10       161     0.35
       16       0.17       302     0.72
       32       0.35       528     1.52
       64       0.40       808     2.95
```

This is a navigable small-world graph — **HNSW without the layer hierarchy**, and the class name says so. Real HNSW stacks several such graphs at decreasing density and enters at the sparsest, replacing most of the walk with a few long hops.

The idea is here; the constant is not. Quoting published HNSW numbers next to this table would be dishonest.

## The measurement that nearly buried the day

```
        N    exact   IVF gathered   IVF blocked   recall
   20,000    0.33m         0.38m         0.24m     0.55
   50,000    0.63m         1.18m         0.44m     0.56
  100,000    1.69m         2.21m         0.61m     0.64
  200,000    2.90m         4.46m         0.97m     0.61
```

The middle column is where this day spent most of its time. That IVF scans an eighth of the collection and is **slower than scanning all of it**, at every size, and the gap does not close as the collection grows.

The algorithm is not wrong. It really does do an eighth of the arithmetic:

```
full multiply, all 200,000 rows         0.95 ms
multiply over 1/8, already contiguous   0.17 ms  (5.7x cheaper, as promised)
gathering that 1/8 out of the matrix    2.00 ms  (209% of the full multiply)
```

**The copy costs more than the multiply it avoids.** `matrix[candidates]` materializes a new array of the candidate rows, and on contiguous float32 that gather is more expensive than simply multiplying everything.

Storing each list's vectors contiguously at build time removes the copy entirely, and that single change is the difference between the middle column and the right one — from 0.7× to 3.4× against exact search, improving with N.

It is also why every serious ANN library is written in C++ with its own memory layout. The asymptotic win is real and the constant factor eats it unless the data sits where the arithmetic wants it.

Both paths are kept, and a test asserts they return **identical answers** — the day turns on them differing in cost and not in result.

## What an approximate index actually costs

- **Recall below 1.0, silently.** Day 7's exact search could not return a wrong answer. This can, and only a comparison against exact search reveals it.
- **A build step.** k-means over the collection, redone whenever the collection changes enough.
- **A parameter nobody can set for you.** `n_probe` and `ef` are recall-versus-latency dials whose right value depends on the data, the query distribution, and what you can tolerate.

And what it buys — a constant-factor speedup that only appears once the memory layout cooperates — does not arrive at all until the collection is large enough that Day 7's single BLAS call has stopped being fast enough.

## Run it

```bash
# demo (the recall dials, the graph, the constant factor)
python ann.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_retrieval/day08_ann

# tests — from inside this folder
python -m unittest
```

Needs only numpy. The tests assert on **work** — vectors scanned, arrays copied — rather than on wall-clock time, for the reason Day 7's slope test records.

## Where this leads

Level 2 now has two retrievers that fail differently: BM25 misses on vocabulary, dense costs and truncates. **Day 9** stops choosing between them.
