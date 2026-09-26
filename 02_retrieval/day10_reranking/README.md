# Day 10 · Cross-Encoder Reranking: Slow, Accurate, and Bounded

> A bi-encoder embeds the query and the chunk separately — which is exactly why the chunk can be precomputed and stored. A cross-encoder gives that up and reads both together.

```
'what are the core hours'
      5.85  the core-hours sentence
    -11.40  the laptop sentence

'when am I expected to be online'
    -10.76  the core-hours sentence
    -11.23  the laptop sentence
```

The scores are logits: comparable within one query, meaningless across queries. Note the paraphrase scores the right sentence **negatively in absolute terms** while still ranking it far above the wrong one. Only the order is usable, which is why this reranks and does not threshold.

## The ceiling, before any gain

A reranker reorders a shortlist. It cannot add to it.

```
 depth   BM25 orig   BM25 para   dense orig   dense para
     3         94%         24%          94%          88%
     5         94%         29%         100%          94%
    10         94%         35%         100%         100%
    20         94%         41%         100%         100%
    26         94%         41%         100%         100%
```

BM25 on the paraphrased set tops out at **41% at depth 26 — the entire index**. That is not a ranking problem a better reranker could fix:

```
BM25 returns 8.9 of 26 chunks on average for these queries,
and 1 of them returns nothing at all.
```

A chunk sharing no term with the query is not a candidate at *any* depth. Three tests pin this without needing a model, because it is a fact about BM25's candidate set rather than about reranking.

## What reranking buys

```
first stage   set              plain   reranked   ceiling
sparse        original           94%        94%       94%
sparse        paraphrased        24%        41%       41%
dense         original           94%       100%      100%
dense         paraphrased        88%       100%      100%
```

**Every reranked number equals its ceiling exactly.** The cross-encoder extracts everything the first stage made available, in all four cases — so the only thing limiting it is first-stage recall. A test asserts the equality, not merely the bound.

Dense + reranking reaches **100% on both question sets** — the first configuration in this repo that does.

The same holds against depth:

```
 depth   ceiling   reranked
     3       88%        88%
     5       94%        94%
    10      100%       100%
    20      100%       100%
    26      100%       100%
```

Identical at every depth. Deepening the shortlist helps only while it raises the ceiling.

## What it costs

```
  pairs        ms   ms/pair
      1     132.1    132.06
     10     130.0     13.00
     20     151.1      7.56
     50     258.8      5.18
    100     513.5      5.14

BM25, top 20                    0.11 ms per query
reranking 20 candidates        151.1 ms per query
ratio                           1404x
```

One forward pass per `(query, chunk)` pair, and nothing can be precomputed because the pair does not exist until the query does. Linear in the shortlist, so depth is a latency budget.

Reranking the whole index is affordable here only because the index is 26 chunks. At a million it would be impossible — which is the entire reason this is a second stage and not a first one.

## What Level 2 established

| Day | The finding |
|-----|-------------|
| 5 | BM25 reaches 94%, and its famous `b` default does not fix a padded chunk |
| 6 | Embeddings buy vocabulary mismatch and nothing else — worth 6 points, or 60, depending on your users |
| 7 | Exact search is one BLAS call; the wall is memory, not latency |
| 8 | Approximate search trades recall for a constant factor that a bad memory layout eats entirely |
| 9 | Fuse only where the retrievers are complementary; RRF's standard constant made things worse here |
| 10 | Reranking fixes ordering and never recall |

The configuration that answers every question in this corpus is **dense retrieval over a shortlist, reranked** — at roughly 1400× BM25's cost, for six points on questions phrased the way the documents are.

## Run it

```bash
python reranking.py       # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

Tests needing the cross-encoder skip cleanly when it is unavailable; the ones about BM25's structural ceiling run regardless.

## Where this leads

Every number in Level 2 is `hit@3` on seventeen questions. **Level 3** stops trusting that and asks what these metrics actually measure.
