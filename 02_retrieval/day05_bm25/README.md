# Day 5 · BM25, the Baseline That Must Be Beaten

> Thirty years old, no training, no GPU, no dependency beyond the standard library — and still the number a retrieval system has to beat before anything more expensive is worth its cost.

Its three ideas each repair a specific failure of the scorers Day 3 used.

## 1. IDF — a term in every chunk says nothing

```
       'the'  df 4/4  idf 0.105
    'policy'  df 3/4  idf 0.357
 'forfeited'  df 1/4  idf 1.204
```

Elevenfold between the commonest and the rarest. Day 3's scorers weighted all three identically.

The form used here is `log(1 + (N - df + 0.5) / (df + 0.5))`. The `+1` inside the log matters: the textbook Robertson–Sparck Jones version goes **negative** for a term appearing in more than half the collection, so a chunk could be penalised for containing a query term. A test checks the whole vocabulary stays non-negative.

## 2. Saturation — the ninth mention is not worth the first

```
   tf  contribution  gain over tf-1
    1         1.000           1.000
    2         1.429           0.429
    3         1.667           0.238
    4         1.818           0.152
    5         1.923           0.105
  ...
   10         2.174
  100         2.463
10000         2.500
```

The ceiling is `k1 + 1 = 2.5`, approached and never passed. `score_occurrences` believed the hundredth mention was worth a hundred firsts; here it is worth less than the second.

## 3. Length normalization — the bias Day 3 measured

Two chunks. One short and correct, one long and padded with the query terms:

```
short  ( 30 chars): 'Core hours are 10:00 to 16:00.'
padded (180 chars): 'Core hours matter. Employees ask about core hours often...'
query: 'what are the core hours'   (the answer is only in the short one)

     b     short    padded   ranked first
  0.00     1.058     1.885   padded
  0.25     1.177     1.799   padded
  0.50     1.327     1.721   padded
  0.75     1.520     1.650   padded
  1.00     1.779     1.585   short
```

`b=0` is Day 3's `score_occurrences` exactly: no correction, padding wins.

**And the part worth saying out loud: the universal default is `b=0.75`, and at `b=0.75` the padded chunk still ranks first.** The default is a compromise tuned for ordinary collections, not a cure for a chunk that repeats the query five times. "BM25 handles length" is true on average and false here. A test pins it.

## Does `b` matter on the actual handbook? No — and that is a result

```
hit rate at a 600-character budget, by b:
 size    b=0.0    b=0.5    b=1.0
   40      94%      94%      94%
   80      94%      94%      94%
  120      94%      94%      94%
  160      88%      88%      88%
  240      94%      94%      94%
  320      94%      94%      94%
  480      88%      88%      88%
```

Nothing moves, and the reason is measurable. `b` acts only through `|d| / avgdl`, and sentence chunks of one handbook have a coefficient of variation of **0.16** — every chunk is about average length, so the correction has nothing to correct.

Mix in Day 4's tables and code, and the spread rises to **0.80**:

```
33 chunks, 22 to 389 characters, cv 0.80
     b   hit@3
  0.00     88%
  0.25     94%
  0.50     94%
  1.00     94%
```

One question's worth — the honest size of the effect on a corpus this small, appearing exactly where the theory says it should. A knob everyone tunes, doing nothing until the thing it corrects for is actually present.

## The baseline

```
 size   occurrences   coverage    BM25
   40           94%        94%     94%
   80           94%        94%     94%
  120           94%        94%     94%
  160           88%        88%     88%
  240           94%        94%     94%
  320           82%        88%     94%
  480           71%        88%     88%
```

BM25 is never worse, and is better exactly where Day 3's two scorers contradicted each other — 94% against 82% and 88% at size 320. It also removes the disagreement, which was the real problem: one principled scorer instead of two defensible ones that disagree.

**Best BM25 hit rate at a 600-character budget: 94%.**

That is the bar. Day 6 loads a neural embedding model, costing a dependency, a download, and orders of magnitude more compute per query — and it has to beat 94% to justify any of it. A great many RAG systems never run this comparison.

## The inverted index

```
26 chunks, 191 distinct terms
   6/26 chunks scored for 'how many vacation days accrue each month'
  14/26 chunks scored for 'what happens to unused leave above ten days'
   9/26 chunks scored for 'who approves more than five consecutive days'
```

A query only touches chunks sharing a term with it. At this size that saves nothing — it is the reason BM25 still runs on collections where embedding every document per query would be impossible.

## Run it

```bash
# demo (the three ideas, the default that does not fix padding, the baseline)
python bm25.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_retrieval/day05_bm25

# tests — from inside this folder
python -m unittest
python test_bm25.py
```

## Where this leads

**Day 6** brings `sentence-transformers` and the first real dependency in this repo. The question it has to answer is not "do embeddings work" but "do they beat 94% by enough to pay for themselves".
