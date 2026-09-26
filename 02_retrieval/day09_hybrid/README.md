# Day 9 · Hybrid Retrieval: Score Fusion Versus Rank Fusion

> Day 5 built a retriever that misses on vocabulary. Day 6 built one that costs 244× as much and truncates. The obvious move is to use both — and the numbers say "sometimes".

## First: is there anything to fuse?

```
question set     BM25  dense  union  BM25 only  dense only
original           16     16     17          1           1
paraphrased         4     15     15          0          11
```

On the **original** questions each retriever finds exactly one the other misses, so the union is 17/17 and fusion has somewhere to go.

On the **paraphrased** set BM25 finds *nothing* dense missed. The union equals dense alone, so no combination rule can beat dense there — and some will do worse.

That table costs two retrievals per question and settles whether the rest of the day is worth doing. It is the step most hybrid write-ups skip. A test asserts the union bounds every fusion method.

## Score fusion

```
  alpha   original   paraphrased    mean
   0.00        94%           29%     62%
   0.25       100%           53%     76%
   0.50       100%           82%     91%
   0.75       100%           82%     91%
   1.00        94%           88%     91%
```

At `alpha` 0.25–0.75 the original set reaches **100%**, which neither retriever manages alone. Fusion doing exactly what it promises, on the set where the complementarity table said it could.

The normalization underneath is shakier than it looks. Min–max uses the minimum and maximum *of the result list*, so the same chunk with the same BM25 score normalizes differently depending on what else came back:

```
raw BM25 scores, top 3       [4.543, 1.499, 1.499]
normalized within top 3      [1.0, 0.0, 0.0]
the same three within top 8  [1.0, 0.061, 0.061]
```

A score that changes when you ask for more results is not a property of the chunk.

## Rank fusion, and a default that is wrong here

RRF sums `1 / (constant + rank)`, so there is nothing to normalize.

```
 constant   original   paraphrased    mean
        0       100%           76%     88%
        1       100%           76%     88%
        5        94%           41%     68%
       20        94%           35%     65%
       60        94%           35%     65%   <- the standard default
      200        94%           35%     65%
```

**The published default of 60 is the worst setting on this data**, by 41 points on the paraphrased set.

The reason is what the constant does. At 60, ranks 1 through 20 differ in weight by under 25%, so RRF is close to counting "appeared in either list". That is the right behaviour when the inputs are several systems of comparable quality — which is what the original RRF paper was fusing — and the wrong behaviour when one list is nearly noise.

At constant 0, rank 1 is worth twice rank 2 and a confident top hit can outvote a bad list. Still not enough to beat dense alone, because **RRF has no way to know that one of its two inputs is worthless.**

## The comparison that matters

```
method                     original   paraphrased    mean
BM25 alone                      94%           24%     59%
dense alone                     94%           88%     91%
score fusion, alpha=0.5        100%           82%     91%
RRF, constant=60                94%           35%     65%
RRF, constant=0                100%           76%     88%
```

Tuned score fusion beats everything on the original set and matches dense alone on the mean. RRF at its standard setting is **worse than either retriever by itself**.

"Always use hybrid" is not what these numbers say. What they say is: measure the complementarity first. Where both retrievers contribute, fusion pays. Where one is carrying the other, fusion is a way to make it worse — and rank fusion, which cannot tell a good list from a bad one, is the most likely to.

## Run it

```bash
python hybrid.py          # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

The fusion rules are tested without any model — they combine lists of `(id, score)` pairs, so only the end-to-end comparison needs embeddings.

## Where this leads

Every method here reorders a candidate list using cheap signals. **Day 10** brings a cross-encoder, which reads the query and the chunk *together* — far slower, and the first thing in this repo that can tell a good list from a bad one.
