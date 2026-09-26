# Day 13 · Ablations: Turning Each Component Off and Reading the Damage

> The table is easy. Reading it correctly is not.

## The table everybody publishes

```
baseline: sentence(80) + dense + rerank, 34 questions

all 34 questions:
  ablation                       MRR    delta   hit@3
  full pipeline                0.912   +0.000   1.000
  no reranker                  0.848   -0.064   0.912
  no embeddings (BM25)         0.647   -0.265   0.676
  hybrid instead of dense      0.912   +0.000   1.000
  fixed chunks                 0.715   -0.197   0.794
  tiny chunks (40)             0.912   +0.000   1.000
  large chunks (320)           0.922   +0.010   1.000
  shallow shortlist (5)        0.912   +0.000   0.971
```

Read straight down: embeddings are worth the most, chunking strategy a lot, the reranker a little, chunk size nothing.

Two rows deserve suspicion before belief. **"Large chunks" comes out *above* the baseline** by 0.010, and two others come out exactly level. Day 11's interval on 34 questions is wider than any of those, so the honest reading of all three is *no measurable effect*.

## The table is not additive

If each delta were a property of its component, removing two would cost the sum. It does not:

```
pair removed                        both  additive  interaction
no reranker + no embeddings        0.466     0.583       -0.118
no reranker + fixed chunks         0.598     0.652       -0.053
no embeddings + fixed chunks       0.532     0.451       +0.081
```

The signs go both ways, and both mean something.

**Negative:** removing the reranker and the embeddings together costs 0.118 *more* than the sum. Each was covering for the other, so measuring either alone understates it.

**Positive:** removing the embeddings and using fixed chunks costs 0.081 *less* than the sum. Their damage overlaps — both lose the same questions, and a question can only be lost once.

## What the reranker is worth depends on what precedes it

```
first stage       no rerank   rerank     gain
sparse                0.466    0.647   +0.181
rrf                   0.632    0.912   +0.279
score                 0.721    0.912   +0.191
dense                 0.848    0.912   +0.064
```

**The reranker is worth +0.064 on one first stage and +0.279 on another** — a factor of 4.4 — and the ablation table above reported one of those as though it were the answer.

Look at the `rerank` column: three of the four first stages land on exactly **0.912**. The reranker does not improve their ranking, it *replaces* it, so whatever order they produced stops mattering and only their candidate sets do. The better the first stage, the less visible the reranker — which is precisely why the single-ablation table called it the cheapest component. It was measured on top of the best first stage available.

BM25 is the exception at 0.647, and [Day 10](../../02_retrieval/day10_reranking/) said why: its candidate set does not contain the answer for most paraphrased questions, so there is nothing to reorder. A reranker converges first stages that *retrieve* the same things, and BM25 does not.

## What an ablation actually measures

Not the value of a component — the damage from removing it **from this pipeline, on this corpus, with these questions**. All three qualifiers do real work here:

| Qualifier | Evidence |
|-----------|----------|
| this pipeline | the reranker is worth 0.064 or 0.279 depending only on what runs before it |
| this corpus | chunk size is worth nothing here; Day 3 showed it is worth a great deal on fixed chunks |
| these questions | embeddings cost 0.265 over all 34 questions and 0.471 over the paraphrased half |

A component that looks free may simply be one another component is covering for. The only way to see that is to remove both — and a single-ablation table never does.

## Run it

```bash
python ablations.py       # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

`Pipeline` is frozen and every entry in `ABLATIONS` changes exactly one field — a test asserts that, because a delta attributed to one component has to come from one component.

## Where this leads

Level 3 is done. The retrieval side is as measured as 34 questions permit, and Day 11 was clear about what that permits. **Level 4** starts generating answers, where the measurement problem gets considerably worse.
