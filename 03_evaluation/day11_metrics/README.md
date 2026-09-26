# Day 11 · `recall@k`, `MRR`, `nDCG`, and Which One Answers Your Question

> Every number in Level 2 was `hit@3` on seventeen questions. This day asks what that measured.

## Level 2, rescored

```
original questions, k=3:
  method             hit@3     P@3     MRR     MAP   nDCG@3
  BM25               0.941   0.314   0.814   0.814    0.847
  dense              0.941   0.314   0.882   0.882    0.898
  score fusion       1.000   0.333   0.912   0.912    0.935
  RRF c=60           0.941   0.314   0.912   0.912    0.919
  dense + rerank     1.000   0.333   0.971   0.971    0.978

paraphrased questions, k=3:
  BM25               0.235   0.078   0.118   0.118    0.147
  dense              0.882   0.294   0.814   0.814    0.831
  score fusion       0.824   0.275   0.529   0.529    0.605
  RRF c=60           0.353   0.118   0.353   0.353    0.353
  dense + rerank     1.000   0.333   0.853   0.853    0.891
```

**MAP and MRR are identical in every row.** This corpus has exactly one relevant chunk per question, and average precision over a single relevant item *is* the reciprocal rank. Printing both looks like two pieces of evidence and is one. A test proves the identity directly.

**precision@3 tops out at 0.333** for the same reason — one relevant chunk in three retrieved. It is measuring the value of `k`, not the retriever.

## Where `hit@3` was hiding something

```
paraphrased questions:
  dense          hit@3 0.882   MRR 0.814
  score fusion   hit@3 0.824   MRR 0.529
```

Day 9 read those hit rates as 88% against 82% and called it close. By MRR it is **0.814 against 0.529**, which is not close at all: score fusion finds the answer and ranks it badly. A binary hit cannot tell first place from third.

## Two metrics that agree on every query and disagree about the system

With one relevant chunk, MRR and nDCG are both functions of its rank alone:

```
 rank    1/rank   1/log2(rank+1)
    1    1.0000           1.0000
    2    0.5000           0.6309
    3    0.3333           0.5000
    5    0.2000           0.3869
   15    0.0667           0.2500
```

So they cannot disagree about a query. They can still disagree about an average:

```
score fusion   answer ranks {1: 14, 2: 3}
               MRR 0.9118    nDCG 0.9349
RRF c=60       answer ranks {1: 15, 2: 1, 15: 1}
               MRR 0.9157    nDCG 0.9342
```

**MRR prefers RRF. nDCG prefers score fusion.** Same result lists, same judgements, opposite verdicts — because `1/rank` punishes the single rank-15 result down to 0.067 while `1/log2(rank+1)` leaves it at 0.25, and nDCG pays more for a rank 2 than MRR does. Averaging a different curve over the same ranks is enough to swap the winner.

A test reproduces this without a retriever at all, from two hand-built rank distributions.

## And none of it survives the sample size

```
  hits     rate          95% interval
 10/17      59%            35% to 82%
 14/17      82%           64% to 100%
 16/17      94%           83% to 105%
 17/17     100%          100% to 100%
```

94% and 100% — the difference Level 2 kept reporting — are not distinguishable: the interval for 16/17 contains 100% comfortably.

Both boundary rows are nonsense in their own way. 105% is not a rate, and 17/17 reports zero width because the normal approximation computes a variance of `p(1-p) = 0`. Neither is a bug; both are the approximation failing exactly where Level 2 spent its time.

The MRR gap in the section above is **0.004** — on seventeen questions, one answer moving from rank 2 to rank 1. Reporting it to four decimals implies a precision the evaluation set cannot support.

## The honest summary of Level 2

Dense + reranking is clearly better than BM25 alone. BM25 is clearly bad on paraphrases. **Every finer distinction than that is inside the noise.**

## Run it

```bash
python metrics.py         # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

The metrics are arithmetic over lists of 0s and 1s, so their tests need no model and no corpus.

## Where this leads

**Day 12** builds an evaluation set large enough to say more — without hand-labelling a thousand queries.
