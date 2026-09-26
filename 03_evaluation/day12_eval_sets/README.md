# Day 12 · Building an Evaluation Set Without Hand-Labelling Everything

> Day 11 ended with seventeen questions being too few to separate the systems. The obvious fix is to generate more.

## Generating questions is easy

```
Q 'Annual leave accrues at ____ per month of continuous service'
A '1.75 days'
Q 'Any balance above ____ is forfeited on 31 December and is not paid out'
A 'ten days'
```

26 known-item and 17 cloze questions, from the same documents, with no labelling and no model.

## And the questions are useless

```
MRR by evaluation set:
set                   BM25     dense   score fusion      RRF
hand-written         0.814     0.882          0.912    0.912
paraphrased          0.118     0.814          0.529    0.353
known-item           1.000     1.000          1.000    1.000
cloze                1.000     1.000          1.000    1.000
```

```
set               n  vocab overlap   resolving power
hand-written     17            45%               83%
paraphrased      17             9%              100%
known-item       26           100%                0%
cloze            17            83%                0%
```

Both generated sets score **every system at 1.000**. Their resolving power is zero: they cannot tell any two retrievers apart, so they cannot be used to choose one.

The vocabulary-overlap column predicts it. A cloze query is the document's own sentence with one span removed — 83% of its words are in the target — so finding the target is not a retrieval problem. Generating questions is easy; generating an evaluation set with an **opinion** is the hard part.

### A measurement error of mine

The first version of `agreement()` scored a tie as agreement, and reported that these degenerate sets agreed with the hand-written one **83% of the time**. Corrected, it separates the three cases:

```
vs paraphrased    3/5 of resolved pairs
vs known-item     no pair resolved by both
vs cloze          no pair resolved by both
```

A set with no opinion cannot agree with anything. `resolving_power()` now exists to be checked *first*.

## Pooling: judge a few chunks instead of all of them

```
34 real queries, top-n from each of three systems:
  depth   chunks judged   answer in pool
      1       1.7 of 26              85%
      3       5.5 of 26              94%
      5       8.4 of 26              97%
     10      14.3 of 26             100%
```

At depth 10 the pool holds the answer to every query while judging 14 chunks of 26. On a real collection that ratio is the whole point: judge hundreds instead of millions.

## What pooling costs, and who pays

The assumption is that anything outside the pool is irrelevant. So hold one system out of the pool and score it anyway:

```
system    scored by the pool   scored by truth   penalty
BM25                     59%               59%        0%
dense                    62%               91%       29%
```

**BM25 pays nothing. Dense pays 29 points.**

The asymmetry is the finding, not the numbers. BM25 finds nothing the other systems miss, so a pool built without it already contains everything it would have contributed. Dense finds chunks no lexical system surfaces — the whole reason it exists — and those are exactly the chunks the pool never judged.

**Pooling penalises a system in proportion to how different it is from the systems that built the pool.** A genuinely novel retriever, evaluated against someone else's pool, looks worse the more novel it is. Both halves are pinned by tests.

## What this day settles

Neither answer is free. The cheap *set* had no opinion; the cheap *judgements* have a preference.

## Run it

```bash
python eval_sets.py       # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

Generation, `resolving_power` and `agreement` are tested without a model; only the end-to-end findings need embeddings.

## Where this leads

**Day 13** turns each component off in turn and reads the damage, using the sets that survived today.
