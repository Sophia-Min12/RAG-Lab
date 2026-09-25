# Day 3 · Chunk Size Versus Answer Quality, Measured

> The question was "what chunk size gives the best answers". The measurement says the question is under-specified — and that is the result.

A five-document policy handbook, 1,939 characters, and 17 questions whose answers are literal spans of the text. A chunk either contains the answer or it does not, so the hit rate needs no human judgement.

## The measurement everyone makes first

```
sentence chunking                  fixed chunking
 size  chunks  hit@3   context      size  chunks  hit@3   context
   40      26    88%       218        40      51     6%       120
   80      26    88%       218        80      26    65%       233
  120      25    88%       225       120      19    71%       356
  160      15    88%       435       160      15    82%       448
  240      10    94%       627       240      10    88%       666
  320      10   100%       826       320      10   100%       897
  480       5    94%      1182       480       5    94%      1182
```

Read the hit column alone and the answer is obvious: bigger is better, use 320. Now read the context column. Top-3 of a corpus cut at 40 characters is 120 characters of text; top-3 of the same corpus cut at 480 is nearly 1,200.

**The comparison was never between two chunk sizes.** It was between ten times as much context and one.

## Holding the budget fixed

Retrieve as many chunks as fit in a fixed character budget instead of a fixed `k`:

```
sentence chunking — hit rate, and how many chunks fit
 size    300 chars     600 chars    1200 chars
   40   88% (3.6)     94% (7.6)    94% (15.6)
   80   88% (3.6)     94% (7.6)    94% (15.6)
  120   88% (3.5)     94% (7.4)    94% (15.0)
  160   82% (1.6)!    88% (3.8)    100% (7.8)
  240   82% (1.0)!    94% (2.2)     94% (5.2)
  320   88% (1.0)!    88% (1.3)!   100% (3.7)
  480   88% (1.0)!    88% (1.0)!    94% (2.8)

fixed chunking
   40   12% (7.0)    24% (15.0)    24% (30.6)
   80   65% (3.2)     88% (7.0)    94% (15.0)
  120   71% (2.0)     82% (5.0)    82% (10.4)
  160   59% (1.2)!    82% (3.4)     94% (7.8)
  240   76% (1.1)!    88% (2.1)     88% (5.2)
  320   76% (1.0)!    88% (1.2)!   100% (3.5)
  480   88% (1.0)!    88% (1.0)!    94% (2.8)
```

`!` marks a cell where fewer than two chunks fit. There the budget has collapsed to top-1, and the number says more about one chunk boundary than about retrieval. A budget comparison only means something when the budget is a few times the chunk size.

Reading the unflagged cells: **sentence chunking is flat** — every size answers about as well as every other. **Fixed chunking is not** — at 40 characters it answers a quarter of the questions and no budget rescues it.

### A bug I nearly explained away

An earlier version of that table had `fixed(320)` at **18%** and `sentence(320)` at **47%**, and I was about to write them up as exactly the quantization effect described above. They were not.

`retrieve_within_budget` was skipping any chunk too large to fit and *continuing* down the ranking, so it packed four short tail-end remainder chunks instead of the single chunk that held the answer. Taking chunks in rank order and stopping at the first overflow gives 76% and 88%.

The bug was mine, the explanation was plausible, and the two had nothing to do with each other. A test now asserts the packed chunks are a prefix of the ranking.

## Why 40-character fixed chunks cannot be rescued

```
 size   sentence intact   fixed intact
   40             100%            41%
   80             100%           100%
  120             100%            88%
  160             100%           100%
```

`intact` asks whether **any** chunk holds the whole answer, before retrieval is involved at all. It is a ceiling: below it no retriever, reranker or model recovers the fact, because no single chunk contains it. A test asserts the hit rate never exceeds it, at any size or strategy.

The ceiling is also **not monotone** — `fixed(80)` keeps every answer intact and `fixed(120)` loses two. Where a boundary lands is luck, and a larger chunk is not reliably a safer one.

## And the part that undoes the rest of the day

Same corpus, same questions, same 600-character budget. The only difference is how a chunk is scored:

```
              sentence                    fixed
 size  occurrences  coverage  gap   occurrences  coverage  gap
   40          94%       94%   0%           24%       24%   0%
   80          94%       94%   0%           82%       88%   6%
  120          94%       94%   0%           76%       82%   6%
  240          94%       94%   0%           82%       88%   6%
  320          82%       88%   6%           65%       88%  24%
  480          71%       88%  18%           71%       88%  18%
```

`score_occurrences` counts every matching token, so a long chunk wins by being long. `score_coverage` counts distinct query terms and does not. Neither is wrong — they answer different questions. They agree perfectly on small chunks and disagree by **up to 24 points** on large ones, which is the difference between "large chunks are fine" and "large chunks are broken".

```
which size does each scorer call best?
  sentence  occurrences  ->   40 (94%)
  sentence  coverage     ->   40 (94%)
  fixed     occurrences  ->   80 (82%)
  fixed     coverage     ->   80 (88%)
```

## What this day established

Not a best chunk size. "What chunk size gives the best answers" has no answer until two other things are fixed:

1. **The context budget.** Comparing `hit@k` across chunk sizes compares different amounts of retrieved text, and the larger-is-better conclusion is mostly that.
2. **The retriever.** A length-biased scorer and a length-neutral one disagree by up to 24 points about the same configuration.

What survives both is the **intact ceiling**: chunking that splits answers caps everything downstream, it is measurable before any retriever exists, and it is the one number here that is a property of the chunker alone.

[Day 5](../../02_retrieval/) replaces the scorer with BM25, whose length normalization exists precisely because of the bias measured above.

## Run it

```bash
# demo (the rigged comparison, the budget, the ceiling, the scorer gap)
python chunk_size.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_chunking/day03_chunk_size

# tests — from inside this folder
python -m unittest
python test_chunk_size.py

# the docstring examples are runnable too
python -m doctest chunk_size.py
```

## Where this leads

The chunkers work and the metrics are honest about what they cannot tell you. **Day 4** hands them documents built to break them: tables, code, and Korean.
