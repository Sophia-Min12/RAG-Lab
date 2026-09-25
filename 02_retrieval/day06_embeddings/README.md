# Day 6 · Real Sentence Embeddings — and Whether They Are Worth It

> The first dependency in this repo. Days 1–5 need nothing but the standard library. The question is not "do embeddings work" — it is whether they beat Day 5's **94%** by enough to pay for themselves.

## On the handbook's own questions: barely

```
              original questions         paraphrased
size            BM25       dense      BM25     dense
40               94%        100%       35%      100%
80               94%        100%       35%      100%
120              94%        100%       35%      100%
160              88%        100%       41%      100%
240              94%         88%       24%       88%
320              94%         94%       29%       94%
480              88%         94%       47%       88%
```

Read the left half alone: six points at the smaller sizes, and at 240 the model **loses**. For a dependency, a download, and the cost measured below, that is a thin case — on that evidence the honest recommendation is to keep BM25.

## On the same questions in different words: decisively

The paraphrased set asks the **same seventeen questions** with the **same seventeen answers**, worded the way somebody who had not read the handbook would ask:

```
original:    'what are the core hours'
paraphrased: 'when am I expected to be online'
both answer: '10:00 to 16:00'
```

Nothing about them is harder. The only thing that changed is vocabulary:

```
original      45% of query words appear in the answer's own sentence
paraphrased    9%
```

BM25 falls to a third. The model does not move.

**That gap is the entire product.** Embeddings buy vocabulary mismatch and nothing else here, so "are embeddings worth it" has no general answer — it depends on whether the people asking use the words the documents use, which is a property of your users rather than of your retriever.

## What it costs

```
17 queries over 26 chunks:
  BM25        0.7 ms
  dense     189.0 ms   (244x)
  plus 4.8s to load the model, once

the index itself:
  BM25   191 posting lists, built from counts
  dense  26 x 384 floats = 78 KB, all of which must be
         recomputed if the model changes
```

## The failure with no symptom

A chunk longer than the model's token limit is truncated — with no error at all.

```
   0 padding words ->    33 tokens, similarity to the question 0.490
 100 padding words ->   233 tokens, similarity to the question 0.314
 300 padding words ->   633 tokens, similarity to the question 0.184
 600 padding words ->  1233 tokens, similarity to the question 0.184
```

**The last two rows are identical.** Everything past 256 tokens was dropped, so two chunks differing by three hundred words receive byte-identical embeddings — a test asserts that equality directly. The phrase at the end of the chunk (`The emergency contact code is ZEBRA`) is not in the index, and no exception, warning or empty result ever said so.

The handbook index has zero truncated chunks, because Day 3 settled on chunks far smaller than 256 tokens. But **the token limit is a hard ceiling on chunk size that Days 3 and 4 never mentioned**, because until today nothing had one. `truncated_chunks()` exists to be called on any real index.

## Where this leaves the baseline

```
          original   paraphrased   17 queries
BM25           94%           35%        0.7 ms
dense         100%          100%      189.0 ms
```

Neither dominates on cost-adjusted terms, and they fail differently — one on vocabulary, the other on cost and on anything past its token limit. **Day 9** stops choosing and combines them.

## Run it

```bash
pip install sentence-transformers       # the first dependency in this repo

# demo (the comparison, the cost, the silent truncation)
python embeddings.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_retrieval/day06_embeddings

# tests — from inside this folder
python -m unittest
```

Without `sentence-transformers` the model tests **skip** (21 of 28) and the demo says so in words. Tests about the *corpus* — including BM25's collapse on the paraphrased set — run either way, because showing the problem needs no model; only solving it does.

CI installs the library so these actually run. If the model cannot be fetched, the tests skip with the reason printed rather than turning the badge red.

> ⚠ On Windows, NumPy and PyTorch may both link OpenMP and abort with `OMP: Error #15`. `KMP_DUPLICATE_LIB_OK=TRUE` works around it.

## Where this leads

**Day 7** builds a vector store from scratch — exact search, and the point at which it stops scaling.
