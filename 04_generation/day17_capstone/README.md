# Day 17 · Capstone: The Full Loop, a CLI, and the Honest Writeup

> Seventeen days wired together, with defaults the measurements actually support rather than the ones with the most parts.

```bash
python capstone.py ask "what are the core hours"
python capstone.py evaluate
python capstone.py probe
python capstone.py index
python capstone.py writeup
```

```
Q  what are the core hours
A  10:00 to 16:00
   Remote (handbook/remote.md) line 2, chars 56..145
   'Employees working remotely must be reachable during core hou...'
   citation verifies: True
```

## What it scores

```
question set      ret hit  ret MRR  correct  grounded  abstained
original             100%    0.971      82%       94%         6%
paraphrased          100%    0.853      47%       71%        24%
```

**Retrieval is perfect on both sets.** Seventeen days of it, and the answer is in the context every single time. Generation then gets 82% and 47%.

The bottleneck moved, and it moved early — Day 10 already had retrieval at the ceiling. Everything after that was measuring a generator.

(The paraphrased figure is pessimistic by about 9 points: three of those questions are the yes/no evaluation-set bug from [Day 15](../day15_grounding/), left in place.)

## The configuration, and why each part is there

| Component | Setting | Why |
|-----------|---------|-----|
| chunking | sentence, 80 chars | Day 3: size makes no measurable difference; Day 4: fixed chunks cost a great deal |
| retrieval | dense | Day 6: worth 6 points on document-phrased questions, 60 on others |
| reranking | cross-encoder over 20 | Day 10: reaches the first-stage ceiling in every case measured |
| context | 3 chunks | Day 14: accuracy peaks there, and more measured *worse* |
| fusion | none | Day 9: beats dense only where the retrievers are complementary, and this corpus is not |

## What the measurements support

```
chunking strategy   worth 0.197 MRR
embeddings          worth 0.265 MRR overall, 0.471 on rephrased questions
reranking           worth 0.064 to 0.279, depending entirely on what precedes it
chunk size          no measurable effect
hybrid fusion       no measurable effect; RRF at its default made things worse
```

BM25 answers a query in 0.11 ms. The full pipeline takes about **1400×** that, and on questions phrased the way the documents are it buys six points.

## The uncomfortable results

**The whole corpus fits in the context window.** 437 tokens against a 512-token limit. Passing all of it with no retrieval at all scores 56%, against the pipeline's 65% — so retrieval here is a *distractor filter*, not a finder, and nine points is what filtering is worth.

**The model follows a poisoned context 100% of the time**, and hands each fabrication over with an accurate citation. It also recites from its own weights 0% of the time. Those are the same property, and there is no setting of it that is safe.

**Thirty-four questions cannot separate most of these systems.** The 95% interval on 16/17 runs from 83% to 105%. Day 11 said so, and every number above should be read with it.

## What was wrong before it was measured

`python capstone.py writeup` lists all fourteen, each documented where it happened rather than quietly fixed. A sample:

- **Day 2** — Day 1's offsets pointed into a normalized copy, so citations on decomposed Korean quoted the wrong sentence, and the failure was byte-identical to correct behaviour on ASCII
- **Day 3** — my own budget packer skipped over-budget chunks and kept going; I was about to explain the resulting 18% as a quantization effect
- **Day 8** — my IVF was slower than brute force at every size, because gathering the candidate rows cost 209% of the multiply it was meant to avoid
- **Day 12** — my agreement metric counted a tie as agreement, and reported that an evaluation set with no opinion agreed 83% of the time
- **Day 15** — three questions in my own evaluation set are phrased as yes/no and graded against an extractive span

Tests assert the writeup names them, so it cannot quietly shrink.

## What remains unmeasured

Whether an answer follows from its context without distorting it. Day 15's containment check catches an answer that is *absent* from the context and cannot catch one that is present and misused — which is precisely the 18% that are grounded and wrong.

A judge model would produce a number for it. It would be an unverified number from an unverified judge, and seventeen days were spent not doing that.

## Run it

```bash
pip install sentence-transformers transformers

python capstone.py writeup    # needs no models at all
python capstone.py ask "how much is the meal allowance"

pytest                        # from the repo root
python -m unittest            # from inside this folder
```

`writeup` is the one command that runs on a machine with nothing installed — a test asserts that.
