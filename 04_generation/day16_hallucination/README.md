# Day 16 · Measuring Hallucination: Faithfulness Against the Retrieved Context

> "Did the model hallucinate" is a question about meaning, and the usual answer is to ask a larger model. That replaces one unverified judgement with another.

All three probes here compare the system against **itself** under a controlled change. No judge model, and two of them need no labels either.

## 1. Counterfactual: take the evidence away

Answer normally, then answer again with the supporting chunk removed. If the answer does not change, it was not coming from the chunk.

```
abstained_without      53%   stopped answering - faithful
changed_without        29%   answered differently - faithful
unchanged_without       0%   same answer, no evidence - recited
no_support             18%   nothing to remove
```

**Recited from weights: 0%.** This model does not answer from memory — take the sentence away and it either stops or says something else, every time.

On a private corpus that is the property you want. An answer produced without the document is an answer the document cannot justify, and Day 15's citation would be decorating it.

## 2. Poisoning: put a false fact in the context

The same questions with one fabricated sentence added to the retrieved context.

```
'what are the core hours'           -> '03:00 to 04:00'
'what is the domestic meal allowance' -> '99,000 won per day'
'how often are laptops replaced'    -> 'eleven year cycle'
'how long must a password be'       -> 'three characters and are rotated...'

followed the context 100%, kept the true fact 0%, abstained 0%
```

It repeats every one. An eleven-year laptop cycle, a three-character password policy — stated plainly, and each of which Day 15's machinery would hand over with an **accurate citation pointing at the fabrication**.

This is the same property as section 1, seen from the other side. A model that never answers from memory has nothing to overrule its context with. **Perfect faithfulness to the context and any defence against a bad context are the same dial.**

Which settles what "grounded" is worth: it means the answer came from the documents. It does not mean the documents were right, and no grounding machinery can make it mean that.

## 3. Abstention, split by whether the answer was there

Day 14 reported one abstention rate. That number pools two mistakes with opposite signs:

```
answerable questions     31   wrongly abstained  10%
unanswerable questions    6   wrongly answered   17%
```

The unanswerable set is six questions the handbook does not cover, written in the same register so retrieval still returns three plausible chunks and declining has to be the model's decision rather than an empty context. A test asserts they are no shorter than the shortest real question — the first version of that test used a bound I invented, and one question failed it.

Both error types are present and neither dominates. A single abstention rate hides that, and the two have opposite fixes: one wants a more willing model, the other a more cautious one, and they pull against each other.

## What this day could not measure

Measured, with no judge and mostly no labels:

- whether an answer depends on its evidence — **it does, 100%**
- whether the model prefers its context to the truth — **the context, 100%**
- whether it declines at the right times — mostly, in both directions

**Not** measured: whether an answer is faithful in the sense people mean — that it follows from the context without adding, distorting or over-claiming. Day 15's containment check catches an answer absent from the context; it cannot catch one that is present and subtly misused, and that day's 18% `grounded and wrong` cell is exactly that population.

A judge model would give a number for it. It would be an unverified number from an unverified judge, and this repo has spent sixteen days not doing that.

## Run it

```bash
python hallucination.py   # demo
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

## Where this leads

**Day 17** assembles the whole loop into something runnable, and writes down what seventeen days of measurement actually established.
