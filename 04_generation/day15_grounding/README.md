# Day 15 · Grounding and Citation: Making the Answer Point at Its Evidence

> Day 2 built offsets that survive normalization. This is what they were for.

```
Q 'what are the core hours'
A '10:00 to 16:00'
  Remote (handbook/remote.md) line 2, chars 56..145
  'Employees working remotely must be reachable during core hou...'
```

Those offsets are Day 2's, and they still point at the **original** document rather than at a normalized copy of it. Every grounded answer here verifies with `verify_citation` against the un-normalized source.

## Correctness and groundedness are different questions

```
  k   grounded+right   grounded+wrong   ungrounded+right   ungrounded+wrong   abstained
  2              65%              18%                 0%                 6%         12%
  3              65%              18%                 0%                 6%         12%
  8              59%              29%                 0%                 6%          6%
```

**`ungrounded and correct` is 0%.** This model does not paraphrase — it lifts spans — so every right answer is traceable. A larger model would fill that cell, and every entry in it would be an answer a user cannot check.

**`grounded and wrong` is the cell worth staring at.** Those answers quote a real chunk, the quote is accurate, and the answer is still wrong — so the citation makes them *more* convincing rather than less.

Citation is a property of provenance, not of truth. A test asserts that cell is non-empty, because if it were ever 0 the claim would need revisiting.

## What the scorer is doing to those numbers

Of the answers scored wrong at k=3, **half are the right fact stated more briefly than the gold answer**:

```
TERSE  expected '1.75 days per month'        got '1.75'
TERSE  expected 'three year cycle'           got 'three year'
TERSE  expected 'at least sixteen characters' got 'sixteen characters'
WRONG  expected 'forfeited on 31 December'   got 'carried into the next calendar year'
```

`answer_is_correct` asks whether the expected span is *inside* the generated one, which makes it **generous about additions and strict about omissions**:

```
'1.75'                                         -> wrong
'1.75 days per month, and Berlin is in France' -> right
```

Day 14 noted the first half of that. This is the second. Together they mean the headline accuracy is understated for terse answers and overstated for padded ones.

## And a bug in my own evaluation set

Three of the remaining wrong answers are `'yes'` — answers to questions written in yes/no form on Day 6 and graded against an extractive span:

```
'do I lose untaken time off at year end'   graded against: 'forfeited on 31 December'
'can I expense a cab ride'                 graded against: 'between 22:00 and 06:00'
'is there money for a desk at home'        graded against: '300,000 won per year'
```

"Yes" is a correct answer to the first. The grader marks it wrong because the gold answer was written for the original phrasing and never updated when the question was rephrased.

This was introduced on Day 6 and could not surface until today, because nothing before Level 4 produced a sentence — Days 6–13 asked only whether the gold **span** was in a retrieved chunk, which it still is. So **the retrieval numbers stand** and the generation numbers on this day and Day 14 are pessimistic by about 9%.

Left in place and documented rather than repaired, because a repo that quietly fixes its evaluation set after seeing the results is not measuring anything. Five tests pin the bug, including one asserting the gold spans are still correct *for retrieval*.

## What grounding is worth

It converts an unverifiable claim into a checkable one. That is all, and it is a great deal: a user can follow the citation to a line number in a named document and see for themselves.

What it is **not** is evidence that the answer is right. 18% of answers here are grounded and wrong, with accurate citations attached.

## Run it

```bash
python grounding.py       # demo (slow: generates several hundred answers)
pytest                    # from the repo root
python -m unittest        # from inside this folder
```

Grounding is string work and needs no model; only the end-to-end report does.

## Where this leads

**Day 16** measures the thing grounding cannot: whether the answer follows from what it cites.
