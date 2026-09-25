# Day 4 · Documents That Break Chunkers: Tables, Code, and Korean

> Level 1 has treated every document as prose. Three kinds are not, and each breaks a different assumption.

## 1. A table row without its header answers nothing

```
strategy                      chunks  header kept   index
fixed(120)                         4          25%    1.00
fixed(200)                         3          33%    1.00
fixed(320)                         2          50%    1.00
chunk_table(2 rows)                6         100%    1.57
chunk_table(4 rows)                3         100%    1.23
chunk_table(6 rows)                2         100%    1.11
```

Here is a chunk that lost it:

```
'uipment | 500,000 | yes |'
'| software licence | 200,000 | yes |'
'| conference fee | 800,000 | yes |'
'| client entertainment'
```

Values with no idea which column is which — the answer to *"do I need a receipt for a taxi"* is in that table and unreadable. The first line is not even a whole row.

Repeating the header in every chunk fixes it and costs index size: **1.11× to 1.57×**, depending on rows per chunk. That is Day 1's overlap trade in a different shape — buy recall, pay in duplication. A test asserts the header repeats and the *data* does not.

## 2. A code fragment that does not parse

```
strategy                      chunks   parse   mean    max
fixed(120)                         6     0/6    109    120
fixed(240)                         3     0/3    219    240
fixed(480)                         2     0/2    328    480
chunk_python()                     5     5/5    131    389
```

**Not one character-count chunk parses, at any size.** Cutting anywhere inside a function leaves an indented fragment, and Python cannot read that as a program:

```
' * ACCRUAL_PER_MONTH, 2)'
```

`chunk_python` takes boundaries from `ast` instead — each top-level statement, class or function, with its decorators:

```
  28 chars  '"""Leave accrual rules."""'
  25 chars  'ACCRUAL_PER_MONTH = 1.75'
  22 chars  'CARRY_OVER_CAP = 10'
 192 chars  'def accrue(months: float) -> float:'
 389 chars  'class LeaveAccount:'
```

The price is the last column of the table above: **28 to 389 characters**. Character chunking guarantees a maximum size and guarantees nothing about meaning; structure chunking is the other way round. A single large function is one large chunk, because the alternatives are a fragment that does not parse or a cut inside a loop body. A test asserts that size spread directly.

## 3. The same budget is not the same budget

```
english   295 chars    59.0 chars/sentence
korean    146 chars    29.2 chars/sentence
```

The same five facts. Korean says it in **49%** of the characters, so a character budget buys different amounts of content:

```
 budget  en chunks  en facts/chunk  ko chunks  ko facts/chunk
     60          5             1.0          3             1.7
    100          5             1.0          2             2.5
    150          3             1.7          1             5.0
    200          2             2.5          1             5.0
```

A size that fits **one** fact in English fits **two** in Korean. Day 3 found that "best chunk size" is undefined until the context budget and the retriever are pinned. This is the third thing it depends on, and it is the language.

## 4. The abbreviation bug Day 1 declared and did not fix

Day 1's splitter cuts after any of `.` `!` `?` followed by whitespace, and its own docstring says it is wrong on abbreviations. This is not a discovery; it is the bill arriving.

```
Day 1's splitter:              with abbreviations guarded:
  'See Fig.'                     'See Fig. 3 for the accrual curve.'
  '3 for the accrual curve.'     'Leave accrues at 1.75 days per month.'
  'Leave accrues at 1.75 ...'    'Contact the finance dept. before submitting.'
  'Contact the finance dept.'    'Approx. 60 days is the limit.'
  'before submitting.'
  'Approx.'
  '60 days is the limit.'
```

Seven sentences where there are four, three of them fragments.

Note what was *never* broken: `1.75` survives both, because a digit follows the period rather than whitespace. A test pins that, so the fix is not credited with solving something it never touched.

The fix is a list of abbreviations, which means it is **permanently incomplete** — it knows `Fig.` and not `Abb.`, and no list covers every abbreviation in every language. A test asserts that `Abb.` still breaks it. Better than nothing, worse than a model, and worth saying rather than quietly shipping either.

## What Level 1 established

Chunking is not a preprocessing step that happens before the interesting part. It sets a ceiling nothing downstream can lift:

| Day | The finding |
|-----|-------------|
| 1 | Overlap buys boundary recall, paid for in index size |
| 2 | Offsets into a normalized copy point at nothing, and the failure is silent on ASCII |
| 3 | "Best chunk size" is undefined until the context budget *and* the retriever are fixed |
| 4 | And until the document is prose. Tables, code and a second language each need a different splitter, and each fix costs something measurable |

## Run it

```bash
# demo (all four breakages, and what each fix costs)
python awkward.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_chunking/day04_awkward_documents

# tests — from inside this folder
python -m unittest
python test_awkward.py

# the docstring examples are runnable too
python -m doctest awkward.py
```

## Where this leads

Level 1 is done and the index is honest about what it contains. **Day 5** starts retrieving from it, with BM25 — the baseline that has to be beaten before anything with an embedding model in it is worth the trouble.
