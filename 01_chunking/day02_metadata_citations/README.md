# Day 2 · Metadata and Citation Spans: Pointing Back at the Source

> A retrieved chunk is half an answer. The other half is *where it came from* — which document, which line, which characters — so a reader can check the claim instead of trusting it.

Storing that is harder than it looks, and the reason is Day 1.

## Day 1's invariant is true, and not enough

Day 1 guarantees `text[chunk.start:chunk.end] == chunk.text`. Its tests pass. But Day 1 normalizes its input to NFC first and returns offsets into *that* string, while the caller still holds the original file:

```
document        raw chars  NFC chars   drift
handbook-en           132        132       0
handbook-ko           149         70     -79
notes-fr               63         59      -4
```

Korean written on macOS arrives **decomposed** — `한` stored as three code points, `ᄒ ᅡ ᆫ`. NFC composes it back to one. Three short lines measure 149 characters raw and 70 after normalization, so every offset past the first points somewhere else entirely.

## The failure has no symptom

```
citations built without mapping the offsets back:
document        chunks   verify   first quote
handbook-en          4      4/4   'Chunking decides what can be retri...'
handbook-ko          2      0/2   '청킹은 검색 가능한 단위를...'
notes-fr             2      0/2   'Le café est naïve au sujet des o...'

and with normalize_with_map doing its job:
handbook-en          4      4/4   'Chunking decides what can be retri...'
handbook-ko          2      2/2   '청킹은 검색 가능한 단위를...'
notes-fr             2      2/2   'Le café est naïve au sujet des o...'
```

**The English row is identical in both tables.** That is the whole danger: on ASCII the broken version and the correct one produce byte-identical output, so it passes review, ships, and fails on the first document somebody wrote on a Mac.

Here is what a reader is actually shown:

```
chunk 1, as the pipeline embedded it:
  '정규화된 사본의 오프셋은 아무것도 가리키지 않는다.'

what the unmapped citation quotes from the file:
  chars 40..70 -> 'ᆼ한다.\n인용은 원본 문서'

what the mapped citation quotes:
  chars 89..149 -> '정규화된 사본의 오프셋은 아무것도 가리키지 않는다.'
```

No error, no exception, no empty result. A quotation of the wrong sentence — beginning mid-syllable on an orphaned trailing jamo, a character no reader would ever see in a document — attached to a real answer and looking exactly as trustworthy as a correct one.

## The fix, and the trap inside the fix

`normalize_with_map(raw)` returns the normalized text *and* a list where `mapping[i]` is the offset in `raw` that normalized character `i` came from. A span `[a, b)` in normalized text maps back to `raw[mapping[a]:mapping[b]]`.

It works by normalizing one **combining sequence** at a time — a starter plus everything that attaches to it. Character-by-character would compose nothing; all-at-once gives no way to recover where anything went.

The trap is in deciding what "attaches to it" means. The textbook answer is a non-zero canonical combining class:

```python
unicodedata.combining("́")  # combining acute -> 230
unicodedata.combining("ᅡ")  # hangul jungseong A -> 0
```

**Hangul jamo have a combining class of zero.** NFC composes them algorithmically, not through the combining-class rules. Group by combining class alone and `ᄒ ᅡ ᆫ` becomes three groups, each normalized alone, composing nothing — the function silently returns decomposed text and every test on Latin passes. A test pins that failure mode directly.

## The invariant this day establishes

```
normalize("NFC", document.text[chunk.start:chunk.end]) == chunk.text
```

Verified for all 38 chunks across 3 documents × 2 strategies × 3 sizes, plus every span of a short mixed-script string checked exhaustively — every `(start, end)` pair, not a sample.

Day 1's invariant was the same statement without the `normalize()`, which holds only while the document already happens to be NFC.

## Metadata

```
6 chunks across 3 documents
  select({'lang': 'ko'})                      -> 2 chunks
  select({'lang': ['en', 'fr']})              -> 4 chunks
  select({'section': 'notes'})                -> 1 chunks
  select({'section': 'retrieval', 'lang': 'en'}) -> 3 chunks
```

Filtering *before* the expensive part of retrieval is the practical reason metadata exists: searching one section of one manual beats searching everything and discarding most of it, and the filter costs nothing.

Citing a chunk against the wrong document raises rather than guessing — it is the one mistake nothing downstream can catch:

```
ValueError: chunk belongs to 'handbook-ko', not 'handbook-en'
```

## Run it

```bash
# demo (citations, the drift, the wrong quote, metadata)
python citations.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_chunking/day02_metadata_citations

# tests — from inside this folder
python -m unittest
python test_citations.py

# the docstring examples are runnable too
python -m doctest citations.py
```

## Where this leads

Chunks now carry provenance, so a chunk can be pointed at and a citation can be checked. **Day 3** asks the question this level exists for: what chunk size actually produces better answers — measured, not assumed.
