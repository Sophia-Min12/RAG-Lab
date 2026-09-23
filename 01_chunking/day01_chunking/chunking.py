"""Day 1 — Chunking: what actually goes in the index.

A retriever does not return documents, it returns **chunks**. That choice
is made before any embedding model is loaded, it is almost never revisited,
and it caps everything downstream: a fact split across two chunks can be
retrieved by neither, and no reranker recovers it.

Three strategies, in increasing order of how much they know about text:

- **fixed** — every ``n`` characters. Knows nothing, never fails, and cuts
  words in half.
- **sentence** — pack whole sentences up to a budget. Respects the unit
  that carries meaning, at the cost of uneven chunk sizes.
- **overlap** — repeat the tail of each chunk at the head of the next, so
  a fact spanning a boundary survives in at least one piece. Costs index
  size in exact proportion to the overlap.

The day's measurable claim is the last one: **overlap buys boundary
recall and is paid for in duplication**, and both halves of that sentence
are numbers rather than opinions.
"""

from __future__ import annotations

import re
import unicodedata

#: A chunk knows where it came from. Day 2 turns this into a citation; on
#: Day 1 it is already what makes "which chunk answered this?" answerable.
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, and where it came from.

    ``start`` and ``end`` are character offsets into the original text, so
    ``text[chunk.start:chunk.end] == chunk.text`` always holds — a test
    pins that for every strategy.

    >>> Chunk("hello", 0, 5, 0).text
    'hello'
    """

    text: str
    start: int
    end: int
    index: int

    def __len__(self) -> int:
        return len(self.text)


def _require_text(text) -> str:
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    return unicodedata.normalize("NFC", text)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into sentences, keeping each one's character offsets.

    Reused in spirit from NLP-Lab Day 1, with one change that matters
    here: offsets are carried along. A chunker that loses them cannot
    produce a citation, and Day 2 needs citations.

    The rule is the same naive one — split after ``.``, ``!``, ``?`` or a
    Korean sentence ender followed by whitespace — and it is wrong on
    abbreviations for the same reasons it was wrong there.

    >>> [s for s, _, _ in split_sentences("One. Two! Three?")]
    ['One.', 'Two!', 'Three?']
    """
    text = _require_text(text)
    spans = []
    position = 0
    for match in re.finditer(r"[.!?。！？]+(?:\s+|$)|\n{2,}", text):
        end = match.end()
        piece = text[position:end]
        if piece.strip():
            spans.append((piece.strip(), position + len(piece) - len(piece.lstrip()),
                          position + len(piece.rstrip())))
        position = end
    if position < len(text) and text[position:].strip():
        piece = text[position:]
        spans.append((piece.strip(), position + len(piece) - len(piece.lstrip()),
                      position + len(piece.rstrip())))
    return spans


def chunk_fixed(text: str, size: int = 200, overlap: int = 0) -> list[Chunk]:
    """Cut every ``size`` characters, optionally repeating ``overlap`` of them.

    The stride is ``size - overlap``, which must be positive — an overlap
    equal to the size would advance nowhere and loop forever. That is
    raised rather than clamped, because silently ignoring the argument is
    how a config file ends up producing a 40 GB index.

    >>> [c.text for c in chunk_fixed("abcdefgh", size=3)]
    ['abc', 'def', 'gh']
    >>> [c.text for c in chunk_fixed("abcdefgh", size=4, overlap=2)]
    ['abcd', 'cdef', 'efgh']
    """
    text = _require_text(text)
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"overlap {overlap} must be smaller than size {size}; "
            "otherwise the window never advances"
        )
    if not text:
        return []

    stride = size - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(text[start:end], start, end, len(chunks)))
        if end == len(text):
            break
        start += stride
    return chunks


def chunk_sentences(text: str, max_chars: int = 200) -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``max_chars``.

    A sentence longer than the budget becomes its own chunk rather than
    being split — the strategy's entire premise is that sentence
    boundaries are the ones worth keeping, so breaking one to satisfy a
    size limit would defeat it. The oversized chunk is a real consequence
    and the demo reports it.

    >>> [c.text for c in chunk_sentences("A one. B two. C three.", max_chars=14)]
    ['A one. B two.', 'C three.']
    """
    text = _require_text(text)
    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[tuple[str, int, int]] = []

    def flush() -> None:
        if not current:
            return
        start, end = current[0][1], current[-1][2]
        chunks.append(Chunk(text[start:end], start, end, len(chunks)))
        current.clear()

    for sentence in sentences:
        piece, start, end = sentence
        if current and (end - current[0][1]) > max_chars:
            flush()
        current.append(sentence)
        if len(piece) > max_chars:
            flush()
    flush()
    return chunks


def coverage(text: str, chunks: list[Chunk]) -> float:
    """Share of the original characters that appear in at least one chunk.

    Anything below 1.0 means the index cannot answer questions about the
    missing span, whatever the embedding model does. A test requires every
    strategy to reach full coverage on ordinary prose.
    """
    text = _require_text(text)
    if not text:
        return 1.0
    covered = bytearray(len(text))
    for chunk in chunks:
        covered[chunk.start:chunk.end] = b"\x01" * (chunk.end - chunk.start)
    return sum(covered) / len(text)


def duplication(text: str, chunks: list[Chunk]) -> float:
    """Total chunk length divided by document length.

    1.0 means no overlap. 1.5 means the index is half again as large as
    the corpus, which is exactly what overlap costs — in storage, in
    embedding calls, and in money.
    """
    text = _require_text(text)
    if not text:
        return 0.0
    return sum(len(c) for c in chunks) / len(text)


def spans_a_boundary(chunks: list[Chunk], phrase: str) -> bool:
    """Whether ``phrase`` survives intact inside at least one chunk.

    This is the only question chunking really has to answer. A fact split
    across two chunks is retrievable by neither, because neither contains
    it.

    ``"answer is"`` straddles the cut at character 10, so no chunk holds
    it until the overlap is wide enough to span it.

    >>> spans_a_boundary(chunk_fixed("the answer is 42", size=10), "answer is")
    False
    >>> spans_a_boundary(chunk_fixed("the answer is 42", size=10, overlap=6), "answer is")
    True
    """
    return any(phrase in chunk.text for chunk in chunks)


SAMPLE = (
    "Retrieval-augmented generation retrieves passages and then generates an answer. "
    "The retriever is the part that usually fails. A chunk that splits a fact in half "
    "cannot be retrieved by any query, because neither half contains the fact. "
    "Overlap repeats the end of one chunk at the start of the next. "
    "It costs index size in direct proportion to the overlap fraction. "
    "검색 증강 생성은 문서를 먼저 찾고 그 다음에 답을 만든다. "
    "청킹 전략은 검색 품질의 상한을 결정한다."
)

#: A phrase that straddles the cut a fixed(120) chunker makes at character
#: 120, in the middle of the word "fails". Nothing about it is unusual -
#: that is the point. Any boundary lands somewhere.
BOUNDARY_FACT = "the part that usually fails"


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print(f"document: {len(SAMPLE)} characters\n")

    print("three strategies on the same text:")
    print(f"  {'strategy':<24}{'chunks':>8}{'mean len':>10}{'coverage':>10}{'duplication':>13}")
    for label, chunks in (
        ("fixed(120)", chunk_fixed(SAMPLE, 120)),
        ("fixed(120, overlap=30)", chunk_fixed(SAMPLE, 120, overlap=30)),
        ("fixed(120, overlap=60)", chunk_fixed(SAMPLE, 120, overlap=60)),
        ("sentence(120)", chunk_sentences(SAMPLE, 120)),
    ):
        mean = sum(len(c) for c in chunks) / len(chunks)
        print(f"  {label:<24}{len(chunks):>8}{mean:>10.1f}"
              f"{coverage(SAMPLE, chunks):>10.0%}{duplication(SAMPLE, chunks):>13.2f}")

    print("\nfixed chunking cuts words in half:")
    for chunk in chunk_fixed(SAMPLE, 120)[:3]:
        print(f"  [{chunk.index}] ...{chunk.text[-28:]!r}")
    print("  sentence chunking does not:")
    for chunk in chunk_sentences(SAMPLE, 120)[:3]:
        print(f"  [{chunk.index}] ...{chunk.text[-28:]!r}")

    print(f"\ndoes the fact {BOUNDARY_FACT!r} survive intact?")
    print(f"  {'strategy':<24}{'survives':>10}{'duplication':>13}")
    for label, chunks in (
        ("fixed(120)", chunk_fixed(SAMPLE, 120)),
        ("fixed(120, overlap=30)", chunk_fixed(SAMPLE, 120, overlap=30)),
        ("fixed(120, overlap=60)", chunk_fixed(SAMPLE, 120, overlap=60)),
        ("sentence(120)", chunk_sentences(SAMPLE, 120)),
    ):
        survives = spans_a_boundary(chunks, BOUNDARY_FACT)
        print(f"  {label:<24}{('yes' if survives else 'NO'):>10}"
              f"{duplication(SAMPLE, chunks):>13.2f}")
    print("  that is the whole trade: overlap buys boundary recall and is paid")
    print("  for in index size, embedding calls and money")

    print("\nevery chunk points back at the source:")
    for chunk in chunk_sentences(SAMPLE, 120)[:2]:
        print(f"  [{chunk.index}] chars {chunk.start}..{chunk.end}  "
              f"exact match: {SAMPLE[chunk.start:chunk.end] == chunk.text}")
    print("  Day 2 turns those offsets into a citation")

    print("\nan overlap as large as the chunk is refused, not clamped:")
    try:
        chunk_fixed(SAMPLE, size=100, overlap=100)
    except ValueError as error:
        print(f"  ValueError: {error}")
