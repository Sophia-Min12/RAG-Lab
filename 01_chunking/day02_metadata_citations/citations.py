"""Day 2 — Metadata and citation spans: pointing back at the source.

A retrieved chunk is only half an answer. The other half is *where it came
from*: which document, which line, which characters — so a reader can check
the claim instead of trusting it.

Storing that turns out to be harder than it looks, and the reason is Day 1.
Day 1 normalizes its input to NFC and returns offsets into the normalized
text. Its round-trip test passes. But the caller still holds the original
document, and against **that** string the offsets are wrong by however much
normalization changed the length. Three short lines of Korean stored
decomposed measure 149 characters raw and 70 after NFC: every offset past
the first is wrong, and the first citation quotes text starting in the
middle of a syllable.

The failure has no symptom. Nothing raises, nothing is empty, and on ASCII
the broken version and the correct one are byte-identical. The citation
just quotes the wrong sentence, attached to a real answer, looking exactly
as trustworthy as a right one.

This day builds the offset map that fixes it, the invariant that catches
it, and the metadata that makes filtering possible before retrieval rather
than after.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

# --- reused from day01 -----------------------------------------------------
# Byte-identical to 01_chunking/day01_chunking/chunking.py. Every day folder
# is self-contained, so nothing here imports across folders.



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


# --- Day 2 -----------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """A source document, holding its text **exactly as it arrived**.

    The text is never normalized in place. That is the whole discipline of
    this day: the document is the thing a citation has to point at, so the
    moment you replace it with a cleaned-up copy, every offset you have
    stored refers to a string the reader does not have.

    >>> Document("d1", "hello").doc_id
    'd1'
    """

    doc_id: str
    text: str
    title: str = ""
    uri: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("a document needs an id; citations are keyed by it")


@dataclass(frozen=True)
class SourcedChunk:
    """A chunk that knows which document it came from.

    ``start`` and ``end`` are offsets into ``Document.text`` — the
    original, un-normalized text. ``text`` is the normalized chunk, which
    is what gets embedded. The two are related by

        normalize("NFC", document.text[chunk.start:chunk.end]) == chunk.text

    and **not** by ``document.text[start:end] == text``, which is the
    assumption Day 1 could make only because it never saw the original.
    """

    doc_id: str
    text: str
    start: int
    end: int
    index: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class Citation:
    """Everything needed to point a human at the evidence.

    Offsets alone are useless to a reader — nobody counts to character
    4,217. A citation carries the line and column too, and the quoted
    text, so it can be checked by eye as well as by ``verify_citation``.
    """

    doc_id: str
    start: int
    end: int
    line: int
    column: int
    quote: str
    title: str = ""
    uri: str = ""


def _is_continuation(character: str) -> bool:
    """Does this character combine with the one before it?

    Two cases matter. A combining mark (``U+0301`` and friends) has a
    non-zero canonical combining class, which is the textbook answer.
    Hangul is the case the textbook answer misses: jamo have a combining
    class of **zero**, and NFC composes them algorithmically rather than
    through the combining-class rules. Grouping only by combining class
    therefore splits ``ᄒ ᅡ ᆫ`` into three groups, normalizes each alone,
    composes nothing, and silently returns decomposed text.

    >>> _is_continuation("\\u0301")   # combining acute
    True
    >>> _is_continuation("\\u1161")   # hangul jungseong A
    True
    >>> _is_continuation("a")
    False
    """
    return bool(unicodedata.combining(character)) or 0x1160 <= ord(character) <= 0x11FF


def normalize_with_map(raw: str, form: str = "NFC") -> tuple[str, list[int]]:
    """Normalize ``raw`` and return the offsets that map back to it.

    ``mapping[i]`` is the offset **in raw** where normalized character
    ``i`` came from, and ``mapping`` has one extra entry at the end so a
    half-open span ``[a, b)`` in the normalized text maps to
    ``raw[mapping[a]:mapping[b]]``.

    The text is walked one *combining sequence* at a time — a starter plus
    everything that attaches to it — and each sequence is normalized on
    its own. Normalizing character by character would compose nothing;
    normalizing the whole string at once gives no way to recover where
    anything went.

    >>> normalized, mapping = normalize_with_map("ab")
    >>> normalized, mapping
    ('ab', [0, 1, 2])
    >>> normalized, mapping = normalize_with_map("a\\u0301b")
    >>> normalized, mapping
    ('\\xe1b', [0, 2, 3])
    """
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"unknown normalization form {form!r}")
    pieces: list[str] = []
    mapping: list[int] = []
    start, length = 0, len(raw)
    while start < length:
        end = start + 1
        while end < length and _is_continuation(raw[end]):
            end += 1
        piece = unicodedata.normalize(form, raw[start:end])
        pieces.append(piece)
        mapping.extend([start] * len(piece))
        start = end
    mapping.append(length)
    return "".join(pieces), mapping


def offset_drift(raw: str, form: str = "NFC") -> int:
    """How many characters normalization adds or removes.

    Zero for text that is already in ``form``. Every non-zero value is a
    citation pointing at the wrong place.

    >>> offset_drift("plain ascii")
    0
    """
    return len(unicodedata.normalize(form, raw)) - len(raw)


def line_column(text: str, offset: int) -> tuple[int, int]:
    """1-based line and column of a character offset.

    Both 1-based, because that is what every editor, compiler and code
    reviewer means by "line 3". Returning a 0-based line here would be
    correct and useless.

    >>> line_column("ab\\ncd", 0)
    (1, 1)
    >>> line_column("ab\\ncd", 3)
    (2, 1)
    """
    if not 0 <= offset <= len(text):
        raise IndexError(f"offset {offset} outside 0..{len(text)}")
    before = text[:offset]
    line = before.count("\n") + 1
    column = offset - (before.rfind("\n") + 1) + 1
    return line, column


def chunk_document(document: Document, size: int = 200, overlap: int = 0,
                   strategy: str = "fixed", **metadata) -> list[SourcedChunk]:
    """Chunk a document, keeping every offset pointed at the original.

    The text is normalized once, chunked by Day 1's strategies, and then
    every chunk's span is mapped **back** through ``normalize_with_map``
    so it indexes ``document.text`` rather than the normalized copy.

    That last step is the entire day. Without it the offsets are correct
    against a string that exists only inside this function.
    """
    if strategy not in {"fixed", "sentence"}:
        raise ValueError(f"unknown strategy {strategy!r}; expected fixed or sentence")
    normalized, mapping = normalize_with_map(document.text)
    if strategy == "fixed":
        chunks = chunk_fixed(normalized, size=size, overlap=overlap)
    else:
        chunks = chunk_sentences(normalized, max_chars=size)
    merged = {**dict(document.metadata), **metadata}
    return [
        SourcedChunk(
            doc_id=document.doc_id,
            text=chunk.text,
            start=mapping[chunk.start],
            end=mapping[chunk.end],
            index=chunk.index,
            metadata=merged,
        )
        for chunk in chunks
    ]


def chunk_document_unmapped(document: Document, size: int = 200,
                            overlap: int = 0) -> list[SourcedChunk]:
    """The same thing with the mapping step left out. Kept to measure it.

    This is the version almost everyone writes first, and on ASCII it is
    indistinguishable from the correct one — which is exactly why it
    survives code review and ships. The demo runs both.
    """
    normalized, _ = normalize_with_map(document.text)
    return [
        SourcedChunk(doc_id=document.doc_id, text=chunk.text, start=chunk.start,
                     end=chunk.end, index=chunk.index, metadata=dict(document.metadata))
        for chunk in chunk_fixed(normalized, size=size, overlap=overlap)
    ]


def cite(document: Document, chunk: SourcedChunk, quote_chars: int = 60) -> Citation:
    """Turn a chunk into something a reader can follow.

    The quote is taken from ``document.text`` - the original - so that
    what the citation shows is what the file contains, not what the
    pipeline made of it.
    """
    if chunk.doc_id != document.doc_id:
        raise ValueError(
            f"chunk belongs to {chunk.doc_id!r}, not {document.doc_id!r}; "
            "citing across documents is the one mistake nothing downstream catches"
        )
    line, column = line_column(document.text, chunk.start)
    quote = document.text[chunk.start:chunk.end]
    if len(quote) > quote_chars:
        quote = quote[:quote_chars].rstrip() + "..."
    return Citation(doc_id=document.doc_id, start=chunk.start, end=chunk.end,
                    line=line, column=column, quote=quote,
                    title=document.title, uri=document.uri)


def verify_citation(document: Document, chunk: SourcedChunk) -> bool:
    """Does the span still contain what the chunk says it does?

    Compares under normalization, because the document is raw and the
    chunk is normalized. This is the assertion that would have caught the
    unmapped version before it reached a user.

    >>> doc = Document("d", "hello world")
    >>> verify_citation(doc, chunk_document(doc, size=5)[0])
    True
    """
    slice_ = document.text[chunk.start:chunk.end]
    return unicodedata.normalize("NFC", slice_) == chunk.text


def select(chunks: list[SourcedChunk], **filters) -> list[SourcedChunk]:
    """Chunks whose metadata matches every filter exactly.

    Filtering **before** the expensive part of retrieval is the practical
    reason metadata exists: searching one section of one manual beats
    searching everything and discarding most of it. A value given as a
    list or set matches any member.

    >>> chunks = [SourcedChunk("d", "x", 0, 1, 0, {"lang": "en"})]
    >>> len(select(chunks, lang="en")), len(select(chunks, lang="ko"))
    (1, 0)
    """
    def matches(chunk: SourcedChunk) -> bool:
        for key, wanted in filters.items():
            value = chunk.metadata.get(key)
            if isinstance(wanted, (list, tuple, set, frozenset)):
                if value not in wanted:
                    return False
            elif value != wanted:
                return False
        return True

    return [chunk for chunk in chunks if matches(chunk)]


#: Three documents whose only differences are the ones that break things:
#: one plain ASCII, one with Korean stored decomposed (which is how macOS
#: filesystems hand it to you), one with Latin combining marks.
def _sample_documents() -> list[Document]:
    english = Document(
        doc_id="handbook-en",
        title="Retrieval Handbook",
        uri="docs/handbook.md",
        text=(
            "Chunking decides what can be retrieved.\n"
            "A citation must point at the source document.\n"
            "Offsets into a cleaned copy point at nothing.\n"
        ),
        metadata={"lang": "en", "section": "retrieval"},
    )
    korean = Document(
        doc_id="handbook-ko",
        title="검색 핸드북",
        uri="docs/handbook.ko.md",
        # Stored decomposed, exactly as a file written on macOS arrives.
        text=unicodedata.normalize(
            "NFD",
            "청킹은 검색 가능한 단위를 결정한다.\n"
            "인용은 원본 문서를 가리켜야 한다.\n"
            "정규화된 사본의 오프셋은 아무것도 가리키지 않는다.\n",
        ),
        metadata={"lang": "ko", "section": "retrieval"},
    )
    accented = Document(
        doc_id="notes-fr",
        title="Notes",
        uri="docs/notes.txt",
        text=unicodedata.normalize(
            "NFD", "Le café est naïve au sujet des offsets.\nRéférence: page 3.\n"),
        metadata={"lang": "fr", "section": "notes"},
    )
    return [english, korean, accented]


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    english, korean, accented = _sample_documents()

    print("a chunk that knows where it came from:")
    for chunk in chunk_document(english, size=45, strategy="sentence")[:3]:
        citation = cite(english, chunk)
        print(f"  [{chunk.index}] line {citation.line}, col {citation.column}  "
              f"chars {citation.start}..{citation.end}")
        print(f"      {citation.quote!r}")
    print("  offsets alone are useless to a reader; nobody counts to character 4217")

    print()
    print("=" * 68)
    print("the same code on text that is not ASCII")
    print("=" * 68)
    print(f"  {'document':<14}{'raw chars':>11}{'NFC chars':>11}{'drift':>8}")
    for document in (english, korean, accented):
        print(f"  {document.doc_id:<14}{len(document.text):>11}"
              f"{len(unicodedata.normalize('NFC', document.text)):>11}"
              f"{offset_drift(document.text):>8}")
    print()
    print("  Day 1 normalizes its input to NFC and returns offsets into the")
    print("  result. That is correct, and it is documented, and the round-trip")
    print("  test passes - against the normalized string. The caller holds the")
    print("  original, where those offsets mean something else entirely.")

    print()
    print("  citations built without mapping the offsets back:")
    print(f"  {'document':<14}{'chunks':>8}{'verify':>9}   first quote")
    for document in (english, korean, accented):
        chunks = chunk_document_unmapped(document, size=40)
        good = sum(verify_citation(document, chunk) for chunk in chunks)
        quote = cite(document, chunks[0], quote_chars=34).quote
        print(f"  {document.doc_id:<14}{len(chunks):>8}{f'{good}/{len(chunks)}':>9}   {quote!r}")

    print()
    print("  and with normalize_with_map doing its job:")
    print(f"  {'document':<14}{'chunks':>8}{'verify':>9}   first quote")
    for document in (english, korean, accented):
        chunks = chunk_document(document, size=40)
        good = sum(verify_citation(document, chunk) for chunk in chunks)
        quote = cite(document, chunks[0], quote_chars=34).quote
        print(f"  {document.doc_id:<14}{len(chunks):>8}{f'{good}/{len(chunks)}':>9}   {quote!r}")

    print()
    print("  the English row is identical in both tables. That is the whole")
    print("  danger: on ASCII the broken version is indistinguishable from the")
    print("  correct one, so it passes review, ships, and fails on the first")
    print("  document somebody wrote on a Mac.")

    print()
    print("=" * 68)
    print("what a wrong offset actually shows the reader")
    print("=" * 68)
    chunk = chunk_document_unmapped(korean, size=40)[1]
    correct = chunk_document(korean, size=40)[1]
    print(f"  chunk 1, as the pipeline embedded it:")
    print(f"    {unicodedata.normalize('NFC', chunk.text)[:34]!r}")
    print(f"  what the unmapped citation quotes from the file:")
    print(f"    chars {chunk.start}..{chunk.end} -> "
          f"{unicodedata.normalize('NFC', korean.text[chunk.start:chunk.end])[:34]!r}")
    print(f"  what the mapped citation quotes:")
    print(f"    chars {correct.start}..{correct.end} -> "
          f"{unicodedata.normalize('NFC', korean.text[correct.start:correct.end])[:34]!r}")
    print()
    print("  not an error, not an exception, not a crash. A quotation that is")
    print("  simply of the wrong sentence, attached to a real answer, looking")
    print("  exactly as trustworthy as a correct one.")

    print()
    print("=" * 68)
    print("metadata, and filtering before the expensive part")
    print("=" * 68)
    everything = [chunk for document in (english, korean, accented)
                  for chunk in chunk_document(document, size=60)]
    print(f"  {len(everything)} chunks across 3 documents")
    for filters in ({"lang": "ko"}, {"lang": ["en", "fr"]}, {"section": "notes"},
                    {"section": "retrieval", "lang": "en"}):
        chosen = select(everything, **filters)
        print(f"    select({filters}) -> {len(chosen)} chunks")
    print("  searching one section of one manual beats searching everything")
    print("  and throwing most of it away, and the filter costs nothing")

    print()
    print("=" * 68)
    print("the invariant this day exists to establish")
    print("=" * 68)
    print("    normalize('NFC', document.text[chunk.start:chunk.end]) == chunk.text")
    print()
    total = 0
    for document in (english, korean, accented):
        for strategy in ("fixed", "sentence"):
            for size in (30, 60, 120):
                chunks = chunk_document(document, size=size, strategy=strategy)
                assert all(verify_citation(document, chunk) for chunk in chunks)
                total += len(chunks)
    print(f"  holds for all {total} chunks across 3 documents x 2 strategies x 3 sizes")
    print("  Day 1's invariant was the same statement without the normalize(),")
    print("  which is true only while the document already happens to be NFC")

    print()
    print("citing a chunk against the wrong document is refused, not guessed:")
    try:
        cite(english, chunk_document(korean, size=40)[0])
    except ValueError as error:
        print(f"  ValueError: {error}")
