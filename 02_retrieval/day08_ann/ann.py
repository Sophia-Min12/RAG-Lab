"""Day 8 scratch."""

from __future__ import annotations

import ast
import math
import re
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

# --- reused from days 1-6 --------------------------------------------------
# Byte-identical to 02_retrieval/day06_embeddings/embeddings.py: the
# chunkers, the offset mapping, the handbook, BM25 and the dense index.


# --- reused from days 1-5 --------------------------------------------------
# Byte-identical to 02_retrieval/day05_bm25/bm25.py: the chunkers, the
# offset mapping, the evaluation harness, the handbook and BM25 itself.


# --- reused from days 1-4 --------------------------------------------------
# Byte-identical to 01_chunking/day04_awkward_documents/awkward.py: the
# chunkers, the offset mapping, the evaluation harness and the handbook.


# --- reused from days 1-3 --------------------------------------------------
# Byte-identical to 01_chunking/day03_chunk_size/chunk_size.py, which
# carries Day 1's chunkers and Day 2's offset mapping forward unchanged.


# --- reused from day01 -----------------------------------------------------
# Byte-identical to 01_chunking/day01_chunking/chunking.py.


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


# --- reused from day02 -----------------------------------------------------
# Byte-identical to 01_chunking/day02_metadata_citations/citations.py. The
# offset mapping matters here: every chunk below still points at its source.


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


# --- Day 3 -----------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """A question, and the exact string that answers it.

    ``answer`` is a literal substring of the source document, which is
    what makes this measurable without a human in the loop: a chunk
    either contains it or does not. That is a weaker notion than "the
    answer is derivable from this chunk", and the gap is Day 12's
    subject.
    """

    query: str
    answer: str
    doc_id: str


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs, keeping the punctuation inside numbers.

    ``35,000`` and ``22:00`` are single tokens; splitting them would make
    "what is the meal allowance" unanswerable for reasons that have
    nothing to do with chunking.

    >>> tokenize("Meals: up to 35,000 won")
    ['meals:', 'up', 'to', '35,000', 'won']
    """
    return re.findall(r"[0-9a-z가-힣:,.]+", text.lower())


def score_occurrences(query: str, text: str) -> float:
    """How many query terms occur, counting repeats.

    The obvious first scorer, and **length-biased**: a longer chunk has
    more tokens and therefore more chances to match, whether or not it is
    more relevant. Kept because the demo measures what that bias does to
    the day's conclusion.

    >>> score_occurrences("core hours", "core hours are core hours")
    4.0
    """
    wanted = set(tokenize(query))
    return float(sum(1 for term in tokenize(text) if term in wanted))


def score_coverage(query: str, text: str) -> float:
    """How many *distinct* query terms occur.

    Bounded by the length of the query rather than the length of the
    chunk, so repetition buys nothing. Still mildly length-biased - a
    longer chunk has more chances to contain each term once - but far
    less so.

    >>> score_coverage("core hours", "core hours are core hours")
    2.0
    """
    wanted = set(tokenize(query))
    return float(len(wanted & set(tokenize(text))))


def retrieve(chunks: list[SourcedChunk], query: str, k: int = 3,
             scorer=score_coverage) -> list[SourcedChunk]:
    """The ``k`` highest scoring chunks, ties broken by document order.

    Deterministic on ties, because a retriever that returns a different
    answer on a re-run cannot be measured.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    ranked = sorted(chunks, key=lambda c: (-scorer(query, c.text), c.doc_id, c.index))
    return ranked[:k]


def retrieve_within_budget(chunks: list[SourcedChunk], query: str,
                           budget: int = 600,
                           scorer=score_coverage) -> list[SourcedChunk]:
    """As many top-ranked chunks as fit in ``budget`` characters.

    The comparison that ``retrieve`` cannot make honestly. Top-3 of a
    corpus cut into 40-character chunks is 120 characters of context;
    top-3 of the same corpus cut into 480-character chunks is 1,189. Any
    difference in hit rate between those two settings is partly a
    difference in how much text was handed over, and the demo shows how
    large "partly" gets.

    Chunks are taken in rank order and packing **stops** at the first one
    that does not fit. The obvious alternative - skip it and keep looking
    for something smaller - was what this function did first, and it is
    wrong in a way worth recording: it silently reorders the ranking by
    what happens to fit, so the top-scoring chunk gets dropped in favour
    of several short low-scoring ones. On fixed(320) at a 300-character
    budget that turned 76% into 18%, by packing four tail-end remainder
    chunks instead of the one chunk that held the answer.

    A single chunk that exceeds the budget on its own is still returned -
    returning nothing would score the configuration as a retrieval
    failure when it is really a chunking one.
    """
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    packed: list[SourcedChunk] = []
    used = 0
    for chunk in sorted(chunks, key=lambda c: (-scorer(query, c.text), c.doc_id, c.index)):
        if used + len(chunk.text) > budget:
            break
        packed.append(chunk)
        used += len(chunk.text)
    if not packed and chunks:
        return retrieve(chunks, query, k=1, scorer=scorer)
    return packed


def answer_is_intact(chunks: list[SourcedChunk], answer: str) -> bool:
    """Does any single chunk contain the whole answer?

    The ceiling on everything downstream. If the answer is split across
    two chunks, no retriever, reranker or model recovers it - they can
    only return a piece. Measuring this separately from the hit rate is
    what distinguishes "the chunker lost it" from "the retriever missed
    it", and the two have opposite fixes.
    """
    return any(answer in chunk.text for chunk in chunks)


def evaluate(chunks: list[SourcedChunk], questions: list[Question],
             scorer=score_coverage, k: int | None = 3,
             budget: int | None = None) -> dict:
    """Hit rate, ceiling, and how much context it took to get there.

    Exactly one of ``k`` and ``budget`` selects the retrieval mode.
    """
    if (k is None) == (budget is None):
        raise ValueError("pass exactly one of k or budget")
    hits = intact = 0
    context_used = []
    for question in questions:
        if answer_is_intact(chunks, question.answer):
            intact += 1
        if budget is None:
            retrieved = retrieve(chunks, question.query, k=k, scorer=scorer)
        else:
            retrieved = retrieve_within_budget(chunks, question.query,
                                               budget=budget, scorer=scorer)
        context_used.append(sum(len(chunk.text) for chunk in retrieved))
        if any(question.answer in chunk.text for chunk in retrieved):
            hits += 1
    total = len(questions)
    return {
        "chunks": len(chunks),
        "mean_chunk": sum(len(c.text) for c in chunks) / len(chunks),
        "intact": intact / total,
        "hit": hits / total,
        "context": sum(context_used) / total,
    }


def build_index(documents: list[Document], size: int,
                strategy: str = "sentence") -> list[SourcedChunk]:
    """Chunk every document at one size. Day 2 keeps the offsets honest."""
    return [chunk for document in documents
            for chunk in chunk_document(document, size=size, strategy=strategy)]


def sweep(documents: list[Document], questions: list[Question], sizes,
          strategy: str = "sentence", scorer=score_coverage,
          k: int | None = 3, budget: int | None = None) -> list[dict]:
    """``evaluate`` across a range of chunk sizes."""
    rows = []
    for size in sizes:
        row = evaluate(build_index(documents, size, strategy), questions,
                       scorer=scorer, k=k, budget=budget)
        row["size"] = size
        rows.append(row)
    return rows


def best_size(rows: list[dict]) -> int:
    """The size with the highest hit rate, smallest size winning a tie.

    Smallest rather than largest on purpose: if two configurations answer
    equally well, the one that hands the model less text is the better
    one, and every later day in this repo pays for context by the token.
    """
    return min(rows, key=lambda row: (-row["hit"], row["size"]))["size"]


#: A small policy handbook. Five documents, one topic each, written so
#: that every answer is a short literal span - which is what makes the
#: measurement possible and is also its main limitation.
def _handbook() -> list[Document]:
    sections = {
        "vacation": (
            "Annual leave accrues at 1.75 days per month of continuous service.\n"
            "Unused leave may be carried into the next calendar year, up to a maximum of ten days.\n"
            "Any balance above ten days is forfeited on 31 December and is not paid out.\n"
            "Requests for more than five consecutive days require approval from a second-line manager.\n"
            "Leave taken during the first three months of employment is unpaid."
        ),
        "expenses": (
            "Expense claims must be submitted within sixty days of the expenditure.\n"
            "Claims submitted after sixty days are rejected automatically and cannot be appealed.\n"
            "Meals are reimbursed up to 35,000 won per day when travelling domestically.\n"
            "International travel is reimbursed at 90,000 won per day for meals and incidentals.\n"
            "A receipt is required for any single item above 20,000 won.\n"
            "Taxi fares are reimbursed only between 22:00 and 06:00, or when carrying equipment."
        ),
        "equipment": (
            "Every engineer is issued a laptop on their first day.\n"
            "Laptops are replaced on a three year cycle, or earlier if a fault cannot be repaired.\n"
            "A second monitor may be requested after the probation period ends.\n"
            "Personal equipment may not be connected to the production network under any circumstances.\n"
            "Lost or stolen equipment must be reported to security within twenty four hours."
        ),
        "remote": (
            "Remote work is permitted for up to three days per week.\n"
            "Employees working remotely must be reachable during core hours, which are 10:00 to 16:00.\n"
            "Working from outside the country requires written approval and a tax review.\n"
            "The company contributes 300,000 won per year towards a home office setup.\n"
            "Core hours do not apply on public holidays."
        ),
        "security": (
            "Passwords must be at least sixteen characters and are rotated every ninety days.\n"
            "Two factor authentication is mandatory for all systems that hold customer data.\n"
            "Production access is granted for a maximum of eight hours at a time.\n"
            "Any suspected breach must be reported to the security team immediately and in writing.\n"
            "Personal cloud storage may not be used for company documents."
        ),
    }
    return [Document(doc_id=name, text=text, title=name.title(),
                     uri=f"handbook/{name}.md", metadata={"section": name})
            for name, text in sections.items()]


#: Seventeen questions whose answers are literal spans of the handbook.
QUESTIONS = [
    Question("how many vacation days accrue each month", "1.75 days per month", "vacation"),
    Question("what happens to unused leave above ten days", "forfeited on 31 December", "vacation"),
    Question("who approves more than five consecutive days of leave", "a second-line manager", "vacation"),
    Question("how long do I have to submit an expense claim", "within sixty days", "expenses"),
    Question("what is the domestic meal allowance", "35,000 won per day", "expenses"),
    Question("what is the international meal allowance", "90,000 won per day", "expenses"),
    Question("when is a receipt required", "above 20,000 won", "expenses"),
    Question("when are taxi fares reimbursed", "between 22:00 and 06:00", "expenses"),
    Question("how often are laptops replaced", "three year cycle", "equipment"),
    Question("when can I request a second monitor", "after the probation period ends", "equipment"),
    Question("how quickly must stolen equipment be reported", "within twenty four hours", "equipment"),
    Question("how many days a week can I work remotely", "three days per week", "remote"),
    Question("what are the core hours", "10:00 to 16:00", "remote"),
    Question("what is the home office budget", "300,000 won per year", "remote"),
    Question("how long must a password be", "at least sixteen characters", "security"),
    Question("how often are passwords rotated", "every ninety days", "security"),
    Question("how long does production access last", "eight hours at a time", "security"),
]

#: The sizes swept throughout the demo.
SIZES = (40, 80, 120, 160, 240, 320, 480)


# --- Day 4 -----------------------------------------------------------------

#: Words that end in a period without ending a sentence. Day 1's splitter
#: cuts after every ``.``, which is right for prose and wrong for these.
ABBREVIATIONS = frozenset({
    "fig.", "no.", "vs.", "e.g.", "i.e.", "etc.", "cf.", "approx.",
    "dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "ave.", "dept.",
    "inc.", "ltd.", "co.", "jan.", "feb.", "mar.", "apr.", "jun.",
    "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
})


def split_sentences_guarded(text: str) -> list[tuple[str, int, int]]:
    """Day 1's splitter, with abbreviations rejoined.

    Day 1 splits after ``.`` followed by whitespace and a capital. That
    handles ``1.75`` correctly - no whitespace follows the period - and
    mishandles ``See Fig. 3 for details``, which it cuts into ``See Fig.``
    and ``3 for details.``.

    The fix is a list, which means it is permanently incomplete: it knows
    ``Fig.`` and not ``Abb.``, and no list covers every abbreviation in
    every language. It is better than nothing and worse than a model, and
    saying so is more useful than pretending the problem is solved.

    >>> [s for s, _, _ in split_sentences_guarded("See Fig. 3 here. Then stop.")]
    ['See Fig. 3 here.', 'Then stop.']
    """
    pieces = split_sentences(text)
    merged: list[tuple[str, int, int]] = []
    for piece, start, end in pieces:
        words = merged[-1][0].split() if merged else []
        if words and words[-1].lower() in ABBREVIATIONS:
            previous_start = merged[-1][1]
            merged[-1] = (text[previous_start:end], previous_start, end)
        else:
            merged.append((piece, start, end))
    return merged


def parses_as_python(text: str) -> bool:
    """Is this fragment syntactically valid Python?

    The one objective test of whether a code chunk survived chunking. A
    fragment that does not parse cannot be run, cannot be type-checked,
    and read on its own tells a reader very little.

    >>> parses_as_python("x = 1")
    True
    >>> parses_as_python("    return x")
    False
    """
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def chunk_python(source: str, max_chars: int | None = None) -> list[Chunk]:
    """Split Python at top-level definitions, so each chunk still parses.

    Boundaries come from ``ast``, not from a character count: each
    top-level statement, class or function becomes one chunk together with
    everything up to the next one.

    **This gives up size control**, and that is the trade rather than an
    oversight. A single 4,000-character function is one 4,000-character
    chunk, because the alternatives are a fragment that does not parse or
    a cut in the middle of a loop body. ``max_chars`` therefore reports
    rather than enforces: the demo counts how many chunks exceed it.

    >>> chunks = chunk_python("x = 1\\n\\n\\ndef f():\\n    return 2\\n")
    >>> len(chunks), all(parses_as_python(c.text) for c in chunks)
    (2, True)
    """
    text = _require_text(source)
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise ValueError(f"source does not parse, so it cannot be split: {error}") from error
    if not tree.body:
        return []

    lines = text.splitlines(keepends=True)
    offsets, running = [], 0
    for line in lines:
        offsets.append(running)
        running += len(line)
    offsets.append(running)

    starts = [node.lineno - 1 for node in tree.body]
    # A decorator sits above the node it decorates and belongs with it.
    for index, node in enumerate(tree.body):
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            starts[index] = min(starts[index], min(d.lineno - 1 for d in decorators))
    bounds = starts + [len(lines)]

    chunks = []
    for index, (first, last) in enumerate(zip(bounds, bounds[1:])):
        piece = text[offsets[first]:offsets[last]]
        if not piece.strip():
            continue
        chunks.append(Chunk(piece, offsets[first], offsets[last], len(chunks)))
    return chunks


@dataclass(frozen=True)
class Table:
    """A parsed markdown table: a header, an alignment row, and body rows."""

    header: str
    alignment: str
    rows: list[str]

    @property
    def preamble(self) -> str:
        return f"{self.header}\n{self.alignment}"


def parse_markdown_table(text: str) -> Table | None:
    """Recognize a markdown table, or return None.

    Deliberately strict: a header line, an alignment line of dashes, and
    at least one body row. Anything else is prose and should be chunked
    as prose.

    >>> parse_markdown_table("| a | b |\\n| --- | --- |\\n| 1 | 2 |").rows
    ['| 1 | 2 |']
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return None
    header, alignment, *rows = lines
    if not header.lstrip().startswith("|"):
        return None
    if not re.fullmatch(r"\|(\s*:?-{2,}:?\s*\|)+", alignment.strip()):
        return None
    return Table(header=header, alignment=alignment, rows=rows)


def chunk_table(text: str, rows_per_chunk: int = 4) -> list[Chunk]:
    """Chunk a markdown table, repeating the header in every chunk.

    A table row without its header is a list of values with no idea what
    the columns mean - ``| meals domestic | 35,000 | no |`` answers
    nothing unless you know the third column is "receipt required".

    Repeating the header costs index size, exactly as Day 1's overlap did,
    and the demo prices it. Offsets point at the **body rows**, so a
    citation still lands on real text: the repeated header is added
    context, not something the document contains at that position.

    >>> chunks = chunk_table("| a | b |\\n| -- | -- |\\n| 1 | 2 |\\n| 3 | 4 |", 1)
    >>> len(chunks), all("| a | b |" in c.text for c in chunks)
    (2, True)
    """
    if rows_per_chunk < 1:
        raise ValueError(f"rows_per_chunk must be at least 1, got {rows_per_chunk}")
    table = parse_markdown_table(text)
    if table is None:
        raise ValueError("this does not look like a markdown table")

    chunks = []
    for index in range(0, len(table.rows), rows_per_chunk):
        group = table.rows[index:index + rows_per_chunk]
        start = text.index(group[0])
        end = text.index(group[-1]) + len(group[-1])
        body = "\n".join(group)
        chunks.append(Chunk(f"{table.preamble}\n{body}", start, end, len(chunks)))
    return chunks


def header_retention(chunks: list[Chunk], header: str) -> float:
    """Fraction of chunks that carry the table header.

    The number that decides whether a retrieved row means anything.
    """
    if not chunks:
        return 0.0
    return sum(1 for chunk in chunks if header in chunk.text) / len(chunks)


def index_growth(text: str, chunks: list[Chunk]) -> float:
    """Total chunk characters over original characters.

    Day 1 called this ``duplication`` for overlap; repeating a table
    header is the same cost in a different shape, so it gets the same
    measurement.
    """
    return sum(len(chunk.text) for chunk in chunks) / len(text)


def density(text: str) -> float:
    """Characters per sentence - a crude proxy for how much a character buys.

    Korean writes the same fact in roughly half the characters English
    needs, so a character budget tuned on one language is a different
    setting in the other. The demo measures it on parallel text rather
    than asserting a ratio.
    """
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    return len(text) / len(sentences)


def facts_per_chunk(text: str, budget: int) -> float:
    """Sentences per chunk at a character budget. The thing that differs."""
    sentences = split_sentences(text)
    chunks = chunk_sentences(text, max_chars=budget)
    if not chunks:
        return 0.0
    return len(sentences) / len(chunks)


#: A markdown table. Twelve rows, one header, and no way to read a row
#: without the header.
EXPENSE_TABLE = (
    "| item | limit | receipt required |\n"
    "| --- | --- | --- |\n"
    "| meals domestic | 35,000 | no |\n"
    "| meals international | 90,000 | no |\n"
    "| taxi late night | actual | yes |\n"
    "| taxi daytime | not reimbursed | n/a |\n"
    "| accommodation | 150,000 | yes |\n"
    "| equipment | 500,000 | yes |\n"
    "| software licence | 200,000 | yes |\n"
    "| conference fee | 800,000 | yes |\n"
    "| client entertainment | 100,000 | yes |\n"
    "| home office | 300,000 | yes |\n"
    "| training course | 1,000,000 | yes |\n"
    "| mobile data | 50,000 | no |\n"
)

#: Python source. Splitting it anywhere but a top-level boundary produces
#: something that does not parse.
POLICY_CODE = '''"""Leave accrual rules."""

ACCRUAL_PER_MONTH = 1.75
CARRY_OVER_CAP = 10


def accrue(months: float) -> float:
    """Days of leave earned after a number of months of service."""
    if months < 3:
        return 0.0
    return round(months * ACCRUAL_PER_MONTH, 2)


class LeaveAccount:
    """A running balance, capped at the carry-over limit each year."""

    def __init__(self, balance: float = 0.0) -> None:
        self.balance = balance

    def accrue_month(self) -> None:
        self.balance += ACCRUAL_PER_MONTH

    def carry_over(self) -> float:
        kept = min(self.balance, CARRY_OVER_CAP)
        self.balance = kept
        return kept
'''

#: The same five facts in English and Korean. Parallel text is the only
#: way to compare density without comparing two different documents.
PARALLEL_EN = (
    "Annual leave accrues at 1.75 days per month of continuous service.\n"
    "Unused leave may be carried into the next calendar year.\n"
    "Any balance above ten days is forfeited on 31 December.\n"
    "Expense claims must be submitted within sixty days.\n"
    "Meals are reimbursed up to 35,000 won per day when travelling.\n"
)

PARALLEL_KO = (
    "연차 휴가는 근속 개월당 1.75일씩 적립된다.\n"
    "사용하지 않은 휴가는 다음 해로 이월할 수 있다.\n"
    "10일을 초과하는 잔여 휴가는 12월 31일에 소멸된다.\n"
    "경비 청구는 지출일로부터 60일 이내에 제출해야 한다.\n"
    "출장 중 식비는 1일 35,000원까지 정산된다.\n"
)

#: Prose that Day 1's splitter gets wrong.
ABBREVIATION_PROSE = (
    "See Fig. 3 for the accrual curve. Leave accrues at 1.75 days per month. "
    "Contact the finance dept. before submitting. Approx. 60 days is the limit."
)


# --- Day 5 -----------------------------------------------------------------


class BM25:
    """Okapi BM25 over a list of chunks.

    Three ideas, each answering a failure of the scorers Day 3 used.

    **Term frequency, saturating.** A chunk mentioning "leave" nine times
    is not nine times as relevant as one mentioning it once.
    ``score_occurrences`` believed it was. BM25 sends ``tf`` through
    ``tf / (tf + k1)``, which rises steeply at first and then flattens:
    the second mention adds much less than the first, and the ninth
    almost nothing.

    **Inverse document frequency.** A query term appearing in every chunk
    distinguishes nothing. Both of Day 3's scorers treated "the" and
    "forfeited" as worth the same.

    **Length normalization.** This is the one Day 3 measured directly: a
    long chunk contains more terms and therefore scores higher for no good
    reason. ``b`` controls the correction, from ``b=0`` (none, the bias
    Day 3 found) to ``b=1`` (divide by length relative to the average).
    The demo sweeps it and the bias reappears exactly where it should.

    The scoring function:

        score(q, d) = Σ  IDF(t) · ──────────────────────────────────
                     t∈q           tf + k1 · (1 - b + b · |d| / avgdl)

                    with the numerator tf · (k1 + 1)

    >>> index = BM25([_text_chunk("the cat sat"), _text_chunk("a dog ran")])
    >>> index.search("cat", k=1)[0][0].text
    'the cat sat'
    """

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75,
                 tokenizer=None) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must not be negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be between 0 and 1, got {b}")
        if not chunks:
            raise ValueError("cannot build an index over no chunks")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.tokenize = tokenizer or tokenize

        self.term_counts: list[dict[str, int]] = []
        self.lengths: list[int] = []
        #: term -> list of (chunk position, term frequency). The inverted
        #: index: it is what lets a query touch only the chunks that could
        #: possibly score, instead of all of them.
        self.postings: dict[str, list[tuple[int, int]]] = {}

        for position, chunk in enumerate(self.chunks):
            counts: dict[str, int] = {}
            for term in self.tokenize(chunk.text):
                counts[term] = counts.get(term, 0) + 1
            self.term_counts.append(counts)
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings.setdefault(term, []).append((position, frequency))

        self.average_length = sum(self.lengths) / len(self.lengths)

    def document_frequency(self, term: str) -> int:
        """How many chunks contain the term at least once."""
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Inverse document frequency, in the smoothed BM25 form.

            log(1 + (N - df + 0.5) / (df + 0.5))

        The ``+1`` inside the log is what keeps this non-negative. The
        textbook Robertson-Sparck Jones form without it goes **negative**
        for a term appearing in more than half the collection, so a chunk
        could be penalised for containing a query term. A test pins that
        this implementation never does.
        """
        frequency = self.document_frequency(term)
        total = len(self.chunks)
        return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))

    def score(self, query: str, position: int) -> float:
        """BM25 score of one chunk against a query."""
        counts = self.term_counts[position]
        length_ratio = self.lengths[position] / self.average_length
        denominator_base = self.k1 * (1 - self.b + self.b * length_ratio)
        total = 0.0
        for term in self.tokenize(query):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            total += self.idf(term) * (frequency * (self.k1 + 1)) / (
                frequency + denominator_base)
        return total

    def search(self, query: str, k: int = 3) -> list[tuple[object, float]]:
        """The ``k`` best chunks and their scores, highest first.

        Only chunks sharing a term with the query are scored — that is
        what the inverted index is for. Ties break on chunk order so the
        result is reproducible.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        candidates = set()
        for term in self.tokenize(query):
            candidates.update(position for position, _ in self.postings.get(term, ()))
        scored = [(position, self.score(query, position)) for position in candidates]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [(self.chunks[position], score) for position, score in scored[:k]]

    def search_within_budget(self, query: str, budget: int = 600) -> list:
        """As many top-ranked chunks as fit, in rank order.

        Day 3's rule, and for Day 3's reason: comparing a fixed ``k``
        across chunk sizes compares different amounts of context.
        """
        if budget < 1:
            raise ValueError(f"budget must be positive, got {budget}")
        packed, used = [], 0
        for chunk, _ in self.search(query, k=len(self.chunks)):
            if used + len(chunk.text) > budget:
                break
            packed.append(chunk)
            used += len(chunk.text)
        if not packed:
            ranked = self.search(query, k=1)
            return [ranked[0][0]] if ranked else []
        return packed


def _text_chunk(text: str) -> SourcedChunk:
    """A one-off chunk, for doctests and small examples."""
    return SourcedChunk(doc_id="d", text=text, start=0, end=len(text), index=0)


def saturation(frequency: int, k1: float = 1.5) -> float:
    """The ``tf`` part of BM25, without IDF or length.

    Plotted in the demo. The point is that it is concave: the first
    occurrence of a term buys most of what any number of occurrences
    will ever buy.

    >>> round(saturation(1), 3), round(saturation(100), 3)
    (1.0, 2.463)
    """
    return frequency * (k1 + 1) / (frequency + k1)


def evaluate_bm25(chunks, questions, k1: float = 1.5, b: float = 0.75,
                  k: int | None = None, budget: int | None = 600) -> dict:
    """Day 3's evaluate(), driven by a BM25 index instead of a scorer."""
    if (k is None) == (budget is None):
        raise ValueError("pass exactly one of k or budget")
    index = BM25(chunks, k1=k1, b=b)
    hits = intact = 0
    context = []
    for question in questions:
        if answer_is_intact(chunks, question.answer):
            intact += 1
        if budget is None:
            retrieved = [chunk for chunk, _ in index.search(question.query, k=k)]
        else:
            retrieved = index.search_within_budget(question.query, budget=budget)
        context.append(sum(len(chunk.text) for chunk in retrieved))
        if any(question.answer in chunk.text for chunk in retrieved):
            hits += 1
    total = len(questions)
    return {"chunks": len(chunks), "hit": hits / total, "intact": intact / total,
            "context": sum(context) / total}


# --- Day 6 -----------------------------------------------------------------

try:  # The first real dependency in this repo, and needed only from here on.
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the skip path
    SentenceTransformer = None
    EMBEDDINGS_AVAILABLE = False

#: Small, fast, and the usual first choice. 384 dimensions, 256 tokens.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def require_embeddings() -> None:
    if not EMBEDDINGS_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed. Days 1-5 run without it; "
            "install with `pip install sentence-transformers`."
        )


class DenseIndex:
    """Cosine search over sentence embeddings.

    Where BM25 matches *words*, this matches *directions in a vector
    space*. A question and its answer can share no vocabulary at all and
    still land near each other, which is the whole reason to pay for it —
    and the demo measures exactly how much that is worth, in both
    directions.

    Vectors are L2-normalized once at build time, so cosine similarity is
    a dot product and the whole search is one matrix multiply.

    **Chunks longer than the model's token limit are silently truncated.**
    Nothing raises, nothing warns: the tail of the chunk simply stops
    existing as far as the index is concerned. ``truncated_chunks``
    reports which ones, because the failure has no other symptom.
    """

    def __init__(self, chunks, model=None, model_name: str = DEFAULT_MODEL,
                 batch_size: int = 32) -> None:
        require_embeddings()
        if not chunks:
            raise ValueError("cannot build an index over no chunks")
        self.chunks = list(chunks)
        self.model = model if model is not None else SentenceTransformer(model_name)
        self.matrix = self._encode([chunk.text for chunk in self.chunks], batch_size)

    def _encode(self, texts, batch_size: int = 32) -> np.ndarray:
        vectors = np.asarray(self.model.encode(texts, batch_size=batch_size,
                                               show_progress_bar=False),
                             dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    @property
    def dimension(self) -> int:
        return self.matrix.shape[1]

    @property
    def token_limit(self) -> int:
        return int(self.model.max_seq_length)

    def token_count(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text))

    def truncated_chunks(self) -> list[int]:
        """Positions of chunks the model could not see the end of.

        Worth calling on any real index. A chunk past the token limit is
        indexed by its opening and nothing else, so a fact in its last
        paragraph is unreachable and no error ever said so.
        """
        return [position for position, chunk in enumerate(self.chunks)
                if self.token_count(chunk.text) > self.token_limit]

    def search(self, query: str, k: int = 3) -> list[tuple[object, float]]:
        """The ``k`` nearest chunks by cosine similarity, and their scores."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        scores = self.matrix @ self._encode([query])[0]
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        return [(self.chunks[i], float(scores[i])) for i in order[:k]]

    def search_within_budget(self, query: str, budget: int = 600) -> list:
        """As many nearest chunks as fit, in rank order. Day 3's rule."""
        if budget < 1:
            raise ValueError(f"budget must be positive, got {budget}")
        packed, used = [], 0
        for chunk, _ in self.search(query, k=len(self.chunks)):
            if used + len(chunk.text) > budget:
                break
            packed.append(chunk)
            used += len(chunk.text)
        if not packed:
            return [self.search(query, k=1)[0][0]]
        return packed


def vocabulary_overlap(question: Question, documents: list[Document]) -> float:
    """Fraction of the query's words that appear in the answer's sentence.

    The number that predicts whether BM25 will work. A lexical retriever
    can only match words that are present; when this is high it has
    everything it needs, and when it is low it is guessing.
    """
    document = next(d for d in documents if d.doc_id == question.doc_id)
    sentence = next(line for line in document.text.split("\n")
                    if question.answer in line)
    query_words = set(tokenize(question.query))
    sentence_words = set(tokenize(sentence))
    if not query_words:
        return 0.0
    return len(query_words & sentence_words) / len(query_words)


def evaluate_dense(index: "DenseIndex", questions, budget: int = 600) -> dict:
    """Hit rate at a context budget, for a dense index."""
    hits = 0
    context = []
    for question in questions:
        retrieved = index.search_within_budget(question.query, budget=budget)
        context.append(sum(len(chunk.text) for chunk in retrieved))
        if any(question.answer in chunk.text for chunk in retrieved):
            hits += 1
    return {"hit": hits / len(questions), "context": sum(context) / len(questions)}


#: The same seventeen answers, asked in words the handbook does not use.
#: Not harder questions - the *same* questions, rephrased the way somebody
#: who has not read the document would ask them.
PARAPHRASED_QUESTIONS = [
    Question("what is the monthly holiday entitlement", "1.75 days per month", "vacation"),
    Question("do I lose untaken time off at year end", "forfeited on 31 December", "vacation"),
    Question("who signs off a long absence", "a second-line manager", "vacation"),
    Question("what is the deadline for filing a reimbursement", "within sixty days", "expenses"),
    Question("how much can I spend on food at home", "35,000 won per day", "expenses"),
    Question("how much can I spend on food abroad", "90,000 won per day", "expenses"),
    Question("when do I need proof of purchase", "above 20,000 won", "expenses"),
    Question("can I expense a cab ride", "between 22:00 and 06:00", "expenses"),
    Question("how old can my computer get before renewal", "three year cycle", "equipment"),
    Question("when am I eligible for an extra screen", "after the probation period ends", "equipment"),
    Question("what do I do if my hardware is taken", "within twenty four hours", "equipment"),
    Question("how often may I stay home", "three days per week", "remote"),
    Question("when am I expected to be online", "10:00 to 16:00", "remote"),
    Question("is there money for a desk at home", "300,000 won per year", "remote"),
    Question("what length must my credentials be", "at least sixteen characters", "security"),
    Question("how frequently must credentials change", "every ninety days", "security"),
    Question("how long does elevated access stay open", "eight hours at a time", "security"),
]

#: A chunk built to exceed the model's token limit, with a distinctive
#: phrase at the very end that the index will never see.
def _overlong_chunk(pad_words: int = 400) -> str:
    return ("The core hours are 10:00 to 16:00. " * 2
            + "padding " * pad_words
            + " The emergency contact code is ZEBRA.")


# --- Day 7 -----------------------------------------------------------------


class VectorStore:
    """Exact nearest-neighbour search. One matrix, one multiply.

    Everything a vector database does, minus the parts that make it a
    database. Vectors are stored L2-normalized in a single contiguous
    array, so a cosine search over the whole collection is

        scores = matrix @ query

    and that single BLAS call is the reason "brute force is slow" is
    mostly a statement about implementations. The demo measures it
    against the obvious Python loop: same answers, 28 times the time.

    Exact search has one property nothing on Day 8 will keep: it returns
    the true nearest neighbours, always. ``recall_at_k`` against it is 1.0
    by construction, which is what makes it the reference every
    approximate index is measured against.

    >>> store = VectorStore(dimension=2)
    >>> store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    >>> store.search([1.0, 0.0], k=1)[0][0]
    'a'
    """

    def __init__(self, dimension: int, dtype=np.float32) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.dtype = dtype
        self.ids: list = []
        self.matrix = np.zeros((0, dimension), dtype=dtype)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids, vectors) -> "VectorStore":
        """Append vectors, normalizing as they go in.

        Normalizing once at insert is what lets the query be a plain dot
        product. Doing it per query instead would repeat the same
        divisions on every search forever.
        """
        vectors = np.asarray(vectors, dtype=self.dtype)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected dimension {self.dimension}, got {vectors.shape[1]}")
        ids = list(ids)
        if len(ids) != len(vectors):
            raise ValueError(f"{len(ids)} ids for {len(vectors)} vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.matrix = np.vstack([self.matrix, vectors / np.maximum(norms, 1e-12)])
        self.ids.extend(ids)
        return self

    def search(self, query, k: int = 5) -> list[tuple[object, float]]:
        """The ``k`` nearest ids and their cosine scores.

        ``argpartition`` finds the top ``k`` without sorting the rest,
        which is ``O(n)`` against ``O(n log n)``. At a million vectors
        that is the difference between the sort and the multiply
        dominating.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not len(self):
            return []
        query = np.asarray(query, dtype=self.dtype)
        norm = float(np.linalg.norm(query))
        scores = self.matrix @ (query / max(norm, 1e-12))
        k = min(k, len(self))
        candidates = np.argpartition(-scores, k - 1)[:k]
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in candidates]

    def search_naive(self, query, k: int = 5) -> list[tuple[object, float]]:
        """The same answers, one Python loop. Kept to measure the gap."""
        query = np.asarray(query, dtype=self.dtype)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scored = [(self.ids[i], float(self.matrix[i] @ query)) for i in range(len(self))]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def search_batch(self, queries, k: int = 5) -> list[list[tuple[object, float]]]:
        """Many queries at once: one matrix-matrix multiply instead of many
        matrix-vector ones, which is where the hardware actually wins."""
        queries = np.asarray(queries, dtype=self.dtype)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        scores = self.matrix @ (queries / np.maximum(norms, 1e-12)).T
        results = []
        for column in range(scores.shape[1]):
            column_scores = scores[:, column]
            top = np.argpartition(-column_scores, min(k, len(self)) - 1)[:k]
            top = top[np.argsort(-column_scores[top], kind="stable")]
            results.append([(self.ids[int(i)], float(column_scores[int(i)])) for i in top])
        return results

    @property
    def memory_bytes(self) -> int:
        """What the vectors cost in RAM. The number that bites first."""
        return self.matrix.nbytes


def random_store(count: int, dimension: int = 384, seed: int = 0,
                 dtype=np.float32) -> VectorStore:
    """A store of random unit vectors, for measuring scale rather than quality."""
    rng = np.random.default_rng(seed)
    store = VectorStore(dimension, dtype=dtype)
    store.add(range(count), rng.normal(size=(count, dimension)))
    return store


def recall_at_k(approximate, exact, k: int) -> float:
    """Fraction of the true top-``k`` that an approximate result found.

    The metric Day 8 exists to trade against. For exact search it is 1.0
    by construction, which is exactly why exact search is the reference.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    truth = {identifier for identifier, _ in exact[:k]}
    if not truth:
        return 1.0
    found = {identifier for identifier, _ in approximate[:k]}
    return len(found & truth) / len(truth)


def time_search(store: VectorStore, queries, k: int = 5, repeats: int = 5,
                method: str = "search") -> float:
    """Milliseconds per query, averaged, after a warm-up."""
    search = getattr(store, method)
    search(queries[0], k=k)
    start = time.perf_counter()
    for _ in range(repeats):
        for query in queries:
            search(query, k=k)
    elapsed = time.perf_counter() - start
    return elapsed / (repeats * len(queries)) * 1000


def scaling_table(sizes, dimension: int = 384, queries: int = 8,
                  seed: int = 0) -> list[dict]:
    """Query latency and memory against collection size."""
    rng = np.random.default_rng(seed + 1)
    probes = rng.normal(size=(queries, dimension))
    rows = []
    for size in sizes:
        store = random_store(size, dimension, seed=seed)
        rows.append({
            "count": size,
            "ms": time_search(store, probes, repeats=3),
            "megabytes": store.memory_bytes / 1e6,
        })
    return rows


def extrapolate(rows: list[dict], count: int, dimension: int = 384,
                bytes_per_number: int = 4) -> dict:
    """Latency and memory at a size too large to measure here.

    Linear in the number of vectors, because the multiply is. Stated as
    an extrapolation rather than a measurement, because that is what it
    is.
    """
    per_vector = rows[-1]["ms"] / rows[-1]["count"]
    return {"count": count, "ms": per_vector * count,
            "gigabytes": count * dimension * bytes_per_number / 1e9}


def scaling_exponent(rows: list[dict]) -> float:
    """Slope of log(latency) against log(count): 1.0 is linear."""
    counts = np.log([row["count"] for row in rows])
    times = np.log([row["ms"] for row in rows])
    return float(np.polyfit(counts, times, 1)[0])

# --- Day 8 -----------------------------------------------------------------


def kmeans(vectors: np.ndarray, clusters: int, iterations: int = 12,
           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm on unit vectors, which makes it spherical k-means.

    Returns ``(centroids, assignment)``. Centroids are renormalized after
    each update, so "nearest centroid" stays a cosine question and the
    same dot product serves for both clustering and search.

    Initialization is a random sample of the data rather than k-means++,
    which is a real simplification: it converges to a worse partition
    sometimes, and the demo's recall numbers carry that.

    >>> centroids, assignment = kmeans(np.eye(4), clusters=2, seed=0)
    >>> centroids.shape, assignment.shape
    ((2, 4), (4,))
    """
    if clusters < 1:
        raise ValueError(f"clusters must be positive, got {clusters}")
    count = len(vectors)
    clusters = min(clusters, count)
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(count, size=clusters, replace=False)].copy()

    assignment = np.zeros(count, dtype=int)
    for _ in range(iterations):
        assignment = np.argmax(vectors @ centroids.T, axis=1)
        moved = False
        for index in range(clusters):
            members = vectors[assignment == index]
            if not len(members):
                # An empty cluster is a real failure mode of Lloyd's: it
                # stays empty forever and the list is wasted. Reseed it on
                # the point furthest from its own centroid.
                worst = int(np.argmin(np.max(vectors @ centroids.T, axis=1)))
                centroids[index] = vectors[worst]
                moved = True
                continue
            centre = members.mean(axis=0)
            norm = np.linalg.norm(centre)
            centre = centre / norm if norm > 1e-12 else centroids[index]
            if not np.allclose(centre, centroids[index]):
                moved = True
            centroids[index] = centre
        if not moved:
            break
    assignment = np.argmax(vectors @ centroids.T, axis=1)
    return centroids, assignment


class IVFIndex:
    """Inverted file: cluster the vectors, then search only a few clusters.

    Build once with k-means. At query time, score the query against every
    **centroid** (cheap — there are a few hundred), take the ``n_probe``
    nearest, and scan only the vectors assigned to those.

    The saving is the ratio of the collection to what gets scanned. The
    cost is that the true nearest neighbour may live in a cluster that
    was not probed, and then it is simply not found — no error, a
    slightly worse answer, and `recall@k` below 1.0 is the only trace.

    ``n_probe`` is the dial between those, and at ``n_probe = n_lists``
    it degenerates to exact search with extra steps.
    """

    def __init__(self, dimension: int, n_lists: int = 16, seed: int = 0) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if n_lists < 1:
            raise ValueError(f"n_lists must be positive, got {n_lists}")
        self.dimension = dimension
        self.n_lists = n_lists
        self.seed = seed
        self.ids: list = []
        self.matrix = np.zeros((0, dimension), dtype=np.float32)
        self.centroids = np.zeros((0, dimension), dtype=np.float32)
        self.lists: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.ids)

    def fit(self, ids, vectors) -> "IVFIndex":
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected (n, {self.dimension}) vectors, got {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.matrix = vectors / np.maximum(norms, 1e-12)
        self.ids = list(ids)
        if len(self.ids) != len(self.matrix):
            raise ValueError(f"{len(self.ids)} ids for {len(self.matrix)} vectors")
        self.centroids, assignment = kmeans(self.matrix, self.n_lists, seed=self.seed)
        self.lists = [np.where(assignment == index)[0]
                      for index in range(len(self.centroids))]
        # Each list's vectors, stored contiguously. This is the whole
        # difference between an IVF that beats exact search and one that
        # loses to it: without it, every query pays to gather its
        # candidate rows out of one big matrix, and that copy costs more
        # than the full multiply it was supposed to replace.
        self.blocks = [np.ascontiguousarray(self.matrix[members])
                       for members in self.lists]
        return self

    def search(self, query, k: int = 5, n_probe: int = 1) -> list[tuple[object, float]]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if n_probe < 1:
            raise ValueError(f"n_probe must be at least 1, got {n_probe}")
        if not len(self):
            return []
        query = np.asarray(query, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)

        centroid_scores = self.centroids @ query
        probe = min(n_probe, len(self.centroids))
        chosen = np.argpartition(-centroid_scores, probe - 1)[:probe]

        found: list[tuple[float, int]] = []
        for index in chosen:
            index = int(index)
            members = self.lists[index]
            if not len(members):
                continue
            # Multiply the contiguous block directly. Writing this as
            # self.matrix[members] @ query instead is a 30% slowdown at
            # 200k vectors, and the demo measures that.
            scores = self.blocks[index] @ query
            take = min(k, len(scores))
            top = np.argpartition(-scores, take - 1)[:take]
            found.extend((float(scores[i]), int(members[i])) for i in top)
        if not found:
            return []
        found.sort(key=lambda pair: -pair[0])
        return [(self.ids[position], score) for score, position in found[:k]]

    def search_gathered(self, query, k: int = 5, n_probe: int = 1):
        """The same search, gathering candidate rows out of one big matrix.

        This is how the index was written first, and it is the obvious
        way: collect the candidate indices, then ``self.matrix[candidates]
        @ query``. It returns identical answers and it is **slower than
        exact search at every size measured**, because the fancy-index
        gather copies the rows and that copy costs more than the multiply
        it was meant to avoid. Kept so the demo can price it.
        """
        query = np.asarray(query, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        probe = min(n_probe, len(self.centroids))
        chosen = np.argpartition(-(self.centroids @ query), probe - 1)[:probe]
        candidates = np.concatenate([self.lists[int(i)] for i in chosen])
        if not len(candidates):
            return []
        scores = self.matrix[candidates] @ query
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[int(candidates[i])], float(scores[i])) for i in top]

    def vectors_scanned(self, query, n_probe: int = 1) -> int:
        """How many vectors a query actually touches. The saving, counted."""
        query = np.asarray(query, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        probe = min(n_probe, len(self.centroids))
        chosen = np.argpartition(-(self.centroids @ query), probe - 1)[:probe]
        return int(sum(len(self.lists[int(i)]) for i in chosen))


class NSWIndex:
    """A navigable small-world graph: HNSW without the layer hierarchy.

    Every vector becomes a node linked to its ``m`` nearest neighbours
    among the vectors inserted before it. A search starts at one entry
    point and walks greedily downhill, keeping a candidate set of size
    ``ef`` so it can back out of a local minimum.

    **This is a simplification and the name says so.** Real HNSW stacks
    several such graphs at decreasing density and enters at the sparsest,
    which turns the first part of the walk from O(n^(1/something)) steps
    into a few long hops. Without the hierarchy the walk starts close to
    the data and takes more steps to cross it — the structure is the same
    idea, and the constant is worse. The demo measures what this version
    achieves rather than quoting what the full one would.

    ``ef`` is this index's dial, in the same role as IVF's ``n_probe``:
    larger means more of the graph visited, higher recall, less saving.
    """

    def __init__(self, dimension: int, m: int = 8, seed: int = 0) -> None:
        if m < 1:
            raise ValueError(f"m must be at least 1, got {m}")
        self.dimension = dimension
        self.m = m
        self.seed = seed
        self.ids: list = []
        self.matrix = np.zeros((0, dimension), dtype=np.float32)
        self.neighbours: list[list[int]] = []

    def __len__(self) -> int:
        return len(self.ids)

    def fit(self, ids, vectors) -> "NSWIndex":
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected (n, {self.dimension}) vectors, got {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.matrix = vectors / np.maximum(norms, 1e-12)
        self.ids = list(ids)
        count = len(self.matrix)
        self.neighbours = [[] for _ in range(count)]
        for position in range(1, count):
            scores = self.matrix[:position] @ self.matrix[position]
            links = np.argsort(-scores, kind="stable")[:self.m]
            for other in links:
                other = int(other)
                self.neighbours[position].append(other)
                # Undirected, so early nodes stay reachable from later ones.
                if len(self.neighbours[other]) < 2 * self.m:
                    self.neighbours[other].append(position)
        return self

    def search(self, query, k: int = 5, ef: int = 16) -> list[tuple[object, float]]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if ef < 1:
            raise ValueError(f"ef must be at least 1, got {ef}")
        if not len(self):
            return []
        query = np.asarray(query, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)

        entry = 0
        visited = {entry}
        frontier = [(float(self.matrix[entry] @ query), entry)]
        best = list(frontier)
        while frontier:
            frontier.sort(reverse=True)
            score, node = frontier.pop(0)
            if len(best) >= ef and score < min(s for s, _ in best):
                break
            for neighbour in self.neighbours[node]:
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                neighbour_score = float(self.matrix[neighbour] @ query)
                frontier.append((neighbour_score, neighbour))
                best.append((neighbour_score, neighbour))
                best.sort(reverse=True)
                del best[ef:]
        self._last_visited = len(visited)
        return [(self.ids[node], score) for score, node in best[:k]]

    def vectors_scanned(self, query, ef: int = 16) -> int:
        """Nodes whose distance was computed. The graph's version of work."""
        self.search(query, k=1, ef=ef)
        return self._last_visited


def recall_curve(exact_store: VectorStore, index, queries, k: int = 10,
                 settings=(1, 2, 4, 8), parameter: str = "n_probe") -> list[dict]:
    """Recall and work against the index's one dial.

    The curve every approximate index is really described by: a knob
    trading the answer's correctness for the time it took.
    """
    rows = []
    for setting in settings:
        recalls, scanned, elapsed = [], [], 0.0
        for query in queries:
            truth = exact_store.search(query, k=k)
            start = time.perf_counter()
            found = index.search(query, k=k, **{parameter: setting})
            elapsed += time.perf_counter() - start
            recalls.append(recall_at_k(found, truth, k))
            scanned.append(index.vectors_scanned(query, **{parameter: setting}))
        rows.append({
            parameter: setting,
            "recall": sum(recalls) / len(recalls),
            "scanned": sum(scanned) / len(scanned),
            "ms": elapsed / len(queries) * 1000,
        })
    return rows


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    DIMENSION = 64
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50_000, DIMENSION)).astype(np.float32)
    queries = rng.normal(size=(20, DIMENSION))
    exact = VectorStore(DIMENSION)
    exact.add(range(len(vectors)), vectors)

    print("=" * 72)
    print("1. the dial: recall against work")
    print("=" * 72)
    lists = 128
    ivf = IVFIndex(DIMENSION, n_lists=lists, seed=0).fit(range(len(vectors)), vectors)
    print(f"  IVF over {len(vectors):,} vectors, {lists} lists "
          f"({min(len(l) for l in ivf.lists)}-{max(len(l) for l in ivf.lists)} each)")
    print(f"  {'n_probe':>9}{'recall@10':>11}{'scanned':>10}{'of total':>10}")
    for row in recall_curve(exact, ivf, queries, k=10,
                            settings=(1, 2, 4, 8, 16, 32, 64, 128)):
        print(f"  {row['n_probe']:>9}{row['recall']:>11.2f}{row['scanned']:>10.0f}"
              f"{row['scanned'] / len(vectors):>10.0%}")
    print()
    print("  at n_probe = n_lists it scans everything and recall reaches 1.00:")
    print("  that is exact search with extra steps, and it is the right")
    print("  sanity check - the approximation has a setting where it is not")
    print("  approximate.")
    print()
    print("  everywhere else the missing recall is silent. The true neighbour")
    print("  was in a cluster nobody probed, so it simply is not in the")
    print("  results. No error, no warning, a slightly worse answer.")

    print()
    print("=" * 72)
    print("2. the graph, and what dropping the hierarchy costs")
    print("=" * 72)
    small = vectors[:8_000]
    small_exact = VectorStore(DIMENSION)
    small_exact.add(range(len(small)), small)
    start = time.perf_counter()
    nsw = NSWIndex(DIMENSION, m=8, seed=0).fit(range(len(small)), small)
    build = time.perf_counter() - start
    print(f"  NSW over {len(small):,} vectors, m=8, built in {build:.1f}s")
    print(f"  {'ef':>9}{'recall@10':>11}{'scanned':>10}{'ms':>9}")
    for row in recall_curve(small_exact, nsw, queries[:8], k=10,
                            settings=(8, 16, 32, 64), parameter="ef"):
        print(f"  {row['ef']:>9}{row['recall']:>11.2f}{row['scanned']:>10.0f}"
              f"{row['ms']:>9.2f}")
    print()
    print("  this is the honest state of a hierarchy-free NSW written in")
    print("  Python: the recall is poor for the work done, and the walk is")
    print("  a Python loop over one neighbour at a time. Real HNSW enters")
    print("  at a sparse top layer and drops down, which replaces most of")
    print("  that walk with a few long hops. The idea is here; the constant")
    print("  is not, and quoting published HNSW numbers next to this table")
    print("  would be dishonest.")

    print()
    print("=" * 72)
    print("3. the measurement that nearly buried the day")
    print("=" * 72)
    print("  IVF does less arithmetic. Does it take less time?")
    print()
    print(f"  {'N':>9}{'exact':>9}{'IVF gathered':>15}{'IVF blocked':>14}{'recall':>9}")
    for count in (20_000, 50_000, 100_000, 200_000):
        data = rng.normal(size=(count, DIMENSION)).astype(np.float32)
        store = VectorStore(DIMENSION)
        store.add(range(count), data)
        n_lists = int(math.sqrt(count))
        index = IVFIndex(DIMENSION, n_lists=n_lists, seed=0).fit(range(count), data)
        probe = max(1, n_lists // 8)
        probes = rng.normal(size=(8, DIMENSION))

        store.search(probes[0], k=10)
        index.search(probes[0], k=10, n_probe=probe)
        index.search_gathered(probes[0], k=10, n_probe=probe)

        def timed(call):
            start = time.perf_counter()
            for query in probes:
                call(query)
            return (time.perf_counter() - start) / len(probes) * 1000

        exact_ms = timed(lambda q: store.search(q, k=10))
        gathered_ms = timed(lambda q: index.search_gathered(q, k=10, n_probe=probe))
        blocked_ms = timed(lambda q: index.search(q, k=10, n_probe=probe))
        recall = sum(recall_at_k(index.search(q, k=10, n_probe=probe),
                                 store.search(q, k=10), 10) for q in probes) / len(probes)
        print(f"  {count:>9,}{exact_ms:>8.2f}m"
              f"{gathered_ms:>13.2f}m{blocked_ms:>13.2f}m{recall:>9.2f}")
    print()
    print("  the middle column is where this day spent most of its time. That")
    print("  IVF scans an eighth of the collection and is SLOWER than scanning")
    print("  all of it, at every size. The algorithm is not wrong - it really")
    print("  does do an eighth of the arithmetic:")
    print()
    big = rng.normal(size=(200_000, DIMENSION)).astype(np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    probe_vector = rng.normal(size=DIMENSION).astype(np.float32)
    probe_vector /= np.linalg.norm(probe_vector)
    subset = np.sort(rng.choice(len(big), size=len(big) // 8, replace=False))
    gathered = np.ascontiguousarray(big[subset])

    def bench(call, repeats=20):
        call()
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        return (time.perf_counter() - start) / repeats * 1000

    full = bench(lambda: big @ probe_vector)
    copy = bench(lambda: big[subset])
    part = bench(lambda: gathered @ probe_vector)
    print(f"    full multiply, all 200,000 rows      {full:7.2f} ms")
    print(f"    multiply over 1/8, already contiguous{part:7.2f} ms  "
          f"({full / part:.1f}x cheaper, as promised)")
    print(f"    gathering that 1/8 out of the matrix {copy:7.2f} ms  "
          f"({copy / full:.0%} of the full multiply)")
    print()
    print("  there it is. The copy costs more than the multiply it avoids.")
    print("  Storing each list's vectors contiguously at build time removes")
    print("  the copy entirely, and that single change is the difference")
    print("  between the middle column and the right one.")
    print()
    print("  it is also why every serious ANN library is written in C++ with")
    print("  its own memory layout. The asymptotic win is real and the")
    print("  constant factor eats it unless the data sits where the")
    print("  arithmetic wants it.")

    print()
    print("=" * 72)
    print("what an approximate index actually costs")
    print("=" * 72)
    print("  - recall below 1.0, silently. Day 7's exact search could not")
    print("    return a wrong answer; this can, and only a comparison")
    print("    against exact search reveals it.")
    print("  - a build step. k-means over the collection, redone whenever")
    print("    the collection changes enough.")
    print("  - a parameter nobody can set for you. n_probe and ef are")
    print("    recall-versus-latency dials whose right value depends on")
    print("    the data, the query distribution and what you can tolerate.")
    print()
    print("  and the thing it buys - a constant-factor speedup that only")
    print("  appears once the memory layout cooperates - does not arrive at")
    print("  all until the collection is large enough that Day 7's one BLAS")
    print("  call has stopped being fast enough.")
