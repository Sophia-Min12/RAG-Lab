"""Day 12 — Building an evaluation set without hand-labelling everything.

Day 11 ended with seventeen questions being too few to separate the
systems. The obvious fix is to generate more.

Both obvious generators — using each sentence as its own query, and
removing a fact from a sentence to make a cloze — produce sets on which
**every retriever scores 1.000**. Their resolving power is zero: they
cannot tell any two systems apart, so they cannot be used to choose one.
Generating questions is easy; generating an evaluation set with an
opinion is the hard part, and vocabulary overlap predicts which you got.

What does work is pooling: keep the real queries, and let several systems
decide which chunks are worth judging. At depth 10 the pool holds the
answer to every query while judging 14 chunks of 26.

Its cost is a bias with a direction. Held out of the pool, BM25 loses
nothing and dense loses 29 points — because BM25 finds nothing the others
miss, while dense finds exactly the chunks no lexical system surfaces.
**Pooling penalises a system in proportion to how different it is from
the systems that built the pool.**

This day also contains a measurement error of mine: the first version of
``agreement`` counted a tie as agreement, and cheerfully reported that
the degenerate sets agreed with the hand-written one 83% of the time.
"""

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


# --- Day 9 -----------------------------------------------------------------


def min_max_normalize(scores: list[float]) -> list[float]:
    """Rescale to [0, 1]. The usual first move, and the fragile one.

    Both halves of a hybrid produce numbers, and they are not on the same
    scale: BM25 is unbounded and depends on IDF and document length,
    cosine sits in [-1, 1] and is usually crowded into [0, 0.9]. Adding
    them directly means whichever happens to be larger wins.

    Normalizing fixes the units and introduces a different problem, which
    the demo measures: the mapping depends on **which results came back**.
    The same chunk with the same score normalizes differently depending on
    what else was retrieved, so a score here is not a property of the
    chunk at all.

    >>> min_max_normalize([1.0, 3.0, 2.0])
    [0.0, 1.0, 0.5]
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def score_fusion(sparse_results, dense_results, alpha: float = 0.5,
                 k: int = 5) -> list[tuple[object, float]]:
    """Normalize both score lists, then take a weighted sum.

    ``alpha`` is the dense weight: 0 is pure BM25, 1 is pure embeddings.
    A chunk found by only one retriever contributes its score there and
    zero for the other, which is the standard treatment and is also an
    assumption - it says "absent" and "worst in the list" are the same
    thing, and they are not.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    combined: dict[object, float] = {}
    for results, weight in ((sparse_results, 1.0 - alpha), (dense_results, alpha)):
        identifiers = [identifier for identifier, _ in results]
        normalized = min_max_normalize([score for _, score in results])
        for identifier, score in zip(identifiers, normalized):
            combined[identifier] = combined.get(identifier, 0.0) + weight * score
    ranked = sorted(combined.items(), key=lambda pair: -pair[1])
    return ranked[:k]


def reciprocal_rank_fusion(result_lists, k: int = 5,
                           constant: int = 60) -> list[tuple[object, float]]:
    """Combine by rank alone: sum of ``1 / (constant + rank)``.

    RRF never looks at a score, which removes the entire normalization
    problem above - there is nothing to put on a common scale, because
    rank 1 is rank 1 in any units.

    The ``constant`` (60 in the original paper) flattens the top of the
    curve: without it, rank 1 would be worth twice rank 2, and one
    retriever's confident mistake would dominate. With 60, rank 1 and
    rank 2 differ by under 2%, so agreement between lists matters more
    than position within one.

    >>> reciprocal_rank_fusion([[("a", 9.0), ("b", 8.0)], [("b", 0.5), ("a", 0.4)]], k=2)
    [('a', 0.032266458495966696), ('b', 0.032266458495966696)]
    """
    if constant < 0:
        raise ValueError(f"constant must not be negative, got {constant}")
    combined: dict[object, float] = {}
    for results in result_lists:
        for rank, (identifier, _) in enumerate(results, start=1):
            combined[identifier] = combined.get(identifier, 0.0) + 1.0 / (constant + rank)
    ranked = sorted(combined.items(), key=lambda pair: (-pair[1], str(pair[0])))
    return ranked[:k]


class HybridRetriever:
    """BM25 and a dense index over the same chunks, combined either way."""

    def __init__(self, chunks, dense_index, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.sparse = BM25(self.chunks, k1=k1, b=b)
        self.dense = dense_index
        self.position = {id(chunk): index for index, chunk in enumerate(self.chunks)}
        #: Query results are cached by (query, depth). Sweeping a fusion
        #: parameter re-runs both retrievers on the same queries dozens of
        #: times, and re-encoding a question that has not changed is the
        #: kind of waste that makes an experiment too slow to run.
        self._cache: dict[tuple[str, int, str], list] = {}

    def _sparse(self, query, depth):
        key = (query, depth, "sparse")
        if key not in self._cache:
            self._cache[key] = [(self.position[id(chunk)], score)
                                for chunk, score in self.sparse.search(query, k=depth)]
        return self._cache[key]

    def _dense(self, query, depth):
        key = (query, depth, "dense")
        if key not in self._cache:
            self._cache[key] = [(self.position[id(chunk)], score)
                                for chunk, score in self.dense.search(query, k=depth)]
        return self._cache[key]

    def search(self, query: str, k: int = 5, method: str = "rrf",
               alpha: float = 0.5, depth: int = 20, constant: int = 60):
        """``method`` is 'rrf', 'score', 'sparse' or 'dense'.

        ``depth`` is how many results each half contributes before fusing.
        It matters: a chunk absent from one list is treated as absent
        rather than as ranked last, so a deeper pool changes the answer.
        """
        if method == "sparse":
            chosen = self._sparse(query, k)
        elif method == "dense":
            chosen = self._dense(query, k)
        elif method == "score":
            chosen = score_fusion(self._sparse(query, depth),
                                  self._dense(query, depth), alpha=alpha, k=k)
        elif method == "rrf":
            chosen = reciprocal_rank_fusion(
                [self._sparse(query, depth), self._dense(query, depth)],
                k=k, constant=constant)
        else:
            raise ValueError(f"unknown method {method!r}")
        return [(self.chunks[position], score) for position, score in chosen]


def evaluate_hybrid(retriever: HybridRetriever, questions, k: int = 3,
                    **kwargs) -> float:
    """Hit@k for one retrieval method."""
    hits = 0
    for question in questions:
        found = retriever.search(question.query, k=k, **kwargs)
        hits += any(question.answer in chunk.text for chunk, _ in found)
    return hits / len(questions)


def complementarity(retriever: "HybridRetriever", questions, k: int = 3) -> dict:
    """What each retriever finds that the other does not.

    The number to look at **before** choosing a fusion method. Fusion can
    only reach the union of what the two halves find; if one contributes
    nothing the other missed, no combination rule will help and some will
    hurt. That is measurable without trying a single fusion.
    """
    sparse_hits, dense_hits = set(), set()
    for index, question in enumerate(questions):
        for method, bucket in (("sparse", sparse_hits), ("dense", dense_hits)):
            found = retriever.search(question.query, k=k, method=method)
            if any(question.answer in chunk.text for chunk, _ in found):
                bucket.add(index)
    return {
        "sparse": len(sparse_hits),
        "dense": len(dense_hits),
        "union": len(sparse_hits | dense_hits),
        "sparse_only": len(sparse_hits - dense_hits),
        "dense_only": len(dense_hits - sparse_hits),
        "total": len(questions),
    }


# --- Day 10 ----------------------------------------------------------------

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = EMBEDDINGS_AVAILABLE
except ImportError:  # pragma: no cover - exercised by the skip path
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

#: Trained on MS MARCO, which is web search queries against passages.
#: What it learned there is the whole subject of this day.
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """A cross-encoder: reads the query and the chunk **together**.

    Day 6's bi-encoder embeds the query and the chunk separately and
    compares the two vectors. Everything the chunk's vector knows about
    the query is nothing, because it was computed before the query
    existed — which is exactly why it can be precomputed and stored.

    A cross-encoder gives up that trade. It runs one forward pass per
    ``(query, chunk)`` pair, with attention running across both, so it can
    represent "this sentence answers that question" rather than "these two
    texts are about similar things". Nothing can be precomputed, so the
    cost is ``len(candidates)`` forward passes **per query**.

    That is why it reranks a shortlist instead of searching. The first
    stage is cheap and decides what is reachable; this reorders it.
    """

    def __init__(self, model=None, model_name: str = DEFAULT_CROSS_ENCODER) -> None:
        if not CROSS_ENCODER_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed, so no cross-encoder.")
        self.model = model if model is not None else CrossEncoder(model_name)
        self._cache: dict[tuple[str, str], float] = {}

    def score(self, query: str, texts) -> list[float]:
        """Relevance of each text to the query. Unbounded, higher is better.

        These are logits rather than probabilities, so they are comparable
        within one query and meaningless across queries - which rules out
        the min-max fusion of Day 9 unless it is done per query.
        """
        texts = list(texts)
        missing = [t for t in texts if (query, t) not in self._cache]
        if missing:
            scores = self.model.predict([(query, t) for t in missing],
                                        show_progress_bar=False)
            for text, score in zip(missing, scores):
                self._cache[(query, text)] = float(score)
        return [self._cache[(query, t)] for t in texts]

    def rerank(self, query: str, candidates, k: int = 3):
        """Reorder ``candidates`` by cross-encoder score, keep the best ``k``.

        **It can only reorder what it was given.** If the first stage did
        not retrieve the answer, no amount of reranking finds it, and the
        demo measures that ceiling before measuring the gain.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        candidates = list(candidates)
        if not candidates:
            return []
        scores = self.score(query, [chunk.text for chunk in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [(candidates[i], scores[i]) for i in order[:k]]


def first_stage_recall(retriever: "HybridRetriever", questions, depth: int,
                       method: str = "sparse", **kwargs) -> float:
    """Fraction of answers present anywhere in the first ``depth`` results.

    The ceiling on everything a reranker can do. Day 3 measured the same
    shape of quantity for chunking and called it the intact ceiling; this
    is its retrieval equivalent.
    """
    hits = 0
    for question in questions:
        found = retriever.search(question.query, k=depth, method=method, **kwargs)
        hits += any(question.answer in chunk.text for chunk, _ in found)
    return hits / len(questions)


def evaluate_reranked(retriever: "HybridRetriever", reranker: "Reranker",
                      questions, depth: int = 20, k: int = 3,
                      method: str = "sparse", **kwargs) -> float:
    """Hit@k after retrieving ``depth`` candidates and reranking them."""
    hits = 0
    for question in questions:
        candidates = [chunk for chunk, _ in
                      retriever.search(question.query, k=depth, method=method, **kwargs)]
        best = reranker.rerank(question.query, candidates, k=k)
        hits += any(question.answer in chunk.text for chunk, _ in best)
    return hits / len(questions)


# --- Day 11 ----------------------------------------------------------------


def relevance(results, question: "Question") -> list[int]:
    """1 where a retrieved chunk contains the answer, 0 elsewhere.

    Binary judgements, because the corpus supports nothing better: a
    chunk either holds the answer string or it does not. Day 12 builds a
    graded set, and nDCG only earns its keep there.
    """
    return [1 if question.answer in chunk.text else 0 for chunk, _ in results]


def hit_at_k(judgements: list[int], k: int) -> float:
    """1.0 if anything relevant is in the top ``k``.

    The metric every number in Level 2 was quoted in. It is binary per
    query, so it cannot distinguish "the answer was first" from "the
    answer was third", and with seventeen questions it moves in jumps of
    6 percentage points.

    >>> hit_at_k([0, 1, 0], 3), hit_at_k([0, 1, 0], 1)
    (1.0, 0.0)
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return 1.0 if any(judgements[:k]) else 0.0


def recall_at_k_judged(judgements: list[int], k: int, total_relevant: int) -> float:
    """Fraction of all relevant chunks that appear in the top ``k``.

    Needs ``total_relevant`` — the number that exists in the collection,
    not the number retrieved. Getting that wrong is the commonest way to
    report a recall that cannot be compared with anyone else's.

    >>> recall_at_k_judged([1, 0, 1], 3, total_relevant=4)
    0.5
    """
    if total_relevant < 0:
        raise ValueError("total_relevant must not be negative")
    if not total_relevant:
        return 1.0
    return sum(judgements[:k]) / total_relevant


def precision_at_k(judgements: list[int], k: int) -> float:
    """Fraction of the top ``k`` that is relevant.

    The metric that punishes padding a context window. Retrieving ten
    chunks to find one answer is recall 1.0 and precision 0.1, and Day 14
    will show what the second number costs a generator.

    >>> precision_at_k([1, 0, 0, 1], 4)
    0.5
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    window = judgements[:k]
    return sum(window) / k


def reciprocal_rank(judgements: list[int]) -> float:
    """1 / rank of the first relevant result, or 0 if there is none.

    Answers "how far down did the user have to read", which is the
    question hit@k refuses to answer. Rank 1 scores 1.0, rank 2 scores
    0.5, rank 10 scores 0.1 — it cares about the top of the list and
    almost nothing else.

    >>> reciprocal_rank([0, 1, 1]), reciprocal_rank([0, 0])
    (0.5, 0.0)
    """
    for rank, judged in enumerate(judgements, start=1):
        if judged:
            return 1.0 / rank
    return 0.0


def average_precision(judgements: list[int], total_relevant: int | None = None) -> float:
    """Mean of the precisions at each rank where something relevant appears.

    The one metric here that rewards finding *several* relevant chunks
    early, which matters when an answer is spread across two of them.

    >>> round(average_precision([1, 0, 1]), 4)
    0.8333
    """
    total_relevant = sum(judgements) if total_relevant is None else total_relevant
    if not total_relevant:
        return 0.0
    found = 0
    total = 0.0
    for rank, judged in enumerate(judgements, start=1):
        if judged:
            found += 1
            total += found / rank
    return total / total_relevant


def dcg_at_k(judgements: list[int], k: int) -> float:
    """Discounted cumulative gain: relevance divided by log of the rank.

    The discount is ``1 / log2(rank + 1)``, so rank 1 keeps all of its
    gain, rank 2 keeps 63%, rank 10 keeps 29%. Unlike reciprocal rank it
    accumulates over every relevant result rather than stopping at the
    first.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return sum(judged / math.log2(rank + 1)
               for rank, judged in enumerate(judgements[:k], start=1))


def ndcg_at_k(judgements: list[int], k: int,
              total_relevant: int | None = None) -> float:
    """DCG over the best DCG achievable, so 1.0 means perfectly ordered.

    With **binary** judgements and one relevant chunk per query — which is
    this corpus — both this and ``reciprocal_rank`` are functions of a
    single number, the rank of the answer. They therefore agree on every
    individual query, and it is tempting to conclude one of them is
    redundant.

    They are not, and the demo shows why: averaging a different non-linear
    function of the same ranks can put two systems in the **opposite
    order**. ``1/rank`` punishes a rank-15 result down to 0.067 while
    ``1/log2(rank+1)`` leaves it at 0.25, and it rewards rank 2 with 0.5
    against 0.63. On this corpus those two effects combine to make MRR
    prefer RRF and nDCG prefer score fusion — over the same result lists.

    >>> round(ndcg_at_k([0, 1], 2), 4)
    0.6309
    """
    total_relevant = sum(judgements) if total_relevant is None else total_relevant
    ideal = dcg_at_k([1] * min(total_relevant, k), k)
    if ideal <= 0:
        return 0.0
    return dcg_at_k(judgements, k) / ideal


def evaluate_all(retriever, questions, k: int = 3, depth: int | None = None,
                 reranker=None, **kwargs) -> dict:
    """Every metric above, over one question set and one configuration."""
    depth = depth or k
    scores = {"hit": [], "recall": [], "precision": [], "mrr": [], "map": [], "ndcg": []}
    for question in questions:
        results = retriever.search(question.query, k=depth, **kwargs)
        if reranker is not None:
            results = reranker.rerank(question.query,
                                      [chunk for chunk, _ in results], k=depth)
        judgements = relevance(results, question)
        total = 1  # exactly one chunk holds each answer in this corpus
        scores["hit"].append(hit_at_k(judgements, k))
        scores["recall"].append(recall_at_k_judged(judgements, k, total))
        scores["precision"].append(precision_at_k(judgements, k))
        scores["mrr"].append(reciprocal_rank(judgements))
        scores["map"].append(average_precision(judgements, total))
        scores["ndcg"].append(ndcg_at_k(judgements, k, total))
    return {name: sum(values) / len(values) for name, values in scores.items()}


def confidence_interval(successes: int, total: int, z: float = 1.96) -> tuple:
    """A normal-approximation interval for a proportion.

    Seventeen questions is a small sample and every percentage in this
    repo has been quoted without one. At 16/17 the interval runs from
    83% to 105% — which is nonsense at the top end and exactly the point:
    the approximation breaks near the boundary, and so does the reader's
    intuition that 94% and 100% are different numbers.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    spread = z * math.sqrt(proportion * (1 - proportion) / total)
    return proportion - spread, proportion + spread


def answer_ranks(retriever, questions, reranker=None, depth: int | None = None,
                 **kwargs) -> list[int | None]:
    """Rank of the answer for each question, or None if it never appears.

    Everything in this day is a different average of this one list, which
    is the clearest way to see why the metrics disagree.
    """
    depth = depth or len(retriever.chunks)
    ranks = []
    for question in questions:
        results = retriever.search(question.query, k=depth, **kwargs)
        if reranker is not None:
            results = reranker.rerank(question.query,
                                      [chunk for chunk, _ in results], k=depth)
        judgements = relevance(results, question)
        ranks.append(next((i + 1 for i, judged in enumerate(judgements) if judged), None))
    return ranks


# --- Day 12 ----------------------------------------------------------------

#: Spans worth asking about: amounts, times, durations, dates. Picked by
#: shape rather than by meaning, which is the method's main limitation.
FACT_PATTERN = re.compile(
    r"\b(?:\d[\d,]*(?:\.\d+)?\s*(?:won|days?|hours?|months?|years?|characters?)"
    r"|\d{1,2}:\d{2}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)"
    r"|(?:three|four|five|six|seven|eight|nine|ten|sixteen|sixty|ninety)\s+"
    r"(?:days?|hours?|months?|years?|characters?))\b",
    re.IGNORECASE,
)


def extract_facts(sentence: str) -> list[str]:
    """Spans in a sentence that could be the answer to something.

    >>> extract_facts("Meals are reimbursed up to 35,000 won per day.")
    ['35,000 won']
    """
    return [match.group(0) for match in FACT_PATTERN.finditer(sentence)]


def generate_known_item_questions(documents, limit: int | None = None) -> list[Question]:
    """Use each sentence as its own query. The cheapest possible set.

    It is also nearly worthless as an evaluation, and the demo shows why:
    a query identical to the target shares 100% of its vocabulary with it,
    so every retriever scores near perfectly and the set cannot separate
    them. Included because it is what "generate an eval set" produces if
    nobody asks what the set is for.
    """
    questions = []
    for document in documents:
        for sentence in document.text.split("\n"):
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            questions.append(Question(query=sentence, answer=sentence,
                                      doc_id=document.doc_id))
    return questions[:limit] if limit else questions


def generate_cloze_questions(documents, limit: int | None = None) -> list[Question]:
    """Remove a fact from its sentence and ask for it back.

    The sentence minus its answer becomes the query; the removed span
    becomes the answer. No model, no labelling, and the answers are
    literal spans so the judgements stay objective.

    The catch is measurable and the demo measures it: the query is still
    the document's own words, so this set is **biased towards lexical
    retrieval**. It tests whether a retriever can find a sentence it has
    most of, which is not the task users bring.
    """
    questions = []
    for document in documents:
        for sentence in document.text.split("\n"):
            sentence = sentence.strip()
            facts = extract_facts(sentence)
            if not facts:
                continue
            fact = facts[0]
            query = sentence.replace(fact, "____", 1).rstrip(".")
            questions.append(Question(query=query, answer=fact,
                                      doc_id=document.doc_id))
    return questions[:limit] if limit else questions


def pooled_judgements(retrievers: dict, query: str, depth: int = 5) -> dict:
    """TREC-style pooling: everything any retriever ranked highly.

    The standard way to build judgements without reading the whole
    collection - take the top ``depth`` from each system, judge only the
    union, assume everything outside the pool is irrelevant.

    That assumption is the method's known bias and it has a direction: a
    system **not in the pool** is penalised for finding something true
    that nobody else surfaced, because the pool never judged it. The demo
    measures how large that is here.

    Returns ``{position: contributing systems}``.
    """
    pool: dict = {}
    for name, search in retrievers.items():
        for chunk, _ in search(query, depth):
            pool.setdefault(id(chunk), (chunk, set()))[1].add(name)
    return {key: value for key, value in pool.items()}


def pool_coverage(pool: dict, question: Question) -> bool:
    """Did the pool contain the answer at all?"""
    return any(question.answer in chunk.text for chunk, _ in pool.values())


def vocabulary_overlap_of_set(questions, documents) -> float:
    """Mean fraction of query words that appear in the answer's sentence.

    Day 6 used this to explain why BM25 did well on the original
    questions. Here it is a property of an evaluation SET, and it
    predicts which retriever that set will flatter.
    """
    if not questions:
        return 0.0
    values = []
    for question in questions:
        try:
            values.append(vocabulary_overlap(question, documents))
        except StopIteration:
            continue
    return sum(values) / len(values) if values else 0.0


def resolving_power(scores: dict, tolerance: float = 1e-9) -> float:
    """Fraction of system pairs an evaluation set can tell apart at all.

    Ask this **before** asking whether two evaluations agree. A set on
    which every system scores identically has a resolving power of 0 and
    an opinion about nothing, and it will still look like it agrees with
    anything if ties are counted as agreement - which is exactly the
    mistake the first version of ``agreement`` below made.

    >>> resolving_power({"a": 1.0, "b": 1.0})
    0.0
    >>> resolving_power({"a": 1.0, "b": 0.5})
    1.0
    """
    names = sorted(scores)
    pairs = [(a, b) for index, a in enumerate(names) for b in names[index + 1:]]
    if not pairs:
        return 0.0
    separated = sum(1 for a, b in pairs if abs(scores[a] - scores[b]) > tolerance)
    return separated / len(pairs)


def agreement(first: dict, second: dict, tolerance: float = 1e-9) -> dict:
    """How two evaluations compare, over the pairs both can order.

    Returns ``{"agreed", "disagreed", "unresolved", "rate"}``. ``rate`` is
    computed over resolved pairs only.

    The first version of this returned a single fraction and counted a
    tie as agreement, which made a degenerate set - every system at
    1.000, ordering nothing - report 83% agreement with the hand-written
    one. A set with no opinion cannot agree with anything, and a metric
    that says otherwise is worse than no metric.
    """
    names = sorted(set(first) & set(second))
    pairs = [(a, b) for index, a in enumerate(names) for b in names[index + 1:]]
    agreed = disagreed = unresolved = 0
    for a, b in pairs:
        left, right = first[a] - first[b], second[a] - second[b]
        if abs(left) <= tolerance or abs(right) <= tolerance:
            unresolved += 1
        elif (left > 0) == (right > 0):
            agreed += 1
        else:
            disagreed += 1
    resolved = agreed + disagreed
    return {"agreed": agreed, "disagreed": disagreed, "unresolved": unresolved,
            "rate": agreed / resolved if resolved else float("nan")}


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers is not installed, so this day cannot run.")
        raise SystemExit(0)

    documents = _handbook()
    chunks = build_index(documents, 80, "sentence")
    model = SentenceTransformer(DEFAULT_MODEL)
    retriever = HybridRetriever(chunks, DenseIndex(chunks, model=model))

    print("=" * 72)
    print("1. generating questions is easy")
    print("=" * 72)
    known = generate_known_item_questions(documents)
    cloze = generate_cloze_questions(documents)
    print(f"  {len(known)} known-item and {len(cloze)} cloze questions, from the")
    print(f"  same documents, with no labelling and no model.")
    print()
    print("  cloze takes a fact out of a sentence and asks for it back:")
    for question in cloze[:3]:
        print(f"    Q {question.query[:60]!r}")
        print(f"    A {question.answer!r}")

    print()
    print("=" * 72)
    print("2. and the questions are useless")
    print("=" * 72)
    methods = {"BM25": {"method": "sparse"}, "dense": {"method": "dense"},
               "score fusion": {"method": "score", "alpha": 0.5},
               "RRF": {"method": "rrf"}}
    sets = {"hand-written": list(QUESTIONS),
            "paraphrased": list(PARAPHRASED_QUESTIONS),
            "known-item": known,
            "cloze": cloze}
    scores = {name: {method: evaluate_all(retriever, questions, k=3, depth=3,
                                          **kwargs)["mrr"]
                     for method, kwargs in methods.items()}
              for name, questions in sets.items()}

    print(f"  MRR by evaluation set:")
    print(f"  {'set':<16}" + "".join(f"{name:>14}" for name in methods))
    for name in sets:
        print(f"  {name:<16}" + "".join(f"{scores[name][m]:>14.3f}" for m in methods))
    print()
    print(f"  {'set':<16}{'n':>5}{'vocab overlap':>15}{'resolving power':>18}")
    for name, questions in sets.items():
        print(f"  {name:<16}{len(questions):>5}"
              f"{vocabulary_overlap_of_set(questions, documents):>15.0%}"
              f"{resolving_power(scores[name]):>18.0%}")
    print()
    print("  both generated sets score every system at 1.000. Their resolving")
    print("  power is zero: they cannot tell any two retrievers apart, so they")
    print("  cannot be used to choose one. Generating questions is easy;")
    print("  generating an evaluation set with an opinion is the hard part.")
    print()
    print("  the vocabulary overlap column predicts it. A cloze query is the")
    print("  document's own sentence with one span removed - 83% of its words")
    print("  are in the target - so finding the target is not a retrieval")
    print("  problem. The hand-written questions sit at 45% and separate five")
    print("  of six system pairs.")

    print()
    print("  a measurement error of mine, worth keeping: the first version of")
    print("  agreement() scored a tie as agreement, and reported that these")
    print("  degenerate sets agreed with the hand-written one 83% of the time.")
    for name in ("paraphrased", "known-item", "cloze"):
        verdict = agreement(scores["hand-written"], scores[name])
        resolved = verdict["agreed"] + verdict["disagreed"]
        summary = (f"{verdict['agreed']}/{resolved} of resolved pairs"
                   if resolved else "no pair resolved by both")
        print(f"    vs {name:<14} {summary}")
    print("  a set with no opinion cannot agree with anything.")

    print()
    print("=" * 72)
    print("3. pooling: judge a few chunks instead of all of them")
    print("=" * 72)
    systems = {
        "BM25": lambda query, depth: retriever.search(query, k=depth, method="sparse"),
        "dense": lambda query, depth: retriever.search(query, k=depth, method="dense"),
        "RRF": lambda query, depth: retriever.search(query, k=depth, method="rrf"),
    }
    everything = list(QUESTIONS) + list(PARAPHRASED_QUESTIONS)
    print(f"  {len(everything)} real queries, top-n from each of three systems:")
    print(f"  {'depth':>7}{'chunks judged':>16}{'answer in pool':>17}")
    for depth in (1, 3, 5, 10):
        sizes, covered = [], 0
        for question in everything:
            pool = pooled_judgements(systems, question.query, depth=depth)
            sizes.append(len(pool))
            covered += pool_coverage(pool, question)
        print(f"  {depth:>7}{sum(sizes) / len(sizes):>10.1f} of {len(chunks)}"
              f"{covered / len(everything):>17.0%}")
    print()
    print("  at depth 10 the pool holds the answer for every query while")
    print("  judging 14 chunks of 26. On a real collection that ratio is the")
    print("  whole point: judge hundreds instead of millions.")

    print()
    print("=" * 72)
    print("4. what pooling costs, and who pays")
    print("=" * 72)
    print("  the assumption is that anything outside the pool is irrelevant.")
    print("  so hold one system out of the pool and score it anyway:")
    print()
    print(f"  {'system':<8}{'scored by the pool':>21}{'scored by truth':>18}{'penalty':>10}")
    for excluded in ("BM25", "dense"):
        contributors = {name: search for name, search in systems.items()
                        if name != excluded}
        by_pool = by_truth = 0
        for question in everything:
            pool = pooled_judgements(contributors, question.query, depth=5)
            judged = {id(chunk) for chunk, _ in pool.values()
                      if question.answer in chunk.text}
            results = systems[excluded](question.query, 3)
            if any(question.answer in chunk.text and id(chunk) in judged
                   for chunk, _ in results):
                by_pool += 1
            if any(question.answer in chunk.text for chunk, _ in results):
                by_truth += 1
        total = len(everything)
        print(f"  {excluded:<8}{by_pool / total:>21.0%}{by_truth / total:>18.0%}"
              f"{(by_truth - by_pool) / total:>10.0%}")
    print()
    print("  BM25 pays nothing. Dense pays 29 points.")
    print()
    print("  and the asymmetry is the finding, not the numbers. BM25 finds")
    print("  nothing the other systems miss, so a pool built without it")
    print("  already contains everything it would have contributed. Dense")
    print("  finds chunks no lexical system surfaces - the whole reason it")
    print("  exists - and those are exactly the chunks the pool never judged.")
    print()
    print("  so pooling penalises a system in proportion to how DIFFERENT it")
    print("  is from the systems that built the pool. A genuinely novel")
    print("  retriever, evaluated against someone else's pool, looks worse")
    print("  the more novel it is.")

    print()
    print("=" * 72)
    print("what this day settles")
    print("=" * 72)
    print("  Day 11 ended with seventeen questions being too few to separate")
    print("  the systems. The obvious fix is to generate more, and the two")
    print("  obvious generators produce sets that separate nothing at all.")
    print()
    print("  what does work here is pooling: keep the real queries, and use")
    print("  several systems to decide which chunks are worth judging. It")
    print("  cuts the labelling by a factor of two on this corpus and by far")
    print("  more on a real one - at the cost of a bias that falls hardest on")
    print("  whichever system is least like the ones already in the pool.")
    print()
    print("  neither answer is free. The cheap set had no opinion; the cheap")
    print("  judgements have a preference. Day 13 turns each component off in")
    print("  turn and reads the damage, using the sets that survived today.")
