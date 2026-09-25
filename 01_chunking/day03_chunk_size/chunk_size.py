"""Day 3 — Chunk size versus answer quality, measured.

The question this day set out to answer is "what chunk size gives the best
answers". The measurement says the question is under-specified, and that
is the day's result.

Sweep chunk size against hit@3 and the answer looks obvious: bigger is
better, take the largest. Then look at how much text was retrieved to get
there — top-3 of a corpus cut at 40 characters is 120 characters of
context, top-3 of the same corpus cut at 480 is nearly 1,200. The
comparison was never between two chunk sizes.

Hold the context budget fixed and the picture changes. Change the scorer
from one that counts every matching token to one that counts distinct
query terms, and it changes again — by up to 24 points on the same
configuration, because the first rewards a chunk for being long.

One number survives both: whether any single chunk contains the whole
answer. That is a property of the chunker alone, measurable before a
retriever exists, and it caps everything downstream.

Day 5 replaces the scorer here with BM25, whose length normalization
exists precisely because of the bias this day measures.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

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


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    documents = _handbook()
    characters = sum(len(document.text) for document in documents)
    print(f"handbook: {len(documents)} documents, {characters} characters, "
          f"{len(QUESTIONS)} questions with literal answers")

    print()
    print("=" * 70)
    print("the measurement everyone makes first: hit@3 against chunk size")
    print("=" * 70)
    for strategy in ("sentence", "fixed"):
        print(f"  {strategy} chunking:")
        print(f"    {'size':>5}{'chunks':>8}{'mean':>7}{'hit@3':>8}{'context':>10}")
        for row in sweep(documents, QUESTIONS, SIZES, strategy=strategy, k=3):
            print(f"    {row['size']:>5}{row['chunks']:>8}{row['mean_chunk']:>7.0f}"
                  f"{row['hit']:>8.0%}{row['context']:>10.0f}")
    print()
    print("  read the hit column alone and the answer is obvious: bigger is")
    print("  better, use 480. Now read the context column. Top-3 of a corpus")
    print("  cut at 40 characters is 120 characters of text; top-3 of the same")
    print("  corpus cut at 480 is nearly 1,200. The comparison is not between")
    print("  two chunk sizes, it is between ten times as much context and one.")

    print()
    print("=" * 70)
    print("the same question with the context budget held fixed")
    print("=" * 70)
    budgets = (300, 600, 1200)
    for strategy in ("sentence", "fixed"):
        print(f"  {strategy} chunking - hit rate, and how many chunks fit:")
        print("    size  " + "".join(f"{f'{b} chars':>14}" for b in budgets))
        for size in SIZES:
            chunks = build_index(documents, size, strategy)
            cells = ""
            for budget in budgets:
                hit = evaluate(chunks, QUESTIONS, k=None, budget=budget)["hit"]
                fitted = sum(len(retrieve_within_budget(chunks, question.query,
                                                        budget=budget))
                             for question in QUESTIONS) / len(QUESTIONS)
                flag = " " if fitted >= 2 else "!"
                cells += f"{f'{hit:.0%} ({fitted:.1f}){flag}':>14}"
            print(f"    {size:>4}  {cells}")
    print()
    print("  ! marks a cell where fewer than two chunks fit the budget. There")
    print("  the budget has collapsed to top-1 and the number says more about")
    print("  one chunk boundary than about retrieval; a budget comparison only")
    print("  means something when the budget is a few times the chunk size.")
    print()
    print("  an earlier version of this table had fixed(320) at 18% and")
    print("  sentence(320) at 47%, which I was about to explain as exactly")
    print("  that quantization effect. It was not. retrieve_within_budget was")
    print("  skipping any chunk too large to fit and continuing down the")
    print("  ranking, so it packed four short tail-end chunks instead of the")
    print("  one chunk that held the answer. Taking chunks in rank order and")
    print("  stopping at the first overflow gives 76% and 88%. The bug was")
    print("  mine, the explanation was plausible, and the two had nothing to")
    print("  do with each other.")
    print()
    print("  reading only the unflagged cells: sentence chunking is flat, every")
    print("  size answering about as well as every other. Fixed chunking is")
    print("  not - at 40 characters it answers a quarter of the questions and")
    print("  no budget rescues it.")

    print()
    print("=" * 70)
    print("why 40-character fixed chunks cannot be rescued")
    print("=" * 70)
    print(f"  {'size':>5}{'sentence intact':>18}{'fixed intact':>15}")
    for size in SIZES:
        sentence = evaluate(build_index(documents, size, "sentence"), QUESTIONS, k=3)
        fixed = evaluate(build_index(documents, size, "fixed"), QUESTIONS, k=3)
        print(f"  {size:>5}{sentence['intact']:>18.0%}{fixed['intact']:>15.0%}")
    print()
    print("  'intact' asks whether ANY chunk holds the whole answer, before")
    print("  retrieval is involved at all. It is a ceiling: below it, no")
    print("  retriever, reranker or model can recover the fact, because no")
    print("  single chunk contains it. Fixed(40) tops out at 41%.")
    print()
    print("  the ceiling is also not monotone - fixed(80) keeps every answer")
    print("  intact and fixed(120) loses two. Where a boundary lands is luck,")
    print("  and a larger chunk is not reliably a safer one.")

    print()
    print("=" * 70)
    print("and now the part that undoes the rest of the day")
    print("=" * 70)
    print("  same corpus, same questions, same 600-character budget.")
    print("  the only difference is how a chunk is scored:")
    print()
    for strategy in ("sentence", "fixed"):
        print(f"  {strategy} chunking:")
        print(f"    {'size':>5}{'occurrences':>14}{'coverage':>11}{'gap':>7}")
        for size in SIZES:
            chunks = build_index(documents, size, strategy)
            a = evaluate(chunks, QUESTIONS, scorer=score_occurrences, k=None, budget=600)["hit"]
            b = evaluate(chunks, QUESTIONS, scorer=score_coverage, k=None, budget=600)["hit"]
            print(f"    {size:>5}{a:>14.0%}{b:>11.0%}{abs(a - b):>7.0%}")
    print()
    print("  the two scorers agree perfectly on small chunks and disagree by")
    print("  up to 24 points on large ones. score_occurrences counts every")
    print("  matching token, so a long chunk wins by being long, while")
    print("  score_coverage counts distinct query terms and does not. Neither")
    print("  is wrong; they answer different questions. On the largest chunks")
    print("  that difference decides whether the configuration looks fine or")
    print("  looks broken.")

    print()
    print("  which size does each call best?")
    for strategy in ("sentence", "fixed"):
        for name, scorer in (("occurrences", score_occurrences),
                             ("coverage", score_coverage)):
            rows = sweep(documents, QUESTIONS, SIZES, strategy=strategy,
                         scorer=scorer, k=None, budget=600)
            size = best_size(rows)
            hit = next(row["hit"] for row in rows if row["size"] == size)
            print(f"    {strategy:<9} {name:<12} -> {size:>4} ({hit:.0%})")

    print()
    print("=" * 70)
    print("what this day actually established")
    print("=" * 70)
    print("  not a best chunk size. The question 'what chunk size gives the")
    print("  best answers' has no answer until two other things are fixed:")
    print()
    print("    1. the context budget. Comparing hit@k across chunk sizes")
    print("       compares different amounts of retrieved text, and the")
    print("       larger-is-better conclusion is mostly that.")
    print("    2. the retriever. A length-biased scorer and a length-neutral")
    print("       one disagree by up to 24 points about the same configuration.")
    print()
    print("  what does survive both: the intact ceiling. Chunking that splits")
    print("  answers caps everything downstream, it is measurable before any")
    print("  retriever exists, and it is the one number here that is a")
    print("  property of the chunker alone.")
    print()
    print("  Day 5 replaces the scorer with BM25, whose length normalization")
    print("  exists precisely because of the bias measured above.")
