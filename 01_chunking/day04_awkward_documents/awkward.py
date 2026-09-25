"""Day 4 — Documents that break chunkers: tables, code, and Korean.

Level 1 has treated every document as prose. Three kinds are not, and each
breaks a different assumption.

A **table row** without its header is a list of values with no idea what
the columns mean. Character chunking keeps the header in one chunk out of
three, and the other two are unreadable.

A **code fragment** cut at a character count does not parse — not once, at
any size tried here, because cutting inside a function leaves an indented
fragment that Python cannot read as a program.

**Korean** says the same thing in half the characters, so a chunk size
tuned on English is a different setting entirely. Day 3 found that "best
chunk size" is undefined until the context budget and the retriever are
pinned; the language is the third thing it depends on.

Each fix costs something, and the costs are measured rather than waved at:
repeating a table header inflates the index, and splitting code at
structure boundaries gives up control of chunk size altogether.

It also settles an account Day 1 opened: that splitter's own docstring
says it is wrong on abbreviations, and here it cuts ``See Fig. 3`` into
two sentences. The fix is a list, which means it is permanently
incomplete.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

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


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("=" * 70)
    print("1. tables: a row without its header answers nothing")
    print("=" * 70)
    table = parse_markdown_table(EXPENSE_TABLE)
    print(f"  {len(EXPENSE_TABLE)} characters, 1 header + {len(table.rows)} rows")
    print()
    print(f"  {'strategy':<28}{'chunks':>8}{'header kept':>13}{'index':>8}")
    for size in (120, 200, 320):
        chunks = chunk_fixed(EXPENSE_TABLE, size)
        print(f"  {f'fixed({size})':<28}{len(chunks):>8}"
              f"{header_retention(chunks, table.header):>13.0%}"
              f"{index_growth(EXPENSE_TABLE, chunks):>8.2f}")
    for rows in (2, 4, 6):
        chunks = chunk_table(EXPENSE_TABLE, rows_per_chunk=rows)
        print(f"  {f'chunk_table({rows} rows)':<28}{len(chunks):>8}"
              f"{header_retention(chunks, table.header):>13.0%}"
              f"{index_growth(EXPENSE_TABLE, chunks):>8.2f}")
    print()
    print("  what a chunk actually looks like without the header:")
    orphan = [c for c in chunk_fixed(EXPENSE_TABLE, 120) if table.header not in c.text][1]
    for line in orphan.text.strip().splitlines()[:4]:
        print(f"    {line!r}")
    print("  values with no idea which column is which. The answer to 'do I")
    print("  need a receipt for a taxi' is in there and unreadable - and the")
    print("  first line is not even a whole row.")
    print()
    print("  repeating the header fixes it and costs index size - the same")
    print("  trade Day 1 measured for overlap, in a different shape. Fewer")
    print("  rows per chunk means more copies of the header.")

    print()
    print("=" * 70)
    print("2. code: a fragment that does not parse")
    print("=" * 70)
    print(f"  {len(POLICY_CODE)} characters of Python")
    print()
    print(f"  {'strategy':<28}{'chunks':>8}{'parse':>8}{'mean':>7}{'max':>7}")
    for size in (120, 240, 480):
        chunks = chunk_fixed(POLICY_CODE, size)
        valid = sum(parses_as_python(c.text) for c in chunks)
        sizes = [len(c.text) for c in chunks]
        print(f"  {f'fixed({size})':<28}{len(chunks):>8}"
              f"{f'{valid}/{len(chunks)}':>8}{sum(sizes)/len(sizes):>7.0f}{max(sizes):>7}")
    chunks = chunk_python(POLICY_CODE)
    valid = sum(parses_as_python(c.text) for c in chunks)
    sizes = [len(c.text) for c in chunks]
    print(f"  {'chunk_python()':<28}{len(chunks):>8}"
          f"{f'{valid}/{len(chunks)}':>8}{sum(sizes)/len(sizes):>7.0f}{max(sizes):>7}")
    print()
    print("  not one character-count chunk parses, at any size. Cutting")
    print("  anywhere inside a function leaves an indented fragment, and")
    print("  Python has no way to read that as a program:")
    print(f"    {chunk_fixed(POLICY_CODE, 240)[1].text.splitlines()[0]!r}")
    print()
    print("  the price of chunk_python is in the last column. Character")
    print("  chunking guarantees a maximum size and guarantees nothing about")
    print("  meaning; structure chunking is the other way round. A single")
    print("  large function is one large chunk, because the alternatives are")
    print("  a fragment that does not parse or a cut inside a loop body.")
    print()
    print("  each top-level unit, with its size:")
    for chunk in chunks:
        first = chunk.text.strip().splitlines()[0]
        print(f"    {len(chunk.text):>4} chars  {first[:52]!r}")

    print()
    print("=" * 70)
    print("3. Korean: the same budget is not the same budget")
    print("=" * 70)
    print("  the same five facts, written twice:")
    for label, text in (("english", PARALLEL_EN), ("korean", PARALLEL_KO)):
        print(f"    {label:<8}{len(text):>5} chars   {density(text):>6.1f} chars/sentence")
    print(f"  Korean says it in {len(PARALLEL_KO) / len(PARALLEL_EN):.0%} of the characters")
    print()
    print("  so a character budget buys different amounts of content:")
    print(f"    {'budget':>7}{'en chunks':>11}{'en facts/chunk':>16}"
          f"{'ko chunks':>11}{'ko facts/chunk':>16}")
    for budget in (60, 100, 150, 200):
        english = chunk_sentences(PARALLEL_EN, budget)
        korean = chunk_sentences(PARALLEL_KO, budget)
        print(f"    {budget:>7}{len(english):>11}{facts_per_chunk(PARALLEL_EN, budget):>16.1f}"
              f"{len(korean):>11}{facts_per_chunk(PARALLEL_KO, budget):>16.1f}")
    print()
    print("  a chunk size tuned on English hands a Korean reader twice the")
    print("  content per chunk - or, read the other way, a size that fits one")
    print("  fact in English fits two in Korean. Day 3 showed that chunk size")
    print("  only means something once the budget and retriever are pinned.")
    print("  This is the third thing it depends on, and it is the language.")

    print()
    print("=" * 70)
    print("4. the abbreviation bug Day 1 declared and did not fix")
    print("=" * 70)
    print(f"  {ABBREVIATION_PROSE[:64]!r}...")
    print()
    print("  Day 1's splitter:")
    for sentence, _, _ in split_sentences(ABBREVIATION_PROSE):
        print(f"    {sentence!r}")
    print()
    print("  with abbreviations guarded:")
    for sentence, _, _ in split_sentences_guarded(ABBREVIATION_PROSE):
        print(f"    {sentence!r}")
    print()
    print("  Day 1 cuts after any of . ! ? followed by whitespace, and its")
    print("  own docstring says it is wrong on abbreviations. This is not a")
    print("  discovery, it is the bill arriving. The rule is right for 1.75 -")
    print("  a digit follows the period, not whitespace - and wrong for")
    print("  'Fig. 3', 'dept. before' and 'Approx. 60'. Seven sentences where")
    print("  there are four, three of them fragments.")
    print()
    print("  the fix is a list of abbreviations, which means it is permanently")
    print("  incomplete: it knows Fig. and not Abb., and no list covers every")
    print("  abbreviation in every language. Better than nothing, worse than a")
    print("  model, and worth saying rather than quietly shipping either.")

    print()
    print("=" * 70)
    print("what Level 1 established")
    print("=" * 70)
    print("  chunking is not a preprocessing step that happens before the")
    print("  interesting part. It sets a ceiling nothing downstream can lift:")
    print()
    print("    Day 1  overlap buys boundary recall, paid for in index size")
    print("    Day 2  offsets into a normalized copy point at nothing, and")
    print("           the failure is silent on ASCII")
    print("    Day 3  'best chunk size' is undefined until the context budget")
    print("           and the retriever are both fixed")
    print("    Day 4  and until the document is prose. Tables, code and a")
    print("           second language each need a different splitter, and")
    print("           each fix costs something measurable")
    print()
    print("  Level 2 starts retrieving. Everything it can find was decided here.")
