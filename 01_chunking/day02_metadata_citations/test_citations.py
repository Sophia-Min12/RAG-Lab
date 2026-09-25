"""Day 2 tests - provenance, offset mapping, and the citation that lies.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unicodedata
import unittest

from citations import (
    Chunk,
    Citation,
    Document,
    SourcedChunk,
    _is_continuation,
    _sample_documents,
    chunk_document,
    chunk_document_unmapped,
    chunk_fixed,
    chunk_sentences,
    cite,
    line_column,
    normalize_with_map,
    offset_drift,
    select,
    verify_citation,
)

ENGLISH, KOREAN, ACCENTED = _sample_documents()

#: The same sentence stored both ways. Identical to a reader, different
#: to len(), which is the entire problem.
NFC_TEXT = "한국어 문서"
NFD_TEXT = unicodedata.normalize("NFD", NFC_TEXT)


class TestDocument(unittest.TestCase):
    def test_it_keeps_the_text_exactly_as_given(self):
        # Not normalized, not stripped, not re-encoded. A citation points
        # at this string.
        document = Document("d", NFD_TEXT)
        self.assertEqual(document.text, NFD_TEXT)
        self.assertNotEqual(document.text, NFC_TEXT)

    def test_an_empty_id_is_refused(self):
        with self.assertRaises(ValueError):
            Document("", "text")

    def test_metadata_defaults_to_empty(self):
        self.assertEqual(dict(Document("d", "t").metadata), {})


class TestNormalizeWithMap(unittest.TestCase):
    def test_it_matches_unicodedata_normalize(self):
        for text in (NFD_TEXT, NFC_TEXT, "plain", "café", ACCENTED.text, KOREAN.text):
            normalized, _ = normalize_with_map(text)
            self.assertEqual(normalized, unicodedata.normalize("NFC", text))

    def test_the_map_has_one_entry_per_character_plus_a_sentinel(self):
        normalized, mapping = normalize_with_map(KOREAN.text)
        self.assertEqual(len(mapping), len(normalized) + 1)

    def test_the_sentinel_is_the_length_of_the_original(self):
        mapping = normalize_with_map(KOREAN.text)[1]
        self.assertEqual(mapping[-1], len(KOREAN.text))

    def test_every_span_maps_back_to_text_that_normalizes_to_it(self):
        # The property the whole day rests on, checked exhaustively on a
        # short string rather than sampled.
        raw = unicodedata.normalize("NFD", "한글 café ok")
        normalized, mapping = normalize_with_map(raw)
        for start in range(len(normalized)):
            for end in range(start + 1, len(normalized) + 1):
                recovered = unicodedata.normalize("NFC", raw[mapping[start]:mapping[end]])
                self.assertEqual(recovered, normalized[start:end])

    def test_it_is_the_identity_map_on_ascii(self):
        normalized, mapping = normalize_with_map("hello")
        self.assertEqual(normalized, "hello")
        self.assertEqual(mapping, [0, 1, 2, 3, 4, 5])

    def test_it_is_the_identity_map_on_text_already_composed(self):
        normalized, mapping = normalize_with_map(NFC_TEXT)
        self.assertEqual(normalized, NFC_TEXT)
        self.assertEqual(mapping, list(range(len(NFC_TEXT) + 1)))

    def test_decomposed_hangul_collapses_three_to_one(self):
        normalized, mapping = normalize_with_map(unicodedata.normalize("NFD", "한"))
        self.assertEqual(normalized, "한")
        self.assertEqual(mapping, [0, 3])

    def test_a_combining_mark_collapses_two_to_one(self):
        normalized, mapping = normalize_with_map("á")
        self.assertEqual(normalized, "á")
        self.assertEqual(mapping, [0, 2])

    def test_the_empty_string_maps_to_a_lone_sentinel(self):
        self.assertEqual(normalize_with_map(""), ("", [0]))

    def test_nfd_is_available_too_and_grows_the_text(self):
        normalized, mapping = normalize_with_map(NFC_TEXT, form="NFD")
        self.assertEqual(normalized, unicodedata.normalize("NFD", NFC_TEXT))
        self.assertGreater(len(normalized), len(NFC_TEXT))
        self.assertEqual(len(mapping), len(normalized) + 1)

    def test_an_unknown_form_is_refused(self):
        with self.assertRaises(ValueError):
            normalize_with_map("x", form="NFX")


class TestContinuationRule(unittest.TestCase):
    """Why grouping by combining class alone is not enough."""

    def test_a_combining_mark_continues(self):
        self.assertTrue(_is_continuation("́"))

    def test_hangul_jamo_continue_despite_a_combining_class_of_zero(self):
        # The trap. Jamo have combining class 0, so the textbook rule
        # would start a new group at each one, normalize each alone, and
        # compose nothing.
        for jamo in ("ᅡ", "ᆫ"):
            self.assertEqual(unicodedata.combining(jamo), 0)
            self.assertTrue(_is_continuation(jamo))

    def test_ordinary_characters_do_not(self):
        for character in "aZ1 .한":
            self.assertFalse(_is_continuation(character))

    def test_grouping_by_combining_class_alone_would_fail(self):
        # Demonstrates the bug the rule avoids, rather than asserting the
        # rule is right by restating it.
        raw = unicodedata.normalize("NFD", "한")
        naive = "".join(unicodedata.normalize("NFC", character) for character in raw)
        self.assertNotEqual(naive, "한")
        self.assertEqual(normalize_with_map(raw)[0], "한")


class TestOffsetDrift(unittest.TestCase):
    def test_ascii_does_not_drift(self):
        self.assertEqual(offset_drift(ENGLISH.text), 0)

    def test_decomposed_korean_drifts_a_lot(self):
        self.assertLess(offset_drift(KOREAN.text), -50)

    def test_decomposed_latin_drifts_a_little(self):
        self.assertLess(offset_drift(ACCENTED.text), 0)

    def test_the_drift_is_exactly_the_length_change(self):
        for document in (ENGLISH, KOREAN, ACCENTED):
            expected = len(unicodedata.normalize("NFC", document.text)) - len(document.text)
            self.assertEqual(offset_drift(document.text), expected)


class TestLineColumn(unittest.TestCase):
    def test_the_first_character_is_line_one_column_one(self):
        self.assertEqual(line_column("abc", 0), (1, 1))

    def test_it_counts_lines_from_one(self):
        self.assertEqual(line_column("ab\ncd", 3), (2, 1))

    def test_it_counts_columns_from_one(self):
        self.assertEqual(line_column("ab\ncd", 4), (2, 2))

    def test_the_end_offset_is_addressable(self):
        self.assertEqual(line_column("ab", 2), (1, 3))

    def test_an_offset_past_the_end_is_refused(self):
        with self.assertRaises(IndexError):
            line_column("ab", 3)

    def test_a_negative_offset_is_refused(self):
        with self.assertRaises(IndexError):
            line_column("ab", -1)

    def test_it_agrees_with_splitlines_on_a_real_document(self):
        # Independently-known answer: find the offset of each line start
        # by construction, not by calling line_column twice.
        text = ENGLISH.text
        offset = 0
        for number, line in enumerate(text.split("\n"), start=1):
            self.assertEqual(line_column(text, offset), (number, 1))
            offset += len(line) + 1


class TestTheInvariant(unittest.TestCase):
    """normalize(NFC, document.text[start:end]) == chunk.text"""

    def test_it_holds_for_every_document_strategy_and_size(self):
        checked = 0
        for document in (ENGLISH, KOREAN, ACCENTED):
            for strategy in ("fixed", "sentence"):
                for size in (30, 60, 120):
                    chunks = chunk_document(document, size=size, strategy=strategy)
                    self.assertTrue(chunks)
                    for chunk in chunks:
                        self.assertTrue(verify_citation(document, chunk),
                                        f"{document.doc_id} {strategy} {size} #{chunk.index}")
                        checked += 1
        self.assertGreater(checked, 30)

    def test_it_holds_with_overlap(self):
        for overlap in (0, 10, 25):
            for chunk in chunk_document(KOREAN, size=50, overlap=overlap):
                self.assertTrue(verify_citation(KOREAN, chunk))

    def test_offsets_stay_inside_the_document(self):
        for chunk in chunk_document(KOREAN, size=40):
            self.assertGreaterEqual(chunk.start, 0)
            self.assertLessEqual(chunk.end, len(KOREAN.text))
            self.assertLess(chunk.start, chunk.end)

    def test_an_unknown_strategy_is_refused(self):
        with self.assertRaises(ValueError):
            chunk_document(ENGLISH, strategy="semantic")


class TestTheUnmappedVersionIsWrong(unittest.TestCase):
    """The defect, pinned so it cannot come back.

    Every assertion here is about the version with the mapping step left
    out. It exists to prove the mapping does something, and to record
    that the failure is silent.
    """

    def test_it_is_indistinguishable_on_ascii(self):
        # Why the bug survives review.
        mapped = chunk_document_unmapped(ENGLISH, size=40)
        correct = chunk_document(ENGLISH, size=40)
        self.assertEqual([(c.start, c.end) for c in mapped],
                         [(c.start, c.end) for c in correct])

    def test_it_fails_every_citation_on_decomposed_korean(self):
        chunks = chunk_document_unmapped(KOREAN, size=40)
        self.assertTrue(chunks)
        self.assertEqual(sum(verify_citation(KOREAN, chunk) for chunk in chunks), 0)

    def test_it_fails_on_decomposed_latin_too(self):
        chunks = chunk_document_unmapped(ACCENTED, size=40)
        self.assertEqual(sum(verify_citation(ACCENTED, chunk) for chunk in chunks), 0)

    def test_it_raises_nothing_at_all(self):
        # The point about severity: no exception, no empty result, no
        # warning. Just the wrong text.
        chunks = chunk_document_unmapped(KOREAN, size=40)
        citation = cite(KOREAN, chunks[0])
        self.assertIsInstance(citation, Citation)
        self.assertTrue(citation.quote)

    def test_the_quote_can_begin_mid_syllable(self):
        # An orphaned trailing jamo: text no reader would ever see in a
        # document, produced by slicing decomposed Hangul at an offset
        # computed against the composed form.
        chunk = chunk_document_unmapped(KOREAN, size=40)[1]
        first = KOREAN.text[chunk.start]
        self.assertTrue(_is_continuation(first))

    def test_the_mapped_version_does_not(self):
        chunk = chunk_document(KOREAN, size=40)[1]
        self.assertFalse(_is_continuation(KOREAN.text[chunk.start]))


class TestCite(unittest.TestCase):
    def test_it_quotes_the_original_not_the_normalized_copy(self):
        chunk = chunk_document(KOREAN, size=40)[0]
        citation = cite(KOREAN, chunk, quote_chars=10_000)
        self.assertEqual(citation.quote, KOREAN.text[chunk.start:chunk.end])

    def test_it_carries_the_documents_identity(self):
        citation = cite(ENGLISH, chunk_document(ENGLISH, size=40)[0])
        self.assertEqual(citation.doc_id, ENGLISH.doc_id)
        self.assertEqual(citation.uri, ENGLISH.uri)
        self.assertEqual(citation.title, ENGLISH.title)

    def test_long_quotes_are_truncated_with_an_ellipsis(self):
        citation = cite(ENGLISH, chunk_document(ENGLISH, size=120)[0], quote_chars=20)
        self.assertTrue(citation.quote.endswith("..."))
        self.assertLessEqual(len(citation.quote), 23)

    def test_a_short_quote_is_not_truncated(self):
        citation = cite(ENGLISH, chunk_document(ENGLISH, size=20)[0], quote_chars=500)
        self.assertFalse(citation.quote.endswith("..."))

    def test_citing_across_documents_is_refused(self):
        # Nothing downstream can catch this one, so it is caught here.
        with self.assertRaises(ValueError):
            cite(ENGLISH, chunk_document(KOREAN, size=40)[0])

    def test_the_line_number_points_at_the_right_line(self):
        chunks = chunk_document(ENGLISH, size=45, strategy="sentence")
        lines = [cite(ENGLISH, chunk).line for chunk in chunks[:3]]
        self.assertEqual(lines, [1, 2, 3])


class TestSelect(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk for document in (ENGLISH, KOREAN, ACCENTED)
                       for chunk in chunk_document(document, size=60)]

    def test_it_filters_on_an_exact_value(self):
        self.assertTrue(all(c.metadata["lang"] == "ko"
                            for c in select(self.chunks, lang="ko")))

    def test_it_filters_on_membership(self):
        chosen = select(self.chunks, lang=["en", "fr"])
        self.assertTrue(all(c.metadata["lang"] in {"en", "fr"} for c in chosen))
        self.assertGreater(len(chosen), 0)

    def test_filters_combine_with_and(self):
        chosen = select(self.chunks, section="retrieval", lang="en")
        self.assertTrue(all(c.metadata["lang"] == "en" for c in chosen))
        self.assertLessEqual(len(chosen), len(select(self.chunks, section="retrieval")))

    def test_no_filters_returns_everything(self):
        self.assertEqual(len(select(self.chunks)), len(self.chunks))

    def test_an_unknown_key_matches_nothing_rather_than_raising(self):
        self.assertEqual(select(self.chunks, nonexistent="x"), [])

    def test_the_parts_sum_to_the_whole(self):
        total = sum(len(select(self.chunks, lang=lang)) for lang in ("en", "ko", "fr"))
        self.assertEqual(total, len(self.chunks))

    def test_per_call_metadata_is_merged_over_the_documents(self):
        chunks = chunk_document(ENGLISH, size=60, source="unit-test")
        self.assertEqual(chunks[0].metadata["source"], "unit-test")
        self.assertEqual(chunks[0].metadata["lang"], "en")


class TestDayOneStillWorks(unittest.TestCase):
    """The copied-forward helpers are unchanged."""

    def test_chunk_fixed_round_trips_against_its_own_input(self):
        text = "a" * 250
        self.assertTrue(all(text[c.start:c.end] == c.text for c in chunk_fixed(text, 100)))

    def test_chunk_sentences_still_respects_the_budget(self):
        for chunk in chunk_sentences(ENGLISH.text, max_chars=60):
            self.assertLessEqual(len(chunk.text), 60)

    def test_the_chunk_dataclass_is_the_day_one_one(self):
        self.assertEqual(Chunk("t", 0, 1, 0).text, "t")


if __name__ == "__main__":
    unittest.main()
