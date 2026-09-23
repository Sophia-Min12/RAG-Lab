"""Day 1 tests — offsets, coverage, and what overlap actually buys.

Korean strings below are functional test data. Runnable via `pytest`
(repo root) or `python -m unittest` (this folder).
"""

import unittest

from chunking import (
    BOUNDARY_FACT,
    SAMPLE,
    Chunk,
    chunk_fixed,
    chunk_sentences,
    coverage,
    duplication,
    spans_a_boundary,
    split_sentences,
)

STRATEGIES = {
    "fixed": lambda t: chunk_fixed(t, 120),
    "fixed+30": lambda t: chunk_fixed(t, 120, overlap=30),
    "fixed+60": lambda t: chunk_fixed(t, 120, overlap=60),
    "sentence": lambda t: chunk_sentences(t, 120),
}


class TestChunkFixed(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual([c.text for c in chunk_fixed("abcdef", size=3)], ["abc", "def"])

    def test_ragged_tail(self):
        self.assertEqual([c.text for c in chunk_fixed("abcdefgh", size=3)], ["abc", "def", "gh"])

    def test_overlap_repeats_the_tail(self):
        self.assertEqual(
            [c.text for c in chunk_fixed("abcdefgh", size=4, overlap=2)],
            ["abcd", "cdef", "efgh"],
        )

    def test_size_larger_than_text(self):
        self.assertEqual([c.text for c in chunk_fixed("abc", size=99)], ["abc"])

    def test_empty_text(self):
        self.assertEqual(chunk_fixed("", size=10), [])

    def test_indices_are_sequential(self):
        chunks = chunk_fixed(SAMPLE, 100, overlap=20)
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_overlap_equal_to_size_rejected(self):
        # Would advance zero characters per step and loop forever.
        with self.assertRaises(ValueError):
            chunk_fixed("abcdef", size=3, overlap=3)

    def test_overlap_larger_than_size_rejected(self):
        with self.assertRaises(ValueError):
            chunk_fixed("abcdef", size=3, overlap=5)

    def test_bad_size_rejected(self):
        with self.assertRaises(ValueError):
            chunk_fixed("abc", size=0)

    def test_negative_overlap_rejected(self):
        with self.assertRaises(ValueError):
            chunk_fixed("abc", size=3, overlap=-1)

    def test_rejects_non_string(self):
        with self.assertRaises(TypeError):
            chunk_fixed(["abc"], size=3)


class TestSplitSentences(unittest.TestCase):
    def test_three_sentences(self):
        self.assertEqual([s for s, _, _ in split_sentences("One. Two! Three?")],
                         ["One.", "Two!", "Three?"])

    def test_offsets_point_at_the_original(self):
        text = "One. Two! Three?"
        for piece, start, end in split_sentences(text):
            self.assertEqual(text[start:end], piece)

    def test_no_terminator(self):
        self.assertEqual([s for s, _, _ in split_sentences("no ending")], ["no ending"])

    def test_korean_sentences(self):
        pieces = [s for s, _, _ in split_sentences("청킹은 중요하다. 검색이 먼저다.")]
        self.assertEqual(len(pieces), 2)

    def test_empty(self):
        self.assertEqual(split_sentences(""), [])


class TestChunkSentences(unittest.TestCase):
    def test_packs_up_to_the_budget(self):
        self.assertEqual(
            [c.text for c in chunk_sentences("A one. B two. C three.", max_chars=14)],
            ["A one. B two.", "C three."],
        )

    def test_never_splits_a_sentence(self):
        for chunk in chunk_sentences(SAMPLE, 120):
            # Each chunk starts at a sentence start and ends at a sentence end.
            self.assertEqual(chunk.text, chunk.text.strip())

    def test_an_oversized_sentence_becomes_its_own_chunk(self):
        long_sentence = "x" * 300 + "."
        chunks = chunk_sentences(long_sentence, max_chars=100)
        self.assertEqual(len(chunks), 1)
        self.assertGreater(len(chunks[0]), 100)

    def test_empty(self):
        self.assertEqual(chunk_sentences("", max_chars=50), [])

    def test_bad_budget_rejected(self):
        with self.assertRaises(ValueError):
            chunk_sentences("a.", max_chars=0)


class TestOffsetsAreExact(unittest.TestCase):
    """Every strategy must be able to point back at the source."""

    def test_slicing_the_original_reproduces_the_chunk(self):
        for name, strategy in STRATEGIES.items():
            for chunk in strategy(SAMPLE):
                self.assertEqual(SAMPLE[chunk.start:chunk.end], chunk.text, name)

    def test_offsets_are_ordered(self):
        for name, strategy in STRATEGIES.items():
            chunks = strategy(SAMPLE)
            starts = [c.start for c in chunks]
            self.assertEqual(starts, sorted(starts), name)

    def test_end_is_after_start(self):
        for name, strategy in STRATEGIES.items():
            for chunk in strategy(SAMPLE):
                self.assertGreater(chunk.end, chunk.start, name)

    def test_len_matches_the_text(self):
        for chunk in chunk_fixed(SAMPLE, 120):
            self.assertEqual(len(chunk), len(chunk.text))


class TestCoverage(unittest.TestCase):
    def test_fixed_covers_everything(self):
        self.assertAlmostEqual(coverage(SAMPLE, chunk_fixed(SAMPLE, 120)), 1.0)

    def test_overlap_still_covers_everything(self):
        self.assertAlmostEqual(coverage(SAMPLE, chunk_fixed(SAMPLE, 120, overlap=40)), 1.0)

    def test_sentence_chunking_drops_only_inter_sentence_whitespace(self):
        # It trims, so coverage is just under 1.0 - and what is missing is
        # spaces, not content.
        share = coverage(SAMPLE, chunk_sentences(SAMPLE, 120))
        self.assertGreater(share, 0.95)
        self.assertLessEqual(share, 1.0)

    def test_no_chunks_covers_nothing(self):
        self.assertEqual(coverage(SAMPLE, []), 0.0)

    def test_empty_document(self):
        self.assertEqual(coverage("", []), 1.0)


class TestDuplication(unittest.TestCase):
    def test_no_overlap_is_one(self):
        self.assertAlmostEqual(duplication(SAMPLE, chunk_fixed(SAMPLE, 120)), 1.0)

    def test_overlap_costs_in_proportion(self):
        # 30/120 = 25% overlap, so roughly 1/(1 - 0.25) = 1.33x the text.
        measured = duplication(SAMPLE, chunk_fixed(SAMPLE, 120, overlap=30))
        self.assertGreater(measured, 1.2)
        self.assertLess(measured, 1.4)

    def test_more_overlap_costs_more(self):
        costs = [duplication(SAMPLE, chunk_fixed(SAMPLE, 120, overlap=o))
                 for o in (0, 20, 40, 60, 80)]
        self.assertEqual(costs, sorted(costs))

    def test_empty_document(self):
        self.assertEqual(duplication("", []), 0.0)


class TestTheTrade(unittest.TestCase):
    """Overlap buys boundary recall and is paid for in duplication."""

    def test_the_fact_is_lost_without_overlap(self):
        chunks = chunk_fixed(SAMPLE, 120)
        self.assertFalse(spans_a_boundary(chunks, BOUNDARY_FACT))

    def test_and_recovered_with_it(self):
        chunks = chunk_fixed(SAMPLE, 120, overlap=30)
        self.assertTrue(spans_a_boundary(chunks, BOUNDARY_FACT))

    def test_recovery_costs_index_size(self):
        without = duplication(SAMPLE, chunk_fixed(SAMPLE, 120))
        with_overlap = duplication(SAMPLE, chunk_fixed(SAMPLE, 120, overlap=30))
        self.assertGreater(with_overlap, without)

    def test_sentence_chunking_gets_it_free(self):
        # It never cuts mid-sentence, so a fact inside one sentence always
        # survives - with no duplication at all.
        chunks = chunk_sentences(SAMPLE, 120)
        self.assertTrue(spans_a_boundary(chunks, BOUNDARY_FACT))
        self.assertLessEqual(duplication(SAMPLE, chunks), 1.0)

    def test_the_phrase_really_does_straddle_the_cut(self):
        # Guard against the test passing for the wrong reason: no single
        # fixed(120) chunk may contain it.
        containing = [c for c in chunk_fixed(SAMPLE, 120) if BOUNDARY_FACT in c.text]
        self.assertEqual(containing, [])

    def test_a_wide_enough_overlap_always_wins(self):
        for overlap in (30, 60, 90):
            self.assertTrue(
                spans_a_boundary(chunk_fixed(SAMPLE, 120, overlap=overlap), BOUNDARY_FACT),
                overlap,
            )


class TestChunkDataclass(unittest.TestCase):
    def test_is_frozen(self):
        chunk = Chunk("a", 0, 1, 0)
        with self.assertRaises(Exception):
            chunk.text = "b"

    def test_equality_by_value(self):
        self.assertEqual(Chunk("a", 0, 1, 0), Chunk("a", 0, 1, 0))


if __name__ == "__main__":
    unittest.main()
