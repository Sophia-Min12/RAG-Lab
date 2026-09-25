"""Day 5 tests - the three ideas, and what the default does not fix.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import statistics
import unittest

from bm25 import (
    EXPENSE_TABLE,
    POLICY_CODE,
    PADDING_DEMO,
    QUESTIONS,
    SIZES,
    BM25,
    SourcedChunk,
    _handbook,
    _text_chunk,
    build_index,
    chunk_python,
    chunk_table,
    evaluate,
    evaluate_bm25,
    saturation,
    score_coverage,
    score_occurrences,
    tokenize,
)

DOCUMENTS = _handbook()
CHUNKS = build_index(DOCUMENTS, 80, "sentence")


class TestIndexConstruction(unittest.TestCase):
    def test_it_records_one_length_per_chunk(self):
        index = BM25(CHUNKS)
        self.assertEqual(len(index.lengths), len(CHUNKS))

    def test_the_average_length_is_the_mean_of_the_lengths(self):
        index = BM25(CHUNKS)
        self.assertAlmostEqual(index.average_length, statistics.mean(index.lengths))

    def test_postings_list_every_chunk_containing_a_term(self):
        index = BM25(CHUNKS)
        for term, postings in index.postings.items():
            for position, frequency in postings:
                self.assertIn(term, tokenize(CHUNKS[position].text))
                self.assertGreater(frequency, 0)

    def test_document_frequency_matches_a_direct_count(self):
        index = BM25(CHUNKS)
        for term in ("leave", "won", "laptop"):
            direct = sum(1 for chunk in CHUNKS if term in tokenize(chunk.text))
            self.assertEqual(index.document_frequency(term), direct)

    def test_an_absent_term_has_zero_document_frequency(self):
        self.assertEqual(BM25(CHUNKS).document_frequency("zzzznotaword"), 0)

    def test_an_empty_index_is_refused(self):
        with self.assertRaises(ValueError):
            BM25([])

    def test_bad_parameters_are_refused(self):
        with self.assertRaises(ValueError):
            BM25(CHUNKS, k1=-1)
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                BM25(CHUNKS, b=bad)


class TestIDF(unittest.TestCase):
    def setUp(self):
        self.sample = [_text_chunk(t) for t in (
            "the policy on leave", "the policy on expenses",
            "the policy on equipment", "the forfeited balance rule")]
        self.index = BM25(self.sample)

    def test_a_term_in_every_chunk_scores_near_zero(self):
        self.assertLess(self.index.idf("the"), 0.2)

    def test_a_rare_term_scores_much_higher(self):
        self.assertGreater(self.index.idf("forfeited") / self.index.idf("the"), 10.0)

    def test_idf_decreases_as_document_frequency_rises(self):
        values = [self.index.idf(t) for t in ("forfeited", "policy", "the")]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_it_is_never_negative(self):
        # The textbook Robertson-Sparck Jones form goes negative for a
        # term in more than half the collection, which would penalise a
        # chunk for containing a query term. The +1 inside the log is
        # what prevents it, and this checks the whole vocabulary.
        index = BM25(CHUNKS)
        for term in index.postings:
            self.assertGreaterEqual(index.idf(term), 0.0, term)

    def test_an_unseen_term_gets_the_highest_idf(self):
        seen = self.index.idf("forfeited")
        self.assertGreater(self.index.idf("nonexistent"), seen)

    def test_it_matches_the_formula(self):
        total = len(self.sample)
        for term in ("the", "policy", "forfeited"):
            frequency = self.index.document_frequency(term)
            expected = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            self.assertAlmostEqual(self.index.idf(term), expected)


class TestSaturation(unittest.TestCase):
    def test_the_first_occurrence_is_worth_the_most(self):
        gains = [saturation(n) - saturation(n - 1) for n in range(1, 6)]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_it_is_bounded_by_k1_plus_one(self):
        for k1 in (0.5, 1.2, 1.5, 2.0):
            self.assertLess(saturation(10_000, k1), k1 + 1)
            self.assertGreater(saturation(10_000, k1), (k1 + 1) * 0.999)

    def test_a_hundred_mentions_are_worth_less_than_three_firsts(self):
        # The claim score_occurrences got wrong.
        self.assertLess(saturation(100), 3 * saturation(1))

    def test_zero_occurrences_contribute_nothing(self):
        self.assertEqual(saturation(0), 0.0)


class TestLengthNormalization(unittest.TestCase):
    """The bias Day 3 measured, and the knob that controls it."""

    def setUp(self):
        self.short, self.padded = (_text_chunk(t) for t in PADDING_DEMO)
        self.query = "what are the core hours"

    def test_the_padded_chunk_does_not_contain_the_answer(self):
        self.assertIn("10:00 to 16:00", self.short.text)
        self.assertNotIn("10:00 to 16:00", self.padded.text)

    def test_without_normalization_padding_wins(self):
        index = BM25([self.short, self.padded], b=0.0)
        self.assertGreater(index.score(self.query, 1), index.score(self.query, 0))

    def test_with_full_normalization_the_answer_wins(self):
        index = BM25([self.short, self.padded], b=1.0)
        self.assertGreater(index.score(self.query, 0), index.score(self.query, 1))

    def test_the_default_does_not_fix_this_case(self):
        # The result worth stating plainly: b=0.75 is the universal
        # default and the padded chunk still ranks first. The default is
        # a compromise for ordinary collections, not a cure.
        index = BM25([self.short, self.padded], b=0.75)
        self.assertGreater(index.score(self.query, 1), index.score(self.query, 0))

    def test_raising_b_helps_the_short_chunk_monotonically(self):
        ratios = []
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            index = BM25([self.short, self.padded], b=b)
            ratios.append(index.score(self.query, 0) / index.score(self.query, 1))
        self.assertEqual(ratios, sorted(ratios))

    def test_saturation_alone_does_not_rescue_it(self):
        # Turning tf saturation off (huge k1) widens the gap; turning it
        # up does not flip the ranking either. Length is the axis here.
        for k1 in (0.5, 1.5, 10.0):
            index = BM25([self.short, self.padded], k1=k1, b=0.75)
            self.assertGreater(index.score(self.query, 1), index.score(self.query, 0))


class TestBWhenLengthsAreUniform(unittest.TestCase):
    """A null result, and the reason for it."""

    def test_sentence_chunks_barely_vary_in_length(self):
        lengths = [len(c.text) for c in CHUNKS]
        spread = statistics.pstdev(lengths) / statistics.mean(lengths)
        self.assertLess(spread, 0.25)

    def test_b_changes_nothing_on_the_handbook(self):
        for size in SIZES:
            chunks = build_index(DOCUMENTS, size, "sentence")
            hits = {evaluate_bm25(chunks, QUESTIONS, b=b, budget=600)["hit"]
                    for b in (0.0, 0.5, 1.0)}
            self.assertEqual(len(hits), 1, f"size {size} moved: {hits}")

    def test_mixing_in_tables_and_code_widens_the_spread(self):
        mixed = self.mixed_index()
        lengths = [len(c.text) for c in mixed]
        spread = statistics.pstdev(lengths) / statistics.mean(lengths)
        self.assertGreater(spread, 0.5)

    def test_and_then_b_starts_to_matter(self):
        mixed = self.mixed_index()
        without = self.hits_at_three(BM25(mixed, b=0.0))
        with_it = self.hits_at_three(BM25(mixed, b=1.0))
        self.assertGreater(with_it, without)

    @staticmethod
    def mixed_index():
        mixed = build_index(DOCUMENTS, 80, "sentence")
        mixed += [SourcedChunk("table", c.text, c.start, c.end, i, {})
                  for i, c in enumerate(chunk_table(EXPENSE_TABLE, 6))]
        mixed += [SourcedChunk("code", c.text, c.start, c.end, i, {})
                  for i, c in enumerate(chunk_python(POLICY_CODE))]
        return mixed

    @staticmethod
    def hits_at_three(index):
        return sum(any(question.answer in chunk.text
                       for chunk, _ in index.search(question.query, k=3))
                   for question in QUESTIONS)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.index = BM25(CHUNKS)

    def test_results_come_back_in_descending_score_order(self):
        scores = [score for _, score in self.index.search("core hours", k=5)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_it_returns_at_most_k(self):
        self.assertLessEqual(len(self.index.search("core hours", k=3)), 3)

    def test_it_is_reproducible(self):
        first = [c.text for c, _ in self.index.search("core hours", k=5)]
        second = [c.text for c, _ in self.index.search("core hours", k=5)]
        self.assertEqual(first, second)

    def test_only_chunks_sharing_a_term_are_returned(self):
        for chunk, score in self.index.search("laptop", k=5):
            self.assertGreater(score, 0.0)
            self.assertTrue(set(tokenize("laptop")) & set(tokenize(chunk.text)))

    def test_a_query_with_no_matching_term_returns_nothing(self):
        self.assertEqual(self.index.search("zzzznotaword", k=3), [])

    def test_k_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            self.index.search("x", k=0)

    def test_scoring_a_chunk_without_any_query_term_gives_zero(self):
        position = next(i for i, c in enumerate(CHUNKS) if "laptop" not in c.text)
        self.assertEqual(self.index.score("laptop", position), 0.0)


class TestSearchWithinBudget(unittest.TestCase):
    def setUp(self):
        self.index = BM25(CHUNKS)

    def test_it_respects_the_budget(self):
        packed = self.index.search_within_budget("core hours", budget=300)
        if len(packed) > 1:
            self.assertLessEqual(sum(len(c.text) for c in packed), 300)

    def test_it_is_a_prefix_of_the_ranking(self):
        packed = self.index.search_within_budget("core hours", budget=400)
        ranked = [c.text for c, _ in self.index.search("core hours", k=len(CHUNKS))]
        self.assertEqual([c.text for c in packed], ranked[:len(packed)])

    def test_a_larger_budget_packs_at_least_as_many(self):
        counts = [len(self.index.search_within_budget("core hours", budget=b))
                  for b in (100, 300, 900)]
        self.assertEqual(counts, sorted(counts))

    def test_a_budget_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            self.index.search_within_budget("x", budget=0)


class TestTheBaseline(unittest.TestCase):
    """The number Day 6 has to beat."""

    def test_bm25_is_never_worse_than_day_threes_scorers(self):
        for size in SIZES:
            chunks = build_index(DOCUMENTS, size, "sentence")
            reference = max(
                evaluate(chunks, QUESTIONS, scorer=s, k=None, budget=600)["hit"]
                for s in (score_occurrences, score_coverage))
            self.assertGreaterEqual(
                evaluate_bm25(chunks, QUESTIONS, budget=600)["hit"], reference,
                f"BM25 lost at size {size}")

    def test_it_is_strictly_better_where_they_disagreed(self):
        chunks = build_index(DOCUMENTS, 320, "sentence")
        biased = evaluate(chunks, QUESTIONS, scorer=score_occurrences,
                          k=None, budget=600)["hit"]
        self.assertGreater(evaluate_bm25(chunks, QUESTIONS, budget=600)["hit"], biased)

    def test_the_baseline_is_high_enough_to_be_a_real_bar(self):
        best = max(evaluate_bm25(build_index(DOCUMENTS, size, "sentence"),
                                 QUESTIONS, budget=600)["hit"] for size in SIZES)
        self.assertGreaterEqual(best, 0.9)

    def test_it_never_exceeds_the_intact_ceiling(self):
        # Day 3's invariant, still holding with a real retriever.
        for size in SIZES:
            result = evaluate_bm25(build_index(DOCUMENTS, size, "sentence"),
                                   QUESTIONS, budget=600)
            self.assertLessEqual(result["hit"], result["intact"] + 1e-9)

    def test_passing_both_k_and_budget_is_refused(self):
        with self.assertRaises(ValueError):
            evaluate_bm25(CHUNKS, QUESTIONS, k=3, budget=600)


if __name__ == "__main__":
    unittest.main()
