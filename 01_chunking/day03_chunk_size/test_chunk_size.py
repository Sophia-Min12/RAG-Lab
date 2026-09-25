"""Day 3 tests - the metrics, and the two ways this measurement misleads.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from chunk_size import (
    QUESTIONS,
    SIZES,
    Document,
    Question,
    SourcedChunk,
    _handbook,
    answer_is_intact,
    best_size,
    build_index,
    evaluate,
    retrieve,
    retrieve_within_budget,
    score_coverage,
    score_occurrences,
    sweep,
    tokenize,
)

DOCUMENTS = _handbook()


class TestTheCorpusIsHonest(unittest.TestCase):
    """If the questions are not answerable, nothing below means anything."""

    def test_every_answer_appears_in_its_document(self):
        by_id = {document.doc_id: document for document in DOCUMENTS}
        for question in QUESTIONS:
            self.assertIn(question.answer, by_id[question.doc_id].text,
                          f"{question.query!r} has no answer in {question.doc_id}")

    def test_every_answer_appears_exactly_once_in_the_whole_corpus(self):
        # Otherwise "the chunk contains the answer" could be satisfied by
        # the wrong document, and the hit rate would flatter itself.
        for question in QUESTIONS:
            total = sum(document.text.count(question.answer) for document in DOCUMENTS)
            self.assertEqual(total, 1, f"{question.answer!r} occurs {total} times")

    def test_the_questions_cover_every_document(self):
        asked = {question.doc_id for question in QUESTIONS}
        self.assertEqual(asked, {document.doc_id for document in DOCUMENTS})

    def test_there_are_enough_questions_for_a_percentage_to_mean_something(self):
        self.assertGreaterEqual(len(QUESTIONS), 15)


class TestTokenize(unittest.TestCase):
    def test_it_lowercases(self):
        self.assertEqual(tokenize("Core Hours"), ["core", "hours"])

    def test_it_keeps_numbers_with_separators_together(self):
        # Splitting 35,000 would make the expense questions unanswerable
        # for reasons unrelated to chunking.
        self.assertIn("35,000", tokenize("up to 35,000 won"))

    def test_it_keeps_times_together(self):
        self.assertIn("22:00", tokenize("between 22:00 and 06:00"))

    def test_it_handles_hangul(self):
        self.assertEqual(tokenize("검색 증강"), ["검색", "증강"])

    def test_empty_text_gives_no_tokens(self):
        self.assertEqual(tokenize(""), [])


class TestScorers(unittest.TestCase):
    def test_occurrences_counts_repeats(self):
        self.assertEqual(score_occurrences("core hours", "core hours core hours"), 4.0)

    def test_coverage_does_not(self):
        self.assertEqual(score_coverage("core hours", "core hours core hours"), 2.0)

    def test_coverage_is_bounded_by_the_query_length(self):
        long_text = "core hours " * 100
        self.assertEqual(score_coverage("core hours", long_text), 2.0)

    def test_occurrences_is_not(self):
        long_text = "core hours " * 100
        self.assertEqual(score_occurrences("core hours", long_text), 200.0)

    def test_occurrences_prefers_the_longer_chunk_for_the_same_content(self):
        # The length bias, demonstrated rather than asserted: padding a
        # chunk with more of the same terms raises its score.
        short = "core hours are 10:00 to 16:00"
        padded = short + " " + short
        self.assertGreater(score_occurrences("core hours", padded),
                           score_occurrences("core hours", short))
        self.assertEqual(score_coverage("core hours", padded),
                         score_coverage("core hours", short))

    def test_neither_scores_an_unrelated_chunk(self):
        for scorer in (score_occurrences, score_coverage):
            self.assertEqual(scorer("core hours", "laptops are replaced"), 0.0)


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.chunks = build_index(DOCUMENTS, 120)

    def test_it_returns_k_chunks(self):
        self.assertEqual(len(retrieve(self.chunks, "core hours", k=3)), 3)

    def test_it_is_deterministic(self):
        first = retrieve(self.chunks, "core hours", k=5)
        second = retrieve(self.chunks, "core hours", k=5)
        self.assertEqual([c.index for c in first], [c.index for c in second])

    def test_the_top_result_scores_at_least_as_high_as_the_rest(self):
        ranked = retrieve(self.chunks, "core hours", k=5)
        scores = [score_coverage("core hours", c.text) for c in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_k_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            retrieve(self.chunks, "x", k=0)


class TestRetrieveWithinBudget(unittest.TestCase):
    def setUp(self):
        self.chunks = build_index(DOCUMENTS, 120)

    def test_it_stays_within_the_budget(self):
        for budget in (200, 600, 1200):
            packed = retrieve_within_budget(self.chunks, "core hours", budget=budget)
            if len(packed) > 1:
                self.assertLessEqual(sum(len(c.text) for c in packed), budget)

    def test_a_larger_budget_packs_at_least_as_many(self):
        counts = [len(retrieve_within_budget(self.chunks, "core hours", budget=b))
                  for b in (200, 400, 800, 1600)]
        self.assertEqual(counts, sorted(counts))

    def test_it_never_returns_nothing(self):
        # A chunk larger than the budget is still returned: that is a
        # chunking problem, not a retrieval failure, and scoring it as
        # the latter would blame the wrong component.
        chunks = build_index(DOCUMENTS, 480)
        self.assertTrue(retrieve_within_budget(chunks, "core hours", budget=10))

    def test_it_respects_the_ranking_rather_than_packing_what_fits(self):
        # The bug this function had first. Skipping an over-budget chunk
        # and continuing down the ranking silently reorders results by
        # size: on fixed(320) at budget 300 it packed four short tail
        # chunks instead of the one that held the answer, turning 76%
        # into 18%. Packing must stop at the first overflow.
        chunks = build_index(DOCUMENTS, 320, "fixed")
        packed = retrieve_within_budget(chunks, "how long does production access last",
                                        budget=300)
        top = retrieve(chunks, "how long does production access last", k=1)[0]
        self.assertEqual(packed[0].text, top.text)

    def test_the_packed_chunks_are_a_prefix_of_the_ranking(self):
        query = "what are the core hours"
        packed = retrieve_within_budget(self.chunks, query, budget=600)
        ranked = retrieve(self.chunks, query, k=len(self.chunks))
        self.assertEqual([c.text for c in packed],
                         [c.text for c in ranked[:len(packed)]])

    def test_a_budget_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            retrieve_within_budget(self.chunks, "x", budget=0)


class TestAnswerIsIntact(unittest.TestCase):
    """The ceiling: a property of the chunker, before retrieval exists."""

    def test_small_fixed_chunks_split_most_answers(self):
        chunks = build_index(DOCUMENTS, 40, "fixed")
        intact = sum(answer_is_intact(chunks, q.answer) for q in QUESTIONS)
        self.assertLess(intact / len(QUESTIONS), 0.5)

    def test_sentence_chunks_keep_every_answer_at_every_size(self):
        # Every answer in this handbook sits inside one sentence, so a
        # chunker that never splits sentences cannot split an answer.
        for size in SIZES:
            chunks = build_index(DOCUMENTS, size, "sentence")
            for question in QUESTIONS:
                self.assertTrue(answer_is_intact(chunks, question.answer),
                                f"size {size} split {question.answer!r}")

    def test_the_hit_rate_never_exceeds_the_ceiling(self):
        # The relationship that makes 'intact' worth measuring separately.
        for strategy in ("fixed", "sentence"):
            for size in SIZES:
                result = evaluate(build_index(DOCUMENTS, size, strategy), QUESTIONS, k=3)
                self.assertLessEqual(result["hit"], result["intact"] + 1e-9)

    def test_the_ceiling_is_not_monotone_in_chunk_size(self):
        # Bigger is not reliably safer: where a boundary lands is luck.
        ceilings = {size: evaluate(build_index(DOCUMENTS, size, "fixed"),
                                   QUESTIONS, k=3)["intact"] for size in (80, 120)}
        self.assertEqual(ceilings[80], 1.0)
        self.assertLess(ceilings[120], 1.0)


class TestTheFixedKComparisonIsRigged(unittest.TestCase):
    """Why hit@k across chunk sizes cannot answer the day's question."""

    def test_context_retrieved_grows_with_chunk_size(self):
        contexts = [evaluate(build_index(DOCUMENTS, size), QUESTIONS, k=3)["context"]
                    for size in SIZES]
        self.assertEqual(contexts, sorted(contexts))

    def test_it_grows_by_roughly_an_order_of_magnitude(self):
        small = evaluate(build_index(DOCUMENTS, 40, "fixed"), QUESTIONS, k=3)["context"]
        large = evaluate(build_index(DOCUMENTS, 480, "fixed"), QUESTIONS, k=3)["context"]
        self.assertGreater(large / small, 8.0)

    def test_the_budget_version_holds_context_roughly_constant(self):
        contexts = [evaluate(build_index(DOCUMENTS, size), QUESTIONS,
                             k=None, budget=1200)["context"] for size in (40, 80, 120)]
        self.assertLess(max(contexts) - min(contexts), 120)


class TestTheScorerChangesTheAnswer(unittest.TestCase):
    """Two defensible scorers, one corpus, different verdicts."""

    def test_they_agree_on_small_chunks(self):
        chunks = build_index(DOCUMENTS, 40, "sentence")
        a = evaluate(chunks, QUESTIONS, scorer=score_occurrences, k=None, budget=600)
        b = evaluate(chunks, QUESTIONS, scorer=score_coverage, k=None, budget=600)
        self.assertEqual(a["hit"], b["hit"])

    def test_they_disagree_on_large_ones(self):
        gaps = []
        for strategy in ("sentence", "fixed"):
            for size in (320, 480):
                chunks = build_index(DOCUMENTS, size, strategy)
                a = evaluate(chunks, QUESTIONS, scorer=score_occurrences,
                             k=None, budget=600)["hit"]
                b = evaluate(chunks, QUESTIONS, scorer=score_coverage,
                             k=None, budget=600)["hit"]
                gaps.append(abs(a - b))
        self.assertGreater(max(gaps), 0.15)

    def test_the_length_biased_scorer_is_the_pessimistic_one_on_large_chunks(self):
        chunks = build_index(DOCUMENTS, 480, "sentence")
        biased = evaluate(chunks, QUESTIONS, scorer=score_occurrences,
                          k=None, budget=600)["hit"]
        neutral = evaluate(chunks, QUESTIONS, scorer=score_coverage,
                           k=None, budget=600)["hit"]
        self.assertLess(biased, neutral)


class TestEvaluate(unittest.TestCase):
    def test_rates_are_fractions(self):
        result = evaluate(build_index(DOCUMENTS, 120), QUESTIONS, k=3)
        for key in ("hit", "intact"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    def test_passing_both_k_and_budget_is_refused(self):
        with self.assertRaises(ValueError):
            evaluate(build_index(DOCUMENTS, 120), QUESTIONS, k=3, budget=600)

    def test_passing_neither_is_refused(self):
        with self.assertRaises(ValueError):
            evaluate(build_index(DOCUMENTS, 120), QUESTIONS, k=None, budget=None)

    def test_a_larger_k_never_lowers_the_hit_rate(self):
        chunks = build_index(DOCUMENTS, 120)
        hits = [evaluate(chunks, QUESTIONS, k=k)["hit"] for k in (1, 3, 5, 10)]
        self.assertEqual(hits, sorted(hits))

    def test_retrieving_everything_reaches_the_ceiling(self):
        chunks = build_index(DOCUMENTS, 120)
        result = evaluate(chunks, QUESTIONS, k=len(chunks))
        self.assertEqual(result["hit"], result["intact"])


class TestSweepAndBestSize(unittest.TestCase):
    def test_sweep_returns_one_row_per_size(self):
        rows = sweep(DOCUMENTS, QUESTIONS, SIZES)
        self.assertEqual([row["size"] for row in rows], list(SIZES))

    def test_best_size_picks_the_highest_hit_rate(self):
        rows = [{"size": 100, "hit": 0.5}, {"size": 200, "hit": 0.9}]
        self.assertEqual(best_size(rows), 200)

    def test_ties_go_to_the_smaller_size(self):
        # If two configurations answer equally well, the one handing the
        # model less text is the better one.
        rows = [{"size": 100, "hit": 0.9}, {"size": 200, "hit": 0.9}]
        self.assertEqual(best_size(rows), 100)


class TestProvenanceSurvives(unittest.TestCase):
    """Day 2's invariant still holds for every chunk measured here."""

    def test_every_indexed_chunk_still_points_at_its_source(self):
        import unicodedata
        by_id = {document.doc_id: document for document in DOCUMENTS}
        for size in SIZES:
            for chunk in build_index(DOCUMENTS, size):
                document = by_id[chunk.doc_id]
                recovered = unicodedata.normalize(
                    "NFC", document.text[chunk.start:chunk.end])
                self.assertEqual(recovered, chunk.text)


if __name__ == "__main__":
    unittest.main()
