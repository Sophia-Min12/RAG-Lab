"""Day 6 tests - the dense index, and the case for and against it.

Tests needing the model skip cleanly when sentence-transformers is not
installed, or when the model cannot be fetched. A green badge therefore
means "these ran, or the model was unavailable" - `python embeddings.py`
says which, in words.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from embeddings import (
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    SIZES,
    BM25,
    DenseIndex,
    Question,
    _handbook,
    _overlong_chunk,
    _text_chunk,
    build_index,
    evaluate_bm25,
    evaluate_dense,
    require_embeddings,
    vocabulary_overlap,
)

DOCUMENTS = _handbook()

_MODEL = None
_LOAD_ERROR = None
if EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(DEFAULT_MODEL)
    except Exception as error:  # network, rate limit, disk - all skip alike
        _LOAD_ERROR = error

needs_model = unittest.skipUnless(
    _MODEL is not None,
    f"embedding model unavailable ({_LOAD_ERROR or 'sentence-transformers not installed'})")


class TestTheQuestionSets(unittest.TestCase):
    """These run without the model - they are about the corpus."""

    def test_the_paraphrased_set_asks_the_same_questions(self):
        self.assertEqual(len(PARAPHRASED_QUESTIONS), len(QUESTIONS))
        self.assertEqual([q.answer for q in PARAPHRASED_QUESTIONS],
                         [q.answer for q in QUESTIONS])
        self.assertEqual([q.doc_id for q in PARAPHRASED_QUESTIONS],
                         [q.doc_id for q in QUESTIONS])

    def test_it_asks_them_in_different_words(self):
        for original, paraphrase in zip(QUESTIONS, PARAPHRASED_QUESTIONS):
            self.assertNotEqual(original.query, paraphrase.query)

    def test_the_original_questions_share_vocabulary_with_their_answers(self):
        overlaps = [vocabulary_overlap(q, DOCUMENTS) for q in QUESTIONS]
        self.assertGreater(sum(overlaps) / len(overlaps), 0.35)

    def test_the_paraphrased_ones_barely_do(self):
        overlaps = [vocabulary_overlap(q, DOCUMENTS) for q in PARAPHRASED_QUESTIONS]
        self.assertLess(sum(overlaps) / len(overlaps), 0.20)

    def test_every_paraphrased_answer_is_still_in_its_document(self):
        by_id = {d.doc_id: d for d in DOCUMENTS}
        for question in PARAPHRASED_QUESTIONS:
            self.assertIn(question.answer, by_id[question.doc_id].text)

    def test_bm25_collapses_on_the_paraphrased_set(self):
        # No model needed to show the problem - only to solve it.
        chunks = build_index(DOCUMENTS, 80, "sentence")
        original = evaluate_bm25(chunks, QUESTIONS, budget=600)["hit"]
        paraphrased = evaluate_bm25(chunks, PARAPHRASED_QUESTIONS, budget=600)["hit"]
        self.assertGreater(original, 0.9)
        self.assertLess(paraphrased, 0.5)

    def test_require_embeddings_is_explicit_when_missing(self):
        if EMBEDDINGS_AVAILABLE:
            require_embeddings()
        else:
            with self.assertRaises(RuntimeError):
                require_embeddings()


@needs_model
class TestDenseIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = build_index(DOCUMENTS, 80, "sentence")
        cls.index = DenseIndex(cls.chunks, model=_MODEL)

    def test_one_vector_per_chunk(self):
        self.assertEqual(self.index.matrix.shape[0], len(self.chunks))

    def test_vectors_are_normalized(self):
        norms = np.linalg.norm(self.index.matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(self.chunks)), atol=1e-5)

    def test_similarity_stays_within_bounds(self):
        for _, score in self.index.search("core hours", k=5):
            self.assertLessEqual(score, 1.0 + 1e-6)
            self.assertGreaterEqual(score, -1.0 - 1e-6)

    def test_a_chunk_is_most_similar_to_itself(self):
        text = self.chunks[3].text
        best, score = self.index.search(text, k=1)[0]
        self.assertEqual(best.text, text)
        self.assertGreater(score, 0.99)

    def test_results_are_ordered_by_similarity(self):
        scores = [s for _, s in self.index.search("core hours", k=6)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_it_is_reproducible(self):
        first = [c.text for c, _ in self.index.search("core hours", k=5)]
        second = [c.text for c, _ in self.index.search("core hours", k=5)]
        self.assertEqual(first, second)

    def test_an_unrelated_query_scores_lower_than_a_related_one(self):
        related = self.index.search("what are the core hours", k=1)[0][1]
        unrelated = self.index.search("photosynthesis in marine algae", k=1)[0][1]
        self.assertGreater(related, unrelated)

    def test_budget_packing_is_a_prefix_of_the_ranking(self):
        packed = self.index.search_within_budget("core hours", budget=400)
        ranked = [c.text for c, _ in self.index.search("core hours", k=len(self.chunks))]
        self.assertEqual([c.text for c in packed], ranked[:len(packed)])

    def test_an_empty_index_is_refused(self):
        with self.assertRaises(ValueError):
            DenseIndex([], model=_MODEL)

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            self.index.search("x", k=0)
        with self.assertRaises(ValueError):
            self.index.search_within_budget("x", budget=0)


@needs_model
class TestVocabularyMismatch(unittest.TestCase):
    """The day's result: what the model is actually buying."""

    @classmethod
    def setUpClass(cls):
        cls.chunks = build_index(DOCUMENTS, 80, "sentence")
        cls.index = DenseIndex(cls.chunks, model=_MODEL)

    def test_dense_holds_up_where_bm25_collapses(self):
        dense = evaluate_dense(self.index, PARAPHRASED_QUESTIONS)["hit"]
        sparse = evaluate_bm25(self.chunks, PARAPHRASED_QUESTIONS, budget=600)["hit"]
        self.assertGreater(dense - sparse, 0.4)

    def test_dense_barely_notices_the_rephrasing(self):
        original = evaluate_dense(self.index, QUESTIONS)["hit"]
        paraphrased = evaluate_dense(self.index, PARAPHRASED_QUESTIONS)["hit"]
        self.assertLess(abs(original - paraphrased), 0.15)

    def test_bm25_loses_more_than_half_its_hits(self):
        original = evaluate_bm25(self.chunks, QUESTIONS, budget=600)["hit"]
        paraphrased = evaluate_bm25(self.chunks, PARAPHRASED_QUESTIONS, budget=600)["hit"]
        self.assertLess(paraphrased, original * 0.6)

    def test_the_win_on_the_original_questions_is_slim(self):
        # The honest half: on questions that use the document's own words
        # the model buys a few points, and at one chunk size it loses.
        dense = evaluate_dense(self.index, QUESTIONS)["hit"]
        sparse = evaluate_bm25(self.chunks, QUESTIONS, budget=600)["hit"]
        self.assertLess(dense - sparse, 0.15)

    def test_and_at_one_chunk_size_the_model_is_worse(self):
        chunks = build_index(DOCUMENTS, 240, "sentence")
        dense = evaluate_dense(DenseIndex(chunks, model=_MODEL), QUESTIONS)["hit"]
        sparse = evaluate_bm25(chunks, QUESTIONS, budget=600)["hit"]
        self.assertLess(dense, sparse)


@needs_model
class TestSilentTruncation(unittest.TestCase):
    """A chunk past the token limit is indexed by its opening and nothing else."""

    def test_the_probe_chunk_really_does_exceed_the_limit(self):
        index = DenseIndex([_text_chunk(_overlong_chunk())], model=_MODEL)
        self.assertGreater(index.token_count(_overlong_chunk()), index.token_limit)

    def test_two_chunks_differing_only_past_the_limit_embed_identically(self):
        # The failure, stated as an equality: everything after the limit
        # is not merely down-weighted, it is absent.
        short = DenseIndex([_text_chunk(_overlong_chunk(300))], model=_MODEL)
        longer = DenseIndex([_text_chunk(_overlong_chunk(600))], model=_MODEL)
        np.testing.assert_allclose(short.matrix[0], longer.matrix[0], atol=1e-5)

    def test_nothing_raises_or_warns(self):
        index = DenseIndex([_text_chunk(_overlong_chunk())], model=_MODEL)
        self.assertEqual(index.matrix.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(index.matrix)))

    def test_a_fact_in_the_tail_becomes_unreachable(self):
        text = _overlong_chunk(600)
        self.assertIn("ZEBRA", text)
        index = DenseIndex([_text_chunk(text), _text_chunk("An unrelated sentence.")],
                           model=_MODEL)
        best = index.search("what is the emergency contact code", k=1)[0][1]
        self.assertLess(best, 0.4)

    def test_truncated_chunks_reports_them(self):
        index = DenseIndex([_text_chunk("short"), _text_chunk(_overlong_chunk())],
                           model=_MODEL)
        self.assertEqual(index.truncated_chunks(), [1])

    def test_the_handbook_index_has_none(self):
        index = DenseIndex(build_index(DOCUMENTS, 480, "sentence"), model=_MODEL)
        self.assertEqual(index.truncated_chunks(), [])


if __name__ == "__main__":
    unittest.main()
