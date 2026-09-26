"""Day 10 tests - reranking, and the ceiling it cannot pass.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from reranking import (
    CROSS_ENCODER_AVAILABLE,
    DEFAULT_CROSS_ENCODER,
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    BM25,
    DenseIndex,
    HybridRetriever,
    Reranker,
    _handbook,
    build_index,
    evaluate_hybrid,
    evaluate_reranked,
    first_stage_recall,
)

DOCUMENTS = _handbook()
CHUNKS = build_index(DOCUMENTS, 80, "sentence")

_BI, _CROSS, _ERROR = None, None, None
if EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        _BI = SentenceTransformer(DEFAULT_MODEL)
        _CROSS = CrossEncoder(DEFAULT_CROSS_ENCODER)
    except Exception as error:
        _ERROR = error

needs_models = unittest.skipUnless(
    _CROSS is not None,
    f"cross-encoder unavailable ({_ERROR or 'sentence-transformers not installed'})")


class TestTheCeilingIsStructural(unittest.TestCase):
    """Needs no model: it is a fact about BM25's candidate set."""

    def test_bm25_cannot_return_a_chunk_with_no_shared_term(self):
        index = BM25(CHUNKS)
        for question in PARAPHRASED_QUESTIONS:
            returned = index.search(question.query, k=len(CHUNKS))
            self.assertLessEqual(len(returned), len(CHUNKS))
            for chunk, score in returned:
                self.assertGreater(score, 0.0)

    def test_some_paraphrased_queries_retrieve_almost_nothing(self):
        index = BM25(CHUNKS)
        counts = [len(index.search(q.query, k=len(CHUNKS)))
                  for q in PARAPHRASED_QUESTIONS]
        self.assertLess(sum(counts) / len(counts), len(CHUNKS) * 0.6)

    def test_at_least_one_returns_no_candidates_at_all(self):
        # Nothing to rerank, at any depth, for any reranker.
        index = BM25(CHUNKS)
        empty = sum(1 for q in PARAPHRASED_QUESTIONS
                    if not index.search(q.query, k=len(CHUNKS)))
        self.assertGreaterEqual(empty, 1)


@needs_models
class TestFirstStageRecall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI))

    def test_recall_rises_with_depth(self):
        values = [first_stage_recall(self.retriever, QUESTIONS, d, method="dense")
                  for d in (1, 3, 5, 10)]
        self.assertEqual(values, sorted(values))

    def test_bm25_stops_rising_before_the_index_is_exhausted(self):
        # The structural ceiling, in the recall numbers.
        at_twenty = first_stage_recall(self.retriever, PARAPHRASED_QUESTIONS, 20,
                                       method="sparse")
        at_everything = first_stage_recall(self.retriever, PARAPHRASED_QUESTIONS,
                                           len(CHUNKS), method="sparse")
        self.assertEqual(at_twenty, at_everything)
        self.assertLess(at_everything, 0.6)

    def test_dense_reaches_everything_by_depth_ten(self):
        self.assertEqual(
            first_stage_recall(self.retriever, PARAPHRASED_QUESTIONS, 10, method="dense"),
            1.0)


@needs_models
class TestReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI))
        cls.reranker = Reranker(model=_CROSS)

    def test_it_ranks_a_relevant_chunk_above_an_irrelevant_one(self):
        right = "Employees must be reachable during core hours, which are 10:00 to 16:00."
        wrong = "Laptops are replaced on a three year cycle."
        scores = self.reranker.score("what are the core hours", [right, wrong])
        self.assertGreater(scores[0], scores[1])

    def test_it_does_so_for_a_paraphrase_too(self):
        # The absolute score goes negative; only the order is usable, and
        # the order is still right.
        right = "Employees must be reachable during core hours, which are 10:00 to 16:00."
        wrong = "Laptops are replaced on a three year cycle."
        scores = self.reranker.score("when am I expected to be online", [right, wrong])
        self.assertGreater(scores[0], scores[1])

    def test_rerank_returns_at_most_k(self):
        best = self.reranker.rerank("core hours", CHUNKS, k=3)
        self.assertEqual(len(best), 3)

    def test_rerank_orders_by_score(self):
        best = self.reranker.rerank("core hours", CHUNKS[:8], k=8)
        scores = [s for _, s in best]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_reranking_an_empty_shortlist_returns_nothing(self):
        self.assertEqual(self.reranker.rerank("anything", [], k=3), [])

    def test_it_only_ever_returns_chunks_it_was_given(self):
        # The property the whole day rests on.
        shortlist = CHUNKS[:5]
        best = self.reranker.rerank("core hours", shortlist, k=3)
        for chunk, _ in best:
            self.assertIn(chunk, shortlist)

    def test_k_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            self.reranker.rerank("x", CHUNKS, k=0)


@needs_models
class TestWhatRerankingBuys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI))
        cls.reranker = Reranker(model=_CROSS)

    def test_it_never_exceeds_the_first_stage_ceiling(self):
        for method in ("sparse", "dense"):
            for questions in (QUESTIONS, PARAPHRASED_QUESTIONS):
                ceiling = first_stage_recall(self.retriever, questions, 20, method=method)
                got = evaluate_reranked(self.retriever, self.reranker, questions,
                                        depth=20, k=3, method=method)
                self.assertLessEqual(got, ceiling + 1e-9)

    def test_it_reaches_that_ceiling_in_every_case_here(self):
        # A strong claim, and it held for all four combinations: the
        # cross-encoder extracted everything the shortlist contained.
        for method in ("sparse", "dense"):
            for questions in (QUESTIONS, PARAPHRASED_QUESTIONS):
                ceiling = first_stage_recall(self.retriever, questions, 20, method=method)
                got = evaluate_reranked(self.retriever, self.reranker, questions,
                                        depth=20, k=3, method=method)
                self.assertAlmostEqual(got, ceiling, places=9)

    def test_dense_plus_reranking_answers_everything(self):
        for questions in (QUESTIONS, PARAPHRASED_QUESTIONS):
            self.assertEqual(
                evaluate_reranked(self.retriever, self.reranker, questions,
                                  depth=20, k=3, method="dense"), 1.0)

    def test_it_improves_on_the_unranked_first_stage(self):
        plain = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS, k=3, method="sparse")
        reranked = evaluate_reranked(self.retriever, self.reranker,
                                     PARAPHRASED_QUESTIONS, depth=20, k=3, method="sparse")
        self.assertGreater(reranked, plain)

    def test_deeper_shortlists_help_only_while_the_ceiling_rises(self):
        results = {}
        for depth in (3, 10, 26):
            results[depth] = (
                first_stage_recall(self.retriever, PARAPHRASED_QUESTIONS, depth,
                                   method="dense"),
                evaluate_reranked(self.retriever, self.reranker, PARAPHRASED_QUESTIONS,
                                  depth=depth, k=3, method="dense"))
        for ceiling, got in results.values():
            self.assertAlmostEqual(got, ceiling, places=9)
        self.assertEqual(results[10][1], results[26][1])

    def test_reranking_bm25_cannot_match_dense(self):
        # No depth rescues a candidate set that never contained the answer.
        sparse = evaluate_reranked(self.retriever, self.reranker, PARAPHRASED_QUESTIONS,
                                   depth=len(CHUNKS), k=3, method="sparse")
        dense = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS, k=3, method="dense")
        self.assertLess(sparse, dense)


@needs_models
class TestCaching(unittest.TestCase):
    def test_repeated_scoring_is_consistent(self):
        reranker = Reranker(model=_CROSS)
        first = reranker.score("core hours", [CHUNKS[0].text, CHUNKS[1].text])
        second = reranker.score("core hours", [CHUNKS[0].text, CHUNKS[1].text])
        self.assertEqual(first, second)

    def test_the_cache_fills(self):
        reranker = Reranker(model=_CROSS)
        self.assertEqual(len(reranker._cache), 0)
        reranker.score("core hours", [CHUNKS[0].text])
        self.assertEqual(len(reranker._cache), 1)


if __name__ == "__main__":
    unittest.main()
