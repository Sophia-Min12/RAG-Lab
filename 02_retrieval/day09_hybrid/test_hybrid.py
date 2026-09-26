"""Day 9 tests - fusion rules, and when fusing is the wrong idea.

The fusion rules themselves need no model: they combine lists of
(id, score) pairs. Only the end-to-end comparison needs embeddings.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from hybrid import (
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    DenseIndex,
    HybridRetriever,
    _handbook,
    build_index,
    complementarity,
    evaluate_hybrid,
    min_max_normalize,
    reciprocal_rank_fusion,
    score_fusion,
)

DOCUMENTS = _handbook()
CHUNKS = build_index(DOCUMENTS, 80, "sentence")

_MODEL = None
_LOAD_ERROR = None
if EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(DEFAULT_MODEL)
    except Exception as error:
        _LOAD_ERROR = error

needs_model = unittest.skipUnless(
    _MODEL is not None,
    f"embedding model unavailable ({_LOAD_ERROR or 'sentence-transformers not installed'})")


class TestMinMaxNormalize(unittest.TestCase):
    def test_it_maps_to_zero_and_one(self):
        self.assertEqual(min_max_normalize([1.0, 3.0, 2.0]), [0.0, 1.0, 0.5])

    def test_an_empty_list_stays_empty(self):
        self.assertEqual(min_max_normalize([]), [])

    def test_identical_scores_all_become_one(self):
        # Rather than dividing by zero. Treating them as equally good is
        # the only defensible reading.
        self.assertEqual(min_max_normalize([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0])

    def test_a_single_score_becomes_one(self):
        self.assertEqual(min_max_normalize([7.0]), [1.0])

    def test_the_result_depends_on_the_whole_list(self):
        # The flaw the day is about: the same value normalizes differently
        # depending on what else is present, so a normalized score is not
        # a property of the item.
        shallow = min_max_normalize([10.0, 5.0])
        deep = min_max_normalize([10.0, 5.0, 0.0])
        self.assertNotEqual(shallow[1], deep[1])


class TestScoreFusion(unittest.TestCase):
    def setUp(self):
        self.sparse = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
        self.dense = [("c", 0.9), ("b", 0.8), ("d", 0.1)]

    def test_alpha_zero_is_pure_sparse(self):
        fused = score_fusion(self.sparse, self.dense, alpha=0.0, k=1)
        self.assertEqual(fused[0][0], "a")

    def test_alpha_one_is_pure_dense(self):
        fused = score_fusion(self.sparse, self.dense, alpha=1.0, k=1)
        self.assertEqual(fused[0][0], "c")

    def test_it_returns_at_most_k(self):
        self.assertEqual(len(score_fusion(self.sparse, self.dense, k=2)), 2)

    def test_items_in_both_lists_are_rewarded(self):
        # 'b' is middling in both; at alpha 0.5 it should outrank items
        # that appear strongly in one list and not at all in the other.
        fused = dict(score_fusion(self.sparse, self.dense, alpha=0.5, k=10))
        self.assertGreater(fused["b"], fused["d"])

    def test_results_are_ordered(self):
        scores = [s for _, s in score_fusion(self.sparse, self.dense, k=4)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_an_absent_item_contributes_zero_not_nothing(self):
        # The standard treatment, and an assumption worth naming: it says
        # "absent" and "worst in the list" are the same, which they are not.
        fused = dict(score_fusion(self.sparse, self.dense, alpha=0.5, k=10))
        self.assertIn("d", fused)
        self.assertAlmostEqual(fused["d"], 0.5 * 0.0)

    def test_bad_alpha_is_refused(self):
        for alpha in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                score_fusion(self.sparse, self.dense, alpha=alpha)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_agreement_beats_a_single_strong_hit(self):
        # 'b' is second in both lists; 'a' is first in one and absent from
        # the other. At the default constant, agreeing twice wins.
        first = [("a", 9.0), ("b", 8.0)]
        second = [("c", 9.0), ("b", 8.0)]
        fused = reciprocal_rank_fusion([first, second], k=1)
        self.assertEqual(fused[0][0], "b")

    def test_the_constant_controls_how_much_rank_one_is_worth(self):
        first = [("a", 9.0), ("b", 8.0)]
        second = [("c", 9.0), ("b", 8.0)]
        # At constant 0, rank 1 is worth twice rank 2, so the two
        # second-place votes no longer outweigh a single first place.
        sharp = reciprocal_rank_fusion([first, second], k=1, constant=0)
        self.assertEqual(sharp[0][0], "a")

    def test_scores_are_never_used(self):
        # The property that removes the normalization problem entirely.
        first = [("a", 1000.0), ("b", 0.001)]
        rescaled = [("a", 0.5), ("b", 0.4)]
        self.assertEqual(reciprocal_rank_fusion([first], k=2),
                         reciprocal_rank_fusion([rescaled], k=2))

    def test_it_returns_at_most_k(self):
        lists = [[("a", 1.0), ("b", 0.9), ("c", 0.8)]]
        self.assertEqual(len(reciprocal_rank_fusion(lists, k=2)), 2)

    def test_it_is_deterministic_on_ties(self):
        lists = [[("a", 1.0), ("b", 1.0)], [("b", 1.0), ("a", 1.0)]]
        self.assertEqual(reciprocal_rank_fusion(lists, k=2),
                         reciprocal_rank_fusion(lists, k=2))

    def test_a_negative_constant_is_refused(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[("a", 1.0)]], constant=-1)

    def test_fusing_one_list_preserves_its_order(self):
        single = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        self.assertEqual([i for i, _ in reciprocal_rank_fusion([single], k=3)],
                         ["a", "b", "c"])


@needs_model
class TestComplementarity(unittest.TestCase):
    """The measurement that decides whether to fuse at all."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_MODEL))

    def test_the_original_questions_have_headroom(self):
        counts = complementarity(self.retriever, QUESTIONS)
        self.assertGreater(counts["union"], counts["sparse"])
        self.assertGreater(counts["union"], counts["dense"])

    def test_the_paraphrased_questions_have_none(self):
        # BM25 finds nothing dense missed, so the union equals dense
        # alone and no fusion rule can beat it.
        counts = complementarity(self.retriever, PARAPHRASED_QUESTIONS)
        self.assertEqual(counts["sparse_only"], 0)
        self.assertEqual(counts["union"], counts["dense"])

    def test_the_union_bounds_every_fusion_method(self):
        for questions in (QUESTIONS, PARAPHRASED_QUESTIONS):
            ceiling = complementarity(self.retriever, questions)["union"] / len(questions)
            for kwargs in ({"method": "score", "alpha": 0.5},
                           {"method": "rrf"},
                           {"method": "rrf", "constant": 0}):
                self.assertLessEqual(evaluate_hybrid(self.retriever, questions, **kwargs),
                                     ceiling + 1e-9)


@needs_model
class TestFusionInPractice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_MODEL))

    def test_score_fusion_beats_both_halves_where_they_are_complementary(self):
        sparse = evaluate_hybrid(self.retriever, QUESTIONS, method="sparse")
        dense = evaluate_hybrid(self.retriever, QUESTIONS, method="dense")
        fused = evaluate_hybrid(self.retriever, QUESTIONS, method="score", alpha=0.5)
        self.assertGreater(fused, max(sparse, dense))

    def test_rrf_at_the_standard_constant_is_worse_than_dense_alone(self):
        # The day's uncomfortable result, pinned.
        dense = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS, method="dense")
        fused = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS, method="rrf")
        self.assertLess(fused, dense)

    def test_a_smaller_rrf_constant_recovers_most_of_the_gap(self):
        standard = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS, method="rrf")
        sharp = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS,
                                method="rrf", constant=0)
        self.assertGreater(sharp - standard, 0.3)

    def test_but_still_does_not_beat_dense_alone_there(self):
        dense = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS, method="dense")
        sharp = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS,
                                method="rrf", constant=0)
        self.assertLessEqual(sharp, dense)

    def test_alpha_interpolates_between_the_two_retrievers(self):
        pure_sparse = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS,
                                      method="score", alpha=0.0)
        pure_dense = evaluate_hybrid(self.retriever, PARAPHRASED_QUESTIONS,
                                     method="score", alpha=1.0)
        self.assertLess(pure_sparse, pure_dense)

    def test_an_unknown_method_is_refused(self):
        with self.assertRaises(ValueError):
            self.retriever.search("anything", method="telepathy")

    def test_the_cache_returns_the_same_results(self):
        first = self.retriever.search(QUESTIONS[0].query, k=3, method="rrf")
        second = self.retriever.search(QUESTIONS[0].query, k=3, method="rrf")
        self.assertEqual([c.text for c, _ in first], [c.text for c, _ in second])


if __name__ == "__main__":
    unittest.main()
