"""Day 11 tests - the metrics, checked against hand-computed answers.

The metrics need no model: they are arithmetic over lists of 0s and 1s,
so almost everything here runs anywhere.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

from metrics import (
    average_precision,
    confidence_interval,
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k_judged,
    reciprocal_rank,
)


class TestHitAtK(unittest.TestCase):
    def test_it_is_one_when_something_relevant_is_in_range(self):
        self.assertEqual(hit_at_k([0, 1, 0], 3), 1.0)

    def test_it_is_zero_when_the_relevant_item_is_out_of_range(self):
        self.assertEqual(hit_at_k([0, 1, 0], 1), 0.0)

    def test_it_cannot_tell_first_place_from_third(self):
        # The limitation the day is about.
        self.assertEqual(hit_at_k([1, 0, 0], 3), hit_at_k([0, 0, 1], 3))

    def test_an_empty_result_list_scores_zero(self):
        self.assertEqual(hit_at_k([], 3), 0.0)

    def test_k_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            hit_at_k([1], 0)


class TestPrecisionAtK(unittest.TestCase):
    def test_hand_computed(self):
        self.assertEqual(precision_at_k([1, 0, 0, 1], 4), 0.5)

    def test_it_divides_by_k_not_by_what_was_returned(self):
        # Two relevant in a top-5 is 0.4 even if only three came back.
        self.assertAlmostEqual(precision_at_k([1, 1, 0], 5), 2 / 5)

    def test_with_one_relevant_chunk_it_is_bounded_by_one_over_k(self):
        # Why it is nearly useless on this corpus.
        for k in (1, 3, 5, 10):
            self.assertLessEqual(precision_at_k([1] + [0] * 20, k), 1 / k + 1e-12)

    def test_perfect_precision(self):
        self.assertEqual(precision_at_k([1, 1, 1], 3), 1.0)


class TestRecallAtK(unittest.TestCase):
    def test_hand_computed(self):
        self.assertEqual(recall_at_k_judged([1, 0, 1], 3, total_relevant=4), 0.5)

    def test_it_uses_the_collection_total_not_what_was_retrieved(self):
        # The commonest way to report an incomparable recall.
        judgements = [1, 1]
        self.assertEqual(recall_at_k_judged(judgements, 2, total_relevant=2), 1.0)
        self.assertEqual(recall_at_k_judged(judgements, 2, total_relevant=10), 0.2)

    def test_nothing_relevant_in_the_collection_is_perfect_recall(self):
        self.assertEqual(recall_at_k_judged([0, 0], 2, total_relevant=0), 1.0)

    def test_a_negative_total_is_refused(self):
        with self.assertRaises(ValueError):
            recall_at_k_judged([1], 1, total_relevant=-1)


class TestReciprocalRank(unittest.TestCase):
    def test_rank_one_scores_one(self):
        self.assertEqual(reciprocal_rank([1, 0, 0]), 1.0)

    def test_rank_two_scores_a_half(self):
        self.assertEqual(reciprocal_rank([0, 1, 1]), 0.5)

    def test_nothing_relevant_scores_zero(self):
        self.assertEqual(reciprocal_rank([0, 0]), 0.0)

    def test_it_ignores_everything_after_the_first_hit(self):
        self.assertEqual(reciprocal_rank([0, 1, 0, 0]), reciprocal_rank([0, 1, 1, 1]))

    def test_it_falls_steeply(self):
        self.assertAlmostEqual(reciprocal_rank([0] * 14 + [1]), 1 / 15)


class TestAveragePrecision(unittest.TestCase):
    def test_hand_computed(self):
        # Relevant at ranks 1 and 3: (1/1 + 2/3) / 2
        self.assertAlmostEqual(average_precision([1, 0, 1]), (1.0 + 2 / 3) / 2)

    def test_it_rewards_finding_several_early(self):
        self.assertGreater(average_precision([1, 1, 0, 0]),
                           average_precision([1, 0, 0, 1]))

    def test_with_one_relevant_item_it_equals_reciprocal_rank(self):
        # Why MAP and MRR are the same column on this corpus.
        for position in range(6):
            judgements = [0] * position + [1] + [0] * 3
            self.assertAlmostEqual(average_precision(judgements, total_relevant=1),
                                   reciprocal_rank(judgements))

    def test_nothing_relevant_scores_zero(self):
        self.assertEqual(average_precision([0, 0]), 0.0)


class TestDCGAndNDCG(unittest.TestCase):
    def test_dcg_is_hand_computable(self):
        # Relevant at rank 2 only: 1/log2(3)
        self.assertAlmostEqual(dcg_at_k([0, 1, 0], 3), 1 / math.log2(3))

    def test_a_perfect_ordering_scores_one(self):
        self.assertAlmostEqual(ndcg_at_k([1, 0, 0], 3, total_relevant=1), 1.0)

    def test_it_falls_with_rank(self):
        values = [ndcg_at_k([0] * p + [1], 10, total_relevant=1) for p in range(5)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_the_discount_is_gentler_than_reciprocal_rank(self):
        # The whole reason the two metrics can disagree about a system.
        for rank in (3, 5, 15):
            judgements = [0] * (rank - 1) + [1]
            self.assertGreater(ndcg_at_k(judgements, rank, total_relevant=1),
                               reciprocal_rank(judgements))

    def test_nothing_relevant_scores_zero(self):
        self.assertEqual(ndcg_at_k([0, 0], 2, total_relevant=0), 0.0)

    def test_k_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            dcg_at_k([1], 0)


class TestMetricsCanDisagree(unittest.TestCase):
    """The day's finding, reproduced without needing a retriever."""

    def test_mrr_and_ndcg_can_rank_two_systems_oppositely(self):
        # System A: fourteen answers at rank 1, three at rank 2.
        # System B: fifteen at rank 1, one at rank 2, one at rank 15.
        # Both are functions of rank alone, so they agree on every query.
        def ranks_to_judgements(rank, length=20):
            return [0] * (rank - 1) + [1] + [0] * (length - rank)

        system_a = [1] * 14 + [2] * 3
        system_b = [1] * 15 + [2] + [15]

        def mean(ranks, metric):
            return sum(metric(ranks_to_judgements(r)) for r in ranks) / len(ranks)

        mrr_a = mean(system_a, reciprocal_rank)
        mrr_b = mean(system_b, reciprocal_rank)
        ndcg_a = mean(system_a, lambda j: ndcg_at_k(j, 20, total_relevant=1))
        ndcg_b = mean(system_b, lambda j: ndcg_at_k(j, 20, total_relevant=1))

        self.assertGreater(mrr_b, mrr_a)    # MRR prefers B
        self.assertGreater(ndcg_a, ndcg_b)  # nDCG prefers A

    def test_they_never_disagree_on_a_single_query(self):
        # Both are monotone decreasing in rank, so per query they must
        # order any two results the same way.
        for rank in range(1, 20):
            for other in range(rank + 1, 21):
                first = [0] * (rank - 1) + [1] + [0] * 20
                second = [0] * (other - 1) + [1] + [0] * 20
                self.assertGreater(reciprocal_rank(first), reciprocal_rank(second))
                self.assertGreater(ndcg_at_k(first, 25, total_relevant=1),
                                   ndcg_at_k(second, 25, total_relevant=1))


class TestConfidenceInterval(unittest.TestCase):
    def test_it_brackets_the_estimate(self):
        low, high = confidence_interval(14, 17)
        self.assertLess(low, 14 / 17)
        self.assertGreater(high, 14 / 17)

    def test_a_larger_sample_narrows_it(self):
        small = confidence_interval(80, 100)
        large = confidence_interval(800, 1000)
        self.assertLess(large[1] - large[0], small[1] - small[0])

    def test_sixteen_of_seventeen_contains_one_hundred_percent(self):
        # The point: 94% and 100% are not distinguishable at this size.
        low, high = confidence_interval(16, 17)
        self.assertLess(low, 1.0)
        self.assertGreater(high, 1.0)

    def test_the_approximation_degenerates_at_the_boundary(self):
        # Zero width at p=1, because the variance p(1-p) is zero. Not a
        # bug - a reason not to trust it where Level 2 spent its time.
        low, high = confidence_interval(17, 17)
        self.assertEqual(low, high)

    def test_a_zero_total_is_refused(self):
        with self.assertRaises(ValueError):
            confidence_interval(0, 0)


if __name__ == "__main__":
    unittest.main()
