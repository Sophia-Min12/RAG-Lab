"""Day 7 tests - exact search, and the scaling that motivates Day 8.

Almost all of these need only numpy: the scaling story is about vectors,
not about what produced them.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from vector_store import (
    VectorStore,
    extrapolate,
    random_store,
    recall_at_k,
    scaling_exponent,
    scaling_table,
    time_search,
)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore(dimension=3)
        self.store.add(["x", "y", "z"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_it_finds_the_obvious_neighbour(self):
        self.assertEqual(self.store.search([1, 0, 0], k=1)[0][0], "x")

    def test_an_identical_vector_scores_one(self):
        self.assertAlmostEqual(self.store.search([0, 1, 0], k=1)[0][1], 1.0, places=5)

    def test_an_orthogonal_vector_scores_zero(self):
        scores = dict(self.store.search([1, 0, 0], k=3))
        self.assertAlmostEqual(scores["y"], 0.0, places=5)

    def test_results_are_ordered_by_score(self):
        scores = [s for _, s in self.store.search([1, 1, 0.5], k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_vectors_are_normalized_on_insert(self):
        store = VectorStore(dimension=2)
        store.add(["a"], [[3.0, 4.0]])
        np.testing.assert_allclose(np.linalg.norm(store.matrix, axis=1), [1.0], atol=1e-6)

    def test_an_unnormalized_query_is_handled(self):
        # Scaling a query must not change the ranking.
        small = [i for i, _ in self.store.search([1, 2, 0], k=3)]
        large = [i for i, _ in self.store.search([100, 200, 0], k=3)]
        self.assertEqual(small, large)

    def test_length_reports_the_number_of_vectors(self):
        self.assertEqual(len(self.store), 3)

    def test_adding_more_extends_the_store(self):
        self.store.add(["w"], [[1, 1, 1]])
        self.assertEqual(len(self.store), 4)
        self.assertEqual(self.store.matrix.shape, (4, 3))

    def test_k_larger_than_the_store_returns_everything(self):
        self.assertEqual(len(self.store.search([1, 0, 0], k=99)), 3)

    def test_searching_an_empty_store_returns_nothing(self):
        self.assertEqual(VectorStore(dimension=3).search([1, 0, 0], k=3), [])

    def test_memory_is_count_times_dimension_times_itemsize(self):
        store = random_store(100, dimension=16, dtype=np.float32)
        self.assertEqual(store.memory_bytes, 100 * 16 * 4)

    def test_float64_costs_twice_as_much(self):
        thin = random_store(100, dimension=16, dtype=np.float32)
        thick = random_store(100, dimension=16, dtype=np.float64)
        self.assertEqual(thick.memory_bytes, 2 * thin.memory_bytes)

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            VectorStore(dimension=0)
        with self.assertRaises(ValueError):
            self.store.search([1, 0, 0], k=0)
        with self.assertRaises(ValueError):
            self.store.add(["a"], [[1, 0]])          # wrong dimension
        with self.assertRaises(ValueError):
            self.store.add(["a", "b"], [[1, 0, 0]])  # ids and vectors disagree
        with self.assertRaises(ValueError):
            self.store.add(["a"], [1, 0, 0])         # not 2-D


class TestExactness(unittest.TestCase):
    """The property Day 8 gives up."""

    def setUp(self):
        self.store = random_store(2_000, dimension=32, seed=1)
        self.rng = np.random.default_rng(7)

    def test_the_fast_path_agrees_with_the_naive_one(self):
        for _ in range(5):
            query = self.rng.normal(size=32)
            fast = [i for i, _ in self.store.search(query, k=10)]
            slow = [i for i, _ in self.store.search_naive(query, k=10)]
            self.assertEqual(fast, slow)

    def test_the_scores_agree_too(self):
        query = self.rng.normal(size=32)
        fast = dict(self.store.search(query, k=10))
        slow = dict(self.store.search_naive(query, k=10))
        for identifier, score in fast.items():
            self.assertAlmostEqual(score, slow[identifier], places=5)

    def test_batched_search_agrees_with_single_search(self):
        queries = self.rng.normal(size=(6, 32))
        batched = self.store.search_batch(queries, k=5)
        for query, result in zip(queries, batched):
            self.assertEqual([i for i, _ in result],
                             [i for i, _ in self.store.search(query, k=5)])

    def test_recall_against_itself_is_one(self):
        # True by construction, and the reason exact search is the
        # reference an approximate index is scored against.
        query = self.rng.normal(size=32)
        for k in (1, 5, 10, 50):
            result = self.store.search(query, k=k)
            self.assertEqual(recall_at_k(result, result, k), 1.0)

    def test_it_really_does_return_the_maximum(self):
        # Independently-known answer: check the top score against a full
        # scan computed a different way.
        query = self.rng.normal(size=32)
        best = self.store.search(query, k=1)[0][1]
        direct = float((self.store.matrix @ (query / np.linalg.norm(query))).max())
        self.assertAlmostEqual(best, direct, places=5)


class TestRecallAtK(unittest.TestCase):
    def test_a_perfect_match_is_one(self):
        result = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        self.assertEqual(recall_at_k(result, result, 3), 1.0)

    def test_missing_one_of_three(self):
        exact = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        approximate = [("a", 0.9), ("b", 0.8), ("z", 0.6)]
        self.assertAlmostEqual(recall_at_k(approximate, exact, 3), 2 / 3)

    def test_finding_nothing_is_zero(self):
        exact = [("a", 0.9), ("b", 0.8)]
        approximate = [("y", 0.5), ("z", 0.4)]
        self.assertEqual(recall_at_k(approximate, exact, 2), 0.0)

    def test_order_within_the_top_k_does_not_matter(self):
        # recall@k is a set measure; ranking quality is Day 11's problem.
        exact = [("a", 0.9), ("b", 0.8)]
        self.assertEqual(recall_at_k([("b", 0.8), ("a", 0.9)], exact, 2), 1.0)

    def test_an_empty_truth_is_one(self):
        self.assertEqual(recall_at_k([], [], 3), 1.0)

    def test_k_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            recall_at_k([], [], 0)


class TestScaling(unittest.TestCase):
    """Why exact search stops, measured rather than asserted."""

    @classmethod
    def setUpClass(cls):
        cls.rows = scaling_table([2_000, 8_000, 32_000], dimension=64, queries=4)

    def test_latency_grows_with_the_collection(self):
        times = [row["ms"] for row in self.rows]
        self.assertEqual(times, sorted(times))

    def test_memory_is_exactly_linear(self):
        # Memory is arithmetic, not timing, so this can be asserted
        # tightly where the latency cannot.
        for row in self.rows:
            self.assertAlmostEqual(row["megabytes"], row["count"] * 64 * 4 / 1e6,
                                   places=6)

    def test_the_slope_is_reported_but_not_asserted(self):
        # No bound here, deliberately. At the sizes a test suite can
        # afford - a few thousand vectors - the BLAS call overhead has
        # not amortized and the measured slope comes out around 0.45,
        # against 1.03 over the demo's wider range. A bound tight enough
        # to mean "linear" would fail; one loose enough to pass would
        # assert nothing. The demo measures this where it is meaningful,
        # and what IS linear beyond argument - memory - is asserted
        # exactly, one test above.
        self.assertTrue(math.isfinite(scaling_exponent(self.rows)))

    def test_extrapolation_scales_both_numbers(self):
        small = extrapolate(self.rows, 1_000_000, dimension=64)
        large = extrapolate(self.rows, 10_000_000, dimension=64)
        self.assertAlmostEqual(large["ms"] / small["ms"], 10.0, places=3)
        self.assertAlmostEqual(large["gigabytes"] / small["gigabytes"], 10.0, places=3)

    def test_memory_at_ten_million_exceeds_a_typical_machine(self):
        # The day's conclusion, as a number: the wall is RAM.
        estimate = extrapolate(self.rows, 10_000_000, dimension=384)
        self.assertGreater(estimate["gigabytes"], 15.0)

    def test_a_million_vectors_is_still_tractable(self):
        estimate = extrapolate(self.rows, 1_000_000, dimension=384)
        self.assertLess(estimate["gigabytes"], 2.0)


class TestTiming(unittest.TestCase):
    def test_it_returns_a_positive_duration(self):
        store = random_store(500, dimension=16, seed=0)
        probes = np.random.default_rng(0).normal(size=(3, 16))
        self.assertGreater(time_search(store, probes, repeats=2), 0.0)

    def test_the_naive_path_is_slower(self):
        # Same arithmetic, same answers; the gap is the Python loop.
        store = random_store(4_000, dimension=64, seed=0)
        probes = np.random.default_rng(0).normal(size=(2, 64))
        fast = time_search(store, probes, repeats=3)
        slow = time_search(store, probes, repeats=1, method="search_naive")
        self.assertGreater(slow, fast)


if __name__ == "__main__":
    unittest.main()
