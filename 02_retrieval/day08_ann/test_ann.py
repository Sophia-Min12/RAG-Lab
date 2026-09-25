"""Day 8 tests - approximate search, and the constant factor that eats it.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

import numpy as np

from ann import (
    IVFIndex,
    NSWIndex,
    VectorStore,
    kmeans,
    recall_at_k,
    recall_curve,
)

DIMENSION = 32
RNG = np.random.default_rng(0)
VECTORS = RNG.normal(size=(3_000, DIMENSION)).astype(np.float32)
QUERIES = RNG.normal(size=(12, DIMENSION))
EXACT = VectorStore(DIMENSION)
EXACT.add(range(len(VECTORS)), VECTORS)


class TestKMeans(unittest.TestCase):
    def test_it_returns_the_requested_number_of_centroids(self):
        centroids, _ = kmeans(VECTORS[:500], clusters=8, seed=0)
        self.assertEqual(centroids.shape, (8, DIMENSION))

    def test_every_vector_is_assigned(self):
        _, assignment = kmeans(VECTORS[:500], clusters=8, seed=0)
        self.assertEqual(len(assignment), 500)
        self.assertTrue(np.all(assignment >= 0))
        self.assertTrue(np.all(assignment < 8))

    def test_centroids_stay_unit_length(self):
        # Spherical k-means: renormalizing keeps "nearest centroid" a
        # cosine question, so one dot product serves clustering and search.
        centroids, _ = kmeans(VECTORS[:500], clusters=8, seed=0)
        np.testing.assert_allclose(np.linalg.norm(centroids, axis=1),
                                   np.ones(8), atol=1e-4)

    def test_each_vector_is_assigned_to_its_nearest_centroid(self):
        centroids, assignment = kmeans(VECTORS[:400], clusters=6, seed=0)
        best = np.argmax(VECTORS[:400] @ centroids.T, axis=1)
        np.testing.assert_array_equal(assignment, best)

    def test_it_is_reproducible(self):
        first = kmeans(VECTORS[:400], clusters=6, seed=3)[1]
        second = kmeans(VECTORS[:400], clusters=6, seed=3)[1]
        np.testing.assert_array_equal(first, second)

    def test_no_cluster_is_left_empty(self):
        # Lloyd's fails this way: an empty cluster stays empty forever and
        # the list is wasted. The implementation reseeds instead.
        _, assignment = kmeans(VECTORS[:600], clusters=12, seed=1)
        self.assertEqual(len(set(assignment.tolist())), 12)

    def test_more_clusters_than_vectors_is_clamped(self):
        centroids, _ = kmeans(VECTORS[:5], clusters=50, seed=0)
        self.assertEqual(len(centroids), 5)

    def test_zero_clusters_is_refused(self):
        with self.assertRaises(ValueError):
            kmeans(VECTORS[:10], clusters=0)


class TestIVF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = IVFIndex(DIMENSION, n_lists=32, seed=0).fit(
            range(len(VECTORS)), VECTORS)

    def test_every_vector_lands_in_exactly_one_list(self):
        total = sum(len(members) for members in self.index.lists)
        self.assertEqual(total, len(VECTORS))
        seen = np.concatenate(self.index.lists)
        self.assertEqual(len(set(seen.tolist())), len(VECTORS))

    def test_probing_every_list_is_exact(self):
        # The sanity check an approximation should pass: a setting where
        # it is not approximate.
        for query in QUERIES[:5]:
            found = self.index.search(query, k=10, n_probe=self.index.n_lists)
            truth = EXACT.search(query, k=10)
            self.assertEqual(recall_at_k(found, truth, 10), 1.0)

    def test_probing_every_list_scans_everything(self):
        scanned = self.index.vectors_scanned(QUERIES[0], n_probe=self.index.n_lists)
        self.assertEqual(scanned, len(VECTORS))

    def test_recall_rises_with_n_probe(self):
        rows = recall_curve(EXACT, self.index, QUERIES[:6], k=10,
                            settings=(1, 4, 16, 32))
        recalls = [row["recall"] for row in rows]
        self.assertEqual(recalls, sorted(recalls))

    def test_work_rises_with_n_probe(self):
        scanned = [self.index.vectors_scanned(QUERIES[0], n_probe=p)
                   for p in (1, 4, 16, 32)]
        self.assertEqual(scanned, sorted(scanned))

    def test_one_probe_scans_a_small_fraction(self):
        scanned = self.index.vectors_scanned(QUERIES[0], n_probe=1)
        self.assertLess(scanned / len(VECTORS), 0.15)

    def test_recall_is_below_one_when_probing_little(self):
        # The thing being bought and sold. If this ever passed at 1.0 the
        # index would not be approximating anything.
        rows = recall_curve(EXACT, self.index, QUERIES, k=10, settings=(1,))
        self.assertLess(rows[0]["recall"], 1.0)

    def test_the_blocked_and_gathered_paths_agree_exactly(self):
        # Same answers, different memory layout. The whole day turns on
        # them being identical in result and not in cost.
        for query in QUERIES[:5]:
            blocked = self.index.search(query, k=10, n_probe=4)
            gathered = self.index.search_gathered(query, k=10, n_probe=4)
            self.assertEqual([i for i, _ in blocked], [i for i, _ in gathered])

    def test_blocks_hold_the_same_vectors_as_the_lists(self):
        for members, block in zip(self.index.lists, self.index.blocks):
            np.testing.assert_allclose(block, self.index.matrix[members])

    def test_blocks_are_contiguous(self):
        # The one property that makes the fast path fast.
        for block in self.index.blocks:
            self.assertTrue(block.flags["C_CONTIGUOUS"])

    def test_scores_are_cosines_of_unit_vectors(self):
        for _, score in self.index.search(QUERIES[0], k=5, n_probe=8):
            self.assertLessEqual(score, 1.0 + 1e-5)
            self.assertGreaterEqual(score, -1.0 - 1e-5)

    def test_results_are_ordered(self):
        scores = [s for _, s in self.index.search(QUERIES[0], k=8, n_probe=8)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            IVFIndex(0)
        with self.assertRaises(ValueError):
            IVFIndex(DIMENSION, n_lists=0)
        with self.assertRaises(ValueError):
            self.index.search(QUERIES[0], k=0)
        with self.assertRaises(ValueError):
            self.index.search(QUERIES[0], n_probe=0)

    def test_an_empty_index_returns_nothing(self):
        self.assertEqual(IVFIndex(DIMENSION).search(QUERIES[0]), [])


class TestNSW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = VECTORS[:800]
        cls.exact = VectorStore(DIMENSION)
        cls.exact.add(range(len(cls.small)), cls.small)
        cls.index = NSWIndex(DIMENSION, m=8, seed=0).fit(
            range(len(cls.small)), cls.small)

    def test_every_node_has_neighbours(self):
        for position, links in enumerate(self.index.neighbours):
            if position:
                self.assertTrue(links, f"node {position} is isolated")

    def test_the_graph_is_undirected_enough_to_stay_connected(self):
        # Early nodes must be reachable from later ones, or the entry
        # point becomes a dead end.
        self.assertTrue(self.index.neighbours[0])

    def test_recall_rises_with_ef(self):
        rows = recall_curve(self.exact, self.index, QUERIES[:6], k=10,
                            settings=(4, 16, 64), parameter="ef")
        recalls = [row["recall"] for row in rows]
        self.assertEqual(recalls, sorted(recalls))

    def test_work_rises_with_ef(self):
        scanned = [self.index.vectors_scanned(QUERIES[0], ef=e) for e in (4, 16, 64)]
        self.assertEqual(scanned, sorted(scanned))

    def test_it_visits_far_fewer_nodes_than_the_collection(self):
        self.assertLess(self.index.vectors_scanned(QUERIES[0], ef=16),
                        len(self.small) // 2)

    def test_it_finds_something_reasonable(self):
        rows = recall_curve(self.exact, self.index, QUERIES[:6], k=10,
                            settings=(64,), parameter="ef")
        self.assertGreater(rows[0]["recall"], 0.1)

    def test_results_are_ordered(self):
        scores = [s for _, s in self.index.search(QUERIES[0], k=8, ef=32)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            NSWIndex(DIMENSION, m=0)
        with self.assertRaises(ValueError):
            self.index.search(QUERIES[0], k=0)
        with self.assertRaises(ValueError):
            self.index.search(QUERIES[0], ef=0)


class TestTheConstantFactor(unittest.TestCase):
    """The day's real finding, asserted on work rather than on the clock."""

    def test_the_gather_copies_as_many_rows_as_it_scans(self):
        # Timing is too noisy to assert; the cause is not. The gathered
        # path materializes a new array of the candidate rows, and that
        # array is what costs.
        index = IVFIndex(DIMENSION, n_lists=32, seed=0).fit(
            range(len(VECTORS)), VECTORS)
        scanned = index.vectors_scanned(QUERIES[0], n_probe=4)
        gathered = index.matrix[np.concatenate(
            [index.lists[int(i)] for i in
             np.argsort(-(index.centroids @ (QUERIES[0] /
                                             np.linalg.norm(QUERIES[0]))))[:4]])]
        self.assertEqual(len(gathered), scanned)
        self.assertFalse(np.shares_memory(gathered, index.matrix))

    def test_the_blocked_path_copies_nothing_at_query_time(self):
        index = IVFIndex(DIMENSION, n_lists=32, seed=0).fit(
            range(len(VECTORS)), VECTORS)
        # The blocks were built once; a query multiplies them in place.
        self.assertTrue(all(block.flags["C_CONTIGUOUS"] for block in index.blocks))
        total = sum(len(block) for block in index.blocks)
        self.assertEqual(total, len(VECTORS))

    def test_ivf_does_less_arithmetic_than_exact(self):
        # The asymptotic claim, which is true regardless of wall-clock.
        index = IVFIndex(DIMENSION, n_lists=32, seed=0).fit(
            range(len(VECTORS)), VECTORS)
        scanned = index.vectors_scanned(QUERIES[0], n_probe=4)
        self.assertLess(scanned, len(VECTORS) / 4)


class TestRecallCurve(unittest.TestCase):
    def test_it_returns_one_row_per_setting(self):
        rows = recall_curve(EXACT, IVFIndex(DIMENSION, n_lists=16, seed=0).fit(
            range(len(VECTORS)), VECTORS), QUERIES[:4], k=5, settings=(1, 2, 4))
        self.assertEqual([row["n_probe"] for row in rows], [1, 2, 4])

    def test_recall_is_a_fraction(self):
        rows = recall_curve(EXACT, IVFIndex(DIMENSION, n_lists=16, seed=0).fit(
            range(len(VECTORS)), VECTORS), QUERIES[:4], k=5, settings=(1, 4))
        for row in rows:
            self.assertGreaterEqual(row["recall"], 0.0)
            self.assertLessEqual(row["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
