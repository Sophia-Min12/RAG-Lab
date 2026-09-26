"""Day 13 tests - ablations, interactions, and what they do not measure.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest
from dataclasses import replace

from ablations import (
    ABLATIONS,
    DEFAULT_CROSS_ENCODER,
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    Pipeline,
    _handbook,
    ablate,
    interaction,
    run_pipeline,
)

DOCUMENTS = _handbook()

_BI, _CROSS, _ERROR = None, None, None
if EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        _BI = SentenceTransformer(DEFAULT_MODEL)
        _CROSS = CrossEncoder(DEFAULT_CROSS_ENCODER)
    except Exception as error:
        _ERROR = error

needs_models = unittest.skipUnless(_CROSS is not None, f"models unavailable ({_ERROR})")


class TestPipeline(unittest.TestCase):
    """No model needed: it is a configuration object."""

    def test_the_default_is_the_best_configuration_found(self):
        pipeline = Pipeline()
        self.assertEqual(pipeline.strategy, "sentence")
        self.assertEqual(pipeline.method, "dense")
        self.assertTrue(pipeline.rerank)

    def test_the_label_names_the_components(self):
        self.assertEqual(Pipeline().label(), "sentence(80) + dense + rerank")

    def test_turning_the_reranker_off_shows_in_the_label(self):
        self.assertNotIn("rerank", replace(Pipeline(), rerank=False).label())

    def test_it_is_frozen_so_an_ablation_cannot_leak(self):
        # replace() returns a new pipeline; nothing mutates a shared one.
        with self.assertRaises(Exception):
            Pipeline().rerank = False

    def test_every_ablation_changes_exactly_one_field(self):
        # Otherwise the delta is not attributable to one component.
        baseline = Pipeline()
        for name, changes in ABLATIONS.items():
            if name == "full pipeline":
                self.assertEqual(changes, {})
                continue
            self.assertEqual(len(changes), 1, f"{name} changes {len(changes)} fields")
            for field_name in changes:
                self.assertTrue(hasattr(baseline, field_name))


@needs_models
class TestAblationTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = list(QUESTIONS) + list(PARAPHRASED_QUESTIONS)
        cls.rows = ablate(DOCUMENTS, cls.questions, model=_BI, cross_encoder=_CROSS)
        cls.by_name = {row["name"]: row for row in cls.rows}

    def test_the_baseline_has_zero_delta(self):
        self.assertEqual(self.rows[0]["name"], "full pipeline")
        self.assertEqual(self.rows[0]["delta"], 0.0)

    def test_one_row_per_ablation(self):
        self.assertEqual(len(self.rows), len(ABLATIONS))

    def test_removing_embeddings_is_the_largest_loss(self):
        losses = {row["name"]: row["delta"] for row in self.rows}
        self.assertEqual(min(losses, key=losses.get), "no embeddings (BM25)")

    def test_removing_the_reranker_costs_something(self):
        self.assertLess(self.by_name["no reranker"]["delta"], 0.0)

    def test_fixed_chunks_cost_more_than_the_reranker(self):
        # Day 4's finding, still visible three levels later.
        self.assertLess(self.by_name["fixed chunks"]["delta"],
                        self.by_name["no reranker"]["delta"])

    def test_chunk_size_has_no_measurable_effect(self):
        # Consistent with Day 3: sentence chunking is flat in size.
        for name in ("tiny chunks (40)", "large chunks (320)"):
            self.assertLess(abs(self.by_name[name]["delta"]), 0.05)

    def test_the_paraphrased_half_shows_a_larger_embedding_effect(self):
        # The same ablation, a different question set, a different answer.
        whole = self.by_name["no embeddings (BM25)"]["delta"]
        half = {row["name"]: row["delta"] for row in
                ablate(DOCUMENTS, list(PARAPHRASED_QUESTIONS), model=_BI,
                       cross_encoder=_CROSS)}["no embeddings (BM25)"]
        self.assertLess(half, whole)


@needs_models
class TestInteractions(unittest.TestCase):
    """The day's finding: single-component deltas do not add up."""

    @classmethod
    def setUpClass(cls):
        cls.questions = list(QUESTIONS) + list(PARAPHRASED_QUESTIONS)

    def measure(self, first, second):
        return interaction(DOCUMENTS, self.questions, first, second,
                           model=_BI, cross_encoder=_CROSS)

    def test_removing_both_is_worse_than_removing_either(self):
        result = self.measure({"rerank": False}, {"method": "sparse"})
        self.assertLess(result["without_both"], result["without_first"])
        self.assertLess(result["without_both"], result["without_second"])

    def test_the_reranker_and_embeddings_cover_for_each_other(self):
        # Negative interaction: together they cost more than the sum.
        result = self.measure({"rerank": False}, {"method": "sparse"})
        self.assertLess(result["interaction"], -0.05)

    def test_embeddings_and_chunking_damage_overlaps(self):
        # Positive interaction: a question can only be lost once.
        result = self.measure({"method": "sparse"}, {"strategy": "fixed"})
        self.assertGreater(result["interaction"], 0.0)

    def test_the_additive_prediction_is_arithmetic(self):
        result = self.measure({"rerank": False}, {"strategy": "fixed"})
        expected = (result["full"] + (result["without_first"] - result["full"])
                    + (result["without_second"] - result["full"]))
        self.assertAlmostEqual(result["additive_prediction"], expected, places=9)

    def test_the_interaction_is_the_gap_from_that_prediction(self):
        result = self.measure({"rerank": False}, {"strategy": "fixed"})
        self.assertAlmostEqual(
            result["interaction"],
            result["without_both"] - result["additive_prediction"], places=9)


@needs_models
class TestTheRerankerDependsOnItsFirstStage(unittest.TestCase):
    """Why 'how much does the reranker matter' has four answers here."""

    @classmethod
    def setUpClass(cls):
        cls.questions = list(QUESTIONS) + list(PARAPHRASED_QUESTIONS)
        cls.gains = {}
        cls.with_rerank = {}
        for method in ("sparse", "rrf", "score", "dense"):
            without = run_pipeline(DOCUMENTS, cls.questions,
                                   Pipeline(method=method, rerank=False),
                                   _BI, _CROSS)["mrr"]
            with_it = run_pipeline(DOCUMENTS, cls.questions,
                                   Pipeline(method=method, rerank=True),
                                   _BI, _CROSS)["mrr"]
            cls.gains[method] = with_it - without
            cls.with_rerank[method] = with_it

    def test_reranking_never_hurts(self):
        for method, gain in self.gains.items():
            self.assertGreaterEqual(gain, 0.0, method)

    def test_the_gain_varies_by_several_fold(self):
        self.assertGreater(max(self.gains.values()) / min(self.gains.values()), 3.0)

    def test_it_helps_the_weakest_ordering_most(self):
        self.assertGreater(self.gains["rrf"], self.gains["dense"])

    def test_three_first_stages_converge_to_the_same_score(self):
        # The reranker replaces their ordering, so only their candidate
        # sets matter - and those three retrieve the same things.
        converged = {round(self.with_rerank[m], 6) for m in ("rrf", "score", "dense")}
        self.assertEqual(len(converged), 1)

    def test_bm25_does_not_converge_with_them(self):
        # Day 10: its candidate set lacks the answer, so there is nothing
        # to reorder.
        self.assertLess(self.with_rerank["sparse"], self.with_rerank["dense"])


if __name__ == "__main__":
    unittest.main()
