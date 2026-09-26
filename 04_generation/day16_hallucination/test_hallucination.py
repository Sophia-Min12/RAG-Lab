"""Day 16 tests - faithfulness without a judge model.

The three probes here need the generator; the machinery around them does
not.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from hallucination import (
    CONTRADICTIONS,
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    GENERATOR_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    UNANSWERABLE,
    DenseIndex,
    Generator,
    HybridRetriever,
    _handbook,
    abstained,
    abstention_report,
    build_index,
    counterfactual_faithfulness,
    find_supporting_chunk,
    pack,
    poisoned_chunk,
    poisoning_report,
    remove_supporting_chunk,
)

DOCUMENTS = _handbook()
CHUNKS = build_index(DOCUMENTS, 80, "sentence")

_GEN, _BI, _ERROR = None, None, None
if GENERATOR_AVAILABLE and EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer

        _BI = SentenceTransformer(DEFAULT_MODEL)
        _GEN = Generator()
    except Exception as error:
        _ERROR = error

needs_generator = unittest.skipUnless(_GEN is not None, f"generator unavailable ({_ERROR})")


class TestRemoveSupportingChunk(unittest.TestCase):
    def test_it_removes_exactly_the_chunk_holding_the_answer(self):
        remaining, support = remove_supporting_chunk(CHUNKS, "10:00 to 16:00")
        self.assertIsNotNone(support)
        self.assertEqual(len(remaining), len(CHUNKS) - 1)
        self.assertNotIn(support, remaining)

    def test_the_answer_is_gone_from_what_remains(self):
        # The property that makes the counterfactual a counterfactual.
        remaining, _ = remove_supporting_chunk(CHUNKS, "10:00 to 16:00")
        self.assertIsNone(find_supporting_chunk("10:00 to 16:00", remaining))

    def test_an_unsupported_answer_removes_nothing(self):
        remaining, support = remove_supporting_chunk(CHUNKS, "the capital of France")
        self.assertIsNone(support)
        self.assertEqual(len(remaining), len(CHUNKS))


class TestTheProbeCorpus(unittest.TestCase):
    """The questions and false facts have to be honest for the probes to be."""

    def test_no_unanswerable_question_is_answered_by_the_handbook(self):
        # If one were, the model would be marked down for being right.
        text = " ".join(document.text.lower() for document in DOCUMENTS)
        for query in UNANSWERABLE:
            for probe in ("sick day", "parental", "salary", "salaries", "bonus",
                          "notice period", "transfer"):
                if probe in query.lower():
                    self.assertNotIn(probe, text, f"{query!r} is answerable")

    def test_the_unanswerable_questions_look_like_the_answerable_ones(self):
        # Same register, so retrieval returns something plausible and the
        # model has to decline on the merits rather than on the shape of
        # the question. Compared against the real distribution instead of
        # a bound I picked: the first version asserted "more than four
        # words" and "when are salaries reviewed" has exactly four.
        answerable = [len(q.query.split()) for q in QUESTIONS]
        shortest = min(answerable)
        for query in UNANSWERABLE:
            self.assertGreaterEqual(len(query.split()), shortest)
            self.assertFalse(query.endswith("?"))

    def test_each_contradiction_states_something_the_handbook_denies(self):
        text = " ".join(document.text for document in DOCUMENTS)
        for _, false_sentence, false_answer in CONTRADICTIONS:
            self.assertNotIn(false_answer, text)

    def test_each_contradiction_targets_a_question_the_handbook_answers(self):
        text = " ".join(document.text.lower() for document in DOCUMENTS)
        for query, _, _ in CONTRADICTIONS:
            self.assertTrue(any(word in text for word in query.lower().split()
                                if len(word) > 4))

    def test_a_poisoned_chunk_is_a_chunk(self):
        chunk = poisoned_chunk("Core hours are 03:00 to 04:00.")
        self.assertIn("03:00", chunk.text)
        self.assertIn(chunk.text, pack([chunk]))


@needs_generator
class TestCounterfactualFaithfulness(unittest.TestCase):
    """Does the answer depend on the evidence it cites?"""

    @classmethod
    def setUpClass(cls):
        cls.report = counterfactual_faithfulness(_GEN, _BI and HybridRetriever(
            CHUNKS, DenseIndex(CHUNKS, model=_BI)), list(QUESTIONS)[:12], k=3)

    def test_the_outcomes_sum_to_one(self):
        names = ("abstained_without", "changed_without", "unchanged_without",
                 "no_support")
        self.assertAlmostEqual(sum(self.report[name] for name in names), 1.0,
                               places=6)

    def test_removing_the_evidence_changes_something(self):
        # If nothing ever changed, the context would not be in use at all.
        moved = self.report["abstained_without"] + self.report["changed_without"]
        self.assertGreater(moved, 0.0)

    def test_it_needs_no_labels(self):
        # The probe compares the system against itself, so it works on a
        # corpus with no gold answers at all.
        report = counterfactual_faithfulness(
            _GEN, HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI)),
            list(QUESTIONS)[:4], k=2)
        self.assertEqual(report["total"], 4)


@needs_generator
class TestPoisoning(unittest.TestCase):
    """When the context is wrong, which source wins?"""

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI))
        cls.report = poisoning_report(_GEN, cls.retriever, k=3)

    def test_the_verdicts_sum_to_one(self):
        self.assertAlmostEqual(
            self.report["followed"] + self.report["overruled"] + self.report["other"],
            1.0, places=6)

    def test_one_row_per_contradiction(self):
        self.assertEqual(len(self.report["rows"]), len(CONTRADICTIONS))

    def test_every_row_records_what_happened(self):
        for row in self.report["rows"]:
            self.assertIn(row["verdict"], {"followed the context",
                                           "kept the true fact", "abstained"})

    def test_the_false_fact_really_is_in_the_context(self):
        # Otherwise the probe measures nothing.
        query, false_sentence, false_answer = CONTRADICTIONS[0]
        retrieved = [chunk for chunk, _ in
                     self.retriever.search(query, k=3, method="dense")]
        context = pack([poisoned_chunk(false_sentence)] + retrieved)
        self.assertIn(false_answer, context)


@needs_generator
class TestAbstention(unittest.TestCase):
    """Two populations with opposite signs."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI))
        cls.report = abstention_report(_GEN, cls.retriever,
                                       list(QUESTIONS)[:12], k=3)

    def test_it_counts_both_populations(self):
        self.assertGreater(self.report["answerable"], 0)
        self.assertEqual(self.report["unanswerable"], len(UNANSWERABLE))

    def test_both_rates_are_fractions(self):
        for name in ("wrongly_abstained", "wrongly_answered"):
            self.assertGreaterEqual(self.report[name], 0.0)
            self.assertLessEqual(self.report[name], 1.0)

    def test_the_answerable_population_is_genuinely_answerable(self):
        # By construction: only questions whose gold answer was retrieved.
        questions = list(QUESTIONS)[:12]
        counted = 0
        for question in questions:
            chunks = [c for c, _ in
                      self.retriever.search(question.query, k=3, method="dense")]
            if any(question.answer in chunk.text for chunk in chunks):
                counted += 1
        self.assertEqual(counted, self.report["answerable"])

    def test_an_unanswerable_question_retrieves_something_anyway(self):
        # The retriever always returns k chunks, so declining has to be
        # the model's decision rather than an empty context.
        chunks = [c for c, _ in
                  self.retriever.search(UNANSWERABLE[0], k=3, method="dense")]
        self.assertEqual(len(chunks), 3)
        self.assertTrue(pack(chunks).strip())


if __name__ == "__main__":
    unittest.main()
