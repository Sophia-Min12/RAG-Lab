"""Day 15 tests - grounding, citation, and two flaws in the scoring.

Grounding is string work and needs no model. Only the end-to-end report
does.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from grounding import (
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    GENERATOR_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    YES_NO_MISMATCH,
    DenseIndex,
    Generator,
    GroundedAnswer,
    HybridRetriever,
    _handbook,
    _text_chunk,
    answer_is_correct,
    answer_with_citation,
    build_index,
    find_supporting_chunk,
    ground,
    grounding_report,
    normalize_answer,
    terse_but_right,
    verify_citation,
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


class TestFindSupportingChunk(unittest.TestCase):
    def test_it_finds_the_chunk_containing_the_answer(self):
        chunk = find_supporting_chunk("10:00 to 16:00", CHUNKS)
        self.assertIsNotNone(chunk)
        self.assertIn("10:00 to 16:00", chunk.text)

    def test_it_returns_none_when_nothing_contains_it(self):
        self.assertIsNone(find_supporting_chunk("the capital of France", CHUNKS))

    def test_it_ignores_case_and_punctuation(self):
        self.assertIsNotNone(find_supporting_chunk("10:00 TO 16:00.", CHUNKS))

    def test_an_empty_answer_supports_nothing(self):
        self.assertIsNone(find_supporting_chunk("", CHUNKS))

    def test_it_cannot_recognise_a_paraphrase(self):
        # The stated limitation: literal containment cannot hallucinate,
        # and cannot see that "ten in the morning until four" is the same
        # fact as "10:00 to 16:00".
        self.assertIsNone(find_supporting_chunk("ten in the morning until four",
                                                CHUNKS))


class TestGround(unittest.TestCase):
    def test_a_supported_answer_is_grounded(self):
        result = ground("q", "10:00 to 16:00", CHUNKS, DOCUMENTS)
        self.assertTrue(result.supported)
        self.assertIsNotNone(result.cited)

    def test_an_unsupported_answer_is_not(self):
        result = ground("q", "the capital of France is Paris", CHUNKS, DOCUMENTS)
        self.assertFalse(result.supported)
        self.assertIsNone(result.cited)

    def test_an_abstention_is_not_grounded(self):
        result = ground("q", "not stated", CHUNKS, DOCUMENTS)
        self.assertFalse(result.supported)
        self.assertTrue(result.abstained)

    def test_it_attaches_a_citation_with_a_line_number(self):
        result = ground("q", "10:00 to 16:00", CHUNKS, DOCUMENTS)
        self.assertIsNotNone(result.citation)
        self.assertGreaterEqual(result.citation.line, 1)
        self.assertTrue(result.citation.uri)

    def test_the_citation_verifies_against_the_original_document(self):
        # Day 2's invariant, carried all the way to a generated answer.
        result = ground("q", "10:00 to 16:00", CHUNKS, DOCUMENTS)
        document = next(d for d in DOCUMENTS if d.doc_id == result.cited.doc_id)
        self.assertTrue(verify_citation(document, result.cited))

    def test_without_documents_there_is_no_citation_but_still_grounding(self):
        result = ground("q", "10:00 to 16:00", CHUNKS)
        self.assertTrue(result.supported)
        self.assertIsNone(result.citation)


class TestTheScorerIsAsymmetric(unittest.TestCase):
    """Generous about additions, strict about omissions."""

    def test_a_terse_answer_scores_wrong(self):
        self.assertFalse(answer_is_correct("1.75", "1.75 days per month"))

    def test_and_terse_but_right_recognises_it(self):
        self.assertTrue(terse_but_right("1.75", "1.75 days per month"))

    def test_a_padded_answer_scores_right(self):
        # Including when the padding is false.
        self.assertTrue(answer_is_correct(
            "1.75 days per month, and Berlin is in France", "1.75 days per month"))

    def test_a_genuinely_wrong_answer_is_neither(self):
        self.assertFalse(answer_is_correct("carried into the next calendar year",
                                           "forfeited on 31 December"))
        self.assertFalse(terse_but_right("carried into the next calendar year",
                                         "forfeited on 31 December"))

    def test_an_empty_answer_is_not_terse_but_right(self):
        self.assertFalse(terse_but_right("", "1.75 days per month"))


class TestTheEvaluationSetBug(unittest.TestCase):
    """Documented rather than repaired."""

    def test_the_listed_questions_exist_in_the_paraphrased_set(self):
        queries = {question.query for question in PARAPHRASED_QUESTIONS}
        for query in YES_NO_MISMATCH:
            self.assertIn(query, queries)

    def test_they_are_phrased_as_yes_no_questions(self):
        starts = ("do ", "is ", "can ", "are ", "does ", "will ", "am i")
        for query in YES_NO_MISMATCH:
            self.assertTrue(query.lower().startswith(starts), query)

    def test_but_are_graded_against_an_extractive_span(self):
        by_query = {question.query: question for question in PARAPHRASED_QUESTIONS}
        for query in YES_NO_MISMATCH:
            gold = by_query[query].answer
            self.assertNotIn(gold.lower(), {"yes", "no"})
            self.assertFalse(answer_is_correct("yes", gold))

    def test_the_gold_spans_are_still_correct_for_retrieval(self):
        # Why Days 6-13 are unaffected: those asked whether the span was
        # retrieved, and it is still the right span to look for.
        by_id = {document.doc_id: document for document in DOCUMENTS}
        by_query = {question.query: question for question in PARAPHRASED_QUESTIONS}
        for query in YES_NO_MISMATCH:
            question = by_query[query]
            self.assertIn(question.answer, by_id[question.doc_id].text)

    def test_only_a_handful_are_affected(self):
        self.assertLessEqual(len(YES_NO_MISMATCH), 3)


@needs_generator
class TestGroundingReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI))
        cls.questions = list(QUESTIONS)[:10]
        cls.report = grounding_report(_GEN, cls.retriever, cls.questions,
                                      DOCUMENTS, k=3)

    def test_the_cells_sum_to_one(self):
        cells = ("grounded_correct", "grounded_wrong", "ungrounded_correct",
                 "ungrounded_wrong", "abstained")
        self.assertAlmostEqual(sum(self.report[name] for name in cells), 1.0, places=6)

    def test_most_answers_are_grounded(self):
        grounded = self.report["grounded_correct"] + self.report["grounded_wrong"]
        self.assertGreater(grounded, 0.5)

    def test_every_grounded_citation_verifies(self):
        grounded = round((self.report["grounded_correct"]
                          + self.report["grounded_wrong"]) * self.report["total"])
        self.assertEqual(self.report["citations_verified"], grounded)

    def test_grounded_and_wrong_is_not_empty(self):
        # The cell the day exists for: accurate citations on wrong
        # answers. If this were ever 0 the claim would need revisiting.
        self.assertGreater(self.report["grounded_wrong"], 0.0)

    def test_an_end_to_end_answer_carries_a_citation(self):
        result = answer_with_citation(_GEN, self.retriever,
                                      "what are the core hours", DOCUMENTS, k=3)
        self.assertIsInstance(result, GroundedAnswer)
        if result.supported:
            self.assertIsNotNone(result.citation)


if __name__ == "__main__":
    unittest.main()
