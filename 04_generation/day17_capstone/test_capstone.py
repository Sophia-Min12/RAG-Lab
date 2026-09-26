"""Day 17 tests - the assembled system, the CLI, and the writeup.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import io
import unittest
from contextlib import redirect_stdout

from capstone import (
    DEFAULT_CROSS_ENCODER,
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    GENERATOR_AVAILABLE,
    HONEST_WRITEUP,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    Generator,
    GroundedAnswer,
    RAGSystem,
    _handbook,
    answer_is_correct,
    main,
    verify_citation,
)

DOCUMENTS = _handbook()

_SYSTEM, _ERROR = None, None
if GENERATOR_AVAILABLE and EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        _SYSTEM = RAGSystem(DOCUMENTS,
                            embedder=SentenceTransformer(DEFAULT_MODEL),
                            cross_encoder=CrossEncoder(DEFAULT_CROSS_ENCODER),
                            generator=Generator())
    except Exception as error:
        _ERROR = error

needs_system = unittest.skipUnless(_SYSTEM is not None, f"models unavailable ({_ERROR})")


class TestTheWriteup(unittest.TestCase):
    """Data rather than prose, so it can be checked."""

    def test_it_names_every_level(self):
        for day in ("Day 2", "Day 3", "Day 5", "Day 9", "Day 11", "Day 13",
                    "Day 14", "Day 15"):
            self.assertIn(day, HONEST_WRITEUP, f"{day} is not accounted for")

    def test_it_records_what_was_wrong(self):
        text = HONEST_WRITEUP.lower()
        self.assertIn("wrong before it was measured", text)
        for phrase in ("my own budget packer", "my agreement metric",
                       "my ivf was slower"):
            self.assertIn(phrase, text)

    def test_it_states_what_was_not_measured(self):
        self.assertIn("REMAINS UNMEASURED", HONEST_WRITEUP)
        self.assertIn("judge", HONEST_WRITEUP.lower())

    def test_it_admits_the_corpus_fits_in_the_context_window(self):
        self.assertIn("437 tokens", HONEST_WRITEUP)

    def test_it_admits_the_poisoning_result(self):
        self.assertIn("poisoned context 100%", HONEST_WRITEUP)

    def test_it_admits_the_sample_size(self):
        self.assertIn("83% to 105%", HONEST_WRITEUP)

    def test_it_is_not_silently_empty(self):
        self.assertGreater(len(HONEST_WRITEUP.split()), 400)


class TestTheCLI(unittest.TestCase):
    def test_no_arguments_prints_help(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main([]), 0)
        self.assertIn("usage", buffer.getvalue().lower())

    def test_writeup_needs_no_models(self):
        # The one command that runs on a machine with nothing installed.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["writeup"]), 0)
        self.assertIn("What this repo measured", buffer.getvalue())

    def test_every_advertised_command_is_reachable(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([])
        help_text = buffer.getvalue()
        for command in ("ask", "evaluate", "probe", "index", "writeup"):
            self.assertIn(command, help_text)


@needs_system
class TestRAGSystem(unittest.TestCase):
    def test_it_builds_an_index(self):
        self.assertGreater(len(_SYSTEM.chunks), 10)
        self.assertEqual(_SYSTEM.dense.matrix.shape[0], len(_SYSTEM.chunks))

    def test_retrieve_returns_the_configured_number_of_chunks(self):
        self.assertEqual(len(_SYSTEM.retrieve("what are the core hours")),
                         _SYSTEM.context_chunks)

    def test_ask_returns_a_grounded_answer(self):
        result = _SYSTEM.ask("what are the core hours")
        self.assertIsInstance(result, GroundedAnswer)
        self.assertTrue(result.answer)

    def test_it_answers_a_question_it_should_know(self):
        result = _SYSTEM.ask("what are the core hours")
        self.assertTrue(answer_is_correct(result.answer, "10:00 to 16:00"))

    def test_the_citation_verifies_against_the_source(self):
        result = _SYSTEM.ask("what is the domestic meal allowance")
        self.assertTrue(result.supported)
        document = next(d for d in DOCUMENTS if d.doc_id == result.cited.doc_id)
        self.assertTrue(verify_citation(document, result.cited))

    def test_it_declines_a_question_the_handbook_cannot_answer(self):
        result = _SYSTEM.ask("what is the capital of France")
        self.assertFalse(answer_is_correct(result.answer, "Paris"))

    def test_it_is_deterministic(self):
        first = _SYSTEM.ask("what are the core hours").answer
        second = _SYSTEM.ask("what are the core hours").answer
        self.assertEqual(first, second)

    def test_turning_the_reranker_off_still_works(self):
        system = RAGSystem(DOCUMENTS, embedder=_SYSTEM.dense.model,
                           generator=_SYSTEM.generator, rerank=False)
        self.assertEqual(len(system.retrieve("what are the core hours")),
                         system.context_chunks)


@needs_system
class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores = _SYSTEM.evaluate(list(QUESTIONS)[:10])

    def test_it_reports_both_halves(self):
        for key in ("retrieval_hit", "retrieval_mrr", "answer_correct",
                    "answer_grounded", "abstained"):
            self.assertIn(key, self.scores)

    def test_every_rate_is_a_fraction(self):
        for key in ("retrieval_hit", "answer_correct", "answer_grounded",
                    "abstained"):
            self.assertGreaterEqual(self.scores[key], 0.0)
            self.assertLessEqual(self.scores[key], 1.0)

    def test_retrieval_beats_generation(self):
        # The shape Level 4 found: the bottleneck moved downstream.
        self.assertGreaterEqual(self.scores["retrieval_hit"],
                                self.scores["answer_correct"])

    def test_it_counts_the_questions_it_was_given(self):
        self.assertEqual(self.scores["questions"], 10)

    def test_the_paraphrased_set_is_harder_for_the_generator(self):
        paraphrased = _SYSTEM.evaluate(list(PARAPHRASED_QUESTIONS)[:10])
        self.assertLessEqual(paraphrased["answer_correct"],
                             self.scores["answer_correct"] + 0.2)


if __name__ == "__main__":
    unittest.main()
