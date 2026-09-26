"""Day 14 tests - packing, scoring, and the generator.

Packing and answer-scoring need no model; only the position experiment
does, and it is the slow one.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from packing import (
    DEFAULT_GENERATOR,
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    GENERATOR_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    DenseIndex,
    Generator,
    HybridRetriever,
    _handbook,
    _text_chunk,
    abstained,
    answer_is_correct,
    build_index,
    normalize_answer,
    pack,
    pack_with_answer_at,
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


class TestPack(unittest.TestCase):
    def setUp(self):
        self.chunks = [_text_chunk(letter) for letter in "abcde"]

    def test_relevance_order_is_the_order_given(self):
        self.assertEqual(pack(self.chunks, separator="|"), "a|b|c|d|e")

    def test_reversed_order(self):
        self.assertEqual(pack(self.chunks, order="reversed", separator="|"), "e|d|c|b|a")

    def test_edges_puts_the_best_first_and_the_second_last(self):
        packed = pack(self.chunks, order="edges", separator="|").split("|")
        self.assertEqual(packed[0], "a")
        self.assertEqual(packed[-1], "b")

    def test_every_order_keeps_every_chunk(self):
        for order in ("relevance", "reversed", "edges"):
            packed = pack(self.chunks, order=order, separator="|").split("|")
            self.assertEqual(sorted(packed), sorted("abcde"))

    def test_packing_nothing_gives_an_empty_string(self):
        self.assertEqual(pack([]), "")

    def test_an_unknown_order_is_refused(self):
        with self.assertRaises(ValueError):
            pack(self.chunks, order="alphabetical")


class TestPackWithAnswerAt(unittest.TestCase):
    def setUp(self):
        self.chunks = [_text_chunk(letter) for letter in "abcde"]
        self.gold = self.chunks[2]

    def test_position_zero_puts_it_first(self):
        packed = pack_with_answer_at(self.chunks, self.gold, 0, separator="|")
        self.assertTrue(packed.startswith("c"))

    def test_the_last_position_puts_it_last(self):
        packed = pack_with_answer_at(self.chunks, self.gold, 4, separator="|")
        self.assertTrue(packed.endswith("c"))

    def test_the_context_holds_the_same_text_at_every_position(self):
        # The property that makes the experiment an experiment.
        contents = set()
        for position in range(5):
            packed = pack_with_answer_at(self.chunks, self.gold, position, separator="|")
            contents.add("".join(sorted(packed.split("|"))))
        self.assertEqual(len(contents), 1)

    def test_the_answer_appears_exactly_once(self):
        for position in range(5):
            packed = pack_with_answer_at(self.chunks, self.gold, position, separator="|")
            self.assertEqual(packed.split("|").count("c"), 1)

    def test_a_position_past_the_end_is_clamped(self):
        packed = pack_with_answer_at(self.chunks, self.gold, 99, separator="|")
        self.assertTrue(packed.endswith("c"))


class TestAnswerScoring(unittest.TestCase):
    def test_normalize_strips_articles_and_punctuation(self):
        self.assertEqual(normalize_answer("The 10:00 to 16:00."), "10:00 to 16:00")

    def test_normalize_is_case_insensitive(self):
        self.assertEqual(normalize_answer("Three Year Cycle"), "three year cycle")

    def test_an_exact_answer_is_correct(self):
        self.assertTrue(answer_is_correct("10:00 to 16:00", "10:00 to 16:00"))

    def test_an_answer_in_a_sentence_is_correct(self):
        self.assertTrue(answer_is_correct("The core hours are 10:00 to 16:00.",
                                          "10:00 to 16:00"))

    def test_a_wrong_answer_is_not(self):
        self.assertFalse(answer_is_correct("a three year cycle", "10:00 to 16:00"))

    def test_containment_cannot_catch_a_correct_answer_that_also_lies(self):
        # Stated as a limitation rather than discovered as a surprise:
        # this scorer is generous, and Day 16 is where that starts to
        # matter.
        self.assertTrue(answer_is_correct(
            "10:00 to 16:00, and the capital of France is Berlin", "10:00 to 16:00"))

    def test_abstention_is_detected(self):
        self.assertTrue(abstained("not stated"))
        self.assertTrue(abstained("The answer is Not Stated."))

    def test_a_real_answer_is_not_an_abstention(self):
        self.assertFalse(abstained("10:00 to 16:00"))


@needs_generator
class TestGenerator(unittest.TestCase):
    def test_it_answers_from_the_context(self):
        context = "Core hours are 10:00 to 16:00."
        self.assertTrue(answer_is_correct(
            _GEN.answer("what are the core hours", context), "10:00 to 16:00"))

    def test_it_abstains_when_the_context_cannot_answer(self):
        context = "Laptops are replaced on a three year cycle."
        self.assertTrue(abstained(_GEN.answer("what is the capital of France", context)))

    def test_it_is_deterministic(self):
        context = "Core hours are 10:00 to 16:00."
        first = _GEN.answer("what are the core hours", context)
        second = _GEN.answer("what are the core hours", context)
        self.assertEqual(first, second)

    def test_prompt_tokens_grows_with_the_context(self):
        short = _GEN.prompt_tokens("q", "one sentence.")
        long = _GEN.prompt_tokens("q", "one sentence. " * 50)
        self.assertGreater(long, short)

    def test_the_whole_corpus_fits_in_the_context_window(self):
        # Worth knowing before claiming retrieval was necessary: all 26
        # chunks come to 437 tokens against a 512-token limit, so this
        # corpus could be passed entire. The demo measures what happens
        # when it is - 56%, against 65% for a dense top-2 - which makes
        # the retriever a distractor filter rather than a finder here.
        whole = " ".join(chunk.text for chunk in CHUNKS)
        self.assertLess(_GEN.prompt_tokens("q", whole), _GEN.max_input_tokens)

    def test_a_long_enough_context_would_exceed_it(self):
        huge = " ".join(chunk.text for chunk in CHUNKS) * 4
        self.assertGreater(_GEN.prompt_tokens("q", huge), _GEN.max_input_tokens)


@needs_generator
class TestPositionMatters(unittest.TestCase):
    """The day's measurement, on a subset small enough for a test suite."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_BI))
        cls.questions = list(QUESTIONS)[:8]

    def accuracy_at(self, position):
        correct = 0
        for question in self.questions:
            gold = next((c for c in CHUNKS if question.answer in c.text), None)
            if gold is None:
                continue
            pool = [c for c, _ in
                    self.retriever.search(question.query, k=8, method="dense")][:8]
            if gold not in pool:
                pool = pool[:7] + [gold]
            answer = _GEN.answer(question.query,
                                 pack_with_answer_at(pool, gold, position))
            correct += answer_is_correct(answer, question.answer)
        return correct / len(self.questions)

    def test_the_answer_is_findable_at_the_front(self):
        self.assertGreater(self.accuracy_at(0), 0.4)

    def test_position_is_measured_but_not_asserted_at_this_sample_size(self):
        # No inequality here on purpose. The demo measures a nine-point
        # first-position advantage over 34 questions; on the eight this
        # test can afford, nine points is 0.7 of a question and both
        # positions came out at 0.75. Asserting a difference would be
        # asserting noise - the same lesson Day 11 spent a section on.
        self.assertIsInstance(self.accuracy_at(0), float)
        self.assertIsInstance(self.accuracy_at(4), float)

    def test_the_front_is_not_worse_than_the_middle(self):
        self.assertGreaterEqual(self.accuracy_at(0), self.accuracy_at(4))


if __name__ == "__main__":
    unittest.main()
