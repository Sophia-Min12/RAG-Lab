"""Day 12 tests - generated sets, resolving power, and the pooling bias.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from eval_sets import (
    DEFAULT_MODEL,
    EMBEDDINGS_AVAILABLE,
    PARAPHRASED_QUESTIONS,
    QUESTIONS,
    DenseIndex,
    HybridRetriever,
    _handbook,
    agreement,
    build_index,
    evaluate_all,
    extract_facts,
    generate_cloze_questions,
    generate_known_item_questions,
    pool_coverage,
    pooled_judgements,
    resolving_power,
    vocabulary_overlap_of_set,
)

DOCUMENTS = _handbook()
CHUNKS = build_index(DOCUMENTS, 80, "sentence")

_MODEL, _ERROR = None, None
if EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(DEFAULT_MODEL)
    except Exception as error:
        _ERROR = error

needs_model = unittest.skipUnless(_MODEL is not None, f"model unavailable ({_ERROR})")


class TestFactExtraction(unittest.TestCase):
    def test_it_finds_an_amount(self):
        self.assertEqual(extract_facts("Meals are reimbursed up to 35,000 won per day."),
                         ["35,000 won"])

    def test_it_finds_a_time(self):
        self.assertIn("10:00", extract_facts("Core hours are 10:00 to 16:00."))

    def test_it_finds_a_spelled_out_duration(self):
        self.assertIn("three year", " ".join(
            extract_facts("Laptops are replaced on a three year cycle.")))

    def test_a_sentence_with_no_facts_yields_nothing(self):
        self.assertEqual(extract_facts("Every engineer is issued a laptop."), [])


class TestGeneratedSets(unittest.TestCase):
    def test_known_item_queries_are_their_own_answers(self):
        for question in generate_known_item_questions(DOCUMENTS):
            self.assertEqual(question.query, question.answer)

    def test_cloze_removes_the_answer_from_the_query(self):
        for question in generate_cloze_questions(DOCUMENTS):
            self.assertNotIn(question.answer, question.query)
            self.assertIn("____", question.query)

    def test_cloze_answers_are_still_in_their_documents(self):
        by_id = {document.doc_id: document for document in DOCUMENTS}
        for question in generate_cloze_questions(DOCUMENTS):
            self.assertIn(question.answer, by_id[question.doc_id].text)

    def test_both_generators_produce_something(self):
        self.assertGreater(len(generate_known_item_questions(DOCUMENTS)), 10)
        self.assertGreater(len(generate_cloze_questions(DOCUMENTS)), 10)

    def test_the_limit_is_honoured(self):
        self.assertEqual(len(generate_cloze_questions(DOCUMENTS, limit=3)), 3)

    def test_generated_sets_share_far_more_vocabulary_than_hand_written_ones(self):
        # The number that predicts they will separate nothing.
        hand = vocabulary_overlap_of_set(QUESTIONS, DOCUMENTS)
        cloze = vocabulary_overlap_of_set(generate_cloze_questions(DOCUMENTS), DOCUMENTS)
        self.assertGreater(cloze, hand + 0.2)


class TestResolvingPower(unittest.TestCase):
    def test_all_tied_resolves_nothing(self):
        self.assertEqual(resolving_power({"a": 1.0, "b": 1.0, "c": 1.0}), 0.0)

    def test_all_distinct_resolves_everything(self):
        self.assertEqual(resolving_power({"a": 1.0, "b": 0.5, "c": 0.1}), 1.0)

    def test_partial(self):
        self.assertAlmostEqual(resolving_power({"a": 1.0, "b": 1.0, "c": 0.5}), 2 / 3)

    def test_a_single_system_resolves_nothing(self):
        self.assertEqual(resolving_power({"a": 1.0}), 0.0)


class TestAgreement(unittest.TestCase):
    """Including the mistake the first version made."""

    def test_identical_orderings_agree(self):
        first = {"a": 1.0, "b": 0.5}
        self.assertEqual(agreement(first, {"a": 0.9, "b": 0.1})["rate"], 1.0)

    def test_opposite_orderings_disagree(self):
        first = {"a": 1.0, "b": 0.5}
        self.assertEqual(agreement(first, {"a": 0.1, "b": 0.9})["rate"], 0.0)

    def test_a_tie_is_unresolved_not_agreement(self):
        # The bug: a set with no opinion used to report agreement.
        first = {"a": 1.0, "b": 0.5}
        verdict = agreement(first, {"a": 1.0, "b": 1.0})
        self.assertEqual(verdict["unresolved"], 1)
        self.assertEqual(verdict["agreed"], 0)
        self.assertEqual(verdict["disagreed"], 0)

    def test_a_fully_tied_second_set_resolves_no_pairs(self):
        first = {"a": 1.0, "b": 0.5, "c": 0.2}
        verdict = agreement(first, {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(verdict["unresolved"], 3)
        self.assertNotEqual(verdict["rate"], verdict["rate"])  # nan

    def test_counts_add_up_to_the_number_of_pairs(self):
        first = {"a": 1.0, "b": 0.5, "c": 0.2, "d": 0.1}
        second = {"a": 0.9, "b": 0.9, "c": 0.4, "d": 0.0}
        verdict = agreement(first, second)
        total = verdict["agreed"] + verdict["disagreed"] + verdict["unresolved"]
        self.assertEqual(total, 6)


@needs_model
class TestGeneratedSetsAreUseless(unittest.TestCase):
    """The day's first finding, end to end."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_MODEL))
        cls.methods = {"BM25": {"method": "sparse"}, "dense": {"method": "dense"},
                       "RRF": {"method": "rrf"}}

    def scores(self, questions):
        return {name: evaluate_all(self.retriever, questions, k=3, depth=3,
                                   **kwargs)["mrr"]
                for name, kwargs in self.methods.items()}

    def test_the_cloze_set_separates_nothing(self):
        self.assertEqual(resolving_power(self.scores(
            generate_cloze_questions(DOCUMENTS))), 0.0)

    def test_the_known_item_set_separates_nothing(self):
        self.assertEqual(resolving_power(self.scores(
            generate_known_item_questions(DOCUMENTS))), 0.0)

    def test_the_hand_written_set_does(self):
        self.assertGreater(resolving_power(self.scores(QUESTIONS)), 0.5)

    def test_every_system_scores_perfectly_on_the_generated_sets(self):
        for value in self.scores(generate_cloze_questions(DOCUMENTS)).values():
            self.assertAlmostEqual(value, 1.0, places=6)


@needs_model
class TestPooling(unittest.TestCase):
    """The day's second finding: what pooling costs, and who pays."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(CHUNKS, DenseIndex(CHUNKS, model=_MODEL))
        cls.systems = {
            "BM25": lambda q, d: cls.retriever.search(q, k=d, method="sparse"),
            "dense": lambda q, d: cls.retriever.search(q, k=d, method="dense"),
            "RRF": lambda q, d: cls.retriever.search(q, k=d, method="rrf"),
        }
        cls.questions = list(QUESTIONS) + list(PARAPHRASED_QUESTIONS)

    def test_a_deeper_pool_judges_more_chunks(self):
        sizes = []
        for depth in (1, 3, 5):
            pool = pooled_judgements(self.systems, self.questions[0].query, depth=depth)
            sizes.append(len(pool))
        self.assertEqual(sizes, sorted(sizes))

    def test_the_pool_is_smaller_than_the_collection(self):
        # The entire point: judge a few instead of all.
        pool = pooled_judgements(self.systems, self.questions[0].query, depth=5)
        self.assertLess(len(pool), len(CHUNKS))

    def test_a_deep_enough_pool_covers_every_answer(self):
        covered = sum(pool_coverage(pooled_judgements(self.systems, q.query, depth=10), q)
                      for q in self.questions)
        self.assertEqual(covered, len(self.questions))

    def test_the_pool_records_which_systems_contributed(self):
        pool = pooled_judgements(self.systems, self.questions[0].query, depth=5)
        for _, contributors in pool.values():
            self.assertTrue(contributors <= set(self.systems))
            self.assertTrue(contributors)

    def test_excluding_dense_from_the_pool_penalises_it(self):
        self.assertGreater(self.penalty("dense"), 0.15)

    def test_excluding_bm25_from_the_pool_costs_it_nothing(self):
        # Because it finds nothing the others miss.
        self.assertAlmostEqual(self.penalty("BM25"), 0.0, places=6)

    def penalty(self, excluded):
        contributors = {name: search for name, search in self.systems.items()
                        if name != excluded}
        by_pool = by_truth = 0
        for question in self.questions:
            pool = pooled_judgements(contributors, question.query, depth=5)
            judged = {id(chunk) for chunk, _ in pool.values()
                      if question.answer in chunk.text}
            results = self.systems[excluded](question.query, 3)
            if any(question.answer in chunk.text and id(chunk) in judged
                   for chunk, _ in results):
                by_pool += 1
            if any(question.answer in chunk.text for chunk, _ in results):
                by_truth += 1
        return (by_truth - by_pool) / len(self.questions)


if __name__ == "__main__":
    unittest.main()
