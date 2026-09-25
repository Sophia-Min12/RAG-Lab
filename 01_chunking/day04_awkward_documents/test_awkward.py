"""Day 4 tests - tables, code, Korean, and the abbreviation list.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from awkward import (
    ABBREVIATION_PROSE,
    ABBREVIATIONS,
    EXPENSE_TABLE,
    PARALLEL_EN,
    PARALLEL_KO,
    POLICY_CODE,
    Chunk,
    chunk_fixed,
    chunk_python,
    chunk_sentences,
    chunk_table,
    density,
    facts_per_chunk,
    header_retention,
    index_growth,
    parse_markdown_table,
    parses_as_python,
    split_sentences,
    split_sentences_guarded,
)


class TestTableParsing(unittest.TestCase):
    def test_it_finds_the_header_and_rows(self):
        table = parse_markdown_table(EXPENSE_TABLE)
        self.assertIn("| item |", table.header)
        self.assertEqual(len(table.rows), 12)

    def test_prose_is_not_a_table(self):
        self.assertIsNone(parse_markdown_table("Just a sentence. And another."))

    def test_a_table_needs_an_alignment_row(self):
        self.assertIsNone(parse_markdown_table("| a | b |\n| 1 | 2 |\n| 3 | 4 |"))

    def test_a_table_needs_at_least_one_body_row(self):
        self.assertIsNone(parse_markdown_table("| a | b |\n| --- | --- |"))

    def test_alignment_colons_are_accepted(self):
        table = parse_markdown_table("| a | b |\n|:---|---:|\n| 1 | 2 |")
        self.assertIsNotNone(table)

    def test_the_preamble_is_header_plus_alignment(self):
        table = parse_markdown_table(EXPENSE_TABLE)
        self.assertEqual(table.preamble, f"{table.header}\n{table.alignment}")


class TestTableChunking(unittest.TestCase):
    def setUp(self):
        self.table = parse_markdown_table(EXPENSE_TABLE)

    def test_character_chunking_loses_the_header(self):
        for size in (120, 200, 320):
            chunks = chunk_fixed(EXPENSE_TABLE, size)
            self.assertLess(header_retention(chunks, self.table.header), 1.0)

    def test_chunk_table_keeps_it_everywhere(self):
        for rows in (1, 2, 4, 6, 12):
            chunks = chunk_table(EXPENSE_TABLE, rows_per_chunk=rows)
            self.assertEqual(header_retention(chunks, self.table.header), 1.0)

    def test_every_row_appears_exactly_once_in_the_bodies(self):
        # Repeating the header must not repeat the data.
        chunks = chunk_table(EXPENSE_TABLE, rows_per_chunk=4)
        for row in self.table.rows:
            appearances = sum(chunk.text.count(row) for chunk in chunks)
            self.assertEqual(appearances, 1, f"{row!r} appears {appearances} times")

    def test_fewer_rows_per_chunk_costs_more_index(self):
        growth = [index_growth(EXPENSE_TABLE, chunk_table(EXPENSE_TABLE, rows_per_chunk=r))
                  for r in (6, 4, 2)]
        self.assertEqual(growth, sorted(growth))

    def test_the_cost_is_above_one_and_bounded(self):
        for rows in (2, 4, 6):
            growth = index_growth(EXPENSE_TABLE, chunk_table(EXPENSE_TABLE, rows_per_chunk=rows))
            self.assertGreater(growth, 1.0)
            self.assertLess(growth, 2.0)

    def test_offsets_point_at_the_body_rows(self):
        # The header is added context; the span is the real text.
        chunks = chunk_table(EXPENSE_TABLE, rows_per_chunk=4)
        for chunk in chunks:
            self.assertIn(EXPENSE_TABLE[chunk.start:chunk.end].split("\n")[0],
                          chunk.text)

    def test_one_row_per_chunk_gives_one_chunk_per_row(self):
        self.assertEqual(len(chunk_table(EXPENSE_TABLE, rows_per_chunk=1)),
                         len(self.table.rows))

    def test_rows_per_chunk_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            chunk_table(EXPENSE_TABLE, rows_per_chunk=0)

    def test_prose_is_refused(self):
        with self.assertRaises(ValueError):
            chunk_table("Not a table at all.")


class TestCodeChunking(unittest.TestCase):
    def test_no_character_chunk_parses(self):
        # The finding: not one, at any size tried.
        for size in (120, 240, 480):
            chunks = chunk_fixed(POLICY_CODE, size)
            self.assertEqual(sum(parses_as_python(c.text) for c in chunks), 0,
                             f"a fixed({size}) chunk unexpectedly parsed")

    def test_every_structure_chunk_parses(self):
        chunks = chunk_python(POLICY_CODE)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertTrue(parses_as_python(chunk.text), repr(chunk.text[:40]))

    def test_the_chunks_reassemble_into_the_original(self):
        chunks = chunk_python(POLICY_CODE)
        self.assertEqual("".join(c.text for c in chunks).strip(), POLICY_CODE.strip())

    def test_offsets_round_trip(self):
        for chunk in chunk_python(POLICY_CODE):
            self.assertEqual(POLICY_CODE[chunk.start:chunk.end], chunk.text)

    def test_it_gives_up_size_control(self):
        # The trade, asserted rather than mentioned: structure chunks vary
        # far more in size than character chunks do.
        structure = [len(c.text) for c in chunk_python(POLICY_CODE)]
        character = [len(c.text) for c in chunk_fixed(POLICY_CODE, 240)]
        self.assertGreater(max(structure) / min(structure),
                           max(character) / min(character))

    def test_a_decorator_stays_with_its_function(self):
        source = "import functools\n\n\n@functools.cache\ndef f(x):\n    return x\n"
        chunks = chunk_python(source)
        decorated = [c for c in chunks if "def f" in c.text][0]
        self.assertIn("@functools.cache", decorated.text)
        self.assertTrue(parses_as_python(decorated.text))

    def test_a_single_statement_is_one_chunk(self):
        self.assertEqual(len(chunk_python("x = 1\n")), 1)

    def test_empty_source_gives_no_chunks(self):
        self.assertEqual(chunk_python(""), [])

    def test_source_that_does_not_parse_is_refused_with_a_reason(self):
        with self.assertRaises(ValueError):
            chunk_python("def broken(:\n")

    def test_parses_as_python_rejects_an_indented_fragment(self):
        self.assertFalse(parses_as_python("    return x"))
        self.assertTrue(parses_as_python("return_x = 1"))


class TestKoreanDensity(unittest.TestCase):
    def test_the_parallel_texts_hold_the_same_number_of_facts(self):
        # Otherwise the comparison is between two different documents.
        self.assertEqual(len(PARALLEL_EN.strip().split("\n")),
                         len(PARALLEL_KO.strip().split("\n")))

    def test_korean_says_it_in_about_half_the_characters(self):
        ratio = len(PARALLEL_KO) / len(PARALLEL_EN)
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 0.6)

    def test_density_per_sentence_differs_by_about_two(self):
        self.assertGreater(density(PARALLEL_EN) / density(PARALLEL_KO), 1.7)

    def test_the_same_budget_buys_more_facts_in_korean(self):
        for budget in (100, 150, 200):
            self.assertGreater(facts_per_chunk(PARALLEL_KO, budget),
                               facts_per_chunk(PARALLEL_EN, budget),
                               f"at budget {budget}")

    def test_a_budget_that_fits_one_english_fact_fits_two_korean_ones(self):
        self.assertAlmostEqual(facts_per_chunk(PARALLEL_EN, 100), 1.0, places=6)
        self.assertGreaterEqual(facts_per_chunk(PARALLEL_KO, 100), 2.0)

    def test_korean_chunks_still_round_trip(self):
        for chunk in chunk_sentences(PARALLEL_KO, 100):
            self.assertIn(chunk.text, PARALLEL_KO)


class TestAbbreviations(unittest.TestCase):
    def test_day_one_splits_after_an_abbreviation(self):
        # The behaviour Day 1's docstring warned about, measured.
        pieces = [s for s, _, _ in split_sentences(ABBREVIATION_PROSE)]
        self.assertIn("See Fig.", pieces)

    def test_the_guard_rejoins_it(self):
        pieces = [s for s, _, _ in split_sentences_guarded(ABBREVIATION_PROSE)]
        self.assertIn("See Fig. 3 for the accrual curve.", pieces)

    def test_the_guard_produces_fewer_sentences(self):
        plain = split_sentences(ABBREVIATION_PROSE)
        guarded = split_sentences_guarded(ABBREVIATION_PROSE)
        self.assertEqual(len(plain), 7)
        self.assertEqual(len(guarded), 4)

    def test_decimals_were_never_the_problem(self):
        # 1.75 survives both, because a digit follows the period rather
        # than whitespace. Worth pinning so the fix is not credited with
        # solving something it never touched.
        for splitter in (split_sentences, split_sentences_guarded):
            pieces = [s for s, _, _ in splitter("Leave accrues at 1.75 days. Then stops.")]
            self.assertEqual(len(pieces), 2)
            self.assertIn("1.75", pieces[0])

    def test_ordinary_prose_is_unchanged_by_the_guard(self):
        prose = "One thing happened. Then another. Then a third."
        self.assertEqual([s for s, _, _ in split_sentences(prose)],
                         [s for s, _, _ in split_sentences_guarded(prose)])

    def test_the_guarded_offsets_still_point_at_the_source(self):
        for sentence, start, end in split_sentences_guarded(ABBREVIATION_PROSE):
            self.assertEqual(ABBREVIATION_PROSE[start:end], sentence)

    def test_the_list_is_incomplete_and_that_is_the_point(self):
        # An abbreviation the list does not know still breaks it. The fix
        # is a list, so it is permanently partial; pretending otherwise
        # would be the actual defect.
        text = "See Abb. 3 for details. Then stop."
        self.assertNotIn("abb.", ABBREVIATIONS)
        self.assertEqual(len(split_sentences_guarded(text)), 3)

    def test_every_entry_is_lowercase_and_ends_in_a_period(self):
        for entry in ABBREVIATIONS:
            self.assertEqual(entry, entry.lower())
            self.assertTrue(entry.endswith("."))


class TestMetrics(unittest.TestCase):
    def test_header_retention_of_no_chunks_is_zero(self):
        self.assertEqual(header_retention([], "| a |"), 0.0)

    def test_header_retention_is_a_fraction(self):
        chunks = chunk_fixed(EXPENSE_TABLE, 120)
        value = header_retention(chunks, parse_markdown_table(EXPENSE_TABLE).header)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_index_growth_is_one_for_a_non_overlapping_split(self):
        text = "a" * 300
        self.assertAlmostEqual(index_growth(text, chunk_fixed(text, 100)), 1.0, places=6)

    def test_index_growth_exceeds_one_when_content_is_repeated(self):
        self.assertGreater(index_growth(EXPENSE_TABLE, chunk_table(EXPENSE_TABLE, 2)), 1.0)

    def test_density_of_empty_text_is_zero(self):
        self.assertEqual(density(""), 0.0)


if __name__ == "__main__":
    unittest.main()
