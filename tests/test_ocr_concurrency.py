"""Concurrency tests for ocr_engine._ocr_pages.

The OCR backend is stubbed, so these run offline and spend nothing. They exist
because the parallel path shipped without ever being executed: as_completed()
returns futures in completion order, so the reassembly is what has to restore
page order, and nothing had exercised it.
"""

import unittest
from unittest.mock import patch

import ocr_engine as O


class FakeDoc:
    """doc[n] -> n. _prepare_page is stubbed, so a page can just be its index."""

    def __init__(self, bad_pages=frozenset()):
        self.bad_pages = bad_pages

    def __getitem__(self, n):
        return n


def _prepare(page):
    if isinstance(page, tuple):  # (doc, n) never happens; guard for clarity
        raise AssertionError
    return f"P{page}".encode()


def _backend(data):
    return "text-" + data.decode()


class ReassemblyTests(unittest.TestCase):
    """Sequential and parallel must produce identical dicts."""

    def _run(self, workers, pages):
        with patch.object(O, "OCR_MAX_WORKERS", workers), \
             patch.object(O, "_prepare_page", _prepare), \
             patch.dict(O._OCR_BACKENDS, {O.OCR_BACKEND: _backend}):
            return O._ocr_pages(FakeDoc(), pages)

    def test_parallel_matches_sequential(self):
        pages = list(range(37))          # not a multiple of any chunk size
        self.assertEqual(self._run(1, pages), self._run(8, pages))

    def test_every_page_maps_to_its_own_text(self):
        pages = list(range(37))
        got = self._run(8, pages)
        self.assertEqual(got, {n: f"text-P{n}" for n in pages})

    def test_non_contiguous_page_numbers_survive(self):
        # Real runs pass only the pages that need OCR, so gaps are normal.
        pages = [3, 9, 10, 40, 41, 99]
        self.assertEqual(self._run(8, pages), {n: f"text-P{n}" for n in pages})


class FailureIsolationTests(unittest.TestCase):
    """One bad page must not cost the rest of the document."""

    def test_rasterization_failure_does_not_abort_the_document(self):
        def prepare(page):
            if page == 5:
                raise ValueError("corrupt page")
            return f"P{page}".encode()

        pages = list(range(20))
        with patch.object(O, "OCR_MAX_WORKERS", 8), \
             patch.object(O, "_prepare_page", prepare), \
             patch.dict(O._OCR_BACKENDS, {O.OCR_BACKEND: _backend}):
            got = O._ocr_pages(FakeDoc(), pages)

        self.assertEqual(len(got), len(pages))
        self.assertEqual(got[5], "")
        self.assertEqual(got[6], "text-P6")

    def test_one_ocr_failure_does_not_abort_the_document(self):
        def backend(data):
            if data == b"P5":
                raise RuntimeError("backend blew up")
            return "text-" + data.decode()

        pages = list(range(20))
        with patch.object(O, "OCR_MAX_WORKERS", 8), \
             patch.object(O, "_prepare_page", _prepare), \
             patch.dict(O._OCR_BACKENDS, {O.OCR_BACKEND: backend}):
            got = O._ocr_pages(FakeDoc(), pages)

        self.assertEqual(got[5], "")
        self.assertEqual(len(got), len(pages))


class SystemicFailureTests(unittest.TestCase):
    """Bad credentials must fail loudly, not yield an index full of empties."""

    def test_total_failure_raises_instead_of_returning_empties(self):
        def dead(data):
            raise RuntimeError("403 credentials rejected")

        pages = list(range(30))
        with patch.object(O, "OCR_MAX_WORKERS", 8), \
             patch.object(O, "_prepare_page", _prepare), \
             patch.dict(O._OCR_BACKENDS, {O.OCR_BACKEND: dead}):
            with self.assertRaises(RuntimeError):
                O._ocr_pages(FakeDoc(), pages)

    def test_sequential_path_also_raises_on_total_failure(self):
        def dead(data):
            raise RuntimeError("403 credentials rejected")

        with patch.object(O, "OCR_MAX_WORKERS", 1), \
             patch.object(O, "_prepare_page", _prepare), \
             patch.dict(O._OCR_BACKENDS, {O.OCR_BACKEND: dead}):
            with self.assertRaises(RuntimeError):
                O._ocr_pages(FakeDoc(), list(range(30)))


if __name__ == "__main__":
    unittest.main()
