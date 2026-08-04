import _path  # noqa: F401

import unittest

from src.biological_eval.context_topk import parse_int_list, target_starts
from src.sliding_eval.windows import slice_sequence


class ContextTopKTests(unittest.TestCase):
    def test_fixed_target_suffix_across_context_lengths(self):
        sequence = "ACGT" * 400
        target_start = 512
        truth = slice_sequence(sequence, target_start, 512, circular=True)
        for context_length in (128, 256, 512, 1024):
            prompt = slice_sequence(sequence, target_start - context_length, context_length, circular=True)
            self.assertEqual(len(prompt), context_length)
            self.assertEqual(slice_sequence(sequence, target_start, 512, circular=True), truth)

    def test_target_starts_use_prompt_offset(self):
        self.assertEqual(target_starts(2000, 512, 256, 3), [512, 768, 1024])

    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("128, 512", [1]), [128, 512])
        self.assertEqual(parse_int_list(None, [13]), [13])


if __name__ == "__main__":
    unittest.main()
