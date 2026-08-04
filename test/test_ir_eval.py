import unittest

from src.biological_eval.context_topk import exact_identity
from src.biological_eval.ir import paired_offsets
from src.sliding_eval.regions import Region, reverse_complement


class IrEvalTests(unittest.TestCase):
    def test_reverse_complement_identity(self):
        ira = "AACCGGTT"
        irb = reverse_complement(ira)
        self.assertEqual(exact_identity(ira, reverse_complement(irb)), 100.0)

    def test_paired_offsets_are_bounded(self):
        ira = Region("IRA", 100, 1100)
        irb = Region("IRB", 5000, 6000)
        offsets = paired_offsets(ira, irb, 512, 2)
        self.assertEqual(offsets[0], 0)
        self.assertLessEqual(offsets[-1], 488)


if __name__ == "__main__":
    unittest.main()
