import unittest

from src.biological_eval.features import cds_metrics, translated_identity
from src.biological_eval.annotations import FeatureRecord


class FeatureMetricTests(unittest.TestCase):
    def test_translated_identity_and_internal_stops(self):
        identity, stops = translated_identity("ATGTAGGCC", "ATGAAAGCC")
        self.assertAlmostEqual(identity, 2 / 3 * 100)
        self.assertEqual(stops, 1)

    def test_reverse_strand_cds_analysis_only(self):
        window = {
            "window_start": "0",
            "target_start": "0",
            "target_end": "8",
            "prompt": "",
            "generated_suffix": "GGCCAT",
            "true_suffix": "GGCCAT",
        }
        feature = FeatureRecord("TEST", "CDS", "gene", -1, 0, 6)
        identity, stops = cds_metrics(window, feature, 100)
        self.assertEqual(identity, 100.0)
        self.assertEqual(stops, 0)
        self.assertEqual(window["generated_suffix"], "GGCCAT")


if __name__ == "__main__":
    unittest.main()
