import _path  # noqa: F401

import unittest

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from src.biological_eval.annotations import feature_gene, feature_intervals, interval_overlap


class AnnotationParsingTests(unittest.TestCase):
    def test_circular_interval_overlap(self):
        self.assertEqual(interval_overlap(90, 10, 95, 5, 100), 10)
        self.assertEqual(interval_overlap(20, 40, 30, 50, 100), 10)
        self.assertEqual(interval_overlap(20, 40, 50, 70, 100), 0)

    def test_reverse_strand_feature_is_preserved(self):
        record = SeqRecord(Seq("A" * 100), id="TEST")
        feature = SeqFeature(
            FeatureLocation(10, 30, strand=-1),
            type="CDS",
            qualifiers={"gene": ["rbcL"]},
        )
        record.features.append(feature)
        self.assertEqual(feature.location.strand, -1)
        self.assertEqual(feature_gene(feature), "rbcL")
        self.assertEqual(feature_intervals(feature), [(10, 30)])

    def test_compound_feature_intervals(self):
        feature = SeqFeature(
            CompoundLocation(
                [
                    FeatureLocation(5, 10, strand=1),
                    FeatureLocation(20, 30, strand=1),
                ]
            ),
            type="CDS",
            qualifiers={"gene": ["matK"]},
        )
        self.assertEqual(feature_intervals(feature), [(5, 10), (20, 30)])


if __name__ == "__main__":
    unittest.main()
