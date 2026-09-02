import _path  # noqa: F401

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from src.biological_eval.annotations import (
    feature_gene,
    feature_intervals,
    genbank_region_map,
    interval_overlap,
)


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

    def test_genbank_repeat_regions_are_used(self):
        record = SeqRecord(Seq("A" * 60000), id="TEST.1", name="TEST")
        record.annotations["molecule_type"] = "DNA"
        record.features.extend(
            [
                SeqFeature(
                    FeatureLocation(10000, 22000),
                    type="repeat_region",
                    qualifiers={"note": ["IRb"]},
                ),
                SeqFeature(
                    FeatureLocation(38000, 50000),
                    type="repeat_region",
                    qualifiers={"note": ["IRa"]},
                ),
            ]
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "record.gb"
            SeqIO.write(record, path, "genbank")
            region_map = genbank_region_map(path)
        self.assertIsNotNone(region_map)
        self.assertEqual(region_map.status, "genbank")
        regions = {region.name: region for region in region_map.regions}
        self.assertEqual((regions["IRB"].start, regions["IRB"].end), (10000, 22000))
        self.assertEqual((regions["IRA"].start, regions["IRA"].end), (38000, 50000))

    def test_implausible_repeat_regions_are_rejected(self):
        record = SeqRecord(Seq("A" * 160000), id="TEST.1", name="TEST")
        record.annotations["molecule_type"] = "DNA"
        record.features.extend(
            [
                SeqFeature(
                    FeatureLocation(3, 133652),
                    type="repeat_region",
                    qualifiers={"note": ["inverted repeat A"]},
                ),
                SeqFeature(
                    FeatureLocation(87865, 114268),
                    type="repeat_region",
                    qualifiers={"note": ["inverted repeat B"]},
                ),
            ]
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "record.gb"
            SeqIO.write(record, path, "genbank")
            self.assertIsNone(genbank_region_map(path))


if __name__ == "__main__":
    unittest.main()
