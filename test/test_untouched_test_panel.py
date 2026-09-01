import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.test_panel import (
    PanelRecord,
    accession_from_header,
    build_training_index,
    evaluate_candidates,
    prepare_test_panel,
    reverse_complement,
    sequence_sha256,
    stream_fasta,
)


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


class UntouchedTestPanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.training_fasta = self.root / "training.fasta"
        self.training_sequence = "AAACCCGGTACGTTAGCA"
        write_fasta(
            self.training_fasta,
            [
                ("NC_000001.1 Rosa alpha plastid, complete genome", self.training_sequence),
                ("NC_000002.1 Helianthus beta plastid, complete genome", "CCCGATATAGGCTAACGT"),
            ],
        )
        self.training = build_training_index(self.training_fasta)

    def tearDown(self):
        self.directory.cleanup()

    def record(self, accession: str, species: str, sequence: str) -> PanelRecord:
        header = f"{accession} {species} plastid, complete genome"
        return PanelRecord(accession, header, sequence, "test.fasta", "external-fasta")

    def test_accession_parser_supports_ncbi_pipe_headers(self):
        header = "gi|123|ref|NC_012345.1| Rosa test plastid"
        self.assertEqual(accession_from_header(header), "NC_012345.1")

    def test_existing_accession_root_is_rejected(self):
        candidates = [self.record("NC_000001.9", "Rosa gamma", "ATATCGCGTTAACCGGAA")]
        accepted, rejected = evaluate_candidates(candidates, self.training)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0].reason, "accession-present-in-training")
        self.assertEqual(rejected[0].match, "NC_000001.1")

    def test_exact_and_reverse_complement_duplicates_are_rejected(self):
        candidates = [
            self.record("TEST_1.1", "Rosa gamma", self.training_sequence),
            self.record("TEST_2.1", "Rosa delta", reverse_complement(self.training_sequence)),
        ]
        accepted, rejected = evaluate_candidates(candidates, self.training)
        self.assertEqual(accepted, [])
        self.assertEqual(
            [item.reason for item in rejected],
            ["exact-sequence-present-in-training", "reverse-complement-present-in-training"],
        )

    def test_genus_labels_distinguish_related_and_new_genomes(self):
        candidates = [
            self.record("TEST_3.1", "Rosa gamma", "ATATCGCGTTAACCGGAA"),
            self.record("TEST_4.1", "Solanum delta", "GATTACACCGGTTAATGC"),
        ]
        accepted, rejected = evaluate_candidates(candidates, self.training)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]["labels"], "unseen-accession")
        self.assertEqual(accepted[1]["labels"], "unseen-accession;unseen-genus")
        self.assertEqual(accepted[1]["species"], "Solanum delta")

    def test_development_accession_is_not_accepted(self):
        candidate = self.record("NC_053550.1", "Rosa minutifolia", "ATATCGCGTTAACCGGAA")
        accepted, rejected = evaluate_candidates(
            [candidate],
            self.training,
            development_accessions=["NC_053550.1"],
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0].reason, "development-accession")

    def test_external_fasta_and_frozen_outputs(self):
        test_fasta = self.root / "test.fasta"
        write_fasta(test_fasta, [("TEST_5.1 Solanum epsilon plastid", "AGCTTAGGCCAATTGC")])
        candidates = list(stream_fasta(test_fasta))
        manifest = self.root / "manifest.csv"
        rejections = self.root / "rejections.csv"
        metadata = self.root / "metadata.json"

        accepted, rejected = prepare_test_panel(
            str(self.training_fasta),
            candidates,
            str(manifest),
            str(rejections),
            str(metadata),
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        with manifest.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["accession"], "TEST_5.1")
        self.assertEqual(row["length"], "16")
        self.assertEqual(row["sequence_sha256"], sequence_sha256("AGCTTAGGCCAATTGC"))
        saved_metadata = json.loads(metadata.read_text())
        self.assertEqual(saved_metadata["training_records_scanned"], 2)
        self.assertEqual(saved_metadata["accepted_records"], 1)
        self.assertEqual(len(saved_metadata["training_fasta_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
