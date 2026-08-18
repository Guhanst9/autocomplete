import _path  # noqa: F401

import csv
import tempfile

import torch

from src.sliding_eval.fasta import PlastidRecord
from src.sliding_eval.generation import gc_difference_percent, longest_homopolymer_run
from src.sliding_eval.regions import infer_regions, reverse_complement
from src.models.s4_model import _select_next_token
from src.sliding_eval.windows import SlidingWindow, write_windows_csv


def test_quality_metrics():
    assert longest_homopolymer_run("AAACCG") == 3
    assert longest_homopolymer_run("") == 0
    assert gc_difference_percent("AACC", "GGCC") == 50.0


def test_csv_diagnostics():
    record = PlastidRecord("TEST.1", "TEST.1 synthetic", "AAAACCCC")
    window = SlidingWindow(
        window_start=0,
        prompt_start=0,
        prompt_end=3,
        target_start=4,
        target_end=7,
        region="LSC",
        prompt="AAAA",
        true_suffix="CCCC",
        generated_suffix="CCCA",
        generated_length=4,
        accuracy_percent=75.0,
        decoding_mode="raw_greedy",
        longest_generated_run=3,
        n_count=0,
        gc_difference_percent=25.0,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = write_windows_csv(record, [window], directory)
        with open(path, newline="") as handle:
            row = next(csv.DictReader(handle))

    assert row["decoding_mode"] == "raw_greedy"
    assert row["longest_generated_run"] == "3"
    assert row["n_count"] == "0"
    assert row["gc_difference_percent"] == "25.00"


def test_token_selection():
    logits = torch.tensor([[0.0, 4.0, 1.0]])
    assert _select_next_token(logits, None).item() == 1

    torch.manual_seed(13)
    first = _select_next_token(logits, 1.0)
    torch.manual_seed(13)
    second = _select_next_token(logits, 1.0)
    assert torch.equal(first, second)


def test_region_inference_tolerates_a_repeat_indel():
    generator = torch.Generator().manual_seed(13)
    bases = "ACGT"
    random_dna = lambda length: "".join(
        bases[index] for index in torch.randint(0, 4, (length,), generator=generator)
    )
    repeat = random_dna(1300)
    second_repeat = reverse_complement(repeat[:700] + "A" + repeat[700:])
    sequence = random_dna(2000) + repeat + random_dna(1800) + second_repeat + random_dna(2500)
    region_map = infer_regions(
        sequence,
        min_ir_length=1000,
        min_ir_spacing=1000,
    )
    regions = {region.name: region for region in region_map.regions}
    assert region_map.status == "sequence-inferred"
    assert regions["IRB"].start <= 2010
    assert regions["IRB"].end >= 3290
    assert regions["IRA"].length >= 1290


if __name__ == "__main__":
    test_quality_metrics()
    test_csv_diagnostics()
    test_token_selection()
    test_region_inference_tolerates_a_repeat_indel()
    print("Sliding evaluation tests passed.")
