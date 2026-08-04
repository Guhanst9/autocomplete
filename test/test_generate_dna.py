import tempfile
from pathlib import Path

from generate_dna import clean_prompt, read_prompt_file


def test_clean_prompt():
    assert clean_prompt("acgt\nACGT") == "ACGTACGT"

    try:
        clean_prompt("ACNT")
    except ValueError as error:
        assert "N" in str(error)
    else:
        raise AssertionError("invalid prompt was accepted")


def test_read_prompt_file():
    with tempfile.TemporaryDirectory() as directory:
        fasta = Path(directory) / "prompt.fna"
        fasta.write_text(">first\nACGT\nTGCA\n>second\nAAAA\n")
        assert read_prompt_file(str(fasta)) == "ACGTTGCA"


if __name__ == "__main__":
    test_clean_prompt()
    test_read_prompt_file()
    print("DNA generation tests passed.")
