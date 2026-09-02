import gzip
import json
from pathlib import Path

from src.dna.prediction import TripletCodec


FORMAT_VERSION = 1


def validate_baseline_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported baseline checkpoint format")
    if checkpoint.get("prediction_unit") != "triplet":
        raise ValueError("baseline checkpoint must use triplet prediction")
    TripletCodec(checkpoint.get("triplet_vocabulary"))
    if len(checkpoint.get("base_counts", [])) != 4:
        raise ValueError("baseline checkpoint must contain four base counts")
    if len(checkpoint.get("triplet_counts", [])) != 64:
        raise ValueError("baseline checkpoint must contain 64 triplet counts")
    tables = checkpoint.get("markov_counts", {})
    if any(str(order) not in tables for order in range(1, 7)):
        raise ValueError("baseline checkpoint must contain Markov orders 1 through 6")


def _open(path: Path, mode: str):
    if ".gz" in path.suffixes:
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def save_baseline_checkpoint(checkpoint: dict, path: str | Path) -> None:
    validate_baseline_checkpoint(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with _open(temporary, "wt") as handle:
        json.dump(checkpoint, handle, separators=(",", ":"), sort_keys=True)
    temporary.replace(destination)


def load_baseline_checkpoint(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"baseline checkpoint not found: {source}")
    with _open(source, "rt") as handle:
        checkpoint = json.load(handle)
    validate_baseline_checkpoint(checkpoint)
    return checkpoint
