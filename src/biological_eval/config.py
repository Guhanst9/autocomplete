import json
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open() as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def require_keys(config: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")

