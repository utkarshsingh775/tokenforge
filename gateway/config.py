"""Loads the model registry and API keys from models.json."""
import json
import os
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("TOKENFORGE_CONFIG", Path(__file__).parent.parent / "models.json"))


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)
