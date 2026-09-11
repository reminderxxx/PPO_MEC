"""Isolated immutable code fixture for historical continuation consumers."""
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="session")
def continuation_science(tmp_path_factory):
    root = tmp_path_factory.mktemp("continuation_science") / "checkout"
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(repository), str(root)],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach",
                    "a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d"],
                   check=True, capture_output=True, text=True)
    manifest = json.loads((root / "configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/execution_environment_manifest.json").read_text())
    return str(root), manifest["runtime_location"]["resolved_python_absolute_path"]
