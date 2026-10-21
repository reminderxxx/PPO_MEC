"""Reward visualization must preserve observations and reject inconsistent logs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "plot_training_rewards", Path(__file__).resolve().parents[1] / "scripts/plot_training_rewards.py"
)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def fixture_manifest(tmp_path, values=(1., 2., 3., 4.)):
    root = tmp_path / "run_seed7"
    (root / "checkpoints").mkdir(parents=True)
    (root / "episodes").mkdir()
    checkpoint = root / "checkpoints" / "latest.pt"
    checkpoint.write_bytes(b"test checkpoint, not deserialized")
    rows = [{"episode_index": i, "total_reward": v} for i, v in enumerate(values, 1)]
    summary = {"agent_name": "ppo", "run_id": "run_seed7", "episodes": len(rows),
               "update_count": 2, "rows": rows}
    (root / "train_summary.json").write_text(json.dumps(summary))
    for row in rows:
        receipt = {"reward_breakdown": {"total": {"sum": row["total_reward"]}}}
        (root / "episodes" / f"episode_{row['episode_index']:04d}.summary.json").write_text(json.dumps(receipt))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"ppo": {"7": str(checkpoint)}}))
    return path, root


def test_no_future_leakage_or_padding():
    a = plots.trailing_mean([1, 2, 3, 100], 2)
    assert np.isnan(a[0])
    np.testing.assert_allclose(a[1:], [1.5, 2.5, 51.5])
    b = plots.trailing_mean([1, 2, 3, -100], 2)
    np.testing.assert_allclose(a[1:3], b[1:3])
    assert len(a) == 4


def test_exact_logged_length_retained(tmp_path):
    path, root = fixture_manifest(tmp_path)
    runs, sources = plots.load_runs(path, 2)
    np.testing.assert_array_equal(runs["ppo"]["7"]["values"], [1, 2, 3, 4])
    assert sources[0]["episodes"] == 4
    assert len(sources[0]["episode_receipt_chain_sha256"]) == 64
    assert sources[0]["checkpoint_selection"] == "latest.pt"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "receipt", "seed", "agent"])
def test_bad_sources_fail_closed(tmp_path, fault):
    path, root = fixture_manifest(tmp_path)
    summary_path = root / "train_summary.json"
    summary = json.loads(summary_path.read_text())
    if fault == "missing":
        summary["rows"].pop()
    elif fault == "duplicate":
        summary["rows"][1]["episode_index"] = 1
    elif fault == "nan":
        summary["rows"][0]["total_reward"] = float("nan")
    elif fault == "receipt":
        summary["rows"][0]["total_reward"] = 99
    elif fault == "seed":
        summary["run_id"] = "run_seed13"
    elif fault == "agent":
        summary["agent_name"] = "dqn"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        plots.load_runs(path, 2)


def test_flat_or_improving_curve_does_not_prove_convergence():
    flat = plots.summarize([[2.] * 8, [2.] * 8], 2)
    assert flat["last_minus_previous"] == 0
    assert flat["convergence_proven"] is False
    rising = plots.summarize([list(range(8)), list(range(8))], 2)
    assert rising["last_minus_first"] == 6
    assert rising["convergence_proven"] is False
