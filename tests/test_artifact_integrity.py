import json
from pathlib import Path

from scripts.audit_artifact_integrity import audit, looks_like_path


def test_audit_hashes_run_files_and_external_references(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "artifacts" / "run"
    external = repo_root / "data" / "input.csv"
    output_dir = repo_root / "artifacts" / "audit"
    run_root.mkdir(parents=True)
    external.parent.mkdir(parents=True)
    external.write_text("value\n1\n", encoding="utf-8")
    (run_root / "result.json").write_text(
        json.dumps({"input_path": "data/input.csv", "score": 1.0}), encoding="utf-8"
    )

    report = audit([run_root], output_dir=output_dir, repo_root=repo_root)

    assert report["passed"] is True
    assert report["external_reference_count"] == 1
    manifest = (output_dir / "sha256_manifest.txt").read_text(encoding="utf-8")
    assert "artifacts/run/result.json" in manifest
    assert "data/input.csv" in manifest


def test_audit_reports_missing_reference(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "artifacts" / "run"
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(
        json.dumps({"checkpoint_path": "artifacts/missing.pt"}), encoding="utf-8"
    )

    report = audit(
        [run_root], output_dir=repo_root / "artifacts" / "audit", repo_root=repo_root
    )

    assert report["passed"] is False
    assert report["missing_reference_count"] == 1
    assert report["missing_references"][0]["reference"] == "artifacts/missing.pt"


def test_audit_allows_explicitly_absent_optional_checkpoint(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "artifacts" / "run"
    run_root.mkdir(parents=True)
    (run_root / "audit.json").write_text(
        json.dumps(
            {
                "audited_checkpoints": [
                    {
                        "checkpoint_path": "artifacts/run/checkpoints/warm_start.pt",
                        "exists": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit(
        [run_root], output_dir=repo_root / "artifacts" / "audit", repo_root=repo_root
    )

    assert report["passed"] is True
    assert report["missing_reference_count"] == 0


def test_provenance_description_with_embedded_paths_is_not_a_reference() -> None:
    assert looks_like_path("scenario=/tmp/scenario; trace_csv=/tmp/trace.csv") is False
