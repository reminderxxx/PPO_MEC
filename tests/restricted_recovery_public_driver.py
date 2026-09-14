"""Test-only child adapter for the public restricted-recovery main path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.formal_cell_transaction import (
    resolve_child_output_descriptor,
    write_child_output_descriptor,
)
from src.runtime.evaluation_only_execution import file_sha256
from src.runtime.restricted_recovery import ALLOWED_CELL_IDS, PHASE


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SyntheticRecoveryChild:
    def __init__(self, scope: str | Path):
        self.scope = Path(scope).resolve()

    def validate(self, request):
        binding = json.loads((self.scope / "synthetic_binding.json").read_text())
        execution = request["recovery_execution"]
        if (
            not self.scope.name.startswith("synthetic_")
            or Path(execution["recovery_root"]) != self.scope / "synthetic_recovery_run"
            or binding
            != {
                "test_only": True,
                "scope": str(self.scope),
                "request_sha256": request["authorization_request_sha256"],
            }
            or execution["allowed_phase"] != PHASE
            or execution["allowed_cell_ids"] != list(ALLOWED_CELL_IDS)
            or execution["holdout_capability"] is not False
        ):
            raise ValueError("synthetic restricted recovery adapter scope mismatch")

    def adapt(self, request, cell_id, builder, resolver):
        self.validate(request)

        def build(staging, actual_cell_id):
            staged = builder(staging, actual_cell_id)
            write_json(
                self.scope / f"{cell_id}_original_staged_argv.json",
                staged,
            )
            return [
                request["recovery_execution"]["python_executable"],
                str(ROOT / "tests/restricted_recovery_public_driver.py"),
                "child",
                str(self.scope),
                str(staging),
                actual_cell_id,
                request["recovery_execution"]["command_plan"]["matrix_contexts"][
                    list(ALLOWED_CELL_IDS).index(cell_id)
                ]["ablation_setting_id"],
            ]

        def resolve(staging, actual_cell_id, completed):
            child = staging / "child_output"
            artifact, _ = resolve_child_output_descriptor(
                child / "cell_child_output.json",
                output_root=child,
                expected_cell_id=actual_cell_id,
                expected_phase=PHASE,
                expected_setting_id=request["recovery_execution"]["command_plan"][
                    "matrix_contexts"
                ][list(ALLOWED_CELL_IDS).index(cell_id)]["ablation_setting_id"],
            )
            native_artifact, required, native_child = resolver(
                staging, actual_cell_id, completed
            )
            if artifact != native_artifact or native_child != child:
                raise ValueError("synthetic recovery descriptor/native resolver disagreement")
            return native_artifact, required, native_child

        return build, resolve


def child(scope: str, staging: str, cell_id: str, setting_id: str) -> None:
    scope_path = Path(scope)
    staging_path = Path(staging)
    assert scope_path.name.startswith("synthetic_") and scope_path in staging_path.parents
    assert os.environ.get("PYTHONPATH") == str(ROOT)
    with (scope_path / "dispatch.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "cell_id": cell_id,
                    "setting_id": setting_id,
                    "test_only": True,
                    "real_model_open_count": 0,
                    "scientific_rollout_count": 0,
                }
            )
            + "\n"
        )
    if (scope_path / "fail_child").exists():
        raise RuntimeError("synthetic restricted recovery child failure")
    child_root = staging_path / "child_output"
    artifact = child_root / setting_id
    artifact.mkdir(parents=True)
    write_json(artifact / "support_provenance.json", {"setting_id": setting_id, "test_only": True})
    write_json(artifact / "aggregate_summary.json", {"test_only": True, "metric": 1.0})
    (artifact / "benchmark_rows.csv").write_text("test_only,metric\ntrue,1.0\n", encoding="utf-8")
    files = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(artifact.iterdir())
    ]
    write_json(
        artifact / "artifact_integrity_manifest.json",
        {"integrity_manifest_version": "1.0.0", "files": files},
    )
    write_child_output_descriptor(
        child_root / "cell_child_output.json",
        cell_id=cell_id,
        phase=PHASE,
        logical_setting_id=setting_id,
        output_root=child_root,
        artifact_root=artifact,
        producer_kind="synthetic_restricted_recovery_only",
        required_payload=[
            "support_provenance.json",
            "aggregate_summary.json",
            "benchmark_rows.csv",
        ],
    )


def host(scope: str, argv: list[str]) -> None:
    from scripts import run_typed_model_cache_restricted_recovery as public

    scope_path = Path(scope).resolve()
    audit = {
        "pid": os.getpid(),
        "argv": argv,
        "cwd": os.getcwd(),
        "import_origin": public.__file__,
        "python": sys.executable,
        "test_only": True,
    }
    print(json.dumps({"host_audit": audit}), flush=True)
    write_json(scope_path / f"host_{os.getpid()}.json", audit)
    sys.argv = [public.__file__, *argv]
    public.main(scientific_child_adapter=SyntheticRecoveryChild(scope_path))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(*sys.argv[2:])
    else:
        from tests.restricted_recovery_public_driver import host

        host(sys.argv[2], sys.argv[3:])
