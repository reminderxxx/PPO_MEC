"""Test-only scientific child adapter; public main and all identity gates stay real.

Not a production CLI option. It accepts only a hash-bound, synthetic_* fixture
under an explicit test scope, with the real prepared source/command contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.formal_cell_transaction import (
    resolve_child_output_descriptor, write_child_output_descriptor,
)
from src.runtime.evaluation_only_execution import canonical_sha256, file_sha256


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class SyntheticScientificChild:
    def __init__(self, scope):
        self.scope = Path(scope).resolve()

    def validate(self, package):
        binding = json.loads((self.scope / "synthetic_binding.json").read_text())
        execution = package["evaluation_execution_contract"]
        run = Path(execution["evaluation_run_root"])
        if (not self.scope.name.startswith("synthetic_")
                or run != self.scope / "synthetic_evaluation_run"
                or binding != {"test_only": True, "scope": str(self.scope),
                               "request_sha256": package["authorization_request_sha256"]}
                or execution["holdout_capability"] is not False):
            raise ValueError("test-only scientific adapter scope mismatch")

    def adapt(self, package, phase, builder, resolver):
        self.validate(package)
        if phase not in ("formal_cache_policy", "formal_controller"):
            raise ValueError("test-only adapter covers exactly two phases")

        def build(staging, cell_id):
            # Exercise the production staging command builder unchanged, but run
            # a tiny, explicitly test-only child instead of the scientific CLI.
            staged = builder(staging, cell_id)
            write_json(self.scope / (cell_id + "_synthetic_original_staged_argv.json"), staged)
            return [package["evaluation_execution_contract"]["python_executable"],
                    str(ROOT / "tests/evaluation_only_public_driver.py"), "child",
                    str(self.scope), phase, str(staging), cell_id]

        def resolve(staging, cell_id, completed):
            artifact, descriptor = resolve_child_output_descriptor(
                staging / "cell_child_output.json", output_root=staging,
                expected_cell_id=cell_id, expected_phase=phase,
                expected_setting_id="synthetic")
            native_artifact, required, child_root = resolver(staging, cell_id, completed)
            if artifact != native_artifact:
                raise ValueError("synthetic descriptor/native resolver disagreement")
            return native_artifact, required, child_root
        return build, resolve


def child(scope, phase, staging, cell_id):
    scope, staging = Path(scope), Path(staging)
    assert scope.name.startswith("synthetic_") and scope in staging.parents
    assert os.environ.get("PYTHONPATH") == str(ROOT)
    assert not any("acceptance_env" in p for p in sys.path)
    with (scope / "dispatch.jsonl").open("a") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "phase": phase, "argv": sys.argv,
                                 "cwd": os.getcwd(), "sys_path": sys.path,
                                 "pythonpath": os.environ.get("PYTHONPATH"),
                                 "test_only": True, "scientific_rollout": 0}) + "\n")
    if (scope / "fail_child").exists():
        raise RuntimeError("test-only child failure")
    if (scope / "pause_child").exists():
        (scope / "child_ready").touch()
        deadline = time.monotonic() + 30
        while not (scope / "release_child").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("test-only barrier timeout")
            time.sleep(.02)
    if phase == "formal_cache_policy":
        artifact = staging / "artifact"
        producer = artifact / "benchmark" / "synthetic_benchmark"
        producer.mkdir(parents=True)
        write_json(artifact / "request_replay.json", {"test_only": True})
    else:
        artifact = staging / "child_output" / "synthetic_benchmark"
        producer = artifact
        producer.mkdir(parents=True)
    write_json(producer / "aggregate_summary.json", {"test_only": True, "metric": 1.25})
    (producer / "benchmark_rows.csv").write_text("test_only,metric\ntrue,1.25\n")
    files = [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": file_sha256(p)}
             for p in sorted(producer.iterdir())]
    write_json(producer / "artifact_integrity_manifest.json",
               {"integrity_manifest_version": "1.0.0", "files": files})
    write_child_output_descriptor(
        staging / "cell_child_output.json", cell_id=cell_id, phase=phase,
        logical_setting_id="synthetic", output_root=staging, artifact_root=artifact,
        producer_kind="test_only", required_payload=[
            "request_replay.json" if phase == "formal_cache_policy" else "aggregate_summary.json"])


def host(scope, fault, argv):
    from scripts import run_typed_model_cache_evaluation_only as public
    scope = Path(scope).resolve()
    run = scope / "synthetic_evaluation_run"
    if fault != "none" or (scope / "pause_before_initialize").exists():
        def trace(frame, event, arg):
            if (event == "call" and frame.f_code.co_name == "_initialize_run"
                    and frame.f_code.co_filename == public.__file__
                    and (scope / "pause_before_initialize").exists()):
                (scope / "initializer_ready").touch()
                deadline = time.monotonic() + 60
                while not (scope / "release_initializer").exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError("test-only initialization barrier timeout")
                    time.sleep(.02)
            # Inject OS-like failure after an actual boundary; do not replace
            # initializer, loader, lock, ledger, publication or identity checks.
            if (fault != "none" and event == "line"
                    and frame.f_code.co_filename in {public.__file__,
                        str(ROOT / "src/evaluators/formal_cell_transaction.py")}):
                reached = run.is_dir() if fault == "root_created" else (run / fault).exists()
                if reached:
                    sys.settrace(None)
                    raise OSError("test-only initialization write fault: " + fault)
            return trace
        sys.settrace(trace)
    audit = {
        "pid": os.getpid(), "argv": argv, "cwd": os.getcwd(),
        "import_origin": public.__file__, "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"), "test_only": True}
    print(json.dumps({"host_audit": audit}), flush=True)
    if os.access(scope, os.W_OK):
        write_json(scope / ("host_%s.json" % os.getpid()), audit)
    sys.argv = [public.__file__, *argv]
    public.main(scientific_child_adapter=SyntheticScientificChild(scope))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(*sys.argv[2:])
    else:
        # Import canonical module so the public main exact-type check is real.
        from tests.evaluation_only_public_driver import host
        host(sys.argv[2], sys.argv[3], sys.argv[4:])
