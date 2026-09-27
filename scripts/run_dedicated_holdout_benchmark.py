"""Open a sealed holdout once and launch its exact authorized benchmark command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluators.dedicated_holdout_execution import (
    DedicatedHoldoutExecutionError,
    build_benchmark_command,
    load_authorization,
)
from src.evaluators.typed_model_cache_formal_protocol import (
    append_holdout_execution_record,
    canonical_sha256,
    semantic_projection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consume one issued holdout authorization and run its exact benchmark"
    )
    parser.add_argument("--authorization-path", required=True)
    parser.add_argument("--seal-path", required=True)
    parser.add_argument("--execution-token-file", required=True)
    return parser


def _load_object(path: str | Path, label: str) -> dict:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DedicatedHoldoutExecutionError(f"unable to load {label}: {target}") from exc
    if not isinstance(payload, dict):
        raise DedicatedHoldoutExecutionError(f"{label} must be a JSON object")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    authorization, report = load_authorization(args.authorization_path)
    seal = _load_object(args.seal_path, "holdout seal")
    seal_hash = canonical_sha256(semantic_projection(seal))
    if seal.get("hashes", {}).get("semantic_sha256") != seal_hash:
        raise DedicatedHoldoutExecutionError("holdout seal hash mismatch")
    bindings = report["bindings"]
    if bindings["seal_semantic_sha256"] != seal_hash:
        raise DedicatedHoldoutExecutionError("authorization/seal identity mismatch")
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != bindings["execution_commit"]:
        raise DedicatedHoldoutExecutionError("execution Git commit mismatch")
    token = Path(args.execution_token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise DedicatedHoldoutExecutionError("one-time execution token is empty")
    command = build_benchmark_command(
        authorization_path=args.authorization_path,
        python_executable=sys.executable,
        repository_root=ROOT_DIR,
    )
    opening_path = Path(str(bindings["opening_record_path"]))
    append_holdout_execution_record(
        opening_path,
        seal_record=seal,
        execution_token=token,
        gate_results=authorization["opening_gate_results"],
        execution_commit=observed_commit,
        command=authorization["benchmark_argv"],
        output_run_id=str(bindings["output_run_id"]),
        authorization_semantic_sha256=report["authorization_semantic_sha256"],
        command_semantic_sha256=report["command_semantic_sha256"],
    )
    completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    receipt = {
        "status": "completed" if completed.returncode == 0 else "failed_after_consumption",
        "return_code": int(completed.returncode),
        "authorization_id": report["authorization_id"],
        "authorization_semantic_sha256": report["authorization_semantic_sha256"],
        "command_semantic_sha256": report["command_semantic_sha256"],
        "opening_record_path": str(opening_path),
        "consumed_permanently": True,
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
