"""Fail-closed authorization for the one-time holdout benchmark entrypoint.

The ordinary benchmark runner remains unable to open a sealed split.  A
dedicated launcher must first consume the existing seal, persist the
append-only opening record, and then pass both that record and an immutable
authorization document to the benchmark preflight and window consumer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluators.typed_model_cache_formal_protocol import (
    canonical_sha256,
    semantic_projection,
)


DEDICATED_HOLDOUT_AUTHORIZATION_VERSION = "1.0.0"
DEDICATED_HOLDOUT_EXECUTION_MODE = "dedicated_holdout_execution"
DEDICATED_HOLDOUT_CALLER_ROLE = "dedicated_one_time_holdout_executor"
DEDICATED_HOLDOUT_ENTRYPOINT = "scripts/benchmark_main_results.py"


class DedicatedHoldoutExecutionError(ValueError):
    """Raised when a dedicated holdout capability is missing or inconsistent."""


def _require_hex_digest(value: Any, *, field: str, length: int) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise DedicatedHoldoutExecutionError(
            f"dedicated holdout {field} must be a {length}-character hex digest"
        )
    return normalized


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DedicatedHoldoutExecutionError(f"unable to load {label}: {target}") from exc
    if not isinstance(payload, dict):
        raise DedicatedHoldoutExecutionError(f"{label} must be a JSON object")
    return payload


def command_semantic_sha256(benchmark_argv: Sequence[str]) -> str:
    return canonical_sha256(
        {
            "entrypoint": DEDICATED_HOLDOUT_ENTRYPOINT,
            "benchmark_argv": list(benchmark_argv),
        }
    )


def _flag_value(argv: Sequence[str], flag: str) -> str:
    values = [index for index, value in enumerate(argv) if value == flag]
    if len(values) != 1 or values[0] + 1 >= len(argv):
        raise DedicatedHoldoutExecutionError(
            f"dedicated holdout command requires exactly one {flag} value"
        )
    return str(argv[values[0] + 1])


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    if (
        authorization.get("dedicated_holdout_authorization_version")
        != DEDICATED_HOLDOUT_AUTHORIZATION_VERSION
    ):
        raise DedicatedHoldoutExecutionError("unsupported dedicated holdout authorization")
    if authorization.get("state") != "issued" or authorization.get("execution_authorized") is not True:
        raise DedicatedHoldoutExecutionError("dedicated holdout authorization is not issued")
    if authorization.get("caller_role") != DEDICATED_HOLDOUT_CALLER_ROLE:
        raise DedicatedHoldoutExecutionError("dedicated holdout caller role mismatch")
    if authorization.get("target_entrypoint") != DEDICATED_HOLDOUT_ENTRYPOINT:
        raise DedicatedHoldoutExecutionError("dedicated holdout entrypoint mismatch")
    if authorization.get("allowed_split") != "sealed_holdout":
        raise DedicatedHoldoutExecutionError("dedicated holdout split mismatch")
    if authorization.get("window_consumption_mode") != DEDICATED_HOLDOUT_EXECUTION_MODE:
        raise DedicatedHoldoutExecutionError("dedicated holdout execution mode mismatch")
    argv = authorization.get("benchmark_argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise DedicatedHoldoutExecutionError("dedicated holdout benchmark_argv is invalid")
    if _flag_value(argv, "--formal_window_split") != "sealed_holdout":
        raise DedicatedHoldoutExecutionError("dedicated command does not bind sealed_holdout")
    if (
        _flag_value(argv, "--window_consumption_mode")
        != DEDICATED_HOLDOUT_EXECUTION_MODE
    ):
        raise DedicatedHoldoutExecutionError(
            "dedicated command cannot use identity_only or ordinary execution mode"
        )
    bindings = authorization.get("bindings")
    if not isinstance(bindings, Mapping):
        raise DedicatedHoldoutExecutionError("dedicated holdout bindings are missing")
    required_bindings = {
        "seal_semantic_sha256",
        "window_consumption_contract_semantic_sha256",
        "window_plan_file_sha256",
        "execution_commit",
        "opening_record_path",
        "output_run_id",
        "output_root",
        "candidate_checkpoint_manifest_sha256",
        "statistics_contract_sha256",
    }
    missing = sorted(required_bindings.difference(bindings))
    if missing or any(not bindings.get(field) for field in required_bindings):
        raise DedicatedHoldoutExecutionError(
            f"dedicated holdout bindings are incomplete: {missing}"
        )
    for field in (
        "seal_semantic_sha256",
        "window_consumption_contract_semantic_sha256",
        "window_plan_file_sha256",
        "candidate_checkpoint_manifest_sha256",
        "statistics_contract_sha256",
    ):
        _require_hex_digest(bindings[field], field=field, length=64)
    _require_hex_digest(bindings["execution_commit"], field="execution_commit", length=40)
    expected_hash = canonical_sha256(semantic_projection(authorization))
    if authorization.get("hashes", {}).get("semantic_sha256") != expected_hash:
        raise DedicatedHoldoutExecutionError("dedicated holdout authorization hash mismatch")
    expected_command_hash = command_semantic_sha256(argv)
    if authorization.get("command_semantic_sha256") != expected_command_hash:
        raise DedicatedHoldoutExecutionError("dedicated holdout command hash mismatch")
    if authorization_path is not None:
        command_path = Path(
            _flag_value(argv, "--holdout-execution-authorization-path")
        ).resolve()
        if command_path != Path(authorization_path).resolve():
            raise DedicatedHoldoutExecutionError(
                "dedicated command authorization path mismatch"
            )
    if Path(_flag_value(argv, "--holdout-opening-record-path")).resolve() != Path(
        str(bindings["opening_record_path"])
    ).resolve():
        raise DedicatedHoldoutExecutionError("dedicated opening record path mismatch")
    if _flag_value(argv, "--benchmark-run-id") != str(bindings["output_run_id"]):
        raise DedicatedHoldoutExecutionError("dedicated benchmark run ID mismatch")
    if Path(_flag_value(argv, "--output_root")).resolve() != Path(
        str(bindings["output_root"])
    ).resolve():
        raise DedicatedHoldoutExecutionError("dedicated benchmark output root mismatch")
    return {
        "status": "pass",
        "authorization_id": str(authorization.get("authorization_id")),
        "authorization_semantic_sha256": expected_hash,
        "command_semantic_sha256": expected_command_hash,
        "split": "sealed_holdout",
        "mode": DEDICATED_HOLDOUT_EXECUTION_MODE,
        "bindings": dict(bindings),
    }


def load_authorization(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = _load_object(path, "dedicated holdout authorization")
    return authorization, validate_authorization(
        authorization, authorization_path=path
    )


def validate_opening_record(
    opening_record: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    authorization_report: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = authorization_report["bindings"]
    if opening_record.get("consumed_permanently") is not True:
        raise DedicatedHoldoutExecutionError("holdout opening record is not consumed")
    if (
        opening_record.get("authorization_semantic_sha256")
        != authorization_report["authorization_semantic_sha256"]
    ):
        raise DedicatedHoldoutExecutionError("holdout opening authorization identity mismatch")
    if (
        opening_record.get("command_semantic_sha256")
        != authorization_report["command_semantic_sha256"]
    ):
        raise DedicatedHoldoutExecutionError("holdout opening command identity mismatch")
    if opening_record.get("output_run_id") != bindings["output_run_id"]:
        raise DedicatedHoldoutExecutionError("holdout opening output identity mismatch")
    if opening_record.get("command") != authorization.get("benchmark_argv"):
        raise DedicatedHoldoutExecutionError("holdout opening command payload mismatch")
    return {
        "status": "pass",
        "consumed_permanently": True,
        "authorization_semantic_sha256": authorization_report[
            "authorization_semantic_sha256"
        ],
        "command_semantic_sha256": authorization_report["command_semantic_sha256"],
    }


def validate_benchmark_capability(
    *,
    authorization_path: str | Path,
    opening_record_path: str | Path,
    actual_benchmark_argv: Sequence[str],
    contract_semantic_sha256: str,
    plan_path: str | Path,
) -> dict[str, Any]:
    authorization, report = load_authorization(authorization_path)
    if list(actual_benchmark_argv) != authorization["benchmark_argv"]:
        raise DedicatedHoldoutExecutionError(
            "benchmark invocation differs from the authorized production command"
        )
    bindings = report["bindings"]
    if bindings["window_consumption_contract_semantic_sha256"] != contract_semantic_sha256:
        raise DedicatedHoldoutExecutionError("window consumption contract identity mismatch")
    if bindings["window_plan_file_sha256"] != file_sha256(plan_path):
        raise DedicatedHoldoutExecutionError("holdout window plan identity mismatch")
    if Path(opening_record_path).resolve() != Path(
        str(bindings["opening_record_path"])
    ).resolve():
        raise DedicatedHoldoutExecutionError("opening record argument mismatch")
    opening = _load_object(opening_record_path, "holdout opening record")
    opening_report = validate_opening_record(
        opening,
        authorization=authorization,
        authorization_report=report,
    )
    return {**report, "opening_record": opening_report}


def validate_consumer_capability(
    *,
    authorization_path: str | Path,
    opening_record_path: str | Path,
    contract_semantic_sha256: str,
    plan_path: str | Path,
) -> dict[str, Any]:
    authorization, report = load_authorization(authorization_path)
    bindings = report["bindings"]
    if Path(opening_record_path).resolve() != Path(
        str(bindings["opening_record_path"])
    ).resolve():
        raise DedicatedHoldoutExecutionError("consumer opening record path mismatch")
    if bindings["window_consumption_contract_semantic_sha256"] != contract_semantic_sha256:
        raise DedicatedHoldoutExecutionError("consumer contract identity mismatch")
    if bindings["window_plan_file_sha256"] != file_sha256(plan_path):
        raise DedicatedHoldoutExecutionError("consumer plan identity mismatch")
    opening = _load_object(opening_record_path, "holdout opening record")
    validate_opening_record(
        opening,
        authorization=authorization,
        authorization_report=report,
    )
    return report


def build_benchmark_command(
    *,
    authorization_path: str | Path,
    python_executable: str,
    repository_root: str | Path,
) -> list[str]:
    authorization, _ = load_authorization(authorization_path)
    entrypoint = Path(repository_root).resolve() / DEDICATED_HOLDOUT_ENTRYPOINT
    if not entrypoint.is_file():
        raise DedicatedHoldoutExecutionError("benchmark entrypoint is missing")
    return [str(python_executable), str(entrypoint), *authorization["benchmark_argv"]]


__all__ = [
    "DEDICATED_HOLDOUT_AUTHORIZATION_VERSION",
    "DEDICATED_HOLDOUT_CALLER_ROLE",
    "DEDICATED_HOLDOUT_ENTRYPOINT",
    "DEDICATED_HOLDOUT_EXECUTION_MODE",
    "DedicatedHoldoutExecutionError",
    "build_benchmark_command",
    "command_semantic_sha256",
    "file_sha256",
    "load_authorization",
    "validate_authorization",
    "validate_benchmark_capability",
    "validate_consumer_capability",
    "validate_opening_record",
]
