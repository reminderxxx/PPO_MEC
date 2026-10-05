from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


STATE_SCHEMA_VERSION = "ppo_mec.two_node_workflow_state.v1"
PRODUCTION_STATE_SCHEMA_VERSION = "ppo_mec.workflow_state.v2"
PACKAGE_SCHEMA_VERSION = "ppo_mec.workflow_state_package.v1"
EXPECTED_WORKFLOW_ID = "technical_vision_description_suffix_v1"
EXPECTED_NODES = ["n0", "n1"]
EXPECTED_EDGES = [["n0", "n1"]]


class StateValidationError(ValueError):
    """Raised before model loading when a recovery state is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload, ensure_ascii=False))
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "payload": copied,
        "payload_sha256": sha256_bytes(canonical_json_bytes(copied)),
    }


def seal_production_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a generic DAG-boundary state without weakening the v1 witness."""
    copied = json.loads(json.dumps(payload, ensure_ascii=False))
    return {
        "schema_version": PRODUCTION_STATE_SCHEMA_VERSION,
        "payload": copied,
        "payload_sha256": sha256_bytes(canonical_json_bytes(copied)),
    }


def validate_production_state_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_workflow: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    expected_input_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a production action-4 boundary before execution ownership moves."""
    _require_equal(
        envelope.get("schema_version"),
        PRODUCTION_STATE_SCHEMA_VERSION,
        "schema_version",
    )
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise StateValidationError("payload must be an object")
    _require_equal(
        envelope.get("payload_sha256"),
        sha256_bytes(canonical_json_bytes(payload)),
        "payload_sha256",
    )

    workflow = payload.get("workflow")
    if not isinstance(workflow, dict):
        raise StateValidationError("workflow must be an object")
    normalized_expected_workflow = json.loads(json.dumps(expected_workflow, ensure_ascii=False))
    for field in ("workflow_id", "nodes", "edges", "execution_order"):
        _require_equal(
            workflow.get(field),
            normalized_expected_workflow.get(field),
            f"workflow.{field}",
        )
    completed = list(payload.get("completed_nodes") or [])
    remaining = list(payload.get("remaining_nodes") or [])
    next_node = payload.get("next_node")
    order = list(expected_workflow.get("execution_order") or [])
    if len(completed) != len(set(completed)) or any(item not in order for item in completed):
        raise StateValidationError("completed_nodes are not a unique workflow prefix")
    expected_completed = order[: len(completed)]
    _require_equal(completed, expected_completed, "completed_nodes")
    _require_equal(remaining, order[len(completed) :], "remaining_nodes")
    _require_equal(next_node, remaining[0] if remaining else None, "next_node")
    if not completed or not remaining:
        raise StateValidationError("action 4 requires a completed prefix and a remaining suffix")
    _require_equal(payload.get("identity"), dict(expected_identity), "identity")
    _require_equal(
        payload.get("input_identity"),
        dict(expected_input_identity),
        "input_identity",
    )
    node_outputs = payload.get("node_outputs")
    if not isinstance(node_outputs, dict) or set(node_outputs) != set(completed):
        raise StateValidationError("node_outputs must cover exactly the completed prefix")
    if any(value in payload for value in ("final_output", "suffix_output")):
        raise StateValidationError("state must not contain a precomputed suffix result")
    return payload


def serialize_state_envelope(envelope: Mapping[str, Any]) -> bytes:
    return json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise StateValidationError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_state_envelope(
    envelope: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _require_equal(envelope.get("schema_version"), STATE_SCHEMA_VERSION, "schema_version")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise StateValidationError("payload must be an object")
    expected_hash = envelope.get("payload_sha256")
    actual_hash = sha256_bytes(canonical_json_bytes(payload))
    _require_equal(expected_hash, actual_hash, "payload_sha256")

    _require_equal(payload.get("workflow_id"), EXPECTED_WORKFLOW_ID, "workflow_id")
    _require_equal(payload.get("nodes"), EXPECTED_NODES, "nodes")
    _require_equal(payload.get("edges"), EXPECTED_EDGES, "edges")
    _require_equal(payload.get("completed_nodes"), ["n0"], "completed_nodes")
    _require_equal(payload.get("remaining_nodes"), ["n1"], "remaining_nodes")
    _require_equal(payload.get("next_node"), "n1", "next_node")
    if "final_output" in payload or "n1_output" in payload:
        raise StateValidationError("state must not contain a precomputed suffix result")

    n0_output = payload.get("n0_output")
    if not isinstance(n0_output, dict):
        raise StateValidationError("n0_output is required")
    raw_text = n0_output.get("raw_text")
    token_ids = n0_output.get("token_ids")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise StateValidationError("n0_output.raw_text must be non-empty")
    if not isinstance(token_ids, list) or not token_ids or not all(isinstance(item, int) for item in token_ids):
        raise StateValidationError("n0_output.token_ids must be a non-empty integer list")
    _require_equal(payload.get("suffix_input_material"), raw_text, "suffix_input_material")
    _require_equal(payload.get("identity"), dict(expected_identity), "identity")

    generation = payload.get("generation")
    if not isinstance(generation, dict):
        raise StateValidationError("generation must be an object")
    for node in EXPECTED_NODES:
        if node not in generation:
            raise StateValidationError(f"generation.{node} is required")

    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict) or rng_state.get("required") is not False or not rng_state.get("reason"):
        raise StateValidationError("rng_state must explicitly declare why continuation state is not required")
    tensor_state = payload.get("tensor_state")
    if not isinstance(tensor_state, dict) or tensor_state.get("required") is not False or not tensor_state.get("reason"):
        raise StateValidationError("tensor_state must explicitly declare why tensors are not required")
    return payload


def build_n1_prompt(template: str, n0_output: str) -> str:
    if template.count("{n0_output}") != 1:
        raise ValueError("n1 prompt template must contain exactly one {n0_output} placeholder")
    if not n0_output.strip():
        raise ValueError("n0 output must be non-empty")
    prompt = template.replace("{n0_output}", n0_output)
    if n0_output not in prompt:
        raise AssertionError("n0 output was not embedded in n1 prompt")
    return prompt


def input_record(*, prompt: str, rendered_prompt: str, input_ids: list[int]) -> dict[str, Any]:
    content = {
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "input_ids": input_ids,
    }
    return {
        **content,
        "content_sha256": sha256_bytes(canonical_json_bytes(content)),
        "contains_n0_output": None,
    }


def loaded_adapter_names(status: Mapping[str, Any]) -> list[str]:
    available = status.get("available_adapters")
    if not isinstance(available, list) or not all(isinstance(item, str) for item in available):
        raise StateValidationError("available_adapters is missing or invalid")
    legacy = status.get("loaded_adapters")
    if legacy is not None and sorted(legacy) != sorted(available):
        raise StateValidationError("loaded_adapters conflicts with available_adapters")
    return sorted(available)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_state_package(package_dir: Path, envelope: Mapping[str, Any]) -> dict[str, Any]:
    if package_dir.exists():
        raise FileExistsError(f"refusing to overwrite state package: {package_dir}")
    package_dir.mkdir(parents=True)
    state_bytes = serialize_state_envelope(envelope)
    state_path = package_dir / "state.json"
    atomic_write(state_path, state_bytes)
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "allowed_files": ["state.json", "manifest.json"],
        "state": {
            "path": "state.json",
            "bytes": len(state_bytes),
            "sha256": sha256_bytes(state_bytes),
        },
        "contains_model_weights": False,
        "contains_final_output": False,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    atomic_write(package_dir / "manifest.json", manifest_bytes)
    return {
        "payload_bytes": len(canonical_json_bytes(envelope["payload"])),
        "state_file_bytes": len(state_bytes),
        "manifest_bytes": len(manifest_bytes),
        "package_bytes": len(state_bytes) + len(manifest_bytes),
        "state_file_sha256": sha256_bytes(state_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "payload_sha256": envelope["payload_sha256"],
        "files": ["state.json", "manifest.json"],
    }


def read_state_package(package_dir: Path, *, expected_identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope, package = read_state_package_envelope(package_dir)
    payload = validate_state_envelope(envelope, expected_identity=expected_identity)
    return payload, package


def read_state_package_envelope(package_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and integrity-check a package before schema-specific validation."""
    actual_files = sorted(path.name for path in package_dir.iterdir() if path.is_file())
    manifest_path = package_dir / "manifest.json"
    state_path = package_dir / "state.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_equal(manifest.get("schema_version"), PACKAGE_SCHEMA_VERSION, "package schema_version")
    _require_equal(actual_files, sorted(manifest.get("allowed_files", [])), "package files")
    state_bytes = state_path.read_bytes()
    _require_equal(len(state_bytes), manifest["state"]["bytes"], "state bytes")
    _require_equal(sha256_bytes(state_bytes), manifest["state"]["sha256"], "state file hash")
    _require_equal(manifest.get("contains_model_weights"), False, "contains_model_weights")
    _require_equal(manifest.get("contains_final_output"), False, "contains_final_output")
    envelope = json.loads(state_bytes)
    return envelope, {
        "state_file_sha256": sha256_bytes(state_bytes),
        "manifest_sha256": sha256_file(manifest_path),
        "package_bytes": len(state_bytes) + manifest_path.stat().st_size,
        "payload_sha256": envelope["payload_sha256"],
        "files": actual_files,
    }


def read_production_state_package(
    package_dir: Path,
    *,
    expected_workflow: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    expected_input_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope, package = read_state_package_envelope(package_dir)
    payload = validate_production_state_envelope(
        envelope,
        expected_workflow=expected_workflow,
        expected_identity=expected_identity,
        expected_input_identity=expected_input_identity,
    )
    return payload, package
