"""Strict opt-in configuration for the versioned physical VEC environment."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping


PHYSICAL_ENV_CONFIG_VERSION = "physical_vec_env_v1"
_POSITIVE = {
    "control_step_s", "backhaul_bytes_per_s", "wireless_bytes_per_s",
    "compute_rate_work_units_per_s", "cache_capacity_bytes_per_rsu",
}
_INTEGER_POSITIVE = {"compute_slots_per_rsu"}
_OPTIONAL = {"root_input_payload_identity"}


def validate_physical_env_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return constructor kwargs; reject silent unit, identity, or schema drift."""
    if not isinstance(payload, Mapping):
        raise ValueError("physical environment config must be a mapping")
    if payload.get("schema_version") != PHYSICAL_ENV_CONFIG_VERSION:
        raise ValueError("unsupported physical environment config schema_version")
    if payload.get("parameter_evidence") not in {"synthetic", "measured"}:
        raise ValueError("parameter_evidence must be synthetic or measured")
    if payload.get("time_unit") != "s" or payload.get("size_unit") != "byte":
        raise ValueError("physical environment units must be seconds and bytes")
    allowed = _POSITIVE | _INTEGER_POSITIVE | _OPTIONAL | {
        "schema_version", "parameter_evidence", "time_unit", "size_unit"
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown physical environment config fields: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key in _POSITIVE:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a positive finite number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{key} must be a positive finite number")
        if key == "cache_capacity_bytes_per_rsu":
            if not number.is_integer():
                raise ValueError("cache_capacity_bytes_per_rsu must be an integer number of bytes")
            kwargs[key] = int(number)
        else:
            kwargs[key] = number
    for key in _INTEGER_POSITIVE:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        kwargs[key] = value
    if "root_input_payload_identity" in payload:
        identity = payload["root_input_payload_identity"]
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("root_input_payload_identity must be a nonempty string")
        kwargs["root_input_payload_identity"] = identity
    return kwargs


def load_physical_env_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return validate_physical_env_config(payload), dict(payload)


def physical_env_config_identity(payload: Mapping[str, Any]) -> str:
    validate_physical_env_config(payload)
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
