"""Versioned physical-cost inputs and a causally gated output replay.

This module does not contain measured values. Callers must provide values and
their provenance; synthetic parameters remain explicitly marked synthetic.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Mapping


COST_PROFILE_SCHEMA_VERSION = "1.0.0"
CostEvidence = Literal["measured", "synthetic"]


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def canonical_identity(value: Any) -> str:
    """Hash a JSON-compatible model input or generation configuration."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("identity value must be finite JSON-compatible data") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReplayIdentity:
    model_identity: str
    input_identity: str
    generation_identity: str

    def __post_init__(self) -> None:
        for name in ("model_identity", "input_identity", "generation_identity"):
            _nonempty(getattr(self, name), name)

    @classmethod
    def from_components(
        cls, *, model_identity: str, input_value: Any, generation_config: Mapping[str, Any]
    ) -> "ReplayIdentity":
        return cls(model_identity, canonical_identity(input_value), canonical_identity(generation_config))


@dataclass(frozen=True)
class CostParameter:
    seconds: float
    evidence: CostEvidence
    source: str

    def __post_init__(self) -> None:
        _nonnegative(self.seconds, "seconds")
        if self.evidence not in ("measured", "synthetic"):
            raise ValueError("evidence must be measured or synthetic")
        _nonempty(self.source, "source")


@dataclass(frozen=True)
class PhysicalCostProfile:
    """Seconds, bytes and abstract work units are the internal physical units."""

    profile_id: str
    replay_identity: ReplayIdentity
    cold_load: CostParameter
    hot_call: CostParameter
    adapter_switch: CostParameter
    state_restore: CostParameter
    schema_version: str = COST_PROFILE_SCHEMA_VERSION
    time_unit: str = "s"
    size_unit: str = "byte"
    work_unit: str = "work_unit"

    def __post_init__(self) -> None:
        _nonempty(self.profile_id, "profile_id")
        if self.schema_version != COST_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported cost profile schema: {self.schema_version}")
        if (self.time_unit, self.size_unit, self.work_unit) != ("s", "byte", "work_unit"):
            raise ValueError("cost profile units must be seconds, bytes and work units")
        if not isinstance(self.replay_identity, ReplayIdentity):
            raise ValueError("replay_identity must be ReplayIdentity")
        for name in ("cold_load", "hot_call", "adapter_switch", "state_restore"):
            if not isinstance(getattr(self, name), CostParameter):
                raise ValueError(f"{name} must be CostParameter")

    @property
    def evidence(self) -> CostEvidence:
        parameters = (self.cold_load, self.hot_call, self.adapter_switch, self.state_restore)
        return "measured" if all(item.evidence == "measured" for item in parameters) else "synthetic"

    def validate_replay_identity(self, actual: ReplayIdentity) -> None:
        if actual != self.replay_identity:
            raise ValueError("replay model/input/generation identity mismatch")


class ReplayOutputGate:
    """Release a replayed output only after its simulated completion event.

    ``mark_completed`` must be called by the execution engine when it emits
    the compute-completed event. Elapsed wall time alone cannot reveal output.
    """

    def __init__(self, *, profile: PhysicalCostProfile, output: Any, earliest_completion_s: float):
        _nonnegative(earliest_completion_s, "earliest_completion_s")
        self._profile = profile
        self._output = deepcopy(output)
        self._earliest_completion_s = float(earliest_completion_s)
        self._completed_at_s: float | None = None

    @property
    def completed_at_s(self) -> float | None:
        return self._completed_at_s

    def mark_completed(self, *, time_s: float, identity: ReplayIdentity) -> None:
        self._profile.validate_replay_identity(identity)
        _nonnegative(time_s, "time_s")
        if time_s < self._earliest_completion_s:
            raise ValueError("output cannot complete before simulated compute completion")
        if self._completed_at_s is not None:
            raise ValueError("replay output completion already recorded")
        self._completed_at_s = float(time_s)

    def read(self, *, time_s: float, identity: ReplayIdentity) -> Any:
        self._profile.validate_replay_identity(identity)
        _nonnegative(time_s, "time_s")
        if self._completed_at_s is None or time_s < self._completed_at_s:
            raise ValueError("replay output is unavailable before simulated completion")
        return deepcopy(self._output)
