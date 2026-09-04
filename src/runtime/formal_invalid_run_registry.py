"""One fail-closed registry for permanently invalid formal run references."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


PERMANENTLY_INVALID_FORMAL_RUN_IDS = (
    "typed_model_cache_formal_20260820_g14c_351fdb8_v1",
    "typed_model_cache_formal_20260820_164251_g14c_v2",
    "typed_model_cache_formal_20260820_203430_g14c_v3",
    "typed_model_cache_formal_20260824_110016_g14c_v4",
    "typed_model_cache_formal_20260824_235839_g14c_v4",
    "typed_model_cache_formal_20260825_111625_g14c_v5",
    "typed_model_cache_formal_20260825_135122_g14c_v6",
    "typed_model_cache_formal_20260826_233222_g14c_v7",
    "typed_model_cache_formal_20260828_101804_g14c_v8",
    "typed_model_cache_formal_20260830_113339_g14c_v9",
    "typed_model_cache_formal_20260901_155201_g14c_v11",
    "typed_model_cache_formal_20260902_162203_g14c_v12",
)
G14C_V12_RUN_ID = PERMANENTLY_INVALID_FORMAL_RUN_IDS[-1]
G14C_V12_RUN_ROOT = (
    "artifacts/experiments/typed_model_cache_formal/" + G14C_V12_RUN_ID
)


class PermanentlyInvalidFormalReferenceError(ValueError):
    """Raised before an invalid run, staging cell, or checkpoint can be consumed."""


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, (str, Path)):
        yield str(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _strings(item)


def reject_permanently_invalid_formal_references(values: Iterable[Any]) -> None:
    for value in values:
        for text in _strings(value):
            normalized = text.replace("\\", "/")
            matched = next(
                (run_id for run_id in PERMANENTLY_INVALID_FORMAL_RUN_IDS if run_id in normalized),
                None,
            )
            if matched:
                raise PermanentlyInvalidFormalReferenceError(
                    "permanently invalid formal run/staging/checkpoint reference rejected: "
                    + matched
                )


__all__ = [
    "G14C_V12_RUN_ID",
    "G14C_V12_RUN_ROOT",
    "PERMANENTLY_INVALID_FORMAL_RUN_IDS",
    "PermanentlyInvalidFormalReferenceError",
    "reject_permanently_invalid_formal_references",
]
