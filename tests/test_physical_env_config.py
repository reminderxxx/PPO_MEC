from __future__ import annotations

import pytest

from src.runtime.physical_env_config import validate_physical_env_config


BASE = {
    "schema_version": "physical_vec_env_v1",
    "parameter_evidence": "synthetic",
    "time_unit": "s",
    "size_unit": "byte",
    "control_step_s": 1,
    "backhaul_bytes_per_s": 10,
    "wireless_bytes_per_s": 10,
    "compute_slots_per_rsu": 1,
    "compute_rate_work_units_per_s": 10,
    "cache_capacity_bytes_per_rsu": 100,
}


def test_physical_config_accepts_explicit_units() -> None:
    assert validate_physical_env_config(BASE)["cache_capacity_bytes_per_rsu"] == 100


@pytest.mark.parametrize("change", [
    {"time_unit": "ms"},
    {"size_unit": "MiB"},
    {"control_step_s": 0},
    {"compute_slots_per_rsu": 0},
    {"unknown": 1},
])
def test_physical_config_rejects_ambiguous_inputs(change: dict) -> None:
    with pytest.raises(ValueError):
        validate_physical_env_config({**BASE, **change})
