from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_real_cache_victim_reload import (
    CURRENT_ADAPTER,
    FUTURE_ADAPTER,
    _build_env,
    _native_request,
    _preview,
    validate_plan,
)
from src.runtime.peft_adapter_lifecycle import (
    PeftAdapterLifecycleError,
    apply_previewed_adapter_transaction,
    loaded_adapter_names,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/acceptance/real_cache_victim_reload_v1.json"


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_plan_freezes_two_conditions_and_exact_12_call_budget() -> None:
    assert validate_plan(_plan()) == {"planned_generate_calls": 12, "per_condition": 6}


def test_real_byte_capacity_forces_only_adapter_victims() -> None:
    plan = _plan()
    condition = next(row for row in plan["conditions"] if row["condition_id"] == "adapter_victim_reload")
    current = _preview(plan, condition, [], CURRENT_ADAPTER)
    assert current["atomic_transaction_status"] == "committed"
    assert current["evicted_object_ids"] == ["adapter:alpr"]
    assert current["dependency_bundle"]["already_resident_object_ids"] == ["base:smolvlm_500m"]
    future = _preview(plan, condition, [CURRENT_ADAPTER], FUTURE_ADAPTER)
    assert future["atomic_transaction_status"] == "committed"
    assert future["evicted_object_ids"] == ["adapter:helmet"]


def test_no_eviction_control_is_all_resident() -> None:
    plan = _plan()
    condition = next(row for row in plan["conditions"] if row["condition_id"] == "no_eviction_control")
    assert _preview(plan, condition, [], CURRENT_ADAPTER)["atomic_transaction_status"] == "noop_all_resident"
    assert _preview(plan, condition, [CURRENT_ADAPTER], FUTURE_ADAPTER)["atomic_transaction_status"] == "noop_all_resident"


def test_unknown_model_rejection_does_not_commit_logical_state() -> None:
    plan = _plan()
    condition = next(row for row in plan["conditions"] if row["condition_id"] == "adapter_victim_reload")
    env = _build_env(plan, condition)
    before = list(env._typed_resident_object_ids["rsu_target"])
    with pytest.raises(ValueError, match="typed adapter must resolve uniquely"):
        _native_request(env, "missing", 0)
    assert env._typed_resident_object_ids["rsu_target"] == before


class _Parameter:
    shape = (2, 3)
    dtype = "float32"

    def numel(self) -> int:
        return 6

    def element_size(self) -> int:
        return 4


class _BaseModel:
    def __init__(self, parent: "_Model") -> None:
        self.parent = parent

    def delete_adapter(self, name: str) -> None:
        self.parent.peft_config.pop(name)


class _Model:
    def __init__(self, names: list[str]) -> None:
        self.peft_config = {name: {} for name in names}
        self.base_model = _BaseModel(self)
        self.active = names[:1]

    def named_parameters(self):
        for name in self.peft_config:
            yield f"layer.lora_A.{name}.weight", _Parameter()

    def load_adapter(self, path: str, *, adapter_name: str, **kwargs) -> None:
        del path, kwargs
        self.peft_config[adapter_name] = {}

    def set_adapter(self, name: str) -> None:
        self.active = [name]

    def get_model_status(self):
        return SimpleNamespace(active_adapters=self.active)


def test_runtime_bridge_removes_then_loads_real_objects_before_logical_commit(tmp_path: Path) -> None:
    adapter_path = tmp_path / "helmet"
    adapter_path.mkdir()
    (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
    model = _Model([FUTURE_ADAPTER])
    preview = {
        "atomic_transaction_status": "committed",
        "adapter_id": CURRENT_ADAPTER,
        "evicted_typed_objects": [{"object_id": "adapter:alpr", "object_type": "adapter"}],
        "admitted_typed_objects": [{"object_id": "adapter:helmet", "object_type": "adapter"}],
    }
    result = apply_previewed_adapter_transaction(
        model,
        preview=preview,
        adapter_paths={CURRENT_ADAPTER: adapter_path},
        object_to_adapter={"adapter:alpr": FUTURE_ADAPTER, "adapter:helmet": CURRENT_ADAPTER},
        opt_in_enabled=True,
    )
    assert result["status"] == "RUNTIME_APPLIED_PENDING_LOGICAL_COMMIT"
    assert loaded_adapter_names(model) == [CURRENT_ADAPTER]
    assert [row["operation"] for row in result["events"]] == ["unload_adapter", "load_adapter"]


def test_runtime_bridge_rejects_base_victim_before_mutation() -> None:
    model = _Model([FUTURE_ADAPTER])
    with pytest.raises(PeftAdapterLifecycleError, match="adapter-only victims"):
        apply_previewed_adapter_transaction(
            model,
            preview={
                "atomic_transaction_status": "committed",
                "adapter_id": CURRENT_ADAPTER,
                "evicted_typed_objects": [{"object_id": "base:x", "object_type": "base_model"}],
                "admitted_typed_objects": [],
            },
            adapter_paths={},
            object_to_adapter={},
            opt_in_enabled=True,
        )
    assert loaded_adapter_names(model) == [FUTURE_ADAPTER]
