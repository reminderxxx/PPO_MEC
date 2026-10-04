from __future__ import annotations

from scripts.audit_native_typed_cache_probe import (
    CAPACITY_BYTES,
    EXPECTED_PROBE_SHA256,
    MIB_BYTES,
    _base_identity_for_adapter,
    build_audit_catalog,
    execute_config,
)


def _reference(adapter_ids: list[str]) -> dict:
    events = []
    residents: list[str] = []
    for index, adapter_id in enumerate(adapter_ids):
        base_id = adapter_id.split(".", 1)[0]
        before = list(residents)
        loaded = [item for item in (base_id, adapter_id) if item not in before]
        residents = list(dict.fromkeys([*before, *loaded]))
        events.append(
            {
                "run_id": "test",
                "request_index": index,
                "adapter_id": adapter_id,
                "base_id": base_id,
                "base_hit": base_id in before,
                "adapter_hit": adapter_id in before,
                "full_hit": base_id in before and adapter_id in before,
                "base_hit_adapter_miss": base_id in before and adapter_id not in before,
                "loaded_ids": loaded,
                "evicted_ids": [],
                "loaded_bytes": sum(96 * MIB_BYTES if item == "b0" else 128 * MIB_BYTES if item == "b1" else 8 * MIB_BYTES for item in loaded),
                "loaded_base_bytes": sum(96 * MIB_BYTES if item == "b0" else 128 * MIB_BYTES if item == "b1" else 0 for item in loaded),
                "loaded_adapter_bytes": sum(8 * MIB_BYTES for item in loaded if ".a" in item),
                "resident_before": before,
                "resident_after": residents,
                "resident_bytes_after": sum(96 * MIB_BYTES if item == "b0" else 128 * MIB_BYTES if item == "b1" else 8 * MIB_BYTES for item in residents),
            }
        )
    return {"events": events}


def test_explicit_unit_mapping_is_exact_for_frozen_capacity() -> None:
    assert CAPACITY_BYTES == 142_606_336
    assert CAPACITY_BYTES / MIB_BYTES == 136.0
    assert EXPECTED_PROBE_SHA256 == "c0b3325874e7427629af8ed36720ed75db5b292c9658d7f3defd39a9deb44bbe"


def test_sharing_off_maps_each_adapter_to_an_independent_base_copy() -> None:
    catalog = build_audit_catalog(sharing_enabled=False)
    base_ids = {
        catalog.get_typed_adapter(adapter_id).required_base_model_id
        for adapter_id in ["b0.a0", "b0.a1", "b0.a2", "b1.a0", "b1.a1", "b1.a2"]
    }
    assert len(base_ids) == 6
    assert _base_identity_for_adapter("b0.a0", False) == "b0::private::b0.a0"
    assert _base_identity_for_adapter("b0.a1", False) == "b0::private::b0.a1"


def test_native_second_interleaved_request_is_rejected_without_mutation() -> None:
    # Reference rows here only satisfy the comparison schema; the assertions are
    # exclusively about the actual native transaction result.
    reference = _reference(["b0.a0", "b1.a0"])
    rows = execute_config(
        workload_label="interleaved",
        sharing_enabled=True,
        reference=reference,
        request_limit=2,
    )
    first, second = rows
    assert first["native"]["transaction_status"] == "committed"
    assert second["native"]["transaction_status"] == "rolled_back_no_mutation"
    assert second["native"]["admission_rejection_reason"] == "insufficient_dependency_safe_evictable_capacity"
    assert second["native"]["eligible_victim_reference_object_ids"] == ["b0.a0"]
    assert second["native"]["actual_eviction_plan"]["ordered_victim_ids"] == ["adapter:b0.a0"]
    assert second["native"]["actual_eviction_plan"]["sufficient"] is False
    assert second["native"]["rejected_state_unchanged"] is True
    assert second["native"]["service_failure"] is True
    assert second["native"]["origin_execution"] is False
    assert second["native"]["cache_event"]["admission_reason"] == "insufficient_dependency_safe_evictable_capacity"
    assert second["native"]["cache_event"]["hit_source"] == "unserved"
    assert second["native"]["cache_event"]["service_success"] is False
