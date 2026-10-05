"""Explicit opt-in bridge between typed-cache events and a PEFT runtime.

The typed-cache ledger remains the authority for legal victims and admissions.
This module only materializes an already-previewed adapter-only transaction in
one loaded PEFT model.  It deliberately does not delete weight files, clear the
OS file cache, or pretend that RSS is proof of unloading.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Mapping


class PeftAdapterLifecycleError(RuntimeError):
    """Fail-closed lifecycle bridge error."""


def loaded_adapter_names(model: Any) -> list[str]:
    """Return adapter objects registered in the live PEFT model."""

    return sorted(str(name) for name in dict(model.peft_config).keys())


def adapter_tensor_inventory(model: Any, adapter_name: str) -> dict[str, Any]:
    """Count live tensors owned by one named adapter.

    PEFT stores named LoRA parameters under a module key containing the adapter
    name.  The config membership check prevents an absent adapter from being
    mistaken for a zero-sized loaded object.
    """

    registered = adapter_name in loaded_adapter_names(model)
    rows = []
    if registered:
        marker = f".{adapter_name}."
        for name, parameter in model.named_parameters():
            if marker in name:
                rows.append(
                    {
                        "name": str(name),
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "numel": int(parameter.numel()),
                        "bytes": int(parameter.numel() * parameter.element_size()),
                    }
                )
    return {
        "adapter_name": adapter_name,
        "registered": registered,
        "tensor_count": len(rows),
        "tensor_numel": sum(row["numel"] for row in rows),
        "tensor_bytes": sum(row["bytes"] for row in rows),
        "tensor_name_sha256": hashlib.sha256(
            "\n".join(sorted(row["name"] for row in rows)).encode("utf-8")
        ).hexdigest(),
        "tensor_name_examples": [row["name"] for row in rows[:5]],
    }


def runtime_snapshot(model: Any, adapter_names: list[str]) -> dict[str, Any]:
    """Record object registration and tensor evidence without using RSS."""

    registered = loaded_adapter_names(model)
    active: list[str] = []
    try:
        status = model.get_model_status()
        active = [str(name) for name in list(status.active_adapters)]
    except BaseException:
        # A PEFT model with no remaining adapter can make get_model_status fail.
        # Registration and tensor inventories remain the unload authority.
        active = []
    return {
        "registered_adapter_names": registered,
        "active_adapter_names": active,
        "adapter_objects": {
            name: adapter_tensor_inventory(model, name) for name in adapter_names
        },
    }


def unload_adapter(model: Any, adapter_name: str) -> dict[str, Any]:
    """Remove a live adapter object through PEFT's tuner API."""

    before = adapter_tensor_inventory(model, adapter_name)
    if not before["registered"] or before["tensor_count"] <= 0:
        raise PeftAdapterLifecycleError(
            f"adapter is not a loaded tensor object: {adapter_name}"
        )
    started = time.monotonic()
    delete = getattr(model.base_model, "delete_adapter", None)
    if not callable(delete):
        raise PeftAdapterLifecycleError(
            "installed PEFT tuner does not support delete_adapter"
        )
    delete(adapter_name)
    elapsed = time.monotonic() - started
    after = adapter_tensor_inventory(model, adapter_name)
    if after["registered"] or after["tensor_count"] != 0:
        raise PeftAdapterLifecycleError(
            f"adapter remained registered after unload: {adapter_name}"
        )
    return {
        "operation": "unload_adapter",
        "adapter_name": adapter_name,
        "seconds": elapsed,
        "before": before,
        "after": after,
    }


def load_adapter(
    model: Any,
    *,
    adapter_name: str,
    adapter_path: Path,
) -> dict[str, Any]:
    """Load one adapter from existing local files and prove live tensors exist."""

    if adapter_name in loaded_adapter_names(model):
        raise PeftAdapterLifecycleError(f"adapter is already loaded: {adapter_name}")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise PeftAdapterLifecycleError(
            f"missing local adapter weights: {adapter_path}"
        )
    started = time.monotonic()
    model.load_adapter(
        str(adapter_path),
        adapter_name=adapter_name,
        is_trainable=False,
        local_files_only=True,
    )
    elapsed = time.monotonic() - started
    after = adapter_tensor_inventory(model, adapter_name)
    if not after["registered"] or after["tensor_count"] <= 0:
        raise PeftAdapterLifecycleError(
            f"adapter load produced no registered tensors: {adapter_name}"
        )
    return {
        "operation": "load_adapter",
        "adapter_name": adapter_name,
        "adapter_path": str(adapter_path),
        "seconds": elapsed,
        "after": after,
    }


def apply_previewed_adapter_transaction(
    model: Any,
    *,
    preview: Mapping[str, Any],
    adapter_paths: Mapping[str, Path],
    object_to_adapter: Mapping[str, str],
    opt_in_enabled: bool,
) -> dict[str, Any]:
    """Materialize one legal native preview, committing no logical state.

    The caller must commit the same native transaction only after this function
    succeeds.  Unsupported base-model unloading and unexplained victims reject
    before any runtime mutation.
    """

    if not opt_in_enabled:
        raise PeftAdapterLifecycleError("real adapter lifecycle bridge is opt-in only")
    status = str(preview.get("atomic_transaction_status"))
    if status not in {"committed", "noop_all_resident"}:
        raise PeftAdapterLifecycleError(
            f"native preview is not executable: {status}"
        )
    evicted_rows = list(preview.get("evicted_typed_objects") or [])
    admitted_rows = list(preview.get("admitted_typed_objects") or [])
    if any(row.get("object_type") != "adapter" for row in evicted_rows):
        raise PeftAdapterLifecycleError(
            "runtime bridge supports adapter-only victims; base unload is unsupported"
        )
    if any(row.get("object_type") != "adapter" for row in admitted_rows):
        raise PeftAdapterLifecycleError(
            "runtime bridge requires the shared base to remain loaded"
        )

    victim_names = []
    for row in evicted_rows:
        object_id = str(row["object_id"])
        if object_id not in object_to_adapter:
            raise PeftAdapterLifecycleError(f"unmapped legal victim: {object_id}")
        victim_names.append(object_to_adapter[object_id])
    admitted_names = []
    for row in admitted_rows:
        object_id = str(row["object_id"])
        if object_id not in object_to_adapter:
            raise PeftAdapterLifecycleError(f"unmapped admission: {object_id}")
        name = object_to_adapter[object_id]
        if name not in adapter_paths:
            raise PeftAdapterLifecycleError(f"missing adapter path mapping: {name}")
        admitted_names.append(name)

    events = []
    for name in victim_names:
        events.append(unload_adapter(model, name))
    for name in admitted_names:
        events.append(
            load_adapter(model, adapter_name=name, adapter_path=adapter_paths[name])
        )
    requested = str(preview.get("adapter_id"))
    if requested:
        if requested not in loaded_adapter_names(model):
            raise PeftAdapterLifecycleError(
                f"requested adapter is not loaded after transaction: {requested}"
            )
        model.set_adapter(requested)
    return {
        "status": "RUNTIME_APPLIED_PENDING_LOGICAL_COMMIT",
        "native_transaction_status": status,
        "requested_adapter": requested,
        "victim_adapter_names": victim_names,
        "admitted_adapter_names": admitted_names,
        "events": events,
    }
