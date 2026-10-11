"""Export complete pre-action clone states for an already completed branch run."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_calibrated_workflow_interface_repair import _sha256, _write_json
from scripts.run_cscwd_service_feasible_action_branches import (
    OUTPUT, STATE_FIELDS, _hash, _prepare, _rebuild, _source_key, _stable,
)


def main() -> None:
    if not (OUTPUT / "analysis_manifest.json").is_file():
        raise RuntimeError("completed branch analysis absent")
    destination = OUTPUT / "state_snapshots"
    if destination.exists():
        raise RuntimeError("snapshot export already exists")
    config, instances, episodes, mappings, current = _prepare()
    frozen = json.loads((OUTPUT / "preflight_receipt.json").read_text(encoding="utf-8"))
    if current["mappings"] != frozen["mappings"]:
        raise RuntimeError("preflight source mapping drift")
    destination.mkdir()
    unique = {}
    for mapping in mappings:
        env, observation, info, _ = _rebuild(config, instances[mapping["design_id"]],
                                              episodes[_source_key(mapping)], int(mapping["step_index"]))
        if frozenset(vars(env)) != STATE_FIELDS:
            raise RuntimeError("environment state contract drift")
        payload = {key: getattr(env, key) for key in STATE_FIELDS - {"_mask_builder", "_rng", "caches"}}
        payload["_rng"] = env._rng.bit_generator.state
        payload["_mask_builder"] = [vars(item) for item in env._mask_builder._schema._actions]
        payload["caches"] = {key: vars(value) for key, value in env.caches.items()}
        payload["policy_input_observation"] = [float(item) for item in observation]
        payload["policy_input_info"] = info
        if _hash(payload) != mapping["full_state_sha256"]:
            raise RuntimeError("complete snapshot hash mismatch")
        state_hash = mapping["full_state_sha256"]
        if state_hash in unique:
            continue
        path = destination / f"{state_hash}.json"
        _write_json(path, _stable(payload))
        unique[state_hash] = str(path.relative_to(OUTPUT))
    if len(unique) != len({row["full_state_sha256"] for row in mappings}):
        raise RuntimeError("snapshot export count mismatch")
    manifest = {"schema_version": "cscwd_action_branch_state_snapshot_export_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "branch_analysis_manifest_sha256": _sha256(OUTPUT / "analysis_manifest.json"),
                "preflight_receipt_sha256": _sha256(OUTPUT / "preflight_receipt.json"),
                "state_count": len(unique), "source_mapping_count": len(mappings),
                "state_files": {state_hash: {"path": name, "sha256": _sha256(OUTPUT / name)}
                                for state_hash, name in sorted(unique.items())}}
    _write_json(OUTPUT / "snapshot_export_manifest.json", manifest)
    print(json.dumps({"complete_states": len(unique), "source_mapping_rows": len(mappings)}))


if __name__ == "__main__":
    main()
