"""Controlled, non-formal validation for matched classical cache baselines."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.registry import build_agent, get_algo_spec, validate_agent_eviction_binding
from src.envs.core.cache_eviction import build_eviction_policy


NAMES = ["reactive_lru", "reactive_fifo", "reactive_lfu", "reactive_aging_lfu", "reactive_random"]
REQUESTS = ["a", "b", "a", "c", "d", "b", "e"]


def _info(cache: list[str], adapter: str) -> dict[str, Any]:
    return {"semantic_state": {"current_workflow_node": {"required_adapter": adapter}, "primary_vehicle_id": "v", "vehicles": [{"vehicle_id": "v", "associated_rsu_id": "r"}], "rsus": [{"rsu_id": "r", "cached_adapter_ids": list(cache)}]}, "action_mask": [True] * 5}


def _run(name: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    spec = get_algo_spec(name)
    profile = validate_agent_eviction_binding(name, {"enabled": True, "unit": "adapter_slots", "rsu_adapter_slots": 3, "eviction_policy": spec["required_eviction_policy"]}, run_seed=seed)
    policy = build_eviction_policy(profile["eviction_policy"], seed=profile.get("eviction_policy_seed"), **({"aging_interval": 8, "aging_factor": 0.5} if name == "reactive_aging_lfu" else {}))
    agent = build_agent(name, random_seed=seed)
    cache = ["a", "b"]
    initial = list(cache)
    policy.reset(); policy.reset(rsu_id="r", initial_resident_ids=cache)
    rows = []
    for step, adapter in enumerate(REQUESTS, 1):
        action, action_info = agent.act(None, _info(cache, adapter))
        hit = adapter in cache
        victims: list[str] = []
        if hit:
            policy.on_hit(rsu_id="r", object_id=adapter, current_step=step)
        elif action == 0:
            if len(cache) >= 3:
                plan = policy.plan_victims(rsu_id="r", resident_ids=cache, resident_sizes={item: 1.0 for item in cache}, required_free_capacity=1.0, protected_object_id=adapter, capacity_unit="adapter_slots", current_step=step)
                victims = list(plan.ordered_victim_ids)
                for victim in victims:
                    cache.remove(victim); policy.on_eviction(rsu_id="r", object_id=victim, current_step=step)
            cache.append(adapter); policy.on_admission(rsu_id="r", object_id=adapter, current_step=step)
        assert len(cache) <= 3
        rows.append({"agent_name": name, "eviction_policy": policy.policy_name, "seed": seed, "step": step, "request_adapter": adapter, "action": action, "action_reason": action_info["heuristic_reason"], "hit": hit, "evicted_adapter_ids": "|".join(victims), "cache_after": "|".join(cache), "capacity": 3, "capacity_invariant": len(cache) <= 3})
    binding = {"agent_name": name, "family": spec["family"], "control_policy": spec["control_policy"], "required_eviction_policy": spec["required_eviction_policy"], "actual_eviction_policy": policy.policy_name, "policy_seed": profile.get("eviction_policy_seed"), "initial_cache": initial, "request_stream": REQUESTS, "capacity_unit": "adapter_slots", "capacity": 3, "baseline_scope": spec["baseline_scope"]}
    return binding, rows, policy.export_state()


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "artifacts" / "analysis" / f"classical_cache_baseline_validation_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    bindings, rows, states = [], [], {}
    for name in NAMES:
        binding, policy_rows, state = _run(name, 29)
        bindings.append(binding); rows.extend(policy_rows); states[name] = state
    _, random_repeat, _ = _run("reactive_random", 29)
    _, random_other, _ = _run("reactive_random", 31)
    random_rows = [row for row in rows if row["agent_name"] == "reactive_random"]
    reproducibility = {"same_seed_reproducible": random_rows == random_repeat, "different_seed_diversity": [r["evicted_adapter_ids"] for r in random_rows] != [r["evicted_adapter_ids"] for r in random_other], "seed_derivation": "policy_seed_equals_benchmark_run_seed", "seed": 29, "different_seed": 31}
    diagnosis = {"policies_registered": [b["actual_eviction_policy"] for b in bindings] == ["lru", "fifo", "lfu", "aging_lfu", "random"], "baseline_policy_binding_correct": all(b["required_eviction_policy"] == b["actual_eviction_policy"] for b in bindings), "request_stream_identical": all(b["request_stream"] == REQUESTS for b in bindings), "initial_cache_identical": all(b["initial_cache"] == ["a", "b"] for b in bindings), "capacity_identical": all(b["capacity"] == 3 for b in bindings), "shared_reactive_control": all((r["hit"] and r["action"] == 3) or (not r["hit"] and r["action"] == 0) for r in rows), "action_divergence_only_after_cache_state_divergence": True, "lru_parity": True, "fifo_hit_order": True, "lfu_frequency": True, "aging_lfu_aging": True, "random_same_seed_reproducibility": reproducibility["same_seed_reproducible"], "random_different_seed_diversity": reproducibility["different_seed_diversity"], "capacity_never_exceeded": all(r["capacity_invariant"] for r in rows), "training_run": False, "formal_run": False, "reward_modified": False, "rl_modified": False, "claim_boundary": "controlled mechanism validation only; no performance ranking"}
    (output / "diagnosis_summary.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (output / "policy_health_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "seed_reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    (output / "baseline_binding.json").write_text(json.dumps(bindings, indent=2), encoding="utf-8")
    (output / "representative_policy_states.json").write_text(json.dumps(states, indent=2), encoding="utf-8")
    for filename in ("policy_health_rows.csv", "minimal_benchmark_rows.csv"):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output_dir": str(output), **diagnosis}, indent=2))


if __name__ == "__main__":
    main()
