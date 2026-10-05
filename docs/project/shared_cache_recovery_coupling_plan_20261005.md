# Shared cache × workflow recovery coupling — frozen plan (2026-10-05)

## 1. Purpose and preserved baseline

This is a bounded, non-training audit of whether a legal production cache/recovery action can change the state and cost of a later request. It does not introduce an RL method, a network simulator, a new model call, or a new formal benchmark.

- Baseline `main` commit before this task: `3881271a2c740cbcb86cff9bc023b2cf9b134118`.
- Preserved workload-v0.1 audit conclusion commit: `5fd8096d0f3d419e976831bc199aa0f407a456df`.
- The full original audit, corrections, mismatch artifacts and fixed recovery repeats are preserved unchanged in history and in `docs/project/workload_v0_1_self_consistency_audit_20261005.md`.
- The preserved conclusion remains binding: local and exact originally shared the same `restore_cost` truth; the action did not affect later cache/model state; 24/24 agreement was not independent method-validity evidence; three seeds were not independent cost measurements. The three fixed reruns support only same-host low-cost fidelity/savings and do not establish wireless deployment gain.

## 2. Production-path classification before experiment

The inspected path is `GymVecEnv.step` → action decode → `VecWorkflowCoreEnv.step` → native typed-cache transaction → state rebuild. No paper-level capability is assumed.

| Question | Classification | Traceable production evidence / boundary |
|---|---|---|
| Does one legal preparation/recovery change later cache contents? | Implemented and event-traceable, but only inside one environment episode | The native sequential LRU transaction commits `_typed_resident_object_ids`; later cache snapshots and request events observe it. `reset()` reconstructs residents from the catalog template. |
| Does it change another workflow's preparation or eviction cost? | Application-relevant but not yet implemented across benchmark workflows | A shared resident set could cause the effect, but current benchmark workflows instantiate/reset environments independently. This audit therefore uses two requests/nodes of one declared workflow and makes no cross-workflow claim. |
| Does it change available bandwidth, compute resource, or queueing wait? | Application-relevant but not implemented | Transfer and restore/recompute time are accounting estimates. There is no production shared-bandwidth or shared-compute queue depleted by the action. |
| Does it change feasible choices at the next handoff? | Partially implemented and event-traceable inside an episode | Target adapter residency affects the action mask for predictive prefetch, while model readiness/state migration affects whether service can advance. No cross-episode persistence is claimed. |

Thus the only reusable production coupling eligible for a witness is finite-capacity typed-cache residency within one normal episode. Cross-workflow sharing remains a gap.

## 3. Frozen witness and comparison

The machine-readable protocol is `configs/experiment/shared_cache_recovery_coupling_v1.json`. It fixes three points before any result is viewed:

- A: ample capacity, b0 shared base, no competition.
- B: 136 MiB target capacity, b0→b1, empty target; both branches must load b1 later.
- C: 136 MiB target capacity, b0→b1, b1 initially resident; preparing b0 is a legal replacement that may force b1 reload.

All identities and transfers use the existing synthetic native audit catalog and its validated dependency graph. They are contract-legal but non-formal and are not real model artifacts. There are two RSUs, two families, one shared-base relation and one bottleneck (target typed-cache capacity).

Compared methods are frozen to:

1. `current_simple_threshold`: current state/current target residency plus frozen ex-ante costs; no realized future.
2. `existing_two_step_rule`: the existing declared-next-node same-adapter rule; no realized future.
3. `offline_enumeration_reference`: post-hoc enumeration of legal first action 0 versus 4, with action 0 thereafter. This is an oracle reference, not an online method.

The declared next node is known at request time because it is part of the workflow graph. Future arrivals, radio realizations, compute realizations and cache outcomes are not given to either online rule.

## 4. Cost and claim boundary

Native events establish residents, evictions, transfer bytes, action legality, service success/failure and completed nodes. Modeled time adds the already-recorded 100 Mbps transfer conversion, the same-host state package overhead, and the technical-workload prefix recompute value. These are not wall-clock wireless measurements. The objective first equalizes completion, then violations, modeled time, bytes, recompute and failures.

The run is allowed to show no gain or reverse gain. No design point, weight, capacity or comparator may be changed after results. Offline enumeration size is not used as complexity evidence.

## 5. Frozen execution and output list

After committing this plan/config/runner, resolve `${FREEZE_COMMIT}` once as `git rev-parse HEAD`, then execute exactly once:

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/run_shared_cache_recovery_coupling.py \
  --config configs/experiment/shared_cache_recovery_coupling_v1.json \
  --output-root artifacts/shared_cache_recovery_coupling_20261005_v1 \
  --expected-git-commit ${FREEZE_COMMIT}
```

Required outputs are `frozen_protocol.json`, `all_method_results.json`, `all_method_results.csv`, `event_witness.json`, `completion_receipt.json`, `integrity_manifest.json`, and the create-only `runtime_state_packages/` evidence directory. The result audit will be written only after this frozen run.
