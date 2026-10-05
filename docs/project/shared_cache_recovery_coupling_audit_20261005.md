# Shared cache × workflow recovery coupling audit (2026-10-05)

## Review identity

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-09-30`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `shared_cache_recovery_coupling_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- baseline `main` commit: `3881271a2c740cbcb86cff9bc023b2cf9b134118`
- preserved workload-v0.1 audit commit: `5fd8096d0f3d419e976831bc199aa0f407a456df`
- frozen execution commit: `23e0aa94e8359ed87093ac86f23aa3c8deb9097a`
- `evidence_level`: `E2_bounded_native_transition_witness`
- formal / holdout / training status: `not run / not opened / RL not started`

## Verdict

**Bounded coupling exists, but only inside one live environment episode.** A legal action-4 preparation commits a different finite-capacity target-RSU resident set. A later request consumes that committed set through the same native sequential-LRU dependency transaction, so current preparation can cause later eviction/reload cost and can change the later action mask. This is not an artifact of selecting a different `restore_cost`.

The current benchmark does **not** preserve that state across workflow environments: `reset()` restores catalog-template residents and benchmark workflows are instantiated/reset independently. Shared bandwidth, shared compute capacity and queue waiting are also not production resources in this path. Therefore this audit does not establish cross-workflow contention, wireless deployment gain, or a paper-ready joint optimization result.

**Research decision: 3 — there is a coupling and a fair, interpretable strategy gap.** The frozen current threshold is correct on A/B but wrong on C because it prices current recovery without pricing eviction of the already-resident declared next-node model. The minimal next-round design is an eviction-externality-aware extension of the simple rule, followed by an independent, broader validation. No RL is justified or trained in this round.

## Preserved workload-v0.1 correction

The original audit and all correction artifacts remain unchanged. Its conclusions continue to apply:

1. `local` and `exact` originally consumed the same `restore_cost` ground truth.
2. Their action did not affect later cache or model state.
3. `24/24` agreement was not independent method-validity evidence.
4. Three seeds did not constitute independent cost measurements.
5. The three fixed recovery reruns only support fidelity and savings for the same-host low-cost interval. They cannot be extended to real wireless deployment benefit.

This audit does not revise those conclusions; it asks a different question through a state-changing native transaction.

## Production-path answer before the witness

Inspected path: `GymVecEnv.step` → semantic action decode → `VecWorkflowCoreEnv.step` → native typed-cache transaction → workflow-state export/import → rebuilt semantic state and action mask.

| Question | Classification | Production finding |
|---|---|---|
| 1. Does one legal preparation/recovery change later available cache contents? | Implemented and event-traceable | Within one episode, the atomic typed-cache transaction commits `_typed_resident_object_ids`; later snapshots and requests consume it. `CacheEvent`, pre/post cache snapshots and the committed resident list expose the transition. |
| 2. Does it change another workflow's preparation or eviction cost? | Application-relevant but not implemented | There is no production cross-workflow resident-state lifecycle in the benchmark path. A fresh environment or `reset()` reconstructs template residents. The witness is therefore cross-request/cross-node within one workflow, not cross-workflow. |
| 3. Does it change bandwidth, compute resource or waiting time available to others? | Application-relevant but not implemented | Bytes and time are accounting fields. The path has no shared link budget, compute queue or waiting-time resource that is consumed by the action. |
| 4. Does it change feasible choices at the next handoff/request? | Partially implemented and event-traceable | Resident adapter readiness participates in the action-1 mask; state/model readiness controls advancement. In C, the second-step eligible set changes from `[0,2,3,4]` without preparation to `[0,1,2,3,4]` after action 4 because b1 was evicted and predictive prefetch became legal. There is still no cross-episode choice effect. |

## Frozen protocol and information fairness

The protocol was committed before output inspection. The first command launch used an incorrect typed full hash and stopped at the commit guard before output-root creation or any environment step. The actual single execution then used the correct frozen commit and produced six native branches (two legal first actions for each of three design points), with no retry or tuning.

Post-run source identity checking found a protocol-label defect that is preserved rather than overwritten: the frozen runner's `current_simple_threshold` used current target residency to estimate preparation cost, whereas the workload-v0.1 `mechanism_aware_local_incremental_cost` did not use resident state for discrimination and treated target-model preparation as common to restart/recovery. Recomputing the preserved original rule from its already-frozen ex-ante inputs gives restart `11.07072456 s` versus recovery `0.0263142 s`, so it also selects action 4 for A, B and C. The native branch mapping and all numbers below are therefore action-equivalent and unchanged, but the frozen decision record is not faithful evidence of the original method's information use. The immutable correction is `post_run_method_identity_correction.json`.

Online information boundaries:

- observed current state: the preserved current threshold uses its fixed workload-v0.1 cost inputs and action legality; the existing two-step rule uses current/declared-next adapter identity and action mask. Current resident objects are available to the controller and to the future proposed rule, but were not a discriminating input of the preserved threshold;
- ex-ante estimate: frozen 100 Mbps byte-to-time conversion, `0.006151 s` recovery overhead, estimated input/state bytes `192,757/2,040`, and `11.035304 s` technical-workload prefix recompute value;
- future demand prediction: neither online method receives a future arrival/radio/compute realization;
- declared future node: the workflow's already-declared next node and adapter identity, not a realized future arrival or resource outcome;
- oracle-only value: the post-hoc realized result of first action 0 versus 4 under the same transitions and constraints.

The offline reference enumerates only two first-action branches and uses action 0 afterwards. Its size is not used as complexity evidence.

## Legal controlled identities

All model objects come from the existing native audit catalog:

- family b0: `base:b0` = 96 MiB, `adapter:b0.a0` / `adapter:b0.a1` = 8 MiB each, shared compatible base;
- family b1: `base:b1` = 128 MiB, `adapter:b1.a0` = 8 MiB;
- two RSUs: `rsu_a` and `rsu_b`;
- bottleneck: target typed-cache capacity, either 320 MiB or 136 MiB;
- legal actions: action 0 (current reactive cache/service) and action 4 (predicted handoff model/state prepare).

These identities are contract-valid synthetic audit objects with `formal_use_status=non_formal_native_audit_only`. They are not real model weights and do not support deployment claims.

## Results at equal completion

Every branch completed both nodes, all selected actions were legal, and all deadline-violation counts were zero. Time below is the frozen modeled counterfactual, not wall-clock radio time.

| Instance | Method | First action | Steps / failures | Transfer bytes | Recompute (s) | Modeled completion (s) | Result |
|---|---|---:|---:|---:|---:|---:|---|
| A no competition | current threshold | 4 | 2 / 0 | 8,390,785 | 0 | 0.774762 | gain; matches oracle |
| A no competition | existing two-step | 0 | 3 / 1 | 0 | 11.035304 | 11.095304 | no preparation; one continuity stall |
| A no competition | offline oracle | 4 | 2 / 0 | 8,390,785 | 0 | 0.774762 | reference best |
| B competition, same later load | current threshold | 4 | 2 / 0 | 251,660,431 | 0 | 20.256333 | gain; matches oracle |
| B competition, same later load | existing two-step | 0 | 3 / 1 | 142,606,336 | 11.035304 | 22.523811 | later b1 load is identical, but continuity stalls |
| B competition, same later load | offline oracle | 4 | 2 / 0 | 251,660,431 | 0 | 20.256333 | reference best |
| C future residency effect | current threshold | 4 | 2 / 0 | 251,660,435 | 0 | 20.256334 | **reverse gain: +9.161030 s vs action 0** |
| C future residency effect | existing two-step | 0 | 3 / 1 | 0 | 11.035304 | 11.095304 | matches oracle despite one continuity stall |
| C future residency effect | offline oracle | 0 | 3 / 1 | 0 | 11.035304 | 11.095304 | reference best |

Interpretation by required class:

- A: no capacity competition. Action 4 transfers only the missing 8 MiB adapter because the shared b0 base is already resident; the simple threshold is sufficient.
- B: competition exists and action 4 changes the intermediate resident set, but both branches transfer the same 136 MiB b1 bundle for the later request. The current action does not change that later model-preparation cost.
- C: action 4 changes a later legal state and cost. It evicts the already-resident 136 MiB b1 bundle to install the 104 MiB b0 bundle, then the next request reloads b1. The continuity saving is real in the model, but smaller than the induced reload cost, so the current threshold is worse.

## Event-recomputable C witness

Initial target state at `rsu_b` is exactly `[base:b1, adapter:b1.a0]`, using 136 MiB of a legal 136 MiB capacity. The declared workflow is `n0:b0.a0 → n1:b1.a0`.

### Branch 0: do not prepare

1. Step 1, action 0 executes n0 at `rsu_a`; `rsu_b` remains `[base:b1, adapter:b1.a0]`; transfer = 0.
2. Step 2 reaches `rsu_b`; b1 is model-ready but no workflow-state package exists. Native service fails once, no model transfer occurs, and n1 does not advance.
3. Step 3 retries action 0; n1 completes. Final target state remains `[base:b1, adapter:b1.a0]`.
4. Frozen accounting: model/state transfer 0 bytes; one `11.035304 s` prefix recompute; two successful service quanta; total `11.095304 s`.

### Branch 4: prepare model and workflow state

1. Step 1, action 4 is legal. The capacity transaction evicts `[adapter:b1.a0, base:b1]`, transfers `base:b0` (96 MiB) plus `adapter:b0.a0` (8 MiB), and commits `[base:b0, adapter:b0.a0]`. It exports a create-only 2,195-byte workflow-state package.
2. Step 2, action 0 requests b1. The same native transaction evicts `[adapter:b0.a0, base:b0]`, transfers `base:b1` (128 MiB) plus `adapter:b1.a0` (8 MiB), commits `[base:b1, adapter:b1.a0]`, imports the 2,195-byte state package, and completes n1 without a stall.
3. Frozen accounting: base bytes `234,881,024`, adapter bytes `16,777,216`, workflow-state bytes `2,195`; total `251,660,435`; no recompute; total `20.256334 s`.

Therefore the current decision affects later cost because it commits a different finite-capacity resident set that the next request must consume. The explanation is the b1 eviction and reload, not a different `restore_cost` lookup.

## Failure, no-gain and reverse-gain evidence

- Failure: every action-0 branch has one native continuity failure at the handoff because no state package was exported; it then completes on the retry.
- No later-cache gain: A changes target residency but not later model transfer; B changes residency and competition but later b1 transfer is the same 142,606,336 bytes in both branches.
- Reverse gain: C's action 4 eliminates recompute but adds 251,660,435 transfer bytes and is `9.161030 s` slower under the frozen model.
- Deadline: all branches stay under the frozen 30 s deadline, so there is no violation advantage to hide a completion difference.

## Minimal next-round method design (not implemented here)

Keep the simple threshold and add one observable eviction-externality term:

```text
prepare iff
  current recovery saving
  > current preparation cost
    + cost of evicting resident objects required by already-declared near-term nodes
```

The term may use only the current target resident set, legal transaction preview and the declared workflow graph. Unknown arrivals, future radio/compute realizations and post-hoc outcomes remain unavailable. This is a small mechanism-aware rule, not an RL architecture. It requires an independent validation round with broader legal capacities/workflows and must be compared without oracle leakage.

## Evidence and limitations

- Artifact root: `artifacts/shared_cache_recovery_coupling_20261005_v1/`.
- Integrity: the original 11 listed files independently matched SHA-256 after repository LF normalization of the CSV; original manifest SHA-256 is `c1683dafc80c4e7b97c89e0fa2904696f7f18e0aaff1547eb752a0377a91b971`. The post-run method-identity correction is separately bound to that immutable manifest by `post_run_correction_integrity_manifest.json`.
- Native facts: action legality, pre/post residents, eviction identities, transfer bytes, state import/export, service success and completed nodes.
- Modeled facts: 100 Mbps time, same-host serialize/restore overhead, technical-workload recompute time and 30 s deadline.
- Unverified: real wireless latency/throughput, shared bandwidth/compute/queue contention, cross-workflow persistent cache, statistical generality, formal checkpoint performance, holdout and external validity.

This E2 witness is sufficient to reject the claim that actions *never* affect later cost in the current native episode path. It is insufficient to claim that the full cross-workflow joint decision problem, an RL solution, or deployment benefit has been validated.
