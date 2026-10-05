# Eviction-Cost-Aware Recovery for Stateful AI Workflows at the Vehicular Edge

> Internal working draft — 2026-10-05. This is a complete but non-submission-ready draft. It reports a bounded synthetic validation and same-host calibration, not an independent real-world evaluation. The implemented rule is classified as a combination/implementation correction unless broader evidence establishes an additional contribution.

## Abstract

Recovering a stateful AI workflow after a roadside-unit handoff can avoid recomputing completed nodes, but recovery also prepares model objects at the destination. Under finite cache capacity, that preparation can evict objects needed by the workflow's next declared node and cause an additional reload. We study this narrow coupling within one continuous workflow. Starting from an existing typed base/adapter cache and state-package mechanism, we implement an explicit opt-in rule that compares the full estimated increments of rerun and recovery. The rule uses the native dependency-safe victim plan, deduplicates shared objects, and charges the signed change in model preparation for the next declared node. It does not observe realized future links, hits, or completion times, and falls back to rerun when required inputs are unavailable. In 12 pre-frozen synthetic validation instances, all four compared methods complete 24/24 nodes with no modeled deadline violation. The proposed rule and an information-matched two-step lookahead make identical decisions in all 12 instances and match a post-hoc offline reference in 10, compared with 7 for the original threshold. Their aggregate modeled times are both 156.104 s, versus 174.097 s for the original threshold and 144.675 s for the offline reference. Two cost-estimation cases remain wrong, and the proposed rule has no demonstrated advantage over correct two-step lookahead. These results support an auditable cost-accounting correction in the implemented setting; they do not establish a new general scheduling algorithm, real wireless benefit, cross-workflow cache scheduling, or statistical generalization.

## 1. Introduction

Vehicle-associated AI applications increasingly resemble workflows rather than isolated inference calls. A perception or reasoning stage produces intermediate state that a later stage consumes, while vehicle mobility may move execution from one roadside unit (RSU) to another. At a handoff, the system can rerun the completed prefix at the destination or transfer validated execution state and resume the suffix. The second option appears attractive when the state package is small relative to recomputation.

That comparison is incomplete when model preparation changes the destination cache. A recovered suffix still requires compatible model objects. Preparing the current model at a finite-capacity RSU may evict a base model or adapter already required by the next declared node. The recovery decision therefore affects not only the immediate state-transfer cost but also a later model reload inside the same workflow. Shared base models make the accounting object-dependent: evicting one adapter need not evict or reload a base that is shared by another adapter.

This paper asks one narrow question: **within a declared continuous AI workflow, when should a controller recover execution state after accounting for the cache and loading cost that the recovery itself induces for the next node?** We do not study cross-workflow scheduling, link or compute queues, real radio allocation, or full vehicle/RSU-level multi-agent reinforcement learning. We add no neural network and no new action. The decision remains a choice between the existing rerun action and the existing recovery/prepare action.

The work has three parts. First, it specifies full incremental costs for rerun and recovery over typed model dependencies and a legal native victim plan. Second, it implements the comparison as an opt-in deterministic rule with conservative behavior under missing inputs. Third, it evaluates the rule against the preserved threshold, an information-matched two-step lookahead, and a post-hoc offline reference on 12 pre-frozen synthetic instances. The evidence supports a small implementation correction and exposes its sensitivity to estimation error. It does not yet support an algorithmic novelty claim.

## 2. Related Work

### 2.1 Shared model and adapter caching

SLoRA [1] and Punica [2] show that multiple LoRA adapters can share one base model in multi-tenant serving. POLAR [3] studies joint adapter caching and routing for edge LLM serving. In vehicular edge computing (VEC), dual-dependency-aware caching and offloading [4] jointly reasons about task and service dependencies. These works establish that shared bases, adapter placement, dependency-aware caching, and online control are existing techniques. Our narrower distinction is the cost effect of a recovery-triggered legal victim plan on the next node of the same mobile workflow. We do not claim to invent shared-base caching or adapter routing.

### 2.2 Workflow execution and recovery

WfCommons [5] provides a methodology for representing and generating scientific workflow instances, while recent edge workflow scheduling work such as AWTO [6] includes model-loading effects for agentic workflows. These systems motivate explicit workflow and loading models but do not by themselves validate state recovery across moving RSUs. In our implementation, recovery uses a validated node-boundary state package and resumes a fixed suffix. This is an existing mechanism assembled with typed cache accounting; it is not a new checkpointing theory.

### 2.3 Edge migration and mobility

Trajectory-prediction-assisted task migration in VEC [7] and service-migration/resource-allocation work [8] demonstrate that mobility-aware migration and continuity are established research directions. Proactive service reservation and service fetching provide related destination-preparation abstractions. Our scope differs in the migrated object boundary: execution state and typed base/adapter readiness are represented separately, and we examine how one legal preparation affects a declared subsequent node. We do not model real wireless scheduling, shared resource queues, arbitrary service placement, or unfinished-task migration across multiple workflows.

## 3. System Model

### 3.1 Implemented workflow and cache state

A workflow is a directed acyclic graph (G=(V,E)) with a declared execution order in the bounded experiment. Each node (v\in V) requires an adapter and its compatible base-model dependency set (D_v). The target RSU has a finite typed resident set (C) and capacity (K). Objects are identified explicitly as base models or adapters; workflow state is migration-only and does not consume the model-cache capacity in the current contract.

The implemented cache uses deterministic sequential dependency-safe LRU. For a proposed current-node dependency bundle (D_0), the native read-only preview resolves missing objects, required free capacity, and an ordered legal victim plan (V). A base is not evictable while a resident adapter still depends on it. The recovery transaction is atomic: it either commits all required objects after legal eviction or performs no cache mutation.

The environment exposes five historical actions, but this study uses the same feasible first-action set ({0,4}) for every method. Action 0 serves reactively and, when execution state is unavailable after handoff, incurs one modeled prefix rerun before retrying. Action 4 prepares the target model and exports a create-only workflow-state package. No additional object-selection or scheduling authority is introduced.

### 3.2 Information boundary

The online rules observe the current legal-action mask, target residency and capacity, typed catalog dependencies and sizes, native LRU policy state, the submitted workflow graph, and ex-ante link/recompute/restore estimates. They do not receive future realized link rate, cache hit, victim outcome beyond the deterministic current-action preview, service success, or post-hoc completion time. The offline reference is explicitly different: it chooses only after both complete action-0 and action-4 branches have been realized.

### 3.3 Objective and common costs

The lexicographic evaluation objective first maximizes completed nodes, then minimizes deadline violations, modeled completion time, total transfer, recomputation, service failures, and finally action ID. Cost comparisons are interpreted only at equal completion. Successful service time is identical across the two branches and is excluded from the decision difference. Future objects missing on both paths are retained in both full totals and cancel; they are not selectively omitted from one path.

## 4. Method

### 4.1 Omission in the original threshold

The preserved simple threshold compares state transfer plus restore against input transfer plus prefix recomputation. Its original workload contract treated target-model preparation as common to rerun and recovery. That is correct only when the current decision does not change the relevant resident set or when model costs are genuinely equal. It omits both recovery-specific current model preparation and the additional reload caused by evicting near-term dependencies.

### 4.2 Full incremental costs

Let (C) be the current target resident set, (D_0) the current dependency set, and (V) the legal native victim plan. The projected post-recovery resident set is

\[
C'=(C\setminus V)\cup D_0.
\]

Let (H) contain declared near-term nodes and let

\[
U=\bigcup_{v\in H}D_v
\]

be their dependency union. The union is over object IDs, so a shared base is charged once even if several adapters require it. For a unique object set (B), (T(B)) converts estimated transfer bytes to time using the declared rate and one positive-transfer fixed latency. The full estimated increments are

\[
J_{\mathrm{rerun}}=\hat C_{\mathrm{recompute}}+T(\mathrm{input})+T(U\setminus C),
\]

\[
J_{\mathrm{recover}}=T(D_0\setminus C)+T(\mathrm{state})+\hat C_{\mathrm{restore}}+T(U\setminus C').
\]

The signed future cache effect is

\[
\Delta_{\mathrm{future}}=T(U\setminus C')-T(U\setminus C).
\]

The induced reload set is ((U\setminus C')\setminus(U\setminus C)). Every such object must be explained by the legal victim plan; otherwise the implementation fails conservatively to rerun. The rule selects recovery only if action 4 is legal, all required inputs are valid, the native preview is feasible, and (J_{\mathrm{recover}}<J_{\mathrm{rerun}}).

### 4.3 Pseudocode and complexity

```text
if opt_in is disabled:
    return original_threshold(inputs)
if action4 is illegal or native_preview is infeasible:
    return rerun
V = native_dependency_safe_victim_plan(C, D0, capacity)
C_after = (C - V) union D0
U = unique_objects(dependencies(declared_near_term_nodes))
before = U - C
after = U - C_after
if a required cost is missing or (after - before) is unexplained by V:
    return rerun
J_rerun = recompute_est + transfer(input) + transfer(before)
J_recover = transfer(D0 - C) + transfer(state) + restore_est + transfer(after)
return recover if J_recover < J_rerun else rerun
```

After the native preview, the rule requires (O(|D_0|+|V|+|U|)) time and (O(|C|+|U|)) auxiliary space. The existing sequential LRU preview recomputes dependency-safe candidates after each victim and has worst-case (O(|C|^2)) time. The two-step baseline receives the same fields and explicitly scores the two paths. In the fixed two-action, one-future-node setting, it is algebraically equivalent to the proposed rule.

## 5. Evaluation

### 5.1 Protocol and evidence sources

The implementation and 12 design points were committed before the result artifact was created. The frozen execution commit is `f5033c1b10f8c323f2247a09b1f5ab73a8621387`; the parsed baseline is `1a999287c172b8f34d64680aaaa3d8eb6be23508`. The run executed 24 native branches in 0.635 s with zero training, model calls, downloads, or old-holdout operations. All 30 files in the integrity manifest independently match their recorded sizes and SHA-256 values [A1].

The 12 instances are synthetic validation instances that were not selected after seeing their outputs. Existing native object sizes are 96/128 MiB for the two bases and 8 MiB per adapter. Measured technical-workflow inputs include a 11.035304 s prefix, 192,757-byte rerun input, approximately 2.1 KiB state package, and 0.006151 s same-host serialize/restore/rebuild estimate. Link rates of 70/100/200 Mbps, a 14 s high-restore stress, a 9.5 s recompute sensitivity, and estimator multipliers are explicit assumptions, not new measurements [A2].

Earlier same-host calibration executed three fixed restart/recovery pairs. Recovery avoided the prefix and was 10.650--10.868 s faster in those runs, but they shared one host and uncontrolled operating-system file cache and are not independent deployment measurements [A3]. An earlier equal-completion native cache ledger showed 3,264 MiB for blocked requests with sharing versus 8,640 MiB for interleaved sharing, confirming only that request order and shared-base reuse can change bytes in the synthetic cache [A4]. These prior results calibrate the question; they are not independent validation of the new rule.

### 5.2 Compared methods and metrics

We compare (i) the original simple threshold, (ii) the eviction-cost-aware rule, (iii) an information-matched two-step lookahead that correctly charges eviction effects, and (iv) a same-feasible-set offline reference with post-hoc access to both realized branches. We report completion, modeled completion time, transfer decomposition, recomputation, deadline violations, service failures, and pure decision-function overhead. The offline selection overhead is not online-comparable because it excludes the prerequisite realization of both branches.

### 5.3 Aggregate results at equal completion

All methods complete 24/24 nodes and incur zero modeled deadline violations [A1].

| Method | Offline matches | Recoveries | Failures | Time (s) | Total bytes | Base | Adapter | State | Rerun input | Recompute (s) | Decision median (µs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original threshold | 7/12 | 11 | 1 | 174.096793 | 2,038,648,604 | 1,879,048,192 | 159,383,552 | 24,103 | 192,757 | 11.035304 | 2.438 |
| Eviction-aware | 10/12 | 6 | 6 | 156.103518 | 1,066,522,920 | 973,078,528 | 92,274,688 | 13,162 | 1,156,542 | 64.676520 | 7.875 |
| Two-step lookahead | 10/12 | 6 | 6 | 156.103518 | 1,066,522,920 | 973,078,528 | 92,274,688 | 13,162 | 1,156,542 | 64.676520 | 7.896 |
| Offline reference | 12/12 | 6 | 6 | 144.675011 | 923,916,580 | 838,860,800 | 83,886,080 | 13,158 | 1,156,542 | 64.676520 | post-hoc |

The eviction-aware rule reduces the heterogeneous 12-point sum by 17.993274 s and 972,125,684 bytes relative to the original threshold, but it uses five more reruns and 53.641216 additional recomputation seconds. The time-priority objective prefers avoiding several large model transfers even when this adds a modeled retry. This is a trade-off over constructed instances, not evidence that every system metric improves. The eviction-aware and two-step methods are identical in all outcomes. Their 0.021 µs overhead difference is too small and uncontrolled to support an efficiency claim; the analytical rule's only present advantage is a direct object-level explanation of induced reload.

### 5.4 Complete per-instance results and factorized ablation

The table reports first action and modeled completion time for all four methods; `4` denotes recovery and `0` rerun [A1].

| Instance / factor | Original | Eviction-aware | Two-step | Offline | Event-level interpretation |
|---|---:|---:|---:|---:|---|
| d01 no externality, shared base | 4 / 0.777415 | 4 / 0.777415 | 4 / 0.777415 | 4 / 0.777415 | No victim; only the current 8 MiB adapter is missing. |
| d02 common future load, tight | 4 / 20.258986 | 4 / 20.258986 | 4 / 20.258986 | 4 / 20.258986 | The same b1 bundle is missing on both paths and cancels. |
| d03 common future load, exact fit | 4 / 20.258987 | 4 / 20.258987 | 4 / 20.258987 | 4 / 20.258987 | b0+b1 exactly fit without a current victim. |
| d04 full b1 reload | 4 / 20.258985 | 0 / 11.130725 | 0 / 11.130725 | 0 / 11.130725 | Recovery evicts and reloads 136 MiB b1 base+adapter. |
| d05 shared-base adapter reload | 4 / 1.468504 | 4 / 1.468504 | 4 / 1.468504 | 4 / 1.468504 | Dependency protection retains the shared base; reload is 8 MiB. |
| d06 reverse full b0 reload | 4 / 20.258985 | 0 / 11.130725 | 0 / 11.130725 | 0 / 11.130725 | Object identity, not a hard-coded family, drives the 104 MiB reload. |
| d07 recovery unprofitable at 70 Mbps | 4 / 28.887342 | 0 / 27.455200 | 0 / 27.455200 | 0 / 27.455200 | Current preparation alone reverses the decision. |
| d08 14 s restore stress | 0 / 11.130725 | 0 / 11.130725 | 0 / 11.130725 | 0 / 11.130725 | Recovery is intrinsically unprofitable. |
| d09 both models ready | 4 / 0.086326 | 4 / 0.086326 | 4 / 0.086326 | 4 / 0.086326 | State-only recovery; no cache mutation. |
| d10 recompute underestimated 50% | 4 / 20.258984 | 0 / 22.559231 | 0 / 22.559231 | 4 / 20.258984 | Negative result: richer accounting rejects a beneficial recovery. |
| d11 recompute overestimated 2× | 4 / 20.258985 | 4 / 20.258985 | 4 / 20.258985 | 0 / 11.130725 | Negative result: even correct reload accounting is overturned by cost error. |
| d12 reload underestimated 10% | 4 / 10.192569 | 0 / 9.587710 | 0 / 9.587710 | 0 / 9.587710 | The moderate error does not flip the corrected decision. |

The factorized comparisons isolate the intended mechanisms without claiming statistical effects. d01--d03 and d09 show no externality; d04/d06 exercise full-bundle eviction; d05 shows base-sharing deduplication; d07/d08 make recovery itself unprofitable; and d10--d12 test estimator error. The actual state package varies from 2,175 to 2,215 bytes with instance identity, while every online rule uses the frozen 2,185-byte prior estimate. No method reads the realized package size before acting.

## 6. Limitations

The validation uses one process-local simulator, two RSUs, two model families, two-node workflows, a deterministic cache policy, and 12 constructed instances. It is semi-synthetic: object transitions are executed by the native cache and state machinery, but link time, deadlines, and several stress values are modeled. The technical workflow has no task-quality label. Same-host timing does not represent radio transfer, remote storage, packet loss, queuing, or competing compute.

The cache persists only within one environment episode. We do not implement or evaluate cross-workflow shared scheduling, a shared bandwidth or compute queue, real wireless resource allocation, arbitrary multi-RSU placement, or full MARL. The offline reference has future-information access unavailable online. There are no independent mobility clusters, seeds representing cost uncertainty, confidence intervals, or formal/holdout results for this rule.

Most importantly, the proposed rule is identical to correct two-step lookahead over the tested horizon and has no demonstrated computational advantage. Its two errors under d10/d11 show that improved structural accounting cannot replace calibrated uncertainty. A paper centered on algorithmic novelty would require a capability or evidence beyond this equivalence.

## 7. Conclusion

Recovery and rerun cannot always be compared using state bytes and recomputation alone. In the implemented typed-cache path, a legal recovery can evict model objects required by the next declared workflow node; shared bases determine whether the later reload is a full bundle or only an adapter. An explicit opt-in rule can account for this effect using only current state, the declared graph, a native legal victim preview, and ex-ante costs. In the frozen 12-instance validation, the rule corrects three structural misses of the original threshold and matches the offline reference in 10 cases, but it ties information-matched two-step lookahead exactly and fails in two estimation-error cases. The current evidence therefore supports a transparent implementation correction, not a general new scheduling algorithm or deployment claim.

## References

[1] Y. Sheng et al., “SLoRA: Scalable Serving of Thousands of LoRA Adapters,” *MLSys*, 2024.
[2] G. Chen et al., “Punica: Multi-Tenant LoRA Serving,” *MLSys*, 2024.
[3] “POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving,” arXiv:2604.16583, 2026; publication status not verified in this draft.
[4] “Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehicular Edge Computing,” *IEEE TMC*, 2025, doi:10.1109/TMC.2025.3573379.
[5] R. Ferreira da Silva et al., “WfCommons: A Framework for Enabling Scientific Workflow Research and Development,” *FGCS*, vol. 128, pp. 16–27, 2022.
[6] “AWTO: A Latency-Optimized Task Offloading Scheme for LLM-Driven Agentic Workflows on Heterogeneous Edge,” *FGCS*, 2026, doi:10.1016/j.future.2026.108415.
[7] “Multi-Agent Deep Reinforcement Learning With Trajectory Prediction for Task Migration-Assisted Computation Offloading,” *IEEE TMC*, 2025, doi:10.1109/TMC.2025.3539945.
[8] “Service Satisfaction-Aware Adaptive Service Migration and Resource Allocation in Vehicular Edge Computing,” *IEEE TMC*, 2026, doi:10.1109/TMC.2025.3596342.

## Artifact and claim ledger (not manuscript body)

- [A1] `artifacts/analysis/eviction_aware_recovery_validation_20261005_v1/`: `all_method_results.json/csv`, `aggregate_summary.json`, `event_explanations.json`, `completion_receipt.json`, `frozen_protocol.json`, and `integrity_manifest.json`.
- [A2] `configs/experiment/eviction_aware_recovery_v1.json` and `docs/project/eviction_aware_recovery_plan_20261005.md`: pre-result parameter, method, information, and claim freeze.
- [A3] `artifacts/analysis/production_action4_independent_repeat_20261005_v1/` and `docs/project/workload_v0_1_self_consistency_audit_20261005.md`: three same-host paired technical runs.
- [A4] `artifacts/analysis/mechanism_evidence_closure_20261005_v1/experiment_a_shared_cache_cost.json`: equal-completion native cache byte ledger.
- Review identity: `reviewed_at=2026-10-05`; `literature_cutoff=2026-09-30`; `target_venue=IEEE TMC`; `artifact_run_id=eviction_aware_recovery_validation_20261005_v1`; `policy_version=tmc_review_policy_v3_20260621`; `git_commit=f5033c1b10f8c323f2247a09b1f5ab73a8621387`; `evidence_level=E2_ARTIFACT_AUDITED (bounded synthetic native transition only)`.
- Safe claim: an object-deduplicated, legal-victim-plan-aware cost correction changes decisions on bounded native instances and exposes exact reload objects.
- Prohibited claims: superiority to correct two-step lookahead, online optimality, statistical generalization, real wireless gain, cross-workflow sharing, shared resource queues, real task-quality benefit, full MARL, or TMC-ready status.
