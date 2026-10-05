# Cost-Accounted Recovery for Stateful AI Workflows at the Vehicular Edge

> Internal working draft — revised 2026-10-05. This is a complete but non-submission-ready mechanism/empirical draft. It separates new real-model measurements, an estimated cost model, and synthetic development cases. It does not claim a new cache-replacement algorithm, a decision-boundary test, or deployment generality.

## Abstract

Recovering a stateful AI workflow after a roadside-unit handoff can avoid recomputing completed nodes, but recovery also requires model readiness and can change a finite destination cache. We study this coupling within one declared continuous workflow. The implemented system exports a validated node-boundary state package, prepares typed base/adapter dependencies through the native legal victim path, and resumes a fixed suffix. A deterministic opt-in rule compares the estimated full increments of rerun and recovery; in the tested two-action, one-future-node setting, it is algebraically identical to an information-matched two-step lookahead. We froze cost estimates before a new 24-call check using one existing vision-language model, one adapter, one technical input, and the production state path. Across two legally executable target-residency conditions and three paired repetitions each, the predicted recovery decision matched the lower measured cost in all six pairs, with exact suffix-input and token fidelity. However, every condition lay on the recovery side, so no decision boundary was tested. Median absolute completion-cost errors ranged from 0.369 to 1.944 s, and the largest relative errors were 45.9%--49.7% for prepared-model recovery. A separate 12-case synthetic development matrix retains two estimator-error failures and shows that the rule adds recomputation while reducing modeled transfer in some cases. The evidence supports an implemented state-recovery mechanism, explicit cost accounting, and a bounded empirical result—not algorithmic superiority, real wireless benefit, cross-workflow scheduling, or statistical generalization.

## 1. Introduction

Vehicle-associated AI applications can be continuous workflows rather than isolated inference calls. One node produces intermediate state consumed by a later node, while mobility may move execution between roadside units (RSUs). At handoff, the system can rerun the completed prefix at the destination or transfer validated execution state and resume the suffix. Recovery is attractive when state restoration costs less than prefix recomputation, but the comparison also depends on model readiness.

Finite caches make readiness stateful. Preparing the current model can evict a base model or adapter needed by a declared later node. Shared bases make this effect object-dependent: evicting one adapter need not evict or reload its base. The comparison must therefore distinguish execution state, model objects, input bytes, and recomputation instead of collapsing them into one transfer term.

This paper asks one narrow question: **can an ex-ante cost estimate predict rerun-versus-recovery cost and action on a new execution of the implemented workflow path?** We do not study cross-workflow scheduling, link or compute queues, real radio allocation, or full vehicle/RSU-level multi-agent reinforcement learning. We add no neural network and no new action.

The contribution is deliberately system-oriented. First, we connect a validated node-boundary state package to typed base/adapter readiness and a dependency-safe native victim preview. Second, we state auditable incremental costs and preserve conservative fallback when inputs are missing. Third, we freeze estimates and test them on new executions without feeding realized costs back into the decision. The evaluation also preserves unfavorable evidence: the analytical rule ties correct two-step lookahead, absolute predictions are biased high, synthetic estimator errors cause wrong actions, and the corrected rule can increase recomputation.

## 2. Related Work and Attribution

### 2.1 Cost-aware caching and model readiness

Cost-aware eviction predates this work. GreedyDual-Size [1] combines locality, retrieval cost, and object size for cache replacement. Loading-cost-aware model caching and request routing at the edge [2] directly models DNN loading time, and recent layer-granularity model unloading [3] studies the memory-cost/reload-delay trade-off for edge LLMs. These works rule out any claim that charging eviction or model reload is itself new. Our narrower system composition binds legal typed-object victims to a state-recovery decision inside one already declared workflow and audits the estimate against a fixed real execution path.

### 2.2 Shared model and adapter serving

SLoRA [4] and Punica [5] establish multi-adapter serving over shared base models. POLAR [6] studies adapter caching and routing for edge LLM serving, while dual-dependency-aware VEC caching and offloading [7] jointly represents task and service dependencies. We use the same broad insight—that model objects and their dependencies are first-class state—but do not claim to invent shared-base caching, adapter routing, or dependency-aware eviction.

### 2.3 Workflow recovery and edge mobility

WfCommons [8] provides workflow representation and generation methodology, and AWTO [9] includes model-loading effects for agentic workflows at heterogeneous edges. Mobility-aware migration and resource-allocation studies [10,11] establish proactive continuity as an existing direction. Our implemented boundary is smaller: a validated application-node state package resumes one fixed suffix, and model readiness is accounted separately. This is not a new checkpointing theory or a general migration scheduler.

## 3. Implemented System and Information Boundary

### 3.1 Workflow, state, and cache objects

A workflow is a DAG (G=(V,E)). Each node (v) requires an adapter and compatible base-model dependency set (D_v). The target RSU has a finite typed resident set (C) and capacity (K). Workflow state is migration-only and does not consume model-cache capacity under the current contract.

The native cache uses deterministic sequential dependency-safe LRU. For a proposed dependency bundle (D_0), a read-only preview resolves missing objects, required capacity, and an ordered legal victim plan (V). A base cannot be evicted while a resident adapter depends on it. The recovery transaction is atomic.

The measured workflow has two nodes. A source process executes `n0` and exports a create-only production action-4 state package. Restart rereads the original image and executes `n0,n1`; recovery cannot read the source image, validates the package, reconstructs the exact `n1` input, and executes only `n1`. Each scientific role is a new process. The operating-system file cache is uncontrolled and never flushed, so this is not a disk-cold-start study.

### 3.2 Observable and unavailable quantities

The online comparison may use action legality, resident typed objects, capacity, declared dependencies, the native victim preview, input/state bytes, and frozen recompute/restore/link estimates. It may not use the current repetition's realized inference, model-load, restore, or completion time. Model outputs were not read until the full measurement plan, order, predictions, scoring, call ceiling, and timeout were committed.

Measured quantities are single-host action wall, model-load time, node inference, state serialization/validation/rebuild, exact dynamic bytes, completion, and input/token fidelity. The 100 Mbps link and 0.02 s positive-transfer latency remain assumptions. Queueing, remote model transfer, radio scheduling, and task correctness are unavailable.

## 4. Cost Accounting and Decision Rule

Let (C'=(C\setminus V)\cup D_0) be the projected post-recovery resident set. For declared near-term nodes (H), let (U=\bigcup_{v\in H}D_v), deduplicated by object identity. With transfer-time estimate (T(\cdot)), the frozen incremental costs are

\[
J_{\mathrm{rerun}}=\hat C_{\mathrm{recompute}}+T(\mathrm{input})+T(U\setminus C),
\]

\[
J_{\mathrm{recover}}=T(D_0\setminus C)+T(\mathrm{state})+\hat C_{\mathrm{restore}}+T(U\setminus C').
\]

The induced reload set is ((U\setminus C')\setminus(U\setminus C)). Every induced object must be explained by the legal victim plan; otherwise the rule falls back to rerun. Objects missing on both paths remain in both totals and cancel only in the comparison. When restart and recovery require the same base/adapter, local model preparation is likewise recorded on both full paths and is not charged asymmetrically.

The rule selects recovery only when action 4 is legal, the native preview is feasible, all costs are valid, and (J_{\mathrm{recover}}<J_{\mathrm{rerun}}). Its post-preview bookkeeping is linear in the current dependencies, victims, and near-term dependency union. Native sequential victim planning remains worst-case quadratic in resident objects. Correct two-step lookahead evaluates the same two totals with the same information; for two actions and one future node it is the same decision, not a weaker baseline.

## 5. Evaluation

### 5.1 Evidence layers

We separate three evidence sources. First, an earlier three-pair same-host run supplies calibration only. Second, the new check executes fresh model calls after all predictions are frozen. Third, 12 synthetic development cases exercise cache victims and estimator errors using native state transitions plus modeled time. The 12 cases are development/mechanism evidence, not unseen validation.

The new protocol uses the existing SmolVLM base, ALPR adapter, technical image, deterministic generation, and production action-4 state path. It has two legal conditions: the model is loaded before the action window, or it is loaded from existing local files inside that window. Each condition has three paired repetitions. A frozen random seed determines condition and arm order. Each pair consumes one source, two restart, and one recovery call: (2\times3\times4=24), exactly the call ceiling. There are no retries.

The predeclared score first requires equal completion and exact suffix fidelity, then compares local action wall plus the same assumed dynamic-transfer time. Signed error is frozen prediction minus measured scored completion. Realized costs determine only the post-hoc label and never revise the online action.

### 5.2 New measurement results

All 24 calls completed in 243.002 s. Every pair preserved source/restart `n0` output, restored source state, `n1` prompt, rendered prompt, input IDs, input hash, output token IDs, identity, and exact node counts. The package was 2,185 B in all six pairs; restart input was 192,757 B. At the frozen network assumption their transfer terms were 0.020175 and 0.035421 s, respectively.

| Condition | Arm | Frozen completion (s) | Measured scored values (s) | Median local wall (s) | Median absolute error (s) |
|---|---|---:|---|---:|---:|
| Model prepared | Restart | 16.694109 | 15.150372, 15.303677, 15.673569 | 15.268257 | 1.390432 |
| Model prepared | Recovery | 5.874839 | 3.923283, 3.931127, 4.027138 | 3.910952 | 1.943712 |
| Local preparation | Restart | 17.833756 | 17.464548, 17.537555, 17.348280 | 17.429127 | 0.369208 |
| Local preparation | Recovery | 7.020647 | 6.138928, 6.131225, 5.828562 | 6.111050 | 0.889423 |

The predicted action was recovery in all six pairs, and recovery was post-hoc cheaper in all six, so wrong-choice cost was zero. This is directional agreement only. All observed pairs were far from the decision boundary and on the same side; no threshold crossing was tested. Prediction bias is nontrivial: all signed errors are positive, and prepared-model recovery has 45.9%--49.7% relative absolute error. The cost model predicts the action more reliably here than it predicts absolute wall time.

State serialize/save + validation + input rebuild was 0.003399--0.005897 s. Restart executed 11.107--11.639 s of `n0` inference that recovery skipped. Loading the local model inside the action added a measured common component, but did not change the decision. Neither condition caused a real cache victim or later reload.

### 5.3 Synthetic development evidence and trade-offs

The preserved 12-case matrix remains useful for mechanism coverage but not independence. The eviction-aware rule and correct two-step lookahead agree 12/12. They match a post-hoc reference in 10/12, while the original threshold matches 7/12. d10 underestimates recomputation and wrongly rejects recovery; d11 overestimates recomputation and wrongly accepts recovery. These negative cases are unchanged.

Across the heterogeneous synthetic cases, the corrected rule reduces modeled time by 17.993 s and transfer by 972,125,684 B relative to the old threshold, but adds five reruns and 53.641 s of recomputation. This is a trade-off created by the time-priority objective, not a claim that all metrics improve. The matrix provides a legal-victim witness for full-bundle and adapter-only reload; it is not a real-model reload measurement.

## 6. Limitations and Claim Boundary

The independent check uses one host, one model/adapter identity, one technical image, one two-node workflow, greedy decoding, and six pairs. It lacks a task-quality label. It does not flush the OS cache or randomize machines. The network term is simulated, queue time is unavailable, and static model files were already local.

Most importantly, both executable conditions select recovery by a large margin. Directional agreement therefore does not demonstrate boundary calibration, deployment stability, or generalization. The real-model path cannot currently create a legal typed victim followed by a measured later reload without adding system behavior or manually deleting state, so that central cache externality remains supported only by synthetic native transitions.

The rule is not algorithmically superior to correct two-step lookahead. Cost-aware eviction and model-loading trade-offs are established prior ideas. The present evidence supports a mechanism/empirical study: native state export/import, exact suffix fidelity, explicit cost decomposition, a bounded direction check, and transparent failure/uncertainty boundaries. It does not support online optimality, real wireless gain, cross-workflow sharing, shared queues, full MARL, or paper-ready status.

## 7. Conclusion

The implemented path can export validated workflow state and resume a suffix without rereading the original input or rerunning the prefix. Ex-ante costs frozen from earlier measurements predicted recovery in two legal target-model residency conditions, and recovery was cheaper in all six new pairs with exact output fidelity. Absolute cost estimates nevertheless overpredicted by up to 1.952 s, all observations lay on one side of the decision boundary, and no real victim-induced reload was executable. Combined with the two synthetic estimator failures and exact equivalence to two-step lookahead, the evidence favors a bounded systems/mechanism claim rather than a new scheduling algorithm.

## References

[1] P. Cao and S. Irani, “Cost-Aware WWW Proxy Caching Algorithms,” USENIX Symposium on Internet Technologies and Systems, 1997.
[2] M. Yao, L. Chen, J. Zhang, J. Huang, and J. Wu, “Loading Cost-Aware Model Caching and Request Routing for Cooperative Edge Inference,” IEEE ICC, pp. 2327--2332, 2022, doi:10.1109/ICC45855.2022.9838823.
[3] Z. Li, Z. Tang, J. Guo, W. Jia, and W. Zhao, “Efficient Layer-Granularity Unloading for LLMs in Edge Computing,” IEEE TMC, vol. 25, no. 9, pp. 15221--15233, 2026, doi:10.1109/TMC.2026.3697118.
[4] Y. Sheng et al., “SLoRA: Scalable Serving of Thousands of LoRA Adapters,” MLSys, 2024.
[5] G. Chen et al., “Punica: Multi-Tenant LoRA Serving,” MLSys, 2024.
[6] “POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving,” arXiv:2604.16583, 2026; final publication metadata remains to be verified.
[7] “Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehicular Edge Computing,” IEEE TMC, 2025, doi:10.1109/TMC.2025.3573379.
[8] R. Ferreira da Silva et al., “WfCommons: A Framework for Enabling Scientific Workflow Research and Development,” FGCS, vol. 128, pp. 16--27, 2022.
[9] “AWTO: A Latency-Optimized Task Offloading Scheme for LLM-Driven Agentic Workflows on Heterogeneous Edge,” FGCS, 2026, doi:10.1016/j.future.2026.108415.
[10] “Multi-Agent Deep Reinforcement Learning With Trajectory Prediction for Task Migration-Assisted Computation Offloading,” IEEE TMC, 2025, doi:10.1109/TMC.2025.3539945.
[11] “Service Satisfaction-Aware Adaptive Service Migration and Resource Allocation in Vehicular Edge Computing,” IEEE TMC, 2026, doi:10.1109/TMC.2025.3596342.

## Artifact and claim ledger (not manuscript body)

- [A1] `artifacts/analysis/independent_recovery_cost_measurement_20261005_v1/`: frozen plan, all six paired receipts, raw JSON/CSV, aggregate, terminal receipt, and 72-file integrity manifest.
- [A2] `configs/acceptance/independent_recovery_cost_measurement_v1.json` and `docs/project/independent_recovery_cost_measurement_plan_20261005.md`: pre-output instances, order, estimates, predictions, scoring, call budget, and uncovered conditions.
- [A3] `artifacts/analysis/production_action4_independent_repeat_20261005_v1/`: prior calibration only.
- [A4] `artifacts/analysis/eviction_aware_recovery_validation_20261005_v1/`: 12 synthetic development/mechanism cases, including d10/d11 negative results.
- Review identity: `reviewed_at=2026-10-05`; `literature_cutoff=2026-10-05`; `target_venue=IEEE TMC`; `artifact_run_id=independent_recovery_cost_measurement_20261005_v1`; `policy_version=tmc_review_policy_v3_20260621`; `git_commit=b7c8c4dadf7a10944ebff8405b5faa56666aeaa0`; `evidence_level=E2_ARTIFACT_AUDITED (bounded same-host real-model technical execution; network assumed)`.
- Strongest safe claim: ex-ante cost direction selected the lower-cost recovery path in six new same-host pairs across two legal model-residency conditions, with exact suffix fidelity; absolute prediction and boundary validity remain unsupported.
- Prohibited claims: novelty of cost-aware eviction, superiority to correct two-step lookahead, calibrated decision boundary, statistical generalization, real wireless gain, real victim-reload benefit, cross-workflow scheduling, shared queues, task-quality gain, full MARL, or TMC-ready status.
- Remaining submission-critical gaps (maximum two): (1) legal real victim→later reload with independently frozen cases on both decision sides; (2) remote/shared-resource measurement and labeled independent workflow evidence.
