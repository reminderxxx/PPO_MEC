# Dependency-Aware Model and Execution-State Preparation for Mobile Edge Workflows

> Internal working draft — 2026-09-28. Not submission-ready. This draft describes the implemented system abstraction and the scope of available evidence; it does not establish a novel algorithm or independent generalization. Editorial notes are excluded from the proposed manuscript body. No positive result is reserved in advance.

## Abstract

Executing a dependent workflow across roadside units requires more than locating a cached model. The destination must also have the appropriate adapter and compatible execution state, while preparation consumes storage and communication resources before its benefit is realized. We study this preparation problem in a trace-driven vehicular edge simulator with explicitly typed model dependencies and workflow state. Our analysis separates the physical effects of sharing and migration from the contribution of the controller that selects preparation actions. Development experiments illustrate why aggregate communication savings are insufficient evidence of efficiency: a controller can transfer less data while completing fewer workflows. They also expose an inconsistency between hierarchical action sampling and the corresponding policy update. These findings motivate controlled mechanism ablations and a distribution-consistent learning formulation. The current evidence characterizes implementation and development behavior; it does not establish superior performance on independent test data.

## 1. Introduction

Vehicle-associated workflows can remain active while their execution context moves between roadside units (RSUs). A later task may depend on an earlier result, a particular model adapter, and state produced before a handoff. Consequently, the availability of an individual cached object does not imply that the next task is executable, and immediate task readiness does not imply completion of the remaining workflow.

Sharing a base model across adapters reduces duplicated storage, but it also couples admission decisions: the incremental cost of an adapter depends on which related objects are already resident. Multi-adapter serving systems demonstrate the practical relevance of this shared-base organization [R2, R3]. Their GPU-memory abstractions, however, should not be equated with roadside placement or workflow-state migration. In the setting considered here, preparing a destination involves both typed model dependencies and continuity of execution.

Proactive preparation introduces a second tension. Preparing early may avoid a cold destination at handoff, but occupies capacity and incurs transfer even when the predicted destination is not used. Preparing only after demand is observed can avoid speculative work yet leave the workflow unprepared during a transition. A useful controller must therefore distinguish preparation that enables subsequent execution from preparation that merely increases local readiness or cache occupancy.

Relevant work already addresses dependency-aware service caching and offloading [R1], adapter caching and routing [R4], mobility-assisted migration [R5], and cache-aware edge workflow scheduling [R6]. These are direct antecedents, not missing research areas. Our question is narrower: **when shared model dependencies and execution-state readiness interact during a mobile workflow, what preparation decisions are necessary, and when does coordinating those decisions improve completion relative to treating them separately?** Establishing this distinction requires explicit resource semantics and controlled comparisons; combining familiar modules is insufficient.

We organize the investigation around three questions. First, how does shared dependency residency change the incremental cost of making a task executable? Second, when does destination preparation compensate for its communication and occupancy costs? Third, does a structured controller exploit these effects better than a simpler controller with the same information and execution authority? The third question is conditional on demonstrating the first two, rather than assuming that a more complex policy is inherently preferable.

The current study provides an executable typed-resource abstraction and a development diagnosis. Its remaining contribution claims require matched mechanism and controller experiments. We distinguish the system mechanism, the correctness of its learning implementation, and the eventual performance of a proposed algorithm throughout the paper.

## 2. Related Work

### 2.1 Dependency-aware caching and workflow execution

Dual-dependency-aware VEC caching and offloading already combines task/service dependencies, service criticality, and PPO-based decisions [R1]. Edge workflow scheduling also incorporates model-loading and memory effects [R6]. Accordingly, dependency awareness or applying PPO is not, by itself, a contribution of this work. The relevant comparison concerns the decision objects, the treatment of execution state across RSUs, and the consequences for the remaining workflow. An exhaustive claim that no earlier work models this combination is not made.

### 2.2 Shared model resources and mobile preparation

SLoRA and Punica support multiple adapters using shared base-model resources [R2, R3]. POLAR studies coupled adapter caching and routing on two timescales [R4]. These works motivate explicit accounting for shared resources and rule out treating adapter caching or two-timescale learning alone as novel. Mobility-assisted task migration provides another close line of work [R5]. Our evaluation must therefore distinguish the benefit of physical sharing or migration from any additional benefit of coordinating them.

### 2.3 Learning and delayed preparation benefits

Preparation decisions can incur cost before their service benefit becomes observable. Return decomposition is an established approach to delayed reward attribution [R7], while PPO-based cooperative controllers can be strong baselines without specialized architectures [R8]. These observations motivate two safeguards: the update must correspond to the policy that actually generated actions, and any proposed attribution mechanism must be compared against the same system controlled by a strong simpler method. The present implementation does not implement RUDDER or inherit its theoretical guarantees.

## 3. System Model and Research Objective

### 3.1 Workflow and resource state

Represent a workflow as a DAG \(G=(V,E)\). An edge represents a prerequisite, and the completed-node set determines the currently eligible frontier. The model catalog distinguishes base models and adapters, including their compatibility relations. Workflow execution state is a separate typed resource; it is not assumed to be an LLM key–value cache. A vehicle is associated with a serving RSU, which may change along the mobility trace.

Let \(C_r(t)\) denote resident resources at RSU \(r\), with capacity \(K_r\). For a task \(v\), let \(D_v\) be the compatible model dependency set and \(z_v(t)\) the required execution state, when applicable. Model readiness requires the relevant members of \(D_v\) to be available. Execution readiness additionally requires prerequisites, compatible state, and the applicable service conditions. Thus model readiness is necessary in the modeled edge-service path, but is not sufficient for workflow completion.

### 3.2 Incremental preparation accounting

For a proposed set of resource objects \(B\), its missing storage at a destination is

\[
\Delta M(B,r,t)=\sum_{o\in B\setminus C_r(t)} \operatorname{size}(o).
\]

The sum is over unique objects: a shared base appearing in multiple task dependency sets is counted once. A non-sharing intervention instead uses its corresponding replicated-object representation. This distinction must be reflected in the actual resident set and transfer accounting, rather than introduced solely through a reward term. Any admission must respect capacity and compatibility after the applicable eviction operation.

This expression is an accounting definition, not a claim of a new optimization algorithm. State transfer, backhaul volume, and occupancy costs must retain their distinct units; an abstract migration penalty is not automatically a measurement of transferred state bytes.

### 3.3 Causal information and action authority

The controller observes the submitted DAG, completed progress, current typed residency and capacity, and predictions computed from causally available mobility observations. It does not receive future service outcomes. The current prototype exposes five actions: fill the current RSU, prefetch to a predicted destination, use vehicle fallback, offload without a cache change, and prepare for handoff. Legal-action masking constrains execution.

These actions determine preparation timing and mode. They do not permit arbitrary selection of future critical-node object bundles or unrestricted joint allocation across RSUs. Such extensions require a different action interface and must give baselines equivalent authority. The controller is a system-level policy; its multiple heads are not separate vehicle or RSU agents.

### 3.4 Evaluation objective

The principal scientific outcome is completion of the submitted workflow within the specified observation horizon. A horizon is not relabeled as an application deadline without a corresponding timing contract. Communication and migration costs include unsuccessful workflows. We report completion, failures, transfer volume, actual state-transfer volume, and delay together with its availability coverage, rather than selecting a post-hoc weighted score.

A possible future constrained formulation maximizes completion subject to a transfer budget in addition to storage constraints. The current prototype does not enforce a hard transfer-budget constraint, so it cannot be described as already solving that formulation. Cost–completion comparisons currently characterize trade-offs, not constrained optimality.

## 4. Controller Computation and Correctness Boundary

The current mechanism-aware input adds 28 features concerning model dependencies, capacity, remaining workflow progress, predicted destination, and execution-state readiness. Controllers share the mechanism information in the matched development comparison. Additional information alone does not establish an algorithmic contribution.

For the hierarchical controller, let \(e\), \(s\), and \(f\) denote event, slow, and fast head probabilities. The unmasked five-action probabilities are

\[
p=(e_0s_1,\ e_0s_2,\ e_0s_0f_1,\ e_0s_0f_0,\ e_1).
\]

With legal-action mask \(m\), the behavior distribution is

\[
\pi_m(a\mid x)=\frac{m_a p_a(x)}{\sum_j m_jp_j(x)}.
\]

If PPO is used for this behavior distribution, its importance ratio must be computed from the old and new masked environment-action probabilities. Independently clipping ratios for canonical head labels is not generally equivalent to clipping the ratio of the action actually sampled. With only one legal action, the conditional distribution is constant and its direct actor gradient is zero.

The current legacy implementation fails this consistency criterion in controlled mathematical checks. The corrected update remains an implementation specification rather than a validated performance result. Fixing it is a prerequisite for evaluating the hierarchical method, not a novelty claim. Neither convergence nor superior completion follows from consistency alone.

Algorithmic complexity claims are deferred until the final decision interface and implementation are fixed. In particular, graph encoding, candidate construction, masking, dependency lookups, and policy inference must be accounted for separately; no unsupported polynomial-time or optimality guarantee is asserted.

## 5. Evaluation Design

### 5.1 Data and interpretation

The prototype combines NGSIM mobility traces and Alibaba workflow DAGs with a controlled mapping to model requirements. This is a trace-driven simulation, not a joint measurement of vehicles running the specified AI workflows. The catalog, model mapping, transfer assumptions, and state semantics require explicit reporting and sensitivity analysis. Development windows reused for training or design are not an independent test set.

### 5.2 Mechanism and controller comparisons

The planned mechanism study crosses shared-base representation and execution-state migration. A common fixed controller first isolates the physical interventions; separately trained matched arms then measure adaptation to those interventions. Independent, sequential, and coordinated preparation policies are needed to distinguish joint decision value from the sum of two useful modules. Merely enabling both modules is not evidence of coordination.

Comparisons include strong dependency-aware heuristics and flat PPO with equivalent observable information, legal actions, training budget, and checkpoint rules. Graph and hierarchical contributions require separate ablations. Negative controls include no sharing opportunity, no handoff, ample capacity, unreliable predictions, and costly migration. Workload eligibility is determined without inspecting the candidate's performance.

### 5.3 Statistical and cost reporting

Outer sampling units are original mobility windows or appropriately grouped capture runs. Seeds and scenario variants within a window are repeated measurements, not additional independent windows. Completion and all incurred costs are reported together. Delay conditional on completion must not be interpreted as unconditional latency. The protocol for a future confirmatory comparison must freeze estimands, multiplicity, selection rules, and data independence before evaluation.

## 6. Available Development Evidence

The completed development matrix contains 12 trained models and 156 evaluation episodes across three original windows. The learned comparisons have 36 episodes per condition; the deterministic heuristic has 12 and is not expanded through duplicate seeds. An independent derivation verified 1,056 source-file hashes and sizes and recomputed the following quantities from the raw episode records.

| Controller | Completed workflows | Total transfer (MB) | Total transfer / completed workflow (MB) |
|---|---:|---:|---:|
| Full SA | 18/36 | 7,432 | 412.889 |
| Signal-off SA | 14/36 | 9,410 | 672.143 |
| Full MAPPO | 18/36 | 7,432 | 412.889 |
| Full PPO | 27/36 | 10,224 | 378.667 |
| Critical-path heuristic | 6/12 | 3,268 | 544.667 |

The ratios retain transfer incurred by failed workflows. Full SA transfers less overall than PPO but completes fewer workflows and incurs more transfer per completed workflow. Normalizing by successful requests instead gives a different ordering, showing why no unconditional efficiency claim follows from choosing a favorable denominator. These are retrospective descriptive diagnostics, not new confirmatory endpoints.

The completion counts across the three training seeds are 9, 9, and 0 for full SA, versus 9, 9, and 9 for PPO. The nominal 280-MB and 360-MB capacity scenarios produce identical primary outcomes and no eviction or capacity rejection in the audited traces. They therefore do not establish robustness to binding capacity constraints. Only three outer windows and development reuse preclude interpreting these results as independent superiority evidence.

Separately, a mathematical audit enumerated 31 nonempty masks, all legal actions, four fixed logit probes, and both advantage signs. The resulting 640 cases show objective or gradient disagreement between the legacy equal-weight head surrogate and the masked environment-action PPO reference. This controlled slice falsifies their general equivalence; it does not quantify how much of the observed completion gap a correction would remove.

## 7. Limitations and Conclusion Boundary

The evidence does not yet establish a new algorithm's superiority, a benefit from joint preparation beyond additive mechanisms, or generalization to unused data. Capacity stress, causal mechanism effects, the cost of more complex control, and the workload-to-model mapping remain evaluation obligations. The previous one-time test opening failed without usable performance results and cannot supply independent validation. A new confirmatory study requires eligible data and an independently fixed evaluation protocol, not reuse of a consumed test set.

The present conclusion is therefore limited: typed dependencies and execution state provide an explicit setting in which to investigate mobile workflow preparation, while current development results expose important accounting and learning-consistency pitfalls. Establishing a publishable system-mechanism contribution requires demonstrating when coordinated preparation improves meaningful service outcomes, and whether the proposed controller contributes beyond simpler alternatives.

## References

- [R1] [Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehicular Edge Computing](https://doi.org/10.1109/TMC.2025.3573379). IEEE TMC, 2025.
- [R2] [SLoRA: Scalable Serving of Thousands of LoRA Adapters](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html). MLSys, 2024.
- [R3] [Punica: Multi-Tenant LoRA Serving](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html). MLSys, 2024.
- [R4] [POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving](https://arxiv.org/abs/2604.16583). Preprint, 2026; publication status requires final verification.
- [R5] [Mobility-Aware Assisted Deep Reinforcement Learning for Collaborative Task Migration and Resource Allocation in Vehicular Edge Computing](https://doi.org/10.1109/TVT.2026.3660321). IEEE TVT, 2026.
- [R6] [AWTO: A latency-optimized task offloading scheme for LLM-driven agentic workflows on heterogeneous edge](https://doi.org/10.1016/j.future.2026.108415). FGCS, 2026.
- [R7] [RUDDER: Return Decomposition for Delayed Rewards](https://papers.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html). NeurIPS, 2019.
- [R8] [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html). NeurIPS Datasets and Benchmarks Track, 2022.

---

## Editorial evidence map — not manuscript body

- reviewed_at / literature_cutoff: 2026-09-28; target_venue: IEEE TMC; policy_version: tmc_review_policy_v3_20260621.
- drafting_base: eaeb1e9; artifact_run_id: crdcm_performance_matrix_v2_20260928.
- Evidence: numerical re-aggregation E3 within observed-data development; math audit cited E2, not independently rerun here; novelty UNVERIFIED; submission readiness Not ready. No paper-level score assigned.
- Sections 1–3: candidate question and implemented abstraction; see `research_problem_and_evidence_plan_20260928.md` and `problem_literature_traceability_20260928.md`. Final physical/state equations must be cross-checked against the frozen implementation before submission.
- Section 4: current decision contract plus corrected-update specification, not an implemented new algorithm. Source: `/Users/howen/.codex/worktrees/1aed/PPO_MEC/docs/project/hierarchical_credit_consistency_e0_20260928.md`, delivery commit `aacf95e8ab9dd202b13fac330e293a30482ee915`.
- Section 5: prospective design, not completed experiments. No invented power, runtime upper bound or significance threshold.
- Section 6 table: `artifacts/analysis/research_problem_evidence_20260928/failure_inclusive_cost_audit.json`; immutable source experiment commit `4dc5adb6f6e436983745a0cf485f5221ba4caf0a`.
- Section 6 math counts: `/Users/howen/.codex/worktrees/1aed/PPO_MEC/artifacts/analysis/hierarchical_credit_consistency_e0_20260928/summary.json`.
- Older formal results are a separate method/workload lineage: see `g14a01_formal_results_independent_review_20260921.md` and subsequent corrected statistics. Do not merge their 8,100 rows with this development matrix.
- Do not claim first/unique, full vehicle-level MARL, a corrected controller's superiority, an enforced transfer budget, or a real deployed LoRA service.
- Before submission: freeze the actual new mechanism/algorithm; complete matched evidence and independent evaluation; replace the abstract and contribution paragraph with supported findings; complete exact bibliography and full-text nearest-neighbor comparison.
