# Workflow-Aware Model Preparation and State Recovery at the Vehicular Edge

> Author review draft, 9 October 2026. This is the single current manuscript working path for the CSCWD-oriented study. The controller implementation and comparison results are awaiting a separately frozen experimental version. Bracketed entries are required results or definitions, not estimated values. Page count and the current conference template have not been verified for this draft.

## Abstract

A vehicle-associated AI workflow can change its serving roadside unit while later tasks still depend on earlier outputs, compatible model adapters, and execution state. Preparing a destination therefore requires more than placing a model in a cache: the model dependency set must fit, the workflow frontier must remain valid, and state recovery must be compared with recomputing the prefix under the same model-loading costs. We implement a typed base-and-adapter cache with dependency-safe replacement and a node-boundary state interface for a continuous workflow. In bounded same-host experiments, a recovered suffix preserves its constructed inputs and output tokens, and a legal adapter victim is removed from the runtime and later reloaded from retained local files. Symmetric accounting shows that this adapter lifecycle is a common cost of restart and recovery in the tested workflow; a historical advantage attributed to a more elaborate recovery rule disappears when both paths execute the same requests. Six measured restart/recovery pairs favor recovery because it skips prefix inference, whereas a small analytic boundary check exposes two errors caused by link-rate estimates. These results establish an auditable mechanism and its limits. A matched evaluation of workflow-aware learning against simpler controllers remains **[RESULT PENDING: frozen controller version and independent comparison]**. The study does not claim measured wireless savings or general algorithmic superiority.

**Keywords:** vehicular edge computing; dependent AI workflows; adapter cache; handoff; state recovery; cost accounting.

## 1. Introduction

Vehicle-associated AI services can involve a sequence of dependent inference tasks rather than one isolated request. A vehicle can leave the coverage of its serving roadside unit (RSU) between tasks, while edge storage and model-loading capacity remain limited. At the destination, the next task may need the previous output, a compatible adapter, and sufficient space for the adapter and its shared base model. If any of these conditions is absent, a nominal cache hit or a predicted handoff does not guarantee that the workflow can continue.

This creates a coupled preparation decision. Loading an adapter early can reduce a cold start at the next RSU, but consumes capacity and communication before the predicted service location is confirmed. Restarting the workflow can avoid sending saved state but repeats completed computation; restoring state can avoid that computation but requires a valid package and may still need the same model loads. We ask: **as the service location changes, how do workflow dependencies, model residency, and recovery cost jointly determine current execution and preparation for the remaining workflow?**

Prior work provides strong pieces of this problem: task- and service-dependent VEC caching, shared-base adapter serving, cost-aware model placement, and mobility-aware migration [2, 4, 5, 7, 10]. These studies differ in the resource object they control and in whether application state is an executable continuation across a service change. Their results do not by themselves tell us the cost of two legal restart/recovery paths when both require the same current and future adapters. The unresolved question for this study is operational and testable: which dependency and state conditions actually make a continuation executable, and which preparation costs distinguish the available choices?

The system developed here makes those three objects explicit. A workflow frontier records which node is executable. Typed cache objects distinguish shared bases from compatible adapters. A node-boundary package records only declared application state and its identity, while model files remain separate static resources. A legal cache transaction and runtime adapter operation are connected before the next node executes. This interface lets us measure a real adapter unload and reload without treating a cache-residency flag as proof that the runtime changed.

Our current evidence supports three **candidate** contributions, with different maturity. First, an executable model-cache and state-recovery path provides bounded system evidence. Second, workflow-aware preparation control is implemented as a controller-level research candidate, but its incremental benefit over matched simpler methods is not established here. Third, event-level cost and service accounting provides auditable positive and negative observations, including a corrected comparison that removes an earlier apparent method advantage. The experiments below keep actual model calls, native cache simulation, and analytic network terms separate.

## 2. Related Work and Attribution

Dependency-aware service caching and offloading in VEC already combines task and service dependencies with learned control [7]. Shared-base adapter serving is established in SLoRA and Punica [4, 5], and adapter caching and routing has its own recent line of work [6]. Cost-aware model placement [2] and migration studies [10, 11] further show that neither adapter caching, handoff preparation, nor applying reinforcement learning is new by itself. Our comparison concerns their operational interaction within one continuing workflow: what has to be resident, what state survives the handoff, and which costs are common to legal alternatives.

Workflow research provides methods for representing and replaying dependent execution [8], while related edge studies address agentic workflows [9]. Our current technical workflow is deliberately narrower: it has a declared graph and testable input passing, but no labeled driving-task accuracy. We therefore use it to validate continuity and cost accounting, not application quality. The literature and this implementation do not justify a claim that the combined problem has no precedent; the evidence required for a new controller contribution is an information- and budget-matched comparison, not a list of integrated modules.

## 3. System and Decision Problem

### 3.1 Workflow and resource state

Let a submitted workflow be a directed acyclic graph \(G=(V,E)\). A node becomes eligible only when its predecessors have completed. A node \(v\) requests a compatible base-and-adapter dependency set \(D_v\). For RSU \(r\), \(C_r(t)\) is the resident set under capacity \(K_r\). A shared base is counted once even when multiple adapters depend on it; an adapter cannot remain resident without its required base. The application state \(z_v\) needed by a suffix is distinct from a model weight, an adapter file, and an unspecified key-value cache.

We distinguish model readiness from execution readiness. Model readiness requires \(D_v\) to be usable at the selected service location. Execution readiness also requires the graph frontier, the relevant saved state, and the service conditions. The observable outcome is whether the declared workflow completes within its stated evaluation horizon. That horizon is not called an application deadline unless an application timing contract is provided.

### 3.2 Available action and information

The prototype's environment exposes five discrete actions: fill the current RSU, prefetch toward a predicted destination, use vehicle fallback, offload without changing the cache, and prepare a handoff. A legal-action mask limits which actions can execute. A controller may use the submitted workflow, current residency and capacity, and causally available mobility observations. It may not use the actual future service location, realized branch cost, or an offline reference when choosing an action.

This action interface is a single controller with several decision heads. It is not a vehicle/RSU multi-agent joint-action system, and it does not allow arbitrary future-node object placement. Any controller comparison must preserve this authority and disclose any additional information used by a planning baseline.

### 3.3 Symmetric cost comparison

For either legal alternative \(a\in\{\mathrm{restart},\mathrm{recover}\}\), we record a path-specific ledger of model admissions and victims, local inference, state preparation or validation, dynamic payload bytes, and subsequent node requests. A descriptive scored completion is

\[
T_a=T_{\mathrm{model},a}+T_{\mathrm{infer},a}+T_{\mathrm{state},a}+T_{\mathrm{network},a}+T_{\mathrm{future},a}.
\]

Measured local components retain their own timestamps. Where no wireless trace exists, \(T_{\mathrm{network},a}\) is explicitly an analytic term derived from frozen bytes, link rate, and fixed latency. The two paths start from equivalent target cache states and each executes its own current and future dependency requests. A model unload and reload that occurs on both paths belongs in both ledgers. The decision rule may compare estimated path totals, but an offline reference can use realized totals only after both paths have been executed.

## 4. Implemented Mechanism and Controller Boundary

### 4.1 State export and suffix reconstruction

The source completes a declared prefix node and exports a node-boundary package containing its output, graph progress, control fields, resource identity, and integrity checks. A fresh target process validates the package, reconstructs the suffix input from saved content, and executes only the remaining nodes. The base, adapter, and processor are static resources loaded separately at the target. Exact reconstructed prompt, input IDs, input hash, and output tokens are checked against the continuous or restart path; a copied final answer would not pass this test.

### 4.2 Dependency-safe model lifecycle

When capacity is binding, an adapter admission may require legal victim selection. The native transaction checks the base dependency and records pre- and post-residents, selected victims, admitted objects, and bytes. An opt-in runtime bridge then removes the selected adapter from the PEFT runtime, loads the requested adapter from retained local files, and commits the logical cache state only after the runtime operation succeeds. A later node can cause another legal eviction and reload. This is evidence of local runtime lifecycle execution; it does not measure transfer of model weights between RSUs.

### 4.3 Candidate workflow-aware controller

The research controller combines graph-related workflow state, typed cache context, predicted handoff context, and cache/execution/event decisions. Its exact formula, observation projection, action aggregation, checkpoint, and training objective for the conference comparison are **[B-FREEZE: method identity and scientific commit]**. We do not insert an older policy formula into this section while citing a newer calibrated result. Once frozen, this section must state the implemented policy distribution, legal mask handling, learning update, inference-time computation, and parameter count. If that version cannot be matched to the evidence below, the controller remains a system component rather than an independently validated contribution.

## 5. Experimental Method

### 5.1 Evidence layers and reproducibility

We use three noninterchangeable evidence layers. The first runs a fixed two-node technical workflow with real model generation in separate processes, comparing prefix restart with state recovery. The second runs one three-node, two-adapter local victim/reload witness and one no-eviction control. The third uses native cache transactions and analytic network terms for corrected development and boundary matrices. The last layer is synthetic and is not counted as additional independent real-model trials. Each layer retains its frozen inputs, command or completion receipt, raw outputs, and file integrity inventory.

The pending controller comparison has a different statistical unit: nonoverlapping original mobility intervals. Repeated seeds and workflows within one interval are nested observations. Before any confirmatory run, the code and all baselines must consume the same frozen source/window identity, request stream, legal actions, observation information, interaction budget, checkpoint selection opportunities, and endpoint definitions. The primary endpoints are workflow completion and failure, continuity, and complete-workflow cost; reward is reported as a secondary controller diagnostic. Transfer and recomputation are retained even for failed workflows. Delay is reported with its availability and conditioning rule.

### 5.2 Required controller comparison and ablations

The main comparison is **[B-RESULT: frozen candidate versus same-information PPO, DT handoff control, popularity, and a correct two-step planner]**. The plan must identify all checkpoints, seeds, original intervals, training/update budgets, and paired rows. The first bounded ablation disables state recovery while keeping model requests symmetric, testing whether suffix restoration avoids repeated prefix work. The second, conditional on the final policy interface, disables its workflow-sensitive preparation signal under matched training and selection budgets: **[B-ABLATION: exact toggle and checkpoint identities]**. No additional component ablation is promoted without a new prespecified question.

## 6. Results

### 6.1 Suffix fidelity and same-host cost

Six fresh paired comparisons across prepared-model and local-preparation conditions consumed 24 real generation calls. Every pair preserved the source/restart prefix output and the recovered suffix's constructed prompt, input IDs, input hash, and output tokens. Each recovered target executed only the suffix node. The package measured 2,185 bytes and the restart input 192,757 bytes. Under the frozen analytic link term, recovery was cheaper in all six pairs, chiefly because restart repeated about 11.1–11.6 seconds of prefix inference. Both conditions lie on the recovery side of the decision boundary; the six successes do not estimate boundary accuracy or wireless gain. Absolute completion estimates retained a visible bias, including approximately 46–50% relative error for prepared-model recovery.

### 6.2 Real victim, runtime removal, and later reload

The three-node witness begins with a shared base and ALPR resident. A legal request for Helmet removes ALPR from the PEFT runtime, reducing its registered 780 tensors and 154,308,608 tensor bytes to zero. Helmet is loaded, the middle node runs, and a later ALPR request legally evicts Helmet and reloads ALPR from unchanged local files before the final node. The no-eviction control performs no action-window load or unload. Restart and recovery preserve the same suffix inputs and output tokens. In the victim condition, the local adapter lifecycle costs 1.022245 and 1.026997 seconds on the two paths; it is a common cost. Their scored completions are 11.294688 and 4.156329 seconds, respectively, including a modeled dynamic network term. This is one matched witness per condition with unflushed OS cache and no task-quality label.

### 6.3 Corrected rule comparison and estimation boundary

An earlier development comparison omitted current-model preparation from restart and charged later model costs asymmetrically. Re-executing all 12 original synthetic design points with isolated, symmetric ledgers removes the apparent method advantage. A correct simple threshold, the fuller eviction-aware rule, an information-matched two-step lookahead, and a small offline reference choose the same action on all 12 points. All methods complete the same 24 nodes with zero service failures; their modeled time and total bytes also match. A separately frozen six-point analytic check leaves the three online rules tied. They agree with the offline reference on four points and make the same two near-boundary errors when the assumed link rate differs from the evaluated rate. These observations demonstrate accounting correctness and an estimation failure mode, not a novel scheduling gain.

### 6.4 Controller comparison

**[RESULT PENDING: B must supply a frozen implementation, matched baseline package, nonoverlapping interval audit, raw endpoint rows, and prespecified statistics. No historical reward ranking or confidence interval is inserted here.]**

| Method | Workflow completion | Continuity | Complete-workflow delay and coverage | Model/state transfer | Reward, secondary |
| --- | --- | --- | --- | --- | --- |
| Frozen workflow-aware controller | [B] | [B] | [B] | [B] | [B] |
| Matched PPO | [B] | [B] | [B] | [B] | [B] |
| Matched DT handoff controller | [B] | [B] | [B] | [B] | [B] |
| Popularity control | [B] | [B] | [B] | [B] | [B] |
| Same-information two-step rule | [B] | [B] | [B] | [B] | [B] |

## 7. Discussion and Limitations

The verified mechanism addresses one continuous technical workflow, one host, one shared base, and two adapters. It does not establish correct traffic-task answers, remote RSU transfer time, queueing behavior, cold disk performance, multiple competing workflows, or performance under a real request distribution. Dynamic state bytes, local adapter weight bytes, and simulated network time remain separate. The corrected rule ties a correct simple baseline in the tested equal-lifecycle paths; a more complex controller is justified only by a future matched result on a problem where its information or decision structure can matter.

The older controller development package cannot fill that gap. Its recorded frozen-window plan and actual evaluation rows refer to different NGSIM source segments; many windows overlap, and the controller and inherited baselines used different update counts and checkpoint selection rules. Its reward means may describe those stored runs, but its original independent-window confidence intervals and claims of matched algorithmic superiority are not used here. No holdout, broader representativeness, or submission-readiness statement is inferred from the system witnesses.

## 8. Conclusion

Workflow continuation across changing service locations couples graph progress, typed model residency, and recoverable state. This prototype connects a validated suffix package to dependency-safe adapter operations and exposes each path's computation, model lifecycle, and dynamic payload cost. In bounded same-host measurements, recovery preserves suffix execution and avoids repeated prefix inference; in a real victim/reload witness, model-loading cost is shared by restart and recovery. Symmetric native accounting removes an earlier apparent advantage of a more elaborate rule and reveals a small, explicit link-estimation failure boundary. Whether a workflow-aware learned controller improves service completion or cost beyond matched simpler methods remains **[B-RESULT PENDING]**.

## References

[1] P. Cao and S. Irani, “Cost-Aware WWW Proxy Caching Algorithms,” USENIX Symposium on Internet Technologies and Systems, 1997.
[2] M. Yao et al., “Loading Cost-Aware Model Caching and Request Routing for Cooperative Edge Inference,” IEEE ICC, 2022, doi:10.1109/ICC45855.2022.9838823.
[3] Z. Li et al., “Efficient Layer-Granularity Unloading for LLMs in Edge Computing,” IEEE TMC, 2026, doi:10.1109/TMC.2026.3697118.
[4] Y. Sheng et al., “SLoRA: Scalable Serving of Thousands of LoRA Adapters,” MLSys, 2024.
[5] G. Chen et al., “Punica: Multi-Tenant LoRA Serving,” MLSys, 2024.
[6] “POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving,” arXiv:2604.16583, 2026; final publication metadata pending verification.
[7] “Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehicular Edge Computing,” IEEE TMC, 2025, doi:10.1109/TMC.2025.3573379.
[8] R. Ferreira da Silva et al., “WfCommons: A Framework for Enabling Scientific Workflow Research and Development,” FGCS, 2022.
[9] “AWTO: A Latency-Optimized Task Offloading Scheme for LLM-Driven Agentic Workflows on Heterogeneous Edge,” FGCS, 2026, doi:10.1016/j.future.2026.108415.
[10] “Multi-Agent Deep Reinforcement Learning With Trajectory Prediction for Task Migration-Assisted Computation Offloading,” IEEE TMC, 2025, doi:10.1109/TMC.2025.3539945.
[11] “Service Satisfaction-Aware Adaptive Service Migration and Resource Allocation in Vehicular Edge Computing,” IEEE TMC, 2026, doi:10.1109/TMC.2025.3596342.
