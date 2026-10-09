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

Let a submitted workflow be a directed acyclic graph \(G=(V,E)\). A node becomes eligible only when its predecessors have completed. A node \(v\) requests a compatible base-and-adapter dependency set \(D_v\). For RSU \(r\), \(C_r(t)\) is the resident set under capacity \(K_r\), measured in bytes. A shared base is counted once even when multiple adapters depend on it; an adapter cannot remain resident without its required base. The application state \(z_v\) needed by a suffix is distinct from a model weight, an adapter file, and an unspecified key-value cache. Node input, state, and model transfer sizes are bytes; compute, load, restore, recompute, and modeled service times are seconds. A link rate in Mbit/s and fixed latency in seconds produce an analytic transfer term, not a wireless measurement.

We distinguish model readiness from execution readiness. Model readiness requires \(D_v\) to be usable at the selected service location. Execution readiness also requires the graph frontier, the relevant saved state, and the service conditions. The simulator records an instance-level modeled deadline and a finite step limit separately: a workflow can finish late, while an unfinished rollout can stop at the step limit. These are evaluation constructs, not a measured application service-level agreement. Completion, on-time completion, failed service attempts, and completed-sample delay therefore have different denominators.

### 3.2 Available action and information

At each decision step the environment exposes five masked actions, \(a_t\in\{0,1,2,3,4\}\): (0) admit the current node's bundle at the current RSU, (1) prefetch it to the predicted next RSU, (2) serve through vehicle fallback, (3) attempt current-RSU service without a cache change, or (4) prepare the predicted target and a node-boundary state package while attempting current service. A failed current-bundle service attempt is recorded and can be followed by another step; it is not an episode failure. A controller sees a normalized nine-value numeric observation together with causal semantic state: the declared workflow and current node, typed cache residency/capacity, current vehicle association, and predicted RSU sequence, dwell, confidence, uncertainty, and load. Prediction fields are estimates, not the realized future trajectory. Neither realized branch cost nor an offline reference enters the online decision.

This action interface is a single controller with slow cache, fast execution, and handoff-event decision heads that aggregate to one executed environment action. It is not a vehicle/RSU multi-agent joint-action system, and it does not allow arbitrary future-node object placement. A baseline may encode the same causal state differently; its actual information and model access must be disclosed. In particular, an exact-transition planner has stronger model-based capability than a learned actor even when both start from the same visible state.

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

### 4.3 Candidate workflow-aware controller and learning objective

The proposed comparison version uses `calibrated_workflow_interface_v2` and `independent_heads_executed_env_v2`: graph-related state, typed cache context, and predicted handoff context feed one controller with slow cache, fast execution, and event heads. The heads aggregate under the five-action legal mask to a single executed action. The current training interface records that executed action and applies clipped PPO to its action probability (`executed_action_ppo_only=true`, `env_action_ppo_coef=1`); the critic estimates discounted return with episode-local generalized advantage estimation. A completed workflow terminates the episode. A nonterminal step-limit truncation retains a value bootstrap from the final, unreset observation. The exact active feature projection, policy distribution, auxiliary terms, parameter count, and inference-time computation must be reported from **[B-FREEZE: integrated scientific commit, checkpoint, and resolved config]**; a v70 sparse-tail prior or reliability gate is not presumed active.

The proposed `service_aligned_v1` per-step training reward is

\[
r_t=I_{\rm node}+100I_{\rm workflow}-25I_{\rm first\ deadline\ miss}
-2I_{\rm failed\ attempt}-0.02t_{\rm op}-0.25B_{\rm transferred}/\mathrm{GiB}
-0.05t_{\rm recompute}.
\]

Here \(t_{\rm op}\) includes node compute or vehicle fallback, model loading, and state restore, but excludes network transfer, recomputation, and the modeled failed-service delay. \(B_{\rm transferred}\) counts model, state, and input bytes once. The deadline penalty occurs only on the first modeled crossing; a late workflow can still complete. This is an optimization objective, whereas the paper comparison ranks service completion and failures before cost and reports reward as a secondary diagnostic. A bounded service-reward development test found that formula-level ranking did not translate into better learned completion for SA, controller-level MAPPO, or PPO. PopArt value normalization has a separately frozen stability A/B design, with no result yet available to this draft; it is established training machinery rather than a claimed new algorithm. Neither the reward formula nor that A/B establishes policy superiority.

## 5. Experimental Method

### 5.1 Evidence layers and reproducibility

We use three noninterchangeable evidence layers. The first runs a fixed two-node technical workflow with real model generation in separate processes, comparing prefix restart with state recovery. The second runs one three-node, two-adapter local victim/reload witness and one no-eviction control. The third uses native cache transactions and analytic network terms for corrected development and boundary matrices. The last layer is synthetic and is not counted as additional independent real-model trials. Each layer retains its frozen inputs, command or completion receipt, raw outputs, and file integrity inventory.

The controller layer is a calibrated simulation: NGSIM supplies mobility intervals and handoff-pressure features, and Alibaba supplies batch-DAG topology and task resource fields. Their pairing, adapter assignment, state scale, link, and modeled deadline are not jointly observed real VEC requests. Its statistical unit is a nonoverlapping original mobility interval; repeated seeds and workflows within one interval are nested observations. The current 36-instance development manifest (12 train, 4 dev, 12 regression, 8 frozen check) has been fully consumed and cannot become a confirmatory split. A historical registry contains 668 distinct consumed interval records, 418 with unresolved identity; an earlier I-80 inventory of 579 eligible windows is only an eligibility pool, not proof of unused data. Before any confirmatory run, source hashes, original frame/time intervals, and history must establish disjoint train/dev/evaluation windows. No old holdout is reopened.

All methods must consume the same frozen request stream, legal actions, causal information, interaction budget when trained, and checkpoint selection opportunities. The primary endpoints are on-time and total workflow completion, unfinished-after-deadline, failed-service attempts and episodes, continuity/no-progress, and complete-workflow cost. Model, state, and input transfer and recomputation are retained even for failed workflows; reward is a secondary controller diagnostic. Delay is reported only for completed samples together with coverage, never with missing values treated as zero.

### 5.2 Required controller comparison and ablations

The intended learned comparison is SA-GHMAPPO against single-controller PPO, controller-level MAPPO, and the project-native DT handoff DRL baseline. The DT implementation is literature-inspired, not an exact reproduction of one published method, and its prediction features require an explicit information-parity audit. Popularity is a stateful heuristic that counts adapter demand and uses predicted next RSU/handoff target; its memory lifetime and prediction ability must be disclosed. A two-step rule clones exact environment transitions and orders candidates lexicographically by completed nodes, service failures, deadline violations, elapsed seconds, and bytes. It is reported as a separate model-based capability comparator, not a matched-capacity learned controller. The present service runner trains/selects only SA, MAPPO, and PPO and evaluates the two-step rule; DT and Popularity still lack the full runner, checkpoint/artifact, and analysis chain. **[B-FREEZE: integrated method identities, capability parity, and scientific commit]**.

The main table will pair every method on the same new raw intervals and list training steps, optimizer updates, selection opportunities, model information, and cost to decide. The current frozen *design*, not an executed strong-baseline result, assigns learned methods seeds 7/17/29/43/61, 1,440 environment steps, 24 update opportunities, 192 optimizer steps and four dev checkpoint choices. Deterministic rules are evaluated once per interval/workflow and are not copied across training seeds. The first bounded ablation compares state recovery with symmetric restart to test avoided prefix work. A second paper ablation may be chosen only after the final scientific version identifies an independently active mechanism and a prespecified falsifiable question; v70 sparse-tail and reliability switches are not assumed active. PopArt control/treatment is a learning-stability test, not an SA-specific novelty ablation. No result-dependent extra ablation is promoted.

## 6. Results

### 6.1 Suffix fidelity and same-host cost

Six fresh paired comparisons across prepared-model and local-preparation conditions consumed 24 real generation calls. Every pair preserved the source/restart prefix output and the recovered suffix's constructed prompt, input IDs, input hash, and output tokens. Each recovered target executed only the suffix node. The package measured 2,185 bytes and the restart input 192,757 bytes. Under the frozen analytic link term, recovery was cheaper in all six pairs, chiefly because restart repeated about 11.1–11.6 seconds of prefix inference. Both conditions lie on the recovery side of the decision boundary; the six successes do not estimate boundary accuracy or wireless gain. Absolute completion estimates retained a visible bias, including approximately 46–50% relative error for prepared-model recovery.

### 6.2 Real victim, runtime removal, and later reload

The three-node witness begins with a shared base and ALPR resident. A legal request for Helmet removes ALPR from the PEFT runtime, reducing its registered 780 tensors and 154,308,608 tensor bytes to zero. Helmet is loaded, the middle node runs, and a later ALPR request legally evicts Helmet and reloads ALPR from unchanged local files before the final node. The no-eviction control performs no action-window load or unload. Restart and recovery preserve the same suffix inputs and output tokens. In the victim condition, the local adapter lifecycle costs 1.022245 and 1.026997 seconds on the two paths; it is a common cost. Their scored completions are 11.294688 and 4.156329 seconds, respectively, including a modeled dynamic network term. This is one matched witness per condition with unflushed OS cache and no task-quality label.

### 6.3 Corrected rule comparison and estimation boundary

An earlier development comparison omitted current-model preparation from restart and charged later model costs asymmetrically. Re-executing all 12 original synthetic design points with isolated, symmetric ledgers removes the apparent method advantage. A correct simple threshold, the fuller eviction-aware rule, an information-matched two-step lookahead, and a small offline reference choose the same action on all 12 points. All methods complete the same 24 nodes with zero service failures; their modeled time and total bytes also match. A separately frozen six-point analytic check leaves the three online rules tied. They agree with the offline reference on four points and make the same two near-boundary errors when the assumed link rate differs from the evaluated rate. These observations demonstrate accounting correctness and an estimation failure mode, not a novel scheduling gain.

### 6.4 Controller comparison

A separate, nonformal development test on the consumed 36-instance manifest found that the service-aligned reward passed its formula-level ordering checks but lowered total completion for SA-GHMAPPO, controller-level MAPPO, and PPO relative to their original-reward counterparts. In the associated offline diagnosis, the service-reward critic had near-zero last-update explained variance and a value-to-policy gradient-norm ratio of 539–7,907 on fixed development samples. This motivates a single pending PopArt stability test; it is neither causal proof of the failure mechanism nor a successful controller comparison. No result from that exposed manifest is promoted to the table below.

**[RESULT PENDING: integrated scientific commit, DT/Popularity runner integration, independent intervals, matched raw rows and budgets, complete manifests/checkpoint hashes, and prespecified statistics. No historical reward ranking or confidence interval is inserted here.]**

| Method | On-time / total completion | Unfinished after deadline | Failed attempts / episodes | Handoff failure / no-progress streak | Completed delay, s (coverage) |
| --- | --- | --- | --- | --- | --- |
| Frozen workflow-aware controller | [B] | [B] | [B] | [B] | [B] |
| PPO | [B] | [B] | [B] | [B] | [B] |
| Controller-level MAPPO | [B] | [B] | [B] | [B] | [B] |
| Project-native DT handoff DRL | [B] | [B] | [B] | [B] | [B] |
| Popularity, stateful heuristic | [B] | [B] | [B] | [B] | [B] |
| Two-step exact-transition rule, model-based | [B] | [B] | [B] | [B] | [B] |

| Method | Model / state / input transfer, B | Recompute, s | Cache misses / evictions | Training / inference cost | Reward, secondary |
| --- | --- | --- | --- | --- | --- |
| Frozen workflow-aware controller | [B] | [B] | [B-METRIC] | [B] | [B] |
| PPO | [B] | [B] | [B-METRIC] | [B] | [B] |
| Controller-level MAPPO | [B] | [B] | [B-METRIC] | [B] | [B] |
| Project-native DT handoff DRL | [B] | [B] | [B-METRIC] | [B] | [B] |
| Popularity, stateful heuristic | [B] | [B] | [B-METRIC] | [B] | [B] |
| Two-step exact-transition rule, model-based | [B] | [B] | [B-METRIC] | [B] | [B] |

## 7. Discussion and Limitations

The verified mechanism addresses one continuous technical workflow, one host, one shared base, and two adapters. It does not establish correct traffic-task answers, remote RSU transfer time, queueing behavior, cold disk performance, multiple competing workflows, or performance under a real request distribution. Dynamic state bytes, local adapter weight bytes, and simulated network time remain separate. The corrected rule ties a correct simple baseline in the tested equal-lifecycle paths; a more complex controller is justified only by a future matched result on a problem where its information or decision structure can matter.

Earlier controller development comparisons are excluded from the main table because their source-window identity, independent sampling units, and training/selection budgets do not satisfy this protocol. Failed workflows remain in completion, failure, transfer, and recomputation denominators; delay describes completed workflows only and always carries its coverage. No holdout, broader representativeness, or submission-readiness statement is inferred from the system witnesses.

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
