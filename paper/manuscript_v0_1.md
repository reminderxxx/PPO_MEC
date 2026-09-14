# PPO-MEC: Role-Factored Control of Typed Model Caching and Continuous DAG Workflows in Vehicular Edge Systems

> Manuscript status: English v0.1, prepared for an IEEE Transactions on Mobile Computing submission track. This draft contains no numerical performance claim. All result placeholders remain `UNVERIFIED` until the frozen formal pipeline, preregistered statistics, and evidence gate complete. The title and the expansion of SA-GHMAPPO are provisional.

## Abstract

Vehicular edge intelligence must preserve service continuity while a moving vehicle changes roadside-unit (RSU) association, yet an AI workflow may require both a base model and a task-specific adapter whose warm state is not available at the next RSU. This paper studies online control of continuous directed-acyclic-graph (DAG) workflows under typed model-cache dependencies, finite RSU capacity, transfer costs, and uncertain short-horizon mobility information. We formulate the implemented system as a prediction-augmented controller-level decision process with five semantically masked actions spanning cache fill, predictive prefetch, vehicle fallback, steady RSU execution, and handoff preparation. We then present SA-GHMAPPO, a graph-encoded, role-factored PPO controller with slow cache, fast execution, and event-triggered handoff heads and a centralized value function. The heads are controller roles, not independent vehicle or RSU agents. Our trace-driven protocol combines NGSIM mobility and Alibaba batch-DAG workflows with a controlled typed object catalog; it does not constitute a joint real-world vehicular model-cache trace. Evaluation is preregistered across three cache capacities, five training seeds, non-overlapping outer mobility windows, matched reactive policies, learned controllers, and exact finite-horizon report-only references. Six primary endpoints jointly characterize service readiness, transfer overhead, workflow continuity, and complete-workflow delay. **[RESULTS PENDING: insert only gate-approved estimates, confidence intervals, effect sizes, and multiplicity-adjusted tests.]**

## Index Terms

Vehicular edge computing, DAG workflow, model caching, adapter caching, handoff, mobility prediction, proximal policy optimization.

## I. Introduction

AI-driven vehicular services increasingly resemble stateful workflows rather than isolated computation requests. A workflow may contain perception, fusion, decision, and personalization stages linked by data dependencies. When the serving vehicle crosses RSU coverage boundaries, continuity depends on more than selecting an execution location: the target RSU may lack the required base model or adapter, and workflow execution state may need to migrate. Reactive loading can restore readiness, but it consumes transfer capacity and may arrive too late. Predictive loading can prepare the next RSU, but an uncertain target or premature admission can pollute a small cache.

These interactions expose three coupled control timescales. Cache placement changes relatively slowly and must respect typed dependencies. Execution placement is a per-request decision between the current RSU and vehicle fallback. Handoff preparation is sparse and event-sensitive: it is useful only when a distinct target exists and preparation is timely. Collapsing the three roles into an unconstrained flat action obscures feasibility and temporal structure, while treating them as independent vehicle or RSU agents would misrepresent the implemented controller.

We study this problem through a trace-driven, controller-level formulation. The workload combines public NGSIM mobility with public Alibaba 2018 batch DAGs and a controlled typed catalog of base models, adapters, and migration-only workflow state. This composition provides reproducible mobility and dependency structure while preserving explicit control over object semantics. It must not be interpreted as a real trace that jointly records vehicles, model requests, adapter identities, and cache events.

The proposed controller, provisionally named SA-GHMAPPO, uses a graph encoder for workflow dependencies, set-style encodings for RSUs, vehicle and prediction features, a centralized critic, and three hierarchically conditioned policy heads. The slow head selects no cache change, current-RSU fill, or predictive prefetch; the fast head selects RSU execution or vehicle fallback; and the event head selects whether to prepare a handoff. A fixed priority map converts these role decisions into the five-action environment contract. The frozen trained implementation does not enable several mechanisms described in earlier design notes, including a separate graph-continuity critic, head-credit weighting, mechanism-logit bias, continuity guards, or a nonzero mechanism auxiliary loss. This paper therefore attributes only the mechanisms active in the frozen checkpoints.

This work makes the following implementation-level contributions; empirical superiority remains a hypothesis until the formal evidence package is complete.

1. It defines a typed model-cache control problem in which service readiness requires a base-model--adapter dependency bundle, while workflow state is migration-only and does not consume long-term cache capacity.
2. It implements a controller-level, role-factored PPO architecture that aligns slow cache, fast execution, and sparse handoff decisions with a common semantic action contract.
3. It establishes a frozen trace-driven evaluation protocol with policy-neutral request exposure, three cache-capacity strata, matched cache-policy controls, broad learned baselines, and non-overlapping temporal outer clusters.
4. It preregisters a six-endpoint claim structure and an exact finite-horizon opportunity reference whose scope is explicitly limited by horizon and enumerated state.

The remainder of this paper reviews the closest research lines, formalizes the system and optimization problem, describes the implemented controller, specifies the evaluation protocol, and reserves the results section for evidence admitted by the formal gate.

## II. Related Work

### A. Dependent-task offloading and workflow continuity

Prior work jointly studies dependent-task scheduling, service caching, and offloading in MEC systems **[TODO_VERIFY: cite the audited Transactions and conference sources in `docs/project/literature_reference_table.md`]**. Our setting differs in the combination of mobility-driven RSU changes, base-model--adapter readiness, and explicit workflow-state migration. The novelty claim must be comparative rather than absolute: DAG offloading, caching, and mobility-aware control each have strong precedents.

### B. Vehicular caching, migration, and predictive handoff

Mobility-aware service caching and predictive handoff have been investigated for VEC and adjacent mobile-edge systems **[TODO_VERIFY: cite official IEEE/ACM/DOI records]**. The present system treats prediction as a short-horizon controller input, not as an oracle and not as a fully learned digital twin. Its baseline predictor combines trajectory/boundary cues into a heuristic confidence value; supervised prediction is disabled in the frozen main protocol.

### C. Multi-timescale and reinforcement-learning control

PPO, MAPPO, value decomposition, and transformer controllers are established baselines for cooperative and structured control **[TODO_VERIFY: MAPPO, QMIX, HAPPO, and controller-transformer references]**. SA-GHMAPPO does not claim that PPO or centralized training is novel. Moreover, its three heads represent control roles inside one controller; they do not instantiate full vehicle-level or RSU-level multi-agent learning.

### D. AI model, adapter, and state caching

Model serving systems distinguish model weights, adapters, prefixes, and runtime state **[TODO_VERIFY: cite adapter/model-serving sources after P02 review]**. In the implemented catalog, only base models and adapters define persistent readiness and capacity accounting. KV-prefix caching is disabled. Workflow state is transferred for migration but excluded from persistent cache capacity. These boundaries prevent the evaluation from conflating different cache objects.

## III. System Model and Problem Formulation

### A. Entities, time, and workflow state

Time is discretized into decision steps (t=0,1,\ldots,T-1). Let \(\mathcal{R}\) be the set of RSUs and let \(i_t\) denote the primary vehicle selected by the frozen runner rule at step \(t\). The implementation performs one controller call and emits one semantic action per step; it is not a simultaneous joint-action game over all visible vehicles or RSUs.

The active workflow is a DAG \(G=(\mathcal{V},\mathcal{E})\). Each node \(v\in\mathcal{V}\) has predecessor set \(\mathrm{pred}(v)\), successor set, input/output attributes, and a required adapter \(a(v)\). If \(\mathcal{C}_t\) is the set of completed nodes, the executable frontier is

\[
\mathcal{F}_t=\{v\in\mathcal{V}\setminus\mathcal{C}_t:\mathrm{pred}(v)\subseteq\mathcal{C}_t\}.
\]

The formal evaluation replays a frozen request-exposure sequence. A service success or failure does not add, delete, reorder, or retry later exposures. This exogenous exposure contract is essential for paired algorithm comparisons.

### B. Typed model objects and capacity

Let \(\mathcal{B}\) denote base models, \(\mathcal{A}\) adapters, and \(\mathcal{S}\) workflow migration states. Each adapter \(a\in\mathcal{A}\) depends on one base model \(b(a)\in\mathcal{B}\). For RSU \(r\), binary variables \(x_{r,o,t}\) indicate whether object \(o\) is resident. Persistent cache occupancy is

\[
\sum_{o\in\mathcal{B}\cup\mathcal{A}} m_o x_{r,o,t}\le C_r,
\]

where \(m_o\) is resident size and \(C_r\) is the capacity of RSU \(r\). Workflow state \(s\in\mathcal{S}\) has a migration-transfer size but `counts_toward_capacity=false`; it is not a long-lived cache object. KV-prefix objects are disabled and excluded from capacity and primary event denominators.

A request for adapter \(a\) is fully service-ready at RSU \(r\) only if both \(a\) and \(b(a)\) are resident:

\[
q_{r,a,t}=x_{r,a,t}\,x_{r,b(a),t}.
\]

Admission is atomic at the dependency-bundle level, with at most two cache objects per logical admission. The runtime admits the base before its adapter, disallows partial admission, rolls back a failed bundle, and prevents eviction of a base that still supports a resident dependent adapter. Cache-changing actions occur before the same-step lookup. Consequently, a same-step readiness hit may still incur transfer bytes; a high hit rate does not imply zero transfer overhead.

### C. Observation and prediction boundary

The controller observation is summarized as

\[
o_t=(G_t,\mathcal{C}_t,\mathcal{F}_t,\mathcal{R}_t,z_t,p_t),
\]

where \(\mathcal{R}_t\) contains RSU load and typed-cache state, \(z_t\) contains primary-vehicle features, and \(p_t\) contains the predicted next RSU, target/path cues, dwell-time cues, future-load summaries, confidence, and uncertainty. The frozen baseline predictor uses short-horizon trajectory and coverage-boundary information. Its confidence is a bounded heuristic combination of stability, dwell, and handoff evidence rather than a calibrated probability guarantee. The main protocol disables the supervised G12 predictor path.

The graph encoder receives ten-dimensional node features, performs two rounds of predecessor/successor mean aggregation with residual normalization, and produces current-node, frontier, and graph summaries. Separate encoders process ten-dimensional RSU features, ten-dimensional vehicle features, and thirteen-dimensional prediction features. Prediction information is reliability-gated with a nonzero leakage floor, so uncertain predictions are attenuated rather than deleted.

### D. Semantic action contract

The environment exposes five masked semantic actions.

| ID | Action | Implemented meaning | Principal feasibility condition |
|---:|---|---|---|
| 0 | `current_rsu_cache_fill` | Admit the current request's dependency bundle at the current RSU, then execute there. | A current workflow node and current RSU exist. |
| 1 | `predictive_next_rsu_prefetch` | Prefetch the dependency bundle to a distinct predicted next RSU while maintaining current execution. | The predicted target is distinct and its bundle is not already ready. |
| 2 | `vehicle_fallback` | Execute on the vehicle without changing the RSU cache. | A current workflow node exists. |
| 3 | `current_rsu_steady_offload` | Execute at the current RSU without a cache-changing request. | A current workflow node and current RSU exist. |
| 4 | `handoff_migration_prepare` | Prepare the predicted handoff target, including target prefetch and migration preparation, while maintaining current execution. | A distinct handoff target exists. |

Let \(\mathcal{M}(o_t)\subseteq\{0,1,2,3,4\}\) be the semantic action mask. The controller samples \(a_t\in\mathcal{M}(o_t)\). Invalid actions are masked rather than optimized through penalty alone.

### E. Transition, reward, and optimization objective

After the selected action is decoded, the runtime applies cache admission or preparation before lookup, serves the frozen request exposure, accounts for typed transfers, advances mobility and workflow state, and records the next observation. The implemented scalar reward is exactly decomposed as

\[
r_t=R_t^{\mathrm{service}}+R_t^{\mathrm{continuity}}+R_t^{\mathrm{explore}}
-P_t^{\mathrm{delay}}-P_t^{\mathrm{miss}}-P_t^{\mathrm{migration}}-P_t^{\mathrm{constraint}}.
\]

The piecewise component values and conditions are defined by the frozen environment implementation. The `mechanism_exploration_bonus` is reward shaping and is not an evaluation endpoint. The learning objective is the expected discounted return

\[
\max_{\theta}\;J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],
\quad a_t\in\mathcal{M}(o_t),
\]

subject to typed-cache capacity and dependency feasibility. This is the implemented training objective, not a claim that a single scalar reward completely represents the paper's system goals. Empirical conclusions are instead organized around six preregistered endpoints:

1. `full_service_ready_byte_hit_rate`;
2. `joint_base_adapter_hit_rate`;
3. `full_service_ready_request_rate`;
4. `transfer_mb_per_request`;
5. `workflow_continuity_rate`;
6. `end_to_end_workflow_delay`.

The last endpoint is available only for complete, uncensored workflows whose exposed requests all succeed. A null value therefore means unavailable, not zero delay. The protocol does not currently provide `latency_saved` or counterfactual-regret endpoints.

## IV. SA-GHMAPPO Controller

### A. Scope and naming

SA-GHMAPPO is retained as the implementation identifier. For this manuscript, it denotes a surrogate-assisted graph and hierarchical **controller-level** PPO. The historical “multi-agent” expansion is potentially misleading because the executable topology has one controller action per step. The final title and method expansion should be frozen only after author review.

### B. Shared representation

The workflow graph, RSU set, current/target RSU summaries, vehicle state, and prediction embedding are fused into a shared latent representation. A centralized critic consumes the fused controller state. “Centralized” refers to the critic's access to the fused system observation; it does not imply decentralized execution by independent vehicle or RSU actors.

The frozen model configuration enables the graph encoder, centralized critic, hierarchical conditioning, prediction features, uncertainty signal, dependency-aware features, event head, and adapter-prefetch action. The three role heads share the representation:

- slow/cache head: `no_change`, `current_fill`, or `predictive_prefetch`;
- fast/execution head: `current_rsu_offload` or `vehicle_fallback`;
- event/handoff head: `keep` or `handoff_prepare`.

Hierarchical conditioning feeds slow-head probabilities to the fast head and slow-plus-fast probabilities to the event head. The environment action follows a fixed priority rule: event preparation maps to action 4; otherwise slow prefetch maps to 1; slow fill maps to 0; fast fallback maps to 2; and the default is steady RSU offload, action 3.

### C. Masked role policies

For head \(h\in\{s,f,e\}\), let \(\pi_\theta^h(u_t^h\mid o_t)\) be the masked categorical role policy. The role action tuple \((u_t^s,u_t^f,u_t^e)\) is converted by the deterministic priority map \(g\) to \(a_t=g(u_t^s,u_t^f,u_t^e)\). Masking is derived from semantic feasibility and then propagated to the relevant role options. This factorization reduces each head's decision scope while preserving the stable five-action external contract.

### D. Frozen PPO objective

For each enabled role head, the probability ratio at optimization epoch \(k\) is

\[
\rho_{t,h}(\theta)=\frac{\pi_\theta^h(u_{t}^{h}\mid o_t)}{\pi_{\theta_{\mathrm{old}}}^h(u_{t}^{h}\mid o_t)}.
\]

The clipped role loss is

\[
\mathcal{L}_{h}^{\mathrm{clip}}(\theta)=
-\mathbb{E}_t\left[
\min\left(\rho_{t,h}\hat A_{t,h},
\operatorname{clip}(\rho_{t,h},1-\epsilon,1+\epsilon)\hat A_{t,h}\right)
\right],
\]

with \(\epsilon=0.2\) in the selected frozen checkpoint configuration. The event head blends its event-specific advantage with the base advantage using the configured factor 0.85 and applies the configured event actor-loss gain 1.25. Because head-credit weighting is disabled, the manuscript must not describe a learned or counterfactual per-head credit mechanism.

The implemented joint loss is

\[
\mathcal{L}=\sum_h\mathcal{L}_{h}^{\mathrm{clip}}
+0.7\,\mathcal{L}_{V}-0.004\,\mathcal{H}+0.06\,\mathcal{L}_{\mathrm{aux}}.
\]

Here \(\mathcal{L}_{V}\) is value-function mean-squared error and \(\mathcal{H}\) is policy entropy. In the frozen implementation, `_compute_auxiliary_loss` returns zero; thus the nominal coefficient 0.06 does not produce an active auxiliary training signal. The manuscript must not attribute performance to mechanism auxiliary supervision.

### E. Explicitly inactive mechanisms

The selected frozen checkpoints disable the following mechanisms: graph-continuity critic, uncertainty-aware event scaling, uncertainty-aware critic, head credit, mechanism-logit bias, continuity guard, handoff-target-alignment guard, heuristic imitation, digital-twin planning, option gates, and the advanced planner/guard family. These features may remain in the codebase as inactive paths, but they are outside the method claim of this paper version.

### F. Computational scope

Per-step inference applies two graph message-passing rounds over the active DAG plus set aggregation over visible RSUs, followed by small multilayer perceptrons and three categorical heads. A formal complexity expression should be added after the exact hidden-layer and edge-processing notation is frozen. **[TODO_VERIFY: derive parameter count, asymptotic cost, and measured inference latency from the frozen checkpoint and formal scalability artifacts.]**

## V. Experimental Methodology

### A. Evidence and reproducibility boundary

All scientific statements in this draft are traced to scientific commit `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`, protocol version 2.9.0, and the immutable trained-model source run `typed_model_cache_formal_20260906_152847_g14c_v16`. Evaluation orchestration uses executor commit `a028ea291941a484ae9cd2e316d1b52adde3d1f2`. The current evaluation-only run is `typed_model_cache_evaluation_only_20260913_g14r20_i3_pending`.

### B. Trace-driven workload

Mobility comes from NGSIM I-80 data. Workflow structure comes from the Alibaba 2018 batch DAG source, with frozen workflow IDs `j_3`, `j_8`, and `j_15`. A controlled typed catalog supplies two base models, eight adapters, and one migration-only workflow-state object. This is a reproducible composition of three sources, not a joint observational trace of vehicular AI model requests.

The base--adapter catalog fingerprint and dependency graph are frozen by the protocol. Persistent resident sizes and transfer sizes are distinct fields. The three RSU cache-capacity strata are 288 MB (constrained), 576 MB (medium), and 864 MB (relaxed). Each episode has at most 22 environment steps and consumes a frozen 24-frame mobility window.

### C. Data splits and independence

The protocol freezes 24 training windows, 12 development windows, 12 formal windows, and 12 sealed holdout windows. Formal and holdout claims require verification using raw frame/time intervals; differing window identifiers or rank offsets are not sufficient evidence of independence. The formal statistical outer unit is the non-overlapping raw-time mobility window, identified by `(source_segment_run_id, window_id)`. Seeds, workflows, and rows are nested observations and must not be treated as independent outer clusters.

The sealed holdout was not opened in the current run. No holdout result appears in this manuscript version.

### D. Training and model selection

The learned matrix contains ten algorithms, five seeds (`7`, `13`, `29`, `43`, and `71`), and three capacities, for 150 immutable selected models. Each learned agent-capacity-seed run uses 256 episodes, a maximum of 22 steps per episode, batch size 64, 32 updates, checkpointing every four updates, and CPU execution. The protocol equalizes the maximum environment-interaction budget while preserving intrinsic on-policy/off-policy update differences.

Checkpoint selection uses development data only and follows a finite-before-null lexicographic rule: maximize full-service-ready byte hit rate, maximize workflow continuity, minimize transfer MB per request, minimize complete end-to-end workflow delay, then resolve ties by earlier update and immutable hash. Formal and holdout data are forbidden for model selection.

### E. Compared controllers and policies

The paper display order is frozen as follows:

- matched reactive policies: LRU, FIFO, LFU, Aging-LFU, and Random;
- proposed controller: SA-GHMAPPO;
- learned comparisons: PPO, MAPPO, DQN, Dueling DQN, QMIX, Controller-MAT, DAG-Offload-DRL, Cache-Offload-DRL, and DT-Handoff-DRL.

The five reactive policies share controller and admission behavior; their only primary difference is eviction policy. They support a controlled cache-policy comparison but should not be described as full competitors to every learned controller mechanism. The report-only matched popularity heuristic is secondary. Exact finite-horizon references at \(H\in\{1,3,6,12\}\) are admissible only for cells explicitly marked exact. They are tiny-horizon, finite-state opportunity references, not unconditional global upper bounds.

### F. Primary endpoints

`full_service_ready_byte_hit_rate` weights each eligible request by the unique bytes of its required base and adapter; partial readiness contributes zero numerator bytes for that request. `joint_base_adapter_hit_rate` and `full_service_ready_request_rate` capture request-level readiness views. `transfer_mb_per_request` includes base-model, adapter, and workflow-state migration transfers per typed request exposure. `workflow_continuity_rate` uses successful request outcomes over the frozen external request denominator, retaining later exposures after predecessor failure. `end_to_end_workflow_delay` is null for failed, incomplete, or censored workflows.

High readiness and low transfer are distinct objectives because same-step admission precedes lookup. Any result discussion must therefore report readiness and transfer jointly. Null endpoints must be accompanied by availability counts and cannot be converted to zeros.

### G. Statistical analysis plan

All primary comparisons use paired outer windows. The protocol specifies a hierarchical bootstrap with 10,000 replicates and seed 1401: raw-time mobility windows are resampled as outer clusters, while seed and workflow ID remain inner cluster keys. It requests percentile and BCa 95% confidence intervals, paired Cohen's \(d_z\), and an outer-window standardized mean difference.

Paired testing uses the exact two-sided sign test after preregistered numerical ties within tolerance \(10^{-9}\) are dropped. Holm correction at \(\alpha=0.05\) applies jointly to all preregistered primary comparisons across the six endpoints. Only finite available p-values enter Holm correction. Zero available pairs yield null effects, intervals, tests, and Holm results—not a tie, pass, or failure. Algorithmic failures remain missing/failure observations under worst-case sensitivity; post-result seed or window deletion is prohibited.

### H. Current execution status

As of 2026-09-14, the formal cache-policy and controller phases each completed three capacity cells. Each phase records 8,100 rows/episodes over the same 12 formal outer windows; these two related tables must not be combined and reported as 16,200 independent samples. Six of 22 expected cells are committed. The first formal-ablation cell terminated with return code 1 before commitment because the executor and scientific checkout disagreed on the active-bundle repository root. Formal support, scalability, statistics, gate, and completion phases have not started. The held continuation lock remains untouched, and holdout remains sealed.

Accordingly, completed raw cells are inventory evidence only. No comparison, confidence interval, significance statement, ablation attribution, or paper-readiness conclusion is available in v0.1.

## VI. Results

### A. Main controller comparison across capacity

**[R1 UNVERIFIED PLACEHOLDER]** Populate the six primary endpoints for SA-GHMAPPO and all frozen baselines at 288, 576, and 864 MB. Report outer-window paired estimates, 95% intervals, availability counts, effect sizes, exact sign tests, and Holm-adjusted decisions. Do not rank methods using row-level standard errors.

### B. Cache-policy isolation

**[R2 UNVERIFIED PLACEHOLDER]** Compare the five matched reactive eviction policies under their shared controller/admission contract. Discuss cache readiness, transfer, pollution, and churn without attributing differences to controller logic.

### C. Continuity--transfer trade-off

**[R3 UNVERIFIED PLACEHOLDER]** Jointly plot workflow continuity and transfer MB per request. Explain same-step admission semantics and avoid interpreting readiness as avoided transfer.

### D. Capacity sensitivity

**[R4 UNVERIFIED PLACEHOLDER]** Analyze whether conclusions persist under constrained, medium, and relaxed capacity. Use paired outer-window changes and avoid selecting the most favorable stratum post hoc.

### E. Ablation and predictor boundary

**[R5 UNAVAILABLE]** The formal-ablation phase has no committed cell. No component attribution is currently admissible. Predictor robustness and supervised-predictor claims also require their frozen support phases.

### F. Exact finite-horizon opportunity

**[R6 UNAVAILABLE]** Report only cells with explicit exact status for \(H=1,3,6,12\). Label the result as a finite-horizon opportunity gap over the enumerated state/action scope, never as a global optimum or universal upper bound.

## VII. Discussion and Limitations

First, the dataset is composed rather than jointly observed: NGSIM provides mobility, Alibaba provides DAG structure, and a controlled catalog supplies model/cache semantics. This improves reproducibility but limits claims about real vehicular AI request distributions.

Second, the controller acts once per step at the system level. Results cannot be generalized to full vehicle/RSU-agent MARL without a separately frozen observation/action contract and matching baselines.

Third, the main predictor is heuristic and short-horizon. Prediction confidence is an input signal, not a guarantee of calibrated uncertainty. Supervised and stronger prediction claims require separate evidence.

Fourth, workflow-state migration is modeled as transfer-only and does not consume persistent cache capacity. KV-prefix caching is disabled. Conclusions therefore concern base-model--adapter readiness and migration transfer, not every form of AI serving state.

Fifth, complete-workflow delay is subject to principled unavailability. Analyses must report availability and cannot treat missing latency, `latency_saved`, or counterfactual regret as observed zero values.

Finally, the current formal evidence package is incomplete. Raw formal rows from two phases do not establish algorithmic superiority, statistical significance, generalization, or paper readiness.

## VIII. Reproducibility, Data Governance, and Ethics

The protocol freezes scientific code, model selection, split identities, object semantics, endpoint formulas, and statistical units before final reporting. Request exposures are policy-neutral, and formal/holdout data are excluded from checkpoint selection. Public dataset terms and redistribution constraints should be summarized before submission **[TODO_VERIFY]**. No unpublished data, checkpoint, manuscript, or reviewer material should be uploaded to public AI services. Trace-driven results should not be framed as an on-road deployment or safety certification.

## IX. Conclusion

This paper formulates continuous vehicular DAG execution as a joint typed-cache, execution, and handoff-preparation problem and describes a graph-encoded, role-factored PPO controller under a fixed semantic action contract. The methodology separates persistent base/adapter readiness from migration-only workflow state and evaluates system trade-offs through six preregistered endpoints. **[FINAL CONCLUSION PENDING]** Replace this paragraph only after formal statistics and the evidence gate determine which hypotheses are supported, mixed, unsupported, contradicted, or unavailable.

## References

**[TODO_VERIFY]** Build the IEEE-formatted bibliography in P02 from the audited literature table and primary publisher records. Do not cite historical project prose as an external scientific source.
