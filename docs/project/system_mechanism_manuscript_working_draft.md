# Cost-Accounted Recovery for Stateful AI Workflows at the Vehicular Edge

> Internal working draft v1.5 — revised 2026-10-06. This is a complete but non-submission-ready mechanism/empirical draft. It separates real-model measurements, modeled network time, and synthetic development checks. Revision v1.3 corrected path-asymmetric lifecycle accounting and withdrew the historical method advantage. Revision v1.4 added a repaired-interface SA policy-mechanism diagnosis and one frozen single-factor negative ablation. Revision v1.5 adds a matched two-reward experiment whose service-aligned candidate passed unit ordering but degraded workflow completion in all learned methods. It adds no validated algorithm. References are inherited from the preceding review and were not independently reverified in this revision.

## Abstract

Recovering a stateful AI workflow after a roadside-unit handoff can avoid recomputing completed nodes, but both restart and recovery require model readiness and can change a finite destination cache. We study this coupling within one declared continuous workflow. The implemented system exports a validated node-boundary state package, prepares typed base/adapter dependencies through the native legal victim path, and resumes a fixed suffix. A deterministic rule compares two event ledgers; in the tested two-action, one-future-node setting, it is equivalent to an information-matched two-step lookahead. A real three-node witness legally evicted 780 live ALPR tensors, loaded a 520-tensor Helmet adapter, and later reloaded ALPR from retained local files. Restart and recovery paid the same approximately 1.02 s lifecycle and recovery remained cheaper because it skipped the prefix. We then re-executed the original 12 synthetic development points with isolated, symmetric restart and recovery states. The original simple threshold, eviction-aware rule, correct two-step lookahead, and a micro offline reference agreed on all 12 points; the historical 10/12-versus-7/12 method advantage disappeared. A pre-frozen six-point boundary check again tied all online rules, while reciprocal link-estimation errors caused the same two wrong choices. The evidence supports an implemented state-recovery and adapter-lifecycle mechanism, reproducible event-level accounting, and explicit failure boundaries—not a new scheduling algorithm, superiority to a correct simple baseline, real wireless benefit, cross-workflow scheduling, or statistical generalization.

## 1. Introduction

Vehicle-associated AI applications can be continuous workflows rather than isolated inference calls. One node produces intermediate state consumed by a later node, while mobility may move execution between roadside units (RSUs). At handoff, the system can rerun the completed prefix at the destination or transfer validated execution state and resume the suffix. Recovery is attractive when state restoration costs less than prefix recomputation, but the comparison also depends on model readiness.

Finite caches make readiness stateful. Preparing the current model can evict a base model or adapter needed by a declared later node. Shared bases make this effect object-dependent: evicting one adapter need not evict or reload its base. The comparison must therefore distinguish execution state, model objects, input bytes, and recomputation instead of collapsing them into one transfer term.

This paper asks one narrow question: **can an ex-ante cost estimate predict rerun-versus-recovery cost and action on a new execution of the implemented workflow path?** We do not study cross-workflow scheduling, link or compute queues, real radio allocation, or full vehicle/RSU-level multi-agent reinforcement learning. We add no neural network and no new action.

The contribution is deliberately system-oriented. First, we implement validated node-boundary state recovery with explicit model-readiness checks and connect legal native adapter victims to real PEFT object removal and local-file reload. Second, we provide a reproducible lifecycle ledger that executes each branch in isolated cache state and separates decision estimates from realized scoring. Third, we retain the conditions under which the mechanism adds no decision value or fails under estimation error. These are implementation and empirical contributions, not a claim that their underlying ideas are new. The evaluation preserves unfavorable evidence: the historical development formula and offline reference were path-asymmetric, its apparent gain is withdrawn, the full rule ties a correct simple threshold and two-step lookahead, absolute real-model predictions are biased high, and near-boundary link errors still cause wrong choices.

We distinguish three questions: whether state recovery preserves the suffix computation (RQ1); whether frozen estimates predict new same-host action costs (RQ2); and whether cache eviction changes the preferable recovery action (RQ3). RQ1 has real-model fidelity evidence, RQ2 has a same-side directional check, and RQ3 now has a real legal victim-to-reload witness but no differential action effect or decision-boundary crossing. No single experiment closes all three questions.

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

The lifecycle witness extends only the technical measurement to `n0,n1,n2`. It uses a shared SmolVLM base, Helmet for `n0,n1`, and ALPR for `n2`. An explicit opt-in bridge first obtains a native read-only victim preview, removes or loads the corresponding PEFT adapter objects, verifies adapter registration and tensor bytes, and commits the same logical cache transaction only after runtime success. The base remains loaded, weight files remain on disk, and the OS file cache remains uncontrolled. Restart and recovery call this identical bridge; it is measurement plumbing, not a new cache algorithm.

### 3.2 Observable and unavailable quantities

The online comparison may use action legality, resident typed objects, capacity, declared dependencies, the native victim preview, input/state bytes, and frozen recompute/restore/link estimates. It may not use the current repetition's realized inference, model-load, restore, or completion time. Model outputs were not read until the full measurement plan, order, predictions, scoring, call ceiling, and timeout were committed.

Measured quantities are single-host action wall, model-load time, node inference, state serialization/validation/rebuild, exact dynamic bytes, completion, and input/token fidelity. The 100 Mbps link and 0.02 s positive-transfer latency remain assumptions. Queueing, remote model transfer, radio scheduling, and task correctness are unavailable.

## 4. Cost Accounting and Decision Rule

### 4.1 Full-path accounting and comparison boundary

For each action \(a\in\{\mathrm{rerun},\mathrm{recover}\}\), an auditable serial-path decomposition is

\[
\widehat J_a=\widehat P_a+\widehat X_a+\widehat S_a+\widehat E_a+\widehat L_a.
\]

Here \(P_a\) is preparation of the models actually required on that path, \(X_a\) is dynamic input/state transfer, \(S_a\) is state handling, \(E_a\) is computation, and \(L_a\) is additional preparation caused by cache changes at later nodes. Model download and local loading must be separately identified within preparation; they are not both implied by a local-load measurement. Each event is counted once. This additive decomposition applies to the serial path evaluated here, not to an overlapping or queued execution system.

Only terms demonstrated equal on the two paths may be cancelled. In the real two-node check both paths use the same model, but rerun executes the prefix and suffix whereas recovery executes only the suffix. In a general workflow the prefix and suffix can require different objects; preparation must therefore be resolved independently for each path, rather than assumed to be common. The same accounting applies to later reloads: they must follow the action-specific resident-state transitions, not merely a union of model names.

This is a clarification of the reporting contract, not a new validated estimator. The following frozen development equations and their results are retained without retroactive alteration.

### 4.2 Frozen development rule and its scope

Let \(C'=(C\setminus V)\cup D_0\) be the projected post-recovery resident set. For declared near-term nodes \(H\), let \(U=\bigcup_{v\in H}D_v\), deduplicated by object identity. With transfer-time estimate \(T(\cdot)\), the frozen development costs were

\[
J_{\mathrm{rerun}}=\hat C_{\mathrm{recompute}}+T(\mathrm{input})+T(U\setminus C),
\]

\[
J_{\mathrm{recover}}=T(D_0\setminus C)+T(\mathrm{state})+\hat C_{\mathrm{restore}}+T(U\setminus C').
\]

The induced reload set is \((U\setminus C')\setminus(U\setminus C)\). Every induced object must be explained by the legal victim plan; otherwise the rule falls back to rerun. Objects missing on both paths remain in both totals and cancel only in the comparison. These equations project the rerun path from \(C\) and write \(D_0\) preparation explicitly only on recovery. The source-to-production audit therefore records a frozen scientific defect: the equations are not a general full-path formula and cannot justify charging only recovery when rerun requests the same current and future dependencies. The 12 development scores and artifacts are retained without retroactive alteration. The new real lifecycle experiment instead resolves both paths event by event; their victim and reload operations are the same and are retained in both totals.

The rule selects recovery only when action 4 is legal, the native preview is feasible, all costs are valid, and (J_{\mathrm{recover}}<J_{\mathrm{rerun}}). Its post-preview bookkeeping is linear in the current dependencies, victims, and near-term dependency union. Native sequential victim planning remains worst-case quadratic in resident objects. Correct two-step lookahead evaluates the same two totals with the same information; for two actions and one future node it is the same decision, not a weaker baseline.

### 4.3 Corrected symmetric event ledger

Revision v1.3 replaces the comparison implementation, not the historical artifact. For each action \(a\), an isolated native cache state executes the ordered model requests on that path. Each event \(e\) records pre-residents, dependency-safe victims, admitted objects, post-residents, and admitted bytes. The scored serial total is

\[
J_a=\sum_{e\in L_a} T(B_e)+T(X_a)+S_a+E_a+Q_a,
\]

where \(L_a\) is the action-specific lifecycle, \(B_e\) is the admitted model bytes for exactly one event, \(X_a\) is input or state bytes, \(S_a\) is state handling, \(E_a\) is recomputation, and \(Q_a\) is common successful service time. No victim or future reload is copied from the other branch. A cost cancels only after the two isolated ledgers show the same event identity and value.

Online methods receive separate read-only lifecycle previews and ex-ante cost estimates. Scoring branches use new environments and realized frozen parameters. The offline reference receives the two realized branch scores only after both legal transitions complete. Thus, equality between an online method and the reference cannot be created by sharing the realized score. Under the current two-action workflow, restart and recovery request the same current and next models; their lifecycle ledgers are equal in all corrected points. The simple threshold is therefore sufficient for those points, while the full rule and correct two-step lookahead retain no additional decision capability.

## 5. Evaluation

### 5.1 Evidence layers

| Research question | Evidence available | What remains untested |
|---|---|---|
| RQ1: suffix fidelity | New real-model calls, identity and exact token/input checks | Traffic-task correctness, other model/task families |
| RQ2: cost prediction | Six new same-host pairs with frozen predictions | Decision boundary, other hosts and workloads |
| RQ3: eviction externality | Real legal adapter victim→runtime unload→later local-file reload plus corrected native lifecycle ledgers | Any action-specific lifecycle difference or decision capability beyond a correct simple threshold |

We separate five evidence sources. First, an earlier three-pair same-host run supplies calibration only. Second, the 24-call check executes fresh model calls after its predictions are frozen. Third, the historical 12-case matrix is retained only as defective development history. Fourth, a pre-frozen 12-call check connects native legal adapter transactions to real PEFT unload/load events with a no-eviction control. Fifth, revision v1.3 re-executes the same 12 synthetic points under symmetric ledgers and runs six separately frozen mechanism-boundary points. Neither synthetic set is unseen real-workload validation.

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

### 5.3 Corrected original development matrix

The old matrix charged current-model preparation only to recovery and also let the old action-0 execution skip that target-cache transaction. Its online scorer, alleged two-step baseline, and offline reference were therefore affected by the same asymmetry. We preserve the old artifact but withdraw its comparative claims.

The corrected run inherits all 12 instances, order, capacities, adapters, initial residents, cost parameters, and estimator multipliers by hash. Four isolated states per instance separate two online previews from two realized-score paths. All 48 paths were legal. Every method completed 24/24 nodes with zero service failures and zero deadline violations. The original threshold, eviction-aware rule, correct two-step lookahead, and micro offline reference selected recovery on 11 points and restart on the high-restore d08 point. They matched on 12/12 actions and all aggregate outcomes: 174.787877 modeled seconds, 2,047,037,144 total bytes, and 11.035304 seconds of recomputation.

The historical extra five correct decisions are therefore withdrawn. d04/d06 still execute legal full-bundle swap-and-reload and d05 still retains a shared base while swapping adapters, but these lifecycles occur on both actions and do not distinguish the rule. Historical d10/d11 wrong-choice gaps remain visible in the old-to-new ledger but disappear after correction; this does not prove estimation robustness, because the original multipliers no longer cross the corrected boundary.

### 5.4 Real adapter victim and later reload

We froze one matched victim condition and one no-eviction control before reading outputs. Both use an existing SmolVLM base, Helmet and ALPR LoRA weights, the same technical image, and a three-node `n0→n1→n2` workflow. The restart arm executes all three nodes; recovery consumes a validated `n0` package and executes `n1,n2`. `n0,n1` require Helmet and `n2` requires ALPR. The victim capacity is exactly the existing base plus ALPR weight bytes; the control capacity is exactly base plus both adapters. Each condition uses one source, three restart, and two recovery calls, for 12 calls total and no retries.

In both victim arms, native sequential dependency-safe LRU selected ALPR for the Helmet request. The PEFT runtime then removed ALPR from 780 registered tensors and 154,308,608 tensor bytes to zero, loaded Helmet as 520 tensors and 9,568,256 tensor bytes, and committed the matching logical transaction. The later ALPR request legally evicted Helmet, removed its tensors to zero, reloaded ALPR from unchanged local files, and executed `n2`. The shared base remained loaded. In the control, both requests were `noop_all_resident` and no action-window load or unload occurred.

| Condition | Arm | Adapter load/unload (s) | Scored completion (s) | Frozen prediction (s) | Decision |
|---|---|---:|---:|---:|---|
| Victim→reload | Restart | 0.997395 / 0.024849 | 11.294688 | 19.553682 | Recovery |
| Victim→reload | Recovery | 1.002123 / 0.024874 | 4.156329 | 8.509755 | Recovery |
| No eviction | Restart | 0 / 0 | 10.593149 | 18.684786 | Recovery |
| No eviction | Recovery | 0 / 0 | 3.173724 | 7.640859 | Recovery |

The direct lifecycle cost is 1.022245 s for restart and 1.026997 s for recovery. The matched action dependencies therefore make it a common cost; it cannot be used as a recovery-only penalty. Recovery remains cheaper because it skips the measured prefix. Inputs and outputs of both suffix nodes are exact across arms, weight hashes are unchanged, and the 2,195-byte state and 192,757-byte original input remain separate from local model bytes. Network time is still simulated. This experiment closes real lifecycle execution, not eviction-driven action reversal or statistical generalization.

### 5.5 Frozen mechanism-boundary check

After the corrected implementation was fixed, we froze six additional synthetic points and executed them once without model generation. They cover a common full-bundle prepare/reload, a resident current model, shared-base dependency retention, ample capacity with no eviction, and two reciprocal near-break-even link-estimation errors. Restart and recovery lifecycle events were equal on all six points. The three online rules agreed 6/6, confirming that the more elaborate rule adds no action capability here.

All methods completed 12/12 nodes with zero failure or deadline violation. The online rules matched the micro offline reference on four points. At an actual 200 Mbps but estimated 1000 Mbps, they chose restart and lost 0.001472 modeled seconds; at actual 1000 Mbps but estimated 200 Mbps, they chose recovery and lost 0.004626 seconds. These are deliberately small analytic boundary checks, not deployable network measurements or statistical evidence. They show that correct lifecycle accounting removes the old artificial gap but does not remove ordinary estimation risk.

## 6. Limitations and Claim Boundary

The evidence uses one host, one shared base, two adapters, one technical image, two- and three-node workflows, and greedy decoding. The victim witness has only one pair per condition and lacks a task-quality label. It does not flush the OS cache or randomize machines. The network term is simulated, queue time is unavailable, and static model files were already local.

Most importantly, every real condition selects recovery by a large margin. Directional agreement therefore does not demonstrate boundary calibration, deployment stability, or generalization. The opt-in bridge executes a legal typed adapter victim and later reload without deleting weight files, but the same current and future dependencies make that lifecycle common to both actions. The corrected 12-point and six-point checks confirm this equality rather than an action-specific cache benefit. A claim that eviction changes the preferred action still requires a legitimate workflow whose actions produce different model-request sequences or resident transitions; asymmetric accounting is not evidence.

The rule is not algorithmically superior to either the correct simple threshold on the tested equal-lifecycle paths or correct two-step lookahead. Cost-aware eviction and model-loading trade-offs are established prior ideas. The present evidence supports a mechanism/empirical study: native state export/import, exact suffix fidelity, dependency-safe adapter lifecycle execution, reproducible cost decomposition, and transparent failure/uncertainty boundaries. It does not support online optimality, real wireless gain, cross-workflow sharing, shared queues, full MARL, or paper-ready status.

## 7. Conclusion

The implemented path can export validated workflow state and resume a suffix without rereading the original input or rerunning the prefix. The real lifecycle witness proves legal adapter eviction, actual PEFT object removal, later local-file reload, and future-node execution, but the cost is common to matched restart and recovery paths. Once both synthetic actions execute that lifecycle symmetrically, the historical method advantage disappears: the simple threshold, full rule, two-step lookahead, and offline reference tie on all 12 original points. The boundary check adds two transparent estimator-error failures but no algorithmic separation. The strongest defensible contribution is therefore an integrated and auditable state-recovery/adapter-lifecycle system plus an empirical account of when recovery helps, ties, or fails—not a new scheduling algorithm.

## 8. Repaired-interface learning side study

The continuous-workflow learning interface was separately repaired before this revision: actors now consume typed bundle readiness, sizes,
estimated links, contact, and state fields; mobility advances on decision steps; deterministic evaluation uses independent head argmax followed by
mask projection; and PPO optimizes the executed environment-action likelihood. This correction restored SA-GHMAPPO from an interface-induced
failure loop, but it did not create a performance contribution. On the repaired development checks, PPO, controller-MAPPO, and an
information-matched two-step planner still completed every workflow, whereas SA did not.

We then inspected four SA mechanisms without reopening the interface audit. Event temperature was active in rollout, training recomputation, and
deterministic evaluation. Configured event-margin and sharpening adjustments were dormant because the frozen workload selected `raw_policy` in
all three paths. Auxiliary event targets were active only in the training loss and did not require current-node bundle readiness. A fixed
12-state logit trace and the complete 100-state development replay motivated one bounded hypothesis: this supervision could overemphasize future
preparation.

The only candidate set `auxiliary_coef` from 0.1 to 0.0 and retrained three fixed seeds for 192 episodes. Completion was unchanged at 0.944 on
the exposed regression split and 0.958 on the frozen development check. On the latter, current-missing action 4 increased from 50.0% to 57.7%,
all action 4 from 54.7% to 71.6%, and invalid prepares from 56 to 86. The failing seed moved rather than disappeared. The deletion therefore
does not repair the imbalance and is rejected. It also does not collapse to always-action-0 behavior; action 0 decreased while action 2 and
action 4 increased. The result is a configuration-specific negative finding, not proof that the auxiliary loss is generally beneficial or that
one component universally causes the behavior.

## 9. Service-objective alignment side study

The original scalar return could rank partial service above an interrupted but completed workflow because deadline cost was checked only after
completion and a truncated unfinished path had no service-outcome term. We froze one replacement before reading method rankings. It gives a
large workflow-completion utility, a one-time deadline-miss cost, and separate non-overlapping charges for failed current-node attempts,
operation time, transferred GiB, and prefix recomputation. It does not reward action 4, prepare calls, handoff events, or migration logs.
External truncation remains non-terminal and bootstraps the value function.

Both rewards trained SA-GHMAPPO, controller-MAPPO, and PPO for the same three seeds, 192 episodes, and 24-step cap. Checkpoint selection used
only common service metrics. The candidate did not improve SA: completion changed from 0.889 to 0.806 on the exposed regression split and
from 0.917 to 0.833 on the frozen development check. It reduced SA transfer and invalid prepare, but increased consecutive no-progress steps
and reduced completed-sample coverage. MAPPO completion changed from 0.972/0.958 to 0.833/0.875, with more failed-service episodes and
invalid prepares. PPO on-time completion improved descriptively, but total completion fell from 1.000/1.000 to 0.944/0.958 and new service
failures appeared. The unchanged two-step planner remained at 1.000 completion under both scoring formulas.

This is a negative objective-design result. Rescoring an unchanged trajectory under the two formulas changes the numeric return without
changing service behavior, so cross-formula return magnitude is not treated as evidence. The learned-policy behavior did change, but it
introduced completion and reliability trade-offs rather than a common service improvement. We therefore classify the result as category D
and do not proceed to the proposed event-auxiliary-target ablation. Reward alignment is neither an algorithmic contribution nor evidence for
an SA-specific mechanism.

## 10. Contribution logic closure

| Research question | Existing-method gap | Our implemented design | Required evidence | Current result | Defensible claim |
|---|---|---|---|---|---|
| Can a workflow resume a dependency-safe suffix after RSU handoff? | Generic migration/checkpointing does not by itself prove this implementation preserves DAG inputs and execution ownership | Validated node-boundary package, target model gate, exact suffix input reconstruction, single execution-right commit | Cross-process node counts, input/token identity, negative identity tests | Passed on the bounded technical workflow | The implemented mechanism resumes the fixed suffix without rereading the source input; no universal recovery theory claim |
| Does adapter-cache state participate in recovery cost? | Shared-base and adapter-serving work does not expose this workflow's legal victim→runtime unload→reload path | Native typed dependency-safe preview/commit connected to PEFT removal/load | Runtime tensor counts/bytes, legal cache events, later node execution, no-eviction control | Real ALPR→Helmet→ALPR lifecycle executed; cost was common to both arms | Real adapter lifecycle is integrated and auditable; no eviction-driven action advantage |
| Does SA role coordination outperform matched alternatives after interface repair? | PPO/MAPPO, graph encoders, recovery and cost-aware control already exist; composition alone is not novelty | Slow/fast/event heads with executed-action PPO under the repaired semantic interface | Matched budgets, checkpoints, full behavior ledger, strong rule and learned controls | SA completion 0.944/0.958; PPO, MAPPO and rule 1.000 | Interface correctness and residual policy gap only; no SA superiority claim |
| Does removing current-readiness-blind auxiliary supervision fix over-preparation? | A training target can conflict with service return, but deletion is not automatically a method | One-factor `auxiliary_coef: 0.1→0.0` ablation | Fixed 3×192 retrain, paired dev states, current-missing action 4, valid/invalid prepare, completion/cost | Completion unchanged; action 4 and invalid prepare increased | Reject wholesale deletion; retain as negative design evidence, not innovation |
| Does the service-aligned scalar objective improve behavior? | A corrected score does not guarantee stable policy learning | One frozen completion/deadline-first reward with non-overlapping observable cost terms | Two rewards × three methods × three seeds, service-only checkpoint selection, raw metrics and same-trajectory rescoring | Completion decreased for SA, MAPPO, and PPO; planner behavior was unchanged | Formula-level alignment is verified, behavioral benefit is rejected; do not advance auxiliary-target ablation |
| What does the calibrated simulator establish? | Simulated reward alone cannot prove real model or wireless behavior | NGSIM+Alibaba-derived bounded workload with measured size/time inputs and typed cache semantics | Provenance, frozen instances, raw episodes, interface audit | Supports controlled development comparison only | Empirically calibrated simulation evidence, not real deployment or independent generalization |
| What do real-model measurements establish? | Synthetic bytes cannot prove adapter lifecycle or suffix fidelity | Same-host real base/adapters, fixed calls, exact token/input checks | Pre-frozen plans, raw timings, tensor lifecycle, hashes, receipts | Mechanism paths executed; wireless/queue/task quality unavailable | Real mechanism realization and local cost components only |

The logical boundary is explicit. Combining MAPPO, graph representations, state recovery, and cost signals does not itself establish novelty.
Interface repair is correctness work, not performance innovation. Removing a harmful-looking enhancement is an algorithm simplification only if it
improves the declared service trade-off; this ablation did not. The paper therefore retains system-mechanism evidence and the negative learning
result without renaming either as a new algorithm. A learning-method claim would require a separately frozen service-feasible event target and
subsequent independent evidence; this revision does not provide it.

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
- [A5] `artifacts/analysis/real_cache_victim_reload_20261005_v1/`: pre-frozen 12-call real Helmet/ALPR victim→reload and no-eviction control, six process receipts, raw rows, terminal, negative checks, and 22-file integrity manifest.
- [A6] `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/`: corrected original 12-point matrix, 48 isolated paths, complete old-to-new point ledger, aggregate, completion receipt, and 7-file integrity manifest.
- [A7] `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/`: pre-frozen six-point mechanism-boundary check, 24 isolated paths, complete rows, aggregate, completion receipt, and 5-file integrity manifest.
- [A8] `artifacts/analysis/calibrated_workflow_prepare_balance_diagnosis_20261006_v1/`: fixed 12-state logit/probability trace, 100-state dev-path target summary, checkpoint hashes, and diagnostic receipt.
- [A9] `artifacts/benchmarks/calibrated_workflow_prepare_balance_ablation_20261006_v1/`: historical A reuse, 3×192 no-auxiliary training, complete curves, evaluation/behavior rows, paired comparison, strata, receipts, and integrity manifest; checkpoints remain local.
- [A10] `artifacts/benchmarks/calibrated_workflow_service_reward_alignment_20261006_v2/`: both rewards rerun for SA/MAPPO/PPO at 3×192 episodes, 24,952 actual steps, 90 local checkpoints, complete learning/evaluation/behavior rows, same-trajectory rescoring, 5,000-draw window-outer deltas, 41 unfavorable completion strata, receipts, and integrity manifest.
- Review identity: `reviewed_at=2026-10-06`; `literature_cutoff=2026-10-05`; `target_venue=IEEE TMC`; `artifact_run_id=eviction_aware_recovery_corrected_20261006_v2 + recovery_cost_boundary_check_20261006_v1`; `policy_version=tmc_review_policy_v3_20260621`; `git_commit=800f0a12f13e34d7de19aee375124813b8817e3a / 30691177bcac540ec1650d14494fe48326805a8c`; `evidence_level=E2_ARTIFACT_AUDITED (bounded synthetic correction and boundary check; network modeled)`.
- Strongest safe claim: native state export/import and exact suffix recovery are integrated with a dependency-safe adapter victim/removal/reload lifecycle; event-level accounting is reproducible, and equal lifecycles are shown to cancel rather than manufacture an action benefit.
- Prohibited claims: superiority to a correct simple threshold or two-step lookahead, novelty of cost-aware eviction, calibrated deployment boundary, eviction-driven decision reversal, statistical generalization, real wireless gain, cross-workflow scheduling, shared queues, task-quality gain, full MARL, or TMC-ready status.
- Remaining submission-critical gaps (maximum two): (1) an externally motivated workflow with a legitimate action-specific lifecycle difference, if an algorithmic claim is retained; (2) remote/shared-resource measurement and labeled independent workflow evidence.
- Revision v1.5: retains the v1.3 systems correction and v1.4 negative no-auxiliary result, adds the negative matched reward experiment, classifies it as a service trade-off/degeneration result, and explicitly stops before auxiliary-target redesign. Progress is maintained in `manuscript_evidence_progress_20261005.md`.
