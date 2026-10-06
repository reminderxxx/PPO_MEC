# 待合并论文实验章节：SA prepare/execution 单因素负结果（2026-10-06）

## Policy-mechanism diagnosis after interface repair

After repairing the calibrated-workflow observation, mobility, deterministic-action, and executed-likelihood interfaces, SA-GHMAPPO still
selected target preparation while the current bundle was missing. We separated four configured mechanisms by their actual execution path.
Event temperature was applied inside the network during rollout, PPO recomputation, and deterministic evaluation. In contrast, the configured
event-margin and sharpening adjustments were bypassed because the frozen workload explicitly used `raw_policy` in all three paths. Auxiliary
cross-entropy and temporal-consistency terms acted only during optimization. Their event targets used predicted target readiness and temporal
features, but did not require current-node service readiness.

We selected 12 diagnostic states before inspecting logits, using fixed quotas over current readiness, handoff proximity, and target-prepare
feasibility. The complete 100-state development replay contained 17 current-missing states; 12 received a hard auxiliary prepare target and 10
selected action 4. This supported a bounded hypothesis that auxiliary supervision could overemphasize future preparation. It did not establish
causality: two current-missing, target-infeasible states had hard target zero but still selected action 4.

## Frozen single-factor ablation

We therefore changed only `auxiliary_coef` from 0.1 to 0.0. The repaired interface, reward, data, observations, network, seeds, 192-episode
budget, 24-step cap, and checkpoint selection were unchanged. The original three SA checkpoints and learning curves were reused because their
identity matched exactly; the candidate alone was retrained for seeds 7, 17, and 29. Existing PPO, controller-MAPPO, and two-step results were
reused only under the identical environment and evaluation identity.

The removal did not change workflow completion: original/candidate completion was 0.944/0.944 on the exposed regression split and
0.958/0.958 on the frozen development check. The failing seed moved rather than disappeared. On the frozen check, current-missing action 4
increased from 33/66 (50.0%) to 45/78 (57.7%); total action 4 increased from 110/201 (54.7%) to 149/208 (71.6%). Invalid prepares increased
from 56 to 86 and target-infeasible prepares from 34 to 59, while realized prepare rate fell from 48.2% to 41.6%. Source-window bootstrap gave
a completion delta of 0 [0,0] and an invalid-prepare delta of +1.250 [0.458,2.293] attempts/window.

The candidate did not collapse to always executing the current node: action 0 decreased from 27.4% to 13.0%, while action 4 and action 2
increased. Frozen-check on-time completion and modeled cost improved descriptively, but service failure rose from 0.417 to 0.583 and completion
did not improve. Completed-only elapsed excludes the unfinished episode in each arm and is not reported as a speed advantage.

## Interpretation and claim boundary

The experiment rejects wholesale removal of the auxiliary loss as a prepare-balance fix. It also shows why component deletion is not an
algorithmic contribution: the auxiliary target has a real information mismatch, yet other parts of the loss can restrain degenerate policy
solutions. The result does not prove that the auxiliary loss is generally beneficial or identify a universal cause. It narrows the next design
question to a service-feasible event target that conditions a prepare label on both current-node executability and target-prepare feasibility.
That redesign was not implemented in this experiment.

This is nonformal development evidence, not independent generalization or paper-ready validation. The systems claims remain supported by
separate state-recovery, dependency-safety, and real adapter-lifecycle artifacts. The learning evidence supports only an honest negative result:
interface repair restored a functioning policy, but neither the original SA nor the single-factor deletion matches the completion and cost
reliability of the information-matched two-step planner in this workload.
