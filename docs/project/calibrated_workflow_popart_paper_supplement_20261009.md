# A 论文线独立补充：PopArt selected-checkpoint 机制结果

## 可放入 Methods 的边界文本

We evaluated PopArt as a non-original critic-stabilization technique under the service-aligned reward. The treatment normalized
only critic targets and outputs while preserving denormalized bootstrap values; reward, actor advantages, auxiliary losses,
network topology, action space, observations, and budgets were unchanged. Checkpoints were selected on four exposed development
instances using a pre-registered service lexicographic criterion. This was a development-only diagnostic, not a formal or holdout
evaluation.

To align mechanism and service consumers, we froze 48 public states before checkpoint inference (12 states from each existing
development instance, selected by deterministic breadth-first valid-action prefixes). We evaluated all 30 selected checkpoints on
the identical state set without updating parameters or normalization statistics. A complete fixed-behavior continuation supplied a
common discounted-return probe; this target is not an unbiased estimate of each policy's value and was not used to reconstruct GAE.

## 可放入 Results 的边界文本

At the actually selected checkpoints, PopArt reduced absolute error to the fixed-behavior return in all 720 state-level paired
comparisons. Method-level probe RMSE changed from 83.60 to 18.08 for SA-GHMAPPO, 85.86 to 69.16 for controller-MAPPO, and
85.02 to 67.74 for PPO. This confirms reward-scale tracking at the service-consumer checkpoints, extending the update-24 training
mechanism result.

The policy response was not directionally consistent. Mean action-4 probability changed by +0.034 for SA-GHMAPPO, -0.069 for
controller-MAPPO, and +0.027 for PPO, whereas action-4 argmax and executed-action rates generally decreased. The distinction matters
because probability argmax, hierarchical head aggregation, action-mask projection, and executed action are different objects. In the
existing development evaluation, SA-GHMAPPO improved completion (0.87 to 1.00) and on-time completion (0.29 to 0.31), MAPPO was
neutral on both, and PPO improved completion (0.93 to 0.97) but reduced on-time completion (0.29 to 0.19). Thus, critic stabilization
did not establish a method-independent service improvement.

## 必须随段落保留的限制

- PopArt is an existing technique and is not claimed as an algorithmic contribution.
- Raw critic remains the canonical comparison; PopArt was not promoted.
- The four development instances and all evaluation splits are exposed; no formal/independent holdout claim is available.
- Five seeds represent initialization sensitivity, not five independent mobility/workload clusters.
- Only 6/15 control/candidate method-seed pairs selected the same update; selected-checkpoint results include the registered
  checkpoint-selection pathway and are not a same-update pure treatment effect.
- The fixed-behavior return probe is not `V^π`; no exact historical optimizer update or GAE was reconstructed.
- Local SA gains do not establish stable SA superiority or paper-ready evidence.

图：`artifacts/analysis/calibrated_workflow_selected_checkpoint_alignment_20261009_v1/selected_checkpoint_alignment_mechanism.svg`。
本文件是给 A 线人工合并的独立补充，不修改或授权并发修改其主论文稿。
