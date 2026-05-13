# project_state_sync_and_baseline_inventory_round11 报告

## 结论

本轮只做项目状态同步、baseline inventory、IPPO/PPO readiness 和 continuity 指标定义审计。没有训练、没有 freeze、没有修改 reward、policy、baseline、checkpoint selection 或正式 benchmark split。

## 1. 当前项目最新状态是什么？

主线仍是 `NGSIM + Alibaba`。主方法是 `sa_ghmappo`。正式候选结果仍应看 `mixed_informative` 与 `full_stratified`；`multi_adapter_hard_joint_proposal/smoke` 只是 proposal smoke。

## 2. round10 的结果是否可以 freeze？

不可以。round10 是 controlled AI-service/cache-stress smoke，不是新数据集，不是正式论文结果，不可 freeze。

## 3. SA-GHMAPPO 在 round10 smoke 中的表现

- SA: reward `92.556667`, continuity `0.583334`, failure `0.0`, backhaul `221.333333`。
- PPO: reward `72.213333`, continuity `0.527778`, failure `0.166667`。
- Popularity: reward `88.395`, continuity `1.0`, backhaul `424.0`。
- Reactive: reward `65.175`, continuity `0.916666`, failure `0.5`。

SA 相对 PPO 在 reward、continuity、failure、backhaul、miss、eviction 上都更好。SA 相对 popularity reward 更高、backhaul 更低、eviction 更少，但 continuity 更低、adapter miss 更多。

## 4. IPPO 为什么缺失？

`ippo` 缺失原因是 `not_registered_or_not_evaluable`。当前 live `src/agents/registry.py` 没有 IPPO 条目，`list_evaluable_agents()` 不包含 IPPO。虽然历史 paper artifacts 中存在 `ippo` 行和 checkpoint 痕迹，但当前 live eval runner 不会评估它。

## 5. 手写规则 / heuristic baselines 有哪些？

当前 live heuristic baseline 是：

- `popularity_cache_heuristic`
- `reactive_greedy`

inventory 中额外检查了 `reactive_offloading`、`reactive_caching`、`local_only`，当前未发现 live agent/registry 入口。

## 6. 哪些手写规则没有进入 round10？

- `local_only`: not_detected_as_live_agent_or_runner_choice
- `reactive_caching`: not_detected_as_live_agent_or_runner_choice
- `reactive_offloading`: not_detected_as_live_agent_or_runner_choice

## 7. popularity_cache_heuristic 是否是当前最强 hand-written rule？

是，从当前 live heuristic 集合看，`popularity_cache_heuristic` 是最强的手写/规则基线：round10 reward `88.395000` 高于 `reactive_greedy` 的 `65.175000`，且 continuity 为 `1.0`、failure 为 `0.0`。

## 8. PPO row 是否就是 flat PPO / ppo_real alias？

round10 的 `ppo` row 来自 current `PPOAgent`，checkpoint_run_id 包含 `flat_ppo_train_...`，因此是 live `ppo` 名称加载 existing `flat_ppo` checkpoint alias。`ppo_real` 是历史 artifact 名称，不是当前 live registry 名称。

## 9. continuity=0.583334 是什么指标？

`workflow_continuity_rate = count(non-stall steps) / total step_records`。它来自 `src/metrics/paper_metrics.py::PaperMetricSet.compute`，再由 `summary_to_row()` 写入 benchmark rows。round10 SA 均值是 `0.583334`。

## 10. failure=0 但 continuity 低是否矛盾？

不矛盾。`handoff_failure_rate` 的分子是 handoff failed events，分母是 handoff events；`workflow_continuity_rate` 的分子是非 stall step。没有 handoff failure 仍可能因为 cache miss、offload target、base model mismatch 等原因产生 service stall。

## 11. SA continuity 低的最可能原因是什么？

round10 中 SA adapter miss 均值为 `5.0`，popularity 为 `0.0`。最可能原因是 proposal hard-joint 下 SA cache admission/prefetch 覆盖不足造成 step-level stall，而不是 handoff failure。

## 12. 下一轮优先级

1. 先补 IPPO eval/checkpoint rows。
2. 补遗漏 hand-written rule rows；当前没有 live `local_only/reactive_offloading/reactive_caching`。
3. 再做 `hard_joint_policy_failure_diagnosis`。
4. 最后才考虑 policy-side prefetch/cache-admission bias。

## 13. 是否建议现在继续调 SA policy？

不建议。当前应先补齐对照方法与指标定义审计，尤其是 IPPO live rows 和 hard-joint continuity/miss 的 step-level 诊断。

## 输出文件

- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/policy_baseline_inventory.csv`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/ippo_ppo_readiness.csv`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/continuity_metric_audit.csv`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/project_state_summary.json`
- `artifacts/analysis/project_state_sync_and_baseline_inventory_round11/diagnosis_summary.json`
- `docs/agent/project_current_state_round11.md`
