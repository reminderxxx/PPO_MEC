# multi_adapter_hard_joint_smoke_round10 报告

## 范围

本轮不是新数据集，不训练，不 freeze，不修改 reward、policy、baseline 或 checkpoint selection。

它是基于真实 NGSIM mobility trace 与真实 Alibaba DAG structure 的 controlled AI-service/cache-stress smoke。`semantic_ai_service` adapter assignment 和 `rsu_adapter_slots=2` cache stress 是可控构造，只用于验证 proposal 链路与 telemetry，不作为正式论文结论。

## 核心结果

- semantic_ai_service_active: `True`
- adapter_diversity_activated: `True`
- cache_capacity_profile_active: `True`
- eviction_activated: `True`
- handoff_pressure_active: `True`
- do_not_freeze: `true`

## Policy Rows

- `sa_ghmappo`: evaluated, episodes=6
- `ippo`: missing (not_registered_or_not_evaluable)
- `ppo`: evaluated, episodes=6
- `popularity_cache_heuristic`: evaluated, episodes=6
- `reactive_greedy`: evaluated, episodes=6

missing_policy_rows:

```json
{
  "ippo": "not_registered_or_not_evaluable"
}
```

## 问题回答

1. 本轮是不是新数据集？

不是。它是基于真实 mobility trace + 真实 Alibaba DAG structure 的 controlled AI-service/cache-stress smoke。

2. `semantic_ai_service` 是否实际生效？

`True`。benchmark rows 中写入了 `adapter_assignment_profile=semantic_ai_service`。

3. 实际 benchmark rows 中是否出现多个 adapter？

`True`。本轮 rows 的 `required_adapter_count` 至少达到 `4.0`，最大达到 `5.0`。

4. cache capacity profile 是否实际生效？

`True`。rows 中 `cache_capacity_enabled` 大于 0，并记录了 slot、used、remaining、occupancy telemetry。

5. 是否发生 eviction？

`True`。总 `eviction_count=46.0`。

6. 哪些 policy 成功评估，哪些缺失？

见上方 Policy Rows。缺失项没有伪造结果。

7. 如果 IPPO/PPO 缺失，原因是什么？

`ippo` 当前缺失原因：`not_registered_or_not_evaluable`。`ppo` 当前缺失原因：`none`。

8. SA 相对 popularity 的结果如何？

`SA=92.556667, popularity=88.395, delta=4.161667, result=win`。

9. SA 相对 IPPO/PPO 的结果如何？

SA vs IPPO: `candidate_or_baseline_missing`。SA vs PPO 详见 `diagnosis_summary.json` 的 `sa_vs_ppo_result`。

10. 从 reward、continuity、cache miss/cold start、backhaul、eviction 看，SA 的优势或劣势在哪里？

本轮是 smoke，结论只用于定位链路。SA 相对 popularity：

- `total_reward`: sa_ghmappo=92.556667, popularity_cache_heuristic=88.395, delta=4.161667, result=win
- `workflow_continuity_rate`: sa_ghmappo=0.583334, popularity_cache_heuristic=1.0, delta=-0.416666, result=loss
- `handoff_failure_rate`: sa_ghmappo=0.0, popularity_cache_heuristic=0.0, delta=0.0, result=tie
- `backhaul_traffic_cost`: sa_ghmappo=221.333333, popularity_cache_heuristic=424.0, delta=-202.666667, result=win
- `adapter_miss_count`: sa_ghmappo=5.0, popularity_cache_heuristic=0.0, delta=5.0, result=loss
- `adapter_cold_start_count`: sa_ghmappo=0.0, popularity_cache_heuristic=0.5, delta=-0.5, result=win
- `eviction_count`: sa_ghmappo=1.5, popularity_cache_heuristic=3.0, delta=-1.5, result=win

SA 相对 PPO：

- `total_reward`: sa_ghmappo=92.556667, ppo=72.213333, delta=20.343334, result=win
- `workflow_continuity_rate`: sa_ghmappo=0.583334, ppo=0.527778, delta=0.055556, result=win
- `handoff_failure_rate`: sa_ghmappo=0.0, ppo=0.166667, delta=-0.166667, result=win
- `backhaul_traffic_cost`: sa_ghmappo=221.333333, ppo=250.666667, delta=-29.333334, result=win
- `adapter_miss_count`: sa_ghmappo=5.0, ppo=5.666667, delta=-0.666667, result=win
- `adapter_cold_start_count`: sa_ghmappo=0.0, ppo=0.0, delta=0.0, result=tie
- `eviction_count`: sa_ghmappo=1.5, ppo=1.666667, delta=-0.166667, result=win

11. 下一轮建议是什么？

建议先补齐 IPPO checkpoint/eval rows，随后在同一 proposal smoke 上复查 SA vs IPPO/PPO/popularity。若 cache miss/cold start 或 backhaul 差距集中，再考虑 policy-side prefetch/cache-admission bias；不建议直接把本 smoke 当正式 split。

12. 本轮是否可以 freeze？

不可以。本轮只是 proposal smoke，`do_not_freeze=true`。

## 输出

```json
{
  "benchmark_rows": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\benchmark_rows.csv",
  "policy_comparison_summary": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\policy_comparison_summary.csv",
  "adapter_diversity_summary": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\adapter_diversity_summary.csv",
  "cache_eviction_summary": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\cache_eviction_summary.csv",
  "handoff_continuity_summary": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\handoff_continuity_summary.csv",
  "actionmix_summary": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\actionmix_summary.csv",
  "diagnosis_summary": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\diagnosis_summary.json",
  "report": "docs\\agent\\multi_adapter_hard_joint_smoke_round10_report.md",
  "episodes_dir": "artifacts\\analysis\\multi_adapter_hard_joint_smoke_round10\\episodes"
}
```
