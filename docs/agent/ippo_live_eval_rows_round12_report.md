# ippo_live_eval_rows_round12 报告

## 范围

本轮补齐 IPPO live eval/checkpoint rows。没有 freeze，没有修改 reward，没有修改 SA-GHMAPPO policy，没有修改 `popularity_cache_heuristic` / `reactive_greedy` 行为，没有修改 checkpoint selection，也没有替换 mixed/full benchmark。

本轮结果仍是 `multi_adapter_hard_joint` proposal smoke，不是正式论文结果。

## 1. 本轮是否训练？

是，但只训练了 IPPO smoke checkpoint。训练 profile 是 `ippo_smoke_round12`，episodes=`6`，update_count=`3`。该 checkpoint 不是 fully tuned baseline。

## 2. IPPO agent 是否已注册？

是。`src/agents/registry.py` 已注册 `ippo`。

## 3. `list_evaluable_agents()` 是否包含 ippo？

`True`。

## 4. IPPO 是否有 checkpoint？

是。checkpoint: `artifacts\training\ippo_smoke_round12\ippo\ippo_smoke_round12_train_20260426_195604_254218_seed7\checkpoints\latest.pt`。

## 5. IPPO 是否成功产生 benchmark rows？

是。IPPO rows=`6`，总 rows=`30`。

## 6. IPPO 与 SA-GHMAPPO 的结构差异是什么？

IPPO 是 flat semantic encoder + independent critic + shared wrapper decision stream 的 independent-style PPO baseline。SA-GHMAPPO 使用图/层级机制、机制窗口 guard/auxiliary 等主方法能力。

## 7. IPPO 是否使用 centralized critic？

不使用。

## 8. IPPO 是否使用 SA 的 hierarchical mechanism？

不使用。它没有 SA hierarchy、graph-continuity critic、heuristic imitation、mechanism auxiliary 或 mechanism logit prior。
训练日志中若出现 deterministic prepare 等基类字段，只是因为 IPPO 复用通用 PPO base；在 IPPO 中 `use_hierarchy=False` 且 `event_head_enabled=False`，这些 SA 机制路径不激活。

## 9. IPPO 与 flat PPO row 的区别是什么？

`ippo` 是独立注册的 agent/checkpoint/run 名称，policy_type=`ippo_policy`；`ppo` row 是 current `PPOAgent` 加载 existing `flat_ppo` checkpoint alias。两者都基于 flat PPO 基础实现，但 IPPO 本轮有独立 smoke checkpoint 和 live rows。

## 10. SA vs IPPO 在 hard_joint smoke 下结果如何？

SA reward `92.556667`，IPPO reward `52.9`，SA-IPPO reward delta `39.656667`，result `win`。逐指标见 `ippo_vs_sa_summary.csv`。

## 11. SA vs PPO / popularity / reactive 是否与 round10 基本一致？

本轮复用同一 proposal smoke 协议。SA reward `92.556667`，PPO `72.213333`，popularity `88.395`，reactive `65.175`。方向上仍是 SA 高于 PPO/reactive，并在 reward/backhaul 上优于 popularity，但 continuity 低于 popularity。

## 12. continuity 低的问题是否仍然存在？

存在。SA continuity `0.583334`，popularity continuity `1.0`。

## 13. IPPO 是否也出现 stall / adapter miss / cold start？

IPPO continuity `0.416667`，adapter miss `7.0`，cold start `0.0`。它也出现 adapter miss/stall，符合 smoke baseline 预期。

## 14. 下一轮建议

建议先补 `local_only` / `reactive_offloading` / `reactive_caching` live rows，随后做 `hard_joint_policy_failure_diagnosis`，最后再考虑 policy-side limited prefetch/cache-admission bias。

## 15. 本轮是否可以 freeze？

不可以。本轮仍是 proposal smoke / IPPO live baseline check。

## 输出

```json
{
  "benchmark_rows": "artifacts\\analysis\\ippo_live_eval_rows_round12\\benchmark_rows.csv",
  "ippo_training_summary": "artifacts\\analysis\\ippo_live_eval_rows_round12\\ippo_training_summary.json",
  "policy_comparison_summary": "artifacts\\analysis\\ippo_live_eval_rows_round12\\policy_comparison_summary.csv",
  "ippo_vs_sa_summary": "artifacts\\analysis\\ippo_live_eval_rows_round12\\ippo_vs_sa_summary.csv",
  "cache_eviction_summary": "artifacts\\analysis\\ippo_live_eval_rows_round12\\cache_eviction_summary.csv",
  "continuity_stall_summary": "artifacts\\analysis\\ippo_live_eval_rows_round12\\continuity_stall_summary.csv",
  "diagnosis_summary": "artifacts\\analysis\\ippo_live_eval_rows_round12\\diagnosis_summary.json",
  "report": "docs\\agent\\ippo_live_eval_rows_round12_report.md"
}
```
