# Future-Horizon Cache Oracle Contract

版本：`future_horizon_cache_oracle_contract_v1.0.0`（G08，2026-08-19）

## 时序与控制范围

源码审计确认当前环境在同 step 先执行 cache action，随后判定服务 `cache_hit`。所以 G08 冻结：pre-action state → 至多一次 current-object/current-service-RSU admission或 noop → 原子 eviction/admission → post-action lookup；同 step admission 可以命中当前 request。trace 同时输出 `pre_action_hit` 与 `post_action_hit`。这与 lookup 后 admission 的常见模型不同，不能跨合同直接比较。

G08 v1 matched oracle 只控制 current-RSU placement/admission、victim和 noop。mobility、DAG、service topology、vehicle/cloud execution、prediction、prefetch、migration和 state bundle transfer固定且不属于 oracle action。adapter admission 产生 resident size 等量 transfer MB；不允许免费瞬移。每 RSU 独立容量；slot 每 object 占 1，MB 使用 catalog resident size。oversized 在 planning 前不可 admission且不 eviction；multi-victim 必须最小充分并原子提交。capacity disabled 为 `not_applicable`/fail-fast。

## Horizon、objective 与 optimality

finite horizon 支持 `H={1,3,6,12}`，在 request index `t` 只可见 `[t, min(t+H, episode_end))`，包含当前 request并在 episode 尾部截断。每步重算的 rolling horizon 与 `full_trace_exact_diagnostic_v1.0.0` 是不同 identity；后者不能冒充 finite-horizon result。不同 H 不保证性能严格单调。

主 objective `lex_hit_mb_hit_count_transfer_evicted_churn_v1.0.0`：

1. 最大化 visible future hit MB；
2. 最大化 hit count；
3. 最小化 adapter transfer MB；
4. 最小化 evicted/churn MB；
5. 按 canonical `(rsu_id, object_id, action, victims)` tie-break。

`exact_rolling_enumeration_v1.0.0` 对有限窗口穷举可行动作并 memoize state，输出 JSON-safe action/state trace及逐步 capacity invariant。达到 state limit 返回 `unknown_state_limit` 并停止，不执行 silent greedy fallback。approximate solver 当前未实现，因而没有结果可以称 approximate oracle或 upper bound。

## Gap 语义

baseline comparison 必须匹配 replay fingerprint、capacity unit/value、initial-state fingerprint，并来自 raw request outcome。object/byte hit 使用 `oracle - baseline`；transfer/churn 使用 `baseline - oracle`；分母为 0 时 normalized gap 为 `null`。G06 future-reuse proxy、reward和 legacy aggregate 均被拒绝。

这里的 gap 是相同外生 replay 与可行域下的 placement opportunity gap，不是 causal regret、真实 latency gain或端到端系统收益。request-level observed/counterfactual latency 仍缺失，latency gap固定 `unavailable/null`。

## CLI 与 artifact

```bash
.venv/bin/python scripts/run_future_horizon_cache_oracle.py \
  --fairness_manifest_path <g07_manifest.json> \
  --request_replay_path <request_replay.json> \
  --observed_baseline_path <raw_lru.summary.json> <raw_fifo.summary.json> \
  --output_dir artifacts/analysis/future_horizon_cache_oracle_validation_<run_id> \
  --horizons 1 3 6 12
```

入口重验 G07 manifest/replay、拒绝覆盖、不读取 hidden、不运行 formal benchmark，并输出 replay/validation、resolved config、oracle result/action trace、baseline gap、capacity/horizon audits、command log和 integrity manifest。
