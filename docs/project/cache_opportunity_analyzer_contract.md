# Cache Opportunity Analyzer Contract

版本：`cache_opportunity_analyzer_contract_version = 1.0.0`（G09，2026-08-19）

## 目的与 claim boundary

G09 在严格匹配的外生 request replay、G08 exact rolling oracle action trace、G07 fairness identity、initial cache/capacity contract 与 baseline raw CacheEvent outcome 上，独立重算 request/window placement opportunity。它回答机会是否存在、类型、baseline 捕获/错失和集中位置，并输出决策信息需求标签。

结果是相同 replay 与可行域下的 diagnostic placement-opportunity association，不是 causal regret、端到端 latency gain、算法优劣、MARL 必要性或 G10 信息充分性结论。禁止读取 reward、legacy aggregate、learned-policy hidden state、文档摘要或不匹配 artifact。

## 输入、匹配与 provenance

必需输入：G07 manifest；G08 `cache_request_replay_version=1.0.0`；G08 `future_horizon_cache_oracle_contract_v1.0.0` result/action trace；与 replay 逐 request 对齐的 raw baseline outcome；manifest 内 initial cache/capacity；显式 H 与 analyzer config。

运行前必须一致：request replay fingerprint、manifest semantic hash、capacity unit/value、initial-state fingerprint、oracle H、rolling identity、objective `lex_hit_mb_hit_count_transfer_evicted_churn_v1.0.0` 和 exact optimality。duplicate ID、缺 request outcome、缺/非正 object size、负值、NaN/inf、顺序或分母不守恒均 fail-fast。缺失值不转为零。所有输出包含版本、identity、coverage 和重算 fingerprint；大型 request rows 独立保存。

## A. Exogenous demand opportunity

仅由 replay 计算 request/MB、unique object/MB、first/repeated/compulsory-cold count/MB、previous reuse distance、next-use distance、H={1,3,6,12} reuse count/MB/rate、对象 frequency/bytes、object-size concentration、RSU-local/cross-RSU/handoff-adjacent reuse、合法 service/cache target breadth和 topology-ineligible reuse。

first request 不是 eviction failure。没有 next use 且 episode 尾部不足 H 的 request 是 right-censored，从该 H 的 available denominator 排除。跨 RSU 同 object 只有历史 cache target 与当前合法 service target 相交时才标记 directly usable。Demand reuse 不等于 feasible cache hit。

## B. Feasible oracle opportunity

由 G08 action trace 计算 oracle-achievable hit count/MB、admission、eviction-required、capacity-binding、noop/no-benefit、oversized infeasible、transfer-required、multi-victim、per-RSU/per-H opportunity及 density：

```text
oracle opportunity density = oracle post-action hit requests / valid requests
oracle byte opportunity density = oracle post-action hit MB / requested MB
```

单独报告 initial-cache natural hit、`pre_action_hit=false` 后同 step admission hit、victim replacement hit、不可行和无收益。由于 G08 冻结 action-before-service-lookup，同 step admission 可命中当前 request；不能与 lookup-before-admission 合同直接比较。initial natural hit 不称为策略改进机会。

## C. Baseline capture/loss

每个 baseline/H/request 输出四象限：both hit、baseline miss/oracle hit、baseline hit/rolling-oracle miss、both miss。captured 是 both hit；missed 是 baseline miss/oracle hit。报告 count/MB capture rate、object/byte gap、transfer/churn `baseline - oracle`。

Rolling finite-H oracle每步 replanning且有词典序 tie-break，局部 `baseline hit && oracle miss` 不表示 baseline 超过理论上界；固定使用 `baseline_hit_oracle_miss` 独立报告。

## Primary taxonomy 与 secondary evidence

每个 baseline/H/request 恰好一个 primary reason，冻结优先级：

1. `unavailable_or_incomparable`
2. `initial_cache_hit`
3. `captured`
4. `baseline_hit_oracle_miss`
5. `right_censored`
6. `oversized_infeasible`
7. `topology_not_eligible`
8. `compulsory_first_request`
9. `no_reuse_within_horizon`
10. `wrong_cache_target`
11. `eviction_choice`
12. `insufficient_free_capacity`
13. `transfer_tradeoff`
14. `admission_not_selected`
15. `capacity_not_binding`

`wrong_cache_target` 要求 oracle/baseline target 均合法且不同。`eviction_choice` 要求 victim set 不同，且 baseline victim 在 H 内确有后续 request；仍只作合同约束解释，不声称因果 regret。Secondary evidence 可多选，包括 demand reuse、local/cross-RSU、handoff、topology、oracle replacement/multi-victim/transfer和 baseline admission/eviction。G06 future-reuse proxy若未来接入只能作为 secondary evidence。

## Gap decomposition、bucket 与 concentration

请求行冻结维度：baseline、H、capacity unit/value、evaluation unit/window、workflow、object/adapter、size bucket、frequency bucket、reuse-distance bucket、current/request/next及合法 target topology、handoff、capacity pressure、oracle action、primary reason。

固定 bucket：size MB `<=32, (32,64], (64,128], >128`；frequency `1, 2–3, 4–7, >=8`；reuse distance `1, 2–3, 4–6, 7–12, >=13, none`；capacity pressure `<0.5, [0.5,0.85), >=0.85`。不得按当前结果分位数改边界。

Concentration 固定 top-k `{1,3,5}`、positive object-gap Gini、zero-opportunity window ratio。Oracle density strata 固定为 low `[0,0.25)`、medium `[0.25,0.75)`、high `[0.75,1]`；不足 5 个 entity 输出 `small_sample_concentration_unstable`。

## Information requirement labels

标签包括 current object identity/size、current RSU、cache contents、remaining capacity、recency/frequency、future reuse estimate、next-RSU/handoff estimate、cross-RSU cache state、transfer cost、coordination information和 DAG future demand。分类仅为：`decision-time observable`、`history-derived`、`predictor-required`、`oracle-only future information`、`currently absent/unknown`。不得由这些标签推出 MAPPO/MARL 必要性；G10 才讨论信息充分性。

## Availability、right-censoring 与 latency

每组输出 availability、unavailable reason、required fields、available/unavailable request count与 coverage。零分母 rate 为 JSON `null`。当前缺逐 request observed service latency、cold/cloud counterfactual、transfer latency和 stall/restart latency，所以 latency saved/gap/saved-per-MB 永久为 `unavailable/null`，reward和 workflow span 不可替代。

## CLI、artifact 与 G10 接口

```bash
.venv/bin/python scripts/analyze_cache_opportunities.py \
  --fairness_manifest_path <g07_manifest.json> \
  --request_replay_path <g08_request_replay.json> \
  --oracle_results_path <g08_oracle_results.json> \
  --oracle_action_trace_path <g08_oracle_action_trace.json> \
  --baseline_outcome_path <raw_1.summary.json> <raw_2.summary.json> \
  --output_dir artifacts/analysis/cache_opportunity_analyzer_validation_<run_id> \
  --horizons 1 3 6 12
```

默认拒绝覆盖，输出 summary、request rows、by-baseline/H/reason/object/window/RSU、information labels、input validation、reconciliation、command log与 integrity manifest。G10 只可消费版本化 request rows、fixed strata、information labels和 coverage；不得把 G09 association 升格为 information sufficiency 或 MARL conclusion。

## G13 typed consumer

typed artifact使用full-service baseline hit、typed initial object set与oracle total/per-type transfer；不得只读legacy`adapter_transfer_mb`而漏掉base transfer。base hit/adapter miss作为partial readiness evidence，不进入captured full-service opportunity。旧adapter replay继续使用原字段。
