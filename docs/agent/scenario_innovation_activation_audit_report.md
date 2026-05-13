# scenario_innovation_activation_audit

## Scope

本轮只做 benchmark 数据激活度审计。未修改环境逻辑、reward、policy、checkpoint selection、baseline，也未训练。

## Inputs

- Mixed: `artifacts\analysis\sa_mechanism_actionmix_diagnosis_round5\benchmark_mixed\main_results_mixed_informative_20260426_023736_742838`
- Full: `artifacts\analysis\sa_mechanism_actionmix_diagnosis_round5\benchmark_full\main_results_full_stratified_20260426_023920_694043`

## Outputs

- `artifacts\analysis\scenario_innovation_activation_audit\dag_hardness_summary.csv`
- `artifacts\analysis\scenario_innovation_activation_audit\cache_pressure_summary.csv`
- `artifacts\analysis\scenario_innovation_activation_audit\mobility_handoff_pressure_summary.csv`
- `artifacts\analysis\scenario_innovation_activation_audit\innovation_activation_summary.csv`
- `artifacts\analysis\scenario_innovation_activation_audit\scenario_bucket_summary.csv`
- `artifacts\analysis\scenario_innovation_activation_audit\diagnosis_summary.json`

## Key Answers

1. hard_joint / mechanism_activating 占比：
   - mixed hard_joint: `0.666667`
   - mixed mechanism_activating bucket: `0.333333`
   - full hard_joint: `0.666667`
   - full mechanism_activating bucket: `0.333333`
2. 是否存在 easy_static_like 稀释：`easy_static_like rates: mixed=0.0, full=0.0; supplied rows do not contain an easy_static_like bucket under the current thresholds.`
3. DAG 是否足够复杂：`large_dag_rate_mean=1.0, deep_dag_rate_mean=1.0; selected workflows are nontrivial but limited to a small catalog.`
4. cache/model/adapter 压力是否足够：`SA miss_rate_mean=0.028624, prefetch_attempt_mean=0.433333, unique_adapter_per_episode_mean=1.0; cache/prefetch pressure exists mainly in mechanism windows, but model/adapter diversity is low and capacity/occupancy telemetry is missing.`
5. handoff / cross-RSU 是否足够：`handoff_during_workflow_rate_mean=0.6, cross_rsu_workflow_rate_mean=0.6; pressure is stratified but not uniformly hard.`
6. SA 相对 IPPO 的优势是否集中 hard_joint：`IPPO/PPO rows are not present in the supplied round5 benchmark rows.`
7. SA 相对 popularity 的劣势是否在 cache/prefetch 场景：`SA reward losses vs popularity are concentrated in buckets with prefetch/cache-admission tie-break behavior.`
8. 下一轮 split 建议：`Do not modify split in this round. Next benchmark split should explicitly preserve a larger hard_joint slice, report cache capacity/occupancy telemetry, and separate cache-dominant from joint DAG+cache+handoff scenarios.`

## Missing Fields

`cache_capacity, cache_occupancy_rate_mean, deadline_tightness, rsu_dwell_time_mean, rsu_dwell_time_p10, rsu_dwell_time_p90`

## Notes

- DAG 指标来自 Alibaba workflow CSV，经 `WorkflowDatasetBuilder` 离线解析。
- cache/action/migration 指标来自 round5 evaluator rows。
- `rsu_dwell_time_*`、`deadline_tightness`、`cache_capacity`、`cache_occupancy_rate_mean` 当前没有可靠底层字段，本轮只在 missing fields 中记录。
