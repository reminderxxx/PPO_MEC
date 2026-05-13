# cache_capacity_eviction_telemetry_round9 报告

## 结论

本轮只做了 cache capacity / eviction / occupancy telemetry 的最小可关闭实现与验证，没有训练、没有 freeze、没有修改 reward 数值、policy 行为、baseline 行为或 checkpoint selection。

默认正式实验仍保持 legacy append-only cache 行为。新增 capacity/eviction 只有在显式传入 `cache_capacity_profile.enabled=true` 时生效；`mixed_informative` 和 `full_stratified` benchmark 没有被替换。

## 修改范围

- `src/envs/core/vec_workflow_core_env.py`
  - 新增可关闭的 `cache_capacity_profile`。
  - 默认 `enabled=false`，保持旧 cache 行为。
  - `enabled=true` 时按 `rsu_adapter_slots` 限制每个 RSU 的 adapter slot。
  - 新增 LRU metadata，adapter hit / admission 时更新 `last_used_step`。
  - 新增 LRU eviction：cache 满且需要 admission 时，移除最久未使用 adapter。
  - 在 `info.metrics_protocol` 中透传 capacity / eviction / occupancy telemetry。

- `src/evaluators/main_results_support.py`
  - 在 benchmark row 诊断字段中追加 cache capacity / eviction / occupancy telemetry。
  - 老 summary 缺失字段时 graceful fallback 为 0 或默认 unit，不破坏已有 schema。

- `scripts/validate_cache_capacity_eviction.py`
  - 新增只读 smoke/validation 脚本。
  - 构造 5 adapter 的最小 workflow，分别验证 `enabled=false` 与 `enabled=true`。

- `configs/benchmark/multi_adapter_hard_joint_proposal.yaml`
  - 继续保持 `proposal_only: true` 与 `do_not_use_for_freeze: true`。
  - 新增 proposal-only cache capacity profile，建议 `rsu_adapter_slots: 2`、`eviction_policy: lru`。
  - 明确标注这是 controlled cache stress setting，不是新真实数据集。

## 问题回答

1. 当前 cache 原来是不是 append-only？

是。原逻辑在 adapter miss 后 admission 到 RSU cache，没有发现 capacity limit 或 eviction 行为。本轮保留该默认行为。

2. 本轮新增的 capacity profile 默认是否关闭？

是。`VecWorkflowCoreEnv` 默认 `cache_capacity_profile.enabled=false`。只有显式传入 enabled profile 时才限制容量。

3. `enabled=false` 时旧行为是否保持不变？

是。验证脚本显示 `capacity_disabled_preserves_old_behavior=True`，没有 eviction，adapter admission 后继续追加。

4. `enabled=true` 时是否能限制 RSU adapter slots？

是。验证脚本使用 `rsu_adapter_slots=2`，结果显示 `capacity_enabled_limit_respected=True`，最大 cache 使用量不超过 2。

5. LRU eviction 是否生效？

是。验证事件中超过 2 个 adapter 后按 LRU 顺序 eviction，依次移除了较早使用的 adapter。

6. 是否能产生 `eviction_count > 0`？

是。`enabled=true` case 的 `eviction_count=3`。

7. 是否能输出 `cache_occupancy_rate`？

是。`enabled=true` 时可计算 occupancy，验证中最大 occupancy 为 1.0。`enabled=false` 时 capacity 未启用，occupancy 记录为 missing/0 fallback。

8. `benchmark_rows.csv` 是否新增 cache capacity / eviction telemetry？

是。`src/evaluators/main_results_support.py` 已追加以下字段聚合：

- `cache_capacity_enabled`
- `rsu_adapter_slots`
- `cache_capacity`
- `cache_used_size`
- `cache_remaining_size`
- `cache_occupancy_rate`
- `cache_admission_added_new_adapter_count`
- `eviction_count`
- `evicted_adapter_count`

同时保留已有 hit/miss、warm/cold、prefetch/migration、backhaul 字段。

9. 该机制是否改变 reward/policy/baseline？

没有。reward 主公式、policy 网络/动作选择、baseline 策略和 checkpoint selection 都没有修改。

10. `multi_adapter_hard_joint_proposal` 是否仍然是 proposal only？

是。该配置仍声明 `proposal_only: true` 和 `do_not_use_for_freeze: true`，没有接管正式 mixed/full benchmark。

11. 下一轮是否可以开始跑小规模 `multi_adapter_hard_joint` smoke？

可以。建议下一轮只做显式 proposal smoke，不作为 freeze 或论文正式结果，先验证多 adapter assignment + capacity eviction + telemetry 的端到端可观测性。

12. 下一轮是否应补 IPPO/PPO rows，而不是继续调 SA policy？

是。当前瓶颈已经转向场景与 telemetry 可证性。建议先补 proposal smoke 与 IPPO/PPO/popularity rows，再判断 SA policy 是否需要继续优化。

## 验证产物

输出目录：

`artifacts/analysis/cache_capacity_eviction_telemetry_round9/`

生成文件：

- `cache_capacity_validation.csv`
- `eviction_event_validation.csv`
- `telemetry_field_coverage.csv`
- `diagnosis_summary.json`

关键验证结果：

- `capacity_disabled_preserves_old_behavior: True`
- `capacity_enabled_eviction_observed: True`
- `capacity_enabled_limit_respected: True`
- `cache_occupancy_rate_observed: True`

## 风险与边界

- 本轮没有运行正式 benchmark，只运行最小 validation。
- proposal config 仍未作为正式 split 使用。
- 当前 capacity 第一版按 adapter slots 计数，不按 MB/size 精确容量。
- base model 暂不参与 cache competition。
- eviction policy 第一版只实现 LRU，没有实现 predicted-value eviction。
