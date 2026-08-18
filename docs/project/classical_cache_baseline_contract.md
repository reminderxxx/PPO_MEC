# Classical Cache Baseline Contract

版本：`1.0.0`（2026-08-18）

五个 live baseline 为 `reactive_lru`、`reactive_fifo`、`reactive_lfu`、`reactive_aging_lfu`、`reactive_random`。它们共享 `reactive_current_rsu_admission_v1`：无节点走既有合法 fallback，无关联 RSU 走 vehicle fallback，当前 RSU miss 选择 cache fill，hit 选择 steady offload；不使用 prediction、prefetch、handoff prepare、popularity 或学习参数，并遵守相同 action mask。唯一变量是环境拥有的 eviction policy。因此作用域固定为 `reactive placement/admission + selected eviction policy`，不是完整 cooperative caching。

容量、initial cache、request stream、system seed 和 action semantics 必须匹配。registry 的 `required_eviction_policy` 必须与 `cache_capacity_profile.eviction_policy` 相等，否则 evaluation fail-fast。每次 episode 使用新 env 所拥有的 policy 实例；reset 清除 policy state。

机制：

- LRU：按 `(last_used_step, object_id)`；hit/already-cached hit 更新 recency。
- FIFO：按 `(admission_order, object_id)`；hit 和重复 admission 不改变顺序，evict 后 re-admission 成为最新。
- LFU：initial/runtime admission frequency 均为 `0`，hit `+1`，按 `(frequency, last_used_step, object_id)`。
- Aging-LFU：默认 `aging_interval=8`、`aging_factor=0.5`。每 RSU 独立 policy callback clock；在 hit/admission/eviction callback 的业务更新前，每逢 interval 执行 `floor(frequency * factor)`，下界为 `0`。
- Random：只使用私有 `random.Random`；`policy_seed = benchmark run seed`，相同 seed 与事件序列可复现。候选先按 object id 稳定排序，再无放回采样。

所有 policy 支持 `adapter_slots` 和 `mb`，返回统一 `EvictionPlan`，并导出 JSON-safe detached state。validation 入口为 `scripts/validate_classical_cache_baselines.py`，产物写入 `artifacts/analysis/classical_cache_baseline_validation_<run_id>/`。当前状态仅为 benchmark-ready 的 controlled mechanism validation，不是 paper-grade 性能结果；正式 cache metrics、oracle 与 paper fairness manifest 仍未实现。
