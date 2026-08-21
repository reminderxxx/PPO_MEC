# Typed Model Cache Runtime Contract

版本：`typed_model_cache_runtime_contract_v1.0.0`
状态：G14R 已补齐 v1.1 formal execution binding；正式 checkpoint 与正式实验仍未运行。

## 目的与唯一解析入口

`src/runtime/typed_model_cache_runtime.py` 是训练、评估、benchmark 与 fairness 的共享运行时解析器。调用方不得各自推断 catalog、容量、初始状态或 checkpoint 兼容性。解析结果是 JSON-safe 的 `resolved_model_cache_runtime`，其 policy-invariant semantic projection 计算 `runtime_contract_sha256`；本机绝对路径、config 来源、执行时 Git commit和五baseline允许变化的eviction policy/seed/config不进入该 hash。真实policy identity仍在每个summary/row单独输出，并进入checkpoint provenance gate。

旧配置缺少 profile 字段时解析为 `legacy_adapter_only_v1`，不改变历史 adapter-only 默认。`typed_base_adapter_state_v1` 不允许回退 legacy：必须显式给出 repository catalog path、canonical fingerprint，以及 `cache_capacity_profile.enabled=true/unit=mb/capacity_mb>0`。typed slot、非有限/非正 MB、typed catalog 被 legacy runtime 消费、legacy catalog 被 typed runtime 消费均立即失败。

## Resolved 字段

核心字段包括：

- `model_cache_profile`、`typed_model_cache_contract_version=1.0.0`；
- `typed_catalog_path/file_sha256/fingerprint`；
- `cache_capacity_profile.enabled/unit/capacity_mb/rsu_adapter_slots`；
- `object_taxonomy`、resident/transfer size 与 stable object fingerprint；
- `dependency_map/fingerprint`、`compatibility_map/fingerprint`；
- `initial_per_rsu_typed_state/typed_initial_state_fingerprint`；
- `pinned_evictability_metadata/fingerprint`；
- `typed_cache_transaction_contract_version` 与 action-before-lookup、atomic rollback、dependency-safe eviction；
- `cache_event_schema_version=1.3.0`、`cache_efficiency_metrics_contract_version=1.2.0`、`cache_trace_context_version=1.0.0`；
- `request_replay_typed_contract_version` 与 `runtime_contract_sha256`。

Catalog fingerprint 在每次配置加载与 environment 消费前重算。Initial state 从 catalog 的 per-RSU typed profiles 排序解析，计算 resident MB 并验证无需 trim；dependency、compatibility 与 pinned/evictability 也分别重算。任何 config/fairness/checkpoint/benchmark 不一致都不能通过 gate。

Repository controlled catalog 现显式包含 Alibaba `legacy_batch_type` workflow 产生的 `adapter_batch_type_1`，映射到 `base:veh_base_v1`，resident/transfer 均为 64 MB，来源仍是 `repository_native_controlled`。这只是受控 mapping，不是真实联合 model-cache trace。

## Training 与 checkpoint

G14R 新增 `formal_training_contract 1.0.0`。未提供 formal protocol 时，遗漏
`checkpoint_every_updates` 保持 legacy 每 update 保存；提供 v1.1 protocol 时，episodes、update
interval、batch、max steps、expected update count、checkpoint cadence 与 agent config 全部从 manifest
解析，任何 CLI 差异立即拒绝。`latest.pt` 每次 update 保存但永不进入 selection；正式 candidate
只在 cadence 整除 update 保存。Resume checkpoint 必须带相同 schedule。

Agent config companion 逐 agent 传入共享 registry 并通过 `_checkpoint_config()` 审计。SA-GHMAPPO 的
`auxiliary_coef=0.06` 写入 resolved config、summary 与 checkpoint metadata；其他 agent 不接收该
专用字段。该层没有修改 SA loss 公式，也没有改动 SA 专用训练入口。

共享入口：

```bash
.venv/bin/python scripts/train_algo_pool_real_sample.py \
  --agent_name ppo \
  --model_cache_runtime_config configs/benchmark/typed_model_cache_controlled_lru.yaml \
  ...
```

同一入口支持 `sa_ghmappo`、`ppo`、`mappo`、`cache_offload_drl` 及 registry 中其余 live trainable agent。G14A 只增加运行时参数、环境 binding、summary 与 serialization provenance，不修改网络、loss、reward、action 或训练超参数。

Checkpoint 的 `training_metadata.typed_runtime_provenance` 写入 execution Git commit、agent、training seed、profile/version、catalog/initial/runtime hash、capacity unit/value、observation/action contract 与 shape、reward/environment contract、CacheEvent/metrics 版本、train window-plan identity。checkpoint 文件自身 SHA-256 在序列化完成后由外部 provenance manifest 绑定，避免不可实现的 self-hash。

`validate_checkpoint_provenance()` 只返回三种状态：`compatible`、`incompatible`、`unavailable_legacy_metadata`。Typed learned benchmark 必须有外部 per-agent/per-seed binding 且 gate 为 `compatible`；旧 checkpoint、错误 agent/seed/Git/catalog/capacity/shape/reward/environment/window/hash不能冒充 typed checkpoint。

## G14R3 portable resource and checkpoint location binding

训练、dev、formal 与 support 现在共同消费 `portable_resource_identity.py`。Runtime config、dataset、window、
fairness 与 checkpoint manifest 都由 logical ID + role + schema + size/hash 解析；cwd 猜测和隐式 workflow default
不能进入正式链。Checkpoint scientific identity 与 artifact location 分离，freeze hash 不包含主机路径；迁移后
必须重新验证 checkpoint hash、agent、seed、capacity、Protocol/catalog/runtime/split/window 与 selection
provenance。无效 G14C v3 root 有硬拒绝规则。Legacy 未传 registry 的非正式入口仍兼容，但 v1.3 正式模板均
显式传入 registry/root/resource IDs。

## Fairness 与 benchmark

Typed fairness manifest 使用 consumer-safe `1.1.0`，旧 legacy manifest 继续是 `1.0.0`。`cache_contract.typed_model_cache` 冻结 catalog、taxonomy、dependency/compatibility/initial/pinned fingerprints、MB capacity、transaction、CacheEvent 1.3、metrics 1.2、trace context、typed replay 与 oracle compatibility。五个 reactive baseline 的 `baseline_matrix` 仍严格只有五项且 10 组 pairwise diff 只允许 policy identity 字段；可选 `controller_agents` 只扩展实际 benchmark agent matrix，不改变 reactive only-policy-difference 审计。

`scripts/benchmark_main_results.py` 支持：

- legacy adapter slots；
- legacy adapter MB；
- typed MB；
- typed slot 显式拒绝。

Typed formal-capable 路径强制提供 validated fairness manifest；learned agent 还强制提供 external checkpoint provenance manifest。旧 benchmark 不提供 fairness manifest 时继续运行并写 `fairness_manifest_status=unavailable`。Raw episode summary 保留 CacheEvent/trace context；row 只保留 nullable scalar 和轻量 provenance；aggregate/run manifest 写 manifest ID/full/semantic hash 与 runtime hash；benchmark integrity manifest包含 raw episode summary。

## Claim boundary

G14A 仅证明正式入口具备 typed MB 运行能力。它没有冻结 G14 split/protocol，没有正式 checkpoint，没有运行 formal、holdout、hidden 或 G15，也没有形成算法排名、latency-saved 或论文结论。完成 G14A 后必须回到计划窗口重新运行 readiness gate，再决定 G14B。
