# Cache Information Sufficiency and MARL Necessity Audit Contract

版本：`information_sufficiency_audit_contract_version = 1.0.0`（G10，2026-08-19）

可选 trace：`decision_observation_trace_version = 1.0.0`

## 目的、输入与禁止边界

G10 是只读诊断：把源码中的真实 observation/action/actor contract 与严格匹配的 G07 manifest、G08 external request replay/exact oracle action trace、G09 request opportunity rows 对齐，审计 decision-time 信息覆盖、recoverability、aliasing、information gain 和实体级 MARL 必要条件。它不训练或选择 policy，不修改 observation、encoder、agent、critic、reward、action 或 checkpoint，也不运行 formal/holdout/hidden/G11。

输入必须共享 G07 manifest identity 与 G08 replay fingerprint，request ID 必须逐一对齐。拒绝 hidden manifest、reward、aggregate、事后 service/cache result、oracle future、learned hidden state、NaN/inf、duplicate ID 和 fingerprint mismatch。G09 的 baseline×H 复制行不是独立样本；独立性按原始 `evaluation_unit_id` 计。

## 真实 architecture 分类

- `single_controller`：一个 actor 产生唯一环境动作；当前 PPO 属于此类。
- `factorized_controller`：一个 controller 内部分解多个 role head，但仍聚合为一个环境动作。
- `controller_level_ctde`：controller role heads + centralized critic；当前 MAPPO 和 SA-GHMAPPO 属于此类。
- `entity_level_marl`：至少两个真实 vehicle/RSU actor，各有独立局部 observation 和可归属动作；当前实现不属于此类。

`GymVecEnv` 每步只接收一个 `semantic_discrete_5` action。MAPPO/SA-GHMAPPO 的 slow/cache、fast/execution、event/handoff heads 按固定优先级聚合后才由 `ActionAdapter` 解码。因此多个 head、参数共享、centralized critic 和 MAPPO 命名都不等于多个实体。源码证据固定记录在 `architecture_audit.json`。

## Decision-time observation trace

合法 trace 必须在 action 选择前采集，并逐 request 记录 controller identity、observation contract、raw semantic fields、flattened vector/dimension、feature-name→index、availability/null mask、normalization、local/global/predictor/history scope、action mask/eligible actions和作为独立 outcome 的 actual selected action。raw/flattened 映射必须一致，顺序和版本必须冻结。

trace 不得包含 oracle future、reward、post-action state、service result或 learned hidden state。旧 artifact 缺 trace 时返回 `unavailable`，不得通过不同 request rerun、aggregate 或事后 CacheEvent 拼接 observation。G10 v1 没有修改运行时 instrumentation；现有真实 G07–G09 controlled artifact 因缺 trace，只能完成源码 schema audit。

## Field coverage 与 recoverability

固定审计 15 项：object identity/size、current/request RSU、cache contents、remaining capacity、capacity unit/value、recency、frequency、future reuse estimate、next-RSU/handoff estimate、cross-RSU cache、transfer cost、multi-victim/capacity pressure、coordination 和 DAG future demand。

每项严格区分环境内部存在、semantic observation 存在、actor encoder 消费、critic 消费、controller 可见和事后 recorder 才存在。输出 actor/controller/critic-only/predictor/oracle-only/absent count/rate，并按 opportunity reason 与 missed bytes 加权。Coverage 只是必要条件，不等于 sufficiency。

recoverability 使用预先冻结的绝对/相对容差 `1e-9`，状态为 `exact/lossy/absent/inconsistent`。不按结果调容差。若没有匹配 trace，所有动态 recoverability 结论保持 unavailable。

## Aliasing、projection 与信息统计

每个 actor-local、controller-global、critic-only 和 predictor-augmented projection 同时计算 exact semantic key、flattened SHA-256 和 `fixed_information_projection_buckets_v1.0.0` coarsened key。固定 float bucket 为 `[-1,0,.25,.5,.75,1,2,4,8,16,32,64,128]`；不得按结果或分位数调整。连续 observation 没有 exact duplicate 不能证明充分。

离线 feature-removal projection 固定为：移除 cross-RSU/global cache、移除 handoff/next-RSU prediction、移除 cache/capacity/resident state。projection 只遮蔽离线字段，不运行或修改 policy。

统计量使用 `empirical_plugin_discrete_v1.0.0`：`H(action)`、`H(action|projection)`、information gain、NMI 和可选 CMI。默认至少 8 条样本且至少 2 个独立 evaluation units；cell 平均少于 5 条发出 sparse warning。MI/CMI 是 association，不是因果作用；H/baseline 复制不增加独立样本。

## Opportunity identifiability

固定检查 admission、wrong target、eviction、capacity、transfer、topology、handoff-adjacent reuse 和 multi-victim。Verdict 为 `identifiable/partially_identifiable/not_identifiable/oracle-only/unavailable`。G08 future visibility 与 G09 future reuse label不能作为当前可观测输入。

## MARL necessity 门禁

结论等级为 `supported/partially_supported/not_supported/unverifiable`；总 verdict 为 `SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED/UNVERIFIABLE`。

实体级 MARL 必须同时满足：两个以上真实决策实体、独立局部 observation、实体可归属动作、并发或耦合动作、不可消除的局部信息限制、稳定 cross-entity 增量信息、非极小 cross-RSU opportunity、多个独立 evaluation units，以及 centralized controller 未直接拥有全部必要信息。任一关键条件缺失不得声称“MARL 必要”。

`centralized_information_beneficial` 只说明 global view 减少 local ambiguity；`factorized_decision_beneficial` 只说明 controller 分解可能有用；二者都不推出 entity-level MARL。当前源码缺真实 entity actor/local isolation/action ownership；当前真实 controlled artifact又只有 1 request/1 unit且无 observation trace，因此 G10 validation 的 entity-level verdict 为 `unverifiable`，不是 supported。

## GNN/GAT 与禁止主张

图结构、DAG encoder 或 RSU set encoder 不自动证明 GNN/GAT 必要。只有 matched projection 在多个独立 unit 上显示图/跨实体信息稳定降低 oracle-action ambiguity，且非图 controller 不能获得同等信息时，才可讨论结构收益；仍需独立算法实验才能谈性能。

禁止表述：MAPPO 名称/多 head/参数共享/centralized critic 证明实体 MARL；集中信息有用等于 MARL 必要；MI/CMI 是因果效应；1 request×4H×5baseline 是 20 个独立样本；当前 G10 证明 GNN/GAT、算法优劣或 paper readiness。

## CLI、artifact 与 G11 边界

```bash
.venv/bin/python scripts/audit_cache_information_sufficiency.py \
  --fairness_manifest_path <g07.json> \
  --request_replay_path <g08_replay.json> \
  --oracle_action_trace_path <g08_oracle_action_trace.json> \
  --opportunity_rows_path <g09_request_opportunity_rows.json> \
  --agent_identity sa_ghmappo \
  --output_dir artifacts/analysis/cache_information_sufficiency_validation_<run_id>
```

可选 `--observation_trace_path` 与 `--audit_config_path`。CLI 默认拒绝覆盖，输出 architecture、field map、recoverability、aliasing、identifiability、information gain、MARL verdict、input validation、resolved config、synthetic validation、command log 和 integrity manifest。

G11 若未来开展，只能消费本合同的明确 blocker/required evidence；G10 不授权 observation 扩展、entity wrapper、新 agent、训练、调参或 benchmark。
