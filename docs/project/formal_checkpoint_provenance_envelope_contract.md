# Formal Checkpoint Provenance Envelope Contract

更新日期：2026-09-06

## 状态

- 唯一 live Protocol：`2.9.0`
- active index：`configs/experiment/typed_model_cache_formal_protocol_v2_9_20260906/protocol_index.json`
- Readiness：`21.0.0`，状态 `READY_FOR_G14C_V16_CLEAN_TRAIN_AND_FORMAL`
- Protocol 2.8 及更早版本：historical/audit-only
- 证据等级：`E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE`

## 两层 schema

正式 checkpoint companion 是 17 字段 provenance envelope。它包含 8 个共享 training identity 字段：

1. `agent_scientific_config_semantic_sha256`
2. `formal_training_execution_binding_sha256`
3. `formal_protocol_semantic_sha256`
4. `execution_commit`
5. `resolved_execution_context_sha256`
6. `formal_agent_order_contract_semantic_sha256`
7. `active_formal_bundle_sha256`
8. `formal_nullable_metric_aggregation_contract_semantic_sha256`

其余 9 个字段属于 envelope 合同：`checkpoint_sha256`、`execution_git_commit`、
`train_window_plan_identity`、`runtime_contract_sha256`、`resolved_agent_config`、`checkpoint_schedule`、
`selection_sha256`、`checkpoint_identity`、`artifact_location`。它们不是 shared identity 的“多余字段”，不得删除。

共享字段集合由 `src/runtime/formal_training_identity.py` 的 capability-aware parser 定义。producer、benchmark
和 typed runtime 不得各自维护另一份字段列表。

## 信任与验证顺序

正式 benchmark 必须在环境 rollout 前完成：

1. 验证 active Protocol 2.9、resolved execution context、execution binding 与 bundle/resource identity。
2. 加载真实 companion 文件并验证完整 17 字段 envelope、路径、generated registry、checkpoint 文件 size/SHA-256。
3. 从已验证的 Protocol/context/binding 投影可信 expected 8-field identity；不得仅相信 checkpoint 与 companion
   彼此相等。
4. 独立读取 checkpoint 顶层和 nested identity；两者均须字段存在、类型正确、互相一致并等于 expected identity。
5. 严格验证 nullable、order、Protocol、commit、context/binding、execution Git、train window、runtime/capacity、
   agent、seed 和 checkpoint schedule/identity。
6. 任一失败都必须先于 `run_real_episode` 或等价环境交互。

禁止从 nested 回填旧 checkpoint 缺失的顶层字段，禁止把 expected identity 设为 `None`，禁止因缺 formal binding
跳过 active gate，禁止采用 checkpoint 自报版本降低 active Protocol 要求。

## G14R18 验收

test-only 验收使用生产 metadata builder，经 torch save、annotate、read-back、strict selection/freeze、
`write_checkpoint_companions`、真实文件 loader、benchmark gate 和 `validate_checkpoint_provenance`。矩阵为
10 agents × 5 seeds × 3 capacities，共 150 个 compatible checkpoint；覆盖 candidate/latest。完整 envelope 正例
通过，要求的 schema、identity 和 per-cell 漂移负例均在 rollout 前拒绝；正负例环境 rollout 调用数均为 0。

最终被验收的冻结代码 commit 为 `834603a266bf06d30a070588a6e1f633eacf70e3`；最终验收记录由独立 commit
`3a88303` 发布。定向测试 `170 passed`、全仓 `1288 passed`，均 0 skipped；public preflight、150/150
训练入口解析、smoke、compile/import、diff-check、active resource size/SHA-256 全部通过。

## 科学与执行边界

Protocol 2.8→2.9 不改变科学配置、nullable 数值语义、agent/order、seed、预算、split/window、
lifecycle/exogenous、catalog/capacity、selection/statistics 或 holdout 合同。本轮没有执行完整正式训练、formal
performance、holdout、G14D 或 G15。

G14C v16 状态是 `G14C_V16_LAUNCH_AUTHORIZATION_DEFERRED`：未创建、未消耗、未执行。这不是 v16 运行失败；
不存在本轮 v16 run root、ledger、checkpoint 或 invalid-run denylist。G14C v15 及更早 invalid 记录保持不变。

## 审查元数据

- `reviewed_at`: `2026-09-06T15:11:27+08:00`
- `literature_cutoff`: `2026-09-06`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_formal_provenance_envelope_repair_20260906_g14r18_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `834603a266bf06d30a070588a6e1f633eacf70e3`
- `evidence_level`: `E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE`
