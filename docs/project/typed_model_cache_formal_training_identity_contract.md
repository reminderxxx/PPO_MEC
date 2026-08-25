# Formal Training Scientific Config and Execution Binding Contract

## 冻结身份

- `agent_training_scientific_config_contract_version=2.0.0`
- scientific config semantic SHA-256：`f83587cd13c126a0d8a6bdc26402e34ac1391bd6fc8ef504736458872d649bc8`
- `formal_training_execution_binding_version=1.0.0`
- `resolved_formal_execution_context_version=2.0.0`
- 首个消费者：`typed_model_cache_formal_protocol_version=1.6.0`
- Protocol v1.6 semantic SHA-256：`f2c9e729f126d9e87f56fcdccf13f2ecd018c28ca3102b8d02b2bbd6abca95c0`

本合同只修复 formal agent training config 的身份分层，不改变 10-agent matrix、5 seeds、256 episodes、
32 updates、checkpoint cadence 4、288/576/864 MB、数据、split/window、catalog、全部 agent 超参数、
dev selection、formal endpoint/support/statistics 或 holdout seal。

## Scientific config

`configs/experiment/typed_model_cache_formal_protocol_v1_6_20260825/agent_training_scientific_config.json`
是唯一 active scientific config。它只包含 10 个 learned agent 的稳定 identity、learning rate、适用的
entropy/value coefficient、SA-GHMAPPO `auxiliary_coef=0.06`、字段 applicability 与明确
`not_applicable` 语义。其 canonical projection 不含 Protocol、execution commit、run ID、绝对路径、
environment、output/checkpoint path、dev/formal/holdout 结果或身份。

配置使用 UTF-8、sorted-key compact JSON 计算 semantic SHA-256；NaN/Infinity、duplicate key、unknown
field、隐式默认、agent 缺失/重复/未知及 applicability 漂移均 fail-fast。Protocol v1.5
`training_budget.agent_configs` 与新 config 10/10 逐字段相等；本轮没有超参数变化。v1.0–v1.5 的
`agent_training_configs.json` 仅供历史审计，不能作为 v1.6 active 输入。

## Execution binding

Protocol semantic projection只冻结 scientific config hash 与 binding schema/rule，不包含某次 runtime
binding full hash。outer runner 在 Protocol hash、clean Git HEAD、environment/dependency 和完整 command
matrix 均确定后，create-only 生成：

`<durable_run_root>/formal_training_execution_binding.json`

该 artifact 绑定 active Protocol ID/version/semantic SHA、精确 observed execution commit、scientific
config SHA、agent matrix、training budget、resolved context contract、environment/dependency、split/window/
catalog/runtime、command matrix 与 portable resource identity。`binding_full_sha256` 是排除自身字段后的
一次 canonical hash；禁止迭代改 hash、接受冲突 binding 或用路径相同替代 content identity。

binding 不含 host path。内容相同的 scientific config 可迁移路径；同名异 hash 必须拒绝。binding full
hash进入 resolved context v2、phase/cell input identity、run identity、training summary、checkpoint metadata、
dev candidate/selection、freeze manifest、checkpoint provenance 与 formal benchmark checkpoint gate。

## Active training chain

Protocol v1.6 train command必须同时传入：

- `--agent_scientific_config_path`
- `--formal_training_execution_binding_path`
- `--formal_protocol_path`
- `--resolved_execution_context_path`

共享入口在 episode 0 前完成 scientific schema/hash、Protocol parity、binding/commit/environment/data/runtime/
command/context identity与实例化 agent `_checkpoint_config()` 全字段审计。任一缺失或漂移均终止；legacy
`--agent_config_path` 在 v1.6 明确拒绝。

`--formal_contract_preflight_only` 只允许 v1.6，用真实训练入口完成 contract resolution 和 agent
实例化审计后在 episode 0 前退出。其 artifact 必须标记 `formal=false`、`training=false`、
`performance_evidence=false`、`checkpoint_created=false`，不能进入 dev/formal。

## Invalid-run 与结论边界

G14C v6 `typed_model_cache_formal_20260825_135122_g14c_v6` 永久登记为
`invalid_during_first_training_cell_before_episode_zero`：episode/interaction/update/checkpoint均为0，禁止
retry、resume、finalize、salvage与checkpoint reuse。G14C v1–v5 的既有禁止边界继续有效。

Readiness v8只授权未来独立任务从最终 pushed Commit A7 的 clean worktree新建 G14C v7。它不表示正式
training/checkpoint/performance、formal、holdout、G14D、G15 或 paper-ready 已完成；ordinary runner仍无
holdout capability，holdout保持 sealed/unopened。
