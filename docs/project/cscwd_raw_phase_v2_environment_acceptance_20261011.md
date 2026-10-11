# CSCWD raw phase v2 环境验收（2026-10-11）

## 身份与结论边界

- `reviewed_at`: `2026-10-11`；`literature_cutoff`: `not_applicable_no_literature_review`；`target_venue`: `IEEE TMC`；`artifact_run_id`: `cscwd_raw_phase_v2_preflight_20261011_v3`；`policy_version`: `tmc_review_policy_v3_20260621`；`git_commit`: `366c951`（环境实现，预检脚本哈希修正随后提交）；`evidence_level`: `engineering_local_verified / paper E0_UNAVAILABLE`。
- 本报告是已授权的环境实现与两条原始开发轨迹预检，不评价算法排名或论文贡献。正式 checkpoint、formal/holdout/support 原始结果未生成；paper-ready verdict 为 `Unverifiable`。没有读取 formal/hidden 结果，也没有新训练。
- 物理、预算、数据范围以事前冻结的 `cscwd_raw_phase_v2_frozen_contract_20261011.md` 和配置 SHA-256 `2a0cee4b4b2ba512f8015296ee1946698dd51066d1ae5744e22b4d428d176343` 为准；v1 环境源码 SHA-256 `30666d7c874717d8464a64702c811116e96b931a5db6b57287c6c11d2dcc6575` 未改。

## 验收原件与检查

| 本地 create-only 原件 | SHA-256 | 范围 |
|---|---|---|
| `artifacts/analysis/cscwd_raw_phase_v2_preflight_20261011_v3/preflight.json` | `9b26892522a99c5e50189925c8c08affae89ade55fe8512f8f6c0e9cf09245ed` | 两个原始来源各一次真实 `reset/step`；逐阶段、时钟与字节账本 |
| `artifacts/analysis/cscwd_raw_phase_v2_preflight_20261011_v3/public_fixture.json` | `1e7da4d92c5e9387438b2bd33d04d6c1b314cbe4dee4dcd03a72c013a65cd8c2` | US101 `dev_01` 首步仅公开 `_info()` 三元组，内嵌 hash 复算一致 |
| `src/envs/core/raw_ngsim_phase_v2_env.py` | `5a5b888f37a3ec8922d74e3b886d325ae96ee266852d338deb37bf0f0b67d683` | 独立 opt-in 环境，不修改旧 v1 |

来源 manifest SHA-256 `34eee3a8d16f2dfe89f3a8d8dcd27f20159c0488e2f1a9ebc77be3aac747abde`。原始 CSV SHA-256 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`。两个窗口各 118.8 s，均属于已暴露 development 来源。最终预检新增 2 episode/2 real step/0 preview/0 training。早期 v2 fixture 的内嵌哈希取了未保存的完整 `_info()`，B 独立验收拒绝；v2 root 留作失败见证，v3 已修正为保存字段自身的哈希。

- Lankershim 首步公开合法 action 3 完成节点，`input_transfer_bytes=192757`，`step_cost_seconds=1.55060428`，无迁移。
- US101 首步公开合法 action 4 完成节点并交付目标，`input/model/state` 分别为 `192757/9641944/140480` bytes，`step_cost_seconds=1.961859672`；七个实际 phase 事件的耗时和总成本一致。此例只证明新模拟合同中正向链可执行，不代表算法或真实网络效果。
- 合成负向验收覆盖接触内输入未完、模型与状态部分 backhaul 超时、节点跨源接触、目标离开或预测错误、trace end 截断、后缀变更下公开状态不变、单节点不重复完成、cache/state 不凭部分传输提交。`python -m pytest -q tests/test_raw_ngsim_phase_v2_env.py tests/test_raw_ngsim_event_time_env.py tests/test_cscwd_new_development_reachability.py` 为 `18 passed`。

## 当前开放项

- B 线尚需在上述最终公开 fixture 上独立验收 estimator、immediate/two-step、SA 消费路径和 checkpoint profile 拒绝；该门未通过前不启动匹配 pilot。v2 禁止 privileged exact decision clone。
- 相位事件的目标抵达查找基于 100 ms 原始采样及相邻段插值；模拟 RSU 几何、速率与零共享队列不是实测。两个开发窗口和预先暴露的 `dev_01` 不能作为独立确认或统计排名。
- 下一步仅按冻结的 60-cell/2,880-step/10,000-preview 上界执行矩阵并保留全部负结果；若规则消费失败，记录阻断，不用调整窗口、任务或条件补齐。
