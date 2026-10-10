# MIXED 反例时间合同与 event 辅助监督条件弃权独立门禁

## 审查身份与范围

- `reviewed_at`: 2026-10-10 Asia/Shanghai；`literature_cutoff`: 2026-09-28（本轮未评价文献 novelty）；`target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: 来源科学 run `cscwd_causal_prepared_state_visibility_matched_20261010_v1`；既有分支 `cscwd_service_feasible_action_branches_20261010_v1`；本轮只读审计 `cscwd_conditional_abstention_audit_20261010_v2`。来源科学 commit `f46ec72b15f534ac44768a83ef6316c1cfcb6b58`，此前分支门禁 commit `c8aecf19b39c83debf57d20f44a90f6ebb650e68`，B 候选实现 commit `46a68f11c289ccc304b88cdfed01c34ba5f61c3d`、tree `a197c7d7c5fd80619747f34f9c4416babb023d8e`。本报告 Git commit/tree 由同目录 `audit_manifest.json` 固定，以免文件自引用。
- `evidence_level`: 既有 development 分支与本轮身份/梯度检验为 `E2_ARTIFACT_AUDITED` 范围内的局部证据；正式 paper-ready 比较仍为 `Unverifiable`。未读 formal/holdout/support，没有新训练、调参、checkpoint 选择或论文主表更新。
- 本轮审查是独立门禁，判定候选实现是否遵守事前协议；**不改写此前 hard-zero target 候选的 `MIXED` 结论**。门禁 `PASS` 只表示协议、身份和代码层面可进入 B 独立决定的科学执行阶段，不表示性能提升或论文贡献已成立。

## 证据原件与时间合同

已校验旧分支 `analysis_manifest.json` 中 96 个文件 SHA、`snapshot_export_manifest.json` 中 9 个完整环境状态 SHA。30 行来源映射对应 90 个合法 action0/2/4 分支；463 条轨迹的 `step_index`、RSU、`clock_seconds_after`、末端耗时与 deadline 标记逐行一致。**额外重放 0 步**。逐步收据见 `artifacts/analysis/cscwd_conditional_abstention_audit_20261010_v2/time_contract_receipt.json`。

冻结 v4 环境采用 `mobility_progression=decision_step_index`：每次 `env.step()` 将 `step_index` 加一，失败也推进一个 RSU 序列位置。接触预算按将来决策位置数乘 `decision_step_seconds=5.0` 算；deadline 则按动作的 `step_cost` 累加到 `clock_seconds`。两者未绑定为同一物理秒轴。90 个分支的首步成本范围为 action0 `2.538–66.937 s`、action2 `9.544–12.557 s`、action4 `2.000–25.435 s`，却都推进一个移动位置。环境的 `service_operation_seconds` 是报告指标，传输、load、restore 在 `step_cost` 中只计一次；463 条原件未见时钟重复入账。

`rsu_sequence` 是脚本根据 NGSIM 来源窗口的 handoff pressure 生成的 synthetic block 序列；既有 config 明示 `source_type=artificial_time_scale_applied_to_trace_derived_handoff_pressure`，并把 decision time scale、link、deadline 与轨迹/workflow 配对列为 artificial。两个反例的 `source_interval` 各有 24 帧、原始首尾时间相隔 2300 ms，但决策预算每格是人工 5 s。因此这是已冻结**模拟器合同自洽、物理时间外推受限**的情况，未发现本轮可确证的旧合同违反或 clock 双重计费。任何车辆真实逐帧 handoff、接触时间或线上 deadline 优势的 claim 仍无证据。

## 两个事前指定的完整环境反例

1. `selected|regression|sa_ghmappo|17|regression_00|2`：起点 current RSU `rsu_0` 缺 bundle，target `rsu_1` 已 ready；剩余 deadline `58.058 s`。action4 首步花 `2.000 s`、当前服务失败、节点未进展；目标 admission 是 `noop_all_resident`，传输 0，prepared state 只到 `staged_pending_current_node_completion`，**没有 commit**。action0 首步花 `3.489 s`，装入当前 bundle 并完成节点；action2 花 `9.544 s`，fallback 并完成节点。三者下一决策均在 `rsu_1`。冻结策略后缀分别为 action4 分支 `[4,2,4]`、action0/2 分支 `[2,4]`：前者随后按期完成、总剩余 `29.257 s`、recompute `4.316 s`；后两者均逾期、总剩余 `68.670/74.725 s`、recompute 各 `43.785 s`。
2. `selected|frozen_check|sa_ghmappo|29|frozen_check_00|5`：起点 current `rsu_2` 缺 bundle，target `rsu_0` 已 ready；剩余 deadline `13.191 s`。action4 首步仍是 `2.000 s` 失败等待、无节点进展、目标 `noop_all_resident` 且无状态 commit；action0/2 首步服务成功，耗时 `2.924/10.560 s`。三者下一决策均在 `rsu_0`。action4 后缀 `[4,2]`，总失败 1 次、耗时 `70.749 s`、recompute `53.948 s`；action0/2 后缀 `[4,4,4]`，各失败 2 次、耗时 `73.886/79.556 s`、recompute 各 `61.141 s`；三者最终都逾期。

两处 action4 首步没有产生有效 state migration；`noop_all_resident` 可触碰目标 cache 的 LRU 元数据，但不改变 resident 集合。较好后缀来自低成本失败等待后模拟 RSU 前进、DAG 节点保持待处理、随后策略状态和 recompute 路径改变，不能写成“首步准备传输带来收益”或全局最优。相同环境状态的不同 checkpoint/view 不是独立窗口样本。

## 候选实现审查：`PASS`（仅限进入独立科学执行的合同门禁）

在 B clean commit `46a68f11c289ccc304b88cdfed01c34ba5f61c3d` 上执行独立脚本，逐项验证旧 96 文件及 9 份完整快照 SHA，并复用原 30 个状态映射；结果写入 `artifacts/analysis/cscwd_conditional_abstention_audit_20261010_v2/abstention_receipt.json`。本审计不训练、不新造 env step、不读 formal/holdout。

| 项目 | 独立结果 |
|---|---|
| 单变量/网络 | 旧版与候选以相同 seed 初始化，全部网络张量相同；共同环境源码 SHA `c6ded6ce84230519160a40f29e276b5823eb204a35e31c35f914d1c6bfee3a59` 未变；旧 hard-zero gate 保持关闭。 |
| 当前服务已就绪 | 6/6 状态的 slow、fast、event 辅助梯度及总辅助 loss 与旧版逐张量相同。 |
| 当前 bundle 缺失 | 24/24 状态的 event 辅助梯度为零，slow/fast 梯度与旧版相同；单独启用 event CE、单独启用 temporal margin 时均为零。目标 hard/soft 数值并未改标。 |
| batch 分母与优化信号 | 30 个状态均满足原 confidence eligibility，6 受监督、24 弃权；本**审计样本**比例 `6/30=0.20`，不是训练分布估计。混合 batch 的 ready 样本 event 梯度恰为单样本一半，全 missing batch 有限且 event 梯度为零；执行动作 PPO clipped surrogate 的 event 梯度非零。源码 diff 中 PPO ratio、entropy、value loss 与原执行动作 logprob 路径未改变。 |
| 运行时及基线 | 30/30 raw-policy 动作、概率、logprob、entropy、mask、value 一致；PPO/MAPPO/DT 的新开关均为 false、同 seed 初始化网络及 raw 概率相同。开关只由 SA 消费。 |
| 信息与 checkpoint | 篡改 future prediction 后弃权权重不变；旧 v3 profile 被拒绝；旧版与候选 checkpoint 互相加载被语义检查拒绝、各自自载通过。 |
| B 执行门禁 | 冻结协议仍 `execution_authorized=false`；runner 对审计 commit/tree、报告及三份收据逐文件 SHA 和语义字段 fail-closed 校验。B 报告其相关 45 tests 与 smoke 已通过；本 A 审计不把这些 B 测试冒充独立执行。 |

**判定：`PASS`。** 这表示候选符合本轮事前冻结的接口、身份、选择性梯度和时间合同门槛。只有 B 独立核验本报告与 manifest 哈希、在自己的协议中另行绑定 full A commit/tree 并通过 fail-closed preflight，才可按预冻结预算考虑运行；本 A 报告不直接启动训练。旧 hard-zero target 的 `MIXED` 和两个完整环境反例保留。实际训练效果、优秀 baseline 比较、公平性与正式贡献仍未验证。顶刊完整 scorecard 为 `N/S`，因为本轮没有候选科学结果或 formal/holdout/support 原件。

## Claim 边界

- 安全表述：在冻结模拟器内，hard-zero target 的一步当前服务语义成立，但多步 action4 存在既有反例；当前候选只尝试在 current-missing 时停止 event CE 与 temporal margin 的辅助监督，保留真实执行动作的 PPO 信号。
- 禁止表述：hard-zero 候选已经通过、action4 首步完成了 state migration、条件弃权已经优于 PPO/MAPPO/DT 或其它优秀 baseline、时间尺度等价真实 NGSIM 帧、目前达到 IEEE TMC paper-ready。
- 长期风险：decision index 与 modeled seconds 分离，正式论证物理 handoff/contact/deadline 时需要额外时间一致性研究；本轮不修改环境，因为会改变冻结科学身份。
