# 当前服务可行性辅助标签：action0/2/4 有界反事实门禁

## 身份与证据边界

- `reviewed_at`: 2026-10-10 11:16 Asia/Shanghai；`literature_cutoff`: 2026-09-28（本轮不做文献 novelty 判断）；`target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: 科学来源 `cscwd_causal_prepared_state_visibility_matched_20261010_v1`；本门禁 `cscwd_service_feasible_action_branches_20261010_v1`。科学 Git commit `f46ec72b15f534ac44768a83ef6316c1cfcb6b58`；事前规则 Git commit `eefc4a2`；来源实现 commit `709746bc1f1ea3037f27497bdb51cf6f45c8963c`。
- `evidence_level`: 科学来源及本次有界 development 分支为 `E2_ARTIFACT_AUDITED`，不构成正式 comparison 的 `E3_REPRODUCED`。formal/holdout/support 未读、未运行；论文贡献或 TMC-ready 仍为 `Unverifiable`。
- 来源科学 root：`/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1/`。本门禁 root：`/Users/howen/.codex/worktrees/cscwd-window-identity/PPO_MEC/artifacts/analysis/cscwd_service_feasible_action_branches_20261010_v1/`。科学 terminal PASS、completion complete、119/119 SHA/size 匹配。旧事件链 7 个输出 SHA 匹配；没有写入 B、main、旧 artifact 或 checkpoint。

## 事前样本与执行合同

冻结计划 `cscwd_service_feasible_action_branch_plan_20261010.md` 在任何分支前提交、推送。失败来源**完整复用**此前有界探测的 24 行：selected 14、update96 10；不按分支结果重新选点。其中 22 行原 `event_target=1`，2 行为原标签 0。正例按冻结排序取 selected 的前 6 行 current-ready、原标签 1、原 action4 服务与 state commit 成功，均来自 `frozen_check_00` 的 seed 7/17/29、step 2/3；这种集中性明确限制对正例覆盖的外推。

30 行来源逐条有 `source_to_state_rows.csv` 映射；公开状态不等于内部状态。完整 config/instance/RNG/cache/LRU、DAG prefix、prepared state、clock/metrics、mask schema、observation 和原 `info` 规范化后为 **9 个完整环境状态**；再加入各自冻结 checkpoint SHA 后是 **30 个策略分支状态**。`state_snapshots/` 保存 9 份完整状态原件，`snapshot_export_manifest.json` 对其逐一给出 SHA；原 30 行及 checkpoint 身份保留，不把不同 seed/view 当独立窗口。

每状态从同一 `env.clone()` 对称执行合法 action0/2/4，三者在本次 30 个状态全部合法，故 90/90 分支、0 个 `NA_MASKED`。之后使用该行的**同一冻结 SA checkpoint**、`raw_policy`、确定性 `agent.act`，每一步只读该分支当时的公开 observation/info，至完成或原 24 步上限；没有机械重放原动作后缀。RNG、deadline、资源和前缀从 snapshot 保留，没有手工改 cache 或 action mask。实际 463 个额外环境步、373 次后缀 policy forward，低于预冻结 2,160/2,160 上限；90 个首动作不调用 policy。原 action4 的首步结果与科学 ledger 同行核对通过。全部 90 个分支最后完成，不能只用完成率掩盖按期、失败与成本差异。

## 门禁结果：`MIXED`，不得据此启动条件训练

**一步当前可服务性成立。** 24/24 失败来源上，action4 当前服务失败、state 未提交；action0 与 action2 均当前服务成功。6/6 正例的 action4 当前服务成功、state commit 成功，候选公式保持原正标签。因此没有“action4 在当前 missing 状态可服务”或“0/2 都不可服务”的硬矛盾，语义子门禁与正例子门禁通过。

**多步收益不能被同一规则确证。** 在 22 个原正标签、current-missing 的受影响 checkpoint 分支状态中，11 个的 action4 在预冻结无权重 Pareto 指标上仍非支配；其中 3 个 checkpoint 分支（对应 2 个完整环境状态）出现 action4 的终局服务排序严格优于 action0 和 action2。完整 Pareto 集合与互斥理由在 `gate_report.json`，逐状态结果在 `branch_summary_rows.csv`，全部后缀动作和 reward/bytes/recompute 分项在 `branch_trajectory_rows.csv`。这些是重复 development 条件下的描述性反例，不做 22 个独立样本的显著性检验。

最清楚的反例是 `regression_00`、seed 17、step 2（selected 与 update96 两个 checkpoint 身份复现同一完整环境状态，不能算两次独立窗口）：起点剩余 deadline `58.058 s`。action4 首步服务失败，但分支后续按期完成，剩余耗时 `29.257 s`、重算 `4.316 s`、总传输 `0.575 MB`；action0/2 首步均成功，却都逾期，剩余耗时 `68.670/74.725 s`、重算各 `43.785 s`、总传输约 `155.178/0.948 MB`。另一个反例 `frozen_check_00`、seed 29、step 5：三个分支最后均逾期，但 action4 的后缀服务失败 1 次，action0/2 各 2 次，耗时 `70.749/73.886/79.556 s`。所有数值均从同一 snapshot 的增量计算，不能把首步成功当成完整策略收益。

模拟器按决策步推进 mobility；失败 action4 仍可能移动到新的 RSU 并改目标 cache，后续冻结策略因此沿不同状态序列决策。这是已实现 contract 下的真实多步反例，并非已确认代码 bug；也不证明原辅助标签优于候选或可外推到真实车联网。

## 判定与交接

按事前规则，语义和正例均通过，但存在预定义的 action4 严格更好服务结局及非支配多步反例，**总体 `MIXED`**。这轮证据只支持“当前服务可行性标签具有局部语义合理性”，不能支持“把全部 current-missing 的 event hard/soft target 置零必然改善多步结果”，更不能证明辅助标签是 SA 与优秀 baseline 差距的唯一原因。B 的唯一 target-only 训练是条件授权；本门禁不是 `PASS`，因此不得依据它启动训练。若后续研究重新定义候选，须另立事前计划，不可在这些已消费 development 反例上临时调权重、换样本或改门槛。
