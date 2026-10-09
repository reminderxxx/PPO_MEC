# CSCWD 强基线预测权限预检阻断

## 审查身份

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09；未新增文献检索
- `target_venue`: CSCWD 2027（拟投）
- `artifact_run_id`: `none`；科学比较在创建 run root 前停止
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_review`: 接线基线 `d18e7bfc1c81dc2fd99166e61489d66e0c4c0b65`；本阻断记录最终提交见 Git
- `evidence_level`: `E1_SOURCE_AND_MANIFEST_CONTRACT_AUDITED / NO_NEW_SCIENTIFIC_RESULT`
- `verdict`: `BLOCKED_BEFORE_FREEZE_AND_LAUNCH / ACTUAL_FUTURE_MOBILITY_EXPOSED_AS_PREDICTION`

用户已授权一次完整、有界、仅作 development 的四学习方法加两规则方法比较。执行前按授权要求核对所有方法的公共预测字段是否含真实未来执行信息。本报告是该预检结论；没有冻结本次科学执行配置、没有创建新 run root、训练、选模、评价或打开旧 holdout。B 的 PopArt A/B 按 `979420a3` 记录为 `FALSIFIED_OR_NOT_PROMOTED`；拟比较仍应使用 raw critic 与 `original_reward_v1` 单一参考条件，不能借此换臂择优。

## 可复核的第一顺位阻断

`configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json`（SHA-256
`b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc`）的 **36/36** 实例都有用于
环境实际迁移的 `rsu_sequence`，**0/36** 实例提供独立的 `predicted_rsu_sequence`。冻结实例生产端
`scripts/freeze_calibrated_continuous_workflow_pilot.py` 从同一人工 block 规则生成 `rsu_sequence`，只保存
`prediction_profile` 的 confidence/quality/uncertainty 标签，没有另行生成轨迹预测。

`src/envs/core/calibrated_continuous_workflow_env.py::_predicted_sequence` 的精确回退为
`self.instance.get("predicted_rsu_sequence", self.instance["rsu_sequence"])`，随后从当前索引读取下一段序列。
`_semantic_state` 再把这段值写入公共 `predictions.next_rsu_sequence`、`predicted_next_rsu_by_vehicle`、
`predicted_first_handoff_rsu_by_vehicle`；`dwell_time` 和 `calibrated_context.contact_budget_seconds`
也由它计算。`future_load` 则由当前 cache resident/capacity 得到，本身不是这项未来轨迹泄漏。
因此在当前 36 实例上，标作“预测”的 handoff 方向与接触预算实际上引用了环境将要执行的未来 RSU 序列。

SA、PPO、controller-level MAPPO 和 DT 都经公共 `info.semantic_state` 获得该快照；DT 显式使用其 sequence、
next/target、dwell 与边界特征，Popularity 也直接读取 next/target。共享同一个不合格输入不构成合法的信息公平。
two-step 另有 exact-transition clone 和字典序目标，始终只能列作不同能力的 model-based comparator。
此处没有读取任何方法的效果结果，也不据此推断哪一种策略获益更多。

## 停止和恢复条件

本轮授权明确要求“所有方法无真实未来执行信息泄漏”，且不允许改 actor、reward、environment 或模型结构来消除问题。
在旧 36-instance manifest 上继续训练会违反该合同；仅把原始序列改名、复制到
`predicted_rsu_sequence`、调 confidence 或给两种 reward 各跑一次，都不能修复来源独立性。新增 fail-closed
preflight：缺少独立预测序列时，runner 报 `PREDICTION_FUTURE_LEAK_BLOCKER`，阻止结果目录创建与训练。

恢复需另立科学合同任务：用每个决策时点以前可见的轨迹/状态产生预测，保存 predictor/source hash、预测生成
时点和可复核的 prefix-only 证据；在生产端生成新 manifest，消费端逐实例逐时点检查实际后缀不进入预测输入；
训练/dev/evaluation 共用同一信息权限并量化预测误差。之后重新审查原始窗口/工作流祖先、选模和预算，冻结
完整 commit/哈希与唯一 create-only run root，再决定一次性开发比较。本轮不自动改环境或生成新数据，也不把
已消费 36 实例变成新确认性数据。

预检命令与结果：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/run_calibrated_workflow_strong_baselines.py --preflight
```

预期结果为非零退出并包含 `PREDICTION_FUTURE_LEAK_BLOCKER: 36/36`；该失败是科学权限门生效，不是可重试的
训练/launcher 故障。完整 28,800-step 比较、分析表和启动 ACK 均未产生。
