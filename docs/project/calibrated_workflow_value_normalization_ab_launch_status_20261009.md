# Calibrated workflow critic PopArt A/B 启动状态（2026-10-09）

## 审查身份

- `reviewed_at`: `2026-10-09T12:11:39+08:00`
- `literature_cutoff`: `2026-10-09`；本轮未新增文献检索
- `target_venue`: `CSCWD 2027`（同时按项目 TMC 审查规范约束证据边界）
- `artifact_run_id`: `calibrated_workflow_value_normalization_ab_launch_attempt_20261009_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `5ee9f1e8071ad8d9e1992e91564ea32801e3d7a7`
- `evidence_level`: `E1_PREFLIGHT_AND_PRE_RUN_FAILURE_RECEIPT`
- `verdict`: `UNVERIFIABLE_NO_SCIENTIFIC_A_B_RESULT`

## 结果

实现与 create-only preflight 均通过，但唯一授权的后台科学启动在 runner 创建 run root 前结束。即时快照未发现所报告 PID；预期 run root、`run_status.json`、completion/failure receipt 均不存在，重定向日志存在但为 0 bytes。由于 runner 在进入训练前先创建 run root 并写 `run_status.json`，本次可确认 scientific environment steps / updates / optimizer steps / checkpoints / evaluation rows 均为 0。

严格遵守 one-launch/no-retry 约束，本轮停止，不做第二次启动、foreground 替代、resume 或补跑。OS 级退出原因无法从空日志恢复，必须标为 unavailable，而不能猜测成算法异常。

## 已验证范围

- scientific commit 已推送到 `origin/codex/service-reward-popart-ab`；
- frozen design SHA-256=`9630d97d56e8192b6278694980132cc942a67cc87794eb8ddc4b67d97f3cc606`；
- authorization SHA-256=`212cdcf0f880963780fe8ad2cf0fd523f90d5f0574c267f110f63a9fed01db3a`；
- preflight status=`pass`，30 cells / 43,200 steps / 720 updates / 5,760 optimizer steps 的预期预算一致；
- 六个 method×arm acceptance batch 各 60 transitions、8 optimizer steps、无 skipped update、记录值有限；
- 36 个 source interval 身份字段齐全且原始 time interval 两两不重叠；
- 单元/相邻测试 `295 passed`，toy smoke 通过；
- checkpoint、真实数据、模型权重均未提交或上传；formal/holdout/download/model-generate 均为 0。

Preflight 只证明实现和启动前合同可执行，不是 A/B 性能或因果证据。失败回执位于 `artifacts/analysis/calibrated_workflow_value_normalization_ab_launch_attempt_20261009_v1/`；preflight 位于 `artifacts/analysis/calibrated_workflow_value_normalization_ab_preflight_20261009_v1/`。

## 对诊断结论的影响

没有新科学结果改变先前诊断。当前最值得补足的环节仍是 critic value-target/output 尺度：既有 service-reward 九个 learned cells 的 explained variance 近零，固定开发样本 value:policy gradient ratio 为 `539–7,907`，estimated global clip scale 为 `.00116–.00147`；termination/GAE/reset-observation 未发现当前实现错误，auxiliary 冲突也不一致。这只支持 PopArt 为最小可检验候选，不支持它有效。

若未来获得新的、显式的单次执行授权，唯一变量仍应是三方法共享 critic 的 PopArt target/output normalization，预算、instance order、reward、actor、auxiliary 与 dev-only checkpoint selection 均保持冻结。下列任一结果否定该候选：

1. paired median value-to-policy gradient ratio 不下降或 explained variance 不上升；
2. current-model-missing 状态下 action 4 mean probability / raw argmax 不同时下降，或 service failure / consecutive no-progress 增加；
3. 任一方法的 completion 或 on-time completion 下降，或所有 completion/failure 指标都无严格改善。

在新授权前不得自动启动，也不得转向 auxiliary-target 消融、更多 seed、延长预算或用最终评价选择方案。
