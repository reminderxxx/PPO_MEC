# Training signal current-bundle readiness 日志纠错（2026-10-10）

## 结论

已确认并最小修复一个**训练信号日志生产者错误**：
`scripts/run_calibrated_workflow_value_normalization_ab.py::_training_signal_row` 原来从不存在的
`semantic_state.current_rsu_id` 查当前 RSU；v4 公共状态实际通过 primary vehicle 的 `associated_rsu_id` 表示当前位置。
修复后先按 `primary_vehicle_id` 解析 vehicle，找不到时与 agent 合同一致回退首 vehicle，再计算 `bundle_ready`。

这不是算法、reward、计费或环境修复。A 的对象级加载账、4,203 记录动作重放和 26 条有界分支已确认 model bytes、clock 与
fallback 计费守恒；新增传输主要是后续实际使用的跨 RSU base/model 放置，不能删减为“浪费”。

## 影响范围

- 候选 run 的 `training_signal_rows.csv`：28,800/28,800 行旧 `current_bundle_ready=False`，该字段无效。
- v4 四 learned 方法原件：115,200/115,200 行旧字段同样无效。
- agent event abstention 使用独立、正确的 primary-vehicle 路径；optimizer 记录存在 79,796 个 supervised repeated exposures，
  因此不能从错误日志反推训练全部 abstain。
- event-abstention analyzer/finalizer 不消费该 readiness 字段；评价、checkpoint selection、loss、optimizer receipt 和
  `MIXED_STOPPED` verdict 未见受影响。strong-baseline analyzer 只检查训练信号行数；budget-extension 只将旧行用于前缀身份比较。
- `value_normalization` 的 behavior gate 会在未来执行时消费该字段，因此生产者必须修复；仓库本地没有已完成的 PopArt scientific
  artifact 可据此重判。

旧 scientific artifact、checkpoint 与 CSV 均不覆盖。新增 create-only correction sidecar 只标记历史字段
`INVALID_DO_NOT_USE_FOR_READINESS_STRATIFICATION`，不伪造逐行更正值；历史值若未来确需恢复，必须另立确定性 replay 合同。

## 验收与边界

- ready 正例：primary vehicle 不在 vehicles 首位，关联 ready RSU，日志必须为 `True`。
- missing 负例：同一 primary vehicle 关联 missing RSU，日志必须为 `False`。
- 旧 artifact 行数/hash 保持；sidecar 新增 training/evaluation/reselection=`0/0/false`。
- 不因本纠错重训、重评、重选 checkpoint 或改变 `MIXED_STOPPED`；fast/slow 与 PPO 的真实 per-head 梯度冲突仍为 Unknown。

A 独立报告 commit=`7734d5807feaa66ac3f40e89ca8ff86f911f0783`、tree=
`210c1727c7947ba80dce6066fc158f969b6496f5`；报告/摘要/两份机器 manifest SHA-256 分别为
`3354ee27…`、`45498f30…`、`859abe71…`、`ee49c70b…`。本纠错属于 logging correction，不是算法创新或性能证据。
