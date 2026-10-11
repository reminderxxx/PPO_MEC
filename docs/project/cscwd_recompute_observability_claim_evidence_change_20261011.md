# A 论文线主张与证据变更说明：recompute observability

本说明独立于主论文稿，不并发修改 manuscript。

新增可确认事实：v4 public semantic state 过去缺少 `last_execution_rsu/current_handoff`，使 46 个本可由已发生历史和公开 DAG
计算的 recompute 状态被错误标为 unknown；最小接口纠错后，同一冻结 v668 资格 probe 出现 46 个 base prepare，且
calibration-abstention 候选在真实 SA consumer 上对这 46 行全部发生 prepare→abstain。

不变边界：这不是算法创新或性能收益；不支持 SA 稳定领先、paper-ready、formal/holdout 或独立泛化。两步 public rule 仍是
model-based、lexicographic objective 的单独能力参考；privileged exact two-step 只保留 phase-v1 capability reference。

下一项唯一允许的因果证据是一次 4 arms × 5 seeds 匹配 development A/B；唯一变量为 SA 的
`mechanism_aux_positive_prepare_calibration_abstention_enabled`。若训练实际 intervention 为 0，或任一 selected/fixed-update24
视图 completion/on-time 降低、failure 增加，则候选失败；不得按最终检查结果换 checkpoint、追加 seed 或改奖励。

## 运行后追加（2026-10-11）

该 A/B 已完整结束并判定 `FAIL`。候选在 selected/fixed24 reward 为 `9.780383/9.424639`，control 为
`9.044652/9.364771`，但 failure rate 同时由 `0.125/0.175` 增至 `0.225/0.200`。新增失败全部位于 v1
mechanism-off；v2 分层 failure 增量为 0。reward 正增来自 v2 deadline gain 覆盖 v1 failure penalty，不能更新为算法优势。
详见 `cscwd_recompute_observability_calibration_abstention_results_20261011.md`。主论文稿未修改。
