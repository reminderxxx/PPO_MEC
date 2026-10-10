# 条件弃权后模型加载与时效退化：事前有界审计计划

- `frozen_at`: 2026-10-10 Asia/Shanghai；本文件须在新反事实 `env.step()` 前提交并推送。A 仅诊断，不训练、不修改生产 agent/环境/奖励、不访问 formal/holdout、不改 B 工作树或主工作区；保留旧 `MIXED` 与本轮候选的双视角负结果。
- 新候选原件 `cscwd_event_aux_abstention_ab_20261010_v1`：`run_manifest.json` SHA `d9f256b6277c6b358b7eafd329f118f1f56884b67d94640bfcb8775d29fa880c`、`artifact_integrity.json` SHA `fa03b3fe9eed9af66f96e10e10d90db380b13a28893716071a395a8e4eb129cc`。旧 v4 对照 `cscwd_causal_prepared_state_visibility_matched_20261010_v1`：run manifest SHA `48718e48dc55e6958c03b676634dca53e84fd21be251115d4809380a1441c615`、integrity SHA `b66dfb946d2fef8c8e93b3ffacd7f332c1280635c7c635ef1f58f7eb8c4e2ac1`。先核对两者完整清单、checkpoint、config、split 和 selected/fixed96 原件身份，失败则停止。
- 全部 200 个 SA 同 `(view, split, seed, design_id)` 配对，`view=selected/update96` 分列，不把重复窗口或同 seed 跨 view 当独立样本。PPO 对照只用旧 v4 相同 episode/view 原件，并披露 action2 fallback 的能力与成本差异。报告 completion/on-time/failure、共同完成 coverage、elapsed、transfer、recompute 的各 seed/split 和总体变化；不以低 bytes 或单一 selected 改善晋级。
- 加载账本逐实际事件重建：`model_transfer_bytes = base + adapter`，按 resident before/after、victim、后续实际 node 使用，互斥归类首次必要加载、驱逐后重载、RSU 再次迁移加载、未被后续使用的准备及其余明确类别；分类和为所有事件 bytes，另与 episode `model_prepare_mb` 守恒。action0 即时完成服务产生的必要加载不因大小被叫作浪费。后续使用可用模拟真值仅作离线审计，不进策略。
- 梯度/优化日志只读核对 fast/slow auxiliary 的 target、权重、梯度与 PPO advantage/reward 的冲突事实；没有配对梯度证据时只报相关性或 `Unknown`，不擅删 loss。费用分解按 transfer/load/restore/recompute/compute/failure 检查只计一次；当前 `decision_step_index` 移动与人工 5s 接触预算单列模型限制。

## 反事实选择与硬预算

先对每个 episode 对齐两版行为 ledger，从初始状态起找**第一个**同公开状态、同前缀动作而当前动作不同的决策；不存在则不入选。来源 episode 依下面有序 strata 顺次选，每 stratum 按 `split=(regression,frozen_check)`、`seed`、`design_id` 升序取第一个未曾选过的 `(split,seed,design_id)`，不能看反事实结果再换点：

1. fixed96 on-time 下降，seed 17、29、61 各 1 个（3）。
2. fixed96 on-time 上升，seed 7 取 1 个（1）。
3. selected on-time 上升，seed 7、17、29 各 1 个（3）。
4. selected on-time 下降，seed 7、17 各 1 个（2）。
5. selected on-time 不变且 model bytes 增加，seed 29、43 各 1 个（2）。
6. fixed96 on-time 不变且 model bytes 增加，seed 61 取 1 个（1）。

若某 stratum 空，缺口不事后替换；最多 12 个 episode。入选后必须从完全相同的 config/instance/RNG/cache/clock/DAG prefix 重建首个分叉前状态，复核前缀两版动作、公开状态、clock 和本次动作均与原 ledger 一致，否则记录明确 `NA_IDENTITY`，不得改选。每状态最多三个**合法且不同**首动作：候选事实动作、旧策略当步动作、现有 action2 fallback（若合法；否则 action0 且先确认可服务）。三个分支之后都使用该行**预先固定的候选 selected 或 update96 checkpoint**、`raw_policy`、确定性决策，各自从自己的公开状态继续至 workflow 完成或原 episode 剩余 horizon（总 24 步）耗尽；不机械复制原后缀、不更换 checkpoint。每分支相同初始 snapshot/RNG；最多 36 分支、864 新 env.step，超限即停止，0 新训练和真实模型调用。保存完整状态摘要、首步与后缀逐事件费用、结局、source/checkpoint/hash 映射与 create-only manifest。

终局按“必要成本交换 / 可避免策略代价 / 确定实现缺陷 / 未知”逐项判定，既报告改善也报告恶化；不同首动作的后续状态不可当作单步局部因果或全局最优。只有定位到精确消费者、计费或梯度实现违反冻结 contract 才建议 bug 修复；否则只给一个最小可证伪机制候选及现有反例。正式论文、优秀 baseline 排名与时间物理真实性仍 `Unverifiable`。
