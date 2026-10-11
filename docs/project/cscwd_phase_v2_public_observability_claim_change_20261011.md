# CSCWD A 论文线：phase-v2 公共可观测性主张与证据变更说明（2026-10-11）

本文件只提供可合并的 claim boundary，不编辑主论文稿。

## 可新增的安全表述

- 三个已暴露 US101 development 起点中，第二次 action4 面向新的预测目标 `rsu_2`；在固定 continuation 下，action3
  保持相同完成/按期/失败结果并减少传输和时间。这是 bounded post-hoc local dominance，不是在线规则。
- 当前 phase-v2 public-causal auxiliary 在这些起点给出 `prepare, supervision_weight=1`。raw predictor 同时明确标记
  `confidence_calibrated=false`；现有 actor、estimator 和 public rules 均未消费该校准 provenance。
- 合成同前缀双后缀见证证明：完全相同的公共 observation/mask/estimate/label 可对应 action4 成功或失败，以及相反的声明
  局部动作偏好。因此现有公开状态不足以逐状态确定“可避免准备”。
- 唯一后续学习设计候选是：未校准或未知 delivery confidence 对正 prepare auxiliary `abstain`。它尚未实现、训练或证明有效。

## 必须保留的限制

- 不得把三处事后节省写成可部署的“成功迁移后禁止 action4”规则；第二次 action4 的目标已变化。
- 不得把 `prediction_uncertainty=0.5` 称为经校准概率、contact lower bound 或 target dwell bound。
- 不得声称 auxiliary 是 SA 服务退化的唯一或充分根因；历史 optimizer/minibatch 因果链仍不可恢复。
- 不得称 public two-step 具有 phase-v2 的真实二步 mobility 能力；其 projection 明确 fail closed 到 immediate。
- 不得把候选写成算法收益、SA 优势、formal/holdout 或 paper-ready 证据。旧 `MIXED`、负结果和 checkpoint 全部保留。

## 对贡献叙事的实际变化

论文方法段可将问题从“action4 重复”收缩为“未校准 target-delivery 预测下的 positive auxiliary 过度确信”，并把 proposed
mechanism 表述为 epistemic abstention，而不是动作 guard。论文结果段目前只能报告诊断与否证条件；在 source-disjoint、
预注册 A/B 通过前，不得报告 performance contribution。

证据入口：`cscwd_phase_v2_public_observability_audit_20261011.md` 与同名 JSON。
