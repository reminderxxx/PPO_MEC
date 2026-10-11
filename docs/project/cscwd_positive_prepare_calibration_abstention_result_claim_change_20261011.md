# A 论文线：positive-prepare calibration abstention 结果主张变更

本文件只向 A 论文线提供主张边界，不编辑主论文稿。

## 可新增事实

- 一次性 development run 在 commit `56d25519…9b` 完成既定 23,040 environment steps 与 3,840 optimizer steps；
  run/analysis 原件完整性核验通过，未读 formal/holdout。
- 四 learned arms 的全部训练 label 都是 `abstain, weight=0`；SA control/candidate 的 10 对网络张量和 80 对评价指标完全相同。
- 新候选裁决为 `UNINFORMATIVE_ZERO_TREATMENT_EXPOSURE`，原 analyzer 的 `MIXED` 只保留为预注册结果，不代表候选效果。
- 后续 96-step Lankershim 非训练探针在 96/96 状态观察到 `action4_not_legal`，predicted target 均为空；这是当前训练域
  缺少机制暴露的有界见证。

## 必须删除或禁止的主张

- 不得写 calibration abstention 改善、伤害或等效于 control；不得把两臂相同写成消融阴性结果。
- 不得把 SA selected reward 略高于 MAPPO 写成稳定领先；fixed-update24 反向，且 candidate 对该差异无贡献。
- 不得把 96-step 未观察到 prepare 外推为整个 Lankershim/NGSIM 域不存在机会。
- 不得用本 run 支持三头结构、sharing×migration、真实 RSU 净收益、独立泛化、显著性或 paper-ready。

## 论文衔接

安全表述是：一次匹配开发实验因训练源零 treatment exposure 而不可解释，暴露出缺失的训练源机制资格门。下一步不是扩大
claim 或调 reward，而是先用结果盲、非训练 gate 确认实际训练源存在 base-positive prepare opportunity；通过前不授权新训练。

