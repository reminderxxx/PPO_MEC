# 给 CSCWD 2027 A 论文线的 PopArt A/B 主张变更说明

## 身份

- `reviewed_at`: `2026-10-09`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-09`；未新增文献检索
- `target_venue`: `CSCWD 2027`
- `artifact_run_id`: `calibrated_workflow_value_normalization_ab_20261009_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scientific_git_commit`: `858bc797e23b4e56051663f28d6fd7681f5ee77d`
- `evidence_level`: `E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY`
- `verdict`: `FALSIFIED_OR_NOT_PROMOTED`

本说明供 A 线独立回收，没有编辑主论文稿或 A 分支。

## 可以新增的有限主张

在完全匹配的 2-arm × 3-method × 5-seed development A/B 中，PopArt 改善了 shared critic 的数值稳定性：三方法 raw-unit critic RMSE 均下降，value:policy gradient ratio 从约 `740–974` 降至 `13–15`，global clipping 的压缩程度明显减轻。该结论是训练机制证据，不是算法原创性或服务优势。

## 必须拒绝的主张

- 不得写“PopArt 普遍改善服务”：PPO on-time completion `.29→.19`，触发预注册方法级否决；MAPPO completion/on-time 完全中性。
- 不得写“行为问题已解决”：current-missing action-4 raw argmax/executed rate 下降，但总体 mean probability `.444792→.446411`，behavior conjunction gate 失败。
- 不得写“SA 已稳定领先”：SA 的 pooled completion `.87→1.00`，但 on-time 跨 seed 方向混合，completed elapsed `+11.60 s`、recompute `+7.59 s`；且所有数据均已暴露。
- 不得用 normalized value loss 的量级差证明拟合改善；跨臂只引用 denormalized raw RMSE/EV。
- 不得把 5 seeds 当 5 个独立真实 workload，也不得将 regression/frozen-check 称为 holdout。

## 保持不变的边界

- simple/two-step rule 继续明确为 exact-transition、lexicographic model-based planner，能力和目标权限不与 learned actor 等同；
- 旧系统机制负结果、action 4 减少不等于完成率提高、no-aux 未修复过度准备等结论不被本轮推翻；
- PopArt 只保留为有限训练稳定性设置，raw critic 保持 canonical 参照；不进入 formal/paper-ready 主表；
- 不自动进入 auxiliary、reward 或结构搜索。

## 审计限制

30 cells 的 service/behavior 使用 dev-selected checkpoint，但机制门读取 update-24 training batch；28/30 selected checkpoint 早于 update 24。故“训练终点机制通过”和“selected policy 服务表现”不能被写成严格的逐 cell 因果链。该限制不改变 overall fail，也不授权补跑。

如未来需要解释 PPO on-time 下降，先补一条 selected-checkpoint、共同冻结状态上的 value error + action probability/margin + advantage 对齐证据；在此之前不提出新的算法贡献。
