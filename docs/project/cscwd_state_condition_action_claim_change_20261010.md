# CSCWD A 论文线：主张与证据变更说明

日期：2026-10-10。此文件仅供 A 论文线独立消费；本轮没有并发编辑主论文稿。

## 可新增的安全表述

- 在 11 个已暴露 development 分叉状态、26 条既有合法分支中，只有 1 个候选动作被完整服务—资源向量严格支配；
  8 个状态是 Pareto 交换，2 个无法由现有分支归因。
- action 2 在两个公开状态中呈相反效果：deadline slack 不足时可造成可避免超期，slack 足够且后续 recompute 较大时
  可严格改善按期结果。这支持“决策必须状态条件化”，不支持全局偏好或屏蔽某动作。
- current-missing 条件下的 5 条 action 4 分支均首步失败，但 3/5 最终仍按期；首步服务失败与 workflow 结局不可等同。

## 不得新增或必须撤回的表述

- 不得称 event/fast auxiliary、reward 数值尺度或 action 4 本身是已定位根因。
- 不得称新增 model bytes 全部是无效预取，也不得称 action 2 零 model-load bytes 证明其成本更低；车辆 fallback 是共同环境能力。
- 不得把本轮已暴露 development 见证写成 holdout、formal、算法增益或论文创新。
- 不得把 two-step rule 与 learned method 写成同等信息权限；前者仍有 estimated-model clone 和词典序目标权限。

## 当前论文状态

旧负结果、checkpoint 和 `MIXED_STOPPED` 全部保留。当前只可把“状态条件动作信用分配”列为待区分机制，不可列为已实现
贡献。冻结的 elapsed-time 环境敏感性已经显示四种 learned fixed policy 的方向混合，且新 profile 缺少真实逐决策接触时段
支持；这触发停止条件。应先修订并验证任务时间合同，而不是启动算法消融。

证据入口：`docs/project/cscwd_state_condition_action_evidence_20261010.md`；artifact：
`artifacts/analysis/cscwd_state_condition_action_evidence_20261010_v1/`。
