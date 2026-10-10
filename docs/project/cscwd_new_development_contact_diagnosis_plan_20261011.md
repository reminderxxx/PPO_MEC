# CSCWD 新开发来源接触拒绝诊断事前边界（2026-10-11）

## 固定输入和范围

- 唯一输入为已完成 `cscwd_new_development_reachability_20261011_v1` 的 40 episode、320 step 原件；`step_ledger.jsonl` SHA-256 `679942ac74cc618127142abc89a5eaa0cd109ff38cad213dfd77dd850a67bacb`，`reachability_manifest.json` SHA-256 `481dd42489dbb9bab30f297237b4373fee9b1b3823082031ed308532f2332e0c`。
- 遍历全部 320 行，不挑窗口/方法/任务；重点完整列出全部 12 次执行 action4 和 68 次 contact 拒绝。重复方法在同一 `(window, design, step, action, clock, planned)` 的相同行为单列 episode 计数、另列去重事件数，不伪作独立样本。
- A 的账本诊断新增环境 step 预算为 **0**，不加载模型或读取原始 CSV；仅复算 `planned_step_cost` 与 public/actual contact、trace 剩余的差及逐动作/方法/来源分层。记录 action4 是否 public/actual 预算皆不足，以及 public 误判可行的拒绝数。不从 ledger 推断缺失的阶段时间或 RSU 几何。
- B 的阶段补充若另行使用已重建轨迹，额外确定性诊断不超过 12 个起点、288 个额外环境 step；须独立记录输入、source/hash、目标/当前 RSU、距离速度、覆盖半径、预测/实际 contact、模型/输入/计算/状态阶段、回滚原因。A 在收到该原件前不合并阶段比例。

## 停止与主张边界

- 不改变源窗口、旧 split、raw 环境、RSU 几何、full-step gate、配置、公开 estimator 或 reward；不训练、不做新学习矩阵、不读 formal/hidden 结果。
- 一旦 ledger hash 或 40/320/12/68 身份不一致即停止。分类区分：原始源时长、当前几何接触、public 预测误差、整步回滚、native 准备阶段不可行；缺字段则标 `UNVERIFIED`。
- 只给可复现诊断与唯一下一实施建议。任何跨边界服务、部分传输或新几何属于新的物理合同决策，不从本轮优化结果倒推参数。
