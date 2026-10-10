# Service-feasible event target：主张与证据变更说明（2026-10-10）

本文件仅供 A 论文线只读消费；没有编辑论文主稿、论文表或投稿材料。

## 当前新增证据

- 已确认的近端失败仍是：SA 在当前完整 bundle 缺失时选择不修复当前服务的合法动作；环境执行与行为账本未发现实现错误。
- 实际训练的 event auxiliary target 未检查 action4 当前服务可行性；24 条有界失败探测中 22 条仍取正标签，但只有 7 个不同
  公开状态。该证据支持一个学习监督候选，不证明因果充分性。
- B 已用默认关闭开关实现唯一 target-only 候选：仅把 `current_complete_bundle_ready` 合取到 event hard/soft target。关闭时旧
  target 保持，slow/fast target、网络、reward、critic、动作权限、推理与预算不变；checkpoint target 语义显式隔离。
- 预检和局部测试只证明实现范围与 fail-closed 权限正确：固定 seed 下候选/旧版均为 165,512 参数、60 个 state keys 且初始
  张量相同；当前为 0 scientific training、0 new evaluation。

## 尚不能新增的主张

- A 的 action0/2/4 对称分支仍在执行，尚无 PASS/FAIL/MIXED、manifest 或 report hash；因此不能写“标签修正已验证”、
  “辅助监督是根因”或“action4 应在 current-missing 状态一律避免”。
- 不得把代码默认关闭、单元测试或梯度方向改变写成性能证据；不得省略 action4 的目标缓存副作用和可能的多步收益。
- 不得写 SA 稳定领先、算法创新、formal/holdout 有效或 paper-ready。two-step 继续标注 exact-transition model-based 能力与
  lexicographic objective 权限。

## 后续证据如何改变主张

- A 门禁 `FAIL/MIXED`：候选不训练；论文线只保留“发现标签/动作语义疑点但反事实不支持统一收紧”的负结果。
- A 门禁 `PASS`：仅授权一次已冻结 development A/B，不直接授权论文主张。候选还须同时满足 completion 不降、selected 与
  fixed-96 的 on-time/failure 边界，以及两视图至少 3/5 seed 改善且至多 1/5 恶化；trade-off 记 MIXED。
- 即使 A/B `PASS`，也只能写成已暴露 development 上的辅助监督机制证据；独立 source/formal/holdout 仍需另立冻结任务。

`reviewed_at=2026-10-10`；`literature_cutoff=2026-10-10`；`target_venue=IEEE TMC`；
`artifact_run_id=none_preflight_only_awaiting_action_branch_gate`；
`policy_version=tmc_review_policy_v3_20260621`；
`git_commit=d17c374911b4fdcad1c37b11b4087be0f7fa0e22`；
`evidence_level=E1_IMPLEMENTATION_AND_DEVELOPMENT_DIAGNOSIS_NO_CANDIDATE_RUN`；
`verdict=Unverifiable_for_performance_or_paper_claim`。
