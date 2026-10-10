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
- A 对 24 条冻结失败源和 6 条排序正例执行了 90 个 action0/2/4 分支与冻结策略后缀；一步可服务性和正例不变性成立，但
  22 个受影响分支中 action4 有 11 个 Pareto 非支配，3 个终局服务排序严格优于 action0/2。门禁因此为 `MIXED`。

## 尚不能新增的主张

- 不能写“标签修正已验证”“辅助监督是根因”或“action4 应在 current-missing 状态一律避免”；多步反例已经否定最后一种
  统一规则的充分性。
- 不得把代码默认关闭、单元测试或梯度方向改变写成性能证据；不得省略 action4 的目标缓存副作用和可能的多步收益。
- 不得写 SA 稳定领先、算法创新、formal/holdout 有效或 paper-ready。two-step 继续标注 exact-transition model-based 能力与
  lexicographic objective 权限。

## 最终主张边界

- 候选不训练；论文线只保留“发现标签/动作语义疑点，但反事实不支持统一收紧”的负结果。
- 可报告 action4 的双重作用：它可能牺牲当前服务，却通过目标缓存改变冻结策略的后续状态并改善按期/失败结局；这只是有界
  development 反例，不能外推为 action4 或旧辅助标签普遍最优。
- 若未来提出区分短期服务与未来准备价值的新信用分配候选，必须另立事前计划和新证据，不能在本批反例上调权重或选样本。

`reviewed_at=2026-10-10`；`literature_cutoff=2026-09-28`（本轮未做 novelty 检索）；`target_venue=IEEE TMC`；
`artifact_run_id=cscwd_service_feasible_action_branches_20261010_v1`；
`policy_version=tmc_review_policy_v3_20260621`；
`git_commit=c8aecf19b39c83debf57d20f44a90f6ebb650e68`（gate），
`d17c374911b4fdcad1c37b11b4087be0f7fa0e22`（candidate implementation）；
`evidence_level=E2_ARTIFACT_AUDITED_BOUNDED_DEVELOPMENT_GATE_NO_CANDIDATE_RUN`；
`verdict=MIXED_gate_stop / Unverifiable_for_performance_or_paper_claim`。
