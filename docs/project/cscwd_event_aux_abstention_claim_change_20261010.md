# Event auxiliary abstention：A 论文线主张与证据变更说明（2026-10-10）

本文件供 A 论文线只读消费；未并发编辑论文主稿、论文表或投稿材料。

## 新增证据

- service-feasible hard-zero 候选继续是 `MIXED_STOPPED`，其多步反例不得删除或改写。
- B 已实现一个不同的 default-off 候选：current-missing 时只屏蔽 event CE 与 temporal-margin 的样本监督，不改 label；
  current-ready 梯度逐张量不变，PPO actor 仍可从长期 return 学 action4。
- slow/fast、value、entropy、PPO、网络、reward、动作权限和 raw inference 不变；候选只传给 SA，三种强基线的初始化、动作概率、
  mask 和旧 checkpoint 语义保持一致。
- 预检只核验历史控制 hashes、36-instance 区间、474 个公共前缀 tamper、网络身份与待审计状态；scientific steps 和新评价均为 0。
- 现有 mobility/time 是 decision-step 仿真：RSU sequence 为 NGSIM handoff-pressure synthetic block，5 秒为人工接触尺度，
  不是逐 frame 真实轨迹。这是外部有效性限制，不是已确认实现 bug。

## 暂不能新增的主张

- 不能写“abstention 已改善服务”“auxiliary 是根因”“SA 已稳定领先”或“新算法贡献”；当前没有候选科学 run。
- 不能把 hard-zero 的 MIXED 结果解释为 abstention PASS，也不能省略 action4 的目标缓存长期副作用。
- 不能把局部梯度测试、网络同形或独立接口门禁写成性能证据；不能写 formal、holdout、独立测试或 paper-ready。
- two-step 继续明确为 exact-transition model-based、lexicographic objective 权限；不得隐去其能力差异。

## 冻结边界

A 仅在 B clean implementation commit 上做独立接口/时间/监督审计，并发布不可变 PASS/FAIL/MIXED 收据。只有 PASS 且 B 再次
逐 hash 验证后，才可能授权一次固定 5-seed、28,800-step、200-new-evaluation development A/B；否则保持 0 training。
候选结果若未同时通过 selected/update96 completion/on-time/failure 与 seed 方向门，则停止，不产生第二候选。

`reviewed_at=2026-10-10`；`literature_cutoff=2026-09-28`；`target_venue=IEEE TMC`；
`artifact_run_id=cscwd_event_aux_abstention_ab_20261010_v1 (not created)`；
`policy_version=tmc_review_policy_v3_20260621`；`git_commit=pending clean implementation commit`；
`evidence_level=E1_IMPLEMENTED_AND_LOCALLY_VERIFIED_NO_SCIENTIFIC_RUN`；
`verdict=UNVERIFIED / awaiting independent gate`。
