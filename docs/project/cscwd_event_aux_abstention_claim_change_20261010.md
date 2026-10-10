# Event auxiliary abstention：A 论文线主张与证据变更说明（2026-10-10）

本文件供 A 论文线只读消费；未并发编辑论文主稿、论文表或投稿材料。

## 新增证据

- service-feasible hard-zero 候选继续是 `MIXED_STOPPED`，其多步反例不得删除或改写。
- B 已实现一个不同的 default-off 候选：current-missing 时只屏蔽 event CE 与 temporal-margin 的样本监督，不改 label；
  current-ready 梯度逐张量不变，PPO actor 仍可从长期 return 学 action4。
- slow/fast、value、entropy、PPO、网络、reward、动作权限和 raw inference 不变；候选只传给 SA，三种强基线的初始化、动作概率、
  mask 和旧 checkpoint 语义保持一致。
- 预检核验历史控制 hashes、36-instance 区间、474 个公共前缀 tamper 和网络身份；A 又独立确认 30/30 raw forward、
  6 ready 梯度不变、24 missing event 梯度为零，且 B 已逐 hash 复核不可变 PASS。
- 唯一 development A/B 已完成：selected failure episode/attempt `32→0`/`50→0`，但 fixed96 有 3/5 seed on-time 下降，
  两视角 total transfer 均增加；预注册 verdict=`MIXED_STOPPED`。
- 现有 mobility/time 是 decision-step 仿真：RSU sequence 为 NGSIM handoff-pressure synthetic block，5 秒为人工接触尺度，
  不是逐 frame 真实轨迹。这是外部有效性限制，不是已确认实现 bug。

## 暂不能新增的主张

- 只能写“abstention 在本 development run 减少服务失败，但伴随成本交换且未通过双视角 seed gate”；不能写 auxiliary 是根因、
  SA 稳定领先、整体算法成功或新算法贡献。
- 不能把 hard-zero 的 MIXED 结果解释为 abstention PASS，也不能省略 action4 的目标缓存长期副作用。
- 不能把局部梯度测试、网络同形或独立接口门禁写成性能证据；不能写 formal、holdout、独立测试或 paper-ready。
- two-step 继续明确为 exact-transition model-based、lexicographic objective 权限；不得隐去其能力差异。

## 冻结边界

A 已发布不可变接口 PASS；唯一固定 5-seed、28,800-step、200-new-evaluation development A/B 已消费并判为 MIXED。
当前停止，不产生第二候选；主论文、论文表和投稿材料未编辑。

`reviewed_at=2026-10-10`；`literature_cutoff=2026-09-28`；`target_venue=IEEE TMC`；
`artifact_run_id=cscwd_event_aux_abstention_ab_20261010_v1`；
`policy_version=tmc_review_policy_v3_20260621`；`git_commit=abca7e047d672644f3e31d821831e23f80723a52`（执行），
`46a68f11c289ccc304b88cdfed01c34ba5f61c3d`（实现），
`e9b19f6ad2ec432487a0fa7fbdc6542dfb742dc1`（独立门禁）；
`evidence_level=E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY`；
`verdict=MIXED_STOPPED / paper claim UNVERIFIED`。
