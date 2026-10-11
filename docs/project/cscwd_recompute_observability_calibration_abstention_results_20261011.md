# Recompute-observability calibration-abstention A/B 结果审查

- `reviewed_at`: 2026-10-11
- `literature_cutoff`: 2026-10-11（本轮未新增网页检索）
- `target_venue`: IEEE Transactions on Mobile Computing
- `artifact_run_id`: `cscwd_phase_v2_recompute_observability_calibration_abstention_ab_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- scientific Git commit: `77bd5d35ced95e144a55a09210ed0da06f659eb8`
- evidence level: `E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY`
- scientific terminal / verdict: `PASS / FAIL`

## 完整性与干预

科学 child 正常结束，耗时 318.127 s；20 cells、23,040 environment steps、3,840 optimizer steps、80 selection episodes、
320 learned evaluation episodes、18 rule episodes，formal/holdout read=0，automatic retry=false。run 113/113 文件、自动分析
5/5 文件的 SHA-256 与 size 全部一致。

SA control 的 5,760 个训练状态含 prepare/serve/abstain=`1,287/383/4,090`；candidate 为
`0/402/5,358`，其中 `1,325` 行由 `positive_prepare_prediction_confidence_not_calibrated` 实际触发弃权。candidate 并非零处理。

## 全方法双视图汇总

下表每行 40 episode，混合了 v2 的 3 个 condition 与 v1 mechanism-off 的 1 个 condition；因此只能用来复核自动 verdict，
不能替代后续分层表。全部方法 completion 均为 1.0。

| 方法 | 视图 | reward | on-time | failure episode rate | failures/episode |
|---|---|---:|---:|---:|---:|
| SA public-v4 | selected | 9.044652 | 0.400 | 0.125 | 0.225 |
| SA abstention-v2 | selected | 9.780383 | 0.700 | 0.225 | 0.450 |
| PPO | selected | 8.757794 | 0.400 | 0.150 | 0.300 |
| MAPPO | selected | 9.126188 | 0.500 | 0.175 | 0.350 |
| SA public-v4 | fixed-update24 | 9.364771 | 0.525 | 0.175 | 0.350 |
| SA abstention-v2 | fixed-update24 | 9.424639 | 0.600 | 0.200 | 0.425 |
| PPO | fixed-update24 | 8.673826 | 0.350 | 0.125 | 0.250 |
| MAPPO | fixed-update24 | 8.759444 | 0.350 | 0.125 | 0.250 |

候选在两个汇总视图 reward 分别增加 `+0.735731/+0.059869`，但 failure rate 同时增加 `+0.100/+0.025`，违反事前服务
非劣门。selected 的 reward 分解为 time `+0.211052`、deadline `+1.200`、failure `-0.675`、transfer `-0.000321`；fixed24
为 `-0.015262/+0.300/-0.225/+0.000131`。因此 reward 领先是期限收益覆盖失败罚分的权衡，不是整体服务改善。

## SA 的 profile × 来源分层

数值均为 candidate-control；v2 每格 15 episode，v1 每格 5 episode。

| 视图 | profile | 来源 | Δreward | Δon-time | Δfailure rate | Δfailures | Δrecompute(s) | Δtransfer(MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| selected | v2 | selection-v19 | +0.992464 | +0.200 | 0 | 0 | +2.863001 | +1.210670 |
| selected | v2 | check-v749 | +2.784000 | +0.600 | 0 | 0 | 0 | 0 |
| selected | v1 off | selection-v19 | -3.547545 | 0 | +0.200 | +1.200 | 0 | +1.889837 |
| selected | v1 off | check-v749 | -1.896000 | 0 | +0.600 | +0.600 | 0 | 0 |
| fixed24 | v2 | selection-v19 | -0.217835 | 0 | 0 | 0 | +2.287559 | -1.379246 |
| fixed24 | v2 | check-v749 | +0.928000 | +0.200 | 0 | 0 | 0 | 0 |
| fixed24 | v1 off | selection-v19 | -1.019545 | 0 | 0 | +0.400 | 0 | +1.889837 |
| fixed24 | v1 off | check-v749 | -0.632000 | 0 | +0.200 | +0.200 | 0 | 0 |

新增 service failure 全部位于 v1 mechanism-off；v2 所有来源/视图分层的 failure 增量均为 0。自动汇总按每窗口
`3×v2 + 1×v1` 加权，因而 v2 deadline 增益可使总体 reward 为正，同时掩盖 v1 服务退化。

## 最早失败链

共有 11 个评价行的 candidate failure count 高于 control；它们全部属于 seeds 17/43/61 的 v1 mechanism-off，且首次失败原因
全部为 `current_rsu_contact_expires_before_commit`。在共同初始 public-state hash 上，典型首分叉为 control action2，而
candidate 选择 action0/action3；其他分叉包括 control action4 转 candidate action0/action1。候选训练动作相对 control 为：
action2 `2518→2362`、action4 `958→847`，action0 `1250→1427`、action1 `103→139`、action3 `931→985`。

这支持一个有限因果结论：hard event-label abstention 经共享 encoder 改变了不仅是 event/action4，还改变了 slow/fast 路径；
在 v2 中换来更好期限结果，但迁移到 mechanism-off 时从安全 action2 转向 contact-sensitive 动作并新增失败。它不能证明
shared encoder 是唯一充分原因，也不能用 11 个重复 view 行冒充独立样本。

## 公共规则能力表

18 个 rule episode 全部保留。public immediate 与 public two-step 在本 phase-v2 实现中使用同一 public immediate fallback，
结果逐格相同，不能算两个独立强基线；privileged exact two-step 读取隐藏 actual contact，只在 v1 作 capability reference。
三者还使用 service/deadline/readiness/cost lexicographic objective，与 learned reward 不同。

| 规则 | profile / 来源 | n | reward | on-time | failure rate | capability |
|---|---|---:|---:|---:|---:|---|
| public immediate | v2 / selection-v19 | 3 | 13.823593 | 1.0 | 0 | public causal |
| public immediate | v2 / check-v749 | 3 | 12.398302 | 1.0 | 0 | public causal |
| public immediate | v1 / selection-v19 | 1 | 12.400609 | 1.0 | 0 | public causal |
| public immediate | v1 / check-v749 | 1 | 7.758302 | 0.0 | 0 | public causal |
| public two-step | 同上四格 | 8 | 与 immediate 逐格相同 | 同左 | 0 | same-public-immediate fallback，非独立 baseline |
| privileged exact two-step | v1 / selection-v19 | 1 | 12.400609 | 1.0 | 0 | hidden actual contact reference |
| privileged exact two-step | v1 / check-v749 | 1 | 7.758302 | 0.0 | 0 | hidden actual contact reference |

## 结论与唯一后续候选

本轮候选判定 `FAIL`，停止，不追加 seed、预算或训练。自动报告中的 “two exposed mobility sources” 应勘误为“同一
Lankershim recording 的两个评价窗口”；原 `independent_unit_count=2` 只能解释为窗口数，独立评价单元为 0。旧原件不改写。

若另立实现轮，至多冻结一个候选：**event auxiliary head-local gradient isolation**，即 event auxiliary loss 只更新 event head，
不回传 shared encoder；PPO、critic、slow/fast auxiliary、reward、数据、预算和 selection 全部不变。它针对的是本轮观察到的
跨 head/shared-representation 外溢，不是再做一次 target 消融。

否证条件：任一 `view × profile × source` 分层 failure 增加；v2 on-time 增益消失；或实际 event-head intervention 为 0。该方案
当前仅冻结为设计，未实现、未训练。即便未来通过，也必须补 source-independent formal/holdout，当前不支持 SA 优势或 paper-ready。

机器审查包：`artifacts/analysis/cscwd_phase_v2_recompute_observability_calibration_abstention_ab_20261011_v1_result_review_v1/`；
`review_summary.json` SHA-256=`3b132e3c56a5edefe5cf03d19ff68a2dfdf804e9068a39fca1b20f8d22bd5d47`，完整性清单 SHA-256=
`caecf435a0b0d91a370a6a2c0eff2c412e7efdcb938840eba30a45710c47053b`。

