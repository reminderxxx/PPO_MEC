# Positive-prepare calibration abstention：结果与零暴露诊断

- `reviewed_at`: 2026-10-11
- `literature_cutoff`: 2026-10-11
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_phase_v2_positive_prepare_calibration_abstention_ab_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- scientific Git commit: `56d25519a6572fe6822cd5a1c99dcf1a6346929b`
- probe entry Git commit: `77c1a823c697929853dc8ecc357bf216eacfe5e0`
- `evidence_level`: `E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY`
- verdict: `UNINFORMATIVE_ZERO_TREATMENT_EXPOSURE`

## 结论先行

本次 scientific child 有效完成：23,040 environment steps、3,840 optimizer steps、80 checkpoint-selection episodes、
320 learned evaluation episodes、100 checkpoint 文件和 18 条 rule evaluation；未 retry，未读 formal/holdout。run root
的 113 个 manifest 文件和 analysis root 的 5 个文件均无 hash mismatch。supervisor root 没有 producer
`artifact_integrity.json`，其 11 个文件已独立只读计算 hash，但不能写成“与 producer manifest 匹配”。

算法 A/B 没有发生实际处理：四个 learned arm 的 5,760 条训练记录全部是 `abstain, weight=0`；两个 SA arm 的
120 个 update 均累计 `23,040 eligible / 0 supervised / 23,040 abstained`。10 对 selected/fixed SA checkpoint 的
网络张量逐对完全相同，80 对评价的全部指标差均为 0。因此原 analyzer 的 `MIXED` 保留，但不能解释为候选有效、无效或
等效；因果裁决必须是 `UNINFORMATIVE_ZERO_TREATMENT_EXPOSURE`。

## 奖励、服务和成本

下表为 5 seeds 聚合；v2 每格 30 episodes，v1 每格 10 episodes。完整逐 seed、source、condition、view 行在结果审查
artifact 的 `seed_view_profile_summary.csv`、`source_condition_summary.csv` 和 `learned_full_strata.csv`。

| 方法 | view | profile | reward | completion | on-time | failure rate | elapsed s | transfer MB | recompute s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SA control | selected/fixed | v2 | 12.9220 | 1.000 | 1.000 | 0.000 | 88.503 | 12.395 | 4.838 |
| SA candidate | selected/fixed | v2 | 12.9220 | 1.000 | 1.000 | 0.000 | 88.503 | 12.395 | 4.838 |
| PPO | selected/fixed | v2 | 12.8129 | 1.000 | 1.000 | 0.000 | 89.886 | 9.078 | 6.420 |
| MAPPO | selected | v2 | 12.9056 | 1.000 | 1.000 | 0.000 | 88.716 | 11.091 | 5.126 |
| MAPPO | fixed-update24 | v2 | **12.9365** | 1.000 | 1.000 | 0.000 | **88.326** | 11.762 | **4.694** |
| all learned arms | both | v1 mechanism-off | 2.1795 | 1.000 | 0.500 | 1.000 | 104.092 | 2.554 | 0.000 |

SA 在 selected v2 的 reward 略高于 MAPPO，但 fixed-update24 低于 MAPPO；且 SA candidate 与 control 完全相同，不能把
任一差异归因于 calibration abstention。PPO 传输最少但重算最多，MAPPO fixed-update24 的 reward/elapsed/recompute 最优；
这是资源交换，不是 SA 稳定领先。

两条 public rule 在 v2 完全相同：reward `12.6843`、completion/on-time/failure=`1/1/0`、elapsed `91.493 s`、
transfer `9.219 MB`、recompute `0`。v1 public rule 与 privileged exact reference 也相同：reward `10.0795`、
completion/on-time/failure=`1/.5/0`。public two-step 在 v2 是同 public immediate fallback，不计为独立强基线。

## 标签路径与最小见证

旧训练日志足以确认：开关已进入实际 config/update 日志，但两个 SA arm 都没有一条受监督 label；四种 learned policy 的训练
动作也都只有 0/2/3，没有 action 4。旧 `training_signal_rows.csv` 未保存 semantic state 或 label reason，因此不能事后声称
23,040 个旧状态究竟是哪一个前置门失败。

静态实现验收中的唯一 `prepare→abstain` 见证来自 US-101 public fixture，而训练源固定为 Lankershim。为区分“consumer
未接线”和“训练域没有机制机会”，执行了一个预注册、create-only 的 Lankershim/default 非训练探针：seed 7、固定
`0,1,2,3,4` 动作周期、96 environment steps、0 optimizer/parameter/checkpoint/formal/holdout。结果为：

- 96/96 base 与 gated label 都是 `abstain`，原因均为 `action4_not_legal`；
- 96/96 public state 的 predicted target 为 null，action 4 mask 为 false；
- 19 次固定周期希望请求 action 4，均按预注册 mask fallback，实际动作仍为 0/2/3；
- 观察到 26 个唯一 semantic-state hash，所有 96 个完整 public semantic state 均保留；
- 该有界见证不证明 Lankershim 域永远没有 prepare/serve，只证明当前冻结 source/window 与固定覆盖的 96 步未产生处理机会。

这排除了“新开关虽开启但 loss consumer 没有读取”作为首要解释；最短解释是授权协议没有先要求实际训练源具有非零 base-positive
label 暴露，导致用 US-101 静态见证授权了 Lankershim 训练。它是实验资格门缺失，不是可包装为算法创新的实现 bug。

## 已确认、否定和未定位

- 已确认：scientific run 完整；candidate/config/checkpoint 语义接线存在；A/B 零处理暴露；SA 两臂张量与评价严格相同；
  Lankershim 96-step probe 中 action 4 全程不合法。
- 否定：本 run 不支持 calibration abstention 提升 reward；也不支持它伤害 reward或与 control 等效；不能支持 SA 稳定领先。
- 未定位：旧 23,040 行逐状态的 exact abstain reason；更广 Lankershim 域是否存在机会；有处理暴露时该 auxiliary 的净效果。
- 保留：真实 cross-RSU 净收益、sharing×migration 因果交互、三头结构相对参数匹配 baseline、source-disjoint 泛化均为
  `UNVERIFIED`。本 A/B 不提供其中任何一项证据。

## 唯一下一候选与停止门

本轮不再改算法，也不延长预算。唯一优先项是修正实验资格协议：在任何下一次训练授权前，对**实际训练源**执行结果盲、
非训练 mechanism-coverage gate，并要求 base public label 至少出现非零 `prepare`；状态选择只依据事前冻结的 mobility/handoff
几何与 public action eligibility，不读取 reward、checkpoint、评价结果或 future truth。当前 US-101 已用于选模，不能改作训练；
若现有未暴露来源不足，应先另立 source-disjoint 数据资格协议，本轮不搜索或下载。

若资格门通过，下一轮 A/B 的唯一变量仍是
`mechanism_aux_positive_prepare_calibration_abstention_enabled=false/true`；两臂使用相同新训练源、5 seeds、23,040-step 总共同
预算、同选模与 selected/fixed 视角。主要指标为 workflow completion、on-time completion、service failure；共同报告 reward、
elapsed、transfer、recompute 和 label exposure。任一视角 completion/on-time 下降或 failure 增加即否定候选；若 supervised
prepare 仍为 0，则立即判 `UNINFORMATIVE` 并停止，不再训练或自动 retry。

后续实际 consumer 资格核验已执行且失败：96 个 Lankershim state 的 direct label 与两个 SA
`_event_auxiliary_target` 全部一致（mismatch=0），但 base-positive prepare=`0`、candidate prepare→abstain=`0`；US-101 在读取
label/reward 前因 selection 角色排除。故 `gate_passed=false, training_authorized=false`，当前可授权训练来源数为 0，未使用剩余
96-step 上限。下一轮未授权计划见 `cscwd_next_positive_prepare_matched_training_plan_20261011.md`。

## Artifact

- result review root：`artifacts/analysis/cscwd_phase_v2_positive_prepare_calibration_abstention_ab_20261011_v1_result_review_v1/`
  - `artifact_integrity.json` SHA-256=`943017d2c38ecc7b36a6a87756e3ef876d432cf94ffcad5d0e10ce6196ece297`
  - `review.json` SHA-256=`fd1a6f99713756587f4b00ab1d431c5fa9a96ef3e644a7b07ba277f71de221b8`
- probe root：`artifacts/analysis/cscwd_lankershim_event_label_coverage_probe_20261011_v1/`
  - `artifact_integrity.json` SHA-256=`337e29848a4343c86922240b234d4d8d8245ce64d1f3c99df9fa60ace2ce7e86`
  - `summary.json` SHA-256=`8c2d885be45250785d930bef81998eaf65939ee3a5d961e9f982141de3ca54db`
- qualification root：`artifacts/analysis/cscwd_training_source_positive_prepare_qualification_20261011_v1/`
  - `artifact_integrity.json` SHA-256=`b822fd65441ded573cdd6665c78258171689a5bfe038e577250fdb43caa14244`
  - `qualification.json` SHA-256=`704fd05a7f564ee7a670ce17d048fcf13d30c42c40831fe8445b5bb986377e5b`
