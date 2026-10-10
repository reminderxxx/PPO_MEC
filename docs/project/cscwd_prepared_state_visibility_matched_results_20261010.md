# Prepared-state 共同可见性匹配结果（2026-10-10）

## 结论

v4 通过了**接口纠错**验收，但按预注册门槛拒绝为稳定的**性能改进候选**。它消除了同公共状态、同动作却有不同
prepared-state readiness/recompute 的已确认别名，并在两种 checkpoint 视角都显著降低四种 learned 方法的重算和多数
传输成本；然而 SA frozen 的按期改善只出现在 selected 视角，selected 的任一服务失败 episode 从 `8/40` 增至 `13/40`。
fixed update-96 的 SA 按期仍为 `11/40`，失败 episode 仍为 `9/40`。不能用成本改善、省略 fixed-96 或改 failure 定义来
追认成功。

因此，本轮保留 v4 作为共享观测 contract 修复；不声称 SA 领先、算法创新、稳定服务改善或 paper-ready。下一步只能由新的
只读诊断决定一个更近端的学习候选，不能自动删除 auxiliary loss、强制 action 0 或移除 action 4。

## 身份与完整性

- scientific commit=`f46ec72b15f534ac44768a83ef6316c1cfcb6b58`；run root=
  `artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1/`。
- terminal=`PASS`；20 cells、115,200 environment steps、1,920 updates、15,360 optimizer steps、1,200 新评价 episode；
  historical selected 400 与规则 40 行按 hash 复用。119/119 scientific files 独立复核通过。
- canonical post-analysis root=
  `artifacts/analysis/cscwd_causal_prepared_state_visibility_matched_20261010_v1_analysis_v2/`；postprocess commit=`3dc779b`，
  0 training/evaluation/reselection，1,600 learned rows、40 rule rows、800 显式配对。
- source manifest SHA-256=`48718e48dc55e6958c03b676634dca53e84fd21be251115d4809380a1441c615`；source integrity=
  `b66dfb946d2fef8c8e93b3ffacd7f332c1280635c7c635ef1f58f7eb8c4e2ac1`；v2 analysis manifest=
  `13369726f5f4ddf35b3bd96a8ce1cfdadc32bf94b33623d1355511cd6c8687b0`；analysis integrity=
  `92e9697cc634572cd0d60d24e7c30a5e488f7ce362bbe5e9d430750f823dd32b`。

## Frozen development 双视角

下表每个 learned method 都是相同 8 windows × 5 seeds=`40` 行；seed 复用窗口，不是 40 个独立 source clusters。
`failure ep` 是 episode 内 `service_failures > 0` 的指示量，不等于未完成率；括号内为失败尝试总数。

| view | method | completion | on-time | failure ep (attempts) | recompute s | transfer MB | common-completed elapsed Δ s |
|---|---|---|---|---|---:|---:|---:|
| selected | SA | 39→40 | 5→9 | 8→13 (30→19) | 63.93→31.92 | 682.92→94.27 | -32.90, n=39 |
| selected | PPO | 40→40 | 4→8 | 0→0 (0→0) | 26.84→17.11 | 195.98→49.15 | -9.65, n=40 |
| selected | MAPPO | 40→40 | 8→15 | 5→10 (9→16) | 47.11→31.93 | 1203.04→1094.06 | -14.61, n=40 |
| selected | DT | 40→40 | 11→13 | 1→3 (1→5) | 41.14→18.19 | 856.69→487.55 | -20.02, n=40 |
| update-96 | SA | 39→40 | 11→11 | 9→9 (27→12) | 44.81→30.46 | 443.33→165.45 | -15.38, n=39 |
| update-96 | PPO | 40→40 | 5→6 | 0→0 (0→0) | 25.40→10.49 | 192.80→44.84 | -10.10, n=40 |
| update-96 | MAPPO | 40→40 | 10→15 | 2→5 (3→6) | 41.41→23.17 | 1162.01→876.06 | -16.41, n=40 |
| update-96 | DT | 40→40 | 13→14 | 2→1 (2→1) | 33.20→9.53 | 675.59→416.64 | -19.95, n=40 |

SA selected 的失败 episode 增加而失败尝试总数下降，两者并不矛盾：失败分布从较少 episode 内多次失败，变为更多 episode
各含较少失败。预注册首要服务指标使用 episode-level failure，因此该负项必须保留。

## Seed、选模与规则边界

- SA selected update 为 seeds `7/17/43/61→96`、seed `29→72`；主视角不是固定端点的复制，但 4/5 seed 相同，不能将
  两视角当独立证据。
- selected SA frozen 的 on-time 逐 seed 为旧 `0/2/1/2/0`、新 `2/2/1/2/2`（seed 7/17/29/43/61）；失败 episode
  的变化方向并不一致。完整 seed×split 原始计数在 `seed_split_summary.csv`。
- 规则只读引用：two-step frozen `8/8` 完成、`5/8` 按期、0 failure、recompute `2.43 s`、transfer `65.03 MB`；它仍具有
  exact-transition model-based 与 lexicographic objective 权限。Popularity frozen `8/8`、`2/8`、0 failure；没有重评。
- 规则的单轨迹 8 行与 learned 的 5-seed×8 行不能直接当独立样本做显著性或稳定领先结论。

## 预注册门禁裁决

- `PASS`：字段来源仅为过去/当前历史；474 个未来后缀篡改点不改变公开前缀；同动作 transition/reward/cost 不变；
  四方法均消费字段；旧 profile/checkpoint 保持隔离。
- `PASS`：SA recompute 与共同完成 latency 在 selected、update-96 两视角均下降；completion 均 `39→40`。
- `FAIL`：SA on-time 只在 selected 改善，fixed-96 为 `11→11`。
- `FAIL`：selected SA failure episode `8→13`；MAPPO/DT selected 和 MAPPO fixed 也有 failure 恶化，触发“任一方法服务边界
  恶化”门槛。
- verdict：`interface_correctness=accepted_as_shared_observation_contract_fix`；
  `performance_candidate=rejected_by_preregistered_dual_view_gate`。门槛未事后修改。

## 顶刊审查元数据与边界

`reviewed_at=2026-10-10`；`literature_cutoff=2026-10-10`；`target_venue=CSCWD 2027`；
`artifact_run_id=cscwd_causal_prepared_state_visibility_matched_20261010_v1_analysis_v2`；
`policy_version=docs/project/top_journal_review_policy.md@f46ec72`；
`evidence_level=L2_complete_development_artifact_no_independent_test`。

全部 36 实例已暴露；formal/holdout/independent-test 均为 `Unverifiable`。本轮未修改论文、论文表或主稿，未上传 checkpoint、
真实数据或大账本。
