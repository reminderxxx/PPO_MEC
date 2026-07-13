# PPO_MEC Top-Journal Readiness Audit — strict-full v8

- `reviewed_at`: `2026-06-21T20:33:44+08:00`
- `literature_cutoff`: `2026-06-21`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `strict_full_v8_dev_screen_20260621_v2`; `strict_full_v8_formal_all_baselines_20260621_v1`; `strict_full_v8_hidden_holdout_20260621_v1`; `strict_full_v8_external_lust_grid_20260621_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `0db0d9175ecbc2f5db615caa7cf5835150cea746` plus audited working-tree changes
- `evidence_level`: `E2_ARTIFACT_AUDITED`（命令记录为透明的事后重建，未达到 E3）
- `verdict`: `Major revision — 78/100`

本报告审查的是研究贡献与实验证据，不审查尚未提供的论文正文，也不预测 TMC 录用。78 分是 PPO_MEC 内部 operational rubric，不是 IEEE 官方录用线。

## Outcome

v7 的 strict-full scientific blocker 已被实质修复：在新的 outcome-blind、跨 split 互斥、每 split 20 个 outer windows 的协议下，v8 对全部 learned baselines 的 total reward 优势在 formal 与一次性 hidden holdout 上均获得 window-outer hierarchical BCa 95% CI 支持；对 DT baseline 的 workflow continuity 也在两者上为正。

但当前仍不是 `TMC-ready candidate`。hidden 中 v8 相对 PPO 的 handoff failure 显著更差，formal/hidden 的 backhaul cost 也显著更高；对强 supplementary popularity heuristic 的 reward 没有形成优势。LuST 只有 4 个独立窗口，且尚无 v8-current ablation、prediction/system robustness 与 scalability 套件。正确结论是 `Major revision`，不是“全面优于所有方法”。

## Evidence inventory

- 冻结 split：`configs/experiment/top_journal_v8_strict_split_20260621/split_manifest.json`，80 个窗口全局互斥，minimum gap 24 frames；每 split 为 6 mechanism、2 active-nonmechanism、12 idle/sparse。
- 候选与 checkpoints：`artifacts/experiments/top_journal_closed_loop/strict_full_v8_dev_screen_20260621_v2/seed_checkpoint_manifest.json`，5 seeds、10 learned agents。
- formal：`artifacts/experiments/strict_full_v8_formal_all_baselines_20260621_v1/main_results_full_stratified_20260621_025440_591857/`。
- hidden：`artifacts/experiments/strict_full_v8_hidden_holdout_20260621_v1/main_results_full_stratified_20260621_201353_886213/`。
- external：`artifacts/experiments/strict_full_v8_external_lust_grid_20260621_v2/main_results_full_stratified_20260621_202424_612488/`。
- 统计：对应 `artifacts/analysis/strict_full_v8_*_statistics_20260621_v*/paired_statistics.csv`。
- 完整性：`artifacts/audits/strict_full_v8_integrity_20260621/`；11457 个文件进入 SHA-256 inventory，missing reference 与 JSON error 均为 0。
- 执行记录：`docs/project/strict_full_v8_execution_record_20260621.md`。hidden 在候选冻结及 formal 通过后只开启一次；冻结 manifest 未被回写。

## Strict statistical result

所有主 CI 使用 `window_id` 外层、`seed + workflow_id` 内层的 hierarchical BCa bootstrap；每个 NGSIM split 有 20 个 outer windows、200 个 paired rows/baseline。

| Split | Comparator | Metric | Candidate-minus-baseline / signed benefit | BCa 95% CI | Review |
|---|---|---:|---:|---:|---|
| formal | PPO | total reward | `+6.656750` | `[4.013080, 10.782299]` | positive |
| hidden | PPO | total reward | `+3.884200` | `[1.612143, 7.830383]` | positive |
| formal | DT handoff DRL | total reward | `+17.893100` | `[12.429725, 24.247976]` | positive |
| hidden | DT handoff DRL | total reward | `+16.323300` | `[10.347505, 21.426419]` | positive |
| formal | DT handoff DRL | continuity | `+0.280658` | `[0.184057, 0.364369]` | positive |
| hidden | DT handoff DRL | continuity | `+0.266788` | `[0.154254, 0.356273]` | positive |
| hidden | PPO | handoff failure benefit | `-0.074167` | `[-0.198877, -0.002500]` | significant regression; Holm `p=0.011235` |
| formal | PPO | backhaul benefit | `-24.160000` | `[-40.720000, -11.840000]` | significant cost increase |
| hidden | PPO | backhaul benefit | `-14.240000` | `[-25.120000, -6.320000]` | significant cost increase |
| formal | popularity heuristic | total reward | `-0.405700` | `[-1.710537, 0.152826]` | no demonstrated difference |
| hidden | popularity heuristic | total reward | `-0.449450` | `[-2.435498, 0.721750]` | no demonstrated difference |

v8 reward decomposition versus DT on formal shows total `+17.8931`，其中 service、cache、delay、continuity 和 exploration/shaping 分量均有贡献；优势不是单独由 shaping 项构成。该分解不等于每个机制都有独立因果贡献，仍需 current-profile ablation。

LuST `auto_grid_tight` 上，v8 对 PPO reward 为 `+6.9935 [0.454110, 13.443500]`，对 DT 为 `+10.5670 [4.321853, 15.050234]`；但只有 4 个 outer windows，低于 policy 的 12-window 门槛，只能作为 supporting evidence。v8 与 popularity heuristic 在该子集完全相同，不能宣称外部场景优于强 heuristic。

## Hard blockers

本轮未发现仍成立的 strict-full reward/DT-continuity blocker；原 blocker 标记为 resolved。以下 major concerns 在升级为主 claim 时会重新成为 blocker：把 total reward 写成“所有系统指标全面领先”、把 4-window LuST 写成强泛化证据、或用旧 v7 ablation 替代 v8 current-profile 机制归因。

## Major concerns

1. hidden 相对 PPO 的 handoff failure benefit 为负且 BCa CI 全负；这说明 v8 用更积极的机制行为换取 reward，尚未证明 failure-safe。
2. formal/hidden 相对 PPO 的 backhaul benefit 都显著为负。论文必须报告 absolute cost 与 trade-off，不能使用“降低回传开销”作为通用 headline。
3. popularity heuristic 在 formal/hidden 的 reward 点估计仍略高，CI 显示无可证差异；它应保留为 supplementary strong reference。
4. v8 目前没有同一冻结 profile 的 prediction robustness、system robustness、scalability 与逐机制消融；旧 v7 support suite 不能直接继承为 v8 证据。
5. LuST 仅 4 个独立机制窗口，统计功效和覆盖范围不足；默认 layout 的 2-window 无 handoff 运行是无效诊断，不能引用。
6. 执行命令是事后透明重建，不是 wrapper 自动保存的原始 command/stdout/stderr；当前只到 E2，尚未独立重建到 E3。

## Minor concerns

- 当前模型仍使用 calibrated/baseline surrogate interface，不得写成 learned predictor checkpoint。
- MAPPO、QMIX 与 Controller-MAT 都是 controller-level contract，不得称 full vehicle/RSU-agent MARL。
- 多个指标和 baseline 的 sign test 已做 Holm 校正，但主判断仍应以预先指定的 total reward、continuity、failure 与 backhaul hierarchical CI 为准。
- frozen split 基于同一 NGSIM 源的不同时间区间；它验证时间外推，不等同于多城市、多采集条件泛化。

## Scorecard

| Dimension | Score | Evidence and deduction |
|---|---:|---|
| Novelty 与近邻定位 | 14/20 | 完整联合 contract 尚可辩护，但 DAG+mobility+MARL、DAG+cache、hierarchical VEC 子组合已拥挤。 |
| 技术正确性与问题建模 | 12/15 | soft bias 不改 reward/action/env；仍需更清晰的系统代价约束与机制因果解释。 |
| Baseline 公平性与独立性 | 14/15 | 9 个 learned baselines 同窗口、同 workflow、同 seed 与交互预算；controller-level 边界需保留。 |
| 实验设计、统计与 holdout | 18/20 | 20 outer windows、5 seeds、冻结 split、一次性 hidden、hierarchical BCa/Holm；命令 provenance 非原生。 |
| 机制兑现与系统指标 | 7/10 | realization/DT continuity 为正，但 PPO failure 与 backhaul trade-off 明显。 |
| 鲁棒性、泛化与可扩展性 | 6/10 | 有 LuST supporting signal，但仅 4 outer windows，且 v8-current support suites 不完整。 |
| 可复现性与 claim 完整性 | 7/10 | checkpoints、manifest、原始 rows、统计与 11457-file SHA 清单齐全；尚未 E3 独立重建。 |
| **Total** | **78/100** | 无当前致命 blocker，但存在可修复的重要缺口；`Major revision`。 |

## Safe claims

- 在冻结 NGSIM+Alibaba strict-full 协议上，v8 对全部 learned baselines 的 total reward 在 formal 与一次性 hidden holdout 均具有正 hierarchical BCa 95% CI。
- v8 相对 DT handoff DRL 的 workflow continuity 在 formal/hidden 均有正 CI，原 v7 continuity regression 已修复。
- v8 的机制兑现率相对 PPO/DT 多数为正，但伴随可量化的 handoff-failure/backhaul trade-off。
- LuST grid 子集提供方向一致但低功效的外部 supporting evidence。

## Prohibited claims

- “达到 TMC 官方标准”或“可预测录用”。
- “全面优于所有 baseline / 所有系统指标”。
- “显著优于 popularity heuristic”。
- “降低 backhaul”或“handoff failure 全面改善”。
- “LuST 已证明跨城市/跨数据集泛化”。
- “learned surrogate predictor”或“full vehicle/RSU-level MAPPO/QMIX/MAT”。

## Required actions before re-review

1. 在不重新打开 hidden 的前提下，冻结 v8 并运行 prediction robustness、system robustness、scalability 与 current-profile ablation。
2. 将 handoff failure 与 backhaul 写成约束/多目标 trade-off；只在 dev 或新建的 future validation split 上开发 Pareto-safe 版本，现有 hidden 永久只作 consumed evidence。
3. 将 LuST 扩到至少 12 个独立 outer windows；若数据客观不足，增加另一 mobility source 或降低 claim 范围。
4. 让正式 wrapper 自动保存 argv、Git commit/dirty state、环境锁文件、stdout/stderr，再在独立输出目录重建 formal report 以升级 E3。
5. 论文正文中预注册主指标、multiplicity family 和 failure/backhaul non-inferiority margin，并完整报告负结果。

## TITS/TVT fit note

若后续把贡献重点调整为 mobility/handoff-aware VEC controller，并完整报告安全性与通信代价，TITS/TVT 的适配性可能高于当前 TMC 系统面 claim；这不降低本报告的 TMC rubric，也不改变 `Major revision` 判定。
