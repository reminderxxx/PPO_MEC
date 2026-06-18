# PPO_MEC Top-Journal Readiness Audit

- `reviewed_at`: `2026-06-18`
- `literature_cutoff`: `2026-06-18`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `final_submission_v7_latency_fallback_20260618_rebuild_v1`
- `policy_version`: `tmc_review_policy_v2_20260618`
- `git_commit_at_review`: `pending_this_change`
- `evidence_level`: `E3_REPRODUCED`
- `verdict`: `Not TMC-ready`

## Executive Verdict

本机已独立 clean retrain 并重建 v7 closed-loop、9 个 learned baselines、formal/offset-3、prediction/system robustness、scalability 和 comparison package；旧文档中的核心 v7 数值能够复现，artifact 清单、manifest 引用和 checkpoint provenance 均可追溯。因此上一版 `Unverifiable` 结论已被新的 `E3_REPRODUCED` 证据取代。

但硬审发现旧协议的 `offset=3 holdout` 与 formal 使用相邻且重叠的滑动窗口，不能作为独立 holdout；同一 split 内的部分 ranked windows也相互重叠。应用时间不重叠选择和 formal/holdout 排除后，SA-GHMAPPO 在 mixed 模式对 strongest learned baseline `dt_handoff_drl` 的 CI 为正，但 full 模式的 95% CI 跨 0。独立 holdout 是主 claim 的硬要求，因此即使 legacy gate 仍显示 `paper_claim_ready=true`，本轮 verdict 仍为 `Not TMC-ready`。

## Evidence Inventory

| 证据 | 状态 | 结论 |
|---|---|---|
| v7 closed-loop rebuild | complete | `paper_claim_ready=true`；formal mixed/full 核心数值复现 |
| v7 final-submission rebuild | complete | 27 个 baseline checkpoint provenance、formal/legacy holdout/support 完整 |
| comparison package rebuild | complete | `review_ready=true`、legacy `paper_ready_package_ready=true`，但不覆盖严格窗口独立性 |
| artifact integrity | passed | SHA-256 inventory、JSON 引用、外部依赖和 checkpoint 路径无缺失 |
| strict non-overlap formal/holdout | complete | split 内和 split 间窗口均不重叠；full CI 暴露弱证据 |
| current-contract ablations | complete | 5 个机制变体、3 seeds、formal/legacy offset-3 均完成 |
| external mobility transfer | complete with limitation | LuST + Alibaba + `auto_grid_tight` 可运行；reward 对 learned baselines 为正，backhaul 对 heuristic 更差 |
| manuscript | not supplied | 不评价论文写作、公式陈述与图表质量 |

主要路径：

- closed-loop：`artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260618_rebuild_v1/`
- final-submission：`artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260618_rebuild_v1/`
- mechanism ablation：`artifacts/experiments/top_journal_support_suite/top_journal_v7_mechanism_ablation_20260618_v1/`
- integrity：`artifacts/analysis/top_journal_v7_rebuild_integrity_20260618/`

## Reproduced Legacy Results

- Closed-loop formal：mixed `98.396667`，full `90.651296`，与 2026-05-28 记录一致。
- Final legacy protocol：`target_reached=true`、`paper_claim_ready=true`、`blockers=[]`，9 个 learned baselines × 3 seeds clean retrain provenance 完整。
- Comparison package：`review_ready=true`、`paper_ready_package_ready=true`；self-review 为 0 blockers、3 limitations、15 pass。
- Support weakest learned-baseline delta：prediction vs DT `+4.819583`，95% CI `[3.160903, 6.582726]`；system robustness `+9.792153`，95% CI `[8.322102, 11.292569]`；scalability `+4.133380`，95% CI `[3.245373, 5.016079]`。

这些结果可用于说明旧 package 已被独立复现，不能用于证明旧 holdout 独立。

## Strict Window-Independence Re-audit

严格协议要求：split 内窗口不重叠；holdout 与 formal 的 frame interval 不相交；mixed/full 若复用窗口则分别报告，不能当独立 cluster 合并。

| Split | SA | DT | paired total-reward delta vs DT (95% CI) |
|---|---:|---:|---:|
| Formal mixed | `92.993` | `85.318` | `+7.675556 [2.775486, 13.401764]` |
| Formal full | `88.052` | `85.088` | `+2.964167 [-0.202500, 6.453132]` |
| Holdout mixed | `90.436` | `84.837` | `+5.598889 [2.429972, 9.186167]` |
| Holdout full | `85.732` | `84.468` | `+1.263333 [-0.816132, 3.512965]` |

Formal full 和 holdout full 的 CI 均跨 0。点估计仍为正，但不能声称“formal/independent holdout 所有模式均显著优于 strongest learned baseline”。

## Mechanism and External Evidence

当前 v7 profile 的 3-seed 消融在 formal 合并口径下，full method 相对变体的 reward delta 为：

- no latency fallback：`+0.422 [0.268, 0.580]`
- no prediction：`+10.847 [6.465, 15.024]`
- no hierarchy：`+13.176 [7.111, 19.228]`
- no event agent：`+8.127 [4.957, 11.117]`
- no adapter prefetch：`+10.503 [6.085, 14.696]`

`no_prediction` 与 `no_adapter_prefetch` 的多项结果高度相似，说明预测与 prefetch 在当前 policy 中存在机制耦合，不能把两者的 delta 相加或解释为正交因果贡献。

LuST 一维 RSU 布局会导致全程无 association，已判为无效诊断配置。新增二维 `auto_grid_tight` 后，LuST external mobility benchmark 中 SA reward `94.053889`，对 DT 的 paired delta 为 `+5.831944 [4.424042, 7.209472]`；但 SA backhaul `101.333` 高于 popularity heuristic `90.667`，因此只能作为外部 mobility transfer 的初步证据，不能写成全面系统效率优势。

## Scorecard

| 维度 | 得分 | 依据与扣分 |
|---|---:|---|
| Novelty 与近邻文献定位 | 14/20 | 联合问题有区分空间；DAG/cache/migration/mixed-timescale 最近邻密集，尚缺论文逐项 claim matrix |
| 技术正确性与问题建模 | 12/15 | contract 与实现可追溯；controller-level、action granularity 和 predictor 边界需严格表述 |
| Baseline 公平性与独立性 | 14/15 | 9 个 learned baselines clean retrain、预算与 provenance 完整；continuous-control/full-MARL contract 不适配 |
| 实验设计、统计与 holdout | 10/20 | 修复窗口泄漏后 full CI 跨 0；仅 3 seeds；多重比较未校正 |
| 机制兑现与系统指标 | 7/10 | 消融和系统指标已补；机制耦合、heuristic/backhaul 边界仍明显 |
| 鲁棒性、泛化与可扩展性 | 8/10 | prediction/system/scale 和 LuST 已覆盖；外部验证仍是单一 LuST 布局 |
| 可复现性与 claim 完整性 | 9/10 | E3、hash/provenance/command 完整；环境计算成本报告仍可加强 |

总分：`74/100`。此外存在独立 holdout 科学性 blocker，故 verdict 为 `Not TMC-ready`。

## Hard Blockers and Concerns

硬 blocker：

- `science_blocker:strict_full_formal_ci_crosses_zero_against_dt_handoff_drl`
- `science_blocker:strict_full_holdout_ci_crosses_zero_against_dt_handoff_drl`

Major concerns：仅 3 seeds；多个 baseline/metric/split 未做 family-wise/FDR 处理；popularity heuristic reward 很接近；机制耦合削弱单组件因果表述；LuST 只有一个有效二维布局；论文正文尚未接受 claim consistency 审查。

## Safe Claims

- v7 legacy pipeline 已在当前主机独立 clean retrain，并复现 2026-05-28 的核心 formal 结果。
- 严格非重叠 formal/holdout 的 mixed 模式中，SA 对 DT 的 paired reward CI 为正。
- 当前 contract 下的 latency fallback、prediction、hierarchy、event controller 和 adapter prefetch 消融均显示正 reward delta，但 prediction/prefetch 不是正交机制。
- LuST 二维 grid 场景提供了初步外部 mobility transfer 证据，同时暴露 backhaul trade-off。

## Prohibited Claims

- 不得把 legacy `offset=3` 称为独立 holdout，也不得用其正 CI 宣称所有独立 split 显著领先。
- 不得声称当前已经达到、预计达到或保证达到 TMC 录用标准。
- 不得声称全面、大幅优于 heuristic、所有 learned baselines、所有场景或所有系统指标。
- 不得把相关消融 delta 相加，或将 reward 增益直接解释为单一 adapter cache/migration 机制的因果收益。
- 不得把 controller-level MAPPO/QMIX/MAT 写成 vehicle/RSU-level full MARL。

## Required Actions Before Re-review

1. 以严格非重叠协议重新训练/选择 checkpoint；不能只在旧 checkpoint 上反复筛选 holdout。
2. 提高 strict full formal 与 strict full holdout 对 DT 的最弱 CI 下界至正值，并用预先冻结的 selection rule 复核。
3. 将 strict protocol 集成到 final-submission gate，使 legacy offset gate 不再产生 paper-ready 判定。
4. 报告 multiplicity 策略、seed/window/workflow 的层级结构、effect size 和计算成本。
5. 扩展至少一个外部 mobility/workflow 组合，并报告 reward、continuity、failure、backhaul、cache/migration trade-off。
6. 提供论文正文后审查 formalization、contribution、related work、图表和 claim-artifact 映射。

TITS/TVT 可作为场景适配参考，但不能用 venue 调整来绕过窗口独立性和统计证据 blocker。
