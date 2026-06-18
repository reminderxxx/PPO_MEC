# PPO_MEC Top-Journal Readiness Audit

- `reviewed_at`: `2026-06-18`
- `literature_cutoff`: `2026-06-18`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `final_submission_v7_latency_fallback_20260528_v1`
- `policy_version`: `tmc_review_policy_v1_20260618`
- `git_commit_at_review`: `dd40c8f`
- `evidence_level`: `E1_DOCUMENTED`
- `verdict`: `Unverifiable`

## Executive Verdict

本轮不能把 PPO_MEC 判为 `TMC-ready candidate`。原因不是已发现 canonical v7 数值失败，而是本机缺少两个正式 run root，Git 历史也没有这些 artifact：

- `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v7_latency_fallback_20260528_v1/`
- `artifacts/experiments/top_journal_final_submission/final_submission_v7_latency_fallback_20260528_v1/`

因此无法核验 checkpoint provenance、command log、formal/holdout/support 原始 rows、paired cluster unit、bootstrap 输入、comparison report 重建和 manifest 外部引用。`ARTIFACT_RECORDS.md`、`PROGRESS.md` 与 `current_results_audit_20260527.md` 的摘要只能支持条件性 desk review，不能替代硬审。

若上述 artifacts 完整恢复且摘要结论全部复核通过，当前研究显示出接近 TMC 投稿候选的实验框架；但 novelty 防线、极小 heuristic margin、机制独立贡献、外部泛化和统计报告仍是 major-revision 风险。

## Evidence Inventory

| 证据 | 状态 | 结论 |
|---|---|---|
| 源码、配置、测试、NGSIM/Alibaba 数据 | available | 全量单元测试和真实 dry-run 可执行，只证明当前代码/数据链可运行 |
| v7 closed-loop run root | missing | 无法核验 SA seed checkpoint、formal gate 和 action-mix 原始结果 |
| v7 final-submission run root | missing | 无法核验 27 个 baseline checkpoint、holdout/support 和 comparison package |
| SHA-256 文件清单 | missing | 无法判断 artifact 是否完整或被后续修改 |
| 文档化 canonical 摘要 | available | 记录了 3 seed、formal/holdout、paired CI、support 和 limitation；属于 `E1` |
| 论文正文 | not in scope / unavailable | 本轮不评价写作质量、图表清晰度和整篇论证结构 |

Artifact 恢复时必须在源主机生成全文件 SHA-256 清单，复制两个完整 run root，并递归补齐 manifest 引用的外部 checkpoint、数据和 analysis 路径。目标主机校验哈希后，才能进入 `E2_ARTIFACT_AUDITED`。

## Literature-Level Assessment

最近邻文献已覆盖单独或两两组合的关键子问题：

- TMC 2025 的 dual dependency-aware service caching + task offloading 已覆盖 DAG/task dependency、service dependency、hierarchical caching 与 PPO。
- TMC 2025 的 trajectory prediction + migration-assisted offloading 已覆盖 mobility prediction、migration 与 MARL。
- TNSM 2025 的 mobility-aware online DAG offloading 已覆盖 vehicle-edge DAG 和在线 mobility-aware 决策。
- TMC/TITS 2025–2026 工作已覆盖 mixed-timescale offloading、service fetching、DT-assisted caching/offloading 和 vehicle-assisted service caching。
- IET ITS 2025 已出现 DAG offloading、跨 edge migration 与 multi-agent cooperative control 的直接近邻组合。

因此仅声称“DAG + caching + mobility + RL”不足以构成 TMC novelty。PPO_MEC 必须把贡献限定为并证明以下交集：跨 RSU 连续 DAG workflow 状态、车载 base model / 路侧 adapter cache warm state、handoff prepare/migration、预测辅助多控制器 contract，以及 NGSIM + Alibaba 下的联合机制兑现。adapter cache 与 service/model/KV cache 的差异必须落实为状态、动作、成本和指标，而不能只更换术语。

## Conditional Experimental Assessment

以下只复述并审查仓库文档，不确认原始数值：

### Documented strengths

- 文档记录 3 seeds、formal + offset=3 holdout、9 个 clean-retrained learned baselines、paired cluster-bootstrap CI，以及 prediction/system robustness 和 scalability support。
- 文档明确 MAPPO/QMIX/Controller-MAT 只是 controller-level baseline，DAG/cache/DT baseline 受 `semantic_discrete_5` contract 限制，避免夸大为 full vehicle/RSU MARL。
- 文档记录 strongest learned baseline 为 `dt_handoff_drl`，并报告 formal/holdout weakest paired CI 下界为正，而不是只挑弱 MAPPO 比较。
- 文档保留 heuristic gap close、mechanism realization 非逐 split 独立正 CI、backhaul 非 universal headline 三项 limitation。

### Major concerns

1. **Heuristic margin 极小。** 文档化 formal/holdout mixed/full 对 popularity heuristic 的 reward margin仅约 `+0.25` 至 `+0.48`，不支持“大幅优于规则方法”。
2. **主要收益与核心机制存在错位风险。** v7 文档将收益归因于 latency fallback / local execution action mix，而 cache/backhaul/handoff 与 heuristic 基本持平；需要证明论文 headline 不是由通用 reward tie-break 主导。
3. **机制独立贡献不足。** 文档已承认 mechanism realization rate 不是每个 split 的 standalone CI-positive 优势，部分早期 DAG/uncertainty 消融 CI 跨 0。
4. **泛化边界窄。** 正式主线只有 NGSIM mobility + Alibaba workflow；offset holdout 仍来自同一数据源和窗口生成协议，不等价于跨城市、跨轨迹集或跨 workflow domain 泛化。
5. **统计透明度待硬审。** 需要确认 paired cluster 的独立单位、bootstrap 层级、窗口/seed 嵌套关系、多个 baseline/metric/split 比较的 family-wise 风险，以及 hyperparameter selection 是否使用了 holdout 信息。
6. **算法命名与 contract 风险。** 主方法名称含 hierarchical multi-agent PPO，但 live wrapper 和强基线均为 controller-level；论文必须用可验证的 agent/controller 定义消除“full MARL”误读。
7. **可复现性证据缺失。** 当前主机无法从 canonical package 重建论文表，尚不满足 `E2/E3`。

### Minor concerns

- 需要报告训练/评估 wall-clock、硬件、峰值内存、总调参计算量和失败实验选择过程。
- 需要明确 NGSIM 与 Alibaba 数据许可、预处理、窗口扫描和 workflow selector 对样本分布的影响。
- 需要说明 3 seeds 的计算约束及其对 CI 稳定性的影响，避免把“3 seed + bootstrap”写成普适统计充分性。
- `baseline_predictor_v2` / calibrated interface 不能写成已加载 learned surrogate checkpoint。

## Scorecard

| 维度 | 分值 | 本轮结果 | 原因 |
|---|---:|---|---|
| Novelty 与近邻文献定位 | 20 | `N/S` | 可做 desk review，但需要论文贡献陈述和最近邻逐项证据 |
| 技术正确性与问题建模 | 15 | `N/S` | 源码可运行，canonical protocol 与论文形式化尚未硬对照 |
| Baseline 公平性与独立性 | 15 | `N/S` | 文档称已通过，27 个 checkpoint 与 manifest 缺失 |
| 实验设计、统计与 holdout | 20 | `N/S` | 原始 rows、cluster unit 和 bootstrap 输入缺失 |
| 机制兑现与系统指标 | 10 | `N/S` | 只有文档摘要，且已知逐 split 独立贡献有限 |
| 鲁棒性、泛化与可扩展性 | 10 | `N/S` | support artifact 缺失；外部数据泛化未覆盖 |
| 可复现性与 claim 完整性 | 10 | `N/S` | canonical artifacts 不可用 |

总分：`N/S (not scored)`。根据 `tmc_review_policy_v1_20260618`，缺少关键 artifact 时不得给出伪精确分数。

## Hard Blockers

- `evidence_blocker:canonical_v7_closed_loop_artifact_missing`
- `evidence_blocker:canonical_v7_final_submission_artifact_missing`
- `evidence_blocker:artifact_sha256_inventory_missing`
- `evidence_blocker:comparison_rebuild_not_possible`

当前未确认额外科学性 blocker；这不代表它们不存在，而是证据不足以完成判断。

## Safe Claims

恢复 artifacts 前只能写成“仓库文档记录”或“待原始 artifact 复核”：

- PPO_MEC 面向跨 RSU 连续 DAG workflow、adapter cache warm state 与 handoff/migration 的联合控制问题。
- 当前实现和测试支持 NGSIM + Alibaba 的真实数据 dry-run。
- 仓库文档记录 v7 在 formal/holdout 中超过所有列出的 learned baselines，并保留 close heuristic 与机制边界；数值尚未在本机硬复核。

## Prohibited Claims

- 不得称当前已通过本轮 TMC-level hard review 或已达到 TMC 官方录用标准。
- 不得称全面、大幅优于 heuristic、oracle prediction 或所有场景。
- 不得把 reward 增益直接解释为 adapter prefetch/migration 的独立因果收益。
- 不得把 controller-level MAPPO/QMIX/MAT 写成 vehicle-agent / RSU-agent full MARL。
- 不得把 `baseline_predictor_v2` 或 calibrated interface 写成已训练 learned surrogate。
- 不得把 NGSIM + Alibaba 单一组合外推为跨城市、跨数据集或真实部署普适结论。

## Required Actions Before Re-review

1. 恢复两个完整 v7 run root、所有 manifest 外部引用和 SHA-256 清单。
2. 校验 final gate、formal/holdout learned gate、27 个 baseline checkpoint provenance、command log 和 protocol version。
3. 在新的临时目录重建 comparison report，逐字段比对 JSON/CSV/LaTeX 与原 package。
4. 审计 formal/holdout 窗口集合是否重叠，确认 selection/hyperparameter tuning 未使用 holdout。
5. 重算 paired/cluster bootstrap，并记录 cluster unit、sample size、CI 方法与多重比较处理。
6. 将 latency fallback、adapter cache、handoff prepare/migration 的收益拆解到机制窗口和非机制窗口，避免 reward-only 归因。
7. 补充至少一个外部 mobility 或 workflow domain，或者把单数据组合明确列为投稿 limitation。
8. 提供论文正文后另做 contribution、formalization、图表、related work 和 claim consistency 审查。

## Policy Exercise

- `final_submission_v7_latency_fallback_20260528_v1`：当前为 `Unverifiable`，因为 canonical artifact 缺失；文档中的 `paper_claim_ready=true` 不覆盖本规范的证据要求。
- `top_journal_mechanism_v6_strong_competition_20260527_v1`：仓库文档已记录 `paper_claim_ready=false` 和 learned/heuristic blockers，因此即使恢复 artifact，也不能作为 TMC-ready canonical；它应保留为 negative/diagnostic evidence。

该演练确认本规范可以区分“项目 gate 通过但证据未恢复”和“项目 gate 已明确失败”两类状态。

## TITS / TVT Fit Note

若后续仍缺少更强的外部泛化和机制因果证据，TITS/TVT 的场景化 VEC 叙事可能比 TMC 更匹配；但 TITS 同样会关注交通场景真实性和指标意义，TVT 也不应以较低门槛为由省略 baseline 公平性、统计和复现证据。投稿方向只能在 artifacts 硬审和论文正文审查后决定。

## Validation Performed On This Host

- `.venv/bin/python -m pytest tests -q`：`58 passed`。
- `.venv/bin/python scripts/smoke_test.py`：通过；仅证明 toy 链路。
- `.venv/bin/python scripts/run_ngsim_sample.py --max_rows 500`：通过。
- `.venv/bin/python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20`：通过。
- `.venv/bin/python scripts/run_real_sample_dryrun.py ...`：通过，episode success、continuity `1.0`；dry-run 不作为论文结论。
- `build_top_journal_comparison_report.py --final_run_root ...final_submission_v7...`：因 `final_submission_gate_report.json` 不存在而失败，直接确认 comparison package 无法重建。
- 文献表标题与 URL 去重检查：未发现重复项；待核验 venue/DOI 条目继续显式保留。
