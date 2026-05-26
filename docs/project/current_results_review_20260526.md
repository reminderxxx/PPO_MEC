# Current Results Review 2026-05-26

用途：整理本轮工作区清洗时确认的当前结果状态、漏洞和阻塞点。本文只记录已从本地 artifacts 和长期文档核验的事实；`.trae/documents` 中的草稿只作为待核验输入，不作为事实来源。

## 结果状态

- 当前 canonical paper-grade package 仍为 `artifacts/experiments/top_journal_final_submission/final_submission_full_current_baselines_20260511_v1/`。
- 该 canonical package 的 `final_submission_gate_report.json` 记录 `target_reached=true`、`paper_claim_ready=true`、`blockers=[]`。
- 该 canonical package 的 comparison report 记录 `review_ready=true`、`paper_ready_package_ready=true`，自审为 `blocker_count=0`、`limitation_count=5`、`pass_count=13`。
- 最新 v5 性能/robustness 候选为 `artifacts/experiments/top_journal_final_submission/final_submission_v5_perf_robust_20260515_v1/`，但它不是 canonical replacement。
- v5 的 `final_submission_gate_report.json` 记录 `target_reached=false`、`paper_claim_ready=false`。

## 当前阻塞点

- v5 未通过 promotion gate，阻塞项为 `cache_offload_drl` 的 paired reward cluster CI 下界为负：
  - formal offset 0：`ci95_low=-1.008212`
  - holdout offset 3：`ci95_low=-3.372274`
- v5 尽管所有 split 的均值 margin 仍为正，但最弱 split margin 明显收窄：formal full `+0.837777`、holdout mixed `+1.443333`、holdout full `+1.360953`。因此不能写成更强或更稳版本。
- 当前主张不得引用 v5 作为论文主表；论文主表和 paper-ready 包继续引用 `final_submission_full_current_baselines_20260511_v1`。

## 漏洞与风险

- MAPPO 是当前协议有效的 controller-level CTDE + aggregation-reason head-credit baseline，但 canonical comparison report 已审计出 MAPPO action-mix collapse：MAPPO prefetch 行为为 `0.0`，存在 `prefetch_underuse`、`current_rsu_exec_underuse`、`local_exec_overuse` 和 `migration_overuse`。论文中不能把 MAPPO 低分作为主算法优势的核心证据。
- 主 claim 应锚定 SA-GHMAPPO 相对最强 learned baseline `ppo` 的结果，而不是锚定 MAPPO、QMIX 或较弱 DQN-family baseline。
- `popularity_cache_heuristic` 与 SA-GHMAPPO 很接近，最小 reward margin 约为 `+0.183333`；它只能作为 supplementary heuristic reference，不能写成大幅领先手写规则。
- 当前预测层默认仍不是真实 learned predictor checkpoint。论文应使用 prediction-aware / surrogate-feature-assisted 等表述，不能写成已经加载 learned surrogate predictor。
- 当前 action contract 仍是 `semantic_discrete_5`，不能声称 DAG-level target-node action 或 full vehicle/RSU-level MARL wrapper。

## 待核验审计清单

以下来自本轮清洗时发现的 `.trae/documents` 审计草稿，只作为后续 issue triage 输入；未逐项复核前不得写成当前事实。

- 奖励系数配置化、观测空间维度一致性、seed 传递、invalid action contract、预测扰动顺序等问题需要单独对照当前源码和 tests 复核。
- 2026-05-15 已在 `docs/project/PROGRESS.md` 记录修复或覆盖的项不得重复列为当前漏洞，包括 primary vehicle observation cache、oracle fallback audit 和 prediction history trim。
- 若后续要把审计清单转为正式 BUGS 记录，需要逐项给出文件、当前代码证据、影响面和验证命令。

## 引用建议

- 当前论文可写主结果：使用 `final_submission_full_current_baselines_20260511_v1` 的 paper-ready package。
- 当前负向结果：保留 v5 为 performance/robustness 候选失败记录，用于说明调参方向和 cache-offload 强对照压力，不用于主 claim。
- 当前审稿风险说明：主动报告 MAPPO action-mix audit，并明确 MAPPO/QMIX/Controller-MAT 都是 controller-level learned baselines，不是 vehicle-agent / RSU-agent full MARL wrappers。
