# Current Results Audit

更新日期：2026-05-27

用途：汇总当前可写入论文的结果、刚实现但尚未正式验证的算法更新，以及基于 2025-2026 相关顶刊/顶会检索后暴露的结果缺口。本文不替代 `artifacts/` 下的 JSON/CSV 产物；所有数值 claim 仍以 artifact 为准。

## 结果审计表

| 项目 | 当前状态 | 证据路径 / 数值 | 是否可写入主论文 | 缺口 / 阻塞点 |
|---|---|---|---|---|
| canonical final submission | 已通过 | `artifacts/experiments/top_journal_final_submission/final_submission_full_current_baselines_20260511_v1/final_submission_gate_report.json`：`paper_claim_ready=true`、`target_reached=true`、`blockers=[]` | 可以作为当前主结果 | MAPPO 使用的是 2026-05-11 当时 head-credit protocol，不是 2026-05-27 新增 v3；若论文强调优化后 MAPPO，需要重跑。 |
| canonical comparison package | 已通过 | `comparison_report/top_journal_comparison_report.json`：`review_ready=true`、`paper_ready_package_ready=true`、`self_review_summary.blocker_count=0`、`limitation_count=5`、`pass_count=13` | 可以作为当前 paper-ready 表格来源 | 必须保留 5 个 limitation，特别是 MAPPO action-mix 风险、heuristic 接近和 prediction/oracle 边界。 |
| canonical learned-baseline gate | 已通过 | formal/holdout paired total reward 对 9 个 learned baselines 均 positive CI；最弱 vs `ppo`：formal mean `+4.745278`、CI `[2.3372, 7.028835]`；holdout mean `+6.975`、CI `[4.155505, 9.63982]` | 可以写“超过当前 clean-retrained learned baselines” | 主张应锚定 strongest learned baseline，而不是弱 MAPPO；heuristic 只作 supplementary reference。 |
| canonical split-level margin | 已通过但有窄边界 | Formal Mixed `+7.871667`、Formal Full `+3.703148`、Holdout Mixed `+10.097777`、Holdout Full `+5.636667`，最强 learned baseline 均为 `ppo` | 可以写，但不要夸大 | Formal Full 是最窄主 split；后续 v6 必须确认该 split 不被优化后 `cache_offload_drl` / MAPPO v3 拉低。 |
| support suites | 已通过 | vs `ppo` 最弱项：Prediction `+2.794583`、Robustness `+6.879236`、Scalability `+2.159306`，CI 均为正 | 可以作为补充表 | `no_prediction` / `oracle_prediction` 仍是 diagnostic，不支撑 universal prediction dominance。 |
| MAPPO action-mix audit in canonical | 有效风险 | canonical 中 MAPPO vs PPO/DQN 的 8 条 action-mix audit 均为 high risk；MAPPO prefetch 为 `0.0`，vs PPO reward delta 约 `-29.25` 到 `-31.81` | MAPPO 可列为 controller-level CTDE baseline，但不能作为主优势证据 | 已实现 MAPPO v3 后必须重跑同预算 final loop；否则论文只能说“旧 MAPPO 有 action-mix collapse，主 claim 锚定 PPO”。 |
| v5 performance/robust candidate | 未通过 | `final_submission_v5_perf_robust_20260515_v1/final_submission_gate_report.json`：`paper_claim_ready=false`；blockers 为 `cache_offload_drl` formal CI low `-1.008212`、holdout CI low `-3.372274` | 不可替换 canonical | v5 主 split margin 太窄：Formal Full `+0.837777`、Holdout Mixed `+1.443333`、Holdout Full `+1.360953`；不通过 promotion gate。 |
| v5 MAPPO behavior | 有改善但不能单独晋级 | v5 MAPPO action-mix audit 变为 `tracked`，prefetch 约 `0.24` 到 `0.5`，不再是 canonical 那种 `prefetch=0` collapse | 只能作为失败候选分析 | v5 失败主因不是 MAPPO，而是 `cache_offload_drl` CI；不能拿 v5 写 paper-ready claim。 |
| MAPPO v3 implementation | 代码已实现，正式结果未出 | `src/agents/mappo_agent.py`、`configs/algo/mappo.yaml`、`mappo_strong_audit` profile；debug 训练和 checkpoint 恢复已通过 | 不能直接写数值优势 | 缺少 3 seed formal/holdout final-submission 重跑；缺少 MAPPO v3 action-mix audit 和 strongest-baseline ranking。 |
| SA v6 strong-competition profile | 代码已实现，正式结果未出 | `top_journal_mechanism_v6_strong_competition` profile 与配置已加入；2-episode debug train 通过，`paper_claim_ready=false` 为预期 | 不能写为结果 | 必须运行 full final-submission loop，验证 against MAPPO v3、domain baselines、Controller-MAT、QMix、DQN-family。 |
| 文献覆盖 | 已补充 2025-2026 相关条目 | `docs/project/literature_reference_table.md` 新增 TITS 2025 distributed VEC offloading、TITS 2025 SAGIN offloading、FGCS 2026 V2V/MARL offloading、TMC 2025 EdgeLLM、TMC 2026 H2O | 可用于 related work 和 reviewer response | 正式投稿前仍需复核 IEEE Xplore/Elsevier/DBLP 最终卷期、页码、DOI；H2O DOI 当前待核验。 |

## 当前还差什么

| 优先级 | 缺口 | 需要补的结果 | 验收标准 |
|---:|---|---|---|
| P0 | MAPPO v3 + SA v6 同预算正式重跑 | 使用 `top_journal_mechanism_v6_strong_competition` 和 `mappo_strong_audit` 跑 final-submission loop | formal + holdout learned gate 全通过；cluster-bootstrap total reward CI 对全部 learned baselines 为正；`paper_claim_ready=true`。 |
| P0 | 优化后 strongest baseline 变化 | 重新生成 comparison report，确认 strongest learned baseline 是 PPO、MAPPO v3、`cache_offload_drl` 还是其他 | 不能 hard-code PPO；主 claim 必须对实际 strongest learned baseline 成立。 |
| P0 | MAPPO v3 action-mix audit | 生成新版 `mappo_action_mix_audit.csv/.tex` | MAPPO v3 不应再出现 prefetch 全 0 或大幅 action-mix collapse；若仍弱，论文必须主动报告。 |
| P1 | `cache_offload_drl` 阻塞复核 | 针对 v5 中 `cache_offload_drl` CI 下界为负的问题，检查是否因该 baseline 变强、SA v5 变弱或 split 方差过大 | 如果 v6 仍对 `cache_offload_drl` CI 不为正，不能 paper-ready。 |
| P1 | heuristic 边界 | 继续保留 `popularity_cache_heuristic` supplementary 对照，不把它写成 primary gate | 若新 run 中 SA 低于 popularity 或 margin 过窄，论文应写成 learned-baseline 主 claim，heuristic 只作参考。 |
| P1 | 预测 claim 边界 | 保留 learned/noisy predictor setting-level CI；不要把 no-prediction/oracle diagnostic 写成全面优势 | Prediction robustness 对 claim-relevant settings 为正 CI；oracle/no-prediction 只进入诊断讨论。 |
| P2 | 真实 AI model-cache backend | 当前 HF/model-cache 仍是 metadata/file-size profile，不是正式 benchmark trace | 若要写“真实 model cache 请求/事件”，必须先实现 importer、adapter_id 映射和独立 profile。 |
| P2 | full vehicle/RSU MARL | 当前 MAPPO/QMIX/Controller-MAT 是 controller-level baselines | 不写 full MARL；若要写，必须冻结 vehicle/RSU-level observation/action contract 并重训对照。 |

## 推荐下一轮命令

```bash
python scripts/run_top_journal_closed_loop.py --run_id top_journal_mechanism_v6_strong_competition_20260527_v1 --seeds 7 13 29 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
python scripts/run_top_journal_final_submission_loop.py --run_id final_submission_v6_mappo_v3_strong_20260527_v1 --base_manifest_path artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --mappo_baseline_profile mappo_strong_audit --minimum_reward_delta 0.5 --holdout_offsets 3 --seeds 7 13 29
python scripts/build_top_journal_comparison_report.py --final_run_root artifacts/experiments/top_journal_final_submission/final_submission_v6_mappo_v3_strong_20260527_v1 --bootstrap_samples 5000
```
