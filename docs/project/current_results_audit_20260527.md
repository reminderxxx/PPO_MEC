# Current Results Audit

更新日期：2026-05-27

用途：给论文写作和下一轮实验调度提供明确状态判定。本文只汇总已存在 artifact 与已实现代码能力；所有数值结论以 `artifacts/` 下 JSON/CSV 为准。

## 状态判定规则

- `Results 可引用`：已有 formal/holdout/support artifact，且 gate 中 `paper_claim_ready=true` 或 comparison package `paper_ready_package_ready=true`。
- `Methods 可描述`：代码、profile、配置或审计协议已实现，但没有 3 seed formal/holdout final-submission artifact；不得写任何性能提升数值。
- `Negative / Appendix only`：artifact 已生成但 gate 未通过；只能作为失败候选或审计边界，不进入主结果。
- `Related Work only`：外部论文只支撑动机、定位、baseline rationale 或 reviewer response；不得支撑 PPO_MEC 数值 claim。

## 结果审计表

| 对象 | 判定 | 精确证据 | 论文写作处理 | 必须修改 / 补齐的项 |
|---|---|---|---|---|
| `final_submission_full_current_baselines_20260511_v1` final gate | `Results 可引用` | `final_submission_gate_report.json`：`paper_claim_ready=true`、`target_reached=true`、`blockers=[]` | Results 主表可引用；句子必须限定为“under the 2026-05-11 current-baseline protocol”。 | 若论文强调 MAPPO 已被优化，需要另跑 MAPPO v3 / SA v6 final-submission，不得用该 artifact 替代。 |
| `final_submission_full_current_baselines_20260511_v1` comparison package | `Results 可引用` | `top_journal_comparison_report.json`：`review_ready=true`、`paper_ready_package_ready=true`、`self_review_summary.blocker_count=0`、`limitation_count=5`、`pass_count=13` | 可导出 paper-ready table；正文必须同步写 5 个 limitation。 | 不得删除 MAPPO action-mix risk、heuristic 接近、prediction/oracle 边界。 |
| canonical learned-baseline CI | `Results 可引用` | formal/holdout 对 9 个 learned baselines 的 paired total reward CI 全为正；最弱 vs `ppo`：formal mean `+4.745278`、CI `[2.3372, 7.028835]`；holdout mean `+6.975`、CI `[4.155505, 9.63982]` | 可以写“SA-GHMAPPO outperforms all clean-retrained learned baselines in the canonical package”。 | 主 claim 必须按 actual strongest learned baseline 排序，不得硬写“只比 MAPPO 强”。 |
| canonical split-level margin | `Results 可引用，弱项需标注` | Formal Mixed `+7.871667`、Formal Full `+3.703148`、Holdout Mixed `+10.097777`、Holdout Full `+5.636667`；四个 split 的 strongest learned baseline 均为 `ppo` | 可写 split-level margin；Formal Full 是 weakest split，需如实报告。 | 下一轮 v6 必须重新检查 Formal Full；若被 MAPPO v3 或 `cache_offload_drl` 拉低，则不得沿用 canonical margin。 |
| support suites | `Results 可引用` | vs `ppo` 最弱项：Prediction `+2.794583`、Robustness `+6.879236`、Scalability `+2.159306`，CI 均为正 | Supplementary / robustness 表可引用。 | 不得把 `no_prediction` / `oracle_prediction` diagnostic setting 写成 universal prediction dominance。 |
| canonical MAPPO action-mix audit | `Results 可引用为风险说明` | 8 条 MAPPO-vs-PPO/DQN audit 均为 `high`；MAPPO prefetch `0.0`；vs PPO reward delta 约 `-29.25` 到 `-31.81` | MAPPO 只能写作有效 controller-level CTDE baseline，且在该 artifact 中存在 action-mix collapse。 | 主优势证据锚定 PPO / strongest learned baseline；不得把旧 MAPPO 低分写成主贡献。 |
| `final_submission_v5_perf_robust_20260515_v1` | `Negative / Appendix only` | `paper_claim_ready=false`；blockers：`cache_offload_drl` formal CI low `-1.008212`、holdout CI low `-3.372274` | 可在 appendix 或 internal audit 写“v5 failed promotion gate”。 | 不得替换 canonical；不得输出 paper-ready claim。 |
| v5 split margins | `Negative / Appendix only` | Formal Full `+0.837777`、Holdout Mixed `+1.443333`、Holdout Full `+1.360953` | 只能解释 v5 为什么没有晋级。 | 后续优化目标必须提高 weak split margin，并消除 `cache_offload_drl` CI blocker。 |
| v5 MAPPO behavior | `Negative / Appendix only` | MAPPO action-mix audit 降为 `tracked`；prefetch 约 `0.24` 到 `0.5` | 可写“v5 中 MAPPO collapse 缓解，但 package 仍失败”。 | 不能把 MAPPO 改善单独写成主结果，因为 final gate 未通过。 |
| MAPPO v3 / `mappo_strong_audit` | `Methods 可描述` | 已实现 `aggregation_reason_weighted_controller_ppo_v3`、三头 policy/entropy floors、checkpoint config/load 审计；debug training 通过 | Methods / implementation 可描述；不得写“MAPPO v3 更强”或任何 performance claim。 | 必须跑 3 seed formal + holdout + comparison report；必须生成新版 `mappo_action_mix_audit.csv/.tex`。 |
| SA v6 / `top_journal_mechanism_v6_strong_competition` | `Methods 可描述` | profile 与配置已实现；2-episode debug train 通过，`paper_claim_ready=false` 是 debug 预期 | Methods / experimental plan 可描述；不得写“v6 已优于强基线”。 | 必须与 MAPPO v3、`cache_offload_drl`、Controller-MAT、QMix、DQN-family 同预算跑 full final-submission。 |
| 2025-2026 文献补充 | `Related Work only` | `literature_reference_table.md` 新增 TITS 2025 distributed VEC offloading、TITS 2025 SAGIN offloading、FGCS 2026 V2V/MARL offloading、TMC 2025 EdgeLLM、TMC 2026 H2O | 只能用于 Related Work、motivation、baseline rationale 或 reviewer response。 | H2O DOI 仍需 IEEE Xplore 复核；FGCS 条目不可称为 IEEE/ACM 顶刊主证据。 |

## 下一轮实验验收表

| 优先级 | 验收对象 | 必须生成的 artifact | 通过条件 | 失败处理 |
|---:|---|---|---|---|
| P0 | SA v6 + MAPPO v3 final gate | `artifacts/experiments/top_journal_final_submission/final_submission_v6_mappo_v3_strong_20260527_v1/final_submission_gate_report.json` | `paper_claim_ready=true`、`target_reached=true`、`blockers=[]` | 若失败，只能写 Negative / Appendix；不替换 canonical。 |
| P0 | learned-baseline paired CI | final comparison report | formal 和 holdout 对全部 learned baselines 的 total reward CI 下界均 `> 0` | 哪个 baseline CI 下界 `<=0`，哪个就是 blocker，必须继续优化或降级 claim。 |
| P0 | strongest learned baseline 排名 | `strongest_comparator_audit` | 每个 split 中 SA-GHMAPPO 都高于 actual strongest learned baseline | 不允许 hard-code PPO；如果 strongest 变为 MAPPO v3 或 `cache_offload_drl`，按实际结果写。 |
| P0 | MAPPO v3 action-mix audit | `mappo_action_mix_audit.csv` / `.tex` | 不出现 prefetch 全 0；若 risk 为 high，报告风险并避免用 MAPPO 弱点支撑主 claim | high risk 不一定阻断 final gate，但必须进入 limitation。 |
| P1 | `cache_offload_drl` blocker | paired reward statistics | formal/holdout vs `cache_offload_drl` CI 下界均 `> 0` | 若任一下界 `<=0`，final package 不 paper-ready。 |
| P1 | prediction claim boundary | prediction robustness report | learned/noisy predictor setting 对 learned baselines 为正 CI | `oracle_prediction` / `no_prediction` 仍只写 diagnostic。 |
| P2 | real model-cache backend | importer + independent benchmark profile | 有真实 request/event 或 file-size profile contract、adapter_id 映射、独立结果标签 | 未实现前不得声称 benchmark cache events 来自真实 model-cache trace。 |
| P2 | full vehicle/RSU MARL | 新 observation/action contract + retrained baselines | vehicle/RSU-level wrapper、训练、评估、benchmark 消费端全部冻结 | 未完成前 MAPPO/QMIX/Controller-MAT 只写 controller-level。 |

## 精确重跑命令

```bash
python scripts/run_top_journal_closed_loop.py --run_id top_journal_mechanism_v6_strong_competition_20260527_v1 --seeds 7 13 29 --sa_profile top_journal_mechanism_v6_strong_competition --mappo_baseline_profile mappo_strong_audit --baseline_agents ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --primary_vehicle_selection handoff_pressure
python scripts/run_top_journal_final_submission_loop.py --run_id final_submission_v6_mappo_v3_strong_20260527_v1 --base_manifest_path artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v6_strong_competition_20260527_v1/seed_checkpoint_manifest.json --force_retrain_learned --resume_training --resume_benchmark --resume_support --command_retries 2 --baseline_episodes 96 --baseline_update_every 6 --baseline_batch_size 32 --mappo_baseline_profile mappo_strong_audit --minimum_reward_delta 0.5 --holdout_offsets 3 --seeds 7 13 29
python scripts/build_top_journal_comparison_report.py --final_run_root artifacts/experiments/top_journal_final_submission/final_submission_v6_mappo_v3_strong_20260527_v1 --bootstrap_samples 5000
```

写作硬约束：上述三个命令全部完成且 final gate 为 `paper_claim_ready=true` 之前，论文只能引用 `final_submission_full_current_baselines_20260511_v1` 的数值结果；MAPPO v3 和 SA v6 只能作为 Methods / planned rerun 描述。
