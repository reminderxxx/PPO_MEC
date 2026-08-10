# v100 LuST Future Validation Execution Record

- `executed_at`: 2026-08-11
- `candidate`: `top_journal_mechanism_v100_urgency_safe_resource_mappo`
- `validation_split`: `future_validation`
- `artifact_run_id`: `top_journal_v100_lust_future_validation_20260810/main_results_full_stratified_20260811_011340_308565`
- `plan_manifest`: `configs/experiment/top_journal_v100_lust_future_validation_20260810/future_validation_manifest.json`
- `evidence_level`: `E2_ARTIFACT_AUDITED`
- `protocol`: LuST SUMO FCD + Alibaba, 11 agents, 3 seeds (`7,13,29`), 12 frozen windows, 2 workflows, 22 steps, zero reward offset

## Result

| agent | total reward | continuity | handoff ready | mechanism realization |
|---|---:|---:|---:|---:|
| SA-GHMAPPO | -25.638 | 0.201 | 0.333 | 0.333 |
| QMIX | -27.456 | 0.196 | 0.333 | 0.333 |
| Controller-MAT | -27.955 | 0.193 | 0.333 | 0.333 |
| Popularity | -32.961 | 0.189 | 0.244 | 0.333 |
| DAG-Offload | -32.270 | 0.201 | 0.271 | 0.319 |
| MAPPO | -35.535 | 0.201 | 0.222 | 0.222 |
| DQN | -35.760 | 0.198 | 0.222 | 0.222 |
| DT-Handoff | -36.467 | 0.198 | 0.212 | 0.222 |
| Dueling-DQN | -43.749 | 0.201 | 0.111 | 0.111 |
| PPO | -47.946 | 0.131 | 0.111 | 0.222 |
| Cache-Offload | -58.046 | 0.050 | 0.083 | 0.194 |

SA versus Popularity total-reward difference was `+7.3225`, hierarchical window-bootstrap BCa 95% CI `[+2.324167,+15.186198]`, paired `24/48/0`, Holm-adjusted sign-test p=`0.0`. SA versus MAPPO was `+9.896944`, BCa `[+3.993952,+20.822057]`, paired `72/0/0`.

## Strata and Boundary

The 12 windows were frozen as 5 mechanism-activating, 4 active-non-mechanism and 3 idle-or-sparse windows. SA's overall reward lead is real on this split, but the active-non-mechanism and idle-or-sparse strata are difficult for all agents and remain negative; the mechanism-stratum SA mean is `31.218` versus Popularity `13.644`. This supports a mechanism-specific reward lead, not universal superiority across every operating regime.

The split plan excludes the historical LuST support windows with minimum frame gap 24 and has `outcome_blind_selection=true`, `independence_audit.passed=true`, and zero conflicts. It is not pooled with NGSIM rows; mobility is treated as the outer cluster.
