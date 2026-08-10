# v100 LuST External Support Execution Record

- `executed_at`: 2026-08-10
- `candidate`: `top_journal_mechanism_v100_urgency_safe_resource_mappo`
- `evidence_role`: external support only, not formal or hidden
- `artifact_run_id`: `top_journal_v100_lust_external_support_20260810/main_results_full_stratified_20260810_183709_845522`
- `mobility`: LuST SUMO FCD at `data/processed/mobility/lust/lust_fcd.csv`
- `protocol`: 11 agents, 3 seeds (`7,13,29`), 4 windows, 2 workflows, 22 steps, zero reward offset
- `plan_provenance`: historical external 4-window plan from the v8 LuST grid support artifact; its metadata is `outcome_blind_selection=false`

## Result

| agent | total reward | continuity | handoff ready | mechanism realization |
|---|---:|---:|---:|---:|
| SA-GHMAPPO | 34.200 | 1.000 | 1.000 | 1.000 |
| MAPPO | 28.946 | 0.996 | 0.854 | 0.875 |
| Popularity | 27.215 | 1.000 | 0.750 | 1.000 |
| DAG-Offload | 23.606 | 0.998 | 0.542 | 0.833 |
| DQN | 21.683 | 0.932 | 0.667 | 0.667 |
| Dueling-DQN | 18.892 | 1.000 | 0.333 | 0.333 |
| QMIX | 18.492 | 0.758 | 1.000 | 1.000 |
| Controller-MAT | 11.042 | 0.758 | 0.667 | 0.667 |
| DT-Handoff | 9.644 | 0.758 | 0.667 | 0.667 |
| PPO | -8.633 | 0.545 | 0.333 | 0.500 |
| Cache-Offload | -51.125 | 0.000 | 0.000 | 0.000 |

SA versus Popularity reward delta was `+6.985`, hierarchical window-bootstrap BCa 95% CI `[+0.900,+18.780]`, paired `24/0/0`. SA versus MAPPO delta was `+5.254`, BCa `[+2.940,+9.860]`, paired `24/0/0`.

## Evidence Boundary

The direction is consistent with NGSIM: the frozen v100 candidate wins total reward over Popularity and MAPPO on LuST. However, four outer windows, a historical plan with `outcome_blind_selection=false`, and one Alibaba workflow source are insufficient for an independent cross-city claim. This package is support evidence only; a new outcome-blind LuST plan with at least 12 independent outer windows is required before using LuST as a primary generalization result.
