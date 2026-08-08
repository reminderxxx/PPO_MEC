# v98 Candidate Freeze Record

- `freeze_date`: 2026-08-08
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `candidate`: `top_journal_mechanism_v98_ucc_counterfactual_policy_improvement_mappo`
- `candidate_git_commit_before_freeze`: `1c402a5`
- `candidate_manifest`: `artifacts/experiments/top_journal_v98_ucc_full_20260808/seed_checkpoint_manifest.json`
- `candidate_manifest_sha256`: `5aa3e7c4e5fd7c7b49994d5509e9a9efbdbdd2a2bed46efef16cf1452549358f`
- `training_root`: `artifacts/experiments/top_journal_v98_ucc_full_20260808/training/sa/sa_ghmappo/`
- `formal_run_id`: `main_results_full_stratified_20260808_152756_948018`
- `mixed_run_id`: `main_results_mixed_informative_20260808_153526_884824`
- `formal_rows`: `artifacts/experiments/top_journal_v98_ucc_full_20260808/benchmarks/formal_full_stratified/main_results_full_stratified_20260808_152756_948018/benchmark_rows.csv`
- `mixed_rows`: `artifacts/experiments/top_journal_v98_ucc_full_20260808/benchmarks/formal_mixed_informative/main_results_mixed_informative_20260808_153526_884824/benchmark_rows.csv`
- `formal_statistics`: `artifacts/experiments/top_journal_v98_ucc_full_20260808/statistics/formal_full_stratified/paired_statistics.json`
- `formal_statistics_sha256`: `ce1fa939e97c2c3eacb66c1b47d7ec4af635c3fa1a3ad75a668659de97a80bd8`
- `statistics_script`: `scripts/analyze_top_journal_statistics.py`
- `statistics_script_sha256`: `87d1611422a1ebb6fafe48fa1a3eda98b94a5fcad81f798b6c011d594018b6fe`
- `hidden_plan`: `configs/experiment/top_journal_v71_strict_split_20260730/hidden_holdout_window_plan.json`
- `hidden_plan_sha256`: `5b1bef10a181ff7aea90da3bd6b04e36975d12bb9b5baf33508148ea03e00cbd`

## Frozen Claim

The primary claim under one-time hidden evaluation is: on the pre-specified NGSIM + Alibaba protocol, v98 SA-GHMAPPO improves `total_reward` over `popularity_cache_heuristic` under the same three-seed checkpoint manifest and evaluation contract. The mechanism claim is limited to the implemented MAPPO extension: uncertainty-calibrated bootstrap transition targets, exact one-step counterfactual transition calibration, and exact one-step policy improvement using the same environment transition contract.

The claim does not include reward-offset gains, evaluation-only guard gains, universal deployment superiority, or superiority on every mechanism metric. Negative v95/v96 probes and the v97 calibration-only full ablation remain part of the evidence package.

## Frozen Protocol

- Mobility/workflow: NGSIM + Alibaba; `primary_vehicle_selection=handoff_pressure`.
- Seeds: `7, 13, 29`; training budget: 256 episodes, update every 8 episodes, batch size 64.
- Evaluation: formal and hidden use the frozen split plans; `full_stratified` and `mixed_informative` are reported separately and are not merged as independent samples.
- Statistics: `window_id` outer cluster, `seed workflow_id` inner cluster, BCa and percentile 95% intervals, paired sign test and Holm correction.
- Hidden rule: after this record is committed, no checkpoint, hyperparameter, window selection, baseline contract, claim wording, or statistics implementation may be changed in response to hidden output. The hidden output is consumed once and is reported even if negative.

