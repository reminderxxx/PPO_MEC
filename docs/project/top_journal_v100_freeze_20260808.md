# v100 Candidate Freeze Record

- `freeze_date`: 2026-08-08
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `candidate`: `top_journal_mechanism_v100_urgency_safe_resource_mappo`
- `candidate_git_commit_before_freeze`: `9d9bc3f`
- `dev_artifact_run_id`: `main_results_full_stratified_20260808_173756_879061`
- `dev_artifact_root`: `artifacts/experiments/top_journal_v100_urgency_safe_resource_probe/dev_benchmark/main_results_full_stratified_20260808_173756_879061`
- `dev_protocol`: `configs/experiment/top_journal_v71_strict_split_20260730/dev_window_plan.json`
- `hidden_status`: v71 hidden was consumed by the earlier v98 candidate and is not reopened or used for v100 tuning.

## Candidate Selection

The candidate was selected using the pre-existing dev split only. v99 improved reward over Popularity by `+0.1290` but reduced mechanism realization from `0.4000` to `0.3250`; it was rejected. v100 improved mean reward from `18.0823` to `18.7598` (`+0.6775`), reduced delay penalty from `13.6037` to `12.9262`, and tied Popularity on continuity (`0.8444`), handoff-ready (`0.3250`), mechanism realization (`0.4000`), backhaul cost (`120.4`) and migration overhead (`0.1265`). These values are dev evidence only and are not formal claims.

After this record, the algorithm profile, checkpoint selection rule, baseline contract, reward offset, formal window plan and statistics implementation are frozen. No hidden output is available for v100, and no v100 result may be described as an independent hidden-holdout result.

## Frozen Algorithm

v100 is an urgency-safe resource-constrained MAPPO extension. The native MAPPO action policy remains the prior. For each legal environment action, exact branch replay produces multi-horizon TD targets, mechanism realization targets and discounted backhaul/migration costs. The policy-improvement target uses the same adaptive horizon weights for both training-time KL projection and execution-time counterfactual planning. The resource penalty is reduced only as the observed handoff urgency rises, while the branch-derived mechanism advantage increases with that urgency. The environment reward, baseline inputs, evaluator filtering and reward offset are unchanged.

## Formal Protocol

- Mobility/workflow: NGSIM + Alibaba.
- Seeds: `7,13,29`.
- Training: 256 episodes, update every 8 episodes, batch size 64.
- Evaluation: existing v71 `formal_window_plan.json`, 20 non-overlapping windows, `full_stratified`, `max_steps=22`, `reward_positive_offset=0.0`.
- Baselines: SA-GHMAPPO, PPO, MAPPO, DQN, Dueling-DQN, QMIX, Controller-MAT, DAG offload DRL, cache offload DRL, DT handoff DRL and Popularity cache heuristic.
- Statistics: window-outer hierarchical bootstrap with BCa and percentile 95% intervals, paired sign test and Holm correction.

This is a formal replication on an already audited protocol, not a new hidden holdout. The claim is limited to the frozen NGSIM + Alibaba controller contract and will remain `UNVERIFIED` until raw formal artifacts, checkpoint manifest and statistics are complete.
