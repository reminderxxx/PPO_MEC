# SA-GHMAPPO mechanism retention round3 report

Date: 2026-04-25

This round only changed SA-GHMAPPO training-side retention and checkpoint selection. It did not modify `VecWorkflowCoreEnv`, the reward formula, benchmark split, or `popularity_cache_heuristic`.

## Changed Files

- `src/agents/sa_ghmappo_core.py`
- `src/agents/sa_ghmappo_agent.py`
- `scripts/train_sa_ghmappo_real_sample.py`
- `configs/experiment/sa_mechanism_retention_round3.yaml`
- `docs/agent/sa_mechanism_retention_round3_diagnosis.md`
- `docs/agent/sa_mechanism_retention_round3_report.md`

## New Artifacts

- Smoke: `artifacts/training/sa_mechanism_retention_round3_smoke/`
- Training:
  - `artifacts/training/sa_mechanism_retention_round3/sa_ghmappo/sa_ghmappo_train_20260425_184717_670478_seed7/`
  - `artifacts/training/sa_mechanism_retention_round3/sa_ghmappo/sa_ghmappo_train_20260425_185050_681789_seed13/`
  - `artifacts/training/sa_mechanism_retention_round3/sa_ghmappo/sa_ghmappo_train_20260425_185417_979680_seed29/`
- Manifest: `artifacts/training/sa_mechanism_retention_round3/seed_checkpoint_manifest_sa_mechanism_retention_round3_best_by_retained_mechanism_score.json`
- Seed29 diagnosis: `artifacts/training/sa_mechanism_retention_round3/seed29_mechanism_window_diagnostics_round3.csv`
- Mixed benchmark: `artifacts/benchmarks/sa_mechanism_retention_round3/mixed_informative/main_results_mixed_informative_20260425_185741_066192/`
- Full benchmark: `artifacts/benchmarks/sa_mechanism_retention_round3/full_stratified/main_results_full_stratified_20260425_185926_067433/`

## Round2 vs Round3 Mixed

| metric | round2 SA | round3 SA | popularity | round3 SA-pop |
|---|---:|---:|---:|---:|
| total_reward | 83.405000 | 83.405000 | 83.513333 | -0.108333 |
| workflow_continuity_rate | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| handoff_failure_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| backhaul_traffic_cost | 124.444444 | 124.444444 | 170.666667 | -46.222223 |
| handoff_ready_ratio | 0.416667 | 0.416667 | 0.416667 | 0.000000 |
| mechanism_realization_rate | 0.500000 | 0.500000 | 0.500000 | 0.000000 |

Mixed reward gap did not improve, but it also did not expand. Backhaul advantage is unchanged.

## Round2 vs Round3 Full

| metric | round2 SA | round3 SA | popularity | round3 SA-pop |
|---|---:|---:|---:|---:|
| total_reward | 76.654815 | 76.654815 | 75.492778 | 1.162037 |
| workflow_continuity_rate | 0.956229 | 0.956229 | 0.924242 | 0.031987 |
| handoff_failure_rate | 0.097222 | 0.097222 | 0.180556 | -0.083334 |
| backhaul_traffic_cost | 147.555556 | 147.555556 | 158.222222 | -10.666666 |
| handoff_ready_ratio | 0.212963 | 0.212963 | 0.250000 | -0.037037 |
| mechanism_realization_rate | 0.277778 | 0.277778 | 0.277778 | 0.000000 |

Full-stratified four core advantages are preserved. The ready gap is unchanged, and mechanism realization remains tied with popularity.

## Seed29 Mechanism Window

| signal | round2 selected update8 | round3 selected update8 | round3 latest update16 |
|---|---:|---:|---:|
| event_prepare_prob_mean | 0.452759 | 0.452912 | 0.212172 |
| guard_prefetch_to_prepare_count | 26 | 26 | 15 |
| late-stage mechanism_aux_loss_mean | 1.004247 | 0.931426 | 0.931426 |
| mechanism_head_entropy latest | 0.452105 | 0.509287 | 0.509287 |

Round3 retained more entropy late and slightly reduced late aux-loss growth, but the late prepare probability still decayed. The selected checkpoint remained update 8, so benchmark results are unchanged.

## Side Effects

- Backhaul did not increase relative to round2.
- Handoff failure did not worsen relative to round2.
- Continuity did not decline relative to round2.
- Mixed reward did not decline relative to round2.

The retention change is safe in this run, but not useful enough to improve the formal benchmark.

## Success Level

- Minimum success: reached. Full four advantages are preserved; full ready gap and mixed reward gap did not expand.
- Qualified success: not reached. Full ready gap remains `0.037037`, above the `0.02` target, and mixed reward gap remains `0.108333`, above the `0.05` target.
- Ideal success: not reached.

## Freeze Recommendation

Do not freeze round3 as final. Keep round2 as the current qualified candidate.

Round3 should be kept as a safe diagnostic branch/artifact, but it does not justify replacing round2 because the selected checkpoints and benchmark metrics are effectively unchanged.

Next minimal step, if continuing: adjust retention to affect the selected-safe window after update 8 more directly, for example by a slightly stronger late event-head prior only when `timing_active + valid_handoff_target + handoff_ready=false`, or run a separately labeled 96-episode continuation without overwriting the 64-episode artifacts.

