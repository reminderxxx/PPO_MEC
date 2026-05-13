# SA-GHMAPPO mechanism retention round3 diagnosis

Date: 2026-04-25

Scope: this diagnosis compares `sa_mechanism_policy_round2` against `sa_mechanism_retention_round3`. This round did not modify `VecWorkflowCoreEnv`, reward formula, benchmark split, or `popularity_cache_heuristic`.

## Inputs

- Round2 mixed benchmark: `artifacts/benchmarks/sa_mechanism_policy_round2/mixed_informative/main_results_mixed_informative_20260425_182051_736385/`
- Round2 full benchmark: `artifacts/benchmarks/sa_mechanism_policy_round2/full_stratified/main_results_full_stratified_20260425_182234_412726/`
- Round2 seed29 diagnostic CSV: `artifacts/training/sa_mechanism_policy_round2/seed29_mechanism_window_diagnostics_round2.csv`
- Round3 training root: `artifacts/training/sa_mechanism_retention_round3/`
- Round3 seed29 diagnostic CSV: `artifacts/training/sa_mechanism_retention_round3/seed29_mechanism_window_diagnostics_round3.csv`
- Round3 mixed benchmark: `artifacts/benchmarks/sa_mechanism_retention_round3/mixed_informative/main_results_mixed_informative_20260425_185741_066192/`
- Round3 full benchmark: `artifacts/benchmarks/sa_mechanism_retention_round3/full_stratified/main_results_full_stratified_20260425_185926_067433/`

## Late-Stage Decay

Round2 showed a clear drop after update 8:

| seed | selected update | event_prepare@8 | event_prepare@last | guard@8 | guard@last | aux_loss@8 | aux_loss@last | entropy@8 | entropy@last |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6 | 0.325558 | 0.200696 | 19 | 18 | 0.681594 | 1.226395 | 0.629650 | 0.486153 |
| 13 | 8 | 0.420230 | 0.222700 | 28 | 16 | 0.514276 | 1.009161 | 0.569363 | 0.501588 |
| 29 | 8 | 0.452759 | 0.206564 | 26 | 15 | 0.644130 | 1.004247 | 0.520547 | 0.452105 |

`mechanism_aux_loss_mean` did not go to zero, and mechanism-window samples were still present. The issue is not loss disappearance or sample absence. It is that the policy head still drifts away from prepare actions late in training.

Round3 activated retention from update 8 through update 16. The same summary:

| seed | selected update | event_prepare@8 | event_prepare@last | guard@8 | guard@last | aux_loss@8 | aux_loss@last | entropy@8 | entropy@last |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6 | 0.326518 | 0.232100 | 19 | 18 | 0.681029 | 1.086899 | 0.629763 | 0.520581 |
| 13 | 8 | 0.420373 | 0.206422 | 28 | 16 | 0.514229 | 0.857987 | 0.569330 | 0.539966 |
| 29 | 8 | 0.452912 | 0.212172 | 26 | 15 | 0.644056 | 0.931426 | 0.520493 | 0.509287 |

Retention slightly improved late entropy and reduced aux-loss growth, especially seed29 entropy `0.452105 -> 0.509287`, but it did not raise late prepare probability enough to change the selected benchmark checkpoint.

## Selected Checkpoints

`best_by_retained_mechanism_score.pt` selected:

| seed | selected source update |
|---:|---:|
| 7 | 6 |
| 13 | 8 |
| 29 | 8 |

This is effectively the same benchmark-facing checkpoint set as round2. Seed7 is still selected before retention starts, and seed13/seed29 are selected at the retention start. Later retained updates were not selected because they did not improve the reward/continuity/failure/backhaul/ready balance enough.

## Ready Gap Source

The full-stratified ready gap remains concentrated in mechanism windows, especially `j_3 / window_off250_len24_t297_320` for seeds 7 and 29. In round2 this row had SA ready `0.0` while popularity had ready `1.0`, and it also carried a handoff failure. Round3 did not change the selected checkpoint for these seeds in a way that improves this row.

The issue is not missing candidate/timing signal: seed29 still has nonzero `valid_handoff_target_step_count`, `timing_active_step_count`, and `gate_pass_step_count`. It is also not a benchmark split issue. The remaining gap is prepare/cache-ready realization on a small number of difficult mechanism rows.

## Interpretation

Round3 retention is mechanically active and conservative. It does not damage the benchmark, but it also does not produce a better selected checkpoint under the 64-episode budget. The result is a safe but non-improving change relative to round2.

Recommendation from diagnosis: do not freeze round3 as final. Keep round2 as the qualified candidate unless a follow-up run changes the policy retention enough to select later stable checkpoints without increasing failure or backhaul.

