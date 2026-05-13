# sa_reward_tiebreak_round4 Report

## Scope

This round did not modify `VecWorkflowCoreEnv`, the reward main formula, benchmark splits, or `popularity_cache_heuristic`.

The implemented changes are limited to:

- reward-tiebreak checkpoint selection;
- `sa_reward_tiebreak_round4` warm-start fine-tune profile;
- per-seed warm-start checkpoint handling in the training runner;
- diagnostic/report artifacts.

## Modified Files

- `scripts/train_sa_ghmappo_real_sample.py`
- `scripts/select_sa_reward_tiebreak_checkpoints.py`
- `configs/experiment/sa_reward_tiebreak_round4.yaml`
- `src/evaluators/main_results_support.py`
- `docs/agent/sa_reward_tiebreak_round4_diagnosis.md`
- `docs/agent/sa_reward_tiebreak_round4_report.md`

## New Artifacts

- Round2 selection sweep:
  - `artifacts/eval/sa_reward_tiebreak_round4_round2_selection_sweep/round2_update_sweep_rows.csv`
  - `artifacts/eval/sa_reward_tiebreak_round4_round2_selection_sweep/round2_update_sweep_summary.json`
- Round4 training:
  - `artifacts/training/sa_reward_tiebreak_round4/sa_ghmappo/sa_ghmappo_train_20260425_201038_981365_seed7/`
  - `artifacts/training/sa_reward_tiebreak_round4/sa_ghmappo/sa_ghmappo_train_20260425_201259_532650_seed13/`
  - `artifacts/training/sa_reward_tiebreak_round4/sa_ghmappo/sa_ghmappo_train_20260425_201513_279829_seed29/`
  - `artifacts/training/sa_reward_tiebreak_round4/sa_ghmappo/sa_ghmappo_train_20260425_202318_679926_seed7/`
  - `artifacts/training/sa_reward_tiebreak_round4/sa_ghmappo/sa_ghmappo_train_20260425_202503_935765_seed13/`
  - `artifacts/training/sa_reward_tiebreak_round4/sa_ghmappo/sa_ghmappo_train_20260425_202645_707398_seed29/`
- Final round4 manifest and selection:
  - `artifacts/training/sa_reward_tiebreak_round4/seed_checkpoint_manifest_sa_reward_tiebreak_round4_best_by_reward_tiebreak_score.json`
  - `artifacts/training/sa_reward_tiebreak_round4/reward_tiebreak_selection_summary.json`
- Final benchmark:
  - `artifacts/benchmarks/sa_reward_tiebreak_round4/mixed_informative/main_results_mixed_informative_20260425_202820_132071/`
  - `artifacts/benchmarks/sa_reward_tiebreak_round4/full_stratified/main_results_full_stratified_20260425_203004_132529/`
- Analysis export:
  - `artifacts/analysis/sa_reward_tiebreak_round4/comparison_summary.csv`
  - `artifacts/analysis/sa_reward_tiebreak_round4/comparison_summary.json`

## Checkpoint Selection Result

The round2 checkpoint sweep showed selection was not the source of the mixed gap:

| seed | selected update | best mixed update | mixed reward |
|---:|---|---|---:|
| 7 | update_0006 | update_0006 | 83.388333 |
| 13 | update_0008 | update_0008 | 83.438333 |
| 29 | update_0008 | update_0008 | 83.388333 |

Round4 final selector selected `warm_start.pt` for all seeds. That means no fine-tuned update improved the reward-tiebreak score without safety risk.

| seed | round4 selected source | selected update |
|---:|---|---:|
| 7 | warm_start.pt | 0 |
| 13 | warm_start.pt | 0 |
| 29 | warm_start.pt | 0 |

## Mixed Comparison

| run | SA reward | popularity reward | reward gap | SA continuity | popularity continuity | SA failure | popularity failure | SA backhaul | popularity backhaul |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round2 | 83.405000 | 83.513333 | -0.108333 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 124.444444 | 170.666667 |
| round4 final | 83.405000 | 83.513333 | -0.108333 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 124.444444 | 170.666667 |

Mixed reward did not improve. The final round4 manifest reproduces the round2 candidate because that was the safest selected checkpoint.

The first 32-episode round4 attempt was worse:

| run | SA reward | reward gap | SA backhaul | popularity backhaul |
|---|---:|---:|---:|---:|
| round4 first attempt mixed | 82.202222 | -1.311111 | 156.444444 | 170.666667 |

It was not used for the final manifest.

## Full Comparison

| run | SA reward | popularity reward | reward gap | SA continuity | popularity continuity | SA failure | popularity failure | SA backhaul | popularity backhaul |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round2 | 76.654815 | 75.492778 | +1.162037 | 0.956229 | 0.924242 | 0.097222 | 0.180556 | 147.555556 | 158.222222 |
| round4 final | 76.654815 | 75.492778 | +1.162037 | 0.956229 | 0.924242 | 0.097222 | 0.180556 | 147.555556 | 158.222222 |

Full-stratified four core advantages are preserved only because final round4 selected the warm-start checkpoints.

Ready/mechanism:

| run | SA ready | popularity ready | ready gap | SA mechanism | popularity mechanism | mechanism gap |
|---|---:|---:|---:|---:|---:|---:|
| round2 full | 0.212963 | 0.250000 | -0.037037 | 0.277778 | 0.277778 | 0.000000 |
| round4 final full | 0.212963 | 0.250000 | -0.037037 | 0.277778 | 0.277778 | 0.000000 |

## Diagnosis

The mixed gap remains localized to `mechanism_activating` windows:

| window_class | SA reward | popularity reward | reward gap | SA backhaul | popularity backhaul |
|---|---:|---:|---:|---:|---:|
| active_non_mechanism | 69.975000 | 69.975000 | 0.000000 | 64.000000 | 64.000000 |
| mechanism_activating | 90.120000 | 90.282500 | -0.162500 | 154.666667 | 224.000000 |

The exported benchmark fields show delay, continuity, failure, readiness, mechanism realization, and migration overhead tied on mixed, while SA backhaul is lower. The remaining `0.108333` aggregate reward gap is therefore not explained by the current aggregate fields. It likely comes from a small service/cache/action-mix difference inside mechanism windows that is not exposed as a formal benchmark column.

## Success Level

- Minimum success: not reached. Mixed gap stayed `-0.108333`, not within `-0.05`.
- Qualified success: not reached. Mixed reward did not tie or exceed popularity.
- Ideal success: not reached.

Round4 should not be frozen.

## Freeze Recommendation

Do not freeze round4.

Keep `sa_mechanism_policy_round2` as the current qualified candidate. Round4 is useful as a negative result: it shows selection and small warm-start fine-tune do not solve the mixed reward tie-break without either selecting the original round2 checkpoint or creating safety regressions.

## Next Minimal Step

The next round should add reward-breakdown export for formal benchmark rows or a per-step reward component diagnostic, then target the exact hidden component behind the `mechanism_activating` reward gap. Do not change the environment reward formula; expose the existing breakdown and use it to tune policy behavior.
