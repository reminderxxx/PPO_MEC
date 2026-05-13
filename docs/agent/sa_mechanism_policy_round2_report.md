# SA-GHMAPPO mechanism policy round2 report

Date: 2026-04-25

This round changed only SA-GHMAPPO policy-side training behavior, training profile parameters, diagnostics, and checkpoint labels. It did not modify `VecWorkflowCoreEnv`, the reward formula, benchmark split, or `popularity_cache_heuristic`.

## Changed files

- `src/agents/sa_ghmappo_core.py`
- `src/agents/sa_ghmappo_agent.py`
- `scripts/train_sa_ghmappo_real_sample.py`
- `configs/experiment/sa_mechanism_policy_round2.yaml`
- `docs/agent/sa_mechanism_policy_round2_diagnosis.md`
- `docs/agent/sa_mechanism_policy_round2_report.md`

## New training artifacts

- `artifacts/training/sa_mechanism_policy_round2/sa_ghmappo/sa_ghmappo_train_20260425_141124_319819_seed7/`
- `artifacts/training/sa_mechanism_policy_round2/sa_ghmappo/sa_ghmappo_train_20260425_141500_568510_seed13/`
- `artifacts/training/sa_mechanism_policy_round2/sa_ghmappo/sa_ghmappo_train_20260425_181700_754342_seed29/`
- `artifacts/training/sa_mechanism_policy_round2/seed_checkpoint_manifest_sa_mechanism_policy_round2_best_by_round2_mechanism_score.json`
- `artifacts/training/sa_mechanism_policy_round2/seed29_mechanism_window_diagnostics_round2.csv`

Selected checkpoints:

| seed | selected checkpoint | source update |
|---:|---|---:|
| 7 | `best_by_round2_mechanism_score.pt` | 6 |
| 13 | `best_by_round2_mechanism_score.pt` | 8 |
| 29 | `best_by_round2_mechanism_score.pt` | 8 |

## New benchmark artifacts

- Mixed: `artifacts/benchmarks/sa_mechanism_policy_round2/mixed_informative/main_results_mixed_informative_20260425_182051_736385/`
- Full: `artifacts/benchmarks/sa_mechanism_policy_round2/full_stratified/main_results_full_stratified_20260425_182234_412726/`

Each directory contains:

- `aggregate_summary.json`
- `benchmark_rows.csv`
- `comparison_against_popularity.json`
- `sa_advantage_diagnosis.json`

## Mixed informative comparison

| metric | round1 V2 SA | round2 SA | popularity | round2 delta SA-pop |
|---|---:|---:|---:|---:|
| total_reward | 80.872222 | 83.405000 | 83.513333 | -0.108333 |
| workflow_continuity_rate | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| handoff_failure_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| backhaul_traffic_cost | 119.111111 | 124.444444 | 170.666667 | -46.222223 |
| handoff_ready_ratio | 0.361111 | 0.416667 | 0.416667 | 0.000000 |
| mechanism_realization_rate | 0.500000 | 0.500000 | 0.500000 | 0.000000 |
| adapter_state_migration_overhead | 0.544444 | 0.453333 | 0.453333 | 0.000000 |

Mixed reward gap shrank from `-2.641111` to `-0.108333`. Continuity, failure, ready, mechanism realization, and migration overhead now tie popularity, while SA keeps a large backhaul advantage.

By window class, the remaining mixed reward gap is concentrated in mechanism windows: SA mechanism-window reward is `90.120000` versus popularity `90.282500`. Active non-mechanism windows tie exactly.

## Full stratified comparison

| metric | round1 V2 SA | round2 SA | popularity | round2 delta SA-pop |
|---|---:|---:|---:|---:|
| total_reward | 75.525000 | 76.654815 | 75.492778 | 1.162037 |
| workflow_continuity_rate | 0.979938 | 0.956229 | 0.924242 | 0.031987 |
| handoff_failure_rate | 0.037037 | 0.097222 | 0.180556 | -0.083334 |
| backhaul_traffic_cost | 153.777778 | 147.555556 | 158.222222 | -10.666666 |
| handoff_ready_ratio | 0.175926 | 0.212963 | 0.250000 | -0.037037 |
| mechanism_realization_rate | 0.259259 | 0.277778 | 0.277778 | 0.000000 |
| adapter_state_migration_overhead | 1.216667 | 1.189630 | 1.226667 | -0.037037 |

Full stratified keeps all four required advantages:

- reward stays above popularity by `1.162037`;
- continuity stays above popularity by `0.031987`;
- handoff failure stays lower by `0.083334`;
- backhaul stays lower by `10.666666`.

The ready gap shrank from `0.074074` to `0.037037`, a 50% reduction. Mechanism realization gap shrank from `0.018519` to `0.000000`.

By window class, round2 SA mechanism windows have ready `0.638889` and mechanism realization `0.833333`; popularity has ready `0.750000` and mechanism realization `0.833333`. SA's aggregate full reward advantage is reinforced by much better idle/sparse continuity and failure behavior.

## Seed29 mechanism window

Round1 V2 selected seed29 update15:

- `event_prepare_prob_mean=0.139703`
- `guard_prefetch_to_prepare_count=1`
- `migration_prepare_rate=0.062500`
- `handoff_ready_ratio=0.187500`
- `mechanism_realization_rate=0.375000`

Round2 selected seed29 update8:

- `event_prepare_prob_mean=0.452759`
- `guard_prefetch_to_prepare_count=26`
- `migration_prepare_rate=0.357639`
- `handoff_ready_ratio=0.312500`
- `mechanism_realization_rate=0.375000`

The main improvement is not candidate generation or checkpoint selection. It is stronger policy probability on mechanism actions when the candidate and timing signals already exist.

## Side effects

- Backhaul did not become abnormal: mixed SA is still `46.222223` lower than popularity, full SA is `10.666666` lower.
- Full handoff failure worsened relative to round1 V2 (`0.037037` to `0.097222`), but remains clearly below popularity (`0.180556`).
- Full continuity declined relative to round1 V2 (`0.979938` to `0.956229`), but remains above popularity (`0.924242`).
- Mixed reward did not reverse, but it improved sharply and is now nearly tied.

## Success level

- Minimum success: reached.
- Qualified success: reached. Full four advantages are preserved, ready gap shrank by 50%, mechanism gap closed, and mixed reward gap shrank.
- Ideal success: not reached. Full ready still trails popularity, and mixed reward is still slightly lower.

## Freeze recommendation

Recommendation: freeze as `minimum paper candidate` / `qualified mechanism candidate`, not as final mechanism paper candidate.

The version is strong enough to preserve full-stratified advantage and materially fix the mechanism realization gap. The next minimal step before a final freeze is late-training retention: keep the round2 mechanism auxiliary signal effective beyond update 8 without increasing handoff failure or backhaul.

