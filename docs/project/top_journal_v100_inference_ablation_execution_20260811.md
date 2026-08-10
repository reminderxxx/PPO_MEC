# v100 Inference Ablation Execution Record

- `executed_at`: 2026-08-11
- `candidate_reference`: `top_journal_mechanism_v100_urgency_safe_resource_mappo`
- `ablation_role`: inference-side mechanism attribution, not retrained policy evidence
- `artifact_run_id`: `top_journal_v100_inference_ablation_lust_zero_offset_20260811/ablation_full_stratified_20260811_014709_410809`
- `full_reference_run_id`: `top_journal_v100_lust_future_validation_20260810/main_results_full_stratified_20260811_011340_308565`
- `plan`: `configs/experiment/top_journal_v100_lust_future_validation_20260810/future_validation_window_plan.json`
- `protocol`: same 12 LuST windows, 3 seeds, 2 workflows, 22 steps, zero reward offset

## Attribution Result

| policy | total reward | continuity | handoff ready | mechanism realization |
|---|---:|---:|---:|---:|
| full v100 | -25.638 | 0.201 | 0.333 | 0.333 |
| w/o online counterfactual planner | -27.723 | 0.201 | 0.333 | 0.333 |

Matched full-minus-ablation reward delta is `+2.084722`, hierarchical window-bootstrap BCa 95% CI `[+1.733256,+2.467439]`, paired `72/0/0`, sign-test p=`0.0`. The gain is consistent across all 12 outer windows.

This supports the algorithmic attribution that execution-time model-assisted MAPPO policy improvement contributes real reward under the LuST protocol. It is an inference-side ablation: it does not replace matched retraining ablations for every training loss component.

## Contract Audit

The first diagnostic run was invalid because the legacy ablation entry point applied its default positive reward offset of `5.0`. `scripts/benchmark_ablation.py` now defaults to zero, exposes `--reward_positive_offset`, records `reward_protocol`, and consumes a frozen plan without rescanning raw mobility. Only the corrected zero-offset run is reported above.
