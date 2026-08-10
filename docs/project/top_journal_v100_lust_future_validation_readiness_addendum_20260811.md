# v100 LuST Future-Validation Review Addendum

reviewed_at: 2026-08-11
literature_cutoff: 2026-08-11
target_venue: IEEE Transactions on Mobile Computing (TMC)
artifact_run_id: top_journal_v100_lust_future_validation_20260810/main_results_full_stratified_20260811_011340_308565
policy_version: tmc_review_policy_v3_20260621
git_commit: 2416321 (v100 checkpoint and prior audit provenance)
evidence_level: E2_ARTIFACT_AUDITED
verdict: stronger generalization evidence; still not TMC-ready

## Audit

- The LuST plan is outcome-blind, sealed, and independently excludes the historical LuST support windows with a 24-frame gap.
- The full package contains 792 raw summaries from 11 agents, 3 seeds, 12 windows and 2 workflows. Hierarchical outer-window statistics are recorded in the artifact root.
- SA-GHMAPPO remains first in total reward on LuST: `-25.638` versus Popularity `-32.961` and MAPPO `-35.535`. The reward lower bound is positive against all baselines.

## Claim Boundary

This closes a major cross-mobility evidence gap for the frozen checkpoint, but not the entire paper-readiness gap. The absolute LuST rewards are negative and the active/idle strata are hard for every method; the gain concentrates in the mechanism-activating stratum. The paper must report this trade-off, add matched component ablations and unified compute accounting, and must not claim universal all-regime superiority.
