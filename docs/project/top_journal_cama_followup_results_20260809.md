# CAMA-MAPPO Follow-up Results

reviewed_at: `2026-08-09`
literature_cutoff: `2026-08-09` (project reference table; no new external literature claim in this round)
target_venue: `IEEE TMC`
artifact_run_id: `top_journal_v101_cama_full_20260808` through `top_journal_v108_strong_cama_native_probe32`
policy_version: `tmc_review_policy_v3_20260621`
git_commit: `182e369`
evidence_level: `E2_PARTIAL / development-only for v101-v108`

## Objective

This round tested whether a genuine MAPPO learning-side improvement could internalize the online counterfactual planner instead of relying on evaluator logic or reward rewriting. The retained method is CAMA-MAPPO: exact legal alternatives are enumerated for the slow, fast and event heads, a policy-marginal counterfactual advantage is computed, and a detached utility-weighted target distribution is added to the hierarchical PPO actor update.

The environment reward, action contract, baseline contract and evaluator filtering were unchanged. The online planner remains an agent-side model-based policy-improvement component and relabels the selected action statistics for on-policy training.

## Evidence

| Run | Role | Result | Decision |
|---|---|---:|---|
| `top_journal_v102_cama_policy_control_full_20260808` | native MAPPO, planner disabled | dev reward `18.4101` vs Popularity `21.1033` | reject as primary candidate |
| `top_journal_v103_cama_head_policy_full_20260808` | native MAPPO + head target distillation | dev reward `18.4093` vs Popularity `21.1033` | no measurable gain |
| `top_journal_v104_urgency_gated_cama_probe64` | urgency-gated CAMA | dev reward `18.339` vs Popularity `21.1033` | reject |
| `top_journal_v105_resource_gated_cama_probe64` | resource-gated CAMA | dev reward `18.339` vs Popularity `21.1033` | reject |
| `top_journal_v106_cama_planner_fusion_probe64` | CAMA + online planner fusion | no reward improvement over v100 on mechanism/active probes | not promoted |
| `top_journal_v107_cvar_runtime_probe` | lower-tail multi-horizon planner probe | `60.8` for both v100 and CVaR runtime action selection | no action-order change |
| `top_journal_v108_strong_cama_native_probe32` | strong native distillation stress test | mean reward `-15.038`, continuity `0.256` | negative result; removed |

The v100 agent-side planner remains the strongest verified candidate in this round. Its frozen NGSIM + Alibaba formal package reports SA-GHMAPPO `26.430` versus Popularity `26.082`, delta `+0.3475`, BCa 95% CI `[0.1550, 0.6008]`, paired `54/66/0`. The raw package is `artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/`.

The action-level diagnosis is consistent: v100 gains in active/non-mechanism windows by replacing some current-RSU executions with vehicle fallback, while mechanism-activating windows are often tied with Popularity. Therefore the small aggregate gap is a property of the current protocol and reachable action outcomes, not evidence that an evaluator wrapper fabricated a larger reward.

## Review

`top_journal_v100_urgency_safe_resource_full_20260808` has an artifact integrity pass and multi-seed formal statistics, but it has no untouched v100 hidden holdout. The historical hidden split was consumed by v98 and cannot be reused. Mechanism realization, continuity, handoff-ready and backhaul/migration outcomes tie Popularity. Full ablations, full noise sweep, cross-scene evidence and unified compute accounting remain incomplete.

Verdict: `Not TMC-ready; strong formal reward candidate, major revision`. The CAMA follow-ups are development evidence and do not justify a larger paper claim. The strong-distillation failure is retained as a boundary condition: increasing a counterfactual target coefficient is unsafe and can destroy mechanism-window continuity even without triggering the existing collapse flag.

## Next Valid Step

Freeze a new non-overlapping formal/hidden split before any further tuning, bind one candidate checkpoint manifest, and run the complete learned-baseline, heuristic-reference, mechanism-ablation, robustness and compute-cost package once. Do not reopen v71 hidden or promote any v101-v108 probe based on point estimates.
