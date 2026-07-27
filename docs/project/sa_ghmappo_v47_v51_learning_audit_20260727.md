# SA-GHMAPPO v47--v51 Policy-Learning Audit

reviewed_at: 2026-07-27
literature_cutoff: 2026-07-27
target_venue: IEEE Transactions on Mobile Computing
artifact_run_id: top_journal_v47_service_backhaul_dev_probe through top_journal_v51_cache_aided_prepare_full_dev
policy_version: tmc_review_policy_v3_20260621
git_commit: 5648620 (review started with additional uncommitted v47--v51 implementation files)
evidence_level: E1_DOCUMENTED
verdict: Unverifiable; not eligible for TMC-ready or paper-ready promotion

## Evidence inventory

- v47 and v48 train summaries / checkpoint records: artifacts/training/top_journal_v47_service_backhaul_dev_probe/ and artifacts/training/top_journal_v48_service_fill_dev_probe/.
- v49 and predictor-fix records: artifacts/training/top_journal_v49_retrospective_handoff_dev_probe/ and artifacts/training/top_journal_v49_retrospective_handoff_predictor_fix_dev_probe/.
- v51 physical-transfer records: artifacts/training/top_journal_v51_cache_aided_prepare_full_dev/sa_ghmappo/sa_ghmappo_train_20260722_151858_833691_seed7/.
- Source inspection: src/agents/sa_ghmappo_core.py, src/envs/core/predictor_manager.py, src/envs/core/vec_workflow_core_env.py, current profiles and contract tests.
- The required formal/holdout/support raw benchmarks, manifest audit, independent split, multi-seed statistics and command provenance are absent for these candidates.

## Findings

1. v47--v49 update evaluation is action-invariant: each of 16 updates reports identical reward and continuity. v47/v48 have no valid predicted handoff target on the evaluated windows, while v49 retains zero deterministic prepare/mechanism realization. These are not evidence that the new credit signals learned a better handoff policy.
2. v51 exposes physical-transfer opportunities, but its full-dev update evaluation is again identical at every update: reward 15.358, continuity 0.706148, handoff-ready rate 0.45, mechanism-realization rate 0.525. best_by_reward is update 1, while update 16 has the same reported metrics.
3. The v51 selected checkpoint diagnostics show deterministic prepare on every valid target (deterministic_event_prepare_rate_on_valid_target=1.0), mean event prepare probability 0.746491, mean event margin 5.998730, and guard_action_delta_rate=0.684751. The policy therefore contains a large rule/projection component that must be separated from raw MAPPO preference before claiming learned control.
4. The current code applies pseudo-target/logit adjustments and then may apply temporal smoothing, continuity, cache, prefetch, backhaul, idle fallback, or option-gate action replacement in act(). Those mechanisms can be legitimate safety constraints only when their contribution is independently reported; they cannot be folded into an unqualified “MAPPO policy” gain.

## Hard blockers

- No formal or independent holdout package; E2 evidence is unavailable.
- The current checkpoint-selection signal is action-invariant across updates. It cannot distinguish an actual learned-policy improvement from a fixed intervention.
- The current v51 update evaluation combines raw policy and runtime action changes; attribution of a potential gain to the learned actor is therefore unresolved.

## Major concerns

- v47--v49 share dev-only, seed-7 evidence and do not demonstrate executable online handoff candidates.
- v51 physical-transfer features improve candidate visibility but do not by themselves prove causal, non-oracle online prediction or MAPPO learning.
- The semantic_discrete_5 contract remains controller-level; no claim may describe this as full RSU-agent or vehicle-agent MAPPO.

## Scorecard

All dimensions: N/S because the candidates lack the required E2 evidence and learning-attribution gate. A numerical TMC score would be misleading.

## Safe claims

- v51 is an unpromoted dev-stage mechanism probe that adds a trajectory-boundary handoff feature and training-only retrospective labels.
- The audit found a reproducible policy-attribution risk requiring a raw-policy versus safety-projected ablation.
- Existing unit tests cover the current physical-transfer/action-contract behavior; they do not validate a paper-level performance claim.

## Prohibited claims

- “v47--v51 improves MAPPO”, “learned handoff policy”, “outperforms all baselines”, “Digital Twin is validated”, or “paper-ready/TMC-ready”.
- Any statement that deterministic prepare rate or a guard-selected action alone proves the mechanism is learned or beneficial.

## Required actions before re-review

Implement the Policy-Learning Gate in research_skill_integration_20260727.md; then run pre-registered multi-seed dev ablations and only promote a raw-policy winner to a new frozen formal/holdout protocol.

## Post-audit implementation follow-up (2026-07-27)

- The independent implementation task added the required gate. `raw_policy` bypasses policy logit adjustments and runtime guards while retaining the environment action mask; `safety_projected` remains a separately logged hybrid-system mode.
- Saved-update evaluation now records a stable raw action-signature digest. The first checkpoint is not selection-eligible; a run whose raw signatures and raw aggregate metrics remain invariant is blocked from automatic best/Pareto selection.
- Automatic checkpoint selection and checkpoint-consistency re-evaluation now use `raw_policy` metrics, so a safety-projected value cannot overwrite the recorded learned-policy result.
- A minimal real-data smoke verified this implementation path only. It does not change this audit's `E1_DOCUMENTED / Unverifiable` verdict, and it does not evaluate or promote the historical v47--v51 artifacts.

## TITS/TVT fit note

The current evidence is insufficient for a paper-level fit judgement at either venue. A TITS/TVT submission cannot be used to relax the same attribution, fairness, or independent-evidence requirements.
