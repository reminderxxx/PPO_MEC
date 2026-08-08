# v100 Top-Journal Readiness Review

reviewed_at: 2026-08-08
literature_cutoff: 2026-08-08
target_venue: IEEE Transactions on Mobile Computing (TMC)
artifact_run_id: top_journal_v100_urgency_safe_resource_full_20260808
policy_version: tmc_review_policy_v3_20260621
git_commit: ac7491b (candidate freeze; final documentation commit is recorded in Git history)
evidence_level: E2_ARTIFACT_AUDITED
verdict: Not TMC-ready; strong formal reward candidate, major revision

## Evidence Inventory

- Formal full: 11 agents, 20 frozen windows, 2 workflows, 3 seeds, 1320 raw summaries.
- Mixed informative: same frozen windows, reported separately and not pooled as independent evidence.
- Training: seeds `7,13,29`, 256 episodes, 32 updates, all `collapse_detected=false`.
- Support: 5 outcome-blind frozen windows, 4 prediction settings, 240 raw summaries.
- Integrity: `artifact_integrity_report.json` passed with 3574 root files, 3665 referenced files, zero missing references and zero JSON errors; SHA-256 inventory is present.
- Hidden: no v100 hidden run. The v71 hidden plan was consumed by v98 and was not reopened.

## Hard Blockers

- No independent hidden holdout for the frozen v100 checkpoint, so the new candidate has no untouched generalization evidence.
- The main result uses one NGSIM + Alibaba controller-level combination; cross-dataset and cross-workflow generalization is untested.
- Mechanism realization, continuity, handoff-ready ratio and backhaul/migration outcomes tie the Popularity heuristic on formal full. The reward win is mainly lower delay penalty, so it is not universal mechanism superiority.
- Complete component ablations and unified wall-clock/compute accounting are still absent.

## Major Concerns

- SA-GHMAPPO v100 is first in formal total reward (`26.430`) and exceeds Popularity (`26.082`) by `+0.3475`, but the gap is small.
- v100 formal reward rows are identical to v98 under this frozen protocol. The v100 dev uplift over v99 is real dev evidence, but it does not establish a new formal uplift.
- MAPPO remains far below the proposed method (`1.126` vs `26.430`), making the fairness and contract interpretation of this controller-level comparison important to explain.

## Scorecard

| dimension | score | evidence and deduction |
|---|---:|---|
| Novelty and literature positioning | 15/20 | urgency-conditioned multi-horizon resource-constrained MAPPO is a concrete distinction; nearest-neighbor matrix and independent validation are incomplete |
| Technical correctness and modeling | 12/15 | exact legal-action branch targets, MAPPO prior and resource/mechanism targets are implemented and checkpointed; complexity/guarantee analysis is missing |
| Baseline fairness and independence | 11/15 | all 11 methods use the frozen controller contract and 3 seeds; learned-baseline contract and compute parity need fuller audit |
| Experiments, statistics and holdout | 14/20 | formal BCa, paired sign test and Holm correction are present; no v100 hidden and only one data combination |
| Mechanism realization and system metrics | 6/10 | metrics are observable and support is present, but formal mechanism metrics tie the heuristic |
| Robustness, generalization and scalability | 5/10 | compact prediction support is positive; full noise, second scenario and scalability are missing |
| Reproducibility and claim completeness | 8/10 | manifest, checkpoints, raw rows, statistics and integrity audit exist; compute log and complete ablation package are incomplete |
| **Total** | **71/100** | **Not TMC-ready** |

## Safe Claims

- Under the frozen NGSIM + Alibaba controller-level protocol, SA-GHMAPPO v100 ranks first in mean total reward among the 11 evaluated methods.
- Against Popularity, the formal reward delta is `+0.3475`, BCa 95% CI `[0.1550, 0.6008]`, paired wins/ties/losses `54/66/0`, with Holm-adjusted sign-test p=`0.0`.
- The algorithmic contribution is an urgency-safe, resource-constrained MAPPO policy-improvement mechanism using exact multi-horizon legal-action branch targets; environment reward and evaluator filtering were unchanged.
- Compact prediction support shows positive SA reward margins under baseline, noisy, no-prediction and oracle settings, but it is not a generalization proof.

## Prohibited Claims

- Do not claim v100 has a new independent hidden-holdout win, universal mechanism superiority, or superiority on all system metrics.
- Do not claim generalization to all mobility/workflow datasets, full vehicle/RSU-level MARL, real deployment, or TMC acceptance.
- Do not claim v100 itself improved the frozen formal rows over v98; those rows are identical.

## Required Actions Before Re-review

1. Freeze v100 and open one genuinely independent, non-overlapping hidden split, then report raw rows and window-outer statistics without retuning.
2. Add at least one independent mobility/workflow combination and a second system-scale setting.
3. Run component ablations for multi-horizon targets, urgency weighting, resource penalty and planner coupling, with the same seeds and budget.
4. Add wall-clock, parameter, memory and inference-cost accounting for every learned baseline and the proposed planner.
5. Report the mechanism trade-off explicitly and test whether the delay gain survives when mechanism metrics are constrained or primary.

TITS/TVT fit note: the work may be a conditional candidate for a systems-oriented venue after the same blockers are addressed; this does not lower the TMC review bar.
