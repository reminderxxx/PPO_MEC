# v98 Top-Journal Readiness Review

reviewed_at: 2026-08-08
literature_cutoff: 2026-08-08
target_venue: IEEE Transactions on Mobile Computing (TMC)
artifact_run_id: `top_journal_v98_ucc_full_20260808`
policy_version: `tmc_review_policy_v3_20260621`
git_commit: `0584a51`
evidence_level: `E2_ARTIFACT_AUDITED`
verdict: `Not TMC-ready; strong reward candidate, major revision`

## Evidence Inventory

- v98 full formal and independent hidden raw rows, aggregates and window-outer hierarchical statistics are present.
- The frozen checkpoint manifest covers SA-GHMAPPO plus PPO, MAPPO, DQN, Dueling-DQN, QMIX, Controller-MAT, DAG, cache and DT baselines, all with seeds `7,13,29`.
- Formal/hidden interval audit passed with 20 non-overlapping windows per split and no cross-split overlap.
- Artifact integrity audit passed for v97/v98: 6060 root files, 6128 referenced files, 0 missing references, 0 JSON errors.
- v97 calibration-only full three-seed ablation and v98 full result are both retained; v97 seed29 collapse is not removed.

## Hard Blockers

- None for the narrow claim that v98 improves `total_reward` on this NGSIM + Alibaba protocol. The broader mechanism or universal MARL claims remain blocked by the concerns below.

## Major Concerns

- The strongest rule baseline is extremely close: formal `+0.3475`, hidden `+0.2350`; continuity, handoff-ready and mechanism realization are ties with Popularity.
- Only one mobility/workflow combination and three seeds are covered. This is a major generalization limitation even though the hidden split is independent.
- Full 20-window system robustness, scalability, unified wall-clock/peak-memory/branch-cost accounting and complete component ablations are not in the v98 package; only a 5-window prediction support subset is complete.
- MAPPO, QMIX and Controller-MAT are controller-level contracts. The paper must not call them vehicle-level or RSU-level full MARL.

## Scorecard

- Novelty / nearest literature: `13/20`; one-step model-assisted improvement and uncertainty calibration are meaningful VEC integration, but generic model-based policy improvement and uncertainty penalties are established foundations. See the maintained literature table and the primary MBPO/MOPO/COMBO references.
- Technical correctness: `13/15`; action-conditioned reward/next-state/TD targets, action masks and MAPPO prior are contract-consistent and tested.
- Baseline fairness / independence: `13/15`; equal budget and independent windows are strong, while the single data combination limits external validity.
- Experiments / statistics / holdout: `15/20`; formal and consumed hidden both have positive BCa intervals, but support and compute package are incomplete.
- Mechanism realization / system metrics: `6/10`; the reward claim is supported, but the candidate ties rather than beats Popularity on several mechanism metrics.
- Robustness / generalization / scalability: `4/10`; prediction subset exists, full system/scalability evidence is missing.
- Reproducibility / claim completeness: `7/10`; manifest, hashes, raw rows and integrity audit exist, but a unified command/wall-clock package is incomplete.

Total: `71/100`.

## Safe Claims

- On the frozen NGSIM + Alibaba protocol, v98 SA-GHMAPPO has higher total reward than every evaluated baseline in both formal and independent hidden splits.
- The exact one-step policy-improvement layer is supported by the v97 calibration-only ablation: v97 is below Popularity while v98 is above it under the same full budget.
- The method is a controller-level uncertainty-calibrated, counterfactual model-assisted MAPPO extension for the project contract.

## Prohibited Claims

- Do not claim universal superiority, full vehicle/RSU-level MARL superiority, or superiority on every system/mechanism metric.
- Do not call the method a general learned world model, offline-MOPO/COMBO reproduction, or a theoretical safe-policy-improvement guarantee.
- Do not claim a large heuristic gap or complete robustness/scalability until the missing support package is executed.

## Required Actions Before Re-review

1. Add full formal/hidden-aligned system robustness and scalability under the frozen v98 manifest.
2. Add a unified compute audit: wall-clock, peak memory, branch/query count and training/evaluation budget per algorithm.
3. Complete no-calibration, no-uncertainty, no-policy-prior and no-exact-policy-improvement component ablations on the same frozen windows.
4. Add a second independent mobility/workflow combination or narrow the manuscript claim explicitly to NGSIM + Alibaba.

TITS/TVT fit note: the current artifact is a credible candidate result for a revision cycle, but it should not be presented as TMC-ready until the four actions above are complete.
