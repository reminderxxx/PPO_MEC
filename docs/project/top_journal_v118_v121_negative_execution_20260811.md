# v118-v121 Planner Internalization Execution Record

- `reviewed_at`: 2026-08-11
- `literature_cutoff`: 2026-08-11
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `top_journal_v118_full_evaluation_20260811`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_execution`: `91d34a7`
- `evidence_level`: `E2_ARTIFACT_AUDITED` development negative package
- `verdict`: rejected; no new reward winner

## Question

Test whether conservative counterfactual-planner distillation can transfer the
execution-time v100 planner gain into the native factorized MAPPO policy without
changing the environment reward, action schema, evaluator filtering or window
plans.

## Fairness Repair

The optional transition ensemble previously consumed the global PyTorch RNG
before policy initialization. This made the historical `no_learned_dynamics`
retraining ablation causally invalid because its initial policy differed from
the full method. Model initialization now runs in a forked RNG stream, and a
regression test requires equal initial policy parameters with learned dynamics
enabled or disabled.

RNG-aligned v100 and v118 each used seeds `7,13,29`, 256 episodes, 32 updates,
batch size 64, the frozen v71 train/dev plans, 20 training windows, two Alibaba
workflows, 22 steps and zero reward offset. All comparisons used `latest.pt` at
update 32; no reward-based checkpoint selection was used.

## v118 Full Result

Artifacts:

- v100 training: `artifacts/experiments/top_journal_v100_rng_fixed_full_seed{7,13,29}_20260811/`
- v118 training: `artifacts/experiments/top_journal_v118_full_seed{7,13,29}_20260811/`
- dual-domain evaluation: `artifacts/experiments/top_journal_v118_full_evaluation_20260811/`

| Domain / policy | v118 | RNG-aligned v100 | Delta |
|---|---:|---:|---:|
| LuST future raw | -37.4842 | -35.2392 | -2.2450 |
| NGSIM formal raw | 16.4178 | 18.8258 | -2.4081 |
| LuST future planner | -26.0508 | -25.9508 | -0.1000 |
| NGSIM formal planner | 21.1950 | 21.2400 | -0.0450 |

The NGSIM raw-policy BCa 95% interval is `[-6.0801,-0.2678]`; LuST is
`[-9.5791,1.1937]`. v118 reduced mean delay and backhaul on LuST but degraded
handoff readiness and migration overhead. Planner execution masked much of the
native-policy loss but did not create a gain over v100.

Artifact integrity passed at
`artifacts/experiments/top_journal_v118_full_evaluation_20260811/audit_artifact_integrity/artifact_integrity_report.json`:
18 root files, 37 referenced files, zero missing references and zero JSON
errors. This raises the negative development package to E2; it does not make a
rejected candidate canonical or paper-ready.

The fixed update-8 audit also failed the cross-seed/cross-domain criterion:
LuST mean delta was `+0.3667`, concentrated entirely in seed 7, while NGSIM was
`-8.8065` because seed 7 suffered a large cross-domain reversal.

## v119-v121 Fail-Fast Probes

v119 added model-advantage and realized-GAE agreement; v120 increased the same
gated projection strength; v121 added a behavior-KL-constrained logit ranking
margin. Each used three seeds, 64 episodes, eight updates and matched v100
update-8 checkpoints.

- LuST raw-policy: v119, v120 and v121 each tied v100 on all 72 pairs.
- NGSIM raw-policy: each mean delta was `+0.0096`, with `1/119/0` wins/ties/losses.
- These probes changed training losses but did not create a material
  deterministic policy improvement, so none was promoted to full training.

## Root Cause And Boundary

Hard planner labels are vulnerable to model bias: v118 accepted positive model
advantage even when realized MAPPO advantage disagreed, producing rare but
large negative tail outcomes. Realized-advantage gating removed the tail but
made the auxiliary loss too weak to alter the deterministic factorized action
boundary. Increasing coefficients or adding a ranking margin did not solve the
contract-level issue.

The current PPO/MAPPO comparison is controller-level: MAPPO has three
factorized controller heads and a centralized critic, but both methods consume
the same global semantic state and emit one environment action. A large generic
MAPPO-over-PPO gap is therefore not expected. A scientifically stronger next
step requires a separately frozen vehicle/RSU-level observation/action
contract with decentralized actors and a centralized graph critic, not more
reward shaping or planner-label tuning.

## Claim Boundary

The canonical frozen v100 artifacts remain the verified reward-winner record;
this execution does not invalidate their previously audited baseline results.
It does invalidate any claim that v118-v121 improve native MAPPO learning or
widen the reward gap. Because formal/LuST outcomes were repeatedly consumed
during development, none of these profiles may be promoted using the same
splits; a future architecture candidate requires a newly frozen untouched
holdout.
