# v100 Formal Results Record

- `recorded_at`: 2026-08-08
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `candidate`: `top_journal_mechanism_v100_urgency_safe_resource_mappo`
- `git_commit`: `ac7491b`
- `evidence_level`: `E2_ARTIFACT_AUDITED` for the formal/mixed package; v100 hidden evidence is unavailable because the v98 hidden was already consumed.

## Algorithm

v100 is an urgency-safe resource-constrained MAPPO extension. Exact environment branch replay evaluates each legal action at horizons `1,3,6` and returns TD value, mechanism realization and discounted backhaul/migration cost targets. A state-conditioned horizon mixture is used in the MAPPO KL policy-improvement target and the execution-time policy-prior-constrained planner. Resource cost pressure is reduced only when observed handoff urgency increases, while branch-derived mechanism advantage rises with the same urgency. Environment reward, baseline input contract, evaluator filters and reward offset remain unchanged.

## Formal Full

Raw run:
`artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/benchmarks/formal_full_stratified/main_results_full_stratified_20260808_183339_526215/`

The run contains 1320 episode summaries: 11 algorithms, 20 windows, 2 workflows and 3 seeds. Mean total reward:

| agent | total reward | continuity | handoff ready | mechanism realization |
|---|---:|---:|---:|---:|
| SA-GHMAPPO v100 | 26.430 | 0.900 | 0.463 | 0.475 |
| Popularity heuristic | 26.082 | 0.900 | 0.463 | 0.475 |
| DQN | 21.602 | 0.900 | 0.308 | 0.317 |
| Dueling-DQN | 17.341 | 0.900 | 0.154 | 0.158 |
| PPO | 9.283 | 0.725 | 0.210 | 0.217 |
| MAPPO | 1.126 | 0.374 | 0.629 | 0.650 |
| DT handoff DRL | 11.180 | 0.693 | 0.361 | 0.375 |
| DAG offload DRL | 9.320 | 0.549 | 0.574 | 0.592 |
| QMIX | 9.184 | 0.549 | 0.574 | 0.592 |
| Cache offload DRL | 1.333 | 0.399 | 0.613 | 0.625 |
| Controller-MAT | 0.844 | 0.374 | 0.629 | 0.650 |

SA-GHMAPPO is the highest-reward method among all 11 evaluated algorithms. The advantage over Popularity is primarily a lower delay penalty (`13.840` vs `14.188`); system mechanism metrics are ties, so the claim is not universal superiority on every metric.

Window-outer hierarchical statistics:
`artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/statistics/formal_full_stratified/paired_statistics.json`

SA vs Popularity total reward delta is `+0.3475`, BCa 95% CI `[0.1550, 0.6008]`, wins/ties/losses `54/66/0`, Holm-adjusted sign-test p-value `0.0`. SA vs PPO is `+17.146917`, BCa `[12.734327, 21.403285]`; SA vs MAPPO is `+25.304083`, BCa `[14.789211, 39.681286]`.

## Mixed Informative

The same frozen protocol was rerun as `mixed_informative` and produced the same aggregate values. It is reported separately and is not merged with `full_stratified` as independent samples:
`artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/benchmarks/formal_mixed_informative/main_results_mixed_informative_20260808_184603_237845/`

## Prediction Support

Compact support run:
`artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/support/prediction_robustness_compact/prediction_robustness_20260808_214349_192721/`

It contains 5 outcome-blind frozen formal windows, 3 seeds, 2 agents and 4 prediction settings, for 240 episode summaries. SA-GHMAPPO reward was `89.312` under baseline prediction, `89.117` under noisy prediction and `87.855` with prediction disabled; Popularity was `88.152`, `87.937` and `86.760`. The SA reference-minus-no-prediction degradation was `1.457`, versus `1.392` for Popularity. This is support evidence, not an independent formal or hidden split; the attempted full noise sweep did not produce an artifact and is not claimed.

## Provenance

Checkpoint manifest:
`artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/seed_checkpoint_manifest.json`

Manifest SHA-256: `418657c69731b45cfca1df0b45ecc1757adcd555cbde5b27106bc4ef6f7b9d09`.

The v100 training summaries contain exactly 256 episodes and 32 updates for seeds `7,13,29`; all report `collapse_detected=false`. The v71 hidden holdout was consumed by v98 and was not reopened, so no v100 hidden claim is made.

Artifact integrity audit:
`artifacts/experiments/top_journal_v100_urgency_safe_resource_full_20260808/audit_artifact_integrity/artifact_integrity_report.json`

The audit passed with `root_file_count=3574`, `referenced_file_count=3665`, `missing_reference_count=0` and `json_error_count=0`; the SHA-256 inventory is in `audit_artifact_integrity/sha256_manifest.txt`.

## Limitations

The formal result is an audited replication on the NGSIM + Alibaba controller-level contract, not a new independent hidden holdout. v100 and v98 produce identical formal reward rows in this protocol, although v100 improves the pre-frozen dev comparison over v99 (`18.760` vs `18.211`). A second independent mobility/workflow combination, unified compute accounting and complete component ablations remain required before a TMC-ready verdict.
