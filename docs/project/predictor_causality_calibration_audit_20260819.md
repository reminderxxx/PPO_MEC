# G12 Predictor Causality and Calibration Baseline Audit

- `reviewed_at`: `2026-08-19`
- `literature_cutoff`: `2026-08-19`
- `target_venue`: IEEE TMC evidence support; this is not a paper-readiness review
- `artifact_run_id`: `g12_predictor_causality_baseline_audit_20260819`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_reviewed`: `3248c4bfe6a608524823eaa62069a3e322d748c1`
- `evidence_level`: `E2_SOURCE_AND_EXISTING_DEV_ARTIFACT_AUDITED`

## Outcome

The existing supervised handoff predictor is a useful development checkpoint, but it does not satisfy a causal calibrated snapshot contract. The sample builder uses future mobility only for labels and resets position history per selected window, while runtime features are built from current state and previous positions. However, the old two-way train/dev protocol uses dev both for threshold selection and reported evaluation, does not independently fit or persist a probability calibrator, and does not validate raw frame/time interval separation inside the predictor training entry. The runtime exposes no immutable `generated_at/as_of/valid_until/consumed_at` snapshot identity. Its delayed mode computes a new current prediction before returning an older prediction payload, so delay is not yet a frozen slow-snapshot contract.

The existing v112 result therefore remains development evidence. Its high next-RSU accuracy is not evidence of handoff calibration or policy benefit, and the supervised predictor remains disabled from canonical profiles.

## Input-time audit

`build_feature_vector()` consumes these pre-action values:

| Feature | Source time | Causal status |
|---|---|---|
| vehicle `position_x`, `position_y`, `speed` | current observation | available at `observation_as_of` |
| `delta_x`, `delta_y` | current position minus the same vehicle's previous observed position | causal if history is reset per independent window/episode |
| current RSU availability/identity | association computed from current observed vehicle position and current RSU layout | current-observation feature |
| workflow node availability, completed/execution counts | current pre-action workflow state | current-observation feature |
| current required-adapter cache readiness | current pre-action node plus current RSU cache | current-observation feature |
| per-RSU relative position, distance, coverage radius | current vehicle and current static/dynamic RSU state | current-observation feature |
| per-RSU active vehicle count | current pre-action RSU state | current-observation feature |
| per-RSU required-adapter readiness | current pre-action cache state | current-observation feature |

No reward, selected action, service result, benchmark result, future label or oracle action is passed to `build_feature_vector()`. The checkpoint uses a deterministic hand-written scale/clamp transform; there is no learned scaler or normalizer fit. The saved `feature_names` are only positional placeholders (`feature_0...`) rather than semantic names, so feature order is recoverable only from source code plus schema version.

## Label and boundary audit

- `next_rsu_label` is the same vehicle's association at `t+1`.
- `handoff_within_horizon` and `handoff_target_label` inspect at most `horizon` future frames and identify the first association different from the current RSU.
- `handoff_eta_steps` is the first such offset; no-handoff samples use `horizon+1` for training but ETA metrics filter to positive handoff samples.
- The historical checkpoint uses `horizon=3`.
- Future labels are constructed from `frames` loaded for one selected window and never index beyond that window. Position history is initialized inside each window loop, so it does not cross a window boundary.
- The sample builder does not explicitly mark tail labels as censored. Near the end of a window it pads missing future frames with `None`, which may turn insufficient future coverage into a no-handoff label. G12 must report this censoring boundary instead of treating it as a verified negative.
- The builder has no fail-fast interval audit of its own. It trusts plan files and window IDs. Different IDs alone are insufficient evidence; raw `frame_offset`, `time_index`, and segment intervals must be checked before fitting.
- A vehicle identifier can recur in temporally separated windows. Existing strict plans provide interval separation, but the training entry does not group-split by vehicle or report cross-split vehicle overlap. This is a documented residual dependence risk, not proof of leakage.

## Fit, threshold and checkpoint audit

- Network weights are fit only on the named train plan.
- There is no learned scaler. Fixed source-code normalization/clamping is applied equally at train/runtime.
- The old threshold is selected on dev labels to maximize handoff F1, and the same dev rows are then reported as evaluation. This is selection/evaluation reuse.
- Threshold selection is classification-oriented, not calibration-oriented, and does not use RL reward.
- Checkpoint v1 saves schema versions, input dimension, placeholder feature order, RSU ID order/none index, horizon, threshold metadata, target-selection mode, weights, metrics and run identity.
- It does not save a three-way split manifest, semantic feature names, source interval audit, normalization artifact hash, probability-calibration method/parameters, reliability bins, ECE definition, staleness policy, abstention rule or snapshot contract identity.

## Raw outputs and confidence audit

- Next-RSU raw output: multiclass logits over checkpoint RSU slots plus a `none` class; runtime applies softmax and exposes only the winning probability in the legacy score payload.
- Handoff-target raw output: a separate multiclass logit vector over the same classes; runtime applies softmax, rejects a target equal to current RSU, and may hard-gate it by the binary threshold.
- Handoff raw output: one binary logit; runtime applies sigmoid.
- ETA raw output: `softplus(linear)+1`, a point estimate only. No interval or calibrated uncertainty is identified.
- Legacy `prediction_confidence_by_vehicle` for supervised mode is the binary handoff probability. Legacy uncertainty is `1-p`; this is not predictive entropy, epistemic uncertainty, ETA uncertainty or calibrated correctness probability.
- Baseline predictor confidence is a hand-crafted mixture of sequence stability, dwell proxy and handoff presence. It must not be mixed with supervised probability calibration.
- No temperature, Platt or isotonic calibration is currently applied. The field named `calibration` stores an F1-selected decision threshold, not probability calibration.
- The training script reports binary Brier and fixed-bin ECE, plus AUC/F1 and ETA MAE. It does not report binary NLL/MCE/reliability rows, multiclass Brier/NLL/ECE/classwise ECE, eligible-target calibration, ETA RMSE/buckets, or selective risk-coverage.

## Existing dev artifact

The latest inspected v112 soft-target artifact contains 41,059 train and 33,260 dev samples. Reported dev values are next-RSU accuracy `0.956705`, handoff-target accuracy `0.950481`, handoff AUC `0.872972`, Brier `0.041477`, ECE `0.011171`, ETA MAE `0.682158`, and an F1-selected threshold `0.140383`. Precision/recall/F1 at that threshold are `0.227137/0.508505/0.314012`; at threshold 0.5 the handoff classifier predicts no positives. These are historical dev diagnostics with threshold/evaluation reuse. They do not establish an independent calibrated evaluation result.

## Runtime timing and leakage audit

- `VecWorkflowCoreEnv.reset()` builds predictions from the reset observation; each subsequent state builds predictions after the mobility step for consumption by the following controller decision.
- `GymVecEnv` returns the state observation and semantic state before the caller selects the next action. The core environment snapshots these predictions before applying that action.
- Legacy runtime predictions have `prediction_time`, but no stable snapshot ID, `generated_at`, `observation_as_of`, validity interval, age or explicit consumption time.
- `PredictorManager.reset()` clears load, adapter heat, vehicle-position and delay histories, so normal episode reset prevents history carry-over.
- Legacy `prediction_delay_steps` appends a newly computed current payload and returns an older payload. Although the returned values are historical, current-state prediction work and stateful load/demand updates have already occurred. G12 must delay/reuse an actually historical generated snapshot and calculate age explicitly.
- The supervised runtime itself does not read future frames. Oracle mode separately reads future mobility frames. Legacy metadata can report oracle fallback to baseline; supervised mode does not silently load oracle data. G12 must additionally make oracle identity incompatible with supervised snapshot identity.
- The online `prediction_quality_audit` compares a predicted target with a target derived from the same predicted sequence. It is explicitly a proxy, not a realized future-label calibration audit.

## Predictor-policy mismatch

The v112 policy probe collapsed to mean reward `16.366` with mechanism realization `0.000`. The most direct evidence is mismatch rather than predictor-classification failure: handoff positives are rare, threshold 0.5 produces no positive decisions, the old confidence is not a calibrated correctness probability, and the policy path consumed target/prediction semantics for which it had not been retrained or selectively gated. This supports a default-off, abstention-aware interface. It does not support tuning a threshold against RL reward.

## G11 data boundary

- BurstGPT and Azure may support future external arrival/token profile calibration.
- Qwen-Bailian and Mooncake may support KV/prefix reuse profiles.
- HF qwen/cbow/bert candidates may support size metadata only.
- None of these sources observes NGSIM mobility/handoff labels and none may calibrate the current handoff, next-RSU, target or ETA predictor.
- No G11 payload is downloaded or joined in G12. Any future cross-source alignment remains exogenous/synthetic rather than a joint real VEC trace.

## Required G12 corrections

G12 must introduce a versioned JSON-safe snapshot and fail-fast validator; a three-way interval-audited train/calibration/evaluation protocol; pure calibration reducers; deterministic calibration-only fitting and selection; explicit abstention/staleness; a default-off runtime mask; immutable historical delay semantics; and action-preceding observation trace instrumentation. It must not enable a canonical predictor, alter reward/action/RL objectives, or claim policy improvement.
