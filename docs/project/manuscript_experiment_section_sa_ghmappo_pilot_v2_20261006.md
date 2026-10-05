# Pending manuscript patch: interface-blocked sequential-control pilot

Merge target: the limitations or implementation-diagnostics section of `docs/project/system_mechanism_manuscript_working_draft.md`, after the
existing bounded recovery evidence. This patch is kept separate to avoid overwriting concurrent manuscript work. It reports an interface-blocked
pilot and must not be used as a fair algorithm-ranking table or as formal/holdout evidence.

## Measurement-Calibrated Sequential-Control Pilot

### Question and claim boundary

We conducted a small, pre-frozen pilot to test whether the currently implemented dependency-aware representation provides a measurable benefit
before assigning it a paper contribution. The tested mechanism propagates DAG-node features along predecessor and successor edges. These node
features include whether the required adapter is resident at the current, predicted-next, and predicted-handoff RSUs. This is a specific
representation hypothesis, not a claim that MAPPO, graph encoders, shared base models, adapter prefetching, or checkpoint recovery are new.

The experiment uses a measurement-calibrated synthetic workload. NGSIM supplies disjoint source time intervals and handoff-pressure features,
while Alibaba traces supply 5--12-node DAG structures, durations, and memory fields. Model and adapter sizes, same-host load times, application-state
and input sizes, state-restore overhead, node compute time, and prefix-recomputation time are calibrated from the existing measurement artifact.
Adapter assignment, a second base-model identity, state-volume scaling, trajectory--workflow pairing, deadlines, and 200/1000-Mbps link conditions
are controlled synthetic factors. Consequently, the experiment is neither a real-RSU deployment nor a new real-world dataset.

### Attempted matched protocol and post-run interface audit

All methods interact with the same five-action producer, dependency-safe typed cache, atomic admission and rollback logic, application-state
transition, reward, and legal-action mask. The actions represent current-RSU cache fill, current-adapter prefetch to the predicted next RSU,
vehicle fallback, current-RSU steady offload, and current-adapter plus application-state preparation for the predicted handoff target. The interface
does not implement future-adapter-specific preparation, remote continued execution, a shared request queue, or shared radio/compute contention.

The decision state exposes an estimated link rate but not the execution-time actual link rate. Both the immediate and two-step online planners use
that estimated decision model; the environment alone applies the actual rate. A depth-four reachability diagnostic covered 1,183 states and 4,962
legal state--action pairs. Every checked state had at least two different one-step consequences, the symmetric byte-to-time formula had zero
numerical discrepancy, the semantic state contained no actual-rate field, and the immediate and two-step rules disagreed in 406 states. This
diagnostic establishes only that the pilot is not trivially action-invariant; it does not establish an RL advantage.

A post-run, read-only interface audit found that the intended matching was not achieved. The hierarchical PPO loss uses canonical controller-head
log probabilities, whereas the executed action is sampled or selected from a masked five-action distribution. The deterministic aggregation also
concentrates the event-prepare probability on action 4 while distributing event-keep probability across actions 0--3. In addition, mobility remains
derived from completed-node index and therefore freezes after a service failure. Finally, the flat and graph encoders do not consume the exposed
byte-capacity, typed-base dependency, link, state-size, and model-size fields on an equal basis; the flat occupancy feature divides an adapter count
by a byte capacity. The cost rules do not read the actual future rate, but they do possess an exact transition clone and a lexicographic objective,
which is stronger model-based planning capability than the learned scalar-reward actors. These issues invalidate a fair method comparison without
invalidating the preserved execution record.

The frozen split contains 12 training, 4 development, and 12 evaluation instances. Raw source frame intervals do not overlap across these splits.
For each learned method, we use seeds 7, 17, and 29, at most 128 episodes per seed, and at most 24 environment steps per episode. Candidate
checkpoints at episodes 32, 64, 96, and 128 are selected only on the development split by a fixed lexicographic rule prioritizing completion,
deadline/service/handoff failures, modeled time, transfer, and finally return. The evaluation split is not used for selection. The main methods are
SA-GHMAPPO, PPO, and controller-level MAPPO. The single-factor ablation retains the same observations, actions, reward, and budget but disables DAG
edge message passing. “Controller-level MAPPO” refers to cache, execution, and handoff-event policy heads with a centralized critic; it is not a
vehicle-level or RSU-level multi-agent implementation.

The worst-case frozen training budget is 27,648 steps for the three main methods plus 9,216 steps for the ablation. The observed total is 11,027
steps because many workflows terminate before the per-episode cap. No model generation, download, previous consumed holdout, hyperparameter search,
or post-result budget extension is used.

### Preserved diagnostic results

We first average the three seeds within each frozen evaluation window and then report percentile-bootstrap 95% confidence intervals over the 12
windows. Seed repetitions are not treated as independent samples. Because the post-run audit identified action-likelihood, mobility, feature, and
planner-capability mismatches, the table is descriptive diagnostic evidence and is not a valid algorithm ranking.

| Method | Workflow completion | Node coverage | Deadline violation | Elapsed to completion/truncation (s) | Transfer (MB) | Prefix recompute (s) |
|---|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 0.750 [0.667, 0.833] | 0.825 [0.763, 0.893] | 0.556 [0.361, 0.750] | 82.715 [58.034, 115.666] | 641.242 [214.267, 1161.152] | 42.221 [25.876, 59.535] |
| SA-GHMAPPO without dependency messages | 0.917 [0.833, 1.000] | 0.945 [0.882, 1.000] | 0.667 [0.417, 0.889] | 91.973 [64.532, 124.864] | 690.450 [192.031, 1330.623] | 51.250 [32.127, 71.506] |
| Controller-level MAPPO | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.750 [0.500, 1.000] | 104.189 [68.070, 151.077] | 825.278 [235.664, 1508.347] | 60.035 [35.639, 87.903] |
| PPO | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.750 [0.556, 0.917] | 95.973 [66.889, 133.241] | 550.810 [160.822, 999.693] | 40.263 [23.852, 58.580] |
| Immediate cost rule | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.417 [0.167, 0.750] | 59.567 [46.399, 74.778] | 27.651 [1.380, 65.846] | 2.038 [0.719, 3.477] |
| Correct two-step cost rule | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.167 [0.000, 0.417] | 52.300 [36.404, 73.562] | 169.034 [30.374, 376.159] | 4.016 [1.798, 6.174] |

The candidate mechanism is not established. Within the same defective hierarchical interface, disabling dependency messages changes workflow
completion by +0.167 (equivalently, full minus ablation is -0.167; 95% CI [-0.250, -0.083]), node coverage by +0.120, and return by
+6.375. These are negative implementation-sensitivity signals rather than paper-grade causal effects. The apparently lower elapsed time and
deadline-violation rate of the full model are not benefits: more episodes terminate at
the step cap without completing, so elapsed time is censored and terminal deadline accounting is avoided. The full model selects actions
0/3/4 a total of 180/10/120 times across seed--window runs, whereas the two-step rule conditionally uses actions 0/2/4 (52/20/10). This behavior,
together with the strong seed dependence, is consistent with policy/aggregation collapse rather than with a lack of sequential consequences in the
environment.

### Interpretation and disposition

The result blocks rather than ranks the methods. The experiment validates a measurement-calibrated execution substrate and a non-trivial legal
decision process, but the learning/execution interface does not provide a matched executed-action likelihood, mobility progression, encoder feature
contract, or planner capability. It therefore does not validate SA-GHMAPPO performance, establish the dependency-message mechanism, or prove the
superiority of the observed baselines. We retain the run as an interface-blocked diagnostic and do not promote the table to the paper's algorithm
results.

The pilot is limited to three seeds, 128 episodes per seed, 12 evaluation windows, synthetic link and pairing factors, and percentile bootstrap
intervals. It lacks formal/hidden holdout and support runs, BCa or multiplicity-controlled inference, shared bandwidth/compute queues, cross-workflow
persistent cache, a real adapter-request trace, and real-RSU deployment. Any future experiment must first pre-register and validate a minimal
executed-action PPO contract, decision-step mobility progression, and equal public-feature profile for all learned baselines. It must preserve this
artifact and must not select scenarios using the present evaluation outcomes.
