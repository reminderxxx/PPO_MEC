# Pending manuscript patch: calibrated interface repair and bounded development validation

Merge target: implementation diagnostics / limitations of `system_mechanism_manuscript_working_draft.md`. This patch is separate so it does
not overwrite concurrent manuscript work. It is not a formal or holdout main-result table.

## Interface Repair and Bounded Development Validation

A post-pilot audit found three interacting implementation defects in the calibrated sequential-control prototype. First, deterministic hierarchical
inference selected the maximum marginal environment-action probability rather than aggregating the independently most likely controller-head actions;
the event-prepare probability was assigned wholly to migration action 4, whereas event-keep probability was split across actions 0--3. Second, PPO
optimized canonical controller-head log probabilities rather than the likelihood of the executed five-way action. Third, mobility was indexed by the
completed-node count, so a failed service attempt froze the mobility and prediction state. The flat encoder additionally divided an adapter count by a
byte capacity, and both flat and graph policies omitted several public typed-cache, model-size, state-size, link, and contact-budget fields.

We introduced an explicit versioned interface rather than changing the historical contract. Under the repaired profile, mobility advances by decision
step while workflow progress advances only after successful service; all learned methods receive the same public byte occupancy, typed bundle
readiness, size, estimated-link, and contact features. Hierarchical policies use independent head argmax followed by semantic aggregation and legal-mask
projection at deterministic inference, and PPO optimizes only the executed environment-action likelihood. We did not alter the reward, workload,
five-action semantics, or SA-GHMAPPO event, temporal-consistency, and auxiliary coefficients.

We retrained SA-GHMAPPO, PPO, and controller-level MAPPO for three fixed seeds and 192 episodes per seed, with 24 steps per episode and development-only
checkpoint selection. The observed 12-window regression set had previously been exposed. A second eight-window group used previously unused source
windows from the pre-existing development plan, but reused the design-template family and is therefore a frozen development check, not an independent
holdout. A strong two-step rule retained its exact decision-model clone and lexicographic objective; this model-based capability is stronger than the
learned scalar-reward actor even though it does not observe the execution-time actual link.

On the exposed regression set, workflow completion was 0.944 for SA-GHMAPPO and 1.000 for PPO, controller-level MAPPO, and the two-step rule. On the
frozen development check, the corresponding values were 0.958, 1.000, 1.000, and 1.000. SA-GHMAPPO improved descriptively from the defective pilot's
0.750 on the regression windows, but this is an interface-repair recovery rather than an algorithmic contribution. The repaired raw-head/action replay
matched all 1,340 preserved learned-policy decisions and the stored policy log probability equaled the executed-action log probability. Nevertheless,
SA-GHMAPPO still selected action 4 on 50% of current-bundle-missing decisions in the new check and retained short no-progress runs of up to four steps.
Thus the frozen-state implementation loop disappeared, while a legal but inefficient learned event-policy bias remained.

The evidence does not establish an SA-GHMAPPO advantage. The two-step planner completed every workflow and had lower deadline violation, completed-case
latency, and transfer volume in the new check. With no event-mechanism ablation, the remaining gap cannot be attributed causally to one enhancement,
and fixed-budget under-training cannot be excluded. We therefore report the repair as implementation correction, retain the negative ranking, and do
not promote the experiment to a paper-ready algorithm result.
