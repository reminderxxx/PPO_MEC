# Handoff state runtime: bounded implementation task

Date: 2026-10-26. Baseline: b8d7442. Branch: codex/handoff-state-runtime.

Implement the independent task proposed after the read-only evidence inventory.
No new training, canonical update, sealed holdout, or manuscript change.

Stage 1 connects native NGSIM association changes to a versioned optional runtime
in VecWorkflowCoreEnv. The technical two-node workflow uses the existing local
base and adapter. Source and target are separate processes; dynamic inputs or
state cross an actual localhost TCP connection. Target never reads the source
image directly on recovery. All paths use identical model/generation identities.

The controlled trace replay places a completed prefix immediately before an
observed handoff. It is accelerated, blocking replay, not real-time mobility or
wireless deployment. Measure actual workflow/blackout wall time separately from
raw frame/time intervals. Static weights are already on disk for every arm;
resident-memory preparation is measured and charged explicitly.

Compare restart, reactive suffix recovery, and causal target preparation plus
suffix recovery. Retain fixed repetitions, every failure, raw node outputs,
transport counts, commands, source hashes and full timing. Freeze input-only
case selection and budgets before model execution. No outcome-based selection.

Stage 2 checks whether decision-relevant opportunity remains after strong rules,
including transfer/restore overhead and negative boundary cases. Exact diagnostic
future information is marked oracle and never supplied to an online baseline.

Stage 3 trains only if a reproducible controllable opportunity survives stage 2.
Otherwise report the negative gate and the specific missing mainline evidence.
These technical results do not establish full SA-GHMAPPO/GNN superiority,
application task correctness, NGSIM+Alibaba end-to-end deployment or convergence.

## Separate real-time development gate

The accelerated test cannot establish that preparation fits a 0.2-second raw
lead. A second, separately frozen development test therefore uses wall-paced
100-ms observations. Input-only dwell qualification on the same known prefix
finds two eligible events (overlapping intervals, one effective cluster).
An unexecuted 24-second-release input preparation is retained as v1. Kernel
calibration showed it may lack a committed prefix. Before any wall-paced model
execution, v2 instead binds request release to the second observed frame in the
source association, a causal arrival rule independent of future handoff time.
Both eligible events remain. Suffix generation allows 32 tokens.
Use all two events and all three arms once, plus an independent continuous
reference, with at most 22 generation attempts. No outcome-based retries.

`scripts/run_realtime_handoff_recovery.py` extrapolates only past positions by
six seconds. At observed handoff it terminates an unfinished source process.
Only an actually completed n0 can produce a checkpoint. Otherwise recovery falls
back to raw-input restart. Already completed workflows are retained as such;
they are not artificially migrated. Actual target dwell is checked separately.
This is a local two-node technical test, still not wireless/application accuracy
or a main-method algorithm comparison. Both positive and negative paths remain.

## Separate keep-source/forward deployment diagnostic

Vehicle handoff alone does not imply source-RSU failure. The interruption test
therefore cannot prove migration is preferable to continuing on the source.
`run_realtime_source_forwarding.py` uses the same two fixed wall-paced inputs,
one trial each and at most four generation calls. Source execution continues
after handoff; only its final result is transferred over the same paced TCP
link to a separate lightweight target process. Source availability, permitted
backhaul forwarding and absence of competing load are explicit assumptions.
This is outside the legacy five-action contract and remains a development
problem-formulation diagnostic, not a new formal algorithm baseline ranking.
