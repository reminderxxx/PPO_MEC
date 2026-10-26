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
