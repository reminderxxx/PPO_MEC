# Formal Nullable Metric Aggregation Contract 1.0.0

## Scope

This contract is the single active definition of nullable metric handling for Protocol 2.3 train/eval summaries, Dev checkpoint selection, checkpoint freeze, formal benchmark aggregation, paired statistics, integrity, formal gate, and claim-map consumers. The executable reducer is `src/metrics/formal_nullable_metrics.py`.

## Value and aggregation semantics

- Finite `int` and `float` values are available; `0` and `0.0` are valid observations.
- Explicit JSON `null` is unavailable and is never converted to zero or a numeric ordering sentinel.
- A missing required formal field is different from explicit `null` and fails immediately.
- `bool`, invalid strings, NaN, and positive or negative Infinity fail immediately on the formal path.
- Means use only available finite values and their denominator is `available_count`.
- `[null]` gives `mean=null`, available 0, unavailable 1; `[0.0]` gives `mean=0.0`, available 1, unavailable 0; `[null, 6.0]` gives `mean=6.0`, available 1, unavailable 1. Empty input gives `mean=null` and `availability_status=unavailable_no_rows`.
- `mean_metrics[metric]` remains a scalar `float|null`. Availability is carried separately by `total_count`, `available_count`, `unavailable_count`, and `availability_status`.
- CSV blank cells round-trip to unavailable. JSON writers use `allow_nan=false`.

For `end_to_end_workflow_delay`, `available_completed_workflow` requires a finite value. Failed, incomplete, or right-censored workflow reasons require `null`.

## Dev checkpoint selection

The frozen order is unchanged: maximize `full_service_ready_byte_hit_rate`, maximize `workflow_continuity_rate`, minimize `transfer_mb_per_request`, minimize `end_to_end_workflow_delay`, then update index and checkpoint SHA-256. At each metric, a finite candidate precedes an unavailable candidate. If both are unavailable, that dimension is skipped. If every candidate is unavailable, the dimension is recorded as non-participating. No zero, infinity, or reward substitution is allowed.

## Paired statistics and claims

`transfer_mb_per_request` and `end_to_end_workflow_delay` are lower-is-better. A pair enters a metric distribution only when both candidate and baseline are finite. Outputs record total pairs, available pairs, candidate-only, baseline-only, both-unavailable drops, and coverage. With zero available pairs, delta, confidence intervals, effect sizes, sign-test p-value, and Holm-adjusted p-value are all `null`. Holm correction consumes finite available p-values only. The formal gate and claim map label a zero-pair endpoint `UNAVAILABLE`, never tie/pass/fail.

## Version and evidence boundary

Protocol 2.3 binds this contract's content, size, semantic SHA-256, and role into the active bundle, resolved context, training/phase/cell identities, checkpoint provenance, Dev selection, statistics, and integrity. Protocol 2.2 is audit-only. This repair is execution-contract evidence only: it is not formal performance, G14 completion, TMC readiness, or paper readiness, and it does not authorize holdout access.
