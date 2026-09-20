# G14R20-I5-E restricted recovery scientific consumer closure

- `reviewed_at`: 2026-09-20 (Asia/Shanghai)
- `literature_cutoff`: not applicable; engineering interface repair
- `target_venue`: IEEE TMC project target; this is not a paper-readiness review
- `artifact_run_id`: `nonformal_i5e_consumer_acceptance_20260920_f`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `acceptance_executor_commit`: `ea9fae516eb6964dfbdac862f259e93b659bbb58`
- `evidence_level`: E2 for the two-cell non-formal engineering chain; scientific claims UNVERIFIED
- `status_ceiling`: `READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION`

## Root cause and frozen interface

G14E05/I5-D failed before rollout. Recovery initialization produced
`restricted_recovery_execution_contract.json`, while the generated checkpoint
resource resolver unconditionally read `evaluation_execution_contract.json` from
the current context directory. The preserved attempt-2 stderr names this exact
missing file. The failed root, its request, initialization files, ledger,
staging, grant/review references and held lock remain read-only.

Contract `1.3.0` chooses explicit restricted-recovery consumption. The resolver
rejects simultaneous evaluation and recovery contracts, absent contracts and
symlinks. For recovery it validates the unsigned request and exact execution
contract, source reference and 150-model identity, current root/run ID, context,
executor commit/tree, source training context, allowed `formal_ablation` phase,
two cell IDs and command plan. The source checkpoint registry and provenance
remain those of the original v16 training run. Original evaluation provenance
is checked through the preserved evaluation source reference and original-run
audit; it is not copied into a forged current-run evaluation contract.

Before production cell transaction dispatch, the recovery executor checks 14
required scientific input paths and resource identities using the same
generated checkpoint resolver used by support and nested benchmark. Missing
inputs fail before child launch. The nested benchmark carries an explicit
non-formal one-episode option only for isolated acceptance; formal command
construction and frozen scientific arguments remain unchanged.

| Boundary | Producer | Consumer and validation |
| --- | --- | --- |
| Recovery initialization | request, recovery contract, source reference, resolved context | initialization marker and exact-byte reload |
| `support.main` | command-plan entry and staged child command | context/git, generated registry, active bundle, fairness and support binding |
| Generated resources | v16 registry, capacity manifest and provenance | recovery scope then generated registry, source context/binding and capacity match |
| Nested benchmark | support's bound args | same recovery scope, checkpoint gate and real model load |
| Result producer | episode, CacheEvent, rows, aggregate, support provenance | final-byte producer manifest validation |
| Transaction | child descriptor and producer manifest | atomic publication, marker, ledger and read-back |
| Handoff | six immutable external cells plus two new commits | eight unique `(phase, cell_id)` identities, no later-stage authority |

## Attempt lineage and original protection

The first target cell has original attempt 1 `failed_terminal`, G14E05
recovery attempt 2 `failed_terminal`, and a future distinct run records attempt
3. Both prior failed terminal hashes and all 11 files of the G14E05 root are
bound in `previous_recovery_failure`; its separate held lock remains `held`.
The second target cell was never started and remains first attempt 1, dispatched
only after the first commits. Six earlier committed cells remain external
read-only references. No old run is resumed, finalized, unlocked or rewritten.

The seven protected user files use this task's actual start/end snapshots under
`artifacts/analysis/g14r20_i5_e_recovery_consumer_20260920/`. Historical I5
qualification is unchanged: `historical_start_evidence=unavailable` and
`historical_protection_verdict=UNVERIFIED`.

## Isolated acceptance evidence

The preserved failed acceptance roots `nonformal_*_a` through `_e` document
the successive consumer failures: agent order, seed coverage, resource-heavy
full-data read stopped by the harness, mechanism diagnosis membership, and
stale producer manifest after support stamping. None was retried or treated as
formal performance. Root `_f` used real support and nested benchmark in two
separate ordered worker processes. Each worker was preceded by new short-lived
quiescence evidence and a distinct **acceptance-only** review/grant fixture;
these fixtures confer no production recovery authority.

Each `_f` ablation setting loaded the v16 PPO seed-7 medium-capacity checkpoint
with `checkpoint_provenance_status=compatible`, completed one real episode,
recorded 9 CacheEvents, wrote a summary and producer manifest, and passed
transaction publication/read-back. The ledger commits are attempts `[3, 1]`.
The handoff validates 6 old external plus 2 new cells, with 8 unique logical
IDs. The 288/576/864 MB mappings each resolve 50 checkpoint coordinates.
These results are non-formal diagnostic outputs and are excluded from formal
statistics, model selection and paper claims.

Evidence paths:

- `artifacts/analysis/g14r20_i5_e_recovery_consumer_20260920/task_start_snapshot.json`
- `artifacts/analysis/g14r20_i5_e_recovery_consumer_20260920/nonformal_f/`
- `artifacts/analysis/g14r20_i5_e_recovery_consumer_20260920/three_capacity_mapping.json`
- `artifacts/experiments/typed_model_cache_restricted_recovery/nonformal_i5e_consumer_acceptance_20260920_f/`

The final unsigned request is published separately at
`artifacts/analysis/g14r20_i5_e_recovery_consumer_20260920/authorization_request_unsigned.json`
with a fresh execution ID/root bound to the final clean executor commit/tree.
Its grant is not issued. Formal recovery attempt 3, the second formal cell,
holdout, training, selection and statistics are not executed by this task.
