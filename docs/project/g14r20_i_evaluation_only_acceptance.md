# G14R20-I evaluation-only acceptance record

- status: `READY_FOR_EVALUATION_ONLY_AUTHORIZATION`
- formal_execution_authorized: `false`
- formal_execution_started: `false`
- holdout_opened: `false`
- reviewed_at: `2026-09-11` (Asia/Shanghai)
- executor_code_commit: `fcf36f986261587a1daa70805569cfbf25920332`
- evidence commits: separate from the executor code commit; the final evidence
  commit is reported with the handoff
- scientific_commit: `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`
- source_review: `g14r20_h_v16_failed_run_disposition_20260911`

## Fixed checkouts and source reference

The scientific checkout is the persistent directory
`artifacts/execution_checkouts/g14r20_i_scientific_a6d1fd8`, fixed to the
scientific commit above. The executor checkout is
`artifacts/execution_checkouts/g14r20_i_executor_8c0d7e8`; its executable
code is the commit `fcf36f9` above; later checkout HEADs contain only evidence/
documentation commits and must not be substituted as the executor identity. The source contract accepts only the reviewed v16
run, the reviewed disposition eligibility digest, the reviewed generated
registry and the 150 verified checkpoint coordinates (10 agents × 5 seeds × 3
capacities). It rejects source-run, checkpoint, hash, coordinate, context,
registry and symlink/path-boundary drift.

## Consumer matrix

| Boundary | Real consumer | Acceptance evidence |
|---|---|---|
| source model → evaluation identity | `build_model_source_reference`, `validate_model_source_reference` | `tests/test_evaluation_only_identity_contract.py`, `tests/test_evaluation_only_source_boundaries.py` |
| outer context → nested resource/checkpoint | `load_generated_checkpoint_registry`, evaluation-only loader, `benchmark_main_results` | targeted consumer acceptance: 147 passed |
| command matrix → phase consumer | `expand_command_plan`, evaluation-only command parser | `tests/test_evaluation_only_identity_contract.py` |
| producer → transaction publication | `CellTransactionLedger.commit_cell`, producer-manifest validators | `tests/test_evaluation_only_publication_integrity.py` |
| final output → statistics/integrity/gate/completion | persisted-cell verification and existing statistics/gate consumers | `tests/test_evaluation_only_publication_integrity.py` and targeted consumer acceptance |

The only allowed command phases are `formal_cache_policy`,
`formal_controller`, `formal_ablation`, `formal_support`,
`formal_scalability`, `formal_statistics`, `formal_gate`, and
`complete_without_holdout`. No training, dev selection, checkpoint freeze,
recovery, retry or holdout command is generated.

## Publication integrity closure

Publication validates the original producer manifest and payload, applies only
the enumerated path relocations, records before/after hashes and a
path-normalized semantic hash, rebuilds and validates the final producer
manifest, then builds the transaction inventory and marker. Atomic publication
is followed by a fresh read of both integrity layers. Normal, repeated,
recovery and gate/statistics consumption use the same checks. The preserved
regression test demonstrates that a transaction inventory cannot mask a stale
producer manifest; mutations to metrics, nulls, row counts, membership,
duplicates, escapes, symlinks, relocation preimages, interruption and
single-writer state are rejected.

## Legacy exclusion and authorization package

The old 288 MB cell remains exposure/partial evidence only and the old 576 MB
staging remains failure evidence only. Neither is consumable by new statistics;
the new task must reevaluate all three capacities under Protocol 2.9. The
preparation entry point creates only a create-once, hash-bound authorization
request; it does not create a run root, ledger, lock, staging directory, grant,
or result. The generated request is retained outside Git at
`artifacts/analysis/g14r20_i_evaluation_only_readiness_20260911/authorization_request.json`.

## Verification boundary

The repository contains the targeted negative/consumer tests and the recorded
147-test targeted acceptance XML under the readiness artifact directory. A
fresh test rerun on this host was not possible because the available Python
environment has neither `pytest` nor `yaml`; full formal rollout, training,
real-data evaluation and holdout were not run (zero by design). Static
source/import compilation must therefore be rerun in the project's configured
dependency environment before authorization.
