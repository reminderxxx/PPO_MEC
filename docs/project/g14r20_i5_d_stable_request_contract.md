# G14R20-I5-D stable request and quiescence contract 1.2.0

`reviewed_at`: 2026-09-20
`literature_cutoff`: not applicable; engineering repair without literature review
`target_venue`: IEEE TMC (project target; no paper-readiness review)
`artifact_run_id`: G14E04 / new I5-D unsigned request
`policy_version`: `tmc_review_policy_v3_20260621`
`evidence_level`: E2 for engineering artifacts only; scientific outcome UNVERIFIED

## Frozen G14E04 fact

G14E04 remains `FAIL_STOP_NO_GRANT`. The independent report is
`/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i5_c_parent_bootstrap_20260916/independent_pregrant_review_g14e04_20260918.json`, SHA-256
`f1e24409af350d849212e080537d80c29886a20d14853fca8a079dd04628663c`.
No grant, qualification, bootstrap, recovery root, lock, ledger, staging or scientific dispatch followed that review. The I5-C request/review/grant/log are audit-only and cannot be retried or reused. The report remains unmodified.

Subsequent central checks observed that the same I5-C request could pass the same public validator and that a structured live rebuild could equal the frozen request. This is evidence of environment-dependent validation, not proof of persistent scientific-source drift. The old independent verdict remains binding.

## Root cause and projection

The I5-C `immutable_source_audit.held_lock.pid_observation` embedded `ps_return_code`, `live_start_time`, `probe_error`, PID and two authorization booleans. `stable_source_audit_projection` already excluded that entire non-authoritative observation in execute. The public validator instead compared the complete frozen request with a new live build. A changed ps permission/result/PID state therefore changed the verdict while model bytes, six committed cells, ledger prefixes, lock inode/hash/owner and staging stayed fixed.

Contract 1.2.0 has one shared `stable_recovery_request_projection`. It removes the full request's `authorization_request_sha256` only for live comparison and excludes exactly `immutable_source_audit.held_lock.pid_observation` and `immutable_source_audit.i4_review_inputs.declared_audit_bundle.present`. The latter is a live directory-presence flag explicitly marked `used_as_evidence=false`; it can change without changing a scientific original. Before projection, the complete frozen request canonical hash is verified. All other fields remain in comparison, including unknown future fields. The builder, public validator, qualify and execute import or call this projection through the same runtime module. New requests do not embed PID observations at all. Old 1.1 requests fail the new schema and remain audit-only.

No other time-varying field is currently excluded. The live build also checks parent state and Python/runtime audit, but contract validation keeps their identity checks exact. Any future non-authoritative field requires an explicit versioned contract change and a new negative test.

## Independent pre-grant evidence

`scripts/qualify_typed_model_cache_restricted_recovery.py` first validates the complete frozen hash and stable live sources, then writes a create-only observation file. Its schema records request hash, `observed_at`, five-minute `expires_at`, held-lock path/device/inode/size/SHA-256, owner payload identity, PID/start-time/ps result and permission availability, nonblocking kernel-lock result, before/after lock identity and bytes, quiescence result, and `lock_cleanup_authorized=false` / `holdout_capability=false`.

The probe opens the original inode read-only with `O_NOFOLLOW`, attempts `LOCK_EX|LOCK_NB`, closes the descriptor, and rechecks path/device/inode/size/hash. It never changes lock bytes or removes the inode. `ps` unavailability, an active original owner, an occupied/unverifiable kernel lock, or changed lock identity makes qualification `fail` with process-permission, quiescence, kernel-lock or immutable-lock failure codes. A failure cannot be interpreted as immutable request drift. Review and production grant must each bind the same evidence path, size and SHA-256. Grant verification rejects expired evidence and reobserves current quiescence; no observation can be reused past expiry.

The existing synthetic-only acceptance grant is isolated from production authority and remains a test fixture. It cannot authorize the real recovery parent or run root.

This contract authorizes only a new unsigned request and independent central review. It does not issue a grant, start recovery, open holdout, or establish historical task-start protection. `historical_start_evidence=unavailable` and `historical_protection_verdict=UNVERIFIED` remain fixed.
