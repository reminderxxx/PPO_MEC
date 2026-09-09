# G14R20-F1 fast receipt race isolated acceptance

## Outcome

`PASS_ISOLATED_RACE_REPAIR_ONLY` on implementation commit
`b35fbe779d6019efda7ddbcb230377cd2259aef6`. This branch was created directly from F implementation
`abe92a11d8d5bbd0a3313cbffbf6077f29c14cef`; it was not merged into main and did not restore real v16.

Production trust remains `not_installed`, continuation approval remains `not_issued`, the current release qualification
proposal remains unadopted, and `real_execution_authorized=false`.

## Deterministic root-cause reproduction

The regression uses an independent forked test-custodian process. The host sends the flushed challenge over a process pipe;
the custodian signs a real Ed25519 receipt, atomically publishes it, and acknowledges completion over the pipe before the
host is allowed to enter receipt-wait initialization. This fixes the ordering without sleeps or repeated attempts.

On the old F code, the exact two parameterized cases (initial location missing; initial legacy receipt present) both timed out.
The newly published receipt became the old implementation's `before` identity and was therefore ignored until the monotonic
deadline. `old_code_failure.xml` records 2/2 failures by `StartupTimeout`. This is synthetic reproduction evidence only.

On the repaired code, the same two cases pass (`new_code_regression.xml`, 2/2). The host now reads the receipt baseline while
holding the fixed-inode startup lock and before challenge publication. `wait_for_new_receipt` requires that baseline as an
argument and no longer initializes it after challenge. Repository-wide search found no other caller.

## Public-path acceptance

The core positive uses the same public parser/main/startup path as F. Host and custodian are different real PIDs. A
synthetic-only FIFO barrier pauses the host after flushed challenge output and before `waiting`; the independent custodian
then performs real signing and atomic publication. The production binding has no active barrier or installed trust.

- Legacy receipt replaced before `waiting`: accepted, verified, one synthetic dispatch completed.
- Missing receipt created before `waiting`: accepted and verified through qualification.
- Receipt published after `waiting`: accepted.
- No receipt and legacy-only receipt: bounded `STARTUP_RECEIPT_TIMEOUT`, no qualification.
- Fast schema/signature/authority/checkpoint/nonce/PID/expiry errors: all rejected without retry.
- Concurrent host: second host rejected with `STARTUP_HOST_BUSY`; first host completed; lock release allowed a fresh host with
  a new nonce/PID challenge.
- Four handoffs passed independent custody recomputation.

The JSONL event order remains challenge → waiting → qualification → handoff → terminal for successful paths. `waiting` reports
the pre-challenge baseline and is not a custodian signing prerequisite.

## Counts and validation

- synthetic dispatch: 1
- scientific rollout: 0
- real v16 dispatch: 0
- real v16 write: 0
- holdout consumption: 0
- targeted startup/production-trust/executor regression: 176 passed, 0 failed, 0 skipped
- full repository: 1,536 passed, 0 failed, 0 skipped
- smoke, all-Python compile, affected imports, strict JSON/zero-counter assertion and `git diff --check`: pass

The executor identity covers 25 files and has digest
`b534cca7e923f55d500b9f657c4a80ed49d04bdd8aa871016aed563b217664dc`; tree is
`016b1fd129ed0f59c6b27972ac41e6e1612eba07`. F's old identity is not reused.

## Protection and limits

Start/end protection scans covered 107,320 objects with zero mismatch and zero addition. The original scientific worktree
remained clean at commit/tree `a6d1fd8...` / `eb8e83c...`; the original proposal, both ledgers, original run inventory, main
pointer and seven user files were unchanged.

D's expensive eight-stage chain and real checkpoint semantic loading were not rerun. No training, model selection, freeze,
formal, holdout, G14D or G15 execution occurred. F evidence remains reused background; only the evidence named in this package
is an F1 revalidation.
