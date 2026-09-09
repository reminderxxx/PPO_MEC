# G14R20-F1 requirement/evidence matrix

| Requirement | Result | Evidence |
|---|---|---|
| Missing location, immediate valid publication | PASS | `fast_missing_creation_events`; public parser/main/startup; real signature and `verify_approval` |
| Legacy receipt, immediate atomic replacement | PASS | `fast_legacy_replacement_events`; pre-challenge baseline is `present` |
| Publication after `waiting` | PASS | `delayed_receipt_events` |
| Only legacy or no receipt | PASS | `old_receipt_events` and `missing_receipt_events`; bounded timeout, no qualification |
| Fast invalid receipt | PASS | schema/signature/authority/checkpoint/nonce/PID/expired all rejected |
| Concurrent same-context hosts | PASS | second host `STARTUP_HOST_BUSY`; first completes; post-release fresh challenge succeeds |
| Deterministic old/new correspondence | PASS | `old_code_failure.xml` 2 failures; `new_code_regression.xml` same 2 cases pass |
| All wait callers use fixed baseline | PASS | repository search found one caller; `run_startup_operation` supplies required `baseline` argument |
| Real execution boundaries | PASS | synthetic dispatch=1; scientific rollout/real v16 dispatch/real v16 write/holdout=0 |
| Protection | PASS | 107,320 objects; zero start/end mismatch or addition; protected identities unchanged |

The old-code result is a synthetic deterministic reproduction, not a historical production failure or scientific result.
