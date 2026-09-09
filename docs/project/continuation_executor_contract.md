> 2026-09-09 G14R20-D：生产授权与隔离 Ed25519 接口由
> [continuation_production_trust_contract.md](continuation_production_trust_contract.md) 2.0.0 取代下文 B 的
> state/hash 与空 signer 草案。1.0.0 HMAC 仅保留历史合成回归，不能进入生产。
> 八阶段正式隔离验收使用 2.0.0 共用签名、原件认证与撤销核心。生产 installation 仍为 None。

# G14R20-B Continuation Execution Contract 1.0.0

Status: implementation under validation; no production signer installed. Acceptance
completion and exact commit identities must be supplied by the separate evidence
record. This contract does not authorize v16 continuation or checkpoint evaluation.
The A proposal schema and read-only entry remain unchanged.

## Objects and identities

The executor identity is external to its source checkout, with `version`, exact
40-hex `commit`, `git_tree`, and a complete `files` array of relative `path` and
SHA-256. It covers both B CLIs, all `scripts/continuation_executor/*.py`, and the
unchanged A schema validator. Verification requires the exact clean commit/tree,
file equality and no additional source changes. Branch observation is diagnostic;
no moving `origin/main` equality is imposed on an already fixed executor. The full
identity is rechecked at each phase admission and before each child dispatch; the
old scientific source must still have its exact clean HEAD.

The independent execution contract binds the A proposal canonical digest and byte
SHA-256, executor identity digest, original run ID/root, both ledger prefix anchors,
immutable file inventory, full continuation command-plan digest, phase allowlist,
`holdout_capability=false`, domain, expiry, revocation ID, fixture scope and recovery
evidence. Unknown fields fail closed. Scientific execution identity remains
`a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`; B never rewrites context or binding.

The detached approval message contains its version/domain, contract digest, and
three separate evidence records: `launch_approval`, `release_attestation`, and
`continuation_approval`. Each has a state and evidence reference. Production trust
is a pinned empty signer map: supplied JSON, test HMAC keys, Readiness, ancestry or
self-calculated hashes cannot populate it. An independent future reviewed trust
installation and independently verified evidence are prerequisites to production
qualification. Missing original evidence stays unavailable/pending, not invalid.

Synthetic acceptance persists and reads back its proposal, contract, approval,
authority, identity and command plan before admission. The proposal byte hash
includes its actual trailing newline; even whitespace-only file drift is rejected.

Synthetic HMAC authority is limited to its exact fixture root and synthetic run,
contract/executor/commands, allowed phases, deterministic output/coordination
scope and expiry/revocation. It is refused by the production CLI. Test authority
is created only for isolated acceptance; it does not sign production requests.

## Entrypoints and boundaries

`execute_fixed_commit_continuation.py` has compatibility, qualification and execute
modes. Compatibility is read-only. Qualification/execute reject missing independent
approval before scientific imports, lock creation, ledger writes or dispatch.
`run_fixed_commit_continuation_acceptance.py` requires an exact clean executor and a
new `synthetic_*` fixture. It runs with the old checkout as cwd and loads original
scientific modules there. A fixed source commit does not establish launch approval.

The only phases are formal_cache_policy, formal_controller, formal_ablation,
formal_support, formal_scalability, formal_statistics, formal_gate, and
complete_without_holdout. Original train/dev/freeze must already be complete.
Scientific argv and ordering, coordinates, nested expansion, cwd/environment,
expected outputs and identities come from frozen templates/context. Only original
native staging and descriptor transformations are permitted for actual execution.

## Responsibility and retained-check matrix

| Responsibility | B orchestration | Original scientific implementation |
|---|---|---|
| Existing-run moving-main condition | Replaced by separate exact identity and independent approval | New-run public main unchanged |
| Source/environment | Checks exact old HEAD, clean tree, Python/cwd/import origins | Original environment resolver and source-content checks |
| Protocol/bundle/resources | Loads unchanged evidence identities | Original capability, bundle, resource and denylist validators |
| Context/binding | Read-only load and exact expansion comparison | Original context and execution-binding validators |
| Phase/cell identities | Same projections; explicit equivalence tests | Original stable IDs, hashing and ledger validators |
| Progress | Approved byte prefix plus legal successors | Original transaction records and committed payload verification |
| Child execution | Eight-phase allowlist and original command order | Original cell transaction, descriptor, inventory, atomic publication and marker |
| Statistics/gate | Original frozen child commands | Original statistics, integrity, formal gate and completion gate |

The executor does not call the old public runner main or replace validators with
success mocks. Synthetic first-five-phase payload producers are explicitly test
adapters, recorded alongside original commands; they are not scientific argv
compatibility evidence. Statistics/integrity/gate use the actual old scripts.
The consumer test calls actual benchmark main and loaders/gate, substituting only
synthetic mobility bundle preparation and stopping immediately before rollout.

## Single writer and recovery

Authorization and reconciliation precede lock acquisition; authorization and
reconciliation are checked again after acquisition. The persistent inode uses
nonblocking `flock`, never delete-and-recreate. The deterministic lock is outside
the immutable run, at `run_root.parent/.continuation_locks/<sha256(run_root)>.lock`.
The contract explicitly includes this coordination root. Keeping mutable lock
state outside the run preserves the original artifact-integrity consumer.

Owner evidence includes random nonce, host, PID, process start identity and exact
executor identity. A normal exit records release; an interrupted owner remains
recorded. Recovery requires a new approval binding that exact owner digest and
independent no-live-descendants/quiescence evidence, an obtainable kernel lock,
and a departed prior owner identity. PID reuse, elapsed time, or approval expiry
alone never remove a lock or establish recovery permission. No blind lock deletion.

A phase is the admitted transaction. Expiry/revocation blocks the next admission;
an admitted native transaction, its one permitted exit-75 retry and finalization
may finish. Native monotonic duration remains authoritative even if UTC changes.
Non-75 terminal failures are not resumed. Completion candidates use native
finalize-only and committed cells are verified/reused without dispatch. Registry
identity remains anchored to checkpoint-freeze terminal, not the growing tip.

## Acceptance evidence and limitations

Report implementation completion, synthetic compatibility, real qualification,
independent approval and actual execution separately. Three acceptance layers are
required: actual B CLI rejection; original producer/consumer main/loaders/gates;
and the complete synthetic eight-phase transaction chain with fault injection.
Every layer records source identities and actual calls. Report child dispatch,
scientific rollout, real v16 dispatch and real v16 write counts separately.

Synthetic instrumentation uses a fixture-only child `sitecustomize` observer,
explicitly recorded as a test environment difference. Probe `-I/-B` arguments are
not added to frozen scientific argv. Fixture path inspection covers nested argv,
resource references, checkpoint metadata, descriptors and all writes; original
source and installed interpreter/dependency metadata are read-only exceptions.
Synthetic tables/checkpoints/results are never formal scientific evidence. The
fairness builder records a separate fixture data-repository commit; this is not
the scientific execution commit. Its copied input configs/catalog are test data,
while all executed scientific modules remain from the old checkout.

Final evidence must bind the clean implementation commit and later separate
record commit, command logs/JUnit/strict-format and hash checks, all fault cases,
150 real checkpoint checks, 44 active and six generated resource checks, existing
ledger/payload/environment validation, protected inventory and seven user-file
start/end SHA-256. Missing coverage must remain explicitly unverified. This
implementation draft is not the final acceptance report.
