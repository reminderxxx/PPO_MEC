# Formal Protocol Capability Routing Contract 1.0.0

## Purpose

G14R13 replaces scattered version allow-lists with one explicit, fail-closed capability registry. The repair addresses
the Protocol 2.3 preflight defect where the outer runner passed a persisted resolved execution context but the nested
validator ignored it and fell back to the unresolved legacy default context.

## Authority and rules

- Code authority: `src/runtime/formal_protocol_capabilities.py`.
- Frozen machine contract: `configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905/formal_protocol_capability_routing_contract.json`.
- Protocol 2.4.0 is the only version with live execution permission.
- Protocol 1.0.0–2.3.0 are historical audit-only. Their historical capabilities remain explicit, but they cannot start
  a live execution.
- An unregistered version, including 2.5.0, fails closed and inherits no capability.
- No consumer may infer capability from a major version, lexical comparison, open range, current working directory,
  relative `.venv`, or implicit Python discovery.
- Protocol 2.4 requires the persisted resolved execution context, explicit Python/environment identity, execution
  binding, agent order, active bundle/resource resolution, exogenous request, full environment projection, request
  lifecycle, and nullable metric contracts.
- The active route has no holdout capability and may not fall back to `default_expansion_context`.

## Producer/consumer closure

The outer runner and nested validator read the same registry. Active bundle validation, resolved context,
training/binding/provenance, dev selection, checkpoint freeze, cache-policy/controller, ablation/support/scalability,
statistics, integrity, and gate paths also resolve the registered capability row. Historical builders remain audit
tools and are not mechanically rewritten to masquerade as current execution consumers.

## Protocol 2.4 boundary

Protocol 2.4 changes only execution capability routing and the identities derived from the new Protocol/bundle. The
Scientific Config, Nullable Metric Contract 1.0, request lifecycle/exogenous request contracts, NGSIM + Alibaba data,
split/window contracts, typed catalog/dependencies, agent order, seeds, capacities, training budget, SA-GHMAPPO
`auxiliary_coef=0.06`, selection/tie-break, endpoints, support/scalability/statistics/Holm rules, and holdout seal are
unchanged.

G14C v13 is recorded only as `PRE_EXECUTION_STOP / VALIDATOR_VERSION_DISPATCH_MISMATCH`: it created no execution
worktree, durable run root, ledger, checkpoint, candidate, selection, or performance row. It is not an invalid executed
run and creates no checkpoint denylist entry.

Readiness v16 is execution-contract evidence only. It authorizes a future, separately requested G14C v14 clean run;
it does not contain formal training, checkpoint, performance, holdout, G14D, G15, or paper-ready evidence.
