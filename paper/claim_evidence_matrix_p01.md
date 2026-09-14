# P01 Claim–Evidence Matrix and Result/Figure Plan

## 1. Research questions

| ID | Research question | Required evidence | Current status |
|---|---|---|---|
| RQ1 | How does SA-GHMAPPO compare with matched reactive and learned controllers on the six primary endpoints? | Completed `formal_controller`, preregistered statistics, Holm family, gate-approved claim rows | `UNVERIFIED` |
| RQ2 | How does cache capacity alter readiness, transfer, continuity, and complete-workflow delay? | All three capacity strata, paired outer-window analysis, availability accounting | `UNVERIFIED` |
| RQ3 | Under a shared controller/admission contract, how do LRU, FIFO, LFU, Aging-LFU, and Random eviction differ? | Completed `formal_cache_policy`, policy-isolation audit, statistics | `UNVERIFIED` |
| RQ4 | What continuity–transfer trade-off follows from predictive prefetch and handoff preparation? | Paired request/event transfers and outcomes; mechanism/support artifacts | `UNVERIFIED` |
| RQ5 | Which enabled components are necessary for observed behavior? | Committed formal ablations, with checkpoint/config identity and matched exposure | `UNAVAILABLE`—first ablation cell failed before commit |
| RQ6 | How far is the learned controller from an exact finite-horizon opportunity reference? | Explicitly exact H=1/3/6/12 cells and scope audit | `UNAVAILABLE` |

## 2. Contribution-to-evidence map

| Contribution candidate | Safe manuscript wording now | Evidence required for stronger wording | Forbidden wording now |
|---|---|---|---|
| Typed readiness contract | “We define and implement readiness as joint base-model and adapter residency under dependency-aware admission.” | Source/runtime audit already supports implementation; empirical benefit requires typed-full vs no-base-sharing formal evidence. | “Base sharing improves hit rate/transfer.” |
| Role-factored controller | “We implement slow, fast, and event role heads inside one controller.” | Source and checkpoint support architecture; comparison requires formal statistics. | “Multi-agent coordination outperforms MAPPO.” |
| Graph/prediction representation | “The frozen controller encodes DAG, RSU, vehicle, and baseline prediction features.” | Formal ablation/support for necessity or robustness. | “Graph continuity critic/learned predictor drives gains.” |
| Multi-timescale alignment | “The action factorization aligns cache, execution, and handoff roles.” | Event/mechanism diagnostics and ablations for effectiveness. | “The hierarchy significantly improves continuity.” |
| Evaluation protocol | “We preregister paired, window-clustered comparisons across three capacities and six endpoints.” | Completed statistics/gate for result claims. | “The protocol establishes superiority/generalization.” |
| Data composition | “We combine NGSIM mobility, Alibaba DAGs, and a controlled typed catalog.” | Provenance is sufficient for this descriptive statement. | “We use a joint real-world vehicular model-cache trace.” |

## 3. Frozen claim families

| Protocol claim ID | Endpoint/boundary | Required evidence | Manuscript target | Status |
|---|---|---|---|---|
| `typed_base_sharing` | sharing reuse and avoided duplicate transfer | typed full vs no-base-sharing | Sec. VI / ablation table | `UNAVAILABLE` |
| `byte_hit` | `full_service_ready_byte_hit_rate` | typed full vs matched non-oracle baselines | Main table | `UNVERIFIED` |
| `transfer_overhead` | `transfer_mb_per_request` | paired raw CacheEvent transfer bytes | Main table + trade-off figure | `UNVERIFIED` |
| `workflow_continuity` | `workflow_continuity_rate` | paired workflow outcomes by outer window | Main table + trade-off figure | `UNVERIFIED` |
| `capacity_pressure` | six endpoints across 288/576/864 MB | capacity support matrix | Capacity figure | `UNVERIFIED` |
| `eviction_policy` | hit/transfer/pollution/churn | five reactive policies under only-policy-difference contract | Cache-policy table | `UNVERIFIED` |
| `oracle_opportunity` | exact feasible opportunity gap | exact H=1/3/6/12 cells only | Opportunity figure | `UNAVAILABLE` |
| `controller_comparison` | six primary endpoints | SA-GHMAPPO vs preregistered strongest dev baseline | Main table | `UNVERIFIED` |
| `predictor_boundary` | prediction support endpoints | baseline predictor; G12 supervised disabled | Support appendix | `UNAVAILABLE` |
| `data_realism_boundary` | provenance only | NGSIM + Alibaba DAG + controlled catalog statement | Experimental setup/limitations | supported as descriptive provenance only |

No protocol claim row currently has a gate-assigned result status. P01 does not assign `supported`, `mixed`, `unsupported`, or `contradicted` on the gate's behalf.

## 4. Planned tables

### Table I — Symbols and typed object semantics

- Inputs: protocol catalog, dependency manifest, runtime action semantics.
- Columns: symbol/object type, resident bytes, transfer bytes, capacity accounting, dependency, lifecycle.
- Required note: workflow state is migration-only; KV prefix disabled.
- Status: structure ready; exact catalog values may be inserted after a second independent transcription check.

### Table II — Semantic actions and role-head mapping

- Inputs: `action_schema.py`, `sa_ghmappo_core.py`, frozen checkpoint config.
- Columns: environment action, slow option, fast option, event option, mask condition, state/cache effect.
- Required note: event > slow-prefetch > slow-fill > fast-fallback > steady-offload priority.
- Status: source-audited; prose table included in manuscript.

### Table III — Dataset, split, training, and statistical protocol

- Inputs: protocol 2.9 manifest and model-source reference.
- Rows: data sources, workflow IDs, capacity strata, windows, seeds, budget, selection, endpoints, outer cluster, bootstrap/test/Holm.
- Required note: combined sources are not a joint real trace.
- Status: source-audited; detailed setup included in manuscript.

### Table IV — Main formal controller comparison

- Inputs: completed `formal_controller` cells plus `formal_statistics` and gate claim rows.
- Layout: one panel per capacity; six endpoints; estimate, 95% CI, availability count, paired effect, Holm-adjusted status.
- Unit: 12 non-overlapping outer windows; seed/workflow nested.
- Status: `BLOCKED` pending statistics/gate. Existing aggregate summaries must not be copied directly into a paper claim.

### Table V — Matched eviction-policy comparison

- Inputs: completed `formal_cache_policy` cells plus statistics/gate.
- Layout: five reactive policies × three capacities; readiness, transfer, pollution/churn secondary metrics.
- Status: `BLOCKED` pending statistics/gate.

### Table VI — Formal ablation and support

- Inputs: committed `formal_ablation` and `formal_support` cells.
- Minimum rows: full model, no prediction, and only other preregistered identities actually present in the protocol.
- Required note: do not invent a no-hierarchy row if it is absent from the frozen matrix.
- Status: `UNAVAILABLE`; first ablation cell failed before commit.

### Table VII — Reproducibility manifest

- Inputs: complete manifest, command log, scientific/executor commits, model reference, checkpoint hashes.
- Status: partial identifiers ready; completion artifact missing.

## 5. Planned figures

### Fig. 1 — System timeline and typed transaction semantics

Visual sequence: frozen request exposure → controller observation → masked role decisions → five-action aggregation → admission/prepare → same-step lookup → execution/outcome → mobility/workflow transition.

The figure must make two relationships visually explicit:

- base + adapter determine readiness, while workflow state is migration-only;
- cache admission precedes lookup, so a hit can coexist with transfer cost.

Evidence: runtime/action source. Status: can be drawn now without numerical results.

### Fig. 2 — SA-GHMAPPO architecture

Visual blocks: DAG encoder, RSU set encoder, vehicle encoder, prediction encoder/reliability gate, shared fusion, centralized value head, slow/fast/event policy heads, hierarchical conditioning, deterministic action priority map.

Figure caption must state “controller-role heads, not independent vehicle/RSU agents.” Disabled mechanisms must not appear as active blocks. Status: can be drawn now.

### Fig. 3 — Capacity sensitivity of six primary endpoints

Preferred design: small multiples with capacity on x-axis and outer-window paired estimates/95% intervals on y-axis. Separate lower-is-better endpoints. Status: pending statistics/gate.

### Fig. 4 — Continuity versus transfer trade-off

Preferred design: paired outer-window or algorithm-capacity summary points with uncertainty, not 16,200 row-level points. The plot should distinguish same-step transfer from avoided future transfer. Status: pending statistics/gate.

### Fig. 5 — Cache-policy isolation

Preferred design: five matched eviction policies; show readiness and transfer together, with pollution/churn as secondary panels. Status: pending statistics/gate.

### Fig. 6 — Exact finite-horizon opportunity gap

Preferred design: gap by H=1/3/6/12 only for cells explicitly marked exact. Caption must define enumerated state/action scope and deny a global-optimum interpretation. Status: unavailable.

### Fig. 7 — Ablation/predictor support

Only generate after committed formal ablation/support artifacts exist. Status: unavailable.

## 6. Result sentence templates

These templates are intentionally incomplete. Bracketed fields may be filled only from gate-approved statistics.

- “Across the 12 preregistered outer mobility windows at [CAPACITY], SA-GHMAPPO changed [ENDPOINT] relative to [BASELINE] by [PAIRED ESTIMATE], with [95% CI], [AVAILABILITY], and Holm-adjusted [STATUS].”
- “The direction was [CONSISTENT/MIXED] across the three frozen capacity strata; [STATE PRECISE EXCEPTIONS].”
- “Because admission precedes same-step lookup, the observed readiness change coincided with [TRANSFER CHANGE]; this result does not imply avoided transfer unless the paired transfer endpoint supports that interpretation.”
- “Complete-workflow delay was available for [COUNT] paired outer observations; unavailable workflows were not imputed as zero.”
- “The H=[HORIZON] reference was exact within [ENUMERATED SCOPE] and therefore measures a limited finite-horizon opportunity gap rather than a global upper bound.”

Forbidden sentence patterns before gate completion:

- “SA-GHMAPPO significantly outperforms ...”
- “Our method achieves state-of-the-art performance ...”
- “The ablation proves ...”
- “The oracle upper bound shows ...”
- “16,200 independent samples ...”
- “Missing delay/regret is zero ...”

## 7. Provenance for result insertion

Every inserted numerical statement must record:

- `artifact_run_id` and phase/cell IDs;
- scientific and executor commits;
- protocol semantic hash;
- model-source/checkpoint identity;
- exact endpoint field and unit;
- outer-pair availability/drop counts;
- confidence-interval method and bootstrap seed;
- effect size, raw p-value, Holm-adjusted status;
- whether the result is formal, support, scalability, exact report-only, or holdout;
- claim ID and gate-assigned status.

If any required field is missing, the prose remains `UNVERIFIED` or `UNAVAILABLE`.

## 8. P02 writing priorities

P02 should focus on literature and positioning without waiting for result recovery:

1. refresh primary-source literature beyond the audited 2026-06-21 cutoff;
2. test the novelty boundary against recent `DAG + caching`, `VEC predictive handoff`, and `model/adapter serving` papers;
3. decide whether to retire the “multi-agent” expansion of SA-GHMAPPO;
4. populate Related Work with verified publisher metadata and a BibTeX source;
5. keep all empirical contribution sentences conditional until the gate assigns statuses.
