# Research-Skill Integration Contract

- adopted_at: 2026-07-27
- project: PPO_MEC
- mainline: NGSIM + Alibaba
- scope: research operations, experiment integrity, prediction/optimization analysis, literature and paper evidence

## Decision

PPO_MEC adopts a three-core, two-conditional workflow. External skills are advisory workflows, not runtime dependencies of src/; none may change reward, benchmark windows, baseline inputs, or results after seeing formal/holdout outcomes.

| Source | Project role | Adopt now | Explicit boundary |
|---|---|---|---|
| [Claude Scholar](https://github.com/Galaxy-Dawn/claude-scholar) | experiment registry, results report, claim-to-artifact evidence chain, code-review checklist | Yes: map its report/claim-ledger ideas to the existing artifacts/, manifests, checkpoints, command logs and docs/project/ records | Its Claude/Zotero/Obsidian commands are not a dependency of training or evaluation. No unpublished artifact, trace, checkpoint, or manuscript is sent to external services. |
| [scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | statistical design, prediction diagnostics, multi-objective analysis | Yes, selectively: hierarchical statistics already use the project implementation; use aeon only for an out-of-fold mobility/ETA predictor study; use pymoo only after a frozen multi-objective protocol is available | Do not replace the custom MAPPO trainer with PufferLib mid-study. Do not introduce PyMC/PufferLib/PyMOO dependencies until a separate reproducible environment and contract are frozen. Its repository security report flags undeclared/missing optional resources, so whole-repo installation is prohibited. |
| [academic-research-skills-codex](https://github.com/Imbad0202/academic-research-skills-codex) | related-work matrix, citation verification, methodology review, rebuttal | Yes: claim provenance is a required paper gate; every external citation is checked against a primary page and every numerical claim against an artifact path | A claim-provenance check cannot prove the simulator or training code correct; unit tests, trace audits and independent artifact checks remain mandatory. |
| [AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-research-SKILLs) | local training observability and engineering practices | Conditional: retain local JSON checkpoints/logs and add an observability integration only after the learning gate below passes | Do not install its autoresearch orchestration or LLM post-training flows. W&B/MLflow are optional conveniences, not evidence sources, and must not upload private data. |
| scipilot-figure-skill | paper figures and visual QA | Conditional, after a formal candidate exists | It is not a system-architecture tool and must not polish dev-probe plots into paper evidence. Use Mermaid/TikZ/draw.io for architecture. |

## Project-native evidence chain

The existing project layout is the source of truth. The adopted workflow is:

hypothesis/config -> train summary/checkpoint -> frozen benchmark rows -> hierarchical statistics -> mechanism audit -> claim ledger -> paper figure.

For each candidate, record a hypothesis, changed policy mechanism, unchanged contracts, train/dev windows, seed, checkpoint-selection rule, raw artifact paths, negative results, and the exact safe/prohibited wording. docs/project/top_journal_review_policy.md remains the promotion authority.

## Policy-Learning Gate (implemented; required before v47+ promotion)

The v47--v51 audit identified deterministic action invariance and policy overrides. The gate is implemented in `scripts/train_sa_ghmappo_real_sample.py` and `src/agents/sa_ghmappo_core.py`; it is required before any new candidate is benchmarked beyond dev:

1. **Separate feasibility from preference.** The environment action mask may reject infeasible actions. Predictor labels, pseudo-targets, temporal priors, option defaults, and continuity/backhaul rules may contribute differentiable training losses or logged features, but must not deterministically select a valid action at evaluation.
2. **Evaluate two policies.** Record both `raw_policy` (learned logits plus action mask only) and `safety_projected` (optional runtime guard). A paper claim about MAPPO requires the raw policy to improve; a safety-projection gain must be reported as a hybrid-system ablation.
3. **Make learning observable.** At every saved update, evaluate the fixed dev protocol and save a deterministic raw-policy action signature (action, raw/projected action, event choice and margin) with a stable digest.
4. **Block invalid selection.** The first checkpoint is not eligible until a second raw evaluation exists. A run with invariant raw action signatures and raw aggregate metrics cannot supply best/Pareto/paper candidate checkpoints. All automatic best-checkpoint scores and the consistency audit use `raw_policy`, never safety-projected metrics.
5. **Use retrospective labels correctly.** Future trajectory labels are allowed only for offline auxiliary loss or split/window auditing. Online state, action mask, benchmark policy and baseline inputs must not consume them.
6. **Pre-register the next candidate.** Freeze one ablation grid on dev, then freeze a new mutually time-disjoint formal/hidden protocol before observing final results. The existing consumed hidden split remains unavailable.

The gate is intentionally an experiment-integrity change, not a reward-shaping change. It preserves semantic_discrete_5, the shared predictor protocol for all compared agents, raw environment reward, and current baseline contracts.

## Algorithm direction after the gate

The next defensible algorithmic question is not “how to force more prepare actions”, but: **can a learned CTDE event head choose prepare only when the expected continuity benefit exceeds a learned backhaul/failure cost under a causal handoff forecast?**

The proposed experimental implementation is a constrained, policy-side MAPPO variant with: (a) raw action logits; (b) retrospective handoff labels only as an auxiliary calibration target during training; and (c) an explicit dual cost for failed prepare/backhaul in the advantage. It must be compared against a no-auxiliary, no-dual-cost, and safety-projected ablation on exactly the same frozen windows. This is a research hypothesis, not a claimed contribution until multi-seed formal evidence exists.

## Execution order

1. Completed: implement and unit-test the Policy-Learning Gate without changing environment reward or baselines.
2. Run the fixed dev ablation with multi-seed checkpoint/action-signature reports.
3. Audit the winner and its negative controls; only then freeze an unused formal/hidden protocol.
4. Run formal, support suite, robustness, scalability, statistics and claim ledger.
5. Use figure tooling only to render audited formal evidence.
