# Event-triggered joint-control development

Date: 2026-10-26. Branch: `codex/event-joint-control`.

The independent readonly review and implementation plan are stored locally under
`artifacts/analysis/event_joint_control_20261026_v1/`. Prior negative prototypes
remain immutable. This is a bounded research prototype, not the original
SA-GHMAPPO, canonical evidence, or a formal/holdout promotion.

## Contracts and responsibilities

`JointWorkflowEnv.step_joint(node_id, prepare_adapter)` exposes a dependency-ready
service task and an independent released adapter preparation at the observed
current RSU. No automatic current admission, fallback, cache shielding or future
mobility is used. Native typed MB capacity, sequential LRU, serial loading,
eviction costs, and the existing finite-horizon business reward are inherited.
The prior `step_node` entry is exactly compatible.

`run_joint_workflow_probe.py` compares six prespecified causal rules across all 12
previously exposed windows. The opportunity gate uses training windows only.
Detailed outcomes stay in local artifacts; the code does not screen windows by
reward. `run_event_joint_learning.py` makes a learned preparation decision only
when warm service and missing ready-task adapters coexist and the link is idle.
A fixed global cache-first executor continues service while an admitted load is
in flight. No decision reward shaping or expert initialization is used.

`event_joint_encoder.py` produces local node, adapter-candidate and optional
actual-edge features. Graph/set variants share parameters, candidate mask,
local inputs, optimizer and budget. `EventJointPPOAgent` lives in `ppo_agent.py`
and does not enter the live registry. The critic is independent of the actor.
Checkpoint action and feature contracts are validated by the matching loader.

## Execution and validation

Use the shared absolute Python interpreter in a clean isolated worktree. Inputs,
outputs and seeds are explicit; output directories and checkpoints are create-only.
Example entry points: `run_joint_workflow_probe.py --inputs ... --output ...` and
`run_event_joint_learning.py --inputs ... --output ... --seed 7 --graph 1
--episodes 32 --cadence 16`.

The first implementation passes 128 matching tests, including independent
service/preload costs, no silent admission, exact legacy behavior, causal events,
parameter-matched graph/set initialization, permutation invariance, actual PPO
updates, checkpoint replay/rejection, and independent critic gradients.

Short learning tests must precede any fixed longer budget. All unsuccessful
attempts and final checkpoints must be retained. Performance on the old four
validation windows is exposed development evidence only. A paper claim requires
new frozen confirmation, honest time-interval independence, fair strong baselines,
physical metrics, variability and a separate readonly final audit.

## Observable eviction and successful-service LRU (v2)

The first event-policy pilot is retained with no demonstrated greedy learning
improvement. Independent investigation found that its view omitted recency and
that successful service did not refresh native typed object recency. New
`--eviction-aware 1` runs therefore use an explicit, equally shared opt-in
`service_hit_lru=True` physical contract for every baseline and learned policy.
Legacy defaults/checkpoints retain their original semantics. Successful service
refreshes the actual served base/adapter; blocked/loading requests do not.

A read-only native cache transaction forecast exposes current resident sizes,
LRU order, transfer cost and predicted victims. The common candidate proposal
set excludes preloads that would evict the selected warm task's adapter at the
observed RSU. This is an explicit constrained action set, not silent cache
shielding or action substitution. Future mobility is not queried. All methods
receive the same features and proposal constraints. Feature version v2 uses
12 candidate fields; the matching loader preserves v1 and v2 envelopes.

`prepare_event_joint_inputs.py` freezes outcome-blind disjoint raw-time windows
and 16 source-ordered Alibaba DAGs. It excludes raw interval overlap rather
than frame-rank overlap because extending a CSV prefix can change frame ranks.
A maximum-coverage persistent subject (>=80%) is selected using input data only.
Confirmation has unseen DAG identities and is outcome-sealed until a final
freeze. This is not a claim of exhaustive project-wide hidden-holdout novelty.
Failed data preparation attempts are retained, and no performance rows were
produced from their incomplete inputs.

## Fixed long comparison

`configs/experiment/event_joint_learning_v1.json` freezes event set-PPO as the
primary method, graph-PPO as the edge ablation, and primitive-step PPO as a
temporal-control comparator. Three seeds, 256 episodes per cell, 16-episode
checkpoint/evaluation cadence and final-only selection are fixed before the
long run. The ordered input sequence and episode cap match; realized native
steps and gradient minibatches may differ and must be reported rather than
called exactly equal interaction/compute budgets. The primitive controller
receives the same network, observable information, legal preparation proposals,
physical service/cache model and native reward, but chooses a preparation action
every physical step without the event executor's loading commitment/automatic
unavoidable fill. This tests temporal abstraction and the fixed execution prior.

`run_event_joint_study.py` executes one method group without resume or budget
extension. `evaluate_event_joint.py` evaluates five causal baselines or a saved
checkpoint under the identical v2 contract. Confirmation requires the frozen
input hash, checkpoint hash and baseline list. The statistical plan fixes raw
time windows as outer clusters, seed replication within windows, 9,999 bootstrap
resamples, percentile/BCa intervals and sign tests with global Holm correction.
A descriptive late reward plateau does not prove parameter/policy convergence.
