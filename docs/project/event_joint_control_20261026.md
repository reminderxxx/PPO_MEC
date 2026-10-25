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
