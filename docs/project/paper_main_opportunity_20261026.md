# Paper stream main-method opportunity study (2026-10-26)

This opt-in development study starts from the paper's frozen stream source
`35975cb3`, with eight vehicles, request-bound controls, persistent per-session
cache and the original native team reward. It is separate from the five-action
prediction pilot and the one-vehicle event prototype. Default registry and
comparators remain unchanged. No formal or sealed holdout is opened.

First run `scripts/probe_stream_main_opportunity.py` against a verified existing
`joint_stream_learning_v2` protocol and its fixed-final main-method checkpoints.
The input-only selector uses four conditions in each of the three training time
sessions. All three existing compatible `StreamCacheRule` modes and all three
main-method seeds are retained. This establishes only the strongest member of
the existing compatible rule pool, not the strongest rule across other action
contracts. Native reward components, status counts and every physical frame
receipt are saved in a fresh output directory. The `--preflight` flag runs one
case for software/runtime validation only.

The next implementation must be justified by a completed, separate train-only
diagnosis. Candidate and controls must share information, physical costs, source
checkpoint initialization and additional interaction/update budgets; any fresh
optimizer warm start is explicitly identified. Module deletion at evaluation is
a sensitivity diagnostic and cannot be relabelled a retrained ablation.

Private diagnosis, raw data references, checkpoints and detailed reviews belong
under the ignored `artifacts/analysis/` directory. These do not enter public Git.
No component or development finding promotes full-system formal/canonical or
paper-ready claims. Remaining independent-window, strongest-baseline, physical
calibration and module-causality requirements continue to apply.

## Fixed follow-up implementation contract

The independently saved train-only diagnosis justifies a single intervention:
rescale the existing state-restoration prior from remaining-session frames to
remaining current-node compute frames. Its old term is removed before inserting
the new term; there is no additional reward charge or removed legal execution
action. The candidate is explicitly instantiated from `sa_ghmappo_agent.py` and
does not change the shared PPO implementation, physical environment or rules.

`run_stream_main_ablation.py` freezes six variants: original prior, new prior,
new prior without DAG edges, without prediction, without surrogate, and without
optional cache admission. Every variant uses three identical parent seeds,
exactly 256 actual interactions for each of 12 input-selected train conditions
(3072 per cell), three update calls and four epochs per update (KL early stopping
and actual gradient-step counts remain visible). Node features, network size
and the 73940 parameters are retained in the DAG-edge deletion control.

All variants use a fresh optimizer and a reset learning schedule. Fixed-frame
cuts bootstrap the native team tail with the existing central critic. Physical
termination restarts the same condition if its frame budget is not exhausted;
all repeated fragments and costs are counted. This is a shared bounded pilot
training contract, not exact continuation or the old full-episode MC trainer.
Before and fixed-final greedy evaluation uses complete original physical
episodes in three score-blind train cases (short/medium/long across three time
sessions). These three evaluation cells confound stratum and time and cannot
support independent formal inference. No peak selection, extra tuning cycle,
holdout access or baseline improvement is authorized by this script.
