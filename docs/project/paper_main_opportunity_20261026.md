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
