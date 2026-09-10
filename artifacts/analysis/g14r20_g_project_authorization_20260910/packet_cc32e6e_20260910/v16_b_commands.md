# G14C v16-B fixed project-authorization commands

These commands are prepared but were not executed in G14R20-G. Run exactly one phase at a time, only after the
previous phase has a valid completed terminal. Do not remove the stop file or a lock owner record to continue.

Fixed working directory:

```text
/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847
```

Common fixed command prefix:

```bash
env PYTHONPATH=/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847 \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/howen/Projects/PPO_MEC/.venv/bin/python -B \
  /private/tmp/ppo_mec_g14r20_g_executor_cc32e6e_20260910/scripts/execute_fixed_commit_continuation.py \
  --proposal /Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json \
  --contract /private/tmp/ppo_mec_g14r20_g_project_authorization/artifacts/analysis/g14r20_g_project_authorization_20260910/packet_cc32e6e_20260910/execution_contract.json \
  --executor-identity /private/tmp/ppo_mec_g14r20_g_project_authorization/artifacts/analysis/g14r20_g_project_authorization_20260910/packet_cc32e6e_20260910/executor_identity.json \
  --project-authorization /private/tmp/ppo_mec_g14r20_g_project_authorization/artifacts/analysis/g14r20_g_project_authorization_20260910/packet_cc32e6e_20260910/project_grant.json
```

Append exactly one of the following phase suffixes to that prefix:

```bash
--phase formal_cache_policy --check execute
--phase formal_controller --check execute
--phase formal_ablation --check execute
--phase formal_support --check execute
--phase formal_scalability --check execute
--phase formal_statistics --check execute
--phase formal_gate --check execute
--phase complete_without_holdout --check execute
```

The grant expires at `2026-09-30T23:59:59+08:00`. Its fixed local stop path is:

```text
/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/.continuation_locks/01523d80955bf4b2c19d19306eaef41da4fc9e6b4841f80564b884e6d9674c92.project_stop
```
