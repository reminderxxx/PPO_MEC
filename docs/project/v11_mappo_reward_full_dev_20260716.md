# v11 MAPPO Reward Full-Dev Record 2026-07-16

## Scope

- Profile: `top_journal_mechanism_v11_mappo_reward`
- Split: frozen dev only, `configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json`
- Seeds: `7 13 29 41 53`
- Mode: `full_stratified`, 20 windows, 2 workflows, `max_steps=16`
- Boundary: dev evidence only. Existing hidden holdout is consumed and was not used.

## Diagnosis

Initial v11 used MAPPO controller credit but still trailed `popularity_cache_heuristic`. The loss was not from mechanism windows:

- mechanism windows: SA already beat popularity.
- active non-mechanism: tie.
- idle/sparse: SA lost enough reward to pull the total below popularity.

Two follow-up probes were rejected:

- vehicle-only fallback improved total reward from `78.934` to `79.4379`, but remained below popularity `79.46875`.
- global no-RSU local fallback improved idle/sparse but damaged mechanism windows, dropping total reward to `79.34415`.

Final v11 keeps MAPPO as the main policy, uses reward-first checkpoint priority, and enables no-RSU local fallback only through the outcome-blind `idle_or_sparse` window-context gate.

## Final Full-Dev Result

Artifact:

`artifacts/experiments/top_journal_mappo_reward_full_dev_v11_20260716/main_results_full_stratified_window_gate_full/main_results_full_stratified_20260716_181112_383674/aggregate_summary.json`

Total reward ranking:

- `sa_ghmappo`: `79.4944`
- `popularity_cache_heuristic`: `79.46875`
- `ppo`: `77.18775`
- `reactive_greedy`: `75.845`
- `controller_mat`: `74.2805`
- `mappo`: `72.6328`
- `cache_offload_drl`: `71.339`
- `dag_offload_drl`: `69.38515`
- `dt_handoff_drl`: `67.63495`
- `qmix`: `62.56225`
- `dqn`: `55.8715`
- `dueling_dqn`: `49.50075`

`sa_advantage_diagnosis.blockers=[]`; total reward delta over popularity is `+0.02565`.

Window-class result against popularity:

- mechanism: `82.788` vs `82.3425`
- active non-mechanism: `83.275` vs `83.275`
- idle/sparse: `77.2175` vs `77.3975`

Interpretation: v11 wins the full-dev table by preserving MAPPO mechanism-window advantage while using the window-context gate to reduce idle/sparse cost. It does not prove every window class is better than the heuristic.

## Reproduction Command

```bash
.venv/bin/python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo dqn dueling_dqn qmix controller_mat dag_offload_drl cache_offload_drl dt_handoff_drl --seed_checkpoint_manifest_path artifacts/experiments/top_journal_mappo_reward_full_dev_v11_20260716/seed_checkpoint_manifest.json --seeds 7 13 29 41 53 --max_mobility_rows 10000 --max_workflows 2 --max_steps 16 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_count 20 --window_mode full_stratified --window_plan_path configs/experiment/top_journal_v8_strict_split_20260621/dev_window_plan.json --primary_vehicle_selection handoff_pressure --min_tasks 5 --max_tasks 20 --output_root artifacts/experiments/top_journal_mappo_reward_full_dev_v11_20260716/main_results_full_stratified_window_gate_full
```
