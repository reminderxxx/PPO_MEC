# large_scale_real_dataset_round13_report

## ??

????????????????????? `NGSIM mobility + Alibaba cluster-trace-v2018 batch_task DAG`?Hugging Face `ClemSummer/qwen-model-cache` ??? model-cache metadata source ???????? catalog???? benchmark ???? HF ????????? HF ????????? cache event?

## ????

- agents: `sa_ghmappo`, `reactive_greedy`, `popularity_cache_heuristic`, `ppo`, `mappo`
- seeds: `7`, `13`, `29`
- workflows: `j_3`, `j_8`, `j_15`, `j_34`
- NGSIM rows: `5000`
- window_length: `32`
- max_steps: `20`
- mixed episodes: `360` total, `72` per agent
- full-stratified episodes: `1080` total, `216` per agent
- full-stratified selected windows: mechanism `6`, active_non_mechanism `6`, idle_or_sparse `6`

## Mixed Overall

| agent | episodes | reward? | success? | continuity? | handoff_fail? | mechanism? | backhaul? | miss? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sa_ghmappo` | 72 | 72.257 | 0.875 | 0.907 | 0.167 | 0.000 | 112.000 | 1.208 |
| `reactive_greedy` | 72 | 69.840 | 0.875 | 0.880 | 0.167 | 0.000 | 90.667 | 1.625 |
| `popularity_cache_heuristic` | 72 | 71.819 | 0.875 | 0.880 | 0.167 | 0.000 | 117.333 | 1.625 |
| `ppo` | 72 | 67.968 | 0.583 | 0.605 | 0.056 | 0.056 | 82.667 | 7.472 |
| `mappo` | 72 | 67.968 | 0.583 | 0.605 | 0.056 | 0.056 | 82.667 | 7.472 |

## Full-Stratified Overall

| agent | episodes | reward? | success? | continuity? | handoff_fail? | mechanism? | backhaul? | miss? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sa_ghmappo` | 216 | 75.683 | 0.815 | 0.831 | 0.148 | 0.125 | 135.778 | 2.764 |
| `reactive_greedy` | 216 | 70.312 | 0.847 | 0.840 | 0.185 | 0.000 | 111.111 | 2.306 |
| `popularity_cache_heuristic` | 216 | 74.835 | 0.875 | 0.855 | 0.087 | 0.097 | 150.889 | 2.125 |
| `ppo` | 216 | 69.762 | 0.546 | 0.554 | 0.089 | 0.306 | 106.074 | 8.565 |
| `mappo` | 216 | 69.762 | 0.546 | 0.554 | 0.089 | 0.306 | 106.074 | 8.565 |

## SA Pairwise Deltas - Mixed

| baseline | metric | SA | baseline | delta | result |
| --- | --- | ---: | ---: | ---: | --- |
| `reactive_greedy` | adapter_miss | 1.208 | 1.625 | -0.417 | sa_win |
| `reactive_greedy` | backhaul | 112.000 | 90.667 | 21.333 | sa_loss |
| `reactive_greedy` | continuity | 0.907 | 0.880 | 0.027 | sa_win |
| `reactive_greedy` | handoff_failure | 0.167 | 0.167 | 0.000 | tie |
| `reactive_greedy` | mechanism_realization | 0.000 | 0.000 | 0.000 | tie |
| `reactive_greedy` | success_rate | 0.875 | 0.875 | 0.000 | tie |
| `reactive_greedy` | total_reward | 72.257 | 69.840 | 2.417 | sa_win |
| `popularity_cache_heuristic` | adapter_miss | 1.208 | 1.625 | -0.417 | sa_win |
| `popularity_cache_heuristic` | backhaul | 112.000 | 117.333 | -5.333 | sa_win |
| `popularity_cache_heuristic` | continuity | 0.907 | 0.880 | 0.027 | sa_win |
| `popularity_cache_heuristic` | handoff_failure | 0.167 | 0.167 | 0.000 | tie |
| `popularity_cache_heuristic` | mechanism_realization | 0.000 | 0.000 | 0.000 | tie |
| `popularity_cache_heuristic` | success_rate | 0.875 | 0.875 | 0.000 | tie |
| `popularity_cache_heuristic` | total_reward | 72.257 | 71.819 | 0.438 | sa_win |
| `ppo` | adapter_miss | 1.208 | 7.472 | -6.264 | sa_win |
| `ppo` | backhaul | 112.000 | 82.667 | 29.333 | sa_loss |
| `ppo` | continuity | 0.907 | 0.605 | 0.302 | sa_win |
| `ppo` | handoff_failure | 0.167 | 0.056 | 0.111 | sa_loss |
| `ppo` | mechanism_realization | 0.000 | 0.056 | -0.056 | sa_loss |
| `ppo` | success_rate | 0.875 | 0.583 | 0.292 | sa_win |
| `ppo` | total_reward | 72.257 | 67.968 | 4.289 | sa_win |
| `mappo` | adapter_miss | 1.208 | 7.472 | -6.264 | sa_win |
| `mappo` | backhaul | 112.000 | 82.667 | 29.333 | sa_loss |
| `mappo` | continuity | 0.907 | 0.605 | 0.302 | sa_win |
| `mappo` | handoff_failure | 0.167 | 0.056 | 0.111 | sa_loss |
| `mappo` | mechanism_realization | 0.000 | 0.056 | -0.056 | sa_loss |
| `mappo` | success_rate | 0.875 | 0.583 | 0.292 | sa_win |
| `mappo` | total_reward | 72.257 | 67.968 | 4.289 | sa_win |

## SA Pairwise Deltas - Full

| baseline | metric | SA | baseline | delta | result |
| --- | --- | ---: | ---: | ---: | --- |
| `reactive_greedy` | adapter_miss | 2.764 | 2.306 | 0.458 | sa_loss |
| `reactive_greedy` | backhaul | 135.778 | 111.111 | 24.667 | sa_loss |
| `reactive_greedy` | continuity | 0.831 | 0.840 | -0.009 | sa_loss |
| `reactive_greedy` | handoff_failure | 0.148 | 0.185 | -0.037 | sa_win |
| `reactive_greedy` | mechanism_realization | 0.125 | 0.000 | 0.125 | sa_win |
| `reactive_greedy` | success_rate | 0.815 | 0.847 | -0.032 | sa_loss |
| `reactive_greedy` | total_reward | 75.683 | 70.312 | 5.370 | sa_win |
| `popularity_cache_heuristic` | adapter_miss | 2.764 | 2.125 | 0.639 | sa_loss |
| `popularity_cache_heuristic` | backhaul | 135.778 | 150.889 | -15.111 | sa_win |
| `popularity_cache_heuristic` | continuity | 0.831 | 0.855 | -0.024 | sa_loss |
| `popularity_cache_heuristic` | handoff_failure | 0.148 | 0.087 | 0.060 | sa_loss |
| `popularity_cache_heuristic` | mechanism_realization | 0.125 | 0.097 | 0.028 | sa_win |
| `popularity_cache_heuristic` | success_rate | 0.815 | 0.875 | -0.060 | sa_loss |
| `popularity_cache_heuristic` | total_reward | 75.683 | 74.835 | 0.847 | sa_win |
| `ppo` | adapter_miss | 2.764 | 8.565 | -5.801 | sa_win |
| `ppo` | backhaul | 135.778 | 106.074 | 29.704 | sa_loss |
| `ppo` | continuity | 0.831 | 0.554 | 0.277 | sa_win |
| `ppo` | handoff_failure | 0.148 | 0.089 | 0.059 | sa_loss |
| `ppo` | mechanism_realization | 0.125 | 0.306 | -0.181 | sa_loss |
| `ppo` | success_rate | 0.815 | 0.546 | 0.269 | sa_win |
| `ppo` | total_reward | 75.683 | 69.762 | 5.920 | sa_win |
| `mappo` | adapter_miss | 2.764 | 8.565 | -5.801 | sa_win |
| `mappo` | backhaul | 135.778 | 106.074 | 29.704 | sa_loss |
| `mappo` | continuity | 0.831 | 0.554 | 0.277 | sa_win |
| `mappo` | handoff_failure | 0.148 | 0.089 | 0.059 | sa_loss |
| `mappo` | mechanism_realization | 0.125 | 0.306 | -0.181 | sa_loss |
| `mappo` | success_rate | 0.815 | 0.546 | 0.269 | sa_win |
| `mappo` | total_reward | 75.683 | 69.762 | 5.920 | sa_win |

## Full Window-Class Breakdown

| window_class | agent | episodes | reward? | continuity? | handoff_fail? | mechanism? | backhaul? | miss? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mechanism_activating | `sa_ghmappo` | 72 | 78.778 | 0.587 | 0.167 | 0.000 | 181.333 | 7.125 |
| mechanism_activating | `popularity_cache_heuristic` | 72 | 75.400 | 0.666 | 0.167 | 0.000 | 200.000 | 5.125 |
| mechanism_activating | `reactive_greedy` | 72 | 70.971 | 0.666 | 0.167 | 0.000 | 141.333 | 5.125 |
| mechanism_activating | `ppo` | 72 | 72.576 | 0.378 | 0.056 | 0.167 | 143.111 | 11.722 |
| mechanism_activating | `mappo` | 72 | 72.576 | 0.378 | 0.056 | 0.167 | 143.111 | 11.722 |
| active_non_mechanism | `sa_ghmappo` | 72 | 72.986 | 0.965 | 0.181 | 0.292 | 98.889 | 0.347 |
| active_non_mechanism | `popularity_cache_heuristic` | 72 | 73.885 | 0.969 | 0.000 | 0.292 | 100.667 | 0.333 |
| active_non_mechanism | `reactive_greedy` | 72 | 69.737 | 0.942 | 0.292 | 0.000 | 93.333 | 0.625 |
| active_non_mechanism | `ppo` | 72 | 65.619 | 0.655 | 0.111 | 0.417 | 65.778 | 6.778 |
| active_non_mechanism | `mappo` | 72 | 65.619 | 0.655 | 0.111 | 0.417 | 65.778 | 6.778 |
| idle_or_sparse | `sa_ghmappo` | 72 | 75.283 | 0.941 | 0.096 | 0.083 | 127.111 | 0.819 |
| idle_or_sparse | `popularity_cache_heuristic` | 72 | 75.221 | 0.930 | 0.096 | 0.000 | 152.000 | 0.917 |
| idle_or_sparse | `reactive_greedy` | 72 | 70.229 | 0.913 | 0.096 | 0.000 | 98.667 | 1.167 |
| idle_or_sparse | `ppo` | 72 | 71.091 | 0.629 | 0.100 | 0.333 | 109.333 | 7.194 |
| idle_or_sparse | `mappo` | 72 | 71.091 | 0.629 | 0.100 | 0.333 | 109.333 | 7.194 |
| non_mechanism | `sa_ghmappo` | 72 | 72.986 | 0.965 | 0.181 | 0.292 | 98.889 | 0.347 |
| non_mechanism | `popularity_cache_heuristic` | 72 | 73.885 | 0.969 | 0.000 | 0.292 | 100.667 | 0.333 |
| non_mechanism | `reactive_greedy` | 72 | 69.737 | 0.942 | 0.292 | 0.000 | 93.333 | 0.625 |
| non_mechanism | `ppo` | 72 | 65.619 | 0.655 | 0.111 | 0.417 | 65.778 | 6.778 |
| non_mechanism | `mappo` | 72 | 65.619 | 0.655 | 0.111 | 0.417 | 65.778 | 6.778 |

## ?????

- `sa_ghmappo` ? mixed overall ? reward `72.257`??? popularity heuristic `71.819`?reactive `69.840`?PPO/MAPPO `67.968`?
- `sa_ghmappo` ? full-stratified overall ? reward `75.683`???? popularity heuristic `74.835`?reactive `70.312`?PPO/MAPPO `69.762`?
- ?? trainable PPO/MAPPO????? full-stratified ? reward `75.683` vs `69.762`?success `0.815` vs `0.546`?continuity `0.831` vs `0.554`?adapter miss `2.764` vs `8.565`?? handoff failure ?? `0.148` vs `0.089`?
- ?? popularity heuristic????? full-stratified ? reward ? `0.847`?backhaul ? `15.111`?mechanism realization ? `0.0278`?
- ? idle_or_sparse ??????? reward ??? popularity?? active_non_mechanism ????????? popularity `0.899`??????? reactive/PPO/MAPPO?

## ?????

- ?? popularity heuristic???? full-stratified continuity ? `0.0238`?handoff failure ? `0.0602`?handoff_ready_ratio ? `0.0278`?
- mechanism_activating ??? popularity heuristic ? continuity ???handoff failure ?????????? reward ???? adapter miss ? backhaul ???
- ???? cache capacity disabled?`eviction_count=0`????????????? model-cache capacity ? eviction competition ?????
- HF model-cache ?? metadata-only ???????? cache hit/miss ?????? HF model-cache dataset ????????

## ??

????????? NGSIM+Alibaba ????? reward ???????? PPO/MAPPO ? reactive greedy??? popularity heuristic ?????? reward?backhaul ????????????? continuity/handoff reliability????????????????????????????????? handoff ??????????????? handoff-ready / migration-prepare ????? continuity-constrained checkpoint selection ???????????

## ??

- mixed aggregate: `artifacts\benchmarks\large_scale_real_dataset_round13\mixed_informative\main_results_mixed_informative_20260427_134259_037473\aggregate_summary.json`
- mixed rows: `artifacts\benchmarks\large_scale_real_dataset_round13\mixed_informative\main_results_mixed_informative_20260427_134259_037473\benchmark_rows.csv`
- full aggregate: `artifacts\benchmarks\large_scale_real_dataset_round13\full_stratified\main_results_full_stratified_20260427_134547_361821\aggregate_summary.json`
- full rows: `artifacts\benchmarks\large_scale_real_dataset_round13\full_stratified\main_results_full_stratified_20260427_134547_361821\benchmark_rows.csv`
- analysis summary: `artifacts\analysis\large_scale_real_dataset_round13\diagnosis_summary.json`
- agent metrics: `artifacts\analysis\large_scale_real_dataset_round13\agent_metric_summary.csv`
- pairwise deltas: `artifacts\analysis\large_scale_real_dataset_round13\sa_pairwise_deltas.csv`
- window-class metrics: `artifacts\analysis\large_scale_real_dataset_round13\window_class_metric_summary.csv`

## ????

```bash
python scripts/check_data_ready.py
python scripts/validate_dataset_source_declarations.py
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo --seed_checkpoint_manifest_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed_sa_best_continuity.json --seeds 7 13 29 --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 5000 --max_workflows 4 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 32 --window_count 6 --window_scan_stride 2 --window_selector max_handoff_candidate --window_mode mixed_informative --max_steps 20 --min_tasks 5 --max_tasks 32 --output_root artifacts/benchmarks/large_scale_real_dataset_round13/mixed_informative
python scripts/benchmark_main_results.py --agents sa_ghmappo reactive_greedy popularity_cache_heuristic ppo mappo --seed_checkpoint_manifest_path artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest_formal_round1_3seed_sa_best_continuity.json --seeds 7 13 29 --mobility_source ngsim --workflow_csv_path data/raw/workflow/alibaba2018/batch_task.csv --max_mobility_rows 5000 --max_workflows 4 --workflow_selector ordered --rsu_layout auto_dominant_tight --frame_offset 0 --window_length 32 --window_count 6 --window_scan_stride 2 --window_selector max_handoff_candidate --window_mode full_stratified --max_steps 20 --min_tasks 5 --max_tasks 32 --output_root artifacts/benchmarks/large_scale_real_dataset_round13/full_stratified
```
