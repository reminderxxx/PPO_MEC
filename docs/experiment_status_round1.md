# Experiment Status Round 1

更新日期：2026-04-24

本文只记录 formal experiment execution round 1 的执行状态和 artifact 入口，不作为论文表格或论文结论摘要。

## Canonical Round1 Run

- 配置：`configs/experiment/baseline/minimal_ngsim_alibaba.yaml`
- 命令：`python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml`
- run id：`baseline_minimal_ngsim_alibaba_20260424_145836`
- 输出根目录：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/`
- benchmark aggregate：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark/main_results_mixed_informative_20260424_150518_946716/aggregate_summary.json`
- benchmark rows：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark/main_results_mixed_informative_20260424_150518_946716/benchmark_rows.csv`
- full-stratified supplement aggregate：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/benchmark_full_stratified/main_results_full_stratified_20260424_174124_983467/aggregate_summary.json`
- seeds：`7`, `13`

## 正式跑完的 Agent

- `sa_ghmappo`：使用既有 formal checkpoint 做 per-seed eval，并进入 benchmark aggregate。
- `reactive_greedy`：无 checkpoint，完成 seed `7/13` eval 和 benchmark。
- `popularity_cache_heuristic`：无 checkpoint，完成 seed `7/13` eval 和 benchmark。
- `ppo`：当前主名；历史 run 以 `flat_ppo` 路径保存，已完成 seed `7/13` 独立训练、独立 checkpoint、独立 eval 和 benchmark。
- `mappo`：当前主名；历史 run 以 `flat_mappo` 路径保存，已完成 seed `7/13` 独立训练、独立 checkpoint、独立 eval 和 benchmark。当前 contract 下可运行，仍不是完整 CTDE multi-agent contract。

## 只完成 Smoke 的内容

- 历史 smoke run 仍只用于链路验证。
- 当前优先算法池内，`ppo`、`mappo`、`reactive_greedy`、`popularity_cache_heuristic` 已有非 smoke round1 run，不再只停留在 smoke。

## Skeleton / Contract 阻塞

- TD3 / SAC / MADDPG / QMIX 当前不在 live registry 中。
- 当前 `semantic_discrete_5` 动作 schema 不自然支持标准连续控制 TD3 / SAC / MADDPG。
- MADDPG / QMIX 需要先冻结 multi-agent wrapper 和 observation/action contract。
- 本轮没有强行接入这些算法，也没有修改环境 reward、handoff、migration 或 adapter cache 语义。

## 后续正式实验主入口

- 端到端 baseline round1：`configs/experiment/baseline/minimal_ngsim_alibaba.yaml`
- 统一 runner：`scripts/run_baseline_experiment.py`
- 主 benchmark 聚合器：`scripts/benchmark_main_results.py`
- seed checkpoint 映射：`artifacts/experiments/baseline/baseline_minimal_ngsim_alibaba_20260424_145836/seed_checkpoint_manifest.json`

## 后续分析输入 Artifact

- `comparison_summary.csv`：agent × seed 粒度汇总，含 checkpoint、train/eval summary 和 benchmark aggregate path。
- `comparison_summary.json`：同上，JSON 入口。
- `comparison_summary_detailed.json`：包含 agent-level summary、by-window-class rows、benchmark aggregate 子结构和 command log。
- `comparison_summary_by_window_class.csv`：window class × agent × seed 粒度机制诊断入口。
- `run_manifest.json`：包含 `agent_seed_runs`，可直接追踪每个 agent × seed 的训练、评估和 benchmark 产物。
- `command_log.json`：本轮每个 train/eval/benchmark 子命令的 stdout/stderr 和 return code。

## 未覆盖风险

- 当前 round1 是最小正式闭环，不是最终训练预算；`ppo` / `mappo` 只证明 per-seed artifact 闭环可运行，不代表已收敛。
- 原始 `mixed_informative` run 没有选中 `idle_or_sparse`；补充 `full_stratified` benchmark 已覆盖该类窗口。
- `handoff_ready_count_mean` 当前仍为 `0.0`，需要后续扩大窗口和 horizon 后复查。
