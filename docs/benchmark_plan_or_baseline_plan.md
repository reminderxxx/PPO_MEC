# Baseline Plan

更新日期：2026-04-24

用途：记录当前 baseline 对照矩阵和统一执行协议。历史 `flat_ppo` / `flat_mappo` 名称只作为 registry alias 与历史 artifact 路径保留。

## 当前算法盘点

| 算法 | registry 状态 | checkpoint | 当前结论 |
|---|---|---|---|
| `sa_ghmappo` | trainable / main method | 需要 | 主方法可训练、可评估、可 benchmark |
| `reactive_greedy` | heuristic | 不需要 | 可直接评估和 benchmark |
| `popularity_cache_heuristic` | heuristic | 不需要 | 可直接评估和 benchmark |
| `ppo` | trainable | 需要 | 基础 PPO，可训练、可评估、可 benchmark |
| `mappo` | trainable | 需要 | 基础 MAPPO-style baseline，可训练、可评估、可 benchmark；完整 CTDE 仍依赖 multi-agent wrapper |

未进入当前 live registry 的算法：

- TD3 / SAC / MADDPG：当前 `semantic_discrete_5` 动作面不支持标准连续控制。
- QMIX：缺少稳定 multi-agent discrete wrapper。

## 当前对照矩阵

最小 smoke 矩阵：

- `sa_ghmappo`
- `reactive_greedy`
- `popularity_cache_heuristic`
- `ppo`

扩展矩阵：

- 在最小矩阵基础上加入 `mappo`
- 使用多 seed、多 window 的 `configs/experiment/baseline/minimal_ngsim_alibaba.yaml`

## 统一训练与评估协议

- 数据主线：默认 `NGSIM + Alibaba`，不把 toy 结果作为正式结论。
- observation：agent 通过 wrapper 的 `semantic_state` 或既有 encoder 使用环境状态。
- action：统一 `semantic_discrete_5`，通过 `ActionSchema` / `ActionAdapter` 转换为 `cache_action`、`offload_action`、`migration_action`。
- seeds：由 `configs/experiment/baseline/*.yaml` 的 `seeds` 字段统一控制。
- checkpoint：trainable agent 必须导出 per-seed checkpoint；heuristic agent 不需要 checkpoint。
- 导出：训练写 `train.csv` / `summary.json` / checkpoint；评估写 `eval.csv` / `summary.json`；统一对照写 `comparison_summary.*` 和 `run_manifest.json`。

## 运行入口

Smoke 闭环：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml
```

最小正式配置：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/minimal_ngsim_alibaba.yaml
```

只跑某些 agent：

```bash
python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml --agents reactive_greedy popularity_cache_heuristic ppo
```

## 用户手动确认项

- 确认 `data/raw/workflow/alibaba2018/batch_task.csv` 存在。
- 确认 NGSIM 原始轨迹文件存在或可由当前 provider 自动定位。
- 正式训练前确认主方法 checkpoint 是否仍采用 `docs/project/ARTIFACT_RECORDS.md` 记录的当前路径。
- TD3 / SAC / MADDPG / QMIX 不应强行运行；若要启用，需要先冻结连续动作或 multi-agent wrapper contract。
