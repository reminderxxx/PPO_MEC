# Mechanism Completion 诊断与后台启动修复

## 审查元数据

| 字段 | 值 |
|---|---|
| `reviewed_at` | `2026-09-28T08:20:11+08:00` |
| `literature_cutoff` | `2026-09-28` |
| `target_venue` | `IEEE Transactions on Mobile Computing (TMC)`；实际投稿目录未指定 |
| `artifact_run_id` | `mechanism_completion_diagnostic_v1_20260928_v2`；`mechanism_factorial_pilot_v2_20260928_v2`；训练根为 `mechanism_algorithm_retraining_v2_20260928` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| `implementation_git_commit` | `da001e7553bdb35403cc5bce894af6e9fdc007d8` |
| `evidence_level` | `E1_CONTROLLED_OBSERVED_DATA_DIAGNOSTIC_NOT_HOLDOUT` |

## v1 后台启动失效

`artifacts/training/mechanism_algorithm_retraining_v1_20260928/` 没有启动任何训练。旧 launcher 对配置的
`/Users/howen/Projects/PPO_MEC/.venv/bin/python` 调用了 `Path.resolve()`，把虚拟环境入口冻结为 macOS
CommandLineTools Python 3.9；后台进程在导入 `yaml` 时立即退出。审计时原 PID `24188` 已不存在，且没有
`state.json` 或 `completion_receipt.json`；训练 episode、update、checkpoint 均为 0。该目录新增
`startup_failure_audit.json` 后保持不可覆盖、不可原地 retry、不可引用为运行中或已完成训练。

runner v2 保留配置中的 venv 入口路径，不再解引用 interpreter symlink；启动前用该精确入口导入 `yaml`、
`torch`，记录 `sys.executable`、`sys.prefix` 与版本，并导入训练入口。后台 wrapper 必须发布实际 PID、解释器身份和
`RUNNING` state，launcher 才返回成功；所有启动期或 child 非零退出都必须生成终态 receipt。预算还需满足
`agents × episodes_per_agent = total_training_episodes` 和
`total_training_episodes × max_steps = total_environment_step_cap`。

## 12-step completion 根因与 20-step 门槛

冻结 workload 含 9、17、5、10 节点的四个 Alibaba DAG，而 v1 的 `max_steps=12` 使 17 节点 `j_8` 在每个窗口
必然右截断。把 horizon 设为预先存在的 `max_tasks=20` 不读取策略 outcome，并且仍小于 24-frame mobility window。
48 个诊断单元使用 3 个冻结窗口、4 个 workflow、12/20 两个 horizon 和两个固定规则；两个规则在同一 horizon 内
的 request exposure fingerprint 完全匹配。`handoff_first_feasibility` 只读取当前 semantic state、prediction 与
action mask，不访问未来 trace 或 outcome；它未加入 live registry，也不是论文 baseline。

| max steps | 固定规则 | 单元 | completion | right-censored | noncensored failed | request failures |
|---:|---|---:|---:|---:|---:|---:|
| 12 | popularity | 12 | 2 / 12 | 3 | 7 | 21 |
| 12 | handoff-first feasibility | 12 | 1 / 12 | 3 | 8 | 32 |
| 20 | popularity | 12 | 4 / 12 | 0 | 8 | 21 |
| 20 | handoff-first feasibility | 12 | 3 / 12 | 0 | 9 | 30 |

因此旧 completion=0 不能只解释为 horizon 问题：20 步消除了结构性右截断且出现真实完成，但仍保留大量非截断请求
失败。handoff-first 规则没有优于 popularity；其作用仅是证明另一个无学习、无未来信息的可执行对照也能在 20 步
完成 DAG，不能作为策略改进结果。

## 20-step 四臂复制

canonical artifact 为 `artifacts/analysis/mechanism_factorial_pilot_v2_20260928_v2/`，包含 3 个冻结非重叠窗口 ×
4 个 workflow × 4 臂，共 48 episode；八项事件检查均为 true。

| base sharing | state migration | episodes | completion | ready / continuity | MB/request | backhaul | evictions/episode |
|---|---|---:|---:|---:|---:|---:|---:|
| on | on | 12 | 0.333333 | 0.778431 | 33.133224 | 294.166667 | 1.833333 |
| off | on | 12 | 0.333333 | 0.778431 | 102.440414 | 920.833333 | 7.833333 |
| on | off | 12 | 0.166667 | 0.748529 | 31.331481 | 284.166667 | 1.833333 |
| off | off | 12 | 0.166667 | 0.748529 | 100.638671 | 910.833333 | 7.833333 |

描述性主效应为：base sharing 对 completion/readiness 为 0，对 transfer 为 `-69.307190 MB/request`、backhaul
为 `-626.666666`、eviction 为 `-6/episode`；migration 对 completion 为 `+0.166666`、ready/continuity 为
`+0.029902`，代价为 `+1.801743 MB/request` 与 `+10` backhaul。四臂差分中的描述性交互为 0。只有 3 个窗口
outer clusters，未做层级 CI 或 Holm 检验；这些数值不能称为统计显著或独立泛化。

## 获批 v2 训练合同

- agents：`sa_ghmappo`、`mappo`，各 64 episodes；共同 seed `1401`；
- workload：同一 3 个冻结窗口、4 个 workflow、ON/ON 机制臂；
- horizon：每 episode 最多 20 steps；总上限 128 episodes / 2,560 environment steps；
- update：每 4 episodes 更新、batch 32、每 4 updates 保存；
- selection：只保留 fixed-budget 轨迹和 `latest.pt`，不得增加 seed、算法、窗口、workflow 或调参；
- startup：新根 `artifacts/training/mechanism_algorithm_retraining_v2_20260928/`；只允许一次启动，不自动 retry；
  `launch_receipt.json`、`state.json` 和最终 `completion_receipt.json` 是状态权威。

启动前，SA-GHMAPPO 与 MAPPO 已分别通过 2-episode、20-step、真实 NGSIM+Alibaba、同 runtime/window/workflow
的训练链路回归，并各自产生 `latest.pt`。该回归只证明训练入口、更新和 checkpoint 写入可执行。

## Claim boundary

本轮可以支持：受控 mechanism 在观察数据上的资源/就绪 trade-off、12-step 结构性截断诊断、20-step 完成可行性，
以及一次受限 matched retraining 的启动资格。不能支持：算法优越性、独立 test/holdout、统计显著、真实 LoRA/KV
paging 性能、所有失败由 horizon 导致，或 paper-ready。训练只有在最终 receipt 为 `SUCCEEDED`、两个 child 均返回
0 后才能称为执行完成；`RUNNING` 仅表示启动确认。
