# 八车四站 v2 开发集实验审查（2026-10-13）

- `reviewed_at`: `2026-10-13`（Asia/Shanghai）
- `literature_cutoff`: 未执行本轮文献检索；novelty 未评估
- `target_venue`: IEEE Transactions on Mobile Computing（仅用于证据门槛）
- `artifact_run_id`: `20261013T083022Z_development`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `1e343d18e020a067e2c81ba353e78d6099fc08f8`
- `evidence_level`: `E0_UNAVAILABLE`（论文完整证据；本轮 dev 原始记录已独立核算）
- `verdict`: `Unverifiable`（论文算法优势）；v2 开发集的确定性动作坍缩仍是 blocker

## Evidence inventory

- 预先冻结的计划和命令：`artifacts/eight_vehicle_joint_v2_development_20261012/execution_plan.json`、`command_log.json`、`preflight.log`、`training.log`。可见实验对话先因 Codex 使用额度中断，未启动训练；本对话使用冻结命令完成一次运行。没有结果驱动的重试或成本调整。
- 运行根目录：`artifacts/eight_vehicle_joint_v2_development_20261012/20261013T083022Z_development/`。`manifest.json` 标记 `v2`、`development_only=true`、仅最终 checkpoint、3 seed、每格 24 episode、每 4 episode 一次更新。
- 原始文件：504 条 train、126 次 update、21 个 SHA-256 匹配的 checkpoint、168 条 greedy dev、168 条 sampled dev、144,061 条逐请求事件；21,932 个逐帧费用回执与对应奖励重算一致。完整性核算见 `independent_audit.json`，所有 14 项检查通过。
- 8 个 dev 窗口在 seed 和算法间复用，不能把 24 行当 24 个独立窗口。无本轮 formal、holdout、support。先前 v1 formal 已被查看，不能作为 v2 的盲测。

## 开发集描述性结果

每算法 3 seed × 8 窗口；下表是每次窗口执行的均值，完成请求数分母为 8。`sample` 使用同一最终 checkpoint 的随机采样，不是新训练。

| 算法 | greedy 奖励 | greedy 完成请求 | sample 奖励 | sample 完成请求 |
| --- | ---: | ---: | ---: | ---: |
| `sa_ghmappo` | 43.748 | 5.500 | 66.330 | 8.000 |
| `ppo` | 40.944 | 5.333 | 66.292 | 8.000 |
| `mappo` | 42.396 | 5.333 | 66.346 | 8.000 |
| `dqn` | 15.503 | 2.667 | 65.938 | 8.000 |
| `dag_offload_drl` | 66.384 | 8.000 | 66.358 | 8.000 |
| `cache_offload_drl` | 66.384 | 8.000 | 66.338 | 8.000 |
| `dt_handoff_drl` | 42.396 | 5.333 | 66.337 | 8.000 |

SA-GHMAPPO greedy 的 seed 7/17/27 奖励分别为 `66.384/66.384/−1.522`，完成请求为 `8/8/0.5`。MAPPO 分别为 `−5.579/66.384/66.384`，PPO 分别为 `66.384/−9.937/66.384`。SA 对 MAPPO 的 24 行配对奖励均差仅 `+1.352`，胜/平/负 `8/8/8`；sample 均差 `−0.017`。这些行重复使用同一 8 个窗口，且跨 seed 失败位置不同，不构成主方法优势证据。

## 机制与根因边界

- SA 的 greedy 事件动作总数为 action 0 `5,620`、action 3 `7,251`、action 4 `13`；对应 `adapter_not_ready` `7,828` 次。失败 seed 的确定性策略再次长期选 action 3，未先拉取 adapter。采样评估三 seed 均完成 8/8，说明训练 episode 成功不能代表部署时的 greedy 策略成功。PPO/MAPPO/DQN/DT 也出现 seed 依赖的类似坍缩。
- 输入节点工作量全部为 `1.0`，跨帧 RSU 节点比例 `0`。模型工作集 `4,608 B` 大于各站容量 `4,096 B`，确有缓存加载和驱逐；但所有成功的 greedy 策略基本退化为 action 0，`dag_offload_drl` 与 `cache_offload_drl` 三 seed 的 greedy 结果完全相同。SA 成功两 seed 与它们的奖励也相同，无法证明 model cache 友好性更优。
- 成功 greedy 执行没有车端兜底或实际 handoff，`prepared_on_arrival` 与状态迁移字节均为零。少量 handoff 只在拖长的失败执行中出现，故预测、状态共享、surrogate 与 handoff 准备的增益没有有效机会验证。共享状态字节有费用，但不是已证明的收益。
- 同为 24 episode 不等于等交互预算：七算法训练决策数为 `15,868–20,831`。四站计算、链路、缓存字节与 Alibaba profile 工作量均为受控仿真假设；方向型算法也不是外部论文方法的忠实复现。

## Hard blockers / required actions

1. 先独立定位 greedy 与 sampled 分歧：保存 action mask、五动作 logits/概率、训练前后与 seed 间变化；检查 action 3 在 adapter 未就绪时的可行性语义、critic/actor 更新和最终 checkpoint。不能以挑 seed 或改评估规则掩盖失败。
2. 在下一份预先冻结的开发协议里用真实 profile 字段或明示的受控负载映射激活跨帧计算；量化每窗可发生的 handoff、缓存竞争、迁移准备与状态延续机会。不得看本轮胜负后调成本并将同窗口重跑称为独立证据。
3. 在同一 observation/action/reward 合同下加入消融：无预测、无状态共享、无 cache-aware 决策，以及公共 LRU/强规则和足够强的 MAPPO；分别核算 hit、eviction、backhaul、准备成功率、状态字节、连续性与完成时延。预测消融不能同时改变动作可用性。
4. 完成 checkpoint 冻结、公平交互预算、独立 formal/holdout/support、窗口外层统计和近邻文献核验后，才能重审论文 claim。当前不更新 canonical 或论文主表。

## Safe / prohibited claims

- 可写：v2 合同和逐帧账本通过局部及真实轨迹开发集完整性检查；开发集暴露 seed 依赖的贪心动作坍缩和机制机会不足。
- 不可写：SA-GHMAPPO 优于 MAPPO、具备更强 model cache 处理、预测/状态共享产生净收益、v2 结果为 paper-ready 或真实 RSU 性能。
