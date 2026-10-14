# 八车 v2 训练量诊断：24 与 96 episode（2026-10-14）

- `reviewed_at`: `2026-10-14`（Asia/Shanghai）
- `literature_cutoff`: 本轮未检索文献；novelty 未评估
- `target_venue`: IEEE Transactions on Mobile Computing（仅用于证据门槛）
- `artifact_run_id`: `20261014T035709Z_development`；对照 `20261013T083022Z_development`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `8fc6023cf2c9838e047368dbedbfd763f2e06fd8`（运行时）
- `evidence_level`: `E0_UNAVAILABLE`（论文完整证据；两轮开发集原始记录已核算）
- `verdict`: `Unverifiable`（论文优势）；“只增加训练量即可消除贪心坍缩”被本轮开发集结果否定

## Evidence inventory 与比较控制

- 冻结计划、命令和日志：`artifacts/eight_vehicle_joint_v2_budget96_20261014/execution_plan.json`、`command_log.json`、`preflight.log`、`training.log`。只把每算法每 seed 的训练从 24 episode 提至 96；保留 seed `7/17/27`、每 4 episode 更新、相同原始数据和开发窗口。本轮只运行 SA-GHMAPPO、PPO、MAPPO、DAG 与 cache 两个方向基线。
- 新 run 的 `manifest.json` 与旧 run 对比，`splits`、环境选项、原始数据 SHA-256、工作流/adapter、RSU 布局、成本与奖励定义一致。两 commit 之间 `src/`、`scripts/`、`configs/` 无改动。两轮均从各自 seed 初始化；96 episode 不是接续旧 checkpoint。逐 seed 数值见 `artifacts/eight_vehicle_joint_v2_budget96_20261014/20261014T035709Z_development/budget_comparison.json`。
- 本轮原始证据：1,440 train、360 update、15 个 hash 匹配的最终 checkpoint、120 greedy dev、120 sampled dev、100,967 条事件、15,391 个逐帧回执；`independent_audit.json` 的 14 项完整性/奖励重算检查全部通过。原 run 的对应审查见 `eight_vehicle_joint_v2_dev_review_20261013.md`。
- 两轮均只用相同 8 个 dev 窗口；正式 formal、holdout、support 未运行。不得把重复的 seed/窗口行当作独立窗口样本，也不得将本轮用于论文主表。

## 结果：增加训练量没有稳定消除坍缩

每算法 3 seed × 8 dev 窗口；表中为 greedy 均值，完成请求数分母为 8。

| 算法 | 24 episode 奖励 | 96 episode 奖励 | 24 episode 完成 | 96 episode 完成 |
| --- | ---: | ---: | ---: | ---: |
| `sa_ghmappo` | 43.748 | 40.944 | 5.500 | 5.333 |
| `ppo` | 40.944 | 42.396 | 5.333 | 5.333 |
| `mappo` | 42.396 | 66.384 | 5.333 | 8.000 |
| `dag_offload_drl` | 66.384 | 29.945 | 8.000 | 2.667 |
| `cache_offload_drl` | 66.384 | 66.384 | 8.000 | 8.000 |

- SA-GHMAPPO seed 27 从平均 `0.5/8` 降到 `0/8`，其 96 episode greedy 事件 `7,680/7,680` 次选 action 3，全部 `adapter_not_ready`。同一 checkpoint 的 sample 仍 `8/8`，包含 action 0 `1,281` 次、action 3 `1,446` 次与 `100` 次 adapter fetch。另两个 SA seed 的 greedy 仍 `8/8`。
- MAPPO seed 7 从 `0/8` 升至 `8/8`；96 episode 三 seed greedy 全选 action 0 并全部 `8/8`。DAG 基线 seed 7/17 则从 `8/8` 降至 `0/8`，96 episode 的 greedy 动作以 action 3 为主。cache 基线三 seed 两轮均全选 action 0、全部 `8/8`。
- 96 episode 的五算法 sample 均为 `8/8`；SA 对 MAPPO 的 greedy 24 行配对奖励均差为 `−25.440`，胜/平/负 `0/16/8`。该均值由单个 SA 失败 seed 驱动，不是可推广的算法排名。
- 训练采样轨迹在每算法每轮的全部请求上都完成：96 episode 每算法 `2,304/2,304` 个请求；但对应 greedy 可以 `0/8`。96 episode 已给每 seed 24 轮更新、每算法约 64,000 决策；最后一轮 PPO 类 critic 的 value mean 约 `17`、return mean 约 `36`，explained variance 仍接近零。价值学习不充分可能是因素，不能单独解释 action 0/3 的相反坍缩，也不能据此保证继续加预算会修好。

## First-order 机制诊断

- `ActionMaskBuilder` 对有节点的请求始终允许 action 3；`ActionAdapter` 的 action 3 只在当前站 offload，不触发 adapter 拉取。action 0 同样当前站 offload，并在缺 adapter 时发起加载；已经缓存时返回 `already_cached`，不占额外链路槽。因而当前工作负载存在明显的 action 0 安全默认与 action 3 冷缓存死锁。
- 采样策略下，部分车辆选 action 0，可以为同站其他选 action 3 的车辆建立缓存；所有车辆在确定性 argmax 中同时选 action 3，则没有加载发起者。团队奖励使这种搭便车反馈与部署时的全体同动作分布不同。**这是由动作与事件直接支持的解释性推断**；还需记录条件动作概率和反事实回报，才能定量归因于 credit assignment。
- 全部节点仍只需一帧，成功 greedy 轨迹几乎完全相同且没有有效 handoff 准备；缓存方向基线的稳定性不能等同于已证明的 model cache 机制优势。

## Required actions before another training budget sweep

1. 独立实现训练过程诊断：每轮固定开发状态的五动作 logits/概率、mask、冷/热缓存条件动作分布、缓存拉取者与未拉取者的回报、greedy/sample 配对结果；保存中间 checkpoint，避免仅凭最终点判断训练是否收敛。
2. 审查 action 0 与 3 的语义和成本；在所有算法共同的合同中，明确冷缓存且无人加载时 action 3 是否可行，验证修改不会凭规则给主方法单独优势。涉及 action mask 或 schema 的修改须同步生产端、训练、评估、checkpoint 消费端。
3. 预先冻结能够激活跨帧工作量、handoff 和缓存机会的新开发协议；再比较 SA 与 MAPPO、cache 基线及机制消融。当前 8 个 dev 窗口可作开发诊断，不可重标独立 formal/holdout。

可守结论：更长训练改变贪心动作，并救回 MAPPO 一个 seed；它没有修复 SA 的失败 seed。禁止表述：SA 优于 MAPPO、DAG/cache 基线已证明机制优势、增加训练量已排除算法问题。
