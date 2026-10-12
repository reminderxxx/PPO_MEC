# 八车算法对比有效性纠正审查（2026-10-12）

| 字段 | 值 |
|---|---|
| `reviewed_at` | 2026-10-12 07:48 UTC |
| `literature_cutoff` | 2026-10-12；只复核已有文献表中的近邻论文官方页面，不作穷尽检索 |
| `target_venue` | IEEE TMC（项目内部严格审查口径） |
| `artifact_run_id` | `20261012T073902076174Z` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| Git commit | 审查输入 `ded2308`；被审运行代码 `05f7ec2` |
| evidence level | 诊断产物完整性已核；正式算法对比缺主方法及同合同基线，为 `E0_UNAVAILABLE` |
| verdict | **该 run 不具备 SA-GHMAPPO 算法对比资格；论文结论 `Unverifiable`。** |

本轮只读审查了代码、manifest、逐请求结果与已记录的近邻论文；没有改算法、调参、筛窗、重跑或改写历史 artifact。
这是对 2026-10-12 八车算法池的用途纠正，不否定其八车并发环境和日志联调已经成功。

## 一、首要问题：被比较的问题变了

`MultiRequestReplayEnv.step()` 只接受每辆车的 `target_rsu_id`。adapter 缺失时，环境自动拉取并按 LRU 驱逐；
换站时，环境自动进行固定帧数的状态传输；服务节点也由环境自动推进。策略无法决定 cache placement、
future-adapter prefetch、handoff preparation、state migration 或 DAG 节点调度。
因此 `scripts/benchmark_eight_vehicle_algorithms.py` 比较的是 **受覆盖约束的逐车 RSU 选择策略**，
不是 PPO_MEC 主线的 cache/offload/handoff 联合控制。即使将训练 seed 或 episode 再扩大，也无法从该动作空间
测出 SA-GHMAPPO 的机制收益。

`src/agents/registry.py` 的主方法 `sa_ghmappo` 使用 `semantic_discrete_5_multi_head`；正式 `mappo` 是
controller-level CTDE，`ppo` 和 DQN 家族也使用五动作语义接口。新 `trace_ppo`、`trace_mappo` 和
`TraceDrivenMultiRequestQNetwork` 是另外写的四站选址策略；其 checkpoint 不是原算法 checkpoint，
不能把名称前缀删掉后称为正式 PPO、MAPPO、DQN 或 SA-GHMAPPO 的实验结果。

run `20261012T073902076174Z` 的 18 checkpoint、5,760 episode、240 eval 行与 58,861 事件可用于检查
并发回放、合法动作、训练循环和记账一致性。其奖励排行、完成率和训练曲线**不得进入算法主表、摘要、
baseline 胜负或机制贡献论证**。原始文件保留，附此纠正审查；不覆写历史数字。

## 二、真正的对比算法应分三类

| 类别 | 应选对象 | 该对象能检验什么 | 当前状态 |
|---|---|---|---|
| 通用强化学习代表 | PPO、controller-level MAPPO；视共同动作合同纳入 DQN 和 QMIX/MAT | 区分策略梯度、集中 critic、值分解/注意力等算法结构；必须同观测信息、合法动作、奖励、预算和评估器 | 仓库已有原算法，但只接现有 controller 环境；八车入口尚未接通。DDQN/dueling 只是 DQN 近变体，不是四个主要科学问题对照。 |
| 本问题的方向匹配方法 | `dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`；再加同信息/同参数量的 flat controller 与关机制消融 | 分别检验 DAG、缓存、handoff 特征及主方法结构的增量价值 | 前三个是仓库内方向型实现，**不是外部论文原算法复现**；当前也只接五动作 controller 合同。flat matched 对照与关键消融仍缺。 |
| 同方向论文方法 | COHCTO、LDMCO、dependency-aware migration；按实验问题增加 DRLCL/DRIVE、adapter-cache 子问题的 POLAR | 正面应对 DAG+service cache、DAG+MARL、跨站迁移、多 RSU 负载与 adapter routing 等近邻方案 | 文献已定位，当前仓库没有经过原论文方法核验、动作映射与资源模型对齐的复现。不能把仓库方向型 agent 改名充当论文算法。 |

官方论文页面支持上述问题定位：[COHCTO：task/service 双依赖缓存和 PPO 协同卸载](https://ieeexplore.ieee.org/document/11014496/)、
[LDMCO：DAG 时序/数据依赖、LST 资源分配与 MADDPG 卸载](https://ieeexplore.ieee.org/document/11304528/)、
[依赖感知无回传迁移](https://ieeexplore.ieee.org/document/10243639/)、
[DRLCL：多 RSU 通信资源与负载均衡](https://ieeexplore.ieee.org/document/10960753/)、
[DRIVE：TD3 动态资源感知卸载](https://ieeexplore.ieee.org/document/11594938/)。
这些论文已在 `literature_reference_table.md` 登记；官方摘要足以确定比较方向，不足以声称已按原文忠实实现。

## 顶刊审查门禁

按 `tmc_review_policy_v3_20260621`，先看 artifact：`manifest.json`、`results.json`、18 个 checkpoint 和
`audit.json` 可核，执行 commit 为 `05f7ec2`；但没有原主方法 checkpoint、八车联合控制协议、正式/holdout/support
结果。第二步即发现 baseline/action contract 不匹配，足以停止论文性能审查。八个评估窗口在三个 seed 间复用，
外层独立单元至多八个；先前的均值不是正式统计检验。cache、handoff 和 migration 没有策略可控动作，
也无对应机制增益可审。文献最近邻见上一节，均未在本 run 实现。

| 审查维度 | 得分 | 证据与不评分原因 |
|---|---|---|
| Novelty 与近邻文献定位 | N/S | 官方论文页与 `literature_reference_table.md` 仅支持选对照方向；本 run 不包含论文方案。 |
| 技术正确性与问题建模 | N/S | `multi_request_replay_env.py` 的动作只有目标 RSU，无法检验主方法的联合问题。 |
| Baseline 公平性与独立性 | N/S | manifest 明示 `original_registry_agents_reused=false`；SA-GHMAPPO 缺席。 |
| 实验设计、统计与 holdout | N/S | 仅八个复用评估窗口；无正式独立 holdout 或预注册同合同对照。 |
| 机制兑现与系统指标 | N/S | cache/LRU/handoff 由环境自动触发，未由策略选择；没有机制因果比较。 |
| 鲁棒性、泛化与可扩展性 | N/S | 单跨源组合、四逻辑站，缺系统 stress 与独立场景。 |
| 可复现性与 claim 完整性 | N/S | 诊断 checkpoint/事件有完整性核验，但缺论文主方法与正式 package。 |

因此不打总分，不输出算法优胜、TMC-ready 或 paper-ready 判断；正式 verdict 为 `Unverifiable`。

## 三、下一次有效运行的最小门槛

1. 先固定研究问题和共同环境：若主张八车四站并发，环境必须让策略实际控制 cache、执行/卸载与 handoff/migration，
   而不是只选站。写清多车 joint action、容量竞争、合法 mask、状态/adapter 成本和逐请求终点。
2. 先让真正的 `sa_ghmappo` 在该合同上完成训练与评估；再接同合同 PPO、MAPPO、至少一个 DQN/QMIX 类代表、
   三个问题方向对照和方法结构消融。任何论文方案须先核验方法细节、公开实现/复现范围与动作映射；不能用同名替身。
3. 冻结共同 NGSIM/Alibaba 窗口、请求/DAG 映射、train/dev/eval 原始时间区间、seed、训练交互预算、
   checkpoint 选择规则与成本模型。窗口至少按独立原始区间统计；重复 seed 不增加外层道路样本数。
4. 主指标用完整 workflow 完成/连续性、有效 cache 命中、handoff readiness/failure、端到端时延可用性及
   transfer/backhaul/计算代价；奖励仅作训练和辅助指标。机制机会不足或全部方法完成率相同，应报告不可判别，
   不继续用微小 reward 排名代替结论。

在这些门槛满足前，现有五动作 native 算法可以在**原 controller 问题**上做有边界的开发诊断；
它与八车选站 run 不能合并成同一比较表。正式独立 holdout 也仍缺失，不能由本次八窗口替代。
