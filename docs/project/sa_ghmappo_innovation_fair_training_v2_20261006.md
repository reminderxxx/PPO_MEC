# SA-GHMAPPO 创新候选与纠正版小预算训练尝试（2026-10-06）

## 1. 审查元数据与结论

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-06`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `calibrated_continuous_workflow_pilot_v2_20261006`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `run_source_git_commit`: `b418eb4dbfa012e271cb97880693c714e6abf79e`
- `delivery_git_commit`: `be4c3d1bbd9f74d5af8032215e36dc280d480013`（训练后源代码与报告首次提交）
- `corrected_cost_ancestor`: `d67575bc94652f7a6e12c97315ae19c7cd42415e`
- `evidence_level`: `E1_DOCUMENTED_WITH_AUDITED_NONFORMAL_PILOT`
- `paper_ready_verdict`: `Unverifiable`
- `post_run_interface_audit`: `calibrated_workflow_interface_defect_report_20261006.md`
- `post_run_interface_audit_git_commit`: `bc5f2751706d7a7d79c445e95c0133491063f76a`
- `final_decision`: **D — 现有学习/执行接口不满足公平方法比较所需 contract；保留执行原件，不伪造训练结论。**

本轮没有恢复旧 12 点优势。纠正版基础仍是：旧 10/12 vs 7/12 和旧成本/字节优势全部撤销；对称计费下简单阈值、
完整规则、正确两步前瞻与离线参考在纠正矩阵中 12/12 同决策；真实恢复保真、依赖安全 cache transaction 和 adapter
生命周期证据继续保留。恢复二选一本身不作为算法创新。

本轮保留的唯一可证伪算法假设是：

> 在相同可见信息、相同五动作接口和相同训练预算下，DAG 节点特征中“当前/预测/目标 RSU 的 adapter 驻留关系”沿
> predecessor/successor 边的消息传递，是否能提高跨 RSU 连续 workflow 的按期完成与资源效率。

首轮执行得到完整 SA-GHMAPPO 完成率 `0.750`，关闭 dependency message passing 的同架构消融为 `0.917`，配对窗口差
`-0.167`，95% percentile bootstrap CI `[-0.250,-0.083]`。但训练后独立只读接口审计确认：hierarchical PPO 优化的
canonical-head log-prob 与实际执行 env-action distribution 不同，确定性 multi-head 聚合对 action 4 有结构性偏置；失败时
mobility 仍被冻结，flat/graph encoder 对公开 byte/typed dependency/cost 字段的消费也不对等。因此这些数值是保留的
**实现诊断结果**，不能作为公平方法排名或机制因果结论。该机制仍不得进入论文正向贡献列表。

## 2. 最近邻差异矩阵

本轮只选择六篇最相关的一手论文。`全文已核验` 表示本轮读取正式全文；`出版页核验` 表示只把出版页可确认内容用于
矩阵，未把全文细节写成已核事实。“未找到同一组合”不作为首次性证明。

| 最近邻与核验范围 | 研究问题 | 决策变量 | 可见信息 | 模型与状态依赖 | 时间耦合 | 方法 | 我们相同的部分 | 唯一候选新增部分 | 新增部分成立所需实验 |
|---|---|---|---|---|---|---|---|---|---|
| [Dual Dependency-Aware Collaborative Service Caching and Task Offloading in VEC](https://doi.org/10.1109/TMC.2025.3573379), TMC 2025；出版页核验 | 车联网中 task/service 双依赖下的协同缓存与卸载 | service caching、task offloading | task/service dependency、节点状态 | service dependency 与 DAG task dependency | 动态协同优化 | GGRN、active/passive hierarchical caching、PPO | DAG 依赖、缓存、卸载、PPO | 仅候选：连续 workflow 中 adapter-residency 节点关系沿 DAG 传播并服务 handoff continuity | 公平 no-message 消融、独立来源组、正式 checkpoint 与强规则/MAPPO；本轮被接口错配阻断 |
| [AWTO](https://doi.org/10.1016/j.future.2026.108415), FGCS 2026；出版社全文章节核验 | heterogeneous edge 上 LLM agentic workflow 的 latency-aware offloading | 逐 task 设备选择与动态模型加载 | task dependency、设备异构、实时 memory/cache | workflow task 对模型需求与 shared model cache | task-by-task MDP | 三模块 LSTM + PPO | DAG workflow、模型加载、cache-aware sequential scheduling | mobility/handoff 下 typed base/adapter 与 application-state 的连续性关系 | 移动造成的额外困难、同信息基线、模型/状态分项成本、来源时间独立 split；不能只“应用到 VEC” |
| [S-LoRA](https://proceedings.mlsys.org/paper_files/paper/2024/file/906419cd502575b617cc489a1a696a67-Paper-Conference.pdf), MLSys 2024；全文已核验 | 大规模并发 LoRA adapter serving | batch、adapter/KV paging、adapter prefetch | waiting queue、adapter/KV residency | shared base + per-request adapter/KV state | 预测下一 batch 并预取 | Unified Paging 与调度系统 | shared base、adapter residency、prefetch 是复用基础技术 | 不把 shared base 或 prefetch 当原创；候选只涉及 DAG/handoff 关系决策 | 真实 adapter 生命周期、同信息消融、独立 trace；S-LoRA 自身 workload 不能冒充真实 VEC adapter trace |
| [Multi-Agent DRL With Trajectory Prediction for Task Migration-Assisted Offloading](https://doi.org/10.1109/TMC.2025.3539945), TMC 2025；出版页核验 | 移动 VEC 的迁移辅助卸载 | vehicle-level offload、migration、resource allocation | 多步 trajectory prediction 与网络/资源状态 | task migration state | 轨迹预测驱动跨时迁移 | Informer + 两阶段 multi-agent DRL | 预测、迁移、MARL 与 mobility 已有 | 当前只能说 controller-level 三控制头与 DAG/adapter/application-state 的具体 contract 不同 | vehicle/RSU agent contract 若要声称更强必须真实实现；当前禁止写成 full vehicle/RSU MARL |
| [A service cache-based dynamic collaborative task migration technology](https://doi.org/10.1186/s13638-026-02623-8), JWCN 2026；Springer 开放全文已核验 | 动态协作集群中的预测迁移与 service pre-cache | cluster、prediction radius、service cache | mobility prediction、service/cache 状态 | service cached/not cached | 迁移前预缓存与迁移成本 | C-DDQN + FIFO/LRU/LFU/Random/GDS/DQN 对照 | prediction + service cache + migration + DRL 已有 | 连续 DAG frontier、typed base/adapter dependency 与 application-state 的关系表示 | 高速 VEC、估计误差、公平单因素消融；本轮未建立有效性 |
| [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Conference.pdf), NeurIPS 2022 Datasets and Benchmarks；全文已核验 | cooperative DEC-POMDP 强基线与实现因素 | 各 agent 的局部动作 | local actor observation、centralized critic state | 环境 agent 状态 | episodic credit assignment | MAPPO/IPPO | PPO、centralized critic、multi-head controller credit 是复用基础 | 不声称 MAPPO/CTDE 原创；只检验 domain relation message passing | 与同预算 PPO/MAPPO、公平调参和真实 agent contract；本项目当前只有 controller-level MAPPO |

独特性结论：**尚未建立独特性。** 最近邻已分别覆盖 DAG/service 双依赖、workflow 模型加载、shared-base adapter
prefetch、预测迁移和 MAPPO。唯一剩余候选是这些已知对象在当前合法 controller contract 中的特定关系表示，但它既可能
只是已知图消息传递的领域特征化，本轮单因素消融也显示负贡献。

## 3. 当前 SA-GHMAPPO 的真实能力核对

| 项目 | 当前真实实现 | 允许表述 | 禁止外推 |
|---|---|---|---|
| DAG / 依赖表示 | semantic state 提供完整 nodes、edges、execution order、completed/current/frontier；`DAGGraphEncoder` 做两轮 predecessor/successor message passing | DAG-aware encoder；节点特征含 current/predicted/handoff-target RSU 的 adapter-ready bit | 不称新 GNN；不称通用 workflow optimizer |
| 控制角色 | slow/cache、fast/execution、event/handoff 三个 controller actor heads，经固定 aggregation 映射到一个五值环境动作 | controller-level hierarchical MAPPO/CTDE | 不是每车、每 RSU 独立 agent；没有并行 joint action |
| 预测输入 | predicted RSU sequence、first handoff target、contact budget、confidence/uncertainty、当前 resident-derived future load、estimated link Mbps | 决策时可用的预测与估计 | 不读取 actual future link cost；不声称 learned mobility predictor |
| critic 与信用分配 | centralized critic；按 aggregation reason 生成 controller-head credit weights | controller-level centralized critic 与 head credit | 不是 COMA、full counterfactual credit 或 vehicle-level credit assignment |
| 动作聚合与合法性 | action mask 约束五动作分布，multi-head 输出再映射到 action 0–4；本轮 `raw_policy` 关闭 safety/heuristic override | 五动作合法 mask、raw deterministic evaluation | 训练后审计已确认 head log-prob 与 executed-action 概率错配、确定性 action-4 聚合偏置；不能称 matched learning contract |

五个合法动作分别是：当前 RSU cache fill、预测下一 RSU 对**当前节点所需 adapter**的 prefetch、vehicle fallback、当前
RSU steady offload、为预测 handoff target 准备**当前节点 adapter bundle + 当前节点完成后的 state package**。没有
future-adapter-specific prepare、远端继续执行、共享请求队列、共享无线/计算竞争或跨 workflow persistent cache。

## 4. 环境纠正与问题非退化诊断

v2 在历史 v1 之上完成了以下局部纠正：

1. action 4 先暂存迁移，只有当前节点服务成功后才提交 application state；当前节点失败不会伪造 migration success 或
   state bytes。
2. 每个实例分离 `estimated_mbps` 与 `actual_mbps`：policy 和两条在线规则只见估计值，环境执行才用实际值；不存在把真实
   未来成本交给规则的 offline-reference 泄漏。
3. reward 只含单次 node completion、单次 terminal workflow completion、time/transfer/failure/deadline 分项；不奖励
   prepare 次数、切换次数或每步正偏置。
4. 即时与两步规则都通过 estimated decision model、合法 action mask 和同一对称 bytes/time 公式预览。

但训练后接口审计确认本轮未闭合的 blocker：

1. hierarchical PPO 的 actor loss 默认消费 canonical controller-head log-prob，而实际动作从 masked five-action
   distribution 采样/取 argmax；优化对象与执行动作不同口径。
2. 确定性 five-action distribution 会把 event=prepare 的概率质量集中给 action 4，却把 keep 质量分摊到 action 0–3，
   可在各 head argmax 本应聚合为 action 0 时改选 action 4。
3. mobility/contact 仍由 `node_index` 派生；服务失败只推进 step/clock，不推进 mobility，可能形成重复失败闭环。
4. flat actor 不消费 cache readiness/occupancy；flat critic 的 occupancy 用 adapter count 除 byte capacity；graph/flat 均未
   对等消费 typed base dependency、byte occupancy、link estimate、state bytes 和 model size。
5. 两条规则拥有精确 environment clone transition 和 lexicographic objective；学习方法只有 semantic encoder 和 scalar
   reward。规则使用 decision-time estimate，未读 actual future cost，但它是更强的 model-based planner capability，不能称
   与 model-free actor “信息能力相同”。

首次诊断只看 28 个实例的初始状态，发现两条规则都选 action 0，状态为 `fail`；该现场保留在
`artifacts/calibrated_continuous_workflow_non_degeneracy_20261006/diagnostic.json`，未覆盖。覆盖不足纠正后，对 depth-4
可达状态做检查：

| 检查 | 结果 |
|---|---:|
| 可达状态 / 合法状态—动作 | 1,183 / 4,962 |
| 每个检查状态存在不同 action 后果 | 1,183 / 1,183 |
| 对称传输公式最大绝对误差 | 0 s |
| semantic state 中 `actual_mbps` 泄漏 | 0 |
| 即时规则动作计数 | action 0/2 = 713/470 |
| 两步规则动作计数 | action 0/2/4 = 558/332/293 |
| 即时与两步规则分歧状态 | 406 |

这些结果只说明问题不是“所有状态同一个独立动作”的明显退化问题，不证明 RL 优势或最优策略复杂性。

## 5. 冻结 workload、信息权限与预算

### 5.1 数据卡

| 类别 | 字段 | 边界 |
|---|---|---|
| 真实来源 | NGSIM 冻结窗口的 segment/time/frame 与 handoff pressure；Alibaba DAG nodes/edges/duration/plan memory | 两源配对是人工的，不是现实中的同一请求 |
| 实测校准 | base/adapter resident+transfer bytes；本机 load time；state/input bytes；state restore、node compute、prefix recompute | 同机测量，不是无线或真实 RSU 测量 |
| 合成字段 | adapter assignment、第二 base identity、state scale、200/1000 Mbps actual/estimate、20 ms fixed link、5 s 决策尺度、deadline、轨迹—workflow pairing | 必须称“实测校准合成工作负载” |
| 明确未用 | 新模型 generate、下载、旧 consumed holdout、文献填充值 | 调用数均为 0 |

冻结 train/dev/evaluation=`12/4/12`。按同一 `source_segment_id` 的原始 inclusive frame interval
`[frame_offset, frame_offset+window_length-1]` 复核，train–dev、train–evaluation、dev–evaluation 的跨 split 重叠均为
0。evaluation 仍不是 hidden/formal holdout。

八个模板事前覆盖 shared/distinct base、tight/ample cache、state scale `1/64/1,000,000`、competitor/all-ready/empty、
200/1000 Mbps 与 matched/optimistic/pessimistic estimation error；保留规则足够、恢复不划算和无竞争区域。模板按 ordinal
绑定，不按任何方法结果筛选。

### 5.2 方法、信息权限与事后失效边界

- 学习方法：SA-GHMAPPO、PPO、controller-level MAPPO。
- 单因素消融：`sa_ghmappo_no_dependency`，只设 `use_dependency_aware=False`；节点、cache、预测输入、动作、奖励、预算
  均相同，仍保留 node-level adapter residency bits，只关闭 DAG edge message passing。
- model-based 在线规则：immediate cost rule、correct two-step cost rule；两者只使用 estimated decision model，不读
  actual execution rate，但通过 exact clone 枚举候选并优化 lexicographic objective。
- PPO/MAPPO 使用 flat encoder，不与 SA 的 relational representation 同结构；事后审计又确认 flat/graph 的公开字段消费
  不对等，故 SA 对 PPO/MAPPO 的差异不能归因于方法优劣。
- full-vs-no-dependency 保持同一 defective hierarchical action/likelihood contract，只差 dependency message passing，
  因而可以作为“在该实现中的负向敏感性信号”，但不能升级为已验证的公平机制消融。

### 5.3 预算与选模

- 固定 seeds：7、17、29；每个学习方法 128 episodes；每 episode 最多 24 steps；8 episodes/update。
- 事前 candidate episodes：32/64/96/128。
- dev 统一 lexicographic selection：completion 高；deadline/service/handoff failure 低；modeled time、transfer 低；reward 高。
- evaluation split 不参与选模；每方法只有一套配置，无超参数搜索。
- 主比较上限 27,648 steps；消融上限 9,216；总上限 36,864；实际 main/ablation/total =
  `8,154/2,873/11,027`，未因排名扩预算。
- selected episodes：SA `32/32/128`，PPO `32/32/64`，MAPPO `96/32/64`，no-dependency `32/128/32`。

## 6. 执行结果（实现诊断，不是匹配方法排名）

数值先在每个 frozen evaluation window 内平均学习方法的三个 seed，再以 12 个 window 为外层单元做 5,000 次 percentile
bootstrap。重复 seed row 不当作独立样本。未完成 run 的 elapsed 是截断前累计值，不能解释为“更快”。
由于上述 actor/executed-action、mobility 和 encoder contract 缺陷，下表不得进入支持算法优劣的论文主表。

| 方法 | 完成率 ↑ | 节点覆盖 ↑ | deadline violation ↓ | elapsed/truncated (s) ↓ | transfer (MB) ↓ | recompute (s) ↓ | 决策开销 (ms/episode) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 0.750 [0.667,0.833] | 0.825 [0.763,0.893] | 0.556 [0.361,0.750] | 82.715 [58.034,115.666] | 641.242 [214.267,1161.152] | 42.221 [25.876,59.535] | 5.872 [4.440,7.578] |
| no-dependency | 0.917 [0.833,1.000] | 0.945 [0.882,1.000] | 0.667 [0.417,0.889] | 91.973 [64.532,124.864] | 690.450 [192.031,1330.623] | 51.250 [32.127,71.506] | 3.690 [2.855,4.723] |
| controller-level MAPPO | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 0.750 [0.500,1.000] | 104.189 [68.070,151.077] | 825.278 [235.664,1508.347] | 60.035 [35.639,87.903] | 2.391 [1.982,2.873] |
| PPO | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 0.750 [0.556,0.917] | 95.973 [66.889,133.241] | 550.810 [160.822,999.693] | 40.263 [23.852,58.580] | 1.356 [1.132,1.617] |
| immediate cost rule | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 0.417 [0.167,0.750] | 59.567 [46.399,74.778] | 27.651 [1.380,65.846] | 2.038 [0.719,3.477] | 15.570 [11.629,20.264] |
| correct two-step rule | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 0.167 [0.000,0.417] | 52.300 [36.404,73.562] | 169.034 [30.374,376.159] | 4.016 [1.798,6.174] | 72.512 [51.824,99.240] |

### 6.1 全部 seed

| 方法 / seed | completion | coverage | deadline | elapsed (s) | transfer (MB) | recompute (s) | reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| SA / 7 | 1.000 | 1.000 | 0.750 | 104.189 | 825.278 | 60.035 | 5.947 |
| SA / 17 | 1.000 | 1.000 | 0.750 | 104.189 | 825.278 | 60.035 | 5.947 |
| SA / 29 | 0.250 | 0.476 | 0.167 | 39.766 | 273.170 | 6.594 | -23.642 |
| no-dependency / 7 | 1.000 | 1.000 | 0.750 | 104.189 | 825.278 | 60.035 | 5.947 |
| no-dependency / 17 | 0.750 | 0.836 | 0.500 | 67.539 | 420.795 | 33.681 | -4.516 |
| no-dependency / 29 | 1.000 | 1.000 | 0.750 | 104.189 | 825.278 | 60.035 | 5.947 |
| MAPPO / 7,17,29（各自相同） | 1.000 | 1.000 | 0.750 | 104.189 | 825.278 | 60.035 | 5.947 |
| PPO / 7 | 1.000 | 1.000 | 0.750 | 79.541 | 1.874 | 0.719 | 8.303 |
| PPO / 17 | 1.000 | 1.000 | 0.750 | 104.189 | 825.278 | 60.035 | 5.947 |
| PPO / 29 | 1.000 | 1.000 | 0.750 | 104.189 | 825.278 | 60.035 | 5.947 |

raw deterministic action 总计进一步显示 policy collapse：SA action 0/3/4=`180/10/120`，no-dependency
action 0/3=`197/75`，MAPPO action 0/4=`222/24`，PPO action 0/2/3/4=`154/74/10/8`；即时规则 action 0/2=`41/41`，
两步规则 action 0/2/4=`52/20/10`。

## 7. 候选机制敏感性与来源定位

full SA minus no-dependency 的 12-window paired bootstrap（只描述当前实现）：

| 指标 | 差值 | 95% CI | 解释 |
|---|---:|---:|---|
| completion | -0.167 | [-0.250,-0.083] | 当前 defective contract 下的负向敏感性；不是 paper-grade 因果结论 |
| node coverage | -0.120 | [-0.192,-0.050] | 同方向实现信号 |
| service-failure episode rate | +0.167 | [+0.083,+0.250] | 完整方法更常出现服务失败 |
| reward | -6.375 | [-10.336,-2.596] | 机制未转化为训练目标收益 |
| elapsed | -9.258 | [-17.747,-2.475] | 由更多未完成/截断污染，不能解释为效率优势 |
| deadline violation | -0.111 | [-0.194,-0.028] | 同样受未完成 episode 未进入 terminal deadline 记账影响，不是优势 |

分层没有出现可用于保留正向 claim 的稳定区域：shared/distinct 的 full SA completion 为 `0.762/0.733`，消融为
`0.905/0.933`；tight/ample 为 `0.733/0.762`，消融为 `1.000/0.857`。`matched_slow` 与 state scale
`1,000,000` 各只有 1–2 个独立窗口，仅能描述，不能做 subgroup claim。所有层中两步规则 completion 都是 1.0。

可定位的 first-order blocker 是**学习/执行接口错配与聚合偏置**，不是环境无序列性：完整 SA 的两个 seed 退化到与
MAPPO 相同的高 recompute、高 model-transfer 行为，seed 29 又发生大量未完成；当前 contract 没有可靠证明依赖消息传递
能否学出规则在可达状态中采用的 0/2/4 条件切换。按本轮边界只记录并停止，不追加 reward trick、guard、重训或数据筛选。

## 8. 正负结果、限制与论文边界

本轮正向结果仅限：

- 已把实测对象大小/本机 load+restore+recompute 与合法 typed cache/state transition 统一给所有方法。
- 修正了 action-4 state commit 和 actual/estimated cost 信息边界；诊断证明 frozen 问题有动作后果与短时序耦合。
- 完成了一次 3-seed 小预算执行、dev 选模、evaluation、两条 model-based 在线规则和单因素敏感性臂，并保留全部原件。

负向结果：

- 观测数值中 SA-GHMAPPO 没有优于 PPO、controller-level MAPPO 或正确成本规则；接口不公平使方法排名不可验证。
- 唯一候选机制在同一 defective hierarchical contract 下呈负向敏感性；不能声称依赖消息传递有效或无效，更不能声称独特且有效。
- 规则显著更慢的 decision overhead 是 Python clone/enumeration 的描述性开销；没有统一硬件隔离或系统级 latency benchmark，
  不应转化为部署 claim。

未覆盖风险：3 seeds、128 episodes 可能未收敛；只有 12 个 evaluation windows；percentile bootstrap 而非 BCa/Holm；
网络与 pairing 合成；没有共享 queue/带宽/compute contention、跨 workflow cache、真实 RSU、真实 adapter request trace、
formal/hidden holdout/support、能耗与端到端 wall-clock。外部论文两项只核验出版页，因此更细的方法差异待全文再核。

最终决策为 **D**：环境的物理状态转移具有序列能力，但现有学习/执行接口不具备公平比较所需的 executed-action likelihood、
mobility progression 和对等 feature contract。本轮只可作为 interface-blocked diagnostic；不得扩大验证寻找胜出区域，也不得把
旧 v1、旧 action-4 提前提交或旧离线信息权限结果并入当前结论。未来另立任务时必须先独立修复并验证上述 contract，再冻结
全新训练；保留本轮 workload、checkpoint、evaluation 和 failure artifact，不以本轮结果反向选场景。

## 9. 证据位置

- 冻结配置：`configs/experiment/calibrated_continuous_workflow_pilot_v2.json`
- 冻结 manifest：`configs/experiment/calibrated_continuous_workflow_pilot_v2_manifest.json`
- 诊断与 split audit：`artifacts/calibrated_continuous_workflow_non_degeneracy_20261006/`
- 训练 artifact：`artifacts/calibrated_continuous_workflow_pilot_v2_20261006/`
- command log：`artifacts/calibrated_continuous_workflow_pilot_v2_20261006.log`
- completion receipt：`artifacts/calibrated_continuous_workflow_pilot_v2_20261006/completion_receipt.json`
- 逐行结果/曲线/分层/checkpoint hashes：训练 artifact 内 `evaluation_rows.*`、`training_curves.*`、
  `stratified_results.*`、`checkpoint_selection.json`、`training_summary.json`、`artifact_integrity.json`。
