# G14S01：创新定位、算法有效性诊断与最小补证实验设计

## 1. 审查元数据与结论

| 字段 | 值 |
|---|---|
| `reviewed_at` | 2026-09-28（Asia/Shanghai） |
| `literature_cutoff` | 2026-09-28 |
| `target_venue` | 用户暂定 B 类／B 区期刊，目录未指定；本报告不定义通用门槛，也不作“已达标”判断 |
| `policy_version` | `tmc_review_policy_v3_20260621`（按项目规则作为更严格的内部审查框架，不等同于目标期刊） |
| `artifact_run_id` | G14E07 formal：`typed_model_cache_post_ablation_20260921_g14e07_pending`；容量统计：`typed_model_cache_g14r22b_capacity_statistics_20260923_v1`；已消费 holdout：`typed_model_cache_holdout_20260926_g14r22d_once`；G14R23：`g14r23_holdout_interface_and_independent_test_20260927` |
| Git | 当前主线 `ea96d286daa3af6b6e81f818b9feeaa4bdab2f80`；formal scientific `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d` / executor `1770401ccc2f4ce15b5d5adcabdecb386fd90a05`；容量统计 artifact `d6c53b4154f84dcb398ceb0f88ace642142c0f23`、报告分支 `33d8aa8fba2361e4b0598fe193dda7ceae4816f5`；G14R23 最终报告 `5173234666ddbbb53df977a2d3a95a04f7166300` |
| Evidence level | formal 结果为有 provenance 的 E2 级部分证据；容量修正统计为针对性 E3；没有有效独立 holdout 性能结果，因此论文级算法/创新 claim 为 `UNVERIFIED` |
| Verdict | **Unverifiable / Not ready for scientific performance claims**。工程闭环较完整，但当前证据不能证明候选创新带来可重复净收益，也不能证明 B 类／B 区期刊已达标。 |

本轮是只读科学审查。没有训练、评估、调参、筛窗或补写算法；文档更新不改变任何既有 artifact。

## 2. 执行摘要

当前最多只能保留两项候选核心贡献，且两项都尚未被现有证据证实：

1. **跨 RSU 连续 DAG workflow 的 typed base/adapter cache 与 workflow-state migration 联合控制。**
   代码路径真实存在，并执行 byte-capacity、base dependency、adapter placement、handoff prepare/state
   migration 与 continuity 判定；但 formal workload 的全部 1,860 个 SA-GHMAPPO request 只请求同一
   `adapter_batch_type_1 + veh_base_v1` 组合，未形成多 adapter 共享 base 的机会。状态迁移事件也很稀疏，且
   `no_base_sharing`、`no_workflow_state_migration` 两个预注册消融不可用。因此“联合机制有效”仍为
   `UNVERIFIED`。
2. **面向 cache / offload / handoff-event 的 controller-level graph/hierarchical prediction-aware policy。**
   三头动作与语义状态确实被当前单环境控制器消费；算法身份是 **controller-level control**，不是未实现的
   vehicle/RSU-level full MARL。现有 `typed_full` 对 `no_prediction` 只显示 readiness/continuity 与
   transfer/backhaul 的权衡，G12 supervised predictor 在正式路径中关闭且 causal accepted snapshot 为 0；又缺少
   同信息、同动作、同参数量的结构匹配基线，因此算法结构贡献仍为 `UNVERIFIED`。

不能把“多个已知组件被完整集成”直接写成方法创新。公开近邻研究已经分别覆盖
`DAG + service dependency + PPO`、`DAG + VEC + MADDPG`、`mobility prediction + migration + DRL`、
`adapter cache + two-timescale control` 和 mixed-timescale VEC。当前可能有差异的是这些对象在同一跨 RSU
连续 workflow 合同中的耦合，但“交叉点尚无相同论文”不是穷尽性不存在证明，且本项目还没有机制证据证明这种
耦合本身产生价值。

## 3. 原件、统计口径与独立测试状态

### 3.1 当前可用原件

- G14E07 formal-only 根目录：
  `artifacts/experiments/typed_model_cache_evaluation_only/typed_model_cache_post_ablation_20260921_g14e07_pending/`。
  其 `complete_without_holdout` 和 completeness gate 仅证明 formal 执行完整，不是性能 gate。
- 独立 formal 复核：`docs/project/g14a01_formal_results_independent_review_20260921.md`。
- 容量修正报告位于 `origin/codex/g14r22-b-capacity-background`，报告为
  `docs/project/g14r22b_capacity_background_acceptance.md`，统计 artifact 为
  `artifacts/analysis/typed_model_cache_g14r22b_capacity_statistics_20260923_v1/`。三份输入 CSV 的容量身份由
  producer manifest、runtime MB、handoff 坐标和 SHA-256 交叉验证，不是从结果反推。
- G14R22D 一次性 holdout 的 `opening_receipt.json` 明确记录
  `consumed_permanently=true`；`execution_receipt.json` 为 `failed_permanently_consumed`。失败发生在第一档
  rollout episode 之前的 sealed-window validation interface，未形成 benchmark 性能结果。该旧 holdout 不得
  retry、resume 或 reopen。
- G14R23 位于 `origin/codex/g14r23-holdout-interface-review`。它修复并用合成窗口验证专用接口，但没有产生
  新的真实 holdout 结果。对 579 个 I-80 候选的审计在排除已用 interval 后剩 519 个，再按 vehicle recurrence
  约束只剩 14 个且全部来自同一 `i_80_run_001`；无法构造预定的多 run 独立测试，结论为
  `NO_RELIABLE_UNUSED_RANGE_ESTABLISHED`。

### 3.2 不可混用的统计版本

- 正确分类是 **0 supported、72 mixed、12 contradicted**；12 个 contradicted 均为 transfer 方向候选不利。
- 12 个 contradicted **不等于** 12 个 Holm 显著劣势。以 12 个 window 为独立单位重算后，84-family 中没有
  Holm-adjusted `p < 0.05` 的比较；transfer 对 LRU 为 `0/3/9`，raw sign-test `p=0.003906`，Holm 后
  `p=0.328125`。
- capacity-aware 10,000 次 bootstrap 保持 point estimate、window W/T/L、sign test、Holm 与分类不变；相对旧
  结果，**14/84 个 lower bound、59/84 个 upper bound 改变**。后续不得引用旧 CI，也不得把“59 个上界改变”
  改写成“59 个结论改变”。
- delay 只有 `220/540` paired rows 为 finite，覆盖 9/12 窗口；共同 finite pair 的差为 0。它只是
  completed-workflow conditional endpoint，不能支持端到端 delay 改善 claim。

## 4. 最接近研究与创新边界

只保留覆盖直接交叉点的五组近邻，不用大量宽泛论文稀释比较。

| 近邻工作 | 对方问题与机制 | PPO_MEC 可区分点 | 实现状态与组合风险 | 可证伪假设 |
|---|---|---|---|---|
| [Dual Dependency-Aware Collaborative Service Caching and Task Offloading in VEC](https://doi.org/10.1109/TMC.2025.3573379), TMC 2025 | task/service 双依赖，GGRN 关键服务，active/passive hierarchical caching，PPO 协同卸载 | 连续 workflow 跨 RSU 保持 frontier；adapter/base typed dependency；handoff state migration | 差异对象已实现，但现有 workload 没有多 adapter 共享 base，机制收益未证；属于多个已知机制的潜在新耦合，不是天然创新 | 在多 adapter 共享 base 且容量绑定时，base sharing × state migration factorial 是否提高 full-service-ready/continuity，并在 transfer/backhaul 成本后仍有可解释净收益 |
| [Optimization of Task Scheduling With Timing and Data Dependencies in VEC](https://doi.org/10.1109/TMC.2025.3646450), TMC 2026 | DAG timing/data dependency，LST-CRA，MADDPG 协同卸载 | cache warm state 与 handoff state 进入 DAG 连续执行合同 | DAG、图表示、MADDPG/MARL 均不是新颖点；本项目只能证明 cache/handoff 耦合的增量 | 在同 observation/action/budget 下，加入 typed-cache 与 state continuity 是否只在机制机会窗口改善 DAG completion，而非只提高 reward shaping |
| [Mobility-Aware Assisted DRL for Collaborative Task Migration and Resource Allocation](https://doi.org/10.1109/TVT.2026.3660321), TVT 2026 | Mamba 轨迹预测 + SAC，RSU 间 task migration 与资源分配 | workflow frontier、adapter warm cache、state payload 与三类动作联合 | 预测、主动 migration、DRL 均已有；supervised predictor 当前正式路径未激活 | prediction on/off 在相同结构下，是否提高 handoff readiness/continuity，且校准误差、错误预取、transfer cost 同时受控 |
| [Open RAN-Based Mixed-Timescale and Robust Task Offloading in VEC](https://doi.org/10.1109/TMC.2026.3657303), TMC 2026 | O-RAN 分层 large/moderate/small-timescale 资源、任务切分与无线控制 | cache/offload/event 语义三头与 DAG/cache/handoff 状态 | “multi-timescale”本身已有；当前三头是单 controller 的 factorization，不是多主体体系 | 参数量匹配的三头 graph/hierarchy 与 flat single-head controller 做 2×2 prediction factorial，结构主效应能否在机制指标而非仅 reward 上复现 |
| [POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving](https://arxiv.org/abs/2604.16583), 作者页面列为 MobiHoc 2026 accepted，ACM DL 待核验 | LoRA adapter caching/routing，two-timescale contextual bandit，真实 adapter 与 GPU paging latency | 高移动 VEC、跨 RSU handoff、DAG continuity 和 workflow state | adapter cache、two-timescale control 已被直接占据；本项目当前 catalog size/transfer 为受控值，没有真实 adapter paging latency | 在真实或可核验 adapter size/loading latency 下，移动/handoff-aware 联合策略是否优于 workload-matched cache/routing bandit，并报告系统成本 |

截至检索日，没有从作者原文/官方出版页确认一篇同时实现上述全部交叉点的论文；这只能形成
`NOVELTY_UNVERIFIED` 的候选差异，不能形成“首次”或“唯一”声明。POLAR 的 MobiHoc 2026 状态来自作者 publication
page；ACM DL 正式条目尚待核验。

## 5. 算法有效性的 first-order 诊断

### 5.1 正式结果没有证明总体优势

- formal 包含 3 容量 × 15 agents × 5 seeds × 12 windows × 3 workflows；候选与基线共享正式请求、窗口和外生
  轨迹，学习方法每个 seed/capacity 为 256 episodes、每 episode 最多 22 steps，即 5,632 interactions，固定
  32 expected updates，未 early stop。该部分执行预算可追溯。
- 当前可信结论是 0 supported、72 mixed、12 contradicted，且 84-family 无 Holm 显著项。ready/continuity 相对
  LRU 的有利方向只在 2/12 窗口；transfer 候选不利方向出现在 9/12 窗口。收益明显集中，而代价覆盖更广。
- `typed_full - no_prediction` 在 576 MB 上约为 ready/continuity `+0.003268`、handoff-ready
  `+0.033333`、reward `+1.0515`，同时 transfer/request `+3.656616`、backhaul `+34.266667`；success 与
  conditional delay 无差异。它支持 trade-off，不支持 Pareto dominance。
- 所有模型在 finite delay 上共同为零差，且 success 未改善；当前不能把 readiness 或 shaped reward 改善外推为
  任务完成与端到端时延改善。

### 5.2 baseline 公平性边界

公平部分：固定 seeds/windows/workflows/capacity，统一最大交互预算，checkpoint 选择规则预冻结，14 个 baseline
不是按结果事后挑选；formal command、checkpoint manifest 和模型 hash 可追溯。

不足部分：

- 不同算法保留自身优化器/超参数/结构，参数量、更新动力学和信息消费并未匹配；“相同 interaction 数”不等于
  结构因果公平。
- SA-GHMAPPO 独有的 graph/hierarchy/prediction/event 语义没有一个同 observation、同 action mask、同 reward、
  同参数量的 matched controller baseline。现有 MAPPO/QMIX/MAT 均是 controller-level comparator，不能写成
  vehicle/RSU-level multi-agent baseline。
- reactive baselines 对学习预算不适用，但也没有获得相同预测信息；它们适合作系统参照，不足以隔离候选算法结构。
- 现有选择规则在 dev 上按 ready/continuity/transfer/delay lexicographic 选 checkpoint，尚未证明对所有算法的
  inductive bias 同等有利。需要报告每个算法候选分布和 selection sensitivity，而不是只报告 selected checkpoint。

### 5.3 成本证据不足

同容量内的描述性 wall-clock 显示 SA-GHMAPPO inference/step 约 49–50 ms，LRU 约 34 ms，PPO 约 41 ms，
MAPPO 约 42–43 ms；SA 相对 LRU 约慢 46%，相对学习 comparator 约慢 15–22%。但 864 MB 整组 timings 在
输出完全相同的情况下几乎整体翻倍，说明宿主/批次混杂明显。当前 Python peak increment 也不等于总 RSS。
因此只能写“候选有可见推理开销，尚未严格量化”，不能形成计算效率结论。训练 wall-clock、CPU/GPU-hours、
峰值 RSS、checkpoint size 和能耗未形成统一原件。

## 6. 机制审计

分类定义：1 = 已实现、正式启用且可直接公平消融；2 = 已实现/启用，但缺公平消融或没有足够机制机会；
3 = 未实现或正式路径未启用。

| 机制 | 实际调用路径与状态/事件证据 | 分类 | 当前能否声称贡献 |
|---|---|---:|---|
| base sharing / typed dependency | `src/envs/core/vec_workflow_core_env.py` 的 placement bundle、capacity profile、typed cache action 与 readiness 检查把 base+adapter 作为 dependency bundle，禁止驱逐仍被依赖的 base；catalog 的 base 独立计入容量 | 2 | 不能。正式 SA 1,860 个 request 只访问一个 adapter/base 对，没有 sharing opportunity；`no_base_sharing` 不可执行 |
| workflow-state migration | 环境在 handoff 前记录 workflow/vehicle state readiness，event action 执行 prepare/migrate，continuity 消费该状态；576/864 MB 的 180 episodes 中 10 次 handoff、39 次 migration/prepare action、6 次成功，4 个 pre-action state-not-ready | 2 | 不能。机制真实触发但事件过稀，且 `no_workflow_state_migration` 不可执行；不能把相关性写成增益 |
| eviction policy | 环境执行真实 byte capacity 与 LRU/FIFO/LFU/size-aware 等策略；五个 reactive policy 可直接切换 | 1（policy 间）；2（eviction 有无） | policy 比较可做，但现有 576/864 MB eviction 为 0，primary outcomes 全同；288 MB 有 81 次 SA eviction，却仍缺 `fixed_no_eviction` 公平对照 |
| baseline deterministic prediction features | semantic state 和三头 policy 消费 next/target RSU、confidence/uncertainty 等字段；`no_prediction` 可执行 | 1 | 只证明可消融，结果是收益/传输代价 trade-off，不证明净贡献 |
| G12 supervised predictor | 正式配置关闭，1,860 个 pre-action record 的 predictor availability mask 全为 0，causal accepted snapshot 为 0 | 3 | 不能称为当前方法已有贡献；未来启用必须作为新实现/新实验 |
| graph/hierarchy/three-head controller | `src/agents/sa_ghmappo_core.py` 构造 digital-twin/handoff tensor并分 slow-cache、fast-offload、event-prepare 三头；checkpoint contract 为 `graph_surrogate_semantic_encoder`、`semantic_discrete_5_multi_head` | 2 | 确认已实现并消费关键状态，但无 matched flat/single-head 对照，不能归因 |

算法身份应统一写为：**一个 controller-level policy 对单个环境动作合同进行 graph/hierarchical、多头决策**。
当前没有 vehicle-agent、RSU-agent 的独立 policy/critic、通信或 joint-action contract，不得称为完整多智能体控制体系。

## 7. 576/864 MB 重复结果的根因

这不是 capacity 参数未进入约束，而是两档都落在同一个非绑定区间。

- catalog 中所有 capacity-counting typed objects 合计 856 MB；初始 RSU residency 为 A=276 MB、B=228 MB、
  C=252 MB。workflow state 为 handoff payload，但 `counts_toward_capacity=false`。
- 288 MB 下 SA cache-used 为 228–244 MB，发生 81 次 eviction/1,860 requests、1,159 次 adapter miss，
  continuity 约 0.4125；容量真实绑定。
- 576 与 864 MB 下，SA cache-used 都是 244–472 MB（均值约 382.38 MB），eviction 恒为 0，adapter miss 都是
  619/1,860，continuity 都约 0.7354。观察峰值 472 MB 小于 576 MB，因此再增加到 864 MB 不改变可行动作或
  服务状态。
- 两档 2,700 对应行的 reward、success、ready、continuity、handoff、admission、eviction、miss、used bytes、
  base/adapter/state/primary transfer、transfer/request 和三项 primary hit endpoint 全部相同；只有
  remaining-capacity 数值不同。SA 的 1,860 条 action 序列也逐条相同。
- 更根本的问题是 workload reachability：全部 request 只触及同一 adapter/base 对。catalog 总量 856 MB 不等于
  单次窗口的可达 working set，因而 864 MB 不是更强“压力档”。

结论：现有三档只识别“288 MB 绑定”与“≥576 MB 非绑定”，没有识别中间饱和曲线，也没有验证多对象竞争、
base-sharing 或 eviction policy 的作用边界。

## 8. support、scalability 与外部有效性

- prediction noise/confidence/delay/drop 与五个 reactive eviction policies 有原始 support 结果；但四个核心
  typed-semantics ablation 不可用，不能用 support 包替代机制因果证据。
- oracle 只覆盖 9-request replay、20–63 visited states；所有 cell 虽为 `optimal`，但没有 baseline-oracle gap，
  不构成规模化最优性证据。
- RSU 数、vehicle 数、DAG nodes、typed objects、adapter/base families 和 handoff density 均没有系统 scaling
  曲线。当前 576/864 的重复不是 scalability 证据。
- mobility 使用 NGSIM、workflow 来源为 Alibaba，但 adapter/base catalog、大小和 transfer cost 是受控模型；正式
  workload 又只映射到一个 adapter/base 对。真实数据来源不能自动使 adapter-cache 机制具有外部有效性。
- 旧 holdout 已永久消费且无性能结果；G14R23 也没有建立新的可靠 unused range。没有独立测试时，formal 结果只能
  用于开发/诊断，不能转写成 confirmatory generalization claim。

## 9. 最小补证实验设计（本轮不执行）

### 9.1 Stage 0：先建立“机制机会”与新的独立测试资格

这一步只看外生 workload 与机制机会，不看任何算法 outcome。

1. 获取新采集 session、不同公开轨迹源，或可证明未参与开发的新 run；不得重开 G14R22D。formal/test 必须按原始
   frame/time interval 互斥，并检查 vehicle recurrence；至少跨 3 个独立 runs，不能把同一 split 的重叠窗口当
   独立 cluster。
2. 预注册并冻结窗口资格：足够的 handoff opportunity、至少两个 adapter 共享同一 base、至少两个 base families、
   至少一个 workflow 在 handoff 期间有未完成 frontier。不能按 reward、agent 胜负或 seed 筛窗。
3. 在不运行候选算法的 workload-only replay 中估计 reachable working-set peak 分布。容量档从该分布的预注册分位
   点确定，至少包含一个明确绑定档、一个饱和转折档和一个非绑定档；不再用 catalog 总量替代 reachable working set。
4. 若无法获得满足独立性和机制机会的新数据，立即停止 confirmatory claim：保留 `UNVERIFIED`，只把现有结果写作
   受控案例研究。

### 9.2 Stage 1：候选贡献 1 的最小 2×2 机制因子实验

四个 arms：

| base sharing | workflow-state migration | 解释 |
|---|---|---|
| on | on | full candidate |
| off | on | 每个 adapter 使用同尺寸的 adapter-specific base replica；物理容量不变，隔离共享 residency 收益 |
| on | off | handoff 后执行预注册 cold restart；保留动作维度和其余 cache 语义，隔离 state continuity 收益 |
| off | off | 双机制关闭，估计交互项 |

要求：四臂分别按相同 interactions、seeds、window、checkpoint selection 重新训练；不能只用 full checkpoint 在评估期
关开关。固定 eviction policy 和信息可见性，容量作为 blocking factor。主要 endpoints 预注册为
full-service-ready、workflow continuity、完成率/失败率；transfer/backhaul、miss/eviction、conditional delay 与计算
成本作为共同报告的代价，不得用 reward 单独晋级。统计以独立 window/run 为 outer clusters，seed/workflow 为内层；
预注册 base-sharing 主效应、migration 主效应和 interaction 三个 contrasts，并对同一 endpoint family 做 Holm
校正。

### 9.3 Stage 2：候选贡献 2 的结构匹配 2×2 实验

仅当 Stage 1 证明至少一个机制存在可重复贡献后才运行：

| controller architecture | prediction |
|---|---|
| graph/hierarchical three-head | on |
| graph/hierarchical three-head | off |
| flat single-head | on |
| flat single-head | off |

四臂必须共享完全相同的 observation 信息、合法动作集合/mask、reward、交互预算、optimizer search budget 与 checkpoint
规则；参数量和 inference budget 预先设定容差并公开。prediction-on 必须报告 availability、coverage、calibration、
错误 RSU 预测和无效预取率。若 supervised predictor 加入，这是一项新机制，不能回溯称为旧 formal 已有贡献。

主要假设是 architecture main effect、prediction main effect及其 interaction 是否在 handoff/cache 相关 endpoint 上
出现，而非只看 shaped reward。现有 controller-level MAPPO 可以作为额外基线，但只有满足上述 matched contract 时
才可承担结构归因。

### 9.4 公平性、成本与停止规则

- 运行前基于 Stage 0 的独立 cluster 数与方差完成 power/sensitivity analysis；若可用独立 cluster 数不足，结果写
  `inconclusive`，不以更多重叠窗口或 row-level pseudo-replication补数。
- 预先固定主 endpoint、方向、effect-size interpretation、family 和缺失值规则。任何 margin 应由领域意义和 pilot
  noise 共同给出，本报告不编造阈值。
- 每个 arm 报告训练 wall-clock、CPU/GPU-hours、峰值 RSS、checkpoint size、模型参数量、推理 latency 与 transfer
  开销。推理 latency 用同机、交错随机顺序和 warm-up 后测量；不能跨当前 576/864 的宿主批次直接比较。
- 只有机制主效应在独立 clusters 上可复现、完成/continuity 不被 failure 抵消、transfer 与计算成本透明且适用边界
  明确，才可把候选贡献从 `UNVERIFIED` 升级。若只提高 reward 或少数窗口，仍不能写总体有效。
- 不以效果好坏选择 baseline、window、seed 或容量；负结果和 non-binding capacity 档全部保留。

## 10. 当前可写与禁止写法

当前可写：

- “我们实现了一个 controller-level、跨 RSU 连续 DAG workflow 的 typed cache/handoff 联合控制原型。”
- “formal 结果呈现稀疏 readiness/continuity 改善与更广泛 transfer/backhaul 成本的权衡；经窗口级 Holm 校正后没有
  显著比较。”
- “288 MB 为绑定区，576/864 MB 在当前 reachable working set 下均为非绑定区。”

当前禁止：

- “首次提出 DAG + adapter cache + migration + multi-timescale MARL”或任何未经穷尽检索证明的 first/unique claim。
- “显著优于 14 个 baselines”“降低端到端时延”“在 holdout 上泛化良好”。
- 把 12 contradicted 写成 12 个 Holm 显著劣势，或继续使用容量修正前 CI。
- 把 G12 supervised predictor、base-sharing 增益、state-migration 增益、eviction 增益写成已证明贡献。
- 把 controller-level MAPPO/SA-GHMAPPO 写成 vehicle/RSU-level full MARL。
- 宣称已达到未指定目录的 B 类／B 区期刊录用门槛。

## 11. Policy scorecard 与最终判断

| 审查维度 | 判断 | 关键原因 |
|---|---|---|
| Artifact completeness / provenance | 部分满足 | formal 与容量统计可追溯；独立 holdout 性能缺失 |
| Statistics | 部分满足 | 容量修正和窗口级复算已完成；独立 clusters 与 confirmatory test 不足 |
| Baseline fairness | 部分满足 | 外生输入和 interaction budget 可比；结构、信息与参数量未匹配 |
| Mechanism realization | 不满足论文 claim | 代码实现存在，但核心消融不可用或机会缺失；supervised predictor 未启用 |
| Cost / robustness / scalability | 不满足 | timing 混杂，训练成本不完整，容量只识别绑定/非绑定，缺系统规模曲线 |
| Novelty | `UNVERIFIED` | 交叉耦合可能有差异，但组件均有强近邻，且尚无机制价值证据 |
| Claim boundary | 可冻结 | 可写实现与 trade-off；总体优势、显著性、holdout 泛化和 full-MARL claim 禁止 |

最终结论：**当前没有一项已经由科学证据成立的核心创新。** 可以保留上述两项候选贡献进入最小补证，但在新独立
数据、机制机会、factorial ablation 和 matched controller 对照完成前，论文的创新性与算法有效性均应标为
`UNVERIFIED`。这不是否定工程实现，而是区分“真实实现”“能触发”“产生因果收益”和“可泛化”四个不同层级。
