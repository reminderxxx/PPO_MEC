# PPO_MEC Innovation Novelty Review

- `reviewed_at`: `2026-06-21`
- `literature_cutoff`: `2026-06-21`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `final_submission_v7_latency_fallback_20260618_rebuild_v1`
- `policy_version`: `tmc_review_policy_v2_20260618`
- `git_commit_at_review`: `d66b68a`
- `evidence_level`: `E3_REPRODUCED`（项目 artifact）；外部文献 venue/DOI 以一手出版页面核验
- `novelty_verdict`: `Conditionally defensible, but crowded`
- `overall_readiness_verdict`: `Not TMC-ready`

## Executive Verdict

截至 2026-06-21，本轮在 IEEE TMC、TITS、TVT、IoT Journal、ToN、TSC、TC、TNSM 及 ACM/系统论文入口检索后，没有发现一篇正式大刊同时覆盖以下完整交集：

1. 跨 RSU 连续 DAG workflow 执行状态；
2. base-model / adapter warm-state cache 与 predictive prefetch；
3. mobility-aware handoff prepare 和 workflow/state migration；
4. cache、execution/offload、handoff-event 三类异构时间尺度控制；
5. 真实 mobility + workflow 数据链路上的机制兑现指标。

因此，PPO_MEC 的**联合问题与系统 contract 仍有可辩护的新颖性**。但各个组成部分已经高度拥挤：`DAG + service caching + PPO`、`DAG + MADDPG`、`mobility + parallel-task cross-RSU offloading`、`trajectory prediction + migration + MARL`、`DAG + hierarchical offloading`、`adapter cache + two-timescale routing` 均已有近邻工作。不能再把 DAG、GNN、MARL、PPO、mobility prediction、hierarchy、cache 或 migration 中任何一个单独声称为首次创新。

本结论是定向一手页面检索与最近邻矩阵的结果，不是对所有数据库和未公开稿件的穷尽性不存在证明。

## 检索范围

- 主时间窗：2022–2026；基础工作可向前追溯。
- 核心刊物：IEEE TMC。
- 相邻大刊：IEEE TITS、TVT、IoT Journal、TNSM、TC、TSC、IEEE/ACM ToN。
- 系统与 adapter 语义参考：ACM/USENIX/MLSys/HPCA 正式页面；arXiv 只作 `C-Context`。
- 主题：VEC/MEC DAG/dependent task offloading、service/model/adapter caching、handoff/service/task migration、mobility prediction、multi-timescale control、MARL/PPO、edge AI/LoRA serving。
- 未向网页上传 manuscript、artifact、checkpoint、真实数据或审稿材料。

## 最近邻矩阵

符号：`●` 直接覆盖，`◐` 部分或相邻抽象，`—` 未覆盖/非核心。

| 最近邻工作 | Venue | DAG/依赖 | cache 对象 | mobility / RSU | migration / continuity | hierarchy / timescale | 对 PPO_MEC 的压力 |
|---|---|:---:|---|:---:|:---:|:---:|---|
| [Dual Dependency-Aware Collaborative Service Caching and Task Offloading in VEC](https://doi.org/10.1109/TMC.2025.3573379) | TMC 2025 | ● | service cache | ◐ | — | ● | 当前最强 `DAG + cache + PPO` 近邻；击穿“首次联合依赖与缓存” |
| [Optimization of Task Scheduling Strategy With Timing and Data Dependencies in VEC](https://doi.org/10.1109/TMC.2025.3646450) | TMC 2026 | ● | — | ◐ | — | ◐ | 已覆盖 DAG timing/data dependency + MADDPG；击穿“DAG + MARL” |
| [Mobility-Aware Collaborative Task Offloading for Parallel Tasks in VEC](https://doi.org/10.1109/TMC.2025.3631820) | TMC 2026 | ◐ | — | ● | ◐ | ◐ | 已覆盖 mobility、task relationship、cross-RSU collaboration 与 interruption risk |
| [Multi-Agent DRL With Trajectory Prediction for Task Migration-Assisted Offloading](https://doi.org/10.1109/TMC.2025.3539945) | TMC 2025 | — | — | ● | ● | ◐ | 已覆盖 trajectory prediction + migration + MARL；击穿预测迁移组合创新 |
| [Service Satisfaction-Aware Adaptive Service Migration and Resource Allocation in VEC](https://doi.org/10.1109/TMC.2025.3596342) | TMC 2026 | — | service instance | ● | ● | — | service continuity/migration 已是 TMC 明确问题线 |
| [Caching-Assisted Collaborative Task Offloading for VEC](https://doi.org/10.1109/TMC.2025.3650617) | TMC 2026 early access | ◐ | service/component cache | ● | — | ◐ | cache-assisted PPO/offloading 近邻；adapter 语义必须有真实系统证据 |
| [De-Duplicated Hierarchical Offloading in VEC With Task Dependencies](https://doi.org/10.1109/JIOT.2024.3510370) | IoT-J 2025 | ● | — | ● | — | ● | `DAG + hierarchical VEC offloading` 已存在；hierarchy 本身不新 |
| [Online Offloading and Mobility Awareness of DAG Tasks for VEC](https://ieeexplore.ieee.org/document/10700794) | TNSM 2025 | ● | — | ● | ◐ | — | `DAG + mobility-aware online offloading` 已存在 |
| [Multi-Agent Based Online Cooperative Computation Offloading and Migration Strategy for VEC](https://doi.org/10.1049/itr2.70083) | IET ITS 2025 | ● | container/service state | ● | ● | ◐ | 已覆盖 DAG、migration、multi-agent、连续执行的交集 |
| [POLAR: Online Learning for LoRA Adapter Caching and Routing](https://arxiv.org/abs/2604.16583) | arXiv 2026 | — | LoRA adapter | — | routing only | ● | adapter cache + two-timescale 已出现，但没有 VEC mobility/handoff |
| [CCO-DSS: Cache-assisted Offloading Based on DAG Subtask Sorting](https://doi.org/10.1109/NGDN66208.2025.11182155) | NGDN 2025 | ● | cooperative cache | ● | — | ◐ | 非大刊但直接证明 `DAG + cache + DRL` 不能用绝对首次表述 |

## 对当前四项创新点的判定

### 1. 跨 RSU 连续 DAG workflow 与 adapter warm-state 联合建模

判定：`较强，但必须写成完整交集创新`。

可守部分是：执行 frontier、adapter warm state、预测目标 RSU、handoff prepare 与 workflow state migration 在一个连续闭环中的联合 contract。不能只写“首次联合 DAG、cache、migration”，因为 TMC dual-dependency caching、TNSM mobility-aware DAG 和 IET DAG migration 已分别覆盖大部分组合。

建议贡献表述：

> We formulate mobility-driven continuous DAG execution across RSUs, where adapter warm-state placement/prefetch and handoff execution-state preparation are jointly controlled under one workflow-continuity contract.

### 2. DAG–RSU–mobility surrogate 的可靠性门控融合

判定：`中等`。

GNN/DAG encoding、model/data fusion、mobility-aware representation 均已有工作。PPO_MEC 可守的是 prediction confidence、uncertainty、handoff countdown 和 cache/workflow state 如何进入统一 reliability gate；但当前 predictor 不是独立 learned checkpoint，因此不能声称“提出新轨迹预测模型”或“surrogate learning framework”。

需要证据：可靠性门控的独立消融、prediction calibration 指标、低置信/错误预测下的 continuity/cache/migration trade-off。

### 3. slow/fast/event 三控制头的多时间尺度 PPO

判定：`中等，偏 contract novelty`。

多时间尺度 caching/offloading、hierarchical offloading、mixed-timescale VEC 已存在。可守部分是三个控制头分别对应 adapter placement/prefetch、实时 execution/offload 和 handoff-event prepare，并形成层级条件依赖。当前实现是 controller-level multi-controller + centralized critic，不是 vehicle/RSU-level full MARL。

需要证据：与 flat single-head、two-head、equal-timescale、independent-head 的公平消融，以及每个 head 的决策频率、计算开销和机制指标。

### 4. 预测切换机制训练与 policy guards

判定：`系统实现价值高，算法 novelty 偏弱`。

confidence/alignment admission、freshness、backhaul guard、latency fallback 更接近安全控制与工程约束。它们应被写成 safety/feasibility layer，并与 learned core 分开归因；否则 reviewer 容易认为性能来自手写规则或 reward shaping。

需要证据：learned core only、每个 guard 单独加入、全部 guards、oracle/no-prediction 四组；同时报告 reward、continuity、failure、validated hit、realized prepare、backhaul 和 guard trigger rate。

## Novelty Scorecard

| 子项 | 得分 | 理由 |
|---|---:|---|
| 问题交集与系统对象 | 5/6 | 完整交集暂未发现同构大刊工作；adapter warm-state + continuous handoff workflow 有区分 |
| 方法结构 | 3/5 | graph、PPO、MARL、hierarchy、multi-timescale 均有近邻；三控制器 contract 可区分 |
| 机制可验证性 | 3/5 | 已有消融与系统指标，但 prediction/prefetch 耦合、guard 归因和 predictor 边界仍弱 |
| 最近邻定位完整性 | 3/4 | 已建立矩阵并核验新 DOI；尚未得到 manuscript 逐句 contribution/related-work 映射 |

Novelty 总分：`14/20`。与 2026-06-18 readiness audit 相同：新增矩阵补齐了定位证据，但新增 TMC/IoT-J 近邻也同步收紧了可声明边界。

## Hard Blockers

- 如果论文把 `DAG + mobility + MARL`、`DAG + cache`、`trajectory prediction + migration` 或 `hierarchical offloading` 任一组合写成首次创新，将构成 novelty blocker。
- 如果 adapter cache 只有抽象容量、命中和 backhaul 数值，没有 adapter size/load/warm/migration latency 或 serving backend 对应，`adapter` 容易被 reviewer 视为 service cache 改名。
- 当前 strict full formal/holdout 对 DT baseline 的 95% CI 跨 0；即使 novelty 可守，整体仍为 `Not TMC-ready`。

## Major Concerns

- graph encoder 的 action contract 不选择 DAG node/frontier；不能声称 DAG-level parameterized scheduling。
- predictor 仍是 calibrated/baseline interface；不能声称 learned surrogate predictor。
- controller-level multi-controller 不能写成 vehicle/RSU-level full multi-agent framework。
- prediction 与 adapter-prefetch 消融高度耦合；不能把两者增益相加。
- policy guards 较多，必须分离 learned contribution、feasibility constraint 和 heuristic contribution。
- 仅 3 seeds；full split 统计弱，外部 LuST 场景还有 backhaul trade-off。

## Safe Claims

- 现有文献分别研究 DAG/dependent-task offloading、service caching、mobility-aware offloading、trajectory-guided migration 和 multi-timescale control；PPO_MEC 针对它们在 mobility-driven continuous AI workflow 中的联合 contract。
- PPO_MEC 的 cache 对象是 base-model/adapter 相关 warm state；它与传统 service/content cache 有抽象联系，但不能等同。
- 当前方法采用 controller-level hierarchical multi-controller policy，分别处理 cache、execution/offload 和 handoff-event。
- 截至 2026-06-21 的定向检索未发现与完整交集同构的正式大刊工作。

## Prohibited Claims

- “首次将 DAG 用于 VEC offloading”。
- “首次联合 DAG 与 cache/service caching”。
- “首次用 MARL 处理 mobility-aware DAG offloading/migration”。
- “首次提出多时间尺度或层级 VEC offloading”。
- “提出新的 learned mobility predictor”。
- “完整 multi-agent VEC framework”或“每个 vehicle/RSU 一个 agent”。
- “全面显著优于全部 baseline/所有场景”。

## Required Actions Before Re-review

1. 在 manuscript 中加入最近邻 claim matrix，至少逐项对比 dual-dependency caching、LDMCO、CoTOP、trajectory-prediction migration、TNSM mobility-aware DAG 和 IET DAG migration。
2. 将贡献标题从算法组件改为问题/contract：`continuous workflow state`、`adapter warm-state lifecycle`、`predictive handoff preparation`、`heterogeneous-timescale joint control`。
3. 增加 adapter 真实性证据：adapter 大小、装载/冷启动、RSU cache 容量、prefetch/migration latency 和 backhaul 字节；最好接入一个真实 LoRA serving profile。
4. 做 learned-core / guard 分离消融和 flat/two-head/equal-timescale 公平对照。
5. 修复 strict full formal/holdout CI blocker，并增加 seed、multiplicity 策略和计算开销报告。
6. 提供 manuscript 后，对 abstract、contribution、related work、图表逐句做 claim-to-source / claim-to-artifact 审查。

## TITS / TVT Fit Note

如果短期内无法修复 TMC 的 strict full statistical blocker，TITS/TVT 的场景适配可能更自然，但 novelty 表述不能因此放宽。无论目标 venue，`DAG + mobility + cache + migration` 都必须拆成可验证的新增系统对象与机制，而不能靠概念堆叠。

## 本轮新增的一手来源

- [IEEE TMC: Optimization of Task Scheduling Strategy With Timing and Data Dependencies in VEC](https://doi.org/10.1109/TMC.2025.3646450)
- [IEEE TMC: Mobility-Aware Collaborative Task Offloading for Parallel Tasks in VEC](https://doi.org/10.1109/TMC.2025.3631820)
- [IEEE TMC: Federated Meta-Learning Based Computation Offloading in UAV-Assisted VEC](https://doi.org/10.1109/TMC.2025.3573278)
- [IEEE IoT Journal: De-Duplicated Hierarchical Offloading in VEC With Task Dependencies](https://doi.org/10.1109/JIOT.2024.3510370)
- [IEEE NGDN: CCO-DSS Cache-Assisted Computation Offloading Based on DAG Sorting](https://doi.org/10.1109/NGDN66208.2025.11182155)
