# Agent × VEC × Model Cache × Digital Twin 顶刊调研

## 审查元数据与结论边界

- `reviewed_at`: `2026-08-11`
- `literature_cutoff`: `2026-08-11`
- `target_venue`: `IEEE TMC`；交叉参照 `TITS / ToN / INFOCOM / MobiCom / MLSys / OSDI / NeurIPS / ICML / ICLR`
- `artifact_run_id`: `N/A`（本轮为公开文献与仓库 contract 调研，不审计实验包）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_review_start`: `241632117027b055aea6a1fa2c969105d3ae0114`
- `evidence_level`: 外部文献为出版方/标准组织一手页面核验；项目现状为 `E1_DOCUMENTED + source-contract inspection`，不据此复核数值 claim

本报告回答的是“哪些交叉点有研究价值、如何转化为可验证课题”，不是 PPO_MEC 已达到 paper-ready 的审查。文件名沿用 2026-08-10 制定的计划，实际检索完成日为 2026-08-11。完整逐条数据源见 [机器可读证据表](agent_vec_cache_digital_twin_sources_20260810.csv)。

## 1. 执行摘要

最值得推进的不是把一个通用 LLM 接到现有环境，也不是重新包装 `MAPPO + cache`，而是构造**有权限边界的双层 Agent**：上层 research/operations agent 处理长时域目标、证据检索、workflow decomposition、计划缓存和反事实审查；下层 SA-GHMAPPO 只在稳定、可审计的五动作 contract 内执行实时 cache/offload/handoff 控制。两层之间通过 typed proposal、action mask、freshness/confidence、budget 和 verifier gate 连接。

公开证据已经表明：

1. Agent 的底层不等于 prompt。当前顶会工作已把 planning、tool graph、长期 memory、plan cache、verifier、安全和可执行 benchmark 分别做成独立研究对象。Agentic Plan Caching 在 NeurIPS 2025 报告平均成本下降 `50.31%`、延迟下降 `27.28%`，说明“计划”本身可以是 cache object；但它不涉及移动性、RSU 或物理 handoff。
2. `DAG + mobility + MARL`、`service caching + offloading`、`DT + VEC resource allocation`、`cooperative cache + CTDE` 均已有正式近邻，不能单独作为创新。IEEE IoT Journal 2025 已联合调度 vehicle twin maintenance 与 task processing；IEEE TMC 2026 已发表 cooperative edge caching + CTDE MADRL。
3. Adapter/KV/model cache 已是成熟的系统调度对象。Punica、S-LoRA、dLoRA、ServerlessLLM、CacheGen、Mooncake 分别证明 adapter batching/paging、adapter 动态迁移、checkpoint locality/live migration 和 KV-state streaming 的真实系统代价。但这些工作大多是数据中心或静态 edge serving，不提供 VEC handoff trace。
4. 当前最稀缺且仍可验证的交集是：**连续 DAG/agentic workflow 在跨 RSU 移动中，同时缓存计划与 adapter warm state，并由校准的 DT/belief state 决定何时复用、预取、迁移或回退**。截至检索日未找到正式论文同时覆盖全部条件；这是“检索范围内未找到直接近邻”，不是“世界首创”。
5. PPO_MEC 的现实短板不是缺一个 Agent 名称，而是缺少真实 adapter/model serving runtime、agent plan 请求 trace、可校准的 twin synchronization/error model，以及 vehicle/RSU-level multi-agent contract。若不补这些，论文只能声称 trace-driven controller prototype，不能声称真实 agentic VEC platform 或完整 digital twin。

推荐主路线为“**Twin-Verified Plan-and-Adapter Cache for Mobile Agentic Workflows**”：计划 cache 作为 workflow-level reusable state，adapter cache 作为 serving-level executable state，DT belief 估计 reuse validity 与 handoff risk，MARL executor 在 typed safety envelope 内实施动作。该路线同时有明确的系统指标、强 baseline 和可做的负向边界，最符合 TMC。

## 2. Agent 底层：从模型到可运行闭环

### 2.1 分层栈

| 层 | 底层对象 | 主要失败模式 | 应映射到 PPO_MEC 的对象 |
|---|---|---|---|
| Goal / policy | 用户目标、SLO、约束、权限 | 目标漂移、越权 | deadline、continuity、backhaul、privacy/safety budget |
| Planner | DAG decomposition、replanning、option selection | 长链漂移、循环、不可执行计划 | workflow DAG、跨 RSU stage placement、slow-timescale option |
| Memory | working/episodic/semantic/procedural memory | stale retrieval、污染、错误复用 | mobility episode、handoff outcome、plan template、adapter demand history |
| Tool/runtime | API schema、tool graph、sandbox、重试 | 参数错误、部分执行、不可逆副作用 | simulator、predictor、cache manager、benchmark/audit tools |
| World/twin model | state estimation、transition、counterfactual rollout | model bias、telemetry delay、分布漂移 | vehicle/RSU belief state、cache readiness、handoff ETA、branch replay |
| Verifier/critic | rule check、critic、uncertainty、human gate | 自我确认、错误置信度 | action mask、freshness gate、cost/risk critic、fallback |
| Executor | bounded action、transaction、rollback | deadline miss、级联失败 | `semantic_discrete_5` 实时 MARL executor |
| Observability | trace、provenance、cost、failure taxonomy | 只有最终 reward、无法归因 | per-step planner proposal、override、cache hit、migration、DT error |

ReCAP、TaskBench、GTA 与 Embodied Agent Interface 共同说明：长时域 Agent 应把 decomposition、tool selection、parameter prediction、action sequencing 和 transition modeling 分开评估，不能只报最终 success。A-Mem 与 G-Memory 说明 memory organization 也是学习对象；AgentPoison/MINJA 则说明把历史经验自动写入长期 memory 会引入供应链式攻击面。

### 2.2 Cache 语义必须严格分开

| Cache 类型 | 缓存内容 | 命中条件 | 失效原因 | 可测成本 |
|---|---|---|---|---|
| Plan cache | 结构化任务计划/子图模板 | 目标、工具能力和环境前提相容 | 工具/API/路况/资源变化 | planning token、LLM latency、adaptation failure |
| Agent memory | 事实、经历、程序性知识 | retrieval relevance + freshness + trust | stale/poisoned memory | retrieval latency、错误决策率 |
| Semantic/prompt cache | 相似输入的文本结果 | semantic equivalence | 外部状态变化 | token/TTFT |
| KV cache | Transformer attention state | prefix/model/adapter compatibility | prefix 或模型状态变化 | GPU memory、transfer、recompute |
| Model/checkpoint cache | 完整模型权重 | model version 匹配 | version/placement/capacity | load/cold-start time、带宽 |
| Adapter cache | LoRA/任务 adapter 权重与 warm runtime state | base model、adapter/version、device 匹配 | version/eviction/handoff | adapter load、GPU/RSU memory、migration bytes |
| Service cache | 应用/容器/函数 | service image 与依赖匹配 | image/version/resource mismatch | fetch/startup delay |
| DT state | 实体的同步数字表示或 belief | entity identity、时间戳、误差界有效 | telemetry delay/model drift | sync bandwidth、state age、prediction error |

论文中只能建立“共同属于可迁移状态”的抽象，不得把以上对象互称。尤其 `adapter_state_migration_overhead` 只有接入真实 adapter bytes、加载和 warm-up latency 后，才能从模拟 penalty 升级为系统结论。

### 2.3 科研 Agent 的可用边界

ResearchAgent、The AI Scientist、Co-Scientist、Robin 和 Coscientist 证明 Agent 可覆盖检索、假设、实验、分析与审查，但证据同时支持三个限制：

- 自评不能替代独立评审；The AI Scientist 的 workshop 首轮通过不等价于顶刊有效性。
- human-in-the-loop 能提高研究质量，科研 Agent 应输出 evidence ledger 和可复现命令，而不是自主改写结论。
- 对 PPO_MEC 最有价值的科研转化是“审计 Agent”：冻结 split、核对 provenance、检测窗口重叠、将 claim 映射到 artifact，并生成失败案例；不是让 LLM 自动调参后再评价自己。

## 3. 最近邻谱系与 novelty 压力

### 3.1 强近邻矩阵

| 研究线 | 代表一手来源 | 已覆盖 | 未覆盖 / 对 PPO_MEC 的空间 |
|---|---|---|---|
| Agent plan/memory cache | NeurIPS 2025 APC、A-Mem、G-Memory | 计划复用、动态图 memory、多 Agent memory | 没有 VEC mobility、RSU handoff、adapter readiness |
| 可执行工具 Agent | NeurIPS 2024 GTA、TaskBench、EAI | tool graph、真实工具、细粒度错误 | 不做通信/计算资源联合控制 |
| 科研 Agent | NAACL 2025 ResearchAgent；Nature 2026 AI Scientist/Co-Scientist/Robin | 检索、假设、代码、实验、审查 | 不保证网络系统实验 protocol 与 artifact integrity |
| VEC large-model offloading | IEEE TVT 2025 federated RL large-model offloading | 大模型任务、vehicle cooperation、MARL | 无 adapter/plan cache、连续 DAG 与跨 RSU state |
| DT + VEC | IEEE IoT-J 2025 twin maintenance/task processing | twin maintenance 与计算竞争、MADRL | 无 agent memory/plan、adapter cache、真实 serving |
| Cooperative edge cache | IEEE TMC 2026 EC-MADRL | cache/pricing/request scheduling、CTDE、testbed | 非 VEC、非 DAG、无 handoff/adapter state |
| DAG + service cache | IEEE TC/TPDS/TCCN 系列 | dependency-aware offloading、service caching | 通常无连续跨 RSU Agent state 与计划复用 |
| LoRA serving | MLSys 2024 Punica/S-LoRA；OSDI 2024 dLoRA | adapter paging、batching、迁移 | 数据中心 runtime，无车辆移动与 DT belief |
| Model/KV state migration | OSDI 2024 ServerlessLLM；SIGCOMM 2024 CacheGen；FAST 2025 Mooncake | checkpoint locality、live migration、state streaming | 无 VEC workflow semantics |
| DT belief control | INFOCOM 2026 latency-robust ISAC | stale telemetry、belief reconstruction、PPO | 非 VEC workflow/cache，但对“DT 必须校准”构成强方法近邻 |

### 3.2 不能再作为单点创新的表述

- “首次将 MAPPO 用于车联网卸载”：已有大量正式工作。
- “首次联合 cache 与 offloading”：INFOCOM 2018 及后续 TC/TPDS/TMC 已成熟。
- “首次考虑 DAG dependency”：JSAC/TC/TCCN/TPDS 已覆盖。
- “首次使用 digital twin 辅助 VEC”：已有 DT-enabled caching/offloading 和 twin maintenance 工作。
- “首次进行 cooperative edge caching”：TMC 2026 已有 CTDE + edge testbed。
- “把 adapter 当作可缓存对象”本身：MLSys/OSDI 系统和多篇 edge LoRA 工作已覆盖。
- “LLM Agent 具有 planning/memory/tool use”：这是通用架构，不是 VEC 方法贡献。

## 4. PPO_MEC 当前能力与缺口

### 4.1 已存在且可复用

- 正式数据主线为 NGSIM mobility + Alibaba DAG workflow，LuST 仅作低功效外部支持。
- 环境暴露五个语义离散动作：current RSU cache fill、next-RSU prefetch、vehicle fallback、steady offload、handoff migration prepare。
- 已有 cache/execution/handoff-event 三控制头、controller-level MAPPO/CTDE、action mask、handoff predictor、数字孪生式 exact branch replay 与多 horizon planner。
- 已记录 end-to-end delay、continuity、cold start、backhaul、migration overhead、handoff readiness 和 mechanism realization 等机制指标。

### 4.2 关键缺口

| 缺口 | 为什么影响 claim | 最小补齐方式 |
|---|---|---|
| 无真实 adapter serving backend | cache/migration cost 主要是模拟参数 | 接 vLLM/Punica/S-LoRA 类 runtime 或可复现实测 profile |
| Alibaba task 到 adapter 的映射是构造映射 | 不是真实 adapter demand trace | 发布显式映射表，并将 synthetic demand 与真实 runtime measurement 分层报告 |
| DT 不是持续同步的 vehicle twin | exact simulator/预测器不足以等同完整 DT | 定义 entity、sync channel、state age、calibration/error 与 update budget |
| 上层 LLM Agent contract 不存在 | 无法评价 plan cache/tool use | 定义 typed plan proposal 和 verifier，不让 LLM 直接输出环境动作 |
| controller-level 而非 vehicle/RSU-level MARL | 不能扩大 multi-agent scalability claim | 保持诚实命名，或另立 full multi-agent wrapper 课题 |
| v100 planner gain 依赖执行期 planner | native actor 尚未内化收益 | planner-enabled/native-policy 双报告；不要隐藏 query/latency cost |
| 只有有限 mobility/workflow 组合 | 外部有效性不足 | 新冻结 LuST/highD 或第二 workflow/service trace，至少 12 个独立 outer windows |

项目文档记录 v100 在冻结 formal/future 上有 reward 优势，但本轮没有审计相关 artifact；这些数值不能被本报告提升为 `E2` 或 paper-ready 证据。

## 5. 稀缺度矩阵

评分：`0–1` 大量直接近邻；`2` 成熟单领域；`3` 少量双领域交叉；`4` 仅预印本/间接近邻；`5` 本轮未找到满足全部核心条件的直接工作。计数是本轮纳入证据集的保守计数，不是全数据库总量。

| 候选交集 | 直接/间接近邻 | 稀缺度 | 置信度 | 判断 |
|---|---:|---:|---|---|
| MARL + VEC offloading | ≥5 / ≥10 | 1 | 高 | 拥挤，不宜作主创新 |
| DAG + service cache/offloading | ≥4 / ≥8 | 2 | 高 | 成熟；必须增加 mobility/state continuity |
| DT + VEC resource allocation | ≥2 / ≥6 | 3 | 中高 | 增长快，单独使用 DT 不新 |
| adapter/model cache + edge serving | ≥5 / ≥10 | 2 | 高 | 系统领域成熟，VEC 数据仍缺 |
| LLM Agent plan cache + mobility | 0 / 3 | 4 | 中 | 有 plan cache/embodied memory，但无 RSU 系统证据 |
| plan cache + adapter cache 联合有效性 | 0 / 5 | 4 | 中高 | cache 对象跨层耦合尚少见 |
| DT-belief + plan/adapter cache + VEC handoff | 0 / 6 | 5 | 中 | 本轮未见全部交集；需避免“首创”措辞 |
| continuous DAG + cross-RSU agent state migration + real trace | 0 / 7 | 5 | 中高 | 最接近现有主线且可做机制验证 |
| vehicle/RSU full MARL + LLM supervisor + safety verifier | 0 / 5 | 4 | 中 | 新但实现成本高、baseline contract 风险大 |
| audit Agent for network-systems artifact integrity | 0 / 4 | 4 | 中 | 科研工具方向，适合作为独立 artifact/benchmark 贡献 |

主要检索式族：`LLM agent plan caching memory tool benchmark`、`vehicular edge digital twin caching offloading MARL`、`adapter caching routing edge LoRA serving`、`DAG workflow handoff state migration RSU`、`scientific research agent experiment verifier`；数据库覆盖 IEEE Xplore、ACM/USENIX/MLSys、NeurIPS/PMLR/OpenReview、Nature、ACL Anthology、ETSI 与 ISO。时间范围以 2021–2026 为主。

## 6. 五条创新路线与优先级

### R1：Twin-Verified Plan-and-Adapter Cache（推荐主线）

- 外部机制：APC 的 plan template reuse、DT belief-state control、LoRA adapter paging/migration。
- VEC 转化：对每个 continuous workflow 同时维护 `plan_template_id`、`adapter_id/version`、RSU placement、handoff ETA、DT state age/error；DT verifier 判断计划是否仍适用及是否值得预取 adapter。
- 接口：上层只提交 `PlanProposal{workflow_subgraph, candidate_rsu, required_adapter, valid_until, confidence, expected_cost, evidence_ids}`；下层通过 action mask 映射到现有五动作。
- 目标：最小化 workflow delay、planning cost、cold start、backhaul 与 invalid-plan execution，同时维持 continuity。
- baseline：no-plan-cache、semantic-cache-only、APC-like plan cache、popularity adapter cache、DT-only、MARL-only、oracle-validity upper bound。
- 消融：no-DT freshness、no-plan adaptation、no-adapter coupling、no-verifier、single-timescale。
- 可守 claim：联合 validity-aware plan/adapter caching 在移动 handoff 下改善成本—连续性 Pareto；前提是新增真实 runtime 与独立 holdout。
- 评分：novelty 5/5、复用 5/5、数据 3/5、可验证 4/5、成本 3/5、TMC fit 5/5；总优先级第一。

### R2：Mobility-Aware Agent Memory with Handoff Continuity

- 外部机制：A-Mem/G-Memory 的动态 memory graph与 ExRAP 的时变环境 memory。
- 转化：memory node 绑定 vehicle/RSU/time interval，retrieval 受空间距离、handoff path、freshness 和 provenance 约束；迁移的是最小必要 memory/adapter state，而非整段 history。
- 指标：memory bytes、stale-hit rate、poisoned/invalid retrieval、continuity、migration latency。
- 风险：若只有向量检索模拟而无真实数据量与延迟，容易退化为通用 memory 方法。
- 优先级第二，适合 TMC + AI systems 交叉。

### R3：Uncertainty-Calibrated DT Verifier for Safe MARL

- 外部机制：belief state、telemetry-delay robustness、constrained MAPPO、verifier gate。
- 转化：DT 输出 state belief 和 calibration interval；仅在 expected gain 超过 uncertainty/cost margin 时允许 prefetch/prepare，否则 steady/fallback。
- baseline：raw predictor、uncalibrated DT、EKF/belief DT、risk-neutral MAPPO、oracle state。
- 指标：ECE/Brier/NLL、state age、false prepare、missed handoff、CVaR delay、backhaul。
- 可守 claim：不是“完整 DT”，而是 latency-aware calibrated twin-assisted control。
- 优先级第三；与现有 v100 planner最接近，但 novelty 需要真实 telemetry delay protocol。

### R4：Agentic DAG Across RSUs

- 外部机制：TaskBench tool graph、ReCAP hierarchical replanning、AWTO agentic workflow scheduling。
- 转化：DAG node 不只代表 compute task，还包含 tool call、model/adapter dependency、side-effect 和 rollback policy；handoff 时迁移 execution checkpoint 与 tool provenance。
- 指标：tool success、parameter error、rollback rate、workflow makespan、exactly-once/at-least-once violation。
- 风险：工程量大，必须构建公开 agentic VEC workload，不能继续把 Alibaba batch DAG 直接称为 LLM agent workflow。
- 优先级第四，但数据集贡献潜力高。

### R5：Reproducibility/Audit Agent for VEC Experiments

- 外部机制：ResearchAgent reviewer agents、AI Scientist pipeline、AgentAuditor/AgentPoison 风险模型。
- 转化：Agent 读取 manifest/command log/interval，不修改算法，自动核验 split overlap、checkpoint provenance、claim-to-field 和负结果披露。
- 指标：真实 blocker precision/recall、false assurance、审计时间、可复现成功率。
- 风险：更像 research infrastructure，需单独 benchmark；不应混入主算法贡献凑创新点。
- 优先级第五，可作为开源 artifact 或独立论文。

## 7. 推荐路线的 decision-complete 实验蓝图

### 7.1 系统与数据

- Mobility：NGSIM 为主；新冻结 LuST 或 highD 为跨域 mobility，formal/holdout 各至少 12 个互斥 outer windows。
- Workflow：Alibaba DAG 保留为 compute dependency；另构建公开、版本化的 agentic tool-DAG workload，记录 tool schema、side effect、required model/adapter 与 plan template family。
- Runtime：选择一个可复现多 LoRA serving backend，实测 adapter size、load/warm-up、swap、base-model compatibility；把 profile 固化为 manifest，不自动下载或覆盖数据。
- DT：每个 vehicle/RSU state 带 event time、arrival time、state age、noise、missingness；训练/dev/formal/hidden 使用预冻结 telemetry-delay/noise profile。

### 7.2 Contract

- 上层周期显著慢于环境 step，只能产生 typed proposal，不能绕过 action mask。
- proposal 必须含 provenance、confidence、valid-until、estimated token/latency/backhaul cost。
- verifier 检查 tool/schema、adapter version、target RSU、deadline、budget、state freshness；失败时回退下层 native policy。
- 下层动作保持现有 `semantic_discrete_5`，从而允许与 v100、PPO、MAPPO、DT/cache/DAG 专项 baseline 做公平比较。

### 7.3 评测

- 主指标：deadline-constrained workflow completion 或 normalized utility；total reward 只作内部聚合，不替代机制指标。
- 系统指标：E2E delay、TTFT/adapter warm latency、plan hit/adapt success、validated adapter hit、cold start、continuity、handoff failure、backhaul/migration bytes、LLM token/cost、planner wall-clock。
- DT 指标：handoff/next-RSU accuracy、ECE、Brier、NLL、state age、rollout error by horizon。
- 安全指标：invalid proposal、verifier rejection/false rejection、unsafe action prevented、memory poisoning/staleness success rate。
- 统计：window 为 outer independent unit；seed/workflow 在窗口内重采样；报告 percentile 与 BCa 95% CI、window effect、paired sign test、Holm 校正。mixed/full 不合并伪增样本。

### 7.4 Baseline 与消融

- 规则：reactive greedy、popularity cache、LRU/LFU、oracle mobility/cache upper bound。
- 学习：PPO、strong MAPPO、DQN/dueling DQN、QMIX/controller-MAT、DAG/cache/DT 专项 learned baselines。
- Agent：LLM-no-cache、semantic response cache、APC-like plan cache、memory-only、planner without verifier。
- 系统：adapter cache only、plan cache only、joint cache、no-prefetch、no-migration。
- DT：no twin、raw predictor、calibrated belief twin、oracle state。
- 必须报告 planner-enabled 与 native-policy-only，防止把执行期搜索收益写成 actor 学习收益。

### 7.5 Artifact gate

- 冻结 candidate hash、prompt/tool schema、model/API version、runtime profile、split manifest、seed/budget 后才开启 hidden。
- 保存所有 proposal、tool call、verifier decision、environment action、cache event、DT prediction 和 wall-clock trace。
- hidden 一次开启即 consumed；不得根据结果修改 prompt、cache threshold、checkpoint 或 window。
- TMC-ready 仍按项目 rubric：无硬 blocker、至少 E2、总分 ≥85 且各项 ≥60%。

## 8. 标准、行业与公开资源

- ETSI GS MEC 030 定义 V2X Information Services API，说明多厂商、多网络、多接入互操作是现实接口边界；Agent 应调用标准化 capability API，而不是假设全局直接状态。
- ETSI MEC-DEC 050 讨论 MEC 与 oneM2M/IoT/digital-twin 协同，可作为 DT/MEC 接口背景，不是算法 novelty 证据。
- ISO 23247-2 给出 DT reference architecture；其核心启示是物理实体、数字表示和接口/同步关系必须显式存在。只有预测特征或 simulator branch 不自动构成完整 DT。
- NVIDIA Omniverse/DRIVE 是行业可行性证据：物理准确模拟、真实数据同步、cloud-to-car 验证链已被产品化；厂商页面不能替代同行评审性能证据。
- NGSIM、Alibaba、LuST 支撑 mobility/workflow，但没有真实 RSU adapter request、cache hit/miss 或 state-migration trace。这一数据缺口本身是稀缺机会，也是一项有效性威胁。

## 9. Safe claims、禁止表述与投稿建议

### Safe claims

- “截至 2026-08-11 的一手来源检索，Agent plan/memory cache、VEC DT resource control 和 adapter serving 各自已有成熟工作，但本轮未找到同时研究连续 DAG、跨 RSU handoff、plan/adapter 双 cache 与校准 DT verifier 的正式直接近邻。”
- “PPO_MEC 提供了研究该交集的 trace-driven controller substrate；真实 adapter runtime、agentic workload 和 DT synchronization 尚需补齐。”
- “双层设计的贡献候选是可验证的跨层 contract 与有效性门控，而不是 LLM/MAPPO 本身。”

### 禁止表述

- “世界首个 Agent-VEC 数字孪生系统”或“该方向完全空白”。
- 把 exact simulator、handoff predictor 或 branch replay直接称为完整 live digital twin。
- 把 controller-level 三头 MAPPO 称为 vehicle/RSU-level full MARL。
- 把 Alibaba DAG 称为真实 LLM agent workflow，或把构造 adapter label 称为真实 adapter demand trace。
- 把 KV cache、plan cache、adapter cache 和 service cache 互换使用。
- 在未计入 LLM query、runtime、transfer 和 verifier 成本时声称端到端 latency 优势。

### 投稿定位

- TMC：最佳匹配 R1/R2/R3，必须强调 mobility、跨 RSU continuity、系统成本与严格 trace protocol。
- TITS：若强化 vehicle safety、traffic context、CAV workflow 和 handoff reliability。
- ToN/INFOCOM：若强化网络建模、backhaul、状态传输与在线控制理论。
- MobiCom/NSDI/OSDI/MLSys：必须有真实 serving/runtime/testbed，而非纯 simulator。
- NeurIPS/ICML/ICLR：必须把方法抽象成跨领域 Agent memory/world-model/verification 问题，并提供标准 benchmark；仅 VEC case study 不够。

## 10. 最终建议

先做 R1 的最小闭环，而不是立刻引入自由文本 LLM：先冻结 typed proposal 与 plan-validity contract，接入可复现 adapter runtime profile，构造 plan-cache workload，并在已有五动作 executor 上完成 joint-cache/no-cache/DT-verifier 对照。只有当 plan reuse、adapter readiness、handoff continuity 和总成本都能从原始 trace 回溯时，再扩展到真正的上层 LLM planner。这样既最大化复用 PPO_MEC，又避免因 Agent 包装扩大 claim 而触发 TMC novelty 与技术正确性 blocker。
