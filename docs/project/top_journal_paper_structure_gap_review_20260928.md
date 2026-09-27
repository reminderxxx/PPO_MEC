# Top-Journal Paper Structure and Gap Review

- `reviewed_at`: `2026-09-28`
- `literature_cutoff`: `2026-09-28`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `N/A — literature/structure review; G14C v16 training/dev artifacts exist, but the formal phase has not started`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `ea96d286daa3af6b6e81f818b9feeaa4bdab2f80`（审查开始时）
- `evidence_level`: 外部论文三篇为公开全文阅读；其余为官方出版页/摘要核验。PPO_MEC 当前 paper-readiness 只到 `E1_DOCUMENTED`，因为最新 typed-cache 主线尚无正式 checkpoint、formal/holdout/support 原始结果。
- `verdict`: `Unverifiable`（针对当前 PPO_MEC 是否达到 TMC-ready；不是对研究方向价值的否定）

## 1. 审查范围与证据边界

本轮重点拆解三篇可公开读取全文的高相关论文，并用官方出版页补充最新近邻：

1. [Resource Allocation for Twin Maintenance and Task Processing in Vehicular Edge Computing Network](https://doi.org/10.1109/JIOT.2025.3576582)，IEEE Internet of Things Journal, 2025。
2. [DNN Partitioning, Task Offloading, and Resource Allocation in Dynamic Vehicular Networks](https://doi.org/10.1109/TMC.2024.3486728)，IEEE TMC, 2025。
3. [Digital Twin-Enabled Mobility-Aware Cooperative Caching in Vehicular Edge Computing](https://arxiv.org/abs/2603.06653)，arXiv 2026；预印本声称 accepted by TMC，正式 DOI/卷期仍待一手页面核验。
4. [AWTO: A Latency-Optimized Task Offloading Scheme for LLM-Driven Agentic Workflows on Heterogeneous Edge](https://doi.org/10.1016/j.future.2026.108415)，FGCS, 2026。仅按出版社可见全文结构与段落拆解，不将其当作 TMC 等级证据。

外部论文的数值只用于理解作者如何组织证据，不用于 PPO_MEC 的性能结论。PPO_MEC 的“还缺什么”依据当前仓库文档、方法报告和 formal 主线状态判断；未逐项重放全部 artifact，因此 paper-readiness 结论只能是 `Unverifiable`。

## 2. 顶刊常见论证骨架

三篇全文虽然技术不同，但都采用同一个可复用的故事链：

1. **应用压力**：先说明场景为什么必须做，不先讲算法。
2. **已有范式**：说明 VEC、DT、FL、DRL 或 split inference 已经解决了什么。
3. **可观测缺口**：用一个明确的资源冲突、动态性、预测误差或稳定性问题收紧研究空白。
4. **问题定义**：把缺口变成系统模型、动作、约束和目标，而不是一句“现有方法效果不好”。
5. **方法映射**：每个模块必须对应一个前文缺口；模块不能只因“先进”而出现。
6. **证据闭环**：实验按“整体收益—机制指标—规模/参数变化—消融/限制”逐层展开。
7. **边界声明**：高质量论文会说明参数敏感性、部署、安全、数据或多节点扩展仍未解决。

PPO_MEC 最适合模仿的不是具体算法，而是“一个缺口对应一个模块、一个模块对应一组可观测指标”的写法。

## 3. 论文一：Twin Maintenance 与 Task Processing 的资源竞争

### 3.1 Introduction 逐段功能

1. **第 1 段：需求入口。** 从 5G 车载应用和计算任务增长切入，说明车辆本地算力/存储不足，VEC 让车辆可将任务卸载到路侧服务器。作用是建立读者共识，不急于引入 DT 或 MADRL。
2. **第 2 段：引入技术机会。** 指出 VEC 的 mobility 和环境动态问题，再介绍 DT 的双向映射、预测与虚实交互能力。作用是把 DT 写成解决动态性的必要基础设施。
3. **第 3 段：从优势转向冲突。** 先说明 DT 可实时监控、预测、在虚拟环境中验证决策，然后立即指出被忽略的问题：DT maintenance 本身也占用 VEC server 计算资源。
4. **第 4 段：把缺口压成一句问题。** 同一服务器需要同时维护 vehicle twin 和处理 vehicular tasks，因此有限资源如何分配是核心问题。
5. **第 5 段：方法总览。** 给出单 VEC server、多车辆场景，说明目标是兼顾两类时限并最大化资源效用，提出 MADRL-based collaborative scheduling。
6. **贡献段。** 三个贡献分别对应：场景与两类 delay、satisfaction-based optimization、MADRL 求解与比较验证。没有把“用了 MADRL”单独当第一贡献。
7. **组织段。** 用一段将 Related Work、System Model、Method、Simulation、Conclusion 串起来。

### 3.2 全文结构

- **Related Work**：先 general DT，再 DT+VEC。它不是文献罗列，而是逐步证明“现有 DT/VEC 工作没有同时建模 twin maintenance 与 task processing 的资源竞争”。
- **System Model**：先给 Fig. 1 物理架构，再依次定义 digital twin、vehicle movement/channel、communication/computing、两类 delay、satisfaction function 和 constrained optimization。
- **Proposed Solution**：将原问题重构成 multi-agent MDP，明确 state/action/reward，随后给 actor-critic、centralized training/distributed execution、训练/执行伪代码和复杂度。
- **Experiments**：参数表 → baseline 定义 → convergence/reward → resource utilization → per-vehicle utility → 两类 delay 随车辆数变化。
- **Conclusion**：重述问题、方法和机制结果，再明确承认 server number、multi-edge resource competition 与 parameter sensitivity 仍待研究。

### 3.3 PPO_MEC 可模仿点与不能照搬处

可模仿：Introduction 的“先说 DT 带来价值，再指出 DT 也消耗资源”的反转非常有效。PPO_MEC 可以采用同样句式：adapter warm state 和 handoff prepare 能提升 continuity，但它们同时消耗 cache、backhaul 与迁移窗口。

不能照搬：该文是单 VEC server、两类资源竞争，没有 continuous DAG、跨 RSU handoff、typed cache dependency 或 state migration。它也主要是仿真和 point estimate；PPO_MEC 若以 TMC 为目标，应在统计独立性、holdout 和 artifact provenance 上更强。

## 4. 论文二：DNN Partitioning + Offloading + Resource Allocation

### 4.1 Introduction 逐段功能

1. **第 1 段：AI vehicular application。** 用 autonomous driving、SLAM、AR navigation 和 VGG16 inference 建立 DNN task 的应用背景。
2. **第 2 段：定量化算力压力。** 用 ResNet152 操作量、车载摄像头/雷达持续数据说明单车算力不足。它把“算力紧张”从口号变成定量动机。
3. **第 3 段：为什么是 VEC。** 先否定远云的 backhaul latency，再引入 RSU/周边车辆组成的 VEC，并给出 local + edge 两段式 DNN execution。
4. **Motivation 第 1 段：结构异质性。** 用 VGG16 layer-wise latency/output-size pilot study 证明不同 partition point 成本不同。Fig. 1 在提出算法前就出现，承担“问题存在性证据”。
5. **Motivation 第 2 段：动态决策。** mobility 导致 channel uncertainty，实时 partition/offloading 必要；传统 heuristic/decomposition 迭代慢，不适合动态网络。
6. **Motivation 第 3 段：长期稳定性。** 指出现有工作只优化 delay/energy，忽略 task-queue stability；由此把单时隙优化升级成 dynamic long-term optimization。
7. **方法总览段。** 给出 joint DNN partitioning/offloading/resource allocation 与 completion-time + stability 目标。
8. **贡献段。** 依次写 MINLP/NP-hard、Lyapunov decoupling、diffusion-based multi-agent decision + convex subroutine、真实地图/SUMO + 多 DNN 验证。
9. **组织段。** 明确每一节在完整求解链中的位置。

### 4.2 全文结构

- **Related Work** 分四条线：DNN edge inference、DRL resource management、Lyapunov stability、diffusion optimization。每条线都对应方法中的一个模块。
- **System Model** 依次定义车辆/RSU拓扑、DNN layer partition、V2I/V2V communication、local/RSU/service-vehicle computing 和 task queues。
- **Problem Formulation** 先给长期 MINLP 和稳定性约束，不把 reward 当问题定义。
- **Lyapunov Section** 将长期约束问题转成 per-slot deterministic problem，是理论桥梁。
- **Diffusion Section** 先单独解释为什么采用 diffusion，再给 forward/reverse process，避免读者在算法段第一次遇到陌生模型。
- **MAD2RL Section** 按 overview → QMIX rationale → convex resource subroutine → MDP → network architecture → replay/policy update → complexity 展开。
- **Evaluation** 使用 OpenStreetMap+SUMO、AlexNet/ResNet18/VGG16；baseline 为 P-QMIX、genetic、greedy；依次评价 denoising step、reward convergence、queue stability、client/service vehicle 数量和 Lyapunov 参数。

### 4.3 PPO_MEC 可模仿点与不能照搬处

可模仿：在 Introduction 中加入一个“问题存在性图”。PPO_MEC 最需要的不是再画总架构，而是用真实 trace 展示：handoff 前剩余时间、target adapter readiness、workflow frontier 和 migration bytes 的冲突，以及 reactive policy 为什么错过 prepare window。

不能照搬：该文处理 DNN layer partition，不是 DAG node workflow；其 multi-agent contract 是车辆级 QMIX，而 PPO_MEC 当前是 controller-level heads。论文必须主动说明这一差别，不能用“multi-agent”名称让读者误以为每车/每 RSU 都是 agent。

## 5. 论文三：DT + Mobility-Aware Cooperative Caching

### 5.1 Introduction 逐段功能

1. **第 1 段：vehicular caching 背景。** 说明 V2X/5G/6G 下 content distribution 与 cache management 的重要性，并列出 mobility、intermittent connectivity、heterogeneous resources。
2. **第 2 段：为何 centralized 不够。** 引出 FL 的隐私与分布式训练优势，说明 FL 如何为 caching decision 提供 collective knowledge。
3. **第 3 段：两个明确缺口。** 不是泛泛说预测不准，而是拆成 client selection 无视 mobility/data quality，以及 LSTM/simple time model 难以处理复杂时空相关和不确定性。
4. **第 4 段：模块逐项对位。** 三层 physical/digital-twin/intelligent-decision 架构；AFL 解决 client selection；GRU-VAE 解决 prediction；SAC 解决 caching allocation。每个模块都能回指上一段的一项缺口。
5. **贡献段。** framework、mobility/data-quality client selection、GRU-VAE + DRL caching 三项贡献，与第 4 段一一对应。
6. **组织段。** Related Work、System Model、DAPR、Evaluation、Conclusion。

### 5.2 全文结构

- **Related Work** 分 traditional caching/FL、DRL caching、DT vehicular caching 三条线，最后组合出缺口。
- **System Model** 通过三层框架图定义 DT mapping、vehicle movement、communication model 和 content access/caching path。
- **Method** 按 GRU-VAE popularity prediction → asynchronous FL/client selection → SAC cache allocation 展开；模块顺序与数据流一致。
- **Evaluation** 使用 T-Drive 北京出租车轨迹、MovieLens 1M、Top-5k albums、YouTube；先定义 cumulative reward、cache hit、transmission delay、prediction MSE，再给五个 baseline。结果按 dataset consistency、cache capacity、区域密度、vehicle density、prediction loss 和四项 ablation 展开。
- **Discussion** 明确承认低密度 sparse data 下收敛慢，以及 malicious node/privacy threat 缺少防护；这是值得 PPO_MEC 模仿的独立限制段。

### 5.3 PPO_MEC 可模仿点与不能照搬处

可模仿：实验先定义指标，再逐个回答模块是否兑现；四项 ablation 直接对应四个方法模块。PPO_MEC 也应按 graph、typed cache dependency、predictor、handoff-event/migration、policy guard 分组消融。

不能照搬：该文缓存对象是 content popularity，不是 adapter/base/workflow state；MovieLens/YouTube 也不是 VEC adapter request trace。PPO_MEC 必须明确 Alibaba workflow 到 adapter demand 的 mapping 是受控构造，而不是真实 LoRA 请求日志。

## 6. 论文四：AWTO Agentic Workflow Edge Offloading

该文是当前最贴近“LLM agentic DAG + heterogeneous edge + model loading/cache”的工作之一。

Introduction 可见段落的功能是：

1. 从 LLM agentic workflows 的多阶段角色和执行不确定性切入，并明确任务依赖形成 DAG。
2. 说明 private edge 的数据主权价值，同时指出 edge memory、model loading 与 autoregressive latency/footprint 的动态性。
3. 指出现有 DRL task model 过于简化，sequence-to-sequence 方法又依赖完整 workflow 先验。
4. 提出 task-by-task cache-aware dynamic scheduling，把 edge cluster 看作 shared model cache pool。
5. 用三模块 LSTM 分别编码 task dependency、device heterogeneity 和 memory state。
6. 贡献段依次写 MDP、动态 PPO scheduler、三模块 encoder、跨 workflow/edge configuration 的结果与 redundant model loading 机制解释。

对 PPO_MEC 的最大压力是：`DAG + LLM workflow + model loading/cache + PPO` 已不再新。PPO_MEC 的必要差异必须是 mobility-driven cross-RSU continuation、adapter/base/workflow-state typed cache、handoff prepare/migration 与真实车辆轨迹，而不能只强调 graph encoder 或 cache-aware PPO。

## 7. 建议的 PPO_MEC Introduction 逐段模板

1. **场景段**：AI-driven VEC 从单次感知任务转向 vehicle-bound multi-stage workflow；车辆移动导致同一 workflow 跨多个 RSU 持续执行。
2. **工作流压力段**：DAG dependency、frontier、intermediate state 和 adapter requirement 使单任务 offloading 模型不足。
3. **typed cache 压力段**：base model、adapter 与 workflow state 的依赖/大小/迁移语义不同，content/service cache 抽象不能直接替代。
4. **handoff 冲突段**：prefetch/prepare 太晚会 stall，太早会造成 cache pollution/backhaul；prediction 又可能不准。
5. **最近邻边界段**：现有工作已分别研究 DAG offloading、service caching、trajectory migration、DT prediction 和 mixed-timescale control，但尚未在同一可审计闭环中处理上述 typed state 与 continuous workflow。措辞必须写成“截至检索日未发现同时覆盖”，不能写绝对首次。
6. **问题定义段**：明确 observation、semantic action、typed cache capacity、handoff event、deadline/continuity/backhaul objectives 和 constraints。
7. **方法总览段**：graph-surrogate encoder + slow/fast/event controller-level PPO + feasibility/guard layer；每个组件分别对应第 2–4 段。
8. **贡献段**：贡献应是 problem contract、method mechanism、formal protocol/evidence，而不是“用了 PPO/MAPPO/GNN”。
9. **结果预告段**：只有在 G14C formal、独立 holdout、support 和统计完成后，才能填具体数值；当前应留空或标记待补。

## 8. 当前 PPO_MEC 还缺什么

### 8.1 Paper-readiness blocker

1. **最新 typed-model-cache 主线没有正式结果。** 当前文档记录 G14C v16 已有 150 train cells、1,200 candidates、150 selected/frozen checkpoints，但正式 formal/holdout/support 尚未执行，启动授权也未签发。缺少完整原始结果时，结论只能是 `Unverifiable`。
2. **没有 integrated manuscript。** 仓库有 method report 和 paper artifact exporter，但未发现完整 LaTeX/manuscript。问题、方法、实验和 limitation 尚未形成一条投稿级叙事。
3. **最近邻已压缩单点 novelty。** DAG、DT、cache、migration、mixed-timescale、MARL、agentic workflow + model loading 均已有强近邻；必须用交集问题和机制 endpoint 守 novelty。

### 8.2 Method/claim gap

1. `SA-GHMAPPO` 是 controller-level multi-controller PPO，不是 vehicle/RSU-agent full MARL；标题、摘要和 Related Work 必须主动声明。
2. 当前 action contract 是 `semantic_discrete_5`，不是 DAG-node parameterized action 或连续 resource allocation；不能与连续控制论文做不加说明的同类比较。
3. learned handoff predictor 尚未以冻结 checkpoint、quality report 和正式 benchmark 进入主结果；surrogate/baseline predictor 与 policy guard 必须分开归因。
4. policy guards 属于 feasibility/safety layer；需要 `learned core only`、`+guard`、`+predictor` 的分层消融，不能把 guard 收益写成端到端学习收益。

### 8.3 Evaluation gap

1. 正式主表需要至少覆盖最强 learned、domain-specific 和 heuristic/oracle 边界；`popularity_cache_heuristic` 接近主方法，必须如实报告。
2. formal/holdout 要按原始 frame/time interval 互斥；`window_rank_offset` 不得再称独立 holdout。
3. 统计单元应以非重叠时间窗口为外层，seed/workflow 为内层；报告 BCa/percentile 95% CI、paired sign test 和 Holm correction。
4. 机制指标至少包括：typed request/byte hit、base-adapter dependency readiness、realized prefetch/prepare、migration bytes/time、handoff-ready ratio、continuity、handoff failure、deadline violation、backhaul、cache pollution/churn。
5. 需要 graph、typed cache、prediction、three-timescale/event、migration、guard 的独立消融，以及 prediction error、bandwidth、compute、cache capacity、vehicle density、workflow size/shape 的 robustness/scalability。
6. `NGSIM + Alibaba` 是真实数据链，但 adapter request/size mapping 仍是受控构造；需要在 Dataset/Limitations 中单列，不得称真实 adapter trace。

### 8.4 Presentation gap

1. 缺一张“问题存在性图”：展示 handoff countdown、DAG frontier、target cache readiness 和迁移成本如何冲突。
2. 缺一张最近邻矩阵：列出 DAG continuity、typed cache、handoff migration、multi-timescale、real trace、formal statistics 六列，避免 Related Work 只做段落罗列。
3. 缺一张端到端机制图：外部 request → DAG frontier → typed dependency bundle → slow/fast/event action → CacheEvent/handoff endpoint。
4. 缺单独 Limitations/Negative Results：包括 oracle prediction 边界、heuristic gap、controller-level MARL、无真实 adapter request trace 和未完成系统部署。

## 9. 投稿前最小闭环

1. 冻结并执行新的 clean formal run，生成 checkpoint、manifest、command log、raw formal/support 和独立 holdout。
2. 用同一 frozen comparison package 产出主表、机制表、robustness、scalability、ablation 和 statistics。
3. 按本文第 7 节写 Introduction，并把每项 contribution 映射到一项实验表/图。
4. 建立 claim-to-artifact 表；任何摘要数值都能回溯到 JSON/CSV、窗口、seed、checkpoint 和 protocol。
5. 完成独立 TMC review；在证据达到 `E2_ARTIFACT_AUDITED` 前不使用 `paper-ready`、`state-of-the-art` 或“全面优于”措辞。

## 10. 安全表述与禁止表述

安全表述：PPO_MEC targets the intersection of continuous cross-RSU DAG execution, typed AI-state caching, and handoff-aware state preparation under a trace-driven VEC workflow setting.

禁止表述：

- “首次将 DAG、cache、handoff、DT 或 MARL 用于 VEC”。这些单点均已有近邻。
- “full multi-agent cooperation”。当前是 controller-level heads。
- “真实 adapter workload”。当前主线没有公开真实 LoRA/adapter request trace。
- “已达到 TMC-ready”或“全面优于所有 baseline”。最新 formal/holdout/support 尚未闭环，且 heuristic/oracle 边界仍需报告。
