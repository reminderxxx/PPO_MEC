# Top Journal Gap Cause Analysis And Improvement Plan

更新日期：2026-05-04

用途：分析 `PPO_MEC` 当前实验距离顶刊发表要求的核心差距，并给出后续实现路线。本报告只做原因分析和方案设计，不修改环境语义、reward、benchmark protocol、checkpoint 或历史 artifact。

## 结论摘要

当前项目已经具备论文雏形：`NGSIM + Alibaba` 真实数据链路跑通，`sa_ghmappo` 相比 PPO/MAPPO 和 reactive greedy 有较稳定优势，baseline artifact、manifest、训练评估和 benchmark 管理链路已经成型。

但距离顶刊发表仍差一个可冻结的正式实验包。主要原因不是缺单个脚本，而是当前证据链还不能支撑“主方法在 AI-driven VEC 的跨 RSU DAG workflow、adapter cache 协同和 handoff migration 机制上全面优于强对照”的强 claim。

最关键短板：

- 对最强 `popularity_cache_heuristic` 的优势仍薄，且部分可靠性指标落后。
- 机制链条没有稳定闭环，尤其 `handoff_ready_ratio`、`mechanism_realization_rate` 和 seed `29` 的 mechanism window 表现不稳。
- HF model-cache 只达到 file-size/cache-volume profile 级别，不能当作真实 VEC cache request trace。
- 当前 PPO/MAPPO 对照区分度不足，`mappo` 也不是完整 CTDE multi-agent contract。
- 三 seed 和有限窗口不足以证明对强 heuristic 的小幅优势具备统计显著性。
- 当前可引用 artifact 中仍有 supporting / archived / live 待重跑边界，尚未形成最终 canonical paper table。

## 顶刊实验口径对照

近年 VEC / MEC / edge caching / task offloading 顶刊或强刊论文通常具备以下实验要求：

| 顶刊常见要求 | 当前状态 | 差距判断 |
| --- | --- | --- |
| 明确问题建模、约束、复杂度或 NP-hard / MINLP 论证 | 项目有系统设定，但论文级形式化仍需整理 | 需要补强 |
| 真实 trace 或公开数据驱动实验 | `NGSIM + Alibaba` 已就绪；HF cache 为 metadata/profile | mobility/workflow 够用，cache trace 不够 |
| 强 baseline 池 | 有 PPO/MAPPO/IPPO、reactive、popularity heuristic | 需要补强强 heuristic / 优化 baseline / CTDE 解释 |
| 多 seed、多窗口、多负载统计 | round13 为 3 seeds、有限 workflows/windows | 需要扩大并补 CI / significance |
| 消融说明每个模块必要性 | 有 ablation artifact，但部分早于当前 live protocol | 需要按当前协议重跑 |
| 鲁棒性和可扩展性 | 有历史 robustness/scalability | 需要按当前算法池和 checkpoint 重新冻结 |
| 机制可解释性 | 有机制诊断，但 mixed 未改善，ready/realization 不领先 | 需要事件级闭环 case study |
| 结果边界清晰 | 文档已开始区分 canonical/supporting/archived | 需要最终 paper-grade 统一表 |

参考口径包括：

- TMC / VFC 多 agent offloading 类工作通常会同时给出问题形式化、强 baseline、真实 mobility trace 或仿真、多指标性能比较。
- TMC / dependent task offloading 类工作会强调 workflow dependency、mobility-aware decision、时延/资源约束和多场景压力测试。
- Edge model caching / cache-assisted offloading 类工作通常会把 cache hit/miss、eviction、storage capacity、backhaul、latency 和 workload variation 作为核心实验维度。

## 当前证据链梳理

### 已经可作为论文基础的部分

1. `NGSIM + Alibaba` 主线跑通。
   - `python scripts/check_data_ready.py` 当前显示 NGSIM、Alibaba、LuST、HF metadata 就绪，highD 缺失。
   - `python scripts/validate_dataset_source_declarations.py` 当前显示数据源声明和 catalog 声明一致。

2. round13 大规模真实数据对比具备 supporting 价值。
   - agents: `sa_ghmappo`、`reactive_greedy`、`popularity_cache_heuristic`、`ppo`、`mappo`
   - seeds: `7/13/29`
   - NGSIM rows: `5000`
   - workflows: `j_3/j_8/j_15/j_34`
   - mixed episodes: `360`
   - full-stratified episodes: `1080`

3. round13 主方法优势：
   - mixed overall reward: `sa_ghmappo=72.257`，高于 popularity `71.819`、reactive `69.840`、PPO/MAPPO `67.968`。
   - full-stratified reward: `sa_ghmappo=75.683`，高于 popularity `74.835`、reactive `70.312`、PPO/MAPPO `69.762`。
   - 相比 PPO/MAPPO，SA 在 reward、success、continuity、adapter miss 上优势明显。

4. mechanism v2 full-stratified 有最低 freeze candidate 形状。
   - full-stratified 下 SA 相比 popularity：reward `+0.032222`、continuity `+0.055696`、handoff failure `-0.143519`、backhaul `-4.444444`、migration overhead `-0.010000`。

### 还不能作为顶刊强 claim 的部分

1. round13 和 mechanism v2 都不能支持“所有关键机制指标全面领先”。
   - popularity 在部分 full-stratified reliability/cache 指标上仍强于 SA。
   - mechanism v2 明确记录 mixed reward 未改善，handoff ready 和 mechanism realization 未提升。

2. HF model-cache transaction round1 不能作为主结果。
   - 只有 seed `7` 和 `2` 个 benchmark episodes。
   - 报告明确标注 `advantage_claim_supported_in_this_round=False`。
   - HF 数据只作为 file-size/cache-volume profile，不能声称真实 VEC cache hit/miss、RSU locality、handoff demand 或 adapter migration trace。

3. 当前 live paper table 尚未冻结。
   - `docs/project/ARTIFACT_RECORDS.md` 已标注历史 aggregate 可能是 archived/supporting。
   - 当前 live 对照算法池需要通过 current scripts 重新生成 paper-grade 主表。

## 原因分析

### 原因 1：主方法对强 heuristic 的优势还不够大

当前 SA 相对 PPO/MAPPO 的优势清楚，但顶刊审稿更会盯住 `popularity_cache_heuristic`。因为它是 prediction-aware heuristic，能直接利用预测信号做 prefetch/migration 相关动作，对你的核心机制构成强对照。

现象：

- round13 mixed 中 SA 只比 popularity reward 高 `0.438`。
- round13 full 中 SA 只比 popularity reward 高 `0.847`。
- full 中 popularity 在 success、continuity、handoff failure、adapter miss 上仍有指标优势。
- mechanism v2 full 中 SA 的综合优势很小，不能承担“SOTA 大幅领先”叙事。

根因判断：

- 当前 reward 优势部分来自 aggregate utility 权重，而不是所有机制指标全面压制。
- popularity 规则更直接、更确定地执行 predictive prefetch / migration prepare，短窗口内可靠性高。
- SA policy 在 mechanism window 中对 prepare/prefetch 的触发频率和时机仍不稳定。

### 原因 2：机制闭环缺少稳定事件级证据

论文主线是跨 RSU DAG workflow、adapter cache、handoff migration、多时间尺度控制。顶刊需要证明机制链条真实工作：

```text
handoff candidate / prediction signal
-> predictive prefetch
-> adapter warm hit / lower miss
-> migration prepare
-> handoff ready
-> fewer stalls / fewer handoff failures
-> lower delay / higher continuity / better reward
```

当前证据仍有断点：

- round1 诊断中 `handoff_ready_count_mean=0.0` 曾经是明显风险。
- mechanism v2 后，full-stratified 综合指标改善，但 `handoff_ready_ratio` 和 `mechanism_realization_rate` 仍低于 popularity。
- `sa_advantage_round1_mechanism_diagnosis.md` 指出 seed `29` 的 `event_prepare_prob_mean=0.139703`，明显低于 seed `7/13` 的约 `0.45`，`guard_prefetch_to_prepare_count` 也只有 `1`，而 seed `7/13` 为 `30`。

根因判断：

- 机制候选信号存在，不是数据窗口完全没有 signal。
- 问题更像 policy/training 稳定性不足，尤其 imitation decay、mechanism window 采样、prepare action probability 和 guard-to-prepare 转换不足。
- checkpoint selection 已经无法从当前训练预算中选出更优安全点，需要回到训练侧改。

### 原因 3：cache 真实性和压力不足

当前正式主线 `NGSIM + Alibaba` 能支撑 mobility + workflow，但 adapter/model cache 仍主要来自本地 catalog/profile 设定。HF round14 和 transaction round1 做了重要补洞，但边界仍明确：

- HF 候选当前不是真实 VEC cache request trace。
- 没有真实 hit/miss、RSU locality、handoff demand、adapter state migration trace。
- HF transaction round1 只是本地 file-size projection 和小样本 sanity。
- round13 报告中也指出 cache capacity / eviction competition 没有充分成为主表压力源。

根因判断：

- 当前 cache 结果更多证明“系统可以表达 cache 行为”，还不是“真实 cache workload 下稳定领先”。
- 顶刊如果审“AI model cache / adapter cache”，会要求 cache capacity、eviction、admission、backhaul、cold start、warm hit 形成强对比实验。

### 原因 4：baseline contract 还不够强

当前 live 对照算法包括 PPO、MAPPO、IPPO、reactive、popularity。问题在于：

- round13 中 PPO 和 MAPPO 数值完全相同，容易被审稿人质疑对照有效性。
- 文档已说明当前 `mappo` 基于 live wrapper 的单步语义决策 contract，不是完整 CTDE multi-agent contract。
- TD3/SAC/MADDPG/QMIX 已从 live registry 移除，理由合理，但论文里需要解释动作空间和 multi-agent contract 为什么不适配。

根因判断：

- 当前主环境动作是 `semantic_discrete_5`，不自然适配 TD3/SAC 的连续动作。
- 完整 MADDPG/QMIX 需要先冻结 multi-agent observation/action schema。
- 对照算法池方向是对的，但还缺一个“审稿人认可的强离散/图/启发式/优化 baseline”。

### 原因 5：统计证据不足以支撑小幅领先

SA 对 PPO/MAPPO 的效果差距较大，三 seed 可以初步说明趋势。但 SA 对 popularity 的优势很小，必须用统计证明：

- 多 seed 下是否稳定。
- 多 window class 下是否稳定。
- 多 workflow size / DAG shape 下是否稳定。
- 不同 cache capacity / mobility density 下是否稳定。
- 是否显著优于强 heuristic，而不是单个 window selection 或 reward 权重导致。

根因判断：

- 当前 `3 seeds × 有限 windows/workflows` 对大优势够用，对小优势不够。
- 缺 confidence interval、bootstrap、paired test、effect size 和 win/tie/loss 统计。

### 原因 6：artifact 分层还没有收敛到最终 paper package

项目文档已经做了很好的 artifact 边界整理，但现在仍存在：

- `supporting`、`archived`、`canonical` 混合。
- 旧 main table 和 current live 算法池不完全一致。
- ablation、robustness、scalability 有些来自早期协议。

根因判断：

- 工程探索阶段产生了很多有效中间证据，但顶刊投稿需要单一、冻结、可复跑、可追溯的 final protocol。

## 改进目标

最终目标不是“多跑几次让表更好看”，而是形成一个可防守的 paper-grade claim：

> 在 `NGSIM + Alibaba` 真实 mobility/workflow trace 下，`sa_ghmappo` 通过 graph-hierarchical multi-agent policy、prediction-aware prefetch、adapter cache coordination 和 handoff migration preparation，在多窗口、多 seed、多负载和 cache-stress 设置中，相比强 RL baseline 和 prediction-aware heuristic，显著提升 workflow utility，并在 continuity、handoff failure、adapter miss/backhaul 或 cache-stress 指标上保持稳定优势。

该 claim 的最低验收线：

- SA 对 PPO/MAPPO/IPPO 有明显优势。
- SA 对 `popularity_cache_heuristic` 不仅 reward 高，还至少在 full-stratified 和 cache-stress 中稳定赢 continuity 或 handoff failure 或 adapter miss/backhaul 的核心组合。
- 机制指标不要求全部最高，但必须解释清楚 trade-off，并有事件级案例证明 SA 的机制链真实生效。

## 实施方案

### Phase 0：冻结投稿实验协议

目标：先冻结 protocol，避免继续产生互相冲突的 artifact。

动作：

1. 新增 `configs/experiment/paper/top_journal_ngsim_alibaba.yaml`。
2. 明确 agents：
   - `sa_ghmappo`
   - `ippo`
   - `ppo`
   - `mappo`
   - `reactive_greedy`
   - `popularity_cache_heuristic`
   - 可选：新增 `dag_critical_path_cache_heuristic` 或 `oracle_window_greedy` 作为强非学习上界/近似上界。
3. 明确 seeds：至少 `7/13/29/43/71`，若时间允许扩到 10 seeds。
4. 明确 windows：
   - `mechanism_activating`
   - `active_non_mechanism`
   - `idle_or_sparse`
   - 每类至少 `8-12` windows。
5. 明确 workflows：
   - 至少 `8-12` 个 Alibaba DAG workflows。
   - 分层覆盖 small / medium / large DAG。
6. 明确 metrics：
   - utility: reward、success、delay。
   - continuity: workflow_continuity、handoff_failure、stall_count。
   - cache: warm_hit、adapter_miss、cold_start、backhaul、eviction、occupancy。
   - mechanism: prefetch_request、validated_prefetch、migration_prepare、handoff_ready、mechanism_realization。
   - cost: migration overhead、backhaul cost。

验收产物：

- `run_manifest.json`
- `seed_checkpoint_manifest.json`
- `aggregate_summary.json`
- `benchmark_rows.csv`
- `statistical_summary.json`
- `paper_main_table.json`
- `paper_claim_summary.json`

### Phase 1：修复 SA 机制窗口不稳定

目标：解决 seed `29` mechanism window prepare/prefetch 弱的问题，让主方法不是只在 full overall 靠 idle/sparse 胜出。

动作：

1. 增加 mechanism window 训练采样权重。
   - 对 `mechanism_activating` windows 做 replay / oversampling。
   - 保证每个 seed 训练中机制窗口样本数足够。

2. 延缓或分段控制 imitation decay。
   - 当前诊断显示 seed29 的 prepare probability 明显低。
   - 应让 heuristic-guided prepare 行为在训练后期仍保持最低保留强度。

3. 增加 prepare/prefetch 行为的稳定性约束。
   - 对 valid target + timing active 状态，增加 prepare action probability floor。
   - 记录 `event_prepare_prob_mean`、`guard_prefetch_to_prepare_count`、`gate_pass_rate` 作为 checkpoint selection 必选字段。

4. checkpoint selection 保留 V2，但增加 hard filters。
   - continuity floor。
   - handoff failure ceiling。
   - mechanism window reward floor。
   - event prepare probability floor。
   - seed-level worst-case penalty。

验收标准：

- seed `29` 的 `event_prepare_prob_mean` 不再显著低于 seed `7/13`。
- mixed mechanism windows 中 SA 不低于 popularity 的 reward 或 continuity 差距显著收窄。
- full-stratified 中 SA 保持 reward、continuity、handoff failure、backhaul 优势。

### Phase 2：强化 cache-stress 真实口径

目标：把 adapter cache 从“能表达”提升为“能形成顶刊实验维度”。

动作：

1. 将 HF model-cache 继续限定为 file-size/cache-volume profile，不写成真实 trace。
2. 在正式实验中加入 cache-stress profile：
   - small slots: `rsu_adapter_slots=1/2`
   - medium slots: `3/4`
   - relaxed slots: `6+`
3. 输出 cache-specific 表：
   - warm hit ratio
   - miss count
   - cold start frequency
   - eviction count
   - admission count
   - occupancy
   - backhaul traffic
   - migration overhead
4. 如果能获取真实 request trace，再建立独立 `real_cache_trace_profile`，否则只声称 file-size projection。

验收标准：

- 在至少一个 constrained cache setting 中，SA 相比 popularity 体现更优 trade-off。
- 若 SA reward 低于 popularity，必须解释是 backhaul / eviction / migration trade-off，并给出受约束场景优势。

### Phase 3：补强 baseline

目标：让 baseline 体系能经得住审稿。

动作：

1. 保留当前 baseline：
   - `reactive_greedy`
   - `popularity_cache_heuristic`
   - `ippo`
   - `ppo`
   - `mappo`

2. 补一个强启发式 baseline：
   - `dag_critical_path_cache_heuristic`
   - 逻辑：优先 critical path task，结合 predicted next RSU、adapter warm state 和 cache capacity 做 prefetch/migration。

3. 补一个上界/近似上界：
   - `oracle_handoff_cache_greedy`
   - 使用 future handoff 信息，只作为 upper bound，不作为公平在线 baseline。

4. 对 PPO/MAPPO 做 contract 解释：
   - 当前 MAPPO 是 single-wrapper semantic decision，不是完整 CTDE。
   - 不声称 full MARL SOTA，只作为 direction-matched trainable baseline。

验收标准：

- PPO、IPPO、MAPPO 的结果不应长期完全相同；如果相同，报告中必须解释 contract 和实现路径，或修正 baseline。
- SA 对至少一个强 heuristic 和 trainable baseline 有稳定优势。

### Phase 4：统计和显著性

目标：把“看起来赢”变成“统计上可防守”。

动作：

1. 新增统计脚本：
   - paired bootstrap CI
   - paired t-test / Wilcoxon signed-rank
   - Cliff's delta 或 Cohen's d
   - win/tie/loss by window and workflow

2. 输出：
   - `statistical_summary.json`
   - `pairwise_significance.csv`
   - `effect_size.csv`
   - `win_tie_loss_by_window.csv`

3. 对 claim 做分级：
   - strong supported
   - supported with trade-off
   - supporting only
   - not supported

验收标准：

- 对 PPO/MAPPO/IPPO 的 reward 和 continuity 优势显著。
- 对 popularity 的核心 claim 至少在 full-stratified 或 cache-stress 维度显著。
- 如果不显著，只能写成趋势或 trade-off，不写成全面领先。

### Phase 5：当前协议下重跑消融、鲁棒性、可扩展性

目标：形成最终 paper package。

动作：

1. Ablation：
   - full SA
   - no graph encoder
   - no hierarchy
   - no prediction signal
   - no adapter prefetch
   - no migration prepare
   - no uncertainty signal
   - no DAG dependency awareness

2. Robustness：
   - prediction noise
   - missing prediction
   - mobility density shift
   - workflow size shift
   - cache capacity shift
   - RSU count/layout shift

3. Scalability：
   - workflows: `2/4/8/12+`
   - windows: `6/12/24+`
   - RSU count
   - adapter catalog size

验收标准：

- 每个核心模块至少在一个机制指标或 utility 指标上有可解释贡献。
- 鲁棒性下降可控，且强于 baseline 或给出明确 trade-off。
- 可扩展性曲线不出现无法解释的崩溃。

## 推荐最终论文实验表结构

1. Main table：overall mixed/full-stratified performance。
2. Mechanism table：mechanism windows 下 prefetch / migration / ready / continuity。
3. Cache-stress table：slots 变化下 cache/backhaul/cold-start/eviction。
4. Ablation table：模块贡献。
5. Robustness table：prediction noise / mobility density / workflow size。
6. Scalability figure：RSU/workflow/catalog/window scale。
7. Case study：一个 handoff window 的 step-level event trace。
8. Statistical table：pairwise significance and effect size。

## 可执行优先级

最高优先级：

1. 冻结 `top_journal_ngsim_alibaba` 协议。
2. 修 seed `29` mechanism prepare/prefetch 不稳定。
3. 扩大 `5 seeds × more windows × more workflows`。
4. 增加统计脚本。
5. 重跑 main + full-stratified + cache-stress。

第二优先级：

1. 新增强 heuristic / oracle upper bound。
2. 当前协议下重跑 ablation。
3. 当前协议下重跑 prediction robustness 和 scalability。

第三优先级：

1. 尝试接入真实 cache request trace；若没有，保持 HF file-size profile 边界。
2. 完整 multi-agent wrapper，为后续 MADDPG/QMIX 或 CTDE MAPPO 做准备。

## 预期风险

- 若 SA 机制训练增强后仍不能稳定超过 popularity，论文 claim 应收缩为“better trade-off under full-stratified/cache-stress”而不是“全面领先”。
- 若真实 cache trace 无法获得，不应把 HF profile 写成真实 cache workload。
- 若 PPO/MAPPO 继续完全一致，必须在论文中解释 contract 限制，或重做 baseline。
- 若扩大 seeds/windows 后优势消失，应优先修改算法训练侧，而不是调 reward 或选择性过滤窗口。

## 建议下一步实施任务

下一步不应继续做零散实验，而应按以下顺序执行：

1. 新增 paper-grade 实验配置和统计输出 schema。
2. 修改 SA 训练配置，提升 mechanism windows 和 prepare/prefetch 稳定性。
3. 跑 1 个 `5 seeds × 6 windows × 4 workflows` 的 pilot，验证 seed29 是否修复。
4. 若 pilot 通过，再扩到 final protocol。
5. 生成最终 paper tables 和 claim summary。

## 本报告使用的本地依据

- `docs/project/CONTEXT.md`
- `docs/project/PROGRESS.md`
- `docs/project/BUGS.md`
- `docs/project/ARTIFACT_RECORDS.md`
- `docs/agent/large_scale_real_dataset_round13_report.md`
- `docs/agent/sa_advantage_round1_mechanism_diagnosis.md`
- `docs/agent/sa_advantage_round1_mechanism_improvement_report.md`
- `artifacts/experiments/hf_model_cache_transaction_round1/*/hf_model_cache_transaction_round1_report.md`

## 外部论文口径参考

- TMC: `Many-to-Many Task Offloading in Vehicular Fog Computing: A Multi-Agent Deep Reinforcement Learning Approach`, DOI `10.1109/TMC.2023.3250495`
- TMC: `BARGAIN-MATCH: A Game Theoretical Approach for Resource Allocation and Task Offloading in Vehicular Edge Computing Networks`, DOI `10.1109/TMC.2023.3239339`
- TMC: `Dual Dependency-Aware Collaborative Service Caching and Task Offloading for Vehicular Edge Computing`, DOI `10.1109/TMC.2025.3573379`
- Computer Communications: `Optimizing task offloading in cache-assisted vehicular edge computing: A spatio-temporal fusion graph convolutional network approach`
- Computer Networks: `Lyapunov-guided deep reinforcement learning for stable online computation offloading in vehicular edge computing`
