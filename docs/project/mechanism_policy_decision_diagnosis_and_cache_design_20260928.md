# 策略决策能力诊断与 model-cache 核心机制设计（2026-09-28）

## 审查元数据与边界

| 字段 | 值 |
|---|---|
| `reviewed_at` | `2026-09-28T11:42:02+08:00` |
| `literature_cutoff` | `2026-09-28` |
| `target_venue` | IEEE TMC（定位参考，不构成录用判断） |
| `artifact_run_id` | `mechanism_policy_decision_diagnosis_v1_20260928`；源证据为 `mechanism_frozen_evaluation_v3_20260928` / `mechanism_algorithm_retraining_v3_20260928` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| audited Git commit | `2de39b9` |
| evidence level | `E2_ARTIFACT_AUDITED_PLUS_READ_ONLY_NO_STEP_PROBE` |
| verdict | `UNVERIFIED`：策略差异诊断成立；算法优势、性能因果和 paper-ready 均未建立 |

本轮只读现有代码、checkpoint 和训练/评价产物；另在 `/tmp` 中加载冻结 checkpoint 做固定输入探针。
没有调用环境 `step`，没有训练、调参、正式 rollout、holdout 访问或旧产物改写。完整结构化摘要见
`artifacts/analysis/mechanism_policy_decision_diagnosis_v1_20260928/diagnostic_summary.json`。反事实输入探针只回答
“网络是否能感知某字段并改变输出”，不回答该字段或动作是否提高性能。

## 结论先行

SA-GHMAPPO 与 MAPPO 的相同冻结行为不是 checkpoint 误加载，也不能归结为两个网络参数相同。两个模型真实结构不同，
分别为 `SurrogateFusionEncoder` 的 165,320 参数和 `FlatSemanticEncoder` 的 39,176 参数；`update_0004→0016`
分别有 60/60 和 26/28 个 state tensors 改变，加载后的 tensor digest 与各自 checkpoint 完全一致。

一阶原因是**可执行决策接口把不同内部概率压缩到了相同的确定性边界**：

1. 三个 head 先被固定映射为五个环境动作分数，再在 mask 内直接做环境动作 argmax；评价不是三个 head 各自 argmax
   后自由组合。当前两策略虽概率不同，但在这 123 个已观察状态上都只把 action 3/4 置于确定性首位。
2. v3 current-only readiness guard 在原决策之后共同把 38/123 个动作改成 action 0。guard-on 的全部 SA/MAPPO
   最终动作再次逐步相同，系统级 override 因而覆盖了 30.9% 的 learned decision authority。
3. 掩码不是把选择强迫成唯一动作：123 步每步都有至少 3 个合法动作。相同行为不能解释为 mask 单动作退化。
4. 当前 actor observation 对 typed base readiness、缺失 dependency bundle 大小、真实 cache slack/eviction opportunity、
   workflow-state migration readiness 和 remaining-DAG adapter reuse 均不足或完全不可见。策略无法针对这些核心机制状态
   形成可审计的不同决策边界。
5. 训练时 guard 后会用强制动作重算 head log-prob，并将最终 canonical head action 存入 rollout。log-prob **存在**，
   但这不等于 raw policy 获得了可识别的因果 credit：动作由外部 override 产生，却按原网络对该强制动作的概率进入 PPO。
   这会训练网络复现 guard 的输出，但不能证明 raw policy 自己发现了该动作。

因此原因分类为“**结构压缩 + 状态不足 + override credit 不可辨识**”的组合；“训练不足”最多是次级待检假设，不能作为
当前一阶结论。也不能排除两个策略在这组窄状态上碰巧选择同一合理解，但现有产物没有保存逐请求 raw logits，无法用历史
轨迹证明这一解释。

## 逐步动作与参数审计

### 历史冻结轨迹

12 个 matched unit、123 步中：

| 比较 | 逐步完全相同 | 动作计数 | action stream SHA-256 |
|---|---:|---|---|
| SA guard-on vs MAPPO guard-on | 123/123 | `0:38, 3:60, 4:25` | `7a7a135f...dc81bcd9` |
| SA guard-off vs MAPPO guard-off | 123/123 | `3:86, 4:37` | `525c7ebb...d6cb6cc` |

guard-on/off 对每个算法都有 38 个差异位置，且全部是 guard-on 的 action 0；对应 guard-off 为 action 3（26 次）或
action 4（12 次）。合法集合为 `{0,2,3}` 86 次、`{0,1,2,3,4}` 26 次、`{0,2,3,4}` 11 次；单一合法动作
为 0 次。逐 unit 动作串已写入诊断 JSON，可复核“逐步相同”而非仅比较汇总计数。

历史 summary 的可观测边界必须保留：`policy_trace_brief` 和 `decision_observation_trace` 保存最终 action、mask 与有限的
pre-action observation，却没有 `raw_env_action`、`head_actions`、raw logits、`action_probs`、`env_action_probs` 或
`head_log_probs`。因此不能在不重新执行环境的前提下恢复 123 个真实状态的 head-level 原始决策。本轮没有为填补该缺口
重跑环境。

### checkpoint 身份、加载与更新

| 项 | SA-GHMAPPO | MAPPO |
|---|---:|---:|
| class / encoder | `SAGHMAPPOAgent` / `SurrogateFusionEncoder` | `MAPPOAgent` / `FlatSemanticEncoder` |
| 参数量 | 165,320 | 39,176 |
| state tensors | 60 | 28 |
| `update_0004→0016` 改变 | 60/60 | 26/28 |
| relative L2 delta | 0.112417 | 0.185024 |
| loaded `update_count` | 16 | 16 |
| 加载后 tensor digest | 与各自 `update_0016.pt` 完全一致 | 与各自 `update_0016.pt` 完全一致 |

四个被读取 checkpoint 的文件 SHA 在探针前后完全不变。v3 `update_0016.pt` 文件 SHA 分别为
`67c9c940...d31ffeb` 和 `a8d3a9dd...c82d627`，与 v3 freeze manifest / evaluation receipt 一致。所有 12 个
SA/MAPPO summary 的 checkpoint path 也分别指向对应的 v3 agent 路径，未发现串线或回退到 `latest.pt`。

### 不推进环境的固定输入探针

全动作合法、相同受控输入下，两者 deterministic raw action 都是 4，但分布显著不同：SA 的 action 3/4 概率为
`0.456826/0.496796`，MAPPO 为 `0.083787/0.633751`。固定 seed 1401、直接调用冻结策略 sampler 200 次时，
全合法 mask 下 SA 的 `0/1/2/3/4` 计数为 `0/3/8/84/105`，MAPPO 为 `14/23/21/14/128`；在常见
`{0,2,3}` mask 下分别为 `0/15/185` 与 `67/76/57`。这证明内部策略不是同一分布，不能从 deterministic action
相同反推 stepwise policy equivalence。

受控 perturbation 还显示：SA 对当前 adapter readiness 会改变 argmax，二者都对 predicted handoff 和 progress
改变 logits；但单独修改 base readiness、dependency bundle bytes、migration readiness、cache capacity 或 remaining
adapter identity/reuse 时，两者输出均完全不变。SA 对仅增加无关 resident adapters 的 env-score 最大变化只有
`1.19e-4`。这些是输入敏感性证据，不是机制收益证据。

## 原因证据表

| 候选原因 | 分类 | 证据 | 判定 |
|---|---|---|---|
| checkpoint 串线、同权重或未更新 | bug | 路径、SHA、class、encoder、tensor digest、update 4→16 diff 均独立且一致 | 排除 |
| mask 强迫同一唯一动作 | structure | 每步至少 3 个合法动作 | 排除 |
| 五动作聚合压缩 head 差异 | structure | `_hierarchical_env_action_scores` 固定把 head log-prob 映射到 5 个动作；有 mask 时直接从该分布选择 | 一阶成立 |
| readiness guard 主导最终行为 | structure/system | 38/123 最终动作被共同改写；guard-on 两算法逐步完全相同 | 一阶成立 |
| typed cache / migration state 不足 | state | 关键字段源码缺失，且 no-step perturbation 对多个字段为精确 0 | 一阶成立 |
| PPO 有 log-prob 即有有效 credit | training | guard 后重算并保存强制动作 log-prob；动作不是 raw policy sample | 否；数值存在但 attribution 不可辨识 |
| 两算法只需更多训练 | training | 权重确实更新，但当前证据没有收敛曲线、多 seed 或解除结构瓶颈后的结果 | 未证实，不能作一阶解释 |
| 两者在窄任务上合理地找到同一解 | reasonable same solution | raw 概率不同但 deterministic argmax 相同，理论上可能 | 未排除；历史 raw logits 缺失，不能确认 |
| evaluation trace 缺少 raw policy provenance | instrumentation bug/risk | summary 不保存 head/logit/prob/override 前后链 | 成立；阻止精确事后归因，但不等同执行结果错误 |

## 决策 authority 与信息矩阵

### Authority

| 层 | 当前 authority | 应保留/调整 |
|---|---|---|
| action mask / schema | 决定物理合法动作 | 保留；不得为放大算法差异而移除物理约束 |
| slow/fast/event heads | 产生 cache、execution、handoff 偏好 | 保留，但必须保存 raw logits/probs/head action |
| 五动作聚合器 | 把三个 head 压成单一 env action，并在 mask 上采样/argmax | 明确记录 pre/post aggregation；新机制若扩展 contract 必须新版本 |
| readiness guard | policy 后强制 current cache fill | 可作为 safety baseline；不得同时充当“算法学会 readiness”的证据 |
| cache runtime | base→adapter 原子 admission、容量、淘汰、迁移等物理执行 | 完整保留，不交给策略绕过 |
| predictor | 提供 as-of-now 的 next-RSU/confidence/uncertainty | 只用因果预测和 age/mask，禁止 actual future/oracle truth |

### Actor 信息充分性

| 决策信息 | SA actor | MAPPO actor | 当前缺口 |
|---|---|---|---|
| 当前 adapter readiness | 二值可见 | 不可见 | 缺 typed ID/size/missing bundle |
| required base readiness / sharing | 不可见 | 不可见 | 无法决定 base reuse 与原子 admission 成本 |
| remaining DAG topology/progress | 图结构可见 | 仅 progress 与当前节点度数 | 均缺 critical-path 加权 typed reuse demand |
| per-RSU cache competition | resident adapter count 粗粒度可见 | actor 不可见 | 缺 free MB、item size、eviction cost/benefit、pending transfer |
| migration state | 不可见 | 不可见 | 缺 workflow checkpoint bytes/readiness/age |
| handoff risk | predicted target/sequence/confidence 可见 | 存在性、统计量可见 | 缺 snapshot age 与风险分布；不得换成真实未来 |
| dependency bundle | 不可见 | 不可见 | 缺 base+adapter+workflow-state 联合缺口与 transfer size |

## 唯一候选机制：CRDCM

候选名为 **Critical-path Reuse-aware Dependency Cache and Migration（CRDCM）**。它不再增加第二个独立创新点，
而是把项目已有 typed base/adapter cache 与 workflow-state migration 收束为一个可证伪机制：依据**已知 workflow DAG**
的 remaining critical path 和 adapter/base reuse，以及**当前可观测或因果预测**的 handoff risk，在当前/预测目标 RSU
联合选择 dependency bundle admission 与 workflow-state prepare。DAG 定义本身属于到达时已知请求，不是未来 outcome；
RSU 去向只允许使用带 timestamp/age/confidence 的 predictor snapshot。

### 状态、动作与目标

- 状态：当前节点 required base/adapter；当前和候选 RSU 的 typed readiness、free MB、resident item size/last-use、
  pending transfer；remaining DAG 对每个 typed item 的 reuse count、critical-path weight、frontier demand；workflow-state
  checkpoint bytes/readiness；预测 handoff target distribution、ETA、confidence、uncertainty 和 snapshot age。
- 动作：在物理 mask 内选择 `{none, current dependency bundle, next-critical dependency bundle}` ×
  `{current RSU, predicted target RSU}`，并选择 workflow-state `{keep, prepare}`。base→adapter 原子 admission、容量和淘汰
  仍由 runtime 强制；不能通过策略跳过。
- 目标：以 completed-workflow return / right-censor-aware completion credit 为主，显式扣除 transfer、backhaul、eviction、
  stale prefetch 和 migration cost；同时报告而非隐藏 Pareto trade-off。
- credit：执行 trace 必须同时保存 raw selection、masked selection、runtime projection、guard override、executed action 及各自
  log-prob。若 safety override 触发，raw-policy PPO sample 不得伪装成原分布采样；可采用 shield-aware behavior distribution
  或将该步 raw actor credit mask 掉，并把 override 作为独立监督/约束信号。具体选型需在实现任务中冻结。

### 与最近邻工作的差异

最接近的是 2025 TMC 的 Dual Dependency-Aware Collaborative Service Caching and Task Offloading：它做 critical-service
预测、主动/被动 service caching 与 PPO joint caching/offloading。CRDCM 的必要差异必须是 typed base sharing + adapter
dependency + workflow-state checkpoint 在**跨 RSU 连续 DAG**上的联合 dependency bundle 和迁移决策，而不是把该论文的
service cache 换名。2026 TMC 的 timing/data-dependent DAG scheduling 强调 latest-start/longest-path 和 task offloading，
但不解决 typed model-cache sharing；AWTO 处理 agentic workflow 的按任务 offloading、on-demand model loading 与 memory
contention，但不是 VEC handoff 下的跨 RSU state migration。S-LoRA 只为共享 base/adapter 内存模型提供系统合理性，不是
本项目的调度算法。上述论文均已存在于 `literature_reference_table.md`，本轮没有新增重复记录。

若实现后无法在 base reuse、binding capacity、handoff/migration opportunity 同时存在时产生可重复主效应，则 CRDCM
应被否证或降级为工程机制，不得以叙事保留为算法贡献。

## 最小实现范围与预算估计（未授权执行）

任何 observation/action 或 log-prob contract 变化都必须发布新方法版本（建议 `crdcm_v1` / evaluation contract v2），
不得覆盖 v3 checkpoint。最小实现约涉及：

1. 环境 semantic state 只新增因果可见的 typed readiness/slack/pending migration/remaining-DAG reuse summaries；
2. action schema 增加 factorized bundle/target/migration action 和严格 runtime mask/projection；
3. SA/MAPPO/PPO 获得同等 action authority，分别使用 graph、flat-CTDE、flat-PPO 编码；
4. 强 heuristic 使用同一状态、mask、候选 bundle 与成本模型；
5. trace 强制保存 raw logits/probs、head actions、aggregation、projection、override、executed action/log-prob；
6. producer/consumer、checkpoint、manifest、summary 和 benchmark schema 同步升级并加合同测试。

保守工程量为 6–9 个生产/消费模块及 4–6 个定向测试文件，约 3–5 个实现日加 1–2 个审计日；这只是范围估计，
不是进度承诺。首轮 falsification training 若使用 5 个 learned conditions（CRDCM、mechanism-off ablation、matched
SA、MAPPO、PPO）× 3 seeds × 每 condition/seed 64 episodes，则为 960 episodes，上限 19,200 environment steps；
heuristic 不训练。该预算必须在新数据 split、状态/动作合同、停止规则冻结后另行批准，不能复用 v3 继续训练。

## 最小可证伪实验

### 比较与 estimand

- controllers：CRDCM、matched SA-GHMAPPO、MAPPO、PPO、一个 dependency-aware strong heuristic；learned controller
  同 seed、budget、action authority、mask、data exposure，heuristic 与其共享同一 candidate bundle 和 runtime。
- mechanism main effect：CRDCM full vs CRDCM mechanism-off（去掉 critical-path reuse/bundle/migration state，重新训练，
  不是推理时掐开关）；并用 full heuristic vs readiness-only heuristic 检查系统机制是否无需学习也成立。
- policy gain：在 mechanism-on、相同 authority 下比较 CRDCM learned policy 与 strong heuristic；不得把 guard 主效应
  计为 policy gain。
- clusters：新的未消费 train/dev/formal window 分组，按原始 frame/time interval 验证互斥；outer cluster 用 window，
  不能把重叠 window ID 当独立样本。旧 holdout 不重开。

### opportunity strata 与负控

1. 正控：binding capacity + 至少两个 adapter 共享同一 base + remaining DAG 有重复 reuse + 可预测 handoff。
2. no-reuse 负控：每个 remaining adapter/base 只请求一次；CRDCM 不应凭空产生 cache gain。
3. ample-capacity 负控：容量不绑定；eviction-aware 部分的收益应收敛到零。
4. no-migration 负控：无 RSU change 或 state migration disabled；migration 分支收益应收敛到零。
5. unreliable-prediction 负控：只降低 causal predictor confidence/增加 age，不泄露真实未来；策略应减少远端 prepare，
   而不是保持相同激进度。

### Pareto 指标与停止规则

主指标为 completion rate、workflow continuity、completed-only conditional delay 与 right-censored count；成本面为
transfer MB/request、backhaul、eviction、stale/expired prefetch、migration overhead。报告 per-window paired effects、
cluster bootstrap CI、Holm 校正和 Pareto frontier，不构造事后 weighted score。

最小否证条件：正控下 full vs mechanism-off 在 completion/continuity 没有方向一致的 window-level改善，或改善完全由
transfer/backhaul 无界增加换取；policy gain 对 strong heuristic 不成立；或任一负控仍显示同量级“收益”。满足任一项即
停止扩大训练并回到 contract/estimand 审查，不追加 seed 追结果。

## 下一步建议与停点

当前优先级不是继续训练 v3，而是先批准一个独立实现任务：冻结 CRDCM 的 causal state schema、factorized action authority、
shield-aware credit 规则和 raw-policy trace schema，并只做 unit/contract/no-step probe。代码审查通过后，再单独批准新版本
matched retraining 和新 split falsification。当前任务在设计与诊断处停止，不自动实现或启动训练。

## 公开文献核验

- IEEE TMC 2025, [Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehicular Edge Computing](https://ieeexplore.ieee.org/document/11014496/), DOI `10.1109/TMC.2025.3573379`。
- IEEE TMC 2026, [Optimization of Task Scheduling Strategy With Timing and Data Dependencies in Vehicular Edge Computing](https://ieeexplore.ieee.org/document/11304528/), DOI `10.1109/TMC.2025.3646450`。
- FGCS 2026, [AWTO: Adaptive Workflow Task Offloading for Multi-Agent Systems in Edge Computing Environments](https://www.sciencedirect.com/science/article/pii/S0167739X2600049X), DOI `10.1016/j.future.2026.108415`。
- MLSys 2024, [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html)。
