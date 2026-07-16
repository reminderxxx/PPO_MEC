# SA-GHMAPPO Paper Method Report

- `prepared_at`: `2026-07-16`
- `target_use`: 论文 Problem Formulation / Method / Algorithm Design 初稿素材
- `main_algorithm`: `sa_ghmappo`
- `canonical_evidence_context`: strict-full v8, `E2_ARTIFACT_AUDITED`, reviewer verdict `Major revision`
- `source_code_scope`: `src/agents/sa_ghmappo_agent.py`, `src/agents/sa_ghmappo_core.py`, `src/encoders/`, `src/envs/specs/action_schema.py`

## 写作边界

本文档只固化研究问题与方法表述，不把当前结果写成已通过顶刊。论文写作必须保留以下边界：

- 当前主算法是 controller-level hierarchical multi-controller PPO，不是 vehicle-agent / RSU-agent full MARL。
- 当前动作空间是 `semantic_discrete_5`，不是 DAG node parameterized action，也不是连续控制动作。
- 当前 predictor 主结果仍基于 calibrated / baseline surrogate interface；`supervised_handoff_predictor_v1` 已有代码路径，但在冻结 checkpoint、quality report 和 v9 benchmark 前，不写成当前主结果已使用 learned predictor。
- policy guards 是 safety / feasibility layer，应与 learned core 分开归因；不能把 guard 带来的收益全部写成端到端学习能力。
- 可安全 claim 是 strict-full v8 相对 learned baselines 的 total reward 优势和相对 DT handoff baseline 的 continuity 优势；不能声称全面降低 handoff failure 或 backhaul。

## 1. 论文级研究问题

本研究面向 AI-driven vehicular edge computing 中的连续工作流执行问题。车辆在行驶过程中会从一个 RSU 切换到另一个 RSU，而车载 AI 应用通常不是单个独立任务，而是由多个具有先后依赖关系的 DAG workflow 节点组成。每个节点可能需要特定 adapter、输入中间状态和可用计算位置。当车辆接近 RSU 覆盖边界时，下一 RSU 未必已经拥有所需 adapter，当前节点或后继节点的执行状态也未必能无缝迁移。因此，系统需要在车辆移动、DAG 依赖、adapter cache 和 handoff 状态迁移之间做联合控制。

可以在论文中将核心问题表述为：

> We study mobility-driven continuous AI workflow execution in vehicular edge computing, where a vehicle-bound DAG workflow must continue across RSU handoffs while its adapter warm states, execution placement, and handoff preparation are jointly controlled under uncertain mobility predictions.

中文对应表述：

> 本文研究车辆跨 RSU 移动场景下的连续 AI DAG workflow 执行问题。与单任务卸载或静态 service caching 不同，本文将 workflow 执行进度、DAG 依赖 frontier、RSU adapter warm-state、移动预测、handoff prepare 和执行状态迁移纳入同一决策闭环，以提升综合 workflow reward 和 continuity，并显式报告 handoff failure 与 backhaul cost 的 trade-off。

## 2. 系统模型

系统由车辆集合、RSU 集合、adapter catalog 和 DAG workflow 组成。每个车辆在时刻 `t` 关联一个当前 RSU，并携带一个正在执行的 workflow。每个 workflow 节点具有输入大小、输出大小、前驱、后继和所需 adapter。RSU 具有覆盖半径、当前活跃车辆、cache 容量、已缓存 adapter 列表和预测负载。预测层提供 next-RSU sequence、first handoff target、dwell time、future load、confidence 和 uncertainty。

令 workflow DAG 为 `G=(V,E)`。节点 `v in V` 表示一个 AI subtask，边 `(u,v) in E` 表示 `v` 依赖 `u` 的输出。时刻 `t` 的已完成节点集合为 `C_t`，可执行 frontier 为：

```text
F_t = { v in V \ C_t | Pred(v) subset C_t }.
```

当前执行节点 `v_t` 来自 frontier 或执行计划。每个节点有 required adapter `a(v_t)`。当前 RSU 为 `r_t`，预测下一 RSU 为 `\hat r_{t+1}`，预测首个 handoff target 为 `\hat r^h_t`。

### 控制目标

控制器不是单纯最小化时延，而是在以下指标之间进行联合优化：

- workflow service reward / completion progress；
- DAG continuity 与 handoff 后继续执行能力；
- adapter warm hit 与 cache miss；
- delay penalty；
- migration / prepare cost；
- backhaul cost；
- handoff failure rate。

论文中应避免把 reward 写成唯一系统目标。正确写法是：reward 是训练目标，continuity、failure、backhaul、validated prefetch hit、realized prepare 等是机制兑现和系统 trade-off 指标。

## 3. MDP / POMDP 表述

本文将问题建模为带预测观测的 episodic MDP。严格来说，车辆未来移动与 RSU 负载只通过 surrogate prediction 部分可见，因此也可称为 prediction-augmented partially observable control problem。实现上使用 semantic state 构造策略输入。

状态：

```text
s_t = (G_t, v_t, C_t, F_t, R_t, x_t, P_t)
```

其中 `G_t` 是 DAG 结构和执行状态，`v_t` 是当前节点，`R_t` 是 RSU/cache/load 状态集合，`x_t` 是主车辆位置、速度、当前 RSU 和服务剩余步数等车辆状态，`P_t` 是预测层输出，包括 next-RSU sequence、handoff target、dwell time、future load、prediction confidence 和 uncertainty。

动作采用稳定的五类语义离散 contract：

| ID | action name | 系统语义 |
|---:|---|---|
| 0 | `current_rsu_cache_fill` | 当前 RSU 对当前节点所需 adapter 做反应式加载，并在当前 RSU 执行。 |
| 1 | `predictive_next_rsu_prefetch` | 向预测下一 RSU 预取当前 / 即将需要的 adapter，当前节点维持稳态执行。 |
| 2 | `vehicle_fallback` | 当前节点回退到车辆本地执行，避免不可靠 RSU / cache 状态导致 stall。 |
| 3 | `current_rsu_steady_offload` | 在当前 RSU 稳态卸载执行，不改变 cache。 |
| 4 | `handoff_migration_prepare` | 为预测 handoff target 准备 adapter / execution state migration。 |

动作 mask 依据语义前置条件屏蔽非法动作。例如，没有 current workflow node 时全部动作无效；预测下一 RSU 与当前 RSU 不同且目标 adapter 尚未 ready 时才允许 predictive prefetch；存在 distinct handoff target 时才允许 handoff prepare。

转移由 VEC workflow environment 决定，包括 DAG 节点完成、cache 更新、handoff event、migration outcome、RSU 负载变化和 reward 生成。策略学习目标为最大化折扣回报：

```text
max_pi E_pi [ sum_t gamma^t r_t ].
```

但论文评估必须同时报告 reward 以外的系统指标。

## 4. 主算法概述

主算法 `SA-GHMAPPO` 全称为 Surrogate-Assisted Graph Hierarchical Multi-Agent PPO。为了避免命名被误解，论文中建议解释为 controller-level multi-controller PPO：slow / fast / event 三个 controller heads 共享 graph-surrogate encoder 和 centralized critic，并不声称每辆车或每个 RSU 是独立 agent。

算法有四个核心部件：

1. DAG-RSU-mobility surrogate fusion encoder；
2. slow / fast / event 三时间尺度 actor heads；
3. centralized critic with optional continuity / uncertainty-aware context；
4. mechanism-aware auxiliary guidance 与 policy-side safety guards。

## 5. DAG-RSU-Mobility Surrogate Fusion Encoder

### 5.1 DAG graph encoder

DAGGraphEncoder 对 workflow nodes 做轻量 message passing。每个节点使用 10 维特征：

1. normalized input size；
2. normalized output size；
3. predecessor count；
4. successor count；
5. is current node；
6. is completed；
7. is frontier；
8. required adapter cached at current RSU；
9. required adapter cached at predicted next RSU；
10. required adapter cached at handoff target RSU。

节点特征先经 MLP 投影到 64 维。若启用 dependency-aware message passing，则沿 DAG 边分别聚合 predecessor 和 successor embedding。每轮更新形式可写为：

```text
h_v^{k+1} = LN(h_v^k + ReLU(W_self h_v^k
             + W_pred mean_{u in Pred(v)} h_u^k
             + W_succ mean_{w in Succ(v)} h_w^k)).
```

当前实现默认进行 2 轮 message passing。输出包括：

- graph embedding：所有节点 embedding 均值；
- current-node embedding：当前执行节点 embedding；
- frontier embedding：frontier 节点 embedding 均值。

### 5.2 RSU state encoder

RSUStateEncoder 对 RSU 集合、cache 状态和预测负载编码。每个 RSU 使用 10 维特征：

1. coverage radius；
2. active vehicle count；
3. cached adapter count；
4. is current RSU；
5. is predicted next RSU；
6. is predicted handoff target RSU；
7. predicted future load；
8. mean future load；
9. demand score of required adapter；
10. whether required adapter is cached。

每个 RSU 被投影到 64 维，随后产生 RSU set embedding、current RSU embedding、predicted next RSU embedding 和 handoff target RSU embedding。

### 5.3 Vehicle and prediction features

Vehicle branch 使用 10 维特征，包括主车辆位置、速度、当前 RSU 是否存在、当前节点输入 / 输出规模、依赖度、dwell time、prediction confidence 和 uncertainty。

Prediction branch 使用 13 维特征：

1. has predicted next RSU；
2. has predicted handoff target；
3. next-RSU sequence length；
4. dwell time；
5. mean future load；
6. workflow progress；
7. prediction confidence；
8. prediction uncertainty；
9. normalized handoff countdown；
10. boundary margin；
11. boundary urgency；
12. service pressure；
13. temporal urgency。

其中 temporal urgency 由 handoff countdown、车辆到 RSU 覆盖边界的距离和当前节点服务剩余步数组合得到：

```text
u_t = clamp(0.55 * u_countdown + 0.25 * u_boundary + 0.20 * u_service, 0, 1).
```

prepare-window score 使用以 preferred lead steps 为中心的 Gaussian timing score：

```text
g_t = exp(-0.5 * ((countdown_t - lead) / sigma)^2)
p_t = clamp(g_t * (0.45 + 0.55 * u_t), 0, 1).
```

prediction reliability 使用 confidence、uncertainty 和 temporal urgency 计算：

```text
rho_raw = confidence_t * (1 - uncertainty_t)
gate_t  = clamp(rho_raw * (0.7 + 0.3 * u_t), 0, 1)
```

主编码器将 prediction embedding 乘以 gate，避免低置信预测在早期训练或错误预测下完全主导决策；同时保留 `prediction_gate_min_leak`，避免 surrogate branch 被完全静默。

### 5.4 Fusion and head-specific contexts

SurrogateFusionEncoder 拼接 7 个 64 维 block：

```text
[graph, current_node, frontier, rsu_set, current_rsu, target_rsu, vehicle + prediction].
```

拼接后的 448 维向量经 `448 -> 128 -> 64` MLP 得到 shared embedding `z_t`。随后构造三个 head-specific context：

```text
z_slow  = tanh(W_slow  [z_t, target_rsu + prediction])
z_fast  = tanh(W_fast  [z_t, current_node + current_rsu])
z_event = tanh(W_event [z_t, target_rsu + prediction])
```

这个设计对应三类控制对象：slow head 更关注目标 RSU 和预测 cache 准备；fast head 更关注当前节点和当前 RSU 执行；event head 更关注 handoff target 和预测可靠性。

## 6. Hierarchical Three-Head Policy

SA-GHMAPPO 使用三个 actor heads：

| head | action space | 语义 |
|---|---:|---|
| slow | 3 | `no_cache_change`, `current_rsu_cache_fill`, `predictive_next_rsu_prefetch` |
| fast | 2 | `current_rsu_offload`, `vehicle_fallback` |
| event | 2 | `keep`, `handoff_prepare` |

在启用 hierarchical conditioning 时，fast head 的输入包含 slow action distribution，event head 的输入包含 slow 与 fast action distributions：

```text
pi_slow  = softmax(f_slow(z_slow))
pi_fast  = softmax(f_fast([z_fast, pi_slow]))
pi_event = softmax(f_event([z_event, pi_slow, pi_fast]) / tau_event)
```

其中 `tau_event` 是 event logit temperature，可随 update decay，用于让 event head 从较平滑探索逐渐转向更明确的 prepare / keep 判断。

三头输出通过优先级聚合映射到 `semantic_discrete_5`：

```text
if event == handoff_prepare:      action = handoff_migration_prepare
elif slow == predictive_prefetch: action = predictive_next_rsu_prefetch
elif slow == current_cache_fill:  action = current_rsu_cache_fill
elif fast == vehicle_fallback:    action = vehicle_fallback
else:                             action = current_rsu_steady_offload
```

该优先级体现了系统语义：handoff prepare 是事件级安全动作，优先级最高；cache prefetch 与 cache fill 是状态准备动作；vehicle fallback 是当前执行位置选择；steady offload 是默认稳态动作。

## 7. Centralized Critic and Reliability-Aware Value Context

主 critic 默认使用 centralized critic，即三个 actor heads 共享一个 value estimate。critic context 由 shared fusion embedding 和 graph/RSU set context 构成。可选 graph-continuity critic 会额外加入 22 维 continuity features，包括：

- remaining nodes ratio；
- frontier width；
- current / critical path length；
- predicted path switch ratio；
- future unique RSU ratio；
- predicted next / target differs；
- current / target adapter readiness；
- target future load gap；
- prediction confidence / uncertainty / reliability；
- conservative prepare pressure；
- low-reliability switch/load/migration pressure；
- temporal urgency；
- prepare-window score。

若启用 uncertainty-aware critic，则 reliability features 进一步通过 gate 调节 critic context，使 value function 显式感知低置信预测下的迁移风险。

## 8. PPO Training Objective

训练使用 PPO clipped objective。对 rollout 中每个 transition，trainer 提供 return、advantage、old log probability 和 action info。优势标准化后乘以 mechanism transition weight，使机制窗口在训练中获得更高权重。

平面形式的 PPO actor loss 为：

```text
L_actor = - E_t [ min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) ],
r_t = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t)).
```

层级策略中，每个 head 单独计算 ratio 与 clipped surrogate，并用 head credit 组合。event head 可使用 event-specific advantage，并通过 `event_advantage_blend` 与 base advantage 混合。总损失为：

```text
L = L_actor
    + c_v L_value
    - c_H H(pi)
    + c_aux L_aux
    + c_imit L_imit
    + c_mech L_mech
    - c_mech_H H_mech.
```

其中：

- `L_value = ||R_t - V(s_t)||^2`；
- `H(pi)` 是 policy entropy；
- `L_aux` 是机制目标辅助监督；
- `L_imit` 是可选 heuristic imitation warmup；
- `L_mech` 是 mechanism-guidance auxiliary loss；
- `H_mech` 鼓励机制相关 head 在关键窗口保留探索。

实现中还包含 gradient clipping、batch mini-epoch 更新、approx KL 统计和可选 KL early stop。

## 9. Mechanism Guidance and Auxiliary Targets

机制辅助目标来自语义状态和预测时机，而不是事后挑选结果。它关注三类情况：

1. 当前 adapter 缺失时，slow head 应倾向 current cache fill；
2. 预测 handoff target 有效且 prepare-window score / temporal urgency 足够时，event head 应倾向 handoff prepare；
3. 当前 RSU 与未来 RSU / handoff target 的 cache readiness 不一致时，slow head 应倾向 predictive prefetch。

论文中建议把它写成 mechanism-aware training signal，而不是新的环境 reward。它的作用是增加稀有 handoff / prefetch 机制窗口的梯度密度，缓解普通 PPO 在大量 idle / non-mechanism windows 中学不到 prepare 行为的问题。

## 10. Policy-Side Guards

主方法包含若干 policy-side guards。论文写作必须明确它们属于 feasibility / safety constraints：

- continuity guard：在 prediction reliability 与 handoff timing 满足条件时抑制不合适的 prefetch，并提升 prepare 倾向；
- cache warm-start guard：避免在 handoff countdown 超出 freshness window 时过早 prefetch，降低 expired miss；
- predictive prefetch admission guard：当 confidence 低、next-RSU 与 prefetch target 不对齐时，将 prefetch 延后为 handoff prepare；
- backhaul guard：限制重复 reactive fills，避免 backhaul cost 被无约束 cache fill 放大；
- latency fallback bias：在部分高延迟或低可靠执行窗口提升 vehicle fallback / steady execution 的可用性。

这些 guard 对顶刊审稿很敏感。推荐论文中用以下方式处理：

```text
We treat policy guards as a feasibility layer rather than as learned policy capacity.
Their contribution is reported separately through learned-core-only and guard-attribution ablations.
```

## 11. Algorithm Pseudocode

```text
Algorithm: SA-GHMAPPO

Input:
  semantic state stream s_t;
  action mask m_t;
  PPO hyperparameters eps, c_v, c_H;
  mechanism coefficients c_aux, c_mech.

For each training episode:
  Reset VEC workflow environment.
  For t = 1 ... T:
    1. Build semantic state:
       - DAG workflow state, current node, frontier;
       - RSU cache/load states;
       - vehicle state;
       - surrogate prediction confidence, uncertainty, handoff countdown.

    2. Encode state:
       h_DAG = DAGGraphEncoder(G_t)
       h_RSU = RSUStateEncoder(R_t)
       h_pred = PredictionProjection(P_t) * prediction_gate_t
       z_t = Fusion(h_DAG, h_RSU, h_vehicle, h_pred)

    3. Produce three policy distributions:
       pi_slow  = slow_head(z_slow)
       pi_fast  = fast_head(z_fast, pi_slow)
       pi_event = event_head(z_event, pi_slow, pi_fast)

    4. Apply action mask and sample / select semantic action:
       a_t = Aggregate(pi_slow, pi_fast, pi_event, m_t)

    5. Apply policy-side guards when enabled:
       a_t' = Guard(a_t, s_t, prediction reliability, freshness, backhaul)

    6. Execute action a_t' in environment.
       Store reward, value, log probability, head actions, system metrics.

  Compute returns and advantages.
  Annotate mechanism guidance targets and transition weights.

  For K PPO epochs:
    For each mini-batch:
      Recompute policy outputs.
      Compute head-level clipped PPO actor loss.
      Compute centralized value loss.
      Add auxiliary / mechanism / optional imitation losses.
      Update parameters with gradient clipping.

Output:
  trained SA-GHMAPPO checkpoint and audited training / benchmark manifests.
```

## 12. 可直接写进论文的方法段落

下面段落可作为论文 Method 的中文初稿，再翻译成英文。

> 为处理跨 RSU 移动下的连续 AI workflow 执行，本文提出 SA-GHMAPPO。该方法首先使用 DAG graph encoder 对 workflow 依赖、当前节点和 frontier 进行 message passing 编码；同时使用 RSU state encoder 表示当前 RSU、预测目标 RSU、cache readiness 和 future load。移动预测分支输入 next-RSU sequence、first handoff target、dwell time、confidence、uncertainty、handoff countdown 与 boundary urgency，并通过 reliability gate 调节其对融合表示的影响。融合后的状态被送入三个语义控制头：slow head 负责 adapter cache fill / prefetch，fast head 负责 current-RSU offload 与 vehicle fallback，event head 负责 handoff migration prepare。三个控制头共享 centralized critic，并通过层级条件依赖形成从 cache preparation 到 execution placement 再到 event preparation 的联合策略。

> 与平面离散 PPO 不同，SA-GHMAPPO 不直接在五个环境动作上学习单一策略分布，而是先在不同时间尺度的控制头上建模决策意图，再通过语义优先级聚合映射到稳定的 `semantic_discrete_5` action contract。该设计使 slow cache decision、fast execution decision 和 event handoff decision 可以分别获得与其机制相关的上下文，同时仍保持与所有 baseline 一致的环境动作接口。

> 为提升稀有 handoff 机制窗口中的学习信号，训练阶段引入 mechanism-aware auxiliary targets 和 transition reweighting。辅助目标仅由当前 semantic state、prediction reliability、adapter readiness 和 prepare-window score 构造，不读取 formal / hidden outcome。推理阶段的 confidence、freshness 和 backhaul guards 作为 feasibility layer 使用，用于避免低置信预测触发过早 prefetch 或重复 backhaul-heavy cache fill。论文中将 learned core 与 guards 分开报告，防止把 safety constraints 误写成纯学习收益。

## 13. 贡献写法建议

推荐贡献标题：

1. Mobility-driven continuous DAG workflow execution across RSUs；
2. Adapter warm-state lifecycle with predictive handoff preparation；
3. Surrogate-assisted DAG-RSU-mobility fusion with reliability gating；
4. Hierarchical three-timescale controller for cache, execution, and handoff-event decisions；
5. Strict trace-driven protocol with learned domain baselines and system trade-off reporting。

不推荐贡献标题：

- 首次使用 DAG 做 VEC offloading；
- 首次将 cache 与 offloading 联合；
- 首次使用 MARL / PPO 解决 mobility-aware VEC；
- 首次提出 digital twin predictor；
- full multi-agent RSU / vehicle collaborative learning。

## 14. 当前结果在方法报告中的放置方式

如果本报告被拆入论文，结果段应只保留边界化表述：

- strict-full v8 在 frozen NGSIM + Alibaba formal 和一次性 hidden 上，对全部 learned baselines 的 total reward BCa 95% CI 为正；
- 相对 DT handoff DRL 的 workflow continuity CI 为正；
- 相对 PPO 存在 handoff failure 与 backhaul cost trade-off；
- 相对 popularity heuristic 的 reward CI 跨 0；
- LuST 只有 4 个 independent outer windows，只能作为 low-power supporting evidence；
- v8-current robustness、scalability、prediction robustness 和 guard attribution 仍需补齐后才能升级 readiness。

## 15. 论文落地清单

写成正式论文前，需要把以下证据插入 Method / Experiment / Discussion：

- action contract 表：五个 semantic actions 与合法性条件；
- architecture 图：DAG encoder、RSU encoder、prediction branch、fusion、slow/fast/event heads、centralized critic；
- algorithm box：使用本文档第 11 节伪代码；
- ablation 表：no prediction、no hierarchy、no event head、no adapter prefetch、learned-core-only、guard attribution；
- system metrics 表：reward、continuity、handoff failure、backhaul、validated prefetch hit、realized prepare；
- limitation 段：controller-level 而非 full MARL、prediction layer 边界、guard 与 learned policy 分离、backhaul/failure trade-off。

