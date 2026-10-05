# Calibrated workflow pilot 接口与策略适配缺陷报告（2026-10-06）

## 审查元数与边界

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-05`（本轮不新增 novelty claim）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `calibrated_workflow_interface_diagnosis_20261006_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `b418eb4dbfa012e271cb97880693c714e6abf79e`
- `evidence_level`: `E1_DOCUMENTED_WITH_REPRODUCIBLE_NONFORMAL_DIAGNOSIS`；本报告有可重建的旧
  checkpoint 诊断原件，但不含 formal/holdout/support，不满足 policy-level `E2_ARTIFACT_AUDITED`
- `verdict`: `Unverifiable`（针对 paper-ready / 算法优势）；下述接口和环境缺陷已由代码与固定
  轨迹确认

本轮是修复前独立只读审查。诊断直接从固定 Git 提交加载历史环境源码，读取保留的
v1 SA-GHMAPPO checkpoint，不修改奖励，不训练，不下载，不调用真实模型，不读取旧
holdout。诊断包位于
`artifacts/analysis/calibrated_workflow_interface_diagnosis_20261006_v1/`；历史负结果
`artifacts/benchmarks/calibrated_continuous_workflow_pilot_20261006_v1/` 保持只读。

## 已证实的 action-4 失败链

固定实例是按旧 evaluation 行顺序找到的第一个 SA 失败样本：
`evaluation_00 / window_lankershim_off4000_len24_t1118936085200_1118936087500 /
j_1066618`，checkpoint 为 `sa_ghmappo_seed7.pt`。该旧 evaluation 已暴露，今后只能作为
回归集。

1. 前两个节点完成后，当前节点 `2` 需要 `base:family_a + adapter:alpr`；
   `rsu_0` 当前 bundle 缺失，`rsu_1` 目标 bundle 也缺失。
2. 此时五个动作都合法。action 4 只在 `rsu_1` 准备模型和状态，其 offload 语义仍是
   `current_rsu_steady_offload`；它不会自动为 `rsu_0` 填充当前 bundle。因此“当前模型缺失时选
   action 4”是合法但不明智，不应直接从 mask 中屏蔽。
3. 原始 head 在首个失败点并没有各自选 action 4：
   slow 概率为 `[0.326, 0.507, 0.167]`，偏向 `current_rsu_cache_fill`；fast 偏向 RSU
   offload；event 概率为 `[0.634, 0.366]`，也偏向 keep。但旧确定性路径先构造五动作分布，
   action 4 独占全部 `event=prepare` 质量，keep 质量则被 slow/fast 分到 action 0--3；最终概率
   `[0.321, 0.106, 0.030, 0.177, 0.366]` 令 action 4 成为单一最大项。这是已确认的动作聚合
   偏置：独立 head argmax 本应聚合为 action 0，但旧接口报告的 `raw_head_actions`
   已是 action 4 的 canonical 反解，不是原始 head 决策。
4. action 4 把 `adapter:alpr` 准备到 `rsu_1`，同时当前 `rsu_0` 服务失败。历史环境却仍将
   state 记为 `prepared`、将尚未完成的节点放入 target `completed_node_ids`，并增加
   `migration_successes`。这是明确的事务提交错误，使 action-4 success/failure 指标不可相信。
5. 失败后 `step_index` 和 `clock_seconds` 继续增加，但 `current_rsu_id`、vehicle 位置与预测序列
   均从 `node_index` 派生。因为节点未完成，`node_index` 不变，mobility 被冻结。从第一次目标
   bundle 到位后，step 3--9 的 actor logits、动作概率、动作 4 和服务失败完全相同，直到
   `max_steps` 截断。

该链条的可重建原始逐步记录在 `action4_failure_trace.json`。它说明失败不是单一原因，而是
“确定性聚合偏置 + 当前 bundle 缺失 + 迁移过早提交 + mobility 冻结”的闭环。

## 训练与评估差异

- 旧 SA 三个 seed 训练平均完成率为 `0.9375 / 0.9323 / 0.9167`，但这些是训练分布上的
  随机采样轨迹；旧评价是不同场景的确定性决策，不能把两者的差异全部归因于学习不足。
- 在同一固定失败实例和同一 checkpoint 上，事前固定的 16 个采样 seed 中 15 个完成，而确定性
  路径只完成 2/5 节点。这只是 checkpoint 诊断，不是新方法成绩；它证明采样/确定性差异本身
  就能解释大量 gap。
- 将 event temperature 单独设为 `1.0` 不会消除固定失败链。在该 pilot 中
  `run_metadata.policy_evaluation_mode=raw_policy`，因此 safety-projected 路径中的
  `event_logit_sharpening_final_scale` 和 `event_prepare_margin_boost` 并未在评估 forward 中生效；
  不能把这两个开关直接定性为此次失败的 bug。`temporal_consistency_coef=0.35` 与
  `auxiliary_coef=0.1` 会影响训练，但本轮证据不足以将 collapse 单独归因于它们。
- 层级策略实际从 masked 5-action distribution 采样/取 argmax，但默认 PPO actor loss 使用
  canonical head 的加权 log-prob，`env_action_ppo_enabled=false`。固定失败点的 head log-prob 为
  `-2.279641`，真正执行动作的 log-prob 为 `-1.003798`；循环状态中两者分别为
  `-2.069790` 和 `-0.766470`。优化动作与执行动作不同口径是已确认的训练接口缺陷。

## 环境字段和编码器适配

producer 已在 `semantic_state` 中提供 `cache_capacity`、`cache_used_bytes`、
`typed_resident_object_ids` 及 `calibrated_context`，但旧 encoder 不是对等消费：

- flat centralized encoder 用 `len(cached_adapter_ids) / cache_capacity` 计算占用率。新环境的
  `cache_capacity` 单位是 byte，分子却是 adapter 个数；扰动 capacity 产生的特征差仅为
  `1.43e-10`，实质上把容量信号压到 0。
- flat actor 完全不消费 cache ready/occupancy；去掉当前或目标 adapter 对 actor 特征的差为 0。
  centralized critic 只有上述错单位的微小变化。
- graph encoder 能区分当前/目标 adapter ID 是否出现，但不消费 typed base dependency、
  `cache_used_bytes`、byte capacity 或对象大小。保持 adapter ID 不变、只替换 resident base family 时，
  graph/flat 特征差都为 0，但环境 `_bundle_ready` 结果可变。
- 单独改变 `calibrated_context.link`、`calibrated_context.state_bytes` 或 current node 的
  `required_base_model`，graph/flat 特征差均为 0。因此字节模型大小、状态量、链路估计、
  typed resident identity 并未真正进入策略。
- agent `act()` 和 `evaluate_value()` 显式丢弃数组 `observation`，只使用
  `info['semantic_state']`。这本身是已声明的 semantic-state contract，不是单独 bug；但意味着数组
  observation 里已有的 byte occupancy/current-ready 不会补救 semantic encoder 的信息丢失。
- DAG 的结构、当前 node 与 adapter-level 当前/目标准备性对 graph encoder 有影响；不能笼统表述为
  “所有新信息都未编码”。精确丢失的是 byte/typed dependency/cost/contact 信号和 flat actor cache 信号。

特征扰动原件为 `feature_perturbation_audit.json`。由于修复后特征语义改变，旧 checkpoint 即使张量
shape 仍可加载，也不具备语义可比性；所有学习方法必须在同一新 profile 下重训。

## 强两步规则的权限与目标

v1 没有分离 actual/estimated link，因此不存在“用未公开 actual Mbps 击败策略”的已证实泄漏。
但规则与学习策略的信息使用能力并不等价：

- 它 clone 整个 environment，对每个候选动作直接执行精确转移，再读取 completed node、
  service failure、deadline、elapsed 和 transfer 的候选后果。这是显式模型规划能力；策略只收到
  semantic state，且旧 encoder 丢失了部分公开字段。
- 规则优化的是 lexicographic tuple：先最大化节点完成，再最小化 failure/deadline/elapsed/
  transfer；RL 优化的是标量 reward。两者相关但不同一。
- 该规则可保留为 strong model-based baseline，不应被削弱；但历史报告中“只使用同一 info
  字典，因此信息能力同等”的表述需要收缩。后续规则 preview 必须使用 decision-time estimated
  link/prediction，不得读取真实执行 link。

## 缺陷、未证实原因与历史影响

### 已确认缺陷

1. 确定性 multi-head 决策的报告/聚合语义错位，event 质量对 action 4 有结构性偏置。
2. 层级 PPO 默认优化 canonical head log-prob，而执行动作从 env-action distribution 产生。
3. action 4 在当前服务失败后仍提交尚未完成节点的 prepared state 和 success 计数。
4. mobility/contact 派生于 `node_index`，失败时虽有时间成本但不发生移动。
5. flat cache occupancy 单位错误；graph/flat 均未消费 typed dependency、byte occupancy、链路、
   状态量和模型大小，flat actor 还未消费当前/目标 cache readiness。
6. strong rule 具有精确 clone transition 和不同目标，旧“同等信息使用能力”标注不准确。

### 尚待验证

- 修复接口并重训后，SA 是否仍有独立的 event 过激活；本轮证据不支持预先关闭所有增强。
- 在同场景、同一 executed-action PPO contract 下，剩余 gap 有多少来自训练预算/学习不足，
  有多少来自方法设计。
- 强规则在不确定预测与未知 dynamics 下是否仍足够；旧 pilot 只能支持小规模显式模型规划
  在该设计上很强。

### 对历史结果与公平性的影响

- v1 的 SA `0.528` 完成率和 action-4 负结果仍是历史实现的真实记录，不得覆盖或改写。
- 该数值不能再用来判定 SA 方法本身的上限；环境事务错误、mobility 冻结、encoder 丢失和
  actor/executed-action log-prob 错配是重大混杂。
- PPO/MAPPO 的旧结果也不是本轮修复后的公平对照；公共 encoder profile 修复和 hierarchical
  action contract 修复必须同时适用于它们。
- two-step rule 的历史强结果仍可保留，但必须标注为具有显式 transition model 与 lexicographic
  objective 的 strong model-based baseline，不得以“同一 info 字典”宣称能力同等。

## 授权的最小修复边界

后续实现应只处理上述已确认问题：新建显式 calibrated encoder/environment profile，将 byte/typed
readiness/cost/contact 信号同口径提供给全部学习方法；使确定性 head 决策、聚合动作、执行动作
与 PPO log-prob 可对账；将迁移状态提交延迟到当前节点成功之后；令 mobility 随 decision step
而非 node completion 推进；规则 preview 只用 decision-time estimate。不修改奖励，不添加
“当前缺模型就强制 action 0”的 SA 启发式，不削弱 strong rule。
