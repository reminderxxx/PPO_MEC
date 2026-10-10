# CSCWD 状态条件动作选择证据：事前冻结计划

## 身份、边界与预算

- `frozen_at`: 2026-10-10 Asia/Shanghai
- source run：`cscwd_event_aux_abstention_ab_20261010_v1` 与旧 v4 `cscwd_causal_prepared_state_visibility_matched_20261010_v1`
- source branch artifact：`cscwd_abstention_cost_causal_20261010_v2`，manifest SHA-256 `859abe71b5465cceeb20787a429ec932274177311d181a724f47e404cc35c4b7`
- 固定输入：11 个事前选定共同起点、26 个既有合法分支、105 个既有 branch env.step。不得按算法排名、reward、梯度或本轮结果增删起点。
- 本轮首先只消费既有分支。允许为恢复分叉点公开字段而确定性重放 11 个共同前缀；这不是新 action branch，必须记录 replay step 数和逐状态 hash。若字段仍不足，最多另立 12 个预先冻结共同状态、36 个合法分支、864 env.step 的 create-only 补充；未另立计划前不得执行。
- 训练、optimizer update、checkpoint 选择、formal、holdout、模型生成与下载均为 0。不修改 reward、action、数据、科学算法、旧原件或论文。

## 公开在线字段

每个共同分叉状态只读取 policy 当时可见或可由这些字段确定计算的内容：

1. primary vehicle 的 current RSU、当前 DAG node、合法 action mask；
2. current bundle `ready/missing/unknown`，以及 required bundle 在 current RSU 的缺失 object、`resident_bytes` 与 `transfer_bytes`；
3. predicted target RSU、目标 bundle ready/missing 和目标缺失 object/bytes；
4. `state_bytes`、`input_bytes`、node `compute_seconds`；
5. 公开 estimated link Mbps/fixed latency、公开 contact budget；
6. `deadline_seconds - clock_seconds` 的连续剩余量；不从结果选择新的 SLA 阈值；
7. current/target prepared prefix 的 `known/exists/valid/missing_completed_fraction`；
8. causal prediction 的目标、confidence、uncertainty 与公开序列 provenance。

不得读取 actual link、实际未来 RSU、完整后缀结果或 exact environment clone 作为 proposed learned policy 的输入。缺失 bytes 只按公开 typed resident/object catalog 计算；`cross_rsu_placement` 不等于无用预取。

## 预声明条件与比较

- readiness 只分 `ready/missing/unknown`。
- bytes 只分 `zero/nonzero/unknown`，同时保留连续值；不使用 100 MB 等事后阈值。
- deadline 只保留连续剩余量，并比较各 action 的首步成本是否超过该状态自己的剩余 deadline；不另造固定秒数 SLA。
- contact 只比较公开 estimated prepare seconds 是否不大于公开 contact budget；这是环境既有权限，不按结果选阈值。
- prepared prefix 只分 `valid / exists_but_stale / missing / unknown`。
- action0、action2、action4 分别报告首步 current service、model/state/input bytes、首步成本、target preparation/state commit，以及冻结候选后缀的 completion、on-time、failures、elapsed、model/state/input bytes、recompute。缺失合法 action 保留为 `not_observed`，不补齐全矩阵。

同一起点内的判定固定为：

- `strictly_dominated_avoidable`：另一合法分支在 completion、on-time、service failures、elapsed、model/state/input bytes、recompute 全部不差且至少一项严格更好。
- `current_service_recovery_tradeoff`：action0/2 修复另一动作的首步 current-service failure，但全后缀资源或时效向量不构成严格支配。
- `future_prepare_tradeoff`：action4 牺牲首步 current service，却在全后缀 service 向量上改善；同时完整报告资源变化。
- `resource_pareto_exchange`：服务与资源/时效方向互换，任一分支均不支配另一分支。
- `unknown`：缺少对称 action、后缀路径变化不能归因于首动作、跨时间边界语义未闭合，或上述证据不一致。

“action0 字节对当前 RSU 服务是机制必需”只表示该动作加载缺失 bundle 后才能完成当前服务，不表示它相对 action2 或长期策略全局必要。action2 的车辆算力与零 model-load bytes 是所有 learned method 共用的环境假设；本轮不得只给 PPO 加成本。

## 候选产生与停止门

只有当一个完全由上述公开字段定义的条件，在 selected/update96、多个 seed 和现有正反分支中给出一致、可解释的动作方向，才允许提出**一个**未来学习改进假设。候选必须写成“可观测条件 → 信用/结构机制 → 预期动作概率 → 服务及成本否定条件”，且与 event abstention 组合时只新增一个变量。

出现任一情况即停止并交付“无训练候选”：跨 seed/视角方向混合；只有 exact clone/实际未来才能区分；条件只能解释单个已暴露状态；或服务改善与 bytes/recompute/elapsed 的 Pareto 交换缺少外部 SLA。不得继续扫描阈值、堆叠 loss/guard 或搜索到 SA 获胜。

若未来另立匹配训练，SA、PPO、controller-level MAPPO 必须使用同一环境与公共输入、共同预算/选模，并同时报告 selected 与固定终点；强 two-step rule 另列其 estimated-model clone 和词典序目标权限。全部旧 development 状态已暴露，不得称 holdout。

## A elapsed-time 敏感性独立验收门

A 的 opt-in 时间实验必须另行满足：默认旧 profile 完全不变；elapsed 驱动的 mobility/contact 规则在 action 开始前冻结；跨 RSU 边界时 current service、target preparation、费用与 prepared-state commit 的归属明确；不能在单个 action 内无记录地切换 RSU/link；SA/PPO/MAPPO 共享相同成本与信息，规则方法的 clone 权限另列；所有运行仍是 development sensitivity，不改写旧科学结果。B 不重复 A 的矩阵，只验收其计划、原件身份、时间轴与公平性。
