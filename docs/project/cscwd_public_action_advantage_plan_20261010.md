# CSCWD 公共动作估计与 event 优势监督：事前冻结计划

## 身份、阶段与停止边界

- `frozen_at`: 2026-10-10 Asia/Shanghai
- implementation base: `12ba1b9a27c5de67bd2d2662a29cc7eb3e851d56`
- prior evidence: `cscwd_state_condition_action_evidence_20261010_v1`，manifest SHA-256
  `978b94cc4446dcd1ede3c3db017d804d70afd0e28c9a7ba9a7a43568fa835b37`
- 本轮只允许一个候选：以公共因果动作估计产生 event supervision 的 `prepare / serve / abstain` 三态合同。
  不删 fast/slow/event loss，不改 reward、网络、auxiliary 系数、loss 分母、动作 mask、推理 guard 或方法权限。
- 阶段 1/2 可实现并测试，默认关闭。阶段 3 只有在 A 的 source-grounded 时间 profile 已固定 commit、公共字段齐全、
  B 独立通过合同测试且 source-grounded 正负状态门通过时才允许创建一次科学 run。任一条件不满足即停在训练前。
- 不读取 formal/holdout，不下载数据/模型，不上传 checkpoint/真实数据，不修改论文。

## 阶段 1：共同公共动作估计器

新增只读纯函数，唯一输入为 `semantic_state` 与公共 `action_mask`。它不得接收 env、instance、actual link、实际未来 route、
结果标签或 exact transition clone，不得修改输入 resident/prefix/workflow。每个 action 返回三态字段
`yes / no / unknown`、已知成本分量、unknown 原因和逐字段 provenance。

必需公共字段：

1. primary vehicle、current RSU、当前 DAG node、workflow nodes/order/completed、action mask；
2. current/predicted-target typed residents、容量/已用 bytes、required bundle、object catalog；
3. public causal predicted target/sequence/confidence/uncertainty；
4. prepared prefix current/target 的 known/exists/valid/missing fraction；
5. state/input bytes、compute seconds、estimated Mbps/fixed latency、start-current-RSU contact budget；
6. versioned time contract、`clock_seconds`、`deadline_seconds`、`remaining_deadline_seconds`、共同公开的
   `vehicle_fallback_seconds` 与 `failed_service_seconds`。

若 admission 需要未知 eviction/LRU、当前 handoff/recompute 无法由 prefix 确定、时间字段缺失或目标不存在，则相关成本/
比较必须为 `unknown`，不得用默认常数补齐。`action 2` 的零 model-load bytes 与 fallback/input 成本对所有方法相同。

同时提供两个非训练公共规则：

- `causal_public_immediate_rule`：只按同一估计器的当前服务、deadline fit、已知成本和未来 readiness gain 的固定词典序选动作；
- `causal_public_two_step_rule`：只投影第一步可由公开字段确定的 cache/prefix/workflow/clock 效果，再调用同一估计器；预测位置
  只来自 causal public sequence。任何第二步关键字段 unknown 时退化为 immediate score 并记录原因。

原 `TwoStepCostRule` 不改，继续标记 `privileged exact decision clone + future route preview + lexicographic objective`，不得与公共规则
合并或用新规则替换强参考。

阶段 1 测试门：future-suffix/actual-route 扰动不改变估计；输入深拷贝前后 hash 和 resident/prefix 完全不变；容量需 eviction
时为 unknown；公开字段缺失 fail-closed；五动作成本对称；公共规则不调用 `clone/clone_for_decision_model/step`；同输入重复输出一致。

## 阶段 2：唯一 event supervision 候选

候选 flag 默认关闭，并与旧 hard-zero/current-missing abstention 语义互斥。开启时：

- `prepare`：action 4 当前服务=`yes`、目标 bundle/state preparation=`yes`、产生公开 future-readiness gain、action 4
  `deadline_fit=yes`，且至少一个 current-service action 可行；event hard/soft target 都为 1，监督 weight=1。
- `serve`：action 4 当前服务=`no`，或 target preparation=`no`，或 action 4 `deadline_fit=no` 且某个 current-service action
  `deadline_fit=yes`；event hard/soft target 都为 0，监督 weight=1。
- `abstain`：其它全部情况，尤其任何决定性字段 unknown 或服务—资源 Pareto 未闭合；event CE 与 temporal-consistency 同时
  weight=0。slow/fast target、confidence、auxiliary coef 和原 minibatch denominator 不变。

这不是 action guard：PPO 仍可提高任何合法动作概率，raw inference 不读取标签。checkpoint 必须记录独立语义
`causal_public_prepare_advantage_v1` 并拒绝跨语义 load。

预注册合同状态：current missing→`serve`；current ready + target missing/prefix invalid + contact/deadline 全部充足→`prepare`；
prepare 超 contact 或超 deadline而 current service fit→`serve`；eviction/recompute/time 任一决定性未知→`abstain`。除 unit fixture 外，
A source-grounded profile 必须在不读结果的静态条件中至少产生一个 `prepare` 与一个 `serve` 状态，且人工字段审计一致；否则不训练。

## 阶段 3：条件性一次 pilot

预留 run ID：`cscwd_public_action_advantage_pilot_20261010_v1`。启动前另行提交 exact scientific commit、create-only protocol、
source-grounded profile commit/hash、公共输入 schema/hash、正负状态 gate receipt 与 `execution_authorized=true`。未满足时不得创建 run root。

若授权，固定：

- methods：原 event-abstention SA、candidate SA、PPO、controller-level MAPPO；seeds=`7/17/29/43/61`；全部从头训练；
- 每 cell `1,152 env.step = 24 updates × 48 transitions`；4 PPO epochs、minibatch 32，最多 192 optimizer steps；总训练
  `23,040 env.step`。checkpoint updates=`[6,12,18,24]`，fixed endpoint=`24`；不延长、不 retry；
- selection：固定 `dev_00/dev_01`，使用现有共同 `_selection_score`；selected 在全部 regression+frozen-check 20 实例评价；
  fixed endpoint 只在未参与 selection 的 `dev_02` 作单实例诊断；
- non-learned：公共 immediate、公共 two-step、privileged two-step 仅在预冻 6 实例
  `regression_00/regression_05/frozen_check_02/frozen_check_04/frozen_check_05/frozen_check_06` 各一次；
- evaluation 上限：selection 160 episodes + selected 400 + fixed endpoint 20 + rules 18 = `598 episodes`，最多
  `14,352 env.step`；全部是已暴露 development，不称 holdout，不把重复源窗口当独立 cluster；
- reward、环境、public observation、optimizer、网络、slow/fast target、aux coef/分母、checkpoint selection 对四 learned
  方法共同冻结。仅 candidate SA 的 event 三态 supervision 语义不同；PPO/MAPPO 不获得额外车辆成本或隐藏信息。

共同主指标：workflow completion、on-time completion、service failures；完整并列 elapsed、model/state/input bytes、recompute、
各 seed、selected 与 fixed endpoint。结果按支配/Pareto 报告，不要求 SA 第一，也不隐藏 completion 退化。

候选否定条件：completion 任一主要 split 下降；on-time/failure 改善不跨至少 3/5 seed且恶化超过 1 seed；prepare/serve 标签
在 source-grounded 时间合同下方向翻转；收益只来自额外信息/权限；或服务改善仅以未定 SLA 下的成本交换出现。否定后停止，
不改阈值、不追加 seed/预算、不切第二候选。
