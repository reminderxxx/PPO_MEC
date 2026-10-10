# CSCWD 公共动作优势候选与停止裁决（2026-10-10）

## 审查身份

- `reviewed_at`: 2026-10-10 Asia/Shanghai
- `literature_cutoff`: 2026-10-10（本轮未检索新文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_raw_ngsim_event_time_20261010_v4`（仅 source-grounded development sensitivity，不是算法 run）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: B 实现 `5f88785ae4c2951e3826a468d3c17e389365f39c`；A 最终运行 `0704937742ec9218094f380a1b541c4bd2159bcb`
- `evidence_level`: `E1_DOCUMENTED` 对算法效果；公共估计器合同有局部可复现测试，但无本候选训练、checkpoint 或匹配评价
- `verdict`: `IMPLEMENTED_DEFAULT_OFF / PILOT_STOPPED_PRE_TRAINING`

## 2026-10-11 追加澄清：停止是执行资格不足，不是候选否证

后续逐消费者审查确认，三个 2.2 s raw 窗口对完整 workflow 本身不可达，且 fixed-action 探针的 30 个请求中有 10 个被
action mask 改写；因此本报告中“无 prepare/serve 即否定候选”与“下一步必须扩为较长窗口”的表述不再作为当前实施结论。
正确状态是 `UNTESTED / EXECUTION_QUALIFICATION_INSUFFICIENT`：短窗口不能验证，也不能否定学习候选；本轮仍禁止训练、扩窗、
调标签或 reward。历史 v4 原件与本报告原文保留，详细勘误和最小实现优先级见
`cscwd_public_estimator_executor_consistency_20261011.md`。

只读阶段表进一步确认：`dev_01` action1/action4 的 prepare `0.694262/0.724293 s` 均短于当时 `0.728088 s`
current contact，但加入当前 node 服务后整步超过 `2.2 s` raw trace 并被回滚。因此历史“0 成功 prepare”同时受 raw
整步原子准入语义约束，不能解释为候选没有生成可行准备动作。

## 结论

当前最值得补足的是 **event head 的 action-conditioned 信用分配**：现有 timing-only event auxiliary 并不比较 action 4
是否能同时完成当前服务、形成目标端 readiness gain、并在公开 deadline/contact 预算内完成。既有状态条件审查已给出
action 2 正反见证和 action 4 多步反例，因此本轮不做统一压低动作，也不删除 auxiliary；只冻结一个可证伪候选：用所有
方法可见的公共动作成本估计生成 `prepare / serve / abstain` 三态标签，`unknown` 时不给 event CE 与 temporal-margin 梯度。

该候选已经 **default-off** 接入 SA 的训练路径，但没有获得训练许可。A 的 source-grounded raw NGSIM profile 表明三个冻结
窗口每个只剩 2.2 s：15/15 raw episode 截断，0 workflow 完成；30 个决策中 10 次请求被公共 mask 改写，实际 action 4
仅执行 2 次且成功 prepare=0。prepare/serve 可达性门失败，故
按事前协议停止；不扩窗、不换实例、不训练、不据最终结果选候选。

## 公共估计器与权限边界

`src/agents/causal_public_action_estimator.py` 是无环境副作用的共同接口，只消费 public semantic state 与 action mask：当前/
目标 typed residents、容量、DAG prefix、对象/状态/输入 bytes、公开 link/contact/deadline 与 fallback/failure cost。缺字段、
需要私有 eviction 顺序或无法确定重算 predecessor 时返回 `unknown`，不访问 `instance.rsu_sequence`、实际未来位置/接触、
reward/outcome、environment clone 或 `step`。

A v4 的 public target/sequence 来自 observed-prefix constant-velocity forecast；prediction confidence 固定标注为未校准 `0.5`，
不再继承合成路由 profile 的 confidence，也不把该常数当物理准确率证据。

- `CausalPublicImmediateRule`：对共同公开估计做确定性一步选择。
- `CausalPublicTwoStepRule`：只在第一步公共投影可识别时做第二步；否则显式回退。原 exact-clone two-step 保留为 privileged
  model-based reference，不与公共规则混淆。
- `causal_public_prepare_advantage_v1`：action 4 当前服务失败、目标准备不可行，或准备超 deadline 而服务动作可按期时标
  `serve`；当前服务、目标准备、deadline 与未来 readiness gain 均可确认时标 `prepare`；其余 `abstain`。

候选只替换 event auxiliary 的 hard/soft target 与监督权重；slow/fast target、confidence、固定 batch 分母、auxiliary
系数、网络、reward、critic、PPO、selection 和 inference adjustment 均不变。PPO/MAPPO 构造不接收该 SA-only flag；checkpoint
写入显式 label schema 并拒绝跨语义加载。它不是 action mask、policy guard 或 SA 专属信息。

## 证据继承与新增门禁

| 证据块 | 范围 | 本轮可支持结论 |
|---|---|---|
| service-reward learning diagnosis | 两奖励 × SA/MAPPO/PPO × seeds 7/17/29 | critic/advantage/动作保留问题已定位到学习信号，但旧 optimizer/minibatch 不可精确恢复；不直接证明 auxiliary 根因 |
| state-conditioned action evidence | 11 states / 26 existing branches | strict avoidable/Pareto/unknown=`1/8/2`；支持 action-conditioned credit，否定统一禁用 action 2/4 |
| public estimator contracts | 5 个纯函数合同 | 未来字段与隐藏 trace endpoint 扰动不改变估计；未知 eviction 不伪造 infeasible；公共规则不 clone/step；成本字段守恒 |
| raw NGSIM event-time v4 | 3 exposed development instances × 5 fixed actions × 2 profiles | 30 episode/147 step；raw 为 15/15 truncated、2 nodes、0 workflow、28 failures、0 transfer bytes；10/30 请求被 mask 改写，实际 action1/action4 各执行 2 次，0 migration success |

A 原件：`source_manifest.json` SHA-256=
`c0b64fd64c602ef5351c5686d2f315bbd5784e63ebbd08902b297fc5bacc883f`，`summary.json` SHA-256=
`62c1f4c2c7fbb2e2763f6946c0fad29b4cd6a3705ebf01501a7d905d8b1fe30f`。原始 CSV/权重/checkpoint 不提交。早期
v1/v2/v3 在实现身份、执行动作计数或公共预测 provenance 收紧前生成，不作为本报告结果。

## 最小可复现见证

合成公共状态固定 current bundle ready、目标 bundle missing、100 Mbps link、10 s forecast contact、60 s deadline：action 4
公开估计为当前服务 `yes`、目标准备 `yes`、总时长 `2.81 s`、readiness gain=`target_bundle_and_state`，标签为
`prepare`。只把 remaining deadline 改为 `2.0 s` 时，action 3 的 `2.0 s` 可按期而 action 4 不可，标签变为 `serve`；删除
time contract 后为 `abstain`。改变 `actual_future_route`、future positions 或隐藏 trace endpoint 不改变结果。

这只是字段方向和梯度合同见证，不是性能结果。raw NGSIM 的真实开发见证相反地说明当前窗口没有一次成功 prepare，因而不能
用该数据进入匹配训练。

## 下一轮唯一变量与否证

本轮冻结而未执行的唯一算法变量是：control 使用现有 event supervision；candidate 仅启用
`mechanism_aux_causal_public_prepare_advantage_enabled=true`。其余训练/评价预算、seed、network、reward、optimizer、选模、
公共字段和规则权限完全相同。共同主要指标为 workflow completion、on-time completion、service failures；同时报告
elapsed、model/state/input bytes、recompute 和 `prepare/serve/abstain` 覆盖。

下一轮必须先用结果盲、与 formal/holdout 原始 frame/time interval 互斥的较长 development window，证明至少存在一个
source-grounded `prepare` 与一个 `serve` 标签以及成功 prepare 可达性。只改变窗口协议来完成资格检查，不能同时训练候选。
资格通过后才允许原计划 4 methods × 5 seeds × 1,152 environment steps/cell 的一次性 pilot。

以下原预注册项保留为历史设计；2026-10-11 澄清后，第 1 项只构成 execution qualification 不足，不再单独否定候选：

1. 较长 source-grounded development 状态仍不能同时形成 prepare/serve 标签或成功 prepare；
2. candidate 的 event 梯度方向与公共标签不一致，或 slow/fast 梯度、固定分母、网络/基线身份发生变化；
3. 同输入同预算下主要服务指标不改善，或改善只来自更高 bytes/recompute/elapsed 且无预声明 SLA 支持；
4. 信号必须读取真实未来、exact clone、私有 eviction 顺序或 SA 专属字段才能成立。

## 验证与未覆盖风险

- 项目 venv 执行 estimator、causal event auxiliary、既有 abstention/per-head/service-feasible 回归：`27 passed`。
- `python3 -m py_compile` 覆盖 estimator、SA core、构造入口和两份新测试：通过；`git diff --check`：通过。
- `scripts/smoke_test.py`：6/6 toy DAG nodes 完成，`terminated=True, truncated=False`。
- A 报告 21 项相关合同通过；B 独立核对 A commit、公共字段、未来扰动测试源码、v4 hash、执行动作计数
  与 30 行汇总。

未覆盖：候选没有训练/评价、source-grounded 标签覆盖没有逐状态账本、没有 formal/holdout、没有真实无线/跨界 forwarding，
也没有论文 novelty 证据。因此算法收益、SA 优势与 paper-ready 均为 `Unverifiable`。
