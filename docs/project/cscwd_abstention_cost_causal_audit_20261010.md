# 条件弃权后模型加载与时效退化：独立成本及反事实审计

## 审查身份、原件与边界

- `reviewed_at`: 2026-10-10 Asia/Shanghai；`literature_cutoff`: 2026-09-28（本轮没有文献 novelty 判断）；`target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: 候选 `cscwd_event_aux_abstention_ab_20261010_v1`，旧 v4 对照 `cscwd_causal_prepared_state_visibility_matched_20261010_v1`，B 原分析 `cscwd_event_aux_abstention_ab_20261010_v1_analysis_v2`，A 有界反事实 `cscwd_abstention_cost_causal_20261010_v2` 与摘要 `cscwd_abstention_cost_causal_supplement_20261010_v2`。候选执行 commit `abca7e047d672644f3e31d821831e23f80723a52`、审计算法实现 `46a68f11c289ccc304b88cdfed01c34ba5f61c3d`；旧对照科学 commit `f46ec72b15f534ac44768a83ef6316c1cfcb6b58`；A 反事实选择事前 commit `5102b4835794308efa6cb4c500e878c259276840`。本报告最终 Git commit 由提交记录给出。
- `evidence_level`: development 原件局部 `E2_ARTIFACT_AUDITED`，其中成本链做了确定性记录动作重放；正式 TMC-ready/canonical/独立比较仍为 `Unverifiable`。所有 36 个设计实例已暴露，本轮 0 训练、0 checkpoint 选择、0 formal/holdout/support、0 真实模型调用。未检索文献、未改论文。
- 候选科学 `run_manifest.json` 与 `artifact_integrity.json` SHA 分别为 `d9f256b6277c6b358b7eafd329f118f1f56884b67d94640bfcb8775d29fa880c`、`fa03b3fe9eed9af66f96e10e10d90db380b13a28893716071a395a8e4eb129cc`；40/40 文件 hash+size 一致。旧对照分别为 `48718e48dc55e6958c03b676634dca53e84fd21be251115d4809380a1441c615`、`b66dfb946d2fef8c8e93b3ffacd7f332c1280635c7c635ef1f58f7eb8c4e2ac1`；119/119 一致。B v2 分析 manifest SHA `a7ed11673ea9fe1c0e16d0c6b8e80b02cf4468f2fc22e05a14009b67a8ccbf12`，原 verdict `MIXED_STOPPED` 不改写。
- A 完整有界审计与补充分析 manifest SHA 分别为 `859abe71b5465cceeb20787a429ec932274177311d181a724f47e404cc35c4b7`、`ee49c70bff27b14e3360e92872dc957669f72722d8944573e21d5e4f79c15bc9`；两个输出根目录均在本 A worktree 的 `artifacts/analysis/` 下，摘要 JSON 逐项列出原件与结论。

事前计划 `cscwd_abstention_cost_causal_audit_plan_20261010.md` 已在任何新分支前提交、推送。其 12 个有序 strata 实际找到 11 个起点；fixed96 seed61 的“on-time 不变且 model bytes 增加”stratum 为空，依规则不补选。A 首次 runner v1 在第一个分支后的自身结果字段读取上报 `KeyError: clock_seconds_after`，只有部分诊断文件，没有 manifest 或科学训练；保留 v1 原状。修正审计脚本字段后另建 create-only v2，最终为 **11 起点、26 个去重合法分支、105 新反事实 env.step、79 次后缀前向**，均低于 36/864 上限。另有 **4,203 步记录动作重放**用于原始事件成本核算，不是新反事实。事实分支后缀动作与原候选 ledger 全部一致；每个起点两版旧前缀动作、RSU、节点、clock 与 checkpoint 动作对齐。v2 manifest逐文件绑定分支、step 成本与对象加载账本。

## 对称 episode 比较与完整时间账

下表为相同 `(view, split, seed, design_id)` 的 100 episode/视角平均；全部候选与旧 SA 的 workflow completion 均为 100/100，因此 elapsed 平均具有共同完成覆盖 100%，但同一实例跨 seed/view 不当独立窗口。单位 MB 为十进制；`model_prepare_mb` 是完整传输的模型部分。

| 视角 | on-time 旧→新 | failure attempts 旧→新 | elapsed s 旧→新 | model MB 旧→新 | total transfer MB 旧→新 | recompute s 旧→新 |
|---|---:|---:|---:|---:|---:|---:|
| selected | 38→45 | 50→0 | 75.499→73.922 | 129.700→300.948 | 130.926→302.097 | 27.256→25.061 |
| update96 | 43→44 | 39→4 | 73.456→75.814 | 176.557→289.254 | 177.728→290.397 | 25.915→27.697 |

fixed96 的 seed 17/29/61 on-time 分别 `9→8/12→9/8→7`（各 20 episode），seed7 `6→12`，seed43 `8→8`。selected 的 seed17 也 `9→8`，seed7 `6→12`、seed29 `7→9`；不能把 selected 总体改善写成五 seed 稳定优势。原分析的 `MIXED_STOPPED` 因此维持。

记录动作重放对候选/旧 SA/PPO 两个视角共 600 个 episode、4,203 步逐条核对实际 action、当前 RSU/bundle、服务成功、model/state/recompute、`clock_seconds_after`；`step_cost = fallback + node compute 或 failure + recompute + 三类 network + model load + state restore` 在每一步与 clock 相等，无重复计费。selected 新旧每 episode 的成本变化为：fallback `−0.640s`、失败 `−1.000s`、recompute `−2.195s`，network `+2.012s`、model load `+0.247s`，合成 elapsed `−1.577s`；fixed96 对应 `−0.240/−0.700/+1.782/+1.389/+0.129s`，合成 `+2.358s`。compute 均 `29.149s`，因同一节点集全部完成。传输上升既可换掉失败/重算，也可使 deadline 变差；无单一方向结论。

## 对象级加载账：新增字节主要被实际服务使用

每笔 committed admission 按 `object_catalog.transfer_bytes` 分解 base/adapter，并保存 RSU、resident before/after、victim、对象实际使用或被再次驱逐。每 episode `Σ对象字节 = Σtransition.model_transfer_bytes = evaluation.model_prepare_mb×10^6`；总计 `110,983,569,096` 字节守恒。类别按优先级互斥：`unused_preparation`、`eviction_reload`、`cross_rsu_placement`、`first_current_service_load`、`first_useful_preparation`/其余。`cross_rsu_placement` 只表示对象先在其它 RSU 出现过，**不自动等于浪费**；`used` 只说明此 episode 后续在该 RSU 的实际服务使用，不证明全局必要性。

| 平均 MB/episode | selected 旧→新 | fixed96 旧→新 |
|---|---:|---:|
| 模型总字节 | 129.700→300.948 | 176.557→289.254 |
| base 对象 | 111.653→253.756 | 142.104→243.606 |
| 跨 RSU 放置 | 104.285→257.469 | 155.019→257.469 |
| 驱逐后重载 | 11.694→29.566 | 17.871→17.871 |
| 未被后续服务使用的准备 | 10.440→10.247 | 0.193→10.247 |
| 加载后在该 RSU 实际使用 | 119.261→290.702 | 176.365→279.007 |

因此 selected 多出的 `171.248 MB/episode` 主要伴随跨 RSU base 放置 `+142.103 MB` 与后续实际使用字节 `+171.441 MB`；`unused_preparation` 没有随总增量上升。fixed96 多出 `112.697 MB`，base `+101.502 MB`，其中约 `10.054 MB` 是新增未用准备。例：fixed96 seed29 `regression_02` 候选在 rsu0 step1 载入 `base:family_a` 1015.026 MB 与 `adapter:alpr` 154.423 MB，驱逐 `helmet_distinct/family_b`；step4 在 rsu1 载入 `family_b` 1015.026 MB 与 `helmet_distinct` 9.642 MB，驱逐 `alpr/family_a`。四个对象均在该 RSU 后续实际服务使用；该 episode 的 2,194.117 MB 不能称为“无用预取”。旧策略该 episode 也有 2,194.117 MB，但集中在 rsu2 step6/7 的跨 RSU 放置及随后驱逐重载，展示路径位置改变而非凭总字节识别错误。

## PPO fallback 能力与费用对称性

同一 v4 对照的 PPO selected/update96 均 100/100 完成、0 服务失败，平均 model MB `113.452/99.924`，明显低于候选 `300.948/289.254`；PPO action2 fallback 为 `367/395` 步，候选为 `181/172` 步。记录动作重放中三臂合计 1,479 个 action2（PPO 762、旧 SA 364、候选 353）全都服务成功、模型传输 **0**，但每步付固定 `8s` fallback、当前 node compute 与 input 网络费用；PPO selected 每 episode fallback `29.360s`、input `1.460MB`、recompute `11.382s`，候选相应 `14.480s/1.014MB/25.061s`。PPO 以更多车端计算换掉 RSU model/cache load 和重算，成本口径未发现漏收固定 fallback、node compute 或 input transfer。

action2 对所有 learned method 同样合法；模拟器假定车辆已有所需计算能力且没有 vehicle model-load bytes。这是**共享动作合同中的能力/物理假设**，不构成 PPO 专属代码优势或已证实的计费 bug；论文若宣称完整端侧模型获取成本已计入则越界。不能为让 PPO 变差而单边增加 fallback 费用。

## 首次共同状态反事实：改善与恶化都保留

所有分支在相同完整 config/instance/RNG/cache/DAG/clock 前缀，首动作分别为候选事实、旧策略动作、合法 action2 fallback（重复合并）；后缀全部采用预定候选 checkpoint 的 raw deterministic policy。不同首动作之后状态不同，结果只支持**同一起点、同后缀策略规则**的有界因果比较，不把旧 checkpoint 的真实后缀混同进去。

- selected seed7 `regression_10` step4：剩余 deadline `2.987s`，候选 action2 花 `9.546s`、最终逾期；旧动作 action3 花 `1.523s`、当前节点和 workflow 完成且按期。两支都 0 model bytes/失败/recompute。这是明确**可避免的状态条件策略代价**，不是传输计费 bug；同 seed `regression_04` step4 恰相反，action2 避免 action3 后续 `33.086s` 重算、按期而 action3 逾期，否定“永远禁用 fallback”。
- fixed96 seed17 `regression_00` step2：候选 action2 首步 `9.544s` 服务成功但余程 `74.725s` 逾期；旧 action4 首步 `2s` 失败等待、无 state commit，候选策略后缀完成需 `31.615s`、按期但多 5 次失败。seed61 `regression_11` 同样 action2 后缀 `42.128s` 逾期、action4 后缀 `36.111s` 按期且 1 次失败。两例说明固定视角的 deadline 与可靠性是 Pareto 交换，且依赖每决策推进 RSU 的人工时间抽象；旧 hard-zero target 的 `MIXED` 反例仍有效。
- fixed96 seed29 `regression_02` step1：候选 action0、旧 action4、service 替代 action2 的候选后缀**全部逾期**（余程 `119.044/150.352/162.657s`）。旧 checkpoint 的原始 episode 按期，因此下降不能归因于第一次 action0 单独造成；旧/新后续 policy 与状态路径不同，归类 `Unknown`，不得挑此状态只报 action0 更快。
- selected seed29 `regression_00` step2：action0 后缀按期、`27.820s/318.489MB`，action2 后缀也按期、`33.875s/164.065MB`，action4 后缀逾期。额外加载换约 `6.055s`，是**必要性未定的速度—字节 Pareto 交换**，并非 action0 全部无用。其后续 rsu1 adapter 驱逐重载确实发生。

其余 7 个起点和全部 26 分支均在 `branch_summary_rows.csv`、逐分支 JSON；没有从结果中二次筛选起点。`decision_step_index` 使失败等待也推进 synthetic RSU，接触预算按人工 `5s/decision`，deadline 则按 modeled cost 累加；这与既有合同一致但不等价 NGSIM 原始逐帧时间。由此产生的“2s 失败等待换 RSU”效果不能外推真实车速或联络窗口。

## fast/slow 辅助监督与确定日志缺陷

候选训练配置 `auxiliary_coef=0.1`、slow/fast CE 权重 `1.0/0.5` 保留；event CE/temporal margin 只在 current-missing 上弃权。代码 `_build_mechanism_targets` 的 `fast_target` 在当前 RSU 存在且两个 fallback/steady bias 开关关闭时恒为 `0`，而 fast head `1` 映射 action2 fallback；因此对 confidence-eligible 样本，**fast CE 结构上偏向非 fallback**，无论该状态 action2 是否更快。候选 selected/update96 记录动作重放的 `685/689` 个状态 fast target 全为 0，其中 `655/659` 个满足原 confidence eligibility；slow target1 分别 `116/119` 个，而候选 action0 `124/97` 次中仅 `40/41` 次与 slow target1 同时出现。slow 标签没有被改标，也不足以单独解释所有新增 action0。11 个首分叉目标和原 head 动作见 `selected_target_rows.csv`，全评价计数见 `evaluation_target_counts.json`。

训练原件只记录执行动作的 `advantage_raw` 和**合计** weighted auxiliary grad norm，不记录每个 head 的 PPO/CE 梯度向量或其余弦；3840/3840 个 optimizer step 的合计辅助梯度非零，但不能据此证明 fast/slow CE 与 PPO 实际冲突。候选训练 action2 `4,281` 步、平均 raw advantage `+0.424`、正值比例 `57.6%`，action0 `6,515` 步、平均 `+0.195`；这些是不同状态与 rollout 的非配对原始 advantage，不能当作 fast 梯度因果证据。候选 optimizer event 计数 `eligible/supervised/abstained=110,072/79,796/30,276` 是跨 epoch/minibatch **重复暴露次数**，不是独立训练 transition 数。

**确定实现缺陷（日志，不是训练 loss）：** `scripts/run_calibrated_workflow_value_normalization_ab.py::_training_signal_row` 从 `semantic_state.current_rsu_id` 取 RSU，但 v4 公开状态的地址是 `vehicles[0].associated_rsu_id`；于是 `rsu_by_id(..., None)` 为 `{}`，`training_signal_rows.current_bundle_ready` 在候选 SA 28,800/28,800、旧 v4 四 learned 共 115,200/115,200 步全部错误地为 `False`。最小见证 `frozen_check_00` reset 的公开 bundle 实际 ready=`True`、该函数记录=`False`。agent 自身的 event abstention 从 primary vehicle 解析 current RSU，optimizer 的 supervised 计数非零；B v2 verdict 分析从 optimizer/评价 ledger 取数，未消费这个错误字段。因此目前影响是训练行为日志及任何未来按该字段分层的分析；**没有证据表明本轮 checkpoint、loss 或 B 的 `MIXED_STOPPED` 结论被改变**。应在独立实现任务中修生产者、补 ready/missing 回归并新增更正后的分析侧车，不覆盖旧科学原件。

## 分项结论与下一步

| 类型 | 结论 |
|---|---|
| 必要成本交换 | action0 在多个新 RSU 实际载入并使用 base/adapter，降低服务失败或重算；selected 改善以约 `+171MB/episode` 模型传输换约 `−1.58s`，fixed96 则出现反向时效。必要性只在具体受服务的分支成立，不是全局最优。 |
| 可避免策略代价 | selected seed7 `regression_10` 的 action2 相对合法 action3 多 `8.023s` 且越 deadline；反例 `regression_04` 中 action2 反而避免大量重算。需要状态条件决策，禁止硬编码总选 action3/0/2。 |
| 确定实现缺陷 | 训练信号记录 `current_bundle_ready` 读取错误地址；影响日志，未定位到训练/评价消费链。当前模型字节及 `step_cost` 对账未见重复入账或漏项。 |
| 未知 | fixed96 seed29 的实际按期退化不能被所选首分叉的候选后缀单独解释；fast/slow 与 PPO 的**实际梯度冲突**缺 per-head 同 minibatch 证据，不能删另一个 loss 或声称已定位学习根因。 |

唯一最小可证伪机制候选是**固定旧/新 checkpoint 与同一批 rollout，在不更新参数的条件下测 fast/slow CE 梯度与 executed-action PPO 梯度的逐 head 方向，并按公开 ready、预计 model bytes、deadline余量分层**。若 across seed 方向不稳定或分层无法解释上述正反分支，则否决“辅助监督导致过度加载”；即使方向一致，仍须另立事前训练协议和独立数据验证，不得直接调权重。顶刊 scorecard `N/S`：formal/holdout/support 与真实时间尺度缺失，现阶段论文贡献、优秀 baseline 稳定优势与 paper-ready 均 `Unverifiable`。
