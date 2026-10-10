# CSCWD 状态条件动作选择证据审查

## 审查身份

- `reviewed_at`: 2026-10-10 Asia/Shanghai
- `literature_cutoff`: 2026-10-10（本轮未检索新文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_state_condition_action_evidence_20261010_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `834e1ee6d72baff312a06681c7f6c1fc2b4cf7b9`
- `evidence_level`: `E3_REPRODUCED`，仅指本轮有界 development 诊断可由冻结原件逐 hash 重建；不是 formal、holdout 或算法效果证据
- `verdict`: `NO_TRAINING_CANDIDATE / DISTINGUISH_FIRST`

## 范围与完整性

本轮执行了事前冻结计划 `cscwd_state_condition_action_evidence_plan_20261010.md`（SHA-256
`5ad92451b29ba2bb5204e8cb6597ac2c6b34f4268caf914da81222f3b02aa936`）。输入是既有
`cscwd_abstention_cost_causal_20261010_v2` 的 11 个共同状态、26 条合法分支和 105 个既有 branch
`env.step`；source manifest SHA-256 为
`859abe71b5465cceeb20787a429ec932274177311d181a724f47e404cc35c4b7`。本轮只为恢复共同分叉点的公开字段，
确定性重放 19 个共同前缀 step；新增 action branch、训练 step、optimizer update、checkpoint 选择、formal 与
holdout 读取均为 0。

产物位于 `artifacts/analysis/cscwd_state_condition_action_evidence_20261010_v1/`；analysis manifest SHA-256
为 `978b94cc4446dcd1ede3c3db017d804d70afd0e28c9a7ba9a7a43568fa835b37`。独立临时目录重算的四个内容文件
SHA 与 manifest 完全一致。CSV 固定 LF，避免 Git checkout 的行尾规范化破坏 hash。

## 已确认事实

| 维度 | 既有分支结果 | 可支持的结论 |
|---|---:|---|
| 冻结共同状态 | 11 | 全部是已暴露 development 状态，不是 holdout |
| `strictly_dominated_avoidable` | 1/11 | 至少存在一个公开状态下可避免的候选动作 |
| `resource_pareto_exchange` | 8/11 | 大多数现有比较缺少由协议给定的单一 SLA 标量，不能据此宣布某动作全局错误 |
| `unknown` | 2/11 | 现有对称动作或首动作—后缀归因不足 |
| current bundle | ready 4，missing 7 | readiness 是有用条件，但单独不足以决定 action 0/2/4 |
| current prepared prefix | valid 1，missing 10 | prefix 信息稀疏，无法从本样本证明稳定学习规则 |
| action 0 | 5/5 首步服务；4/5 按期 | 当前加载可恢复服务，但 bytes/elapsed 仍可能形成交换，不能把加载字节一律叫浪费 |
| action 2 | 11/11 首步服务；5/11 按期 | fallback 不是全局好或坏；完成当前服务不保证按期完成 workflow |
| action 4 | 0/5 首步服务；3/5 最终按期 | 当前 bundle 缺失时均先失败，但部分后缀仍按期；不能仅凭首步失败全局屏蔽 action 4 |

action 4 的 5 条既有分支均处于 current bundle missing。它们累计 10 次 service failure，但 5/5 最终完成、
3/5 按期。该结果否定“action 4 首步失败即可直接等同整条 workflow 失败”，也不支持把准备行为的字节全部归为
无效预取。

## 最小正反见证

两个见证均来自 seed 7、selected、step 4，动作合法且只使用分叉前公开状态；它们给出相反的 action 2 方向：

1. `regression_10`：current bundle ready、current prefix valid、剩余 deadline `2.986819 s`。公开估计 action 2
   首步需 `9.557929 s`，不可能在剩余 deadline 内完成；action 3 相对 action 2 少 `8.023007 s`，两者都完成当前
   服务，但只有 action 3 按期。这里 action 2 被严格支配，是 1/11 的可避免错误。
2. `regression_04`：current bundle ready、current prefix missing、剩余 deadline `22.873627 s`。公开估计 action 2
   首步需 `12.650780 s`，可在 deadline 内完成；action 2 相对 action 3 少 `25.087902 s` elapsed 和
   `33.087902 s` recompute，且 action 2 按期、action 3 超期。这里 action 2 支配 action 3。

这对反例说明最值得继续定位的是**状态条件的动作信用分配**，尤其是 current readiness、prepared prefix、动作公开
成本和连续 deadline slack 如何共同影响 action 0/2/4；它不支持“统一压低 action 2/4”或“删除某个辅助 loss”。
同一公开状态 hash 还可在 selected/update96 的不同冻结后缀策略中重复出现而产生不同后缀结果，因此不能把完整后缀
回报直接包装成单步全局标签。

## 支持、否定与未定位项

- 支持：存在至少一个公开可辨的 deadline-slack 选错；服务退化可包含 state-conditioned action credit 问题。
- 否定：新增字节全是无用预取；action 2 或 action 4 应全局禁用；fast/event auxiliary 是已证根因；当前证据足以启动
  新 loss、guard 或网络结构训练。
- 未定位：当时 PPO minibatch、old policy、optimizer state 与 clip 后合成更新没有保存，不能精确复现历史更新；11 个
  状态中只有一对公开条件相近而动作方向相反的见证，且 8/11 是 Pareto 交换，尚不足以形成跨 seed/视角稳定的监督规则。
- 时间语义敏感性已独立核验：A 线一次执行 80 episode，真实 `env.step=528`，two-step 预览 step=`2,094`，合计
  `2,622<2,880`；legacy 的 24 个可对照 episode、144 transition 与原 selected ledger 零差异。本线复跑 5 项合同测试并
  核对四个原件 hash，均通过。旧 profile 有 3 次 2s 失败在 modeled clock 尚未跨 5s 边界时就切换 RSU；opt-in profile
  消除此现象，但固定策略方向混合：event SA on-time `6→2`，MAPPO `5→3` 且 failure `2→5`，old SA completion
  `8→7` 且 failure `3→19`，PPO on-time `4→5`。因此时间合同是实质混杂，不能据新 profile 选择算法。
- `synthetic_elapsed_5s_atomic_v1` 只能称合成敏感性，不是 NGSIM 原始物理时间；动作在起点 RSU 原子结算、下一决策才
  切换，且 22 次决策发生在完整合成序列时长之后并钳在末 RSU。two-step 的第二步 clone 还读取实际
  `instance.rsu_sequence` 的未来位置，只能作为更强 model-based/future-route-preview 参照，排除出同公共输入公平排名。
  A final commit=`ad483610ad98d70a019e108fdcc1a1fe66325084`；完整报告位于该提交的
  `docs/project/cscwd_mobility_elapsed_sensitivity_20261010.md`。

## 候选与停止决定

本轮**不冻结算法候选、不授权训练**。可能的单一研究假设是“公开 action-conditioned service/slack credit 能减少
deadline 前后的错误动作”，但它目前只被两个已暴露状态说明，尚未跨 source、seed 和视角稳定复现；把它立即实现为
auxiliary target 或 policy guard 会越过证据。

事前冻结的 mobility 时间 profile 区分已经完成，且证明结果对时间合同敏感，却没有证明哪一套语义物理正确。因此下一轮
**不应是算法 A/B**；唯一可接受的先行变量是 source-grounded mobility/contact 时间合同（逐决策真实接触时段、跨界服务、
目标 RSU 接触、轨迹末端），其余 checkpoint、策略、reward、输入、成本和样本保持固定。当前原件没有这些逐决策时刻，
所以本轮在此停止，不用合成 profile 反向挑算法。

只有时间合同闭合、且 source-disjoint 开发状态再次支持同一公开条件方向后，才允许一个算法 A/B：control 保持现有
event-abstention，candidate 只新增**公开 action-conditioned service/slack credit**，不改 reward、fast/slow/event 目标、网络
权限、动作或预算。共同主指标为 workflow completion、on-time completion、service failures，并完整报告 elapsed、
model/state/input bytes 与 recompute；SA、PPO、controller-level MAPPO 共享同字段与预算，two-step 另列能力权限。

以下任一结果会否定“现在值得补 action-conditioned credit”这一方向：

1. 公开 readiness/prefix/slack 条件在 source-disjoint 开发状态不再给出一致方向；
2. 时间合同改变后 good/bad action 2/4 的方向翻转；A 线混合敏感性已经触发这一停止条件，故当前候选未获授权；
3. 服务改善只以 bytes/recompute/elapsed 恶化交换，且协议仍没有给出外部 SLA；
4. 同输入同预算的 PPO、controller-level MAPPO 与 SA 不能共享该信号，或信号需要实际未来/exact clone。

若上述区分完成后仍稳定支持该假设，才另立一个 source-disjoint、同环境同公共输入、同训练预算与共同选模的匹配 A/B；
SA、PPO、controller-level MAPPO 同时参加，two-step rule 单独标注 estimated-model clone 和词典序目标权限。本轮没有
执行该训练，也没有修改论文或旧 `MIXED_STOPPED` 结论。

## 验证

- `python -m pytest -q tests/test_cscwd_state_conditioned_action_analysis.py`：`3 passed`。
- `python -m pytest -q tests/test_cscwd_state_conditioned_action_analysis.py tests/test_calibrated_continuous_workflow.py tests/test_calibrated_workflow_event_aux_abstention.py`：`21 passed`。
- `python scripts/smoke_test.py`：通过，6/6 toy DAG nodes 完成，`terminated=True, truncated=False`。
- 独立临时目录重算：11 states、26 branches、19 prefix replay steps；四个内容文件 hash 与 manifest 一致。

未覆盖风险：未重放历史 PPO 更新、未新增对称 action 全矩阵、未读 formal/holdout、未做真实物理接触时间验证，也未产生
算法性能证据。
