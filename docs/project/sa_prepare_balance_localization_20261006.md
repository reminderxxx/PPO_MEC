# SA 当前执行—未来准备失衡：单因素定位与冻结候选（2026-10-06）

## 1. 身份与边界

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-06`
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `calibrated_workflow_prepare_balance_diagnosis_20261006_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `baseline_git_commit`: `5709b90b296df63283472dcb2e799cbe796284a9`
- `interface_profile`: `calibrated_workflow_interface_v2`
- `hierarchical_action_contract`: `independent_heads_executed_env_v2`
- `evidence_level`: `E2_ARTIFACT_AUDITED_NONFORMAL_DEVELOPMENT_DIAGNOSIS`
- `paper_ready_verdict`: `Unverifiable`

本报告只定位修复后 SA-GHMAPPO 的 prepare/execution 行为，不重新审计已通过的接口修复，不读取 formal、hidden 或旧
holdout，不调用真实模型，不改变数据、环境、reward 或网络。诊断使用原修复版 3 个 selected checkpoint 和 4 个 dev
实例；其状态与 logits 不能作为方法性能成绩。

原件：

- `artifacts/analysis/calibrated_workflow_prepare_balance_diagnosis_20261006_v1/diagnostic_states.json`
- `artifacts/analysis/calibrated_workflow_prepare_balance_diagnosis_20261006_v1/diagnostic_summary.json`
- `artifacts/analysis/calibrated_workflow_prepare_balance_diagnosis_20261006_v1/run_manifest.json`

## 2. 四项机制的实际作用路径

| 因素 | 作用位置 | 输入与激活 | 感知当前缺模型/当前节点可执行性 | 是否重复强化 prepare | 训练与确定性评估含义 |
|---|---|---|---|---|---|
| `event_prepare_margin_boost=0.35` | `_apply_event_logit_sharpening()` 中直接改 event logits | 有合法预测交接目标、timing support 达阈值、置信门通过时加 prepare margin | 否；不检查 current bundle readiness，也不试算当前服务 | 若实际启用，会在 sharpening 后再次推高 prepare | 本轮 rollout 与确定性评估均携带 `policy_evaluation_mode=raw_policy`，`_forward_policy()` 在进入 adjustment 前返回，因此实际均不生效；不是当前 action 4 的直接原因 |
| event temperature `1.4→0.85` | `分层策略网络.forward_single()` 内用 temperature 除 event logits | 按 update count 线性退火，8 updates 后固定 0.85 | 否 | 低于 1 时提高概率置信度，但正比例缩放不改变二分类 deterministic argmax | rollout、PPO 重算和确定性评估都使用同一 active temperature；会改变采样概率/梯度，不单独改变确定性 argmax |
| sharpening `final_scale=2.3` | `_apply_event_logit_sharpening()` 中围绕 event-logit 均值放大差值 | update schedule、timing support 与 scaling factor | 否 | 若实际启用，会放大已学 event 偏好，并与 margin 串联 | 与 margin 相同，本轮 raw-policy 路径实际跳过；12/12 诊断状态的 runtime `event_sharpening_info={}` |
| `temporal_consistency_coef=0.35` | `SAGHMAPPOBaseAgent._compute_auxiliary_loss()` 内 BCE 子项 | 对有正 confidence 的机制伪标签状态，以 `event_soft_target=prepare_window_score` 监督 event margin | 否；soft target 由 target readiness/timing 构造 | 是 auxiliary 内第二条 event 监督，但 soft target 不总是大于 0.5 | 只作用于训练 loss；确定性评估不直接调用，影响通过 checkpoint 参数保留 |
| `auxiliary_coef=0.1` | 总训练 loss 外层；内部含 slow/fast/event CE 和 temporal BCE | `_build_mechanism_targets()` 的预测目标、target bundle readiness、timing/confidence | event target 不检查 current readiness；`mechanism_aux_current_cache_fill_enabled=false` 时 slow target 也不补当前缺失 | 是；event hard CE 与 temporal BCE 可同时监督 event head，外层再乘 0.1 | 只作用于训练 loss；rollout/评估动作不直接加规则，但 checkpoint 学到该偏好 |

源码对账还确认：learn() 从每条 rollout 原样取回 `decision_info.run_metadata`，PPO 重算继续使用 `raw_policy`，所以本轮
executed-action PPO 的旧错位没有复发。配置了但被 raw-policy 跳过的 margin/sharpening 是 dormant enhancement / contract
语义问题，不是概率计算错误，也不能解释当前确定性 action 4。

## 3. 12 状态诊断

状态选择在 logits 检查前固定：按 `seed 7/17/29 → manifest dev instance 顺序 → step_index` 遍历，以
`current_bundle_ready × countdown<=2.5 × action-4 clone 可行性` 分层，每层最多 2 个；不足 12 时按原顺序补最早未选状态。
100 个候选状态最终选 12 个，覆盖 current missing/ready、near/far、target prepare feasible/infeasible。action-4 可行性由同状态
environment decision clone 试算，`contact_budget_exceeded` 记为不可行；不推进原环境。

关键结果：

- 12/12 状态实际运行 `raw_policy_evaluation=true`，12/12 没有 runtime sharpening info。
- 实际 raw-policy action 计数为 action 0/3/4=`1/4/7`。把 dormant sharpening+margin 反事实打开后为 `1/3/8`；只去掉
  margin 为 `1/4/7`。这说明它们若开启会进一步增加 prepare，但不是本轮已执行行为的来源。
- 12 个状态中 current missing 为 5 个，其中 4 个实际选 action 4，2 个 auxiliary hard event target 为 1。
- 对完整 100-state dev 轨迹的附加源码对账为：current missing 17 个，hard event target=1 有 12 个，实际 action 4 有 10 个；
  current missing 且 action 4 在 clone 中不可行有 2 个。ready 状态 83 个中 hard target=1 有 43 个、action 4 有 64 个。
- temporal soft target 在 current-missing 17 状态中只有 5 个大于 0.5；因此 hard event CE 比 temporal BCE 更直接地提供
  prepare 方向监督。只关闭 temporal consistency 不能移除主要冲突目标。

反例必须保留：两个 current-missing/infeasible 诊断状态的 hard event target 均为 0，checkpoint 仍选择 action 4；因此
auxiliary 不是所有错误动作的充分解释，网络泛化、PPO 信号或有限预算学习仍可能贡献。单因素重训只能判断该 factor 在本配置的作用。

## 4. 主假设与冻结候选

**主假设**：当前 auxiliary target 把 target readiness/timing 当作 event-prepare 监督，却没有把“当前节点能否先执行”纳入 event
target；在 current bundle missing 时，它与服务完成目标可能冲突。event hard CE 与 temporal BCE 又在同一 auxiliary loss 中重复
约束 event margin，使 checkpoint 更容易保留未来准备偏好。

**唯一候选 B**：将 SA 的 `auxiliary_coef` 从 `0.1` 中性化为 `0.0`。其余全部保持原修复版：

- `event_prepare_margin_boost=0.35`、temperature `1.4→0.85`、sharpening `2.3`、
  `temporal_consistency_coef=0.35` 的配置值不改；后两类 loss 因外层 auxiliary coefficient 为 0 不进入优化。
- 不加“当前缺模型强制 action 0”，不改 mask、环境、reward、数据、观测、网络或 checkpoint 选择。
- 候选定位为删除可能有害的训练增强，不作为新算法或原创机制。

配置 A 为历史原修复版 SA。其 commit、接口、3 seeds、192 episodes、实例、随机设置与本轮要求完全匹配，故复用
`calibrated_continuous_workflow_interface_repair_20261006_v1` 的 checkpoint、完整训练曲线和原始结果；本轮仅按新增行为字段
重放 selected checkpoint，不伪称重训。

配置 B 固定为 3 seeds（7/17/29）× 192 episodes × 每 episode 最多 24 steps；新增训练理论上限
`3×192×24=13,824` steps，checkpoint candidates 仍为 48/96/144/192，只用原 4 个 dev instances 和同一字典序选模。
禁止自动重试、追加 seed、延长预算或按结果改 candidate。PPO、controller-MAPPO、two-step rule 只复用接口、实例和评价身份
完全一致的既有结果。

## 5. 判定边界

候选必须同时检查 completion/deadline、current-missing action 4、连续无进展、可行/不可行 prepare、完成样本时间、传输、
重算和其他 strata。若 action 4 与恢复收益一并消失，只能判为保守执行退化；若无稳定变化，则保留负结果并转向重新设计 target，
不扩大本轮实验。
