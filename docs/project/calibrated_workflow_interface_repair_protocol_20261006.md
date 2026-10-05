# Calibrated workflow interface repair：冻结协议（2026-10-06）

## 审查身份

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-06`
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `calibrated_continuous_workflow_interface_repair_20261006_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `protocol_base_git_commit`: `03e4fddec29ad97ca1e16893d7abd89b335a48e7`
- `evidence_level`: `E0_PRE_REGISTERED_IMPLEMENTATION_REPAIR`
- `paper_ready_verdict`: `Unverifiable`

## 已确认修复边界

本轮只修复独立报告 `calibrated_workflow_interface_defect_report_20261006.md` 已确认的接口问题：

1. 新 `calibrated_workflow_interface_v2` 编码 profile 按 bytes 计算 cache occupancy，并让 SA、PPO、controller-MAPPO
   同样消费 current/target typed bundle readiness、base/adapter identity、bundle/state/input bytes、estimated link、fixed cost、
   contact budget 和 node compute time。旧 profile 不改。
2. 新 `independent_heads_executed_env_v2` 动作合同在确定性推理中执行 raw head argmax → aggregate → mask projection；
   训练仍使用五动作 marginal distribution，但 PPO likelihood 只对应实际执行动作。canonical inverse 只作审计标签，不再冒充
   raw head 决策。
3. 新环境 profile 的 mobility 按 decision step 推进；workflow node 仍只在 service success 后推进。action 4 仍只准备目标
   RSU，不增加“当前缺模型就替换 action 0”的规则。
4. 保留 v2 已实现的 staged state migration：当前节点失败时不得提交 prepared state 或 migration success。

奖励、五动作语义、workload templates 和 SA 的
`event_prepare_margin_boost=0.35`、`event_logit_sharpening_final_scale=2.3`、
`temporal_consistency_coef=0.35`、`auxiliary_coef=0.1` 保持不变。本轮不增加 SA 消融。

## 冻结预算与检查

- learned methods：SA-GHMAPPO、PPO、controller-level MAPPO；rule：原 two-step model-based planner。
- seeds：`7/17/29`；每个 learned method/seed 恰好 192 episodes；每 episode 最多 24 steps。
- 理论交互上限：`3 methods × 3 seeds × 192 × 24 = 41,472`；总 wall-clock 上限 1,800 秒。
- checkpoint candidates：48/96/144/192；只用 4 个 dev instances 的冻结字典序选模。
- `regression`：旧 evaluation 的 12 个已暴露 source windows，只检查已知故障。
- `frozen_check`：同一个事前生成 dev plan 中从未用于旧训练、诊断或 evaluation 的 indices 12–19；只冻结一次。
  它不是 formal/hidden holdout，因此最终只能称开发验证。
- 不调用 real model，不下载，不读取 formal/hidden/旧 holdout，不因排名追加 seed、episode 或重跑。

旧 checkpoint 的 tensor shape 可加载，但新编码语义、动作选择和 likelihood contract 已改变，不能作语义可比推理；所有三个
learned methods 必须在新 profile 重训。旧 v1/v2 checkpoint、结果、报告和 integrity 只读保留。

## 强规则权限标注

two-step rule 只在 decision clone 中读取 estimated link，不读 execution actual link；但它拥有显式 transition clone 和
lexicographic objective。这是强 model-based planner 能力，不等同于 learned actor 对同一字典的表示能力。该规则不削弱，比较结果必须
保留此 capability 差异。
