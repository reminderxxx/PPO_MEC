# CRDCM 最小决策合同与真实小样本诊断

## 审查元数据

- `reviewed_at`: `2026-09-28`
- `literature_cutoff`: `2026-09-28`（本实现轮未新增网页检索）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `crdcm_decision_diagnostic_v1_20260928`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `base_git_commit`: `19200c5e9fec1dddd9375834ada3ac1798bf119e`
- `implementation_git_commit`: `47f10523d6da036dcd14c91fed5d6201903aa457`
- `evidence_level`: `E2_ARTIFACT_AUDITED`，仅限非正式 decision-contract diagnostic；不等于 formal performance evidence
- `verdict`: `UNVERIFIED for algorithm advantage / PASS for minimum decision-contract plumbing`

## 结论

`crdcm_decision_v1` 已把当前请求的 typed base/adapter bundle、RSU 实时容量、已提交 DAG 的剩余关键路径与 adapter 复用、因果 handoff 预测、workflow-state readiness/risk 送到决策层，并完成 policy action 到实际执行、reward/mechanism event 和 PPO credit provenance 的逐步 trace。新实现与旧 SA-GHMAPPO/MAPPO/PPO checkpoint 严格分离，旧 checkpoint 加载到 CRDCM agent 会 fail-fast；历史七个受保护用户文件、旧模型和旧 run 不被覆盖。

这仍是最小版本：动作继续使用现有五动作合同。CRDCM 可以决定“当前 RSU 补齐、预测目标预取、vehicle fallback、当前 RSU 稳态、handoff prepare”，但不能直接选择任意 future critical node 的 object ID。若要升级为对象级 bundle 选择或多候选 RSU 联合分配，需要新 action schema、执行事务和 credit derivation，属于后续独立重大设计，不在本轮隐式扩展。

## 冻结状态合同

合同为 `crdcm_observation_v1`，仅在 mechanism profile 显式设置 `crdcm_observation_enabled=true` 时生产；旧 profile 的 state shape 不变。Actor 看不到 oracle future truth、service outcome、reward 或 outcome label。

| 状态组 | 单位 | 来源 | 更新时间 |
|---|---|---|---|
| 当前请求 base/adapter bundle | object ID、MB | 当前 workflow node + typed catalog + 当前 resident set | 每步 action 前 |
| 当前/预测目标容量 | MB、占用率 `[0,1]` | RSU runtime typed residency | 每步 action 前 |
| 剩余 DAG / critical path / reuse | node count、ratio | 已提交完整 DAG + 当前 completed node set | 每步 action 前 |
| handoff target/confidence/uncertainty/ETA | RSU ID、概率、mobility step | causal predictor snapshot | predictor 完成后、action 前 |
| migration readiness/risk | bool、`[0,1]` | workflow-state runtime residency + 当前 profile | 每步 action 前 |

固定 feature vector 共 28 维，字段名与顺序写入 observation；encoder 对版本、长度、范围和 no-oracle 声明 fail-fast。

## Producer 到执行链

1. `VecWorkflowCoreEnv` 在 pre-action phase 生产结构化 CRDCM observation。
2. `CRDCMPolicyMixin` 将同一 28 维向量输入新的 trainable residual；SA-GHMAPPO、MAPPO、PPO 都使用完全相同的 CRDCM 信息。
3. 结构消融只保留既有差异：SA-GHMAPPO 为 graph/surrogate + hierarchical heads，MAPPO 为 flat semantic + three controller heads，PPO 为 flat semantic + single head。
4. 现有 action aggregation/mask 和五动作执行器保持不变；强启发式也只读同一 CRDCM observation 和 mask。
5. Trainer 的 `policy_decision_trace_v2` 记录 pre/post CRDCM logits/probabilities、mask、raw/aggregated/pre/post override/executed action、log-prob、credit provenance、reward components、metrics protocol 和 CacheEvent。

## Credit 合同

`mask_external_override_actor_credit_v1` 规定：只有最终执行动作等于聚合 policy action 时，样本才可进入 CRDCM PPO update。Guard、planner 或 trainer 造成动作偏离时，`actor_credit_weight=0`、`actor_credit_source=external_override_masked`；不再把强制动作静默伪装成 raw policy sample。为避免修改受保护旧核心，CRDCM agent 在新 wrapper 的 `learn()` 中剔除这些样本；该行为只适用于新版本。

本轮未训练，因此没有 optimizer 性能结论。新 checkpoint 包含独立 CRDCM residual state 和版本字段；legacy/non-CRDCM checkpoint 负向测试确认拒绝加载，新 checkpoint save/load round-trip 通过。

## 强启发式与负对照

`crdcm_critical_path_heuristic` 的固定规则为：当前 bundle 缺失时先补齐；有可靠且临近的跨 RSU target、目标 bundle 不 ready、容量不冲突且 critical-path/reuse 压力足够时执行预取或 state prepare；低复用、无 target、容量冲突、目标已 ready 或充足资源稳态时不强行触发机制。规则没有读取 outcome，也没有针对输出回调阈值。

单元测试覆盖容量竞争、共享 adapter 复用、剩余 DAG 进度、跨 RSU state prepare、低复用、无 target/无迁移和 ample-resource negative control。

## 真实小 rollout

冻结配置：`configs/experiment/crdcm_decision_diagnostic_v1.yaml`。矩阵为 4 controllers × 4 scenarios × 3 固定 observed NGSIM/Alibaba units = 48 episodes，每 episode 最多 20 steps；实际执行 496 steps，低于 960-step 上限。没有训练、checkpoint 选择或 holdout 访问。

- 全部 124 个 matched decision cell 都含四个 controller。
- 107/124 cell 存在 action disagreement，率为 `0.862903`。
- SA-GHMAPPO 与 MAPPO 为 `0/124` disagreement；它们在 fresh/untrained、同 seed 的本轮诊断中仍表现为相同动作，不能声称图结构已经带来行为增益。
- SA/MAPPO 对 PPO disagreement 均为 `86/124 = 0.693548`；PPO 对强启发式为 `62/124 = 0.5`。
- 48 个 episode 的 override count 为 0，说明本次实际 trace 的 executed action 都能直接回溯到聚合 policy/heuristic action；外部 override 的 credit mask 由独立负向测试覆盖。
- completion、continuity、failure category、transfer MB/request、migration cost、conditional delay coverage 和 override rate 的完整 16-row controller×scenario 表保存在 artifact；这些 fresh/untrained 数值不能用于算法排序。

可解释的机制信号包括：capacity/reuse 场景中 SA/MAPPO 出现 action 4，PPO 主要选择 current fill，强启发式在 readiness/pressure 不满足时回到 steady action；no-migration 与 ample-resource 场景的 transfer/migration/failure 变化均被记录，但不能视作因果性能结论。

## Artifact 与复现边界

Artifact root：`artifacts/analysis/crdcm_decision_diagnostic_v1_20260928/`。

关键文件：`completion_receipt.json`、`command_log.json`、`artifact_integrity_manifest.json`、`protected_user_files_audit.json`、`diagnostic_rows.csv`、`diagnostic_aggregate.csv`、`decision_divergence.json`、`pairwise_action_disagreement.json`、`plan_snapshot.json` 和 48 个本地 episode summary。

本轮结论只允许表述为“新状态到达决策、动作/credit 可追溯、强控制可运行、真实小样本机制行为可解释”。不得表述为 CRDCM 优于既有算法、具备泛化、收敛或 paper-ready 性能。

## 未来公平训练计划（冻结但未授权）

- 只创建新 CRDCM checkpoint，不适配旧 checkpoint。
- SA/MAPPO/PPO/强启发式共享 observation、mask、执行机制、场景、seed 和请求 exposure；结构差异单独标记。
- 至少 5 seeds；先 development，再由独立批准的新 disjoint formal windows 评估。
- 报告完整 metric vector、window-outer uncertainty、失败类别和 cost/coverage，不使用 post-hoc 单一加权分数。
- 用户提到的未来 960-episode performance matrix 仍为 `frozen_not_authorized`，必须另行批准。
