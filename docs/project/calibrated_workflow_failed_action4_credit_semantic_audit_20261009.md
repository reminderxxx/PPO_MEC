# Calibrated workflow failed-action-4 credit 语义反例审查

## 身份与结论

- `reviewed_at`: `2026-10-09`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-09`；本轮未检索新文献
- `target_venue`: `CSCWD 2027`；同时保留 TMC claim 边界
- `artifact_run_id`: `calibrated_workflow_failed_action4_credit_audit_20261009_v1`
- `source_run_id`: `calibrated_workflow_value_normalization_ab_20261009_v2`
- `source_scientific_commit`: `858bc797e23b4e56051663f28d6fd7681f5ee77d`
- `audit_code_commit`: `d0795da929322d353af76805b2350b7eefdf796b`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `evidence_level`: `E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY_SEMANTIC_REPLAY`
- `verdict`: `CAP_PREDICATE_INSUFFICIENT_NOT_READY`

撤销 `6006ed2...` 中“失败、无节点进展、未提交 migration 的 action 4 不应获得正局部信用”的强断言。
拟议 predicate 同时覆盖有后续 cache 价值、无 resident 变化和带驱逐外部代价的 transition，不能统一执行
`advantage=min(A_t,0)`。原方案保留为历史 `NOT_READY`，不实现、不训练，也不自动替换为新 predicate。

## 源码语义

科学执行 commit 与当前环境文件 SHA-256 均为
`d789e6088e6a6a859de2fc720f78c94410179d2cc6a5a57a9751a9b89fed68bc`。action 4 的顺序是：

1. 先对预测 target 调用 `_admit_bundle()`；准入可以立即改变 target residents/LRU，并发生 victim eviction。
2. 若模型与 state package 能在 contact budget 内准备，则记录 `staged_migration`。
3. 随后检查当前 RSU 是否具备当前节点 bundle；缺失时当前服务失败。
4. 只有当前服务成功，`staged_migration` 才写入 `prepared_state` 并令 `migration_success=true`。

因此 `migration_success=false` 只说明 workflow state migration 未提交，**不说明先前 target model-cache 准入已回滚**。
contact budget 失败时会原子回滚，但当前 bundle 缺失导致的 service failure 不会自动回滚已经可行的 target admission。

## 既有行为账本全量分类

固定 predicate：`executed_action==4 && !service_completed && !progressed && !migration_success`。只读取既有
`behavior_ledger.csv` 的 5,293 行、600 episodes，并按 recorded executed action 在 hash-matched 环境/冻结实例中重放。
service、migration、progress、transfer、recompute、operation time 和 clock 的重放 mismatch=`0`。

原账本没有 cache before/after、admitted/victims、prepared-state 或未来 reuse 字段；这些字段的**直接日志状态均为
unknown**。下表来自确定性重放，不冒充直接日志。反事实“若没做 action 4 会避免多少加载”仍不可由 factual ledger
识别，保持 `unknown_not_identifiable_from_factual_ledger`。

| 固定分类 | 数量 | 解释 |
| --- | ---: | --- |
| target 已 resident，无新准入 | 688 | resident set 不变，但 LRU metadata 被 touch；不能等同无副作用 |
| contact-budget rollback / rejection，无 resident 变化 | 333 | 既有样本均为 contact-budget rollback；准入前状态恢复 |
| 新准入、后续复用、未观察 victim cost | 82 | target warm 在后续同 bundle service 前仍驻留 |
| 新准入、后续复用、同时观察 victim reload/failure | 36 | 同一动作既有 reuse，也有 eviction externality；净值不可由分类决定 |
| 新准入、未观察 reuse、victim externality 未闭合 | 4 | 到 episode 结束未复用；不能外推预算外永不复用 |
| 合计 | 1,143 | predicate 覆盖全部上述语义 |

其中 122 个 trigger 实际改变 target resident set并发生模型传输，总 `11,600,584,304` bytes；118 个在账本
episode 内观察到 target same-bundle reuse，4 个未观察 reuse。36 个观察到 victim 后续 reload 或因 victim 缺失服务失败。
`prepared_state` 在 1,143 个触发 transition 中均未改变，这只验证 state migration 未提交，不能抹去 model-cache 变化。

分方法 trigger 数为 SA/MAPPO/PPO=`395/554/194`；control/PopArt=`668/475`。这些是已暴露 evaluation 行为的
描述性计数，不是独立统计单元或性能比较。

## 六个事前局部见证

六个 case 在脚本常量中固定后一次运行；均使用合法 action、同一 production env.step，不调用模型或 agent，不修改 cache
内部状态。固定未来只作为 environment hidden execution path；public semantic state 均不暴露 `actual_mbps`。

| case | predicate | target cache 结果 | 后续事实 | 裁决 |
| --- | --- | --- | --- | --- |
| committed warm then reuse | true | `[base]→[base, alpr]` | 下一步到 target，action 3 成功，model transfer=0 | 最小反例：`migration_success=false` 仍有 durable cache side effect 与 factual reuse |
| contact-budget rollback | true | `[]→[]` | 临时准入因预算不足原子回滚 | 该子类才满足 resident 无变化 |
| unused warm with victim reload | true | helmet→alpr，helmet 被驱逐 | 后续需要 helmet，action 0 重载 `9,641,944` bytes | predicate 也覆盖带可见负外部性的预热 |
| committed warm never reused | true | `[base]→[base, alpr]` | 固定 episode 内不访问 target | 仅能称 observed-unused，不能称永久无价值 |
| target-ready noop/LRU touch | true | residents 不变 | `last_used` 改变 | resident noop 仍可改变未来 eviction order |
| capacity rejection | true | `[]→[]` | bundle 超总容量，拒绝且无变化 | 明确的无 cache commit 子类 |

最小因果见证是第一行：当前 RSU 缺 alpr，action 4 先在 target 提交 alpr，当前服务随后失败且无节点进展、
`migration_success=false`；下一 decision 到达 target，同一节点 action 3 直接成功且 model transfer=0。该见证不证明 action 4
净收益为正——前一步已支付 transfer/load 成本——但足以否定“predicate 等价于没有长期价值”。第三行则证明相反方向的
外部代价也可达。因此需要比较长期 cache benefit、transfer cost 与 victim cost，不能仅看 migration flag。

## 对原候选的裁决

- **否定 predicate 充分性**：强制 cap 会把 118 个已观察到后续 reuse 的新准入 transition 与 rollback/rejection 一并处理。
- **不接受实现**：本轮不增加 advantage cap，不改 policy objective、reward、mask、environment 或 auxiliary。
- **不把它称为实现 bug**：环境按当前合同同时建模 model prewarm 与 workflow-state migration；问题是拟议 policy objective
  省略了 cache side effect，而不是现有 action 4 忘记 rollback。
- **仍未解决整个 actor credit 问题**：selected-checkpoint 证据仍表明 critic scale 改善不足以稳定改善 service，但当前审查
  只排除了一个过宽 intervention predicate，没有定位可实施的新算法。
- **停止**：不继续搜索 predicate、不重做 PopArt 前向、不训练、不追加 seed，不进入 auxiliary/reward sweep。

机器产物：`artifacts/analysis/calibrated_workflow_failed_action4_credit_audit_20261009_v1/`，含全部 1,143 行分类、
字段覆盖、六个见证、summary 与完整性清单；不含 checkpoint、权重或真实数据副本。

最终回答：原 advantage-cap 候选当前应判为 **NOT_READY**。最小见证已经证明“失败 + 无进展 +
`migration_success=false`”不等于“action 4 没有持久副作用或后续价值”；在能够同时归因 cache reuse、transfer 和 victim
externality 之前，不应改变 actor objective。
