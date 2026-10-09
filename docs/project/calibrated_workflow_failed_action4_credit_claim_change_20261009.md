# 给 A 论文线的主张变更：failed-action-4 credit candidate

`reviewed_at=2026-10-09`；source scientific commit=`858bc797...`；audit artifact=
`calibrated_workflow_failed_action4_credit_audit_20261009_v1`。

请勿写入“服务失败、无节点进展且 migration 未提交的 action 4 没有长期价值”或“下一步应将其 positive advantage 截断”。
源码和既有行为账本重放表明：action 4 在当前服务检查前先执行 target model-bundle admission；即使
`migration_success=false`，target cache commit 仍可保留。

在既有 5,293 行 behavior ledger 中，拟 predicate 触发 1,143 次：688 次 target 已 resident、333 次 contact-budget
rollback、122 次新 target admission。122 次新准入中 118 次在同 episode 内观察到后续 same-bundle reuse；36 次同时出现
victim 后续 reload/失败，说明净值可能正、负或混合。直接账本未记录 cache before/after 与 victims；上述 cache 字段来自
hash-matched deterministic replay，反事实 avoided loading 仍为 unknown。

安全表述：PopArt selected-checkpoint 结果继续支持“critic scale tracking 改善但不足以稳定改善服务”；actor credit 仍未定位。
原 non-committing failed-action advantage cap 仅作为历史 proposal，状态改为 `NOT_READY`，未实现、未训练、未形成算法或
论文贡献。该变更不修改 A 主论文、不改变 A 线 original-reward strong-baseline 的独立协议，也不允许合并两线排名。
