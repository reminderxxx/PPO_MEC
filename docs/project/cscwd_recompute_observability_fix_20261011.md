# Public recompute observability 最小纠错

实现只给 v4 prepared-state public semantic state 增加 `public_execution_history_v1`：last execution RSU、当前是否 handoff 和
已发生历史来源标识。它不含 future route/outcome，不进入现有 encoder feature，不改变 action mask、reward、executor 或网络参数。

Public estimator 在无 handoff 时令 recompute=0；发生 handoff 且 current prepared prefix invalid 时，使用已公开 workflow DAG、
completed nodes 与 measured reference 按 executor 相同 predecessor closure 计算 recompute。旧 profile 不增加字段；缺新字段的
旧 state 保持 fail-closed unknown。Estimator/event/calibration-event/public-rule schema 分别升至 v4/v4/v2/v4，旧 v3 scientific
protocol 在当前代码下拒绝加载，不改旧 artifact。

该改动是接口纠错，不是算法创新。只有全新 create-only v668 probe 在真实 SA consumer 上确认非零 base prepare 与实际
prepare→abstain 后，才可进入一次匹配训练；未通过仍停止。

