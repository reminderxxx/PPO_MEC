# 最小有用任务验收与单一机制试验冻结方案

- `frozen_at`: `2026-10-04`
- `scientific_base_commit`: `01d0617d7fb66793329b3c482f794750fb101063`
- `plan_schema`: `ppo_mec.minimal_useful_task_acceptance.v1`
- `formal_or_holdout`: `forbidden`
- `training`: `forbidden`

本轮先验顺序为：输入与许可资格门禁 → 最多一个任务的开发/锁定正确性 → 仅在正确性成立后选择一个机制对照。
完整机器可读方案见 `configs/acceptance/minimal_useful_task_acceptance_v1.json`。

调用预算在任何模型输出前固定为：开发组最多两套有依据配置、共 8 次；锁定检查组 8 次；单一机制对照 16 次；
总上限 32 次。科学执行累计上限 1,800 秒，单进程上限 300 秒，不自动重试。若没有同时满足本地图像、独立标签、
许可可追溯、任务与模型能力匹配及跨组无近重复的输入，立即以 0 次 `generate` 停止，不执行机制比较。

若后续任务有真实下游依赖，只可选择 A（重算前缀 vs 恢复前缀后只跑后缀）；若只是独立视觉服务，只可选择 B
（逐请求重载 vs 复用同一已加载 base/adapter）。不得同时执行 A、B，也不得为制造 DAG 而串联 ALPR 与 Helmet。
