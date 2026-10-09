# CSCWD 强基线行为差异只读诊断计划

- `frozen_at`: 2026-10-09（Asia/Shanghai）；实施前冻结
- `source_commit`: `a08869388f9962149198054b5f8a0d6dd4d07eae`
- `source_run_id`: `cscwd_causal_strong_baselines_dev_20261009_v1`
- `target_venue`: CSCWD 2027（拟投）；本轮不写论文或论文表
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scope`: 已消费 36 实例的只读开发诊断；不重训、不选模、不调参、不重复正式评价

## 固定输入和停止门

只读取原 run 的 `run_manifest.json`、`artifact_integrity.json`、440 行 `evaluation_rows.csv`、3,308 行 `behavior_ledger.csv`、20 个 selected checkpoint、原 36-instance manifest 与冻结配置。先核对 run terminal PASS、source commit、所有输入文件 hash、行数、方法/seed/实例身份；失败即停止，不修原件。旧 oracle/泄漏轨迹不得混用。原科学 runner、环境、actor、reward、checkpoint 全部只读。完整 5 seed 与 20 个开发评估实例保留；窗口为外层单位，同一窗口不同 seed 不算独立样本。

## 轨迹诊断

按记录的每个 `method×seed×instance` 动作序列在相同冻结环境中重放一次，逐步核对 action、服务、进展、迁移、transfer/recompute 与 episode summary。重放只为补记 cache event、目标 resident before/after、驱逐对象、后续命中与后续重新加载；不生成新 policy evaluation。每步预定分层：当前 bundle ready、目标 prepare feasible、公开预测首个 handoff 距离（unknown/1/2/3+）、目标 resident 变化、实际准备 commit、节点进展。不得仅凭 action 4 次数或低 transfer 宣称机制收益。

逐 `seed×split×instance` 报告首次 SA 与 PPO/MAPPO/DT 动作分歧及之前动作前缀是否一致；只有相同前缀才作为同状态动作分歧，其他差异只属于状态分布差异。逐 episode 给出首次失败、最长/总连续无进展、后续目标 cache reuse、victim reload；逐 seed 保留完整样本的 completion 差异和失败分布。全部实例成本与双方共同完成子集的耗时、字节、重算分开列，附完成覆盖率。不能将少完成造成的低传输视为收益。

## 共同状态前向预算

若轨迹不能明确区分策略动作与状态分布，最多选 60 个 train/dev 状态。只用两条固定合法 replay：`cache_first` 在当前 bundle 缺失时 action 0，否则 action 3；`periodic_prepare` 在当前缺失时 action 0，否则每第 3 个决策若 action 4 合法则 action 4，否则 action 3。每实例最多 24 步，遇完成/截断停止。候选状态按 `current_ready×target_prepare_feasible×predicted_handoff_distance(unknown/1/2/3+)×target_ready` 分桶，各桶按 `(split,design_id,schedule,step_index)` 排序并轮转取样，至多 60；不按奖励、方法动作或最终结果筛状态。

20 个 selected checkpoint 对每个入选状态只读 deterministic forward 一次，上限 1,200。分别记录五动作 mask、策略概率、raw head、聚合原始动作、mask 投影、最终动作；不得把这些字段混为同一决策。核对 checkpoint 文件 hash，调用前后逐 tensor 比较网络、optimizer、value normalization 状态；任何变化停止并记录。策略差异只据共同公开状态解释，不能从相关性直接定性架构问题。

最多 12 个预定义合法 synthetic `env.step` 局部反例，若需要才使用；只比较同状态 action 0/3/4 的即时服务、目标缓存和下一步可见的延迟准备，不做未来 oracle 性能排名。若当前诊断无需它们，记录 `0/12`，不为制造优势而筛/改环境。

## 预测审计与交付

对旧 run 记录的因果预测逐步核对 predictor hash、prefix 边界、known/unknown；后验比较下一步 stay 与实际 handoff 分开计数，并单列 handoff 目标与首个 handoff 时间（截尾不能按正确）。即使逐步准确率 100%，也不得写成迁移预测优秀。

输出 create-only 本地诊断目录中的完整重放账本、共同状态前向记录、机器汇总与完整性回执；Git 只提交脚本、计划、简短报告和小型汇总，不上传 checkpoint、真实数据或大账本。最终定位只可在动作执行投影、策略决策、选模/seed 敏感、信息表示可辨识性四类中选有证据的一类；否则记未知。最多给一个最小干预候选及可证伪条件，本轮不实现或训练。
