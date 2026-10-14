# 参数化控制 v1 独立表示审查

- reviewed_at：2026-10-14；literature_cutoff：2026-10-14；target_venue：IEEE TMC（内部方法边界）。
- artifact_run_id：`joint_parameterized_learning/seed{7,17,27}/20261014T105504Z_development`。
- policy_version：`tmc_review_policy_v3_20260621`；Git：`a7a06a0`；evidence level：论文 `E0_UNAVAILABLE`，本报告为源码与部分开发原始值核验，完整实验尚未回收，整体性能 Unverifiable。

## 按审查顺序记录

1. artifact 完整性：第一轮预算/源/hash 已固定，SA 三 seed 的 128 checkpoint/dev/events 存在，其他配置仍在执行；不可提前宣布全表优劣。
2. protocol/provenance：从原 train/dev 重建，holdout 不评估。预测器 fit16/validation4 都是训练窗口。
3. 统计/baseline：三方法相同动作、掩码与预算，seed7/27 SA 最终 8 dev 共 7,680 request-frame 全部 waiting；seed17 主要 adapter_not_ready。未作独立性或显著性推断。
4. 机制：synthetic 四站最小见证中，b/c/d 的执行 candidate 特征两两完全相同。缓存 head 只条件于已选站的特征，没有站点 identity；因此执行 target 不同但同状态时，条件分布无法识别目标。证据：`artifacts/joint_parameterized_learning_audit_20261014/representation_diagnostic.json`。共享 payload 当前含 cache/association，但没有已声明服务目标的负载。随机 broad cache 选择与等待的长期 credit 很弱；这不是训练时长不足的充分证据。
5. claim：不能把 v1 失败归结为 SA 理论失败；共同控制表示未充分表达资源绑定，也不能把 train validation 的预测 RMSE 改善当调度收益。
6. novelty：本工程审查未重新检索 novelty；first-party cache PPO 仍非外部论文复现。

## 独立实现任务（后续阶段）

保留 v1 代码路径/checkpoint 与整轮结果；另启 opt-in productive v2。全方法共同增加站点 identity、车辆/位置上下文和有延迟的上帧服务负载共享；冷缓存下 conditional cache mask 保证选择执行后在缓存决策周期请求该节点 adapter closure，无动作执行后覆盖。存在可执行站时省去纯等待的控制选项。可执行/热缓存后仍保留未来 DAG base/adapter、跨站 prepare、priority 学习。使用同预算固定最终 checkpoint、同窗口/seed，保留失败与负差。审查轮不调参，修复和再训练以独立 commit/manifest 记录。
