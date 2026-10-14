# 参数化学习 productive v2 实现任务

日期：2026-10-14。任务来自独立 `joint_parameterized_representation_review_20261014.md`，不是在其审查轮修改结果。状态：IMPLEMENTED_VALIDATION_IN_PROGRESS。

## 共同修复

- opt-in `--productive_controls`，旧路径默认 false，旧 checkpoint 不需要迁移；未加该参数的 v1 结果保留。环境 API 仍是 v3，学习 contract 在 manifest 的 protocol/productive_controls 与 checkpoint options 中冻结。
- 四 RSU identity one-hot 放入每个候选特征及上一控制的条件特征，解决两个同状态但不同站的目标绑定歧义。额外提供自身车辆在当前公开 roster 的序号和当前位置相对固定站点的坐标，避免同观察请求无法打破贪心选站对称性。三方法都获得相同信息。
- 环境 opt-in `service_load_sharing=True` 将**上帧声明的服务站**上的未完成请求数及剩余节点工作量加入 RSU 共享 payload。关联负载与服务负载分开；本地当前值、远端按原有 interval/delay 到达，JSON 字节费完整计入，尚不竞争数据链路槽。
- 相同公共可行性约束：有覆盖站或合法车辆执行时不选纯等待；cache 决策周期中，已选 RSU 缺当前 adapter 且未在加载时，cache 条件分布只保留该执行站当前节点的 adapter closure。未到 cache 周期仍 noop，正在加载时可学习其他准备，模型就绪后所有未来/当前 DAG base/adapter 选项恢复。
- 条件 mask 由已选执行 action 与冻结行为 feature/mask 重建，联合 logprob 精确求和。**没有环境执行后动作覆盖，也没有新增免费自动缓存**。所有三方法与两消融使用相同约束，不能把可行性修复单独宣传为 SA 的方法优势。
- 状态/优先级仍由策略学习，显式 victim 不学习；typed LRU 共用。SA 的 surrogate 与 GRU 定义保持 v1，不调其系数以追高分。

## 冻结再训练

与 v1 同数据、同 20 train/8 dev 窗口、同 seed7/17/27、各 128 episode、每32评估、固定最终128 checkpoint、同4ep update/4epoch PPO。配置五组 SA、MAPPO、cache PPO、SA-no-surrogate、SA-no-prediction。新增服务共享字节影响总费，故 v1/v2 的直接奖励差只能是工程变化诊断，主要机制结论必须在 productive v2 内配对比较。输出 `artifacts/joint_parameterized_learning_productive/seed<seed>/<run_id>/`，整轮失败保留，formal/holdout 不运行。

## 验证

针对三方法检查 identity 不混叠、冷站 cache target=current selected target、非周期 noop、条件 mask 零概率拒绝、相同 stored rollout logprob、update 与 opt-in checkpoint；检查服务负载在远端按一帧延迟到达和共享字节计费；相关回归327 passed。4 episode 真实链 smoke 与正式开发预算分别记录，不混作性能证据。
