# 第三阶段：SA学习更新候选与中后期状态诊断

- 冻结日期：2026-10-18；用户授权：“启动第三任务……重点是后续sa的算法提升，以及奖励收敛”。
- 阶段ID：`sa_learning_stage3_20261018`；起点Git `85fb2a205aadc63f4b30b8be15ebdd9dd94f6927`。
- 原科学源码 `35975cb3d73c84ddbabf01f6749bbc2d60c826de`；原结果与所有checkpoint只读。
- 依据：第二阶段独立审查确认policy/value共同裁剪耦合；未建立其与性能退化的唯一因果关系。正向seed7/27裁剪更重的反例保留。

## 1. 优先级与唯一软件候选

本阶段以SA改进为主，DQN缺陷降为后续维护项并继续公开；不利用其较弱结果支撑SA机制结论。只实施一个显式opt-in候选：`separate_policy_value_clip=True`，policy/value各clip norm0.5，surrogate保持独立0.5。保持同一个non-surrogate Adam、相同参数顺序、loss/MC信用/学习率/entropy/样本顺序、网络与推理、预测/共享、prior/mask/预约、物理费用和reward。

policy包括heads、cache/domain/graph encoder、graph message及graph residual gate；value仅critic、prefix_values、action_values。参数组须互斥、无遗漏。独立0.5改变联合范数上界，不称为相同总梯度预算。候选为待验证假设，不预设奖励提升或收敛。

开关默认False并走原更新路径；旧checkpoint缺字段恢复False。True写入options及`training_gradient_contract=sa_policy_value_clip_v1`；加载须检查flag/tag配对，未知或矛盾拒绝。网络/action/checkpoint tensor结构不变，`training_resume_supported=False`保持。新增loss诊断字段须为有限numeric，兼容现有消费者；明确区分policy/value/surrogate。

## 2. 中后期参考状态补证：执行前冻结

固定SA17/dev022、SA17/dev046、SA17/dev017、SA7/dev046、SA27/dev046共5格，各只运行原最终64确定性参考一次；每格≤480帧，总≤**2400评估物理帧**。原15格stage2归档已与原科学事件/收据全内容核对，此处逐格对账stage2已验证的summary/events/receipts。采样frame清单在机器合同中事先冻结，来自原事件：三个中后期120帧段内最早paid-before-compute换站、最早其他同节点换站、固定中点180/300/420及各±1帧；缺类别留空，不补挑。预列合并127帧≤135，全部车/双pass最多2160捕获调用。

采集须在本阶段算法修改前完成，原123源码指纹一致；捕获完整decision、候选、基础/实际mask、旧target是否可选、target-memory age/commitment、prior和learned margin。0/32/64/headzero仅在相同reference prefix下纯评分；不运行其他策略轨迹，不选新checkpoint，不增加case。exclusive启动标记拒绝重跑，失败保留现场。

该补证按物理episode时段报告交付与换站，不能称训练收敛；时序片段和conditional评分变化不是因果收益。所有正向控制与反例保留。

## 3. 软件验收预算

科学训练session、科学optimizer更新均为 **0**；原1/1确认训练预算仍已用完。只在throwaway模型副本与合成/已存tensor上验证更新实现，qualification最多128次Adam.step、单元测试最多128次Adam.step（包括surrogate），无环境训练、无模型晋级。

须验证：默认更新对原父类逐tensor/Adam state一致；两模式更新前推理一致；组划分互斥且覆盖；价值梯度变大不会改变candidate policy裁剪（控制变量软件反例，不作性能结论）；零/未使用参数稳定；候选/旧checkpoint及非空Adam状态保存恢复、flag/tag负例、旧八配置兼容；参数/optimizer调用计数与软件副本隔离；全部真实原件SHA不变。

算法新增仅在`src/agents/sa_ghmappo_agent.py`；相关脚本、测试、文档单独最小增补。原冻结run需要原Git源码或已存源码指纹重建，不将当前修改后的源码追认旧source35975cb；保留Git与原源码快照。

## 4. 阶段出口与下一阶段

本阶段交付候选实现、软件验收、SA中后期诊断与下一阶段密集快照合同草案。没有新的学习性能结果，不能把软件通过称为SA变好或奖励已收敛。按top_journal_review_policy保留独立报告与证据边界，正式论文E0_UNAVAILABLE/Unverifiable，formal/holdout封存、canonical不变。

下一阶段建议对原版SA与唯一候选做相同seed7/17/27、相同64会话日程的有界配对训练，0/8/16/24/32/40/48/56/64固定完整48条件评估；该训练合同目前仅设计，不在第三阶段自动启动。原reward曲线保留，固定面板快照分别画每seed和整体，不插值伪造快照或平滑制造收敛。尾部稳定门槛需事前冻结；即使通过也只称预算内稳定，不称渐近收敛。

现有最终64模型及整体/seed7/27 reward、完成率、模型与归一化传输量非回退门禁按`next_stage_preservation_plan_20261017.json`完整精度和原容差执行。失败保留现有版本，不挑峰、不拼seed，不无限延训。完成阶段后给出第四阶段任务点；脚本/测试/文档验证后scoped commit/push，真实数据/checkpoint不上传。
