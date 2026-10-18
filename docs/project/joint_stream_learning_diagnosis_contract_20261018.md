# 第二阶段：共同学习链与同状态行为诊断合同

- 冻结日期：2026-10-18；授权：用户“推进第二阶段”。
- 阶段 ID：`learning_diagnosis_stage2_20261018`。
- 原科学源码：`35975cb3d73c84ddbabf01f6749bbc2d60c826de`；阶段一评估源码：`4823a1a05ddbd174c3c5079b7b934f46c4d183f6`。
- 本阶段起点：`4ee7fd0504c46da0fe12b00039026a3200f47df9`。
- 原科学 run：`artifacts/joint_stream_learning/20261016T162100Z_confirmation`。
- 输出：`artifacts/joint_stream_learning/learning_diagnosis_stage2_20261018`；机器合同 `contract.json` 冻结确切清单、SHA、预算及命令。

## 目的与边界

先诊断共同学习路径，再判断 seed17 的动作评分与物理行为。仅增加诊断工具，不修改算法、reward、数据、费用、动作权限、预测器、共享机制或任何历史结果。训练与优化预算均为 **0**，不续训、不挑峰、不替换最终64会话模型。原有唯一确认训练额度1/1已用完。本阶段不得启动第三阶段修复或任何新训练。

阶段一的15个见证是读取已消费开发结果后选择的诊断案例，不是独立性能面板、随机抽样或holdout。全部清单从阶段一 `recovery/stage2_task_points.json` 逐项冻结，不根据诊断结果增删。缺少原训练行为张量的地方必须保留缺失标记。

## P0：只读共同学习链核验

检查行为/当前条件概率、两遍执行动作的mask/prefix、团队尾回报与价值/Q单位、DQN跨空闲/换请求/会话边界及回放覆盖。合成张量只可forward/autograd；禁止调用完整update或optimizer.step。回放构造反例只执行原源码的纯transition构造前缀，不运行学习循环。对真实日志核验SHA后重建队列资格，不能冒称实际随机采样索引。

一阶缺陷先形成独立报告；本阶段只记录，不为了消除缺陷修改算法。DQN回放覆盖缺陷已由独立反例确认，当前排名保留为带已知缺陷的历史开发结果，不能作方法机制优势归因。

## P1：有界参考状态重建与纯评分

原紧凑controls没有完整语义状态及行为概率输入，因此用户的阶段二授权在此具体化为以下有界重建，执行前冻结：

1. 仅固定15个 `(method, seed, case_id)`，每格只运行一次原 **64会话最终模型** 的确定性参考轨迹。每格最多480物理帧，总共最多 **15会话 / 7200评估物理帧**。不额外运行0/32轨迹，不重复资格轨迹，不开启新完整面板。
2. 每格存完整参考events/receipts、summary、rollouts。summary逐字段对账原最终64 raw；参考事件/收据可按原identity核验。源码123指纹、protocol/assets、manifest、dev/checkpoint SHA均须匹配。参数执行前后不变。
3. 捕获frame0及其后最早8个存在任意实际head至少2个有效候选的决策帧，保留各选中帧全部车辆及两遍计算；每格最多9帧、每遍最多72行、全格每遍最多1080行。保留完整语义decision、候选、控制器memory/calendar/reservations和张量输入，并记录SHA。
4. 首遍只比较execution。次遍固定参考64的完整autoregressive prefix，使0/32/64与head输出置零的软件控制面对同一batch、实际mask及上下文。比较的是条件评分/argmax，不是替代策略的自主运行结果。
5. 置零控制用临时forward hook，不改参数。数值公共先验仅execution存在；cache/event没有数值先验时使用最低有效索引破同分，不能称为强规则基线。
6. 纯评分器不持有环境、不调用env.step或优化器，校验输入/参数/RNG不变、各变体mask一致、64重算bit-exact一致。DQN仅online heads确定性评分，无epsilon采样。
7. 最多2160捕获调用 ×4评分变体；未达采样数如实记录，不补挑状态。故障保留现场，不自动重跑参考轨迹，不追加案例/帧。

## P2：同参考数据的梯度与物理链条

梯度仅用上述选中帧的second-pass实际reference actions、底层categorical logs/baselines及同次参考MC labels，最多1080行/快照，对0/32/64各一次固定聚合审查。按冻结loss/active-head标准化，分离policy、value_half、entropy、surrogate_point2与模块梯度，计算共同裁剪系数；只允许autograd.grad，不调用update、backward优化步骤或optimizer.step。参考策略是greedy，categorical log不是确定性策略实际采样概率；ratio命名reference_categorical_ratio，64应≈1。这里只计算真实开发状态/固定参考动作与事后MC标签上的PPO形式损失梯度，不是有效off-policy估计或原训练梯度，不能据此直接归因性能。

独立读取原final64事件，核验选站变化、已付状态、模型族重新准入/驻留、实际compute/交付及optional预取对象使用。保留成功对照和反例，不能把先后顺序或总费用变化直接写成因果损失。真实先验已包含状态恢复、冷模型及族冲突费用，不能无证据声称缺少这些成本。

## 审查、交付与下一阶段

按 `top_journal_review_policy.md` 顺序独立审查，列明日期、文献截止、目标venue、run/commit、证据级别与缺失。开发证据不能替代formal/holdout；当前paper readiness仍为Unverifiable。不得声称收敛、预测/GNN/surrogate因果贡献或修复必胜。

完成后交付真实正负结果、复现路径和第三阶段可执行任务点；更新PROGRESS/BUGS/ARTIFACT_RECORDS及目录入口，验证后只提交脚本/测试/文档并真实push。真实数据、checkpoint、私有artifact不提交。

下一阶段若提出最小修复，应先冻结独立软件验收；新训练必须另获明确有界公平合同。保留 `next_stage_preservation_plan_20261017.json` 的整体与seed7/27非回退门禁；未满足时保留现有最终64版本，不能用seed17改善抵消其他优势退化。
