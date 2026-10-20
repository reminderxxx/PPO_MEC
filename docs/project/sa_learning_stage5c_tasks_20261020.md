# 5C任务点：因果可用的有效动作与信用概率合同

- authored_at: 2026-10-20；status: **用户已授权且完成第1、2项合同/默认关闭工具与零环境软件测试；算法候选、软件Adam、真实环境及新训练仍未授权。**
- current_delivery: [5C1/2交付审查](sa_credit_audit_results_review_20261020.md)、[因果概率合同](sa_credit_audit_contract_20261020.md)。
- evidence: [第五阶段完整独立审查](sa_same_state_results_review_20261020.md)。原版SA固定64及已有整体/seed7/27优势保留，分组裁剪候选不晋级。
- 单一优先级：先明确有效动作类在实际采样时是否可知、正确联合/条件概率及真实行为信用记录，再决定一个最小opt-in候选。暂不叠加锁站、sunk-cost惩罚、clip/lr调参或更长训练。

## 1. 先冻结因果时点与概率证明

沿原first/second pass与owner生成顺序逐字段标记：当前decision、prefix、已生成的peer controls/calendar、尚未生成的priority分别何时可见。离线reference co-controls类可能包含后续owner选择，不能直接用于在线mask或把class-mass冒充behavior probability。

分别证明：一帧物理同效、完整controller/RNG续接同效、固定策略未来return同效；未证明后两者不得把类内Q差异称真正反事实Q错误。其他prepare目标尚无匹配claims，不能按原reference预约自动合并。

推导原joint/autoregressive策略下的合法class logprob、old/new PPO ratio与action-independent baseline条件；对固定teammate actions的条件分布不得误作原联合边缘。若只能在joint action生成后判类，可研究事后控制变量的条件无偏性，不能向actor注入未来controls。

拒绝条件：缺因果可用性或正确ratio时，停止在线折叠/损失修改，明确UNIDENTIFIED；不为取得漂亮曲线放宽定义。

## 2. 补齐真实行为信用可审计字段

若用户授权实现，先仅增加独立诊断日志，默认学习规则/动作/旧输出保持：每behavior UID、session/request/frame/vehicle/pass、因果可见性、完整候选/mask/actions、old logprob、team MC/10/8、stored baseline、raw/normalized advantage、selected Q/surrogate/V、有效head权重、sample/minibatch/epoch与KL/真实Adam delta对应关系。明示forced冷模型准入仍可有物理贡献。

日志必须来自未来实际stochastic行为；阶段五greedy结果不能倒填旧训练记录。不要把全局shared model费用强分单请求，不新增sunk-cost reward项。schema变更同时核对生产端/消费端、bounded输出成本、RNG与训练状态无扰动。

## 3. 零环境软件资格，之后只选一个候选或停止

只用现有tensor/纯概率fixture，先覆盖None/forced、单owner、同order/不同order、相同prepare/未识别其他prepare、稀疏cache、FIFO/idle和高logit/subnormal边界；保留原版7/27正例、17负例及必要离站反证。不得用fixture reward挑参数。

默认关闭路径必须与原版旧checkpoint、完整tensor/输出/RNG精确兼容。确认一个first-order缺陷后才设计一个opt-in最小信用/表征变化；不同时改先验、mask、loss、clip和训练长度。当前软件Adam预算仍0；若需要梯度/优化资格，另行批准最多128私有软件Adam/0环境，单列费用，不当原训练梯度或性能证据。

若没有缺陷/无偏性证明，交付“未识别”及可检验缺口，不强行实现算法。实现后先独立审查、匹配测试与scoped commit/push，再提出新学习合同；本任务点不启动它。

## 4. 新学习必须独立授权

旧阶段四1/1、阶段五参考1/1均封存，不能恢复/延训或借额度。新合同须一个差异、同轮原版对照、全3seed、固定全48dev、固定最终64与真实密集快照、原GRU/日程/环境/reward/权限，预算另外批准。不挑峰、不拼seed，不因尾稳失败延长。

继续原完整精度总体/seed7/seed27 reward、完成率、modelKiB、KiB/完成请求门禁（1e−8容差），总体reward须严格增加；同时过同轮原版对照。seed17自身0点奖励/完成率恢复、费用和连续性交付独立列出，不抵消7/27退步。尾48/56/64 reward范围≤1%、完成率≤1pp、每交付模型≤2%保持；若预算改变，先冻结匹配尾点。formal/holdout仍封存、canonical不变、DQN暂缓。

阶段结束交付正负结果、边界及下一阶段任务；没有用户进一步授权，不执行上述实现/优化/学习。
