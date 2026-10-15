# 持续请求学习：预测器划分复核（正式训练前）

- reviewed_at: 2026-10-16 Asia/Shanghai
- literature_cutoff: 2026-10-15（无新文献）
- target_venue: IEEE TMC（政策目标）
- artifact_run_id: joint_delivery_control_v2/protocol/predictor.pt；joint_stream_learning软件联调
- policy_version: tmc_review_policy_v3_20260621
- Git: 2beece5（学习适配实现尚未提交，软件测试源码dirty）
- evidence level: predictor原bundle字段直接读取；paper E0_UNAVAILABLE / Unverifiable

## 只读定位

原GRU horizon=8、seed=101、epochs=30。fit_offsets包括240到2040的base/tight副本；内部validation包括2160/2280/2400/2520。新共同学习合同以原train前12个窗口为train、后8个为dev，则拟作新dev的1680/1800/1920/2040已进入旧fit，2160以后进入旧内部validation。冻结复用不等于新dev独立。所有算法共同使用同一预测器仍不足以支持该划分的全链泛化表述。

完整工程v3是train-only机会诊断，原件不变；目前新学习仅软件smoke，无正式新科学训练。不能依据重叠数据上的预测/奖励优势选模型。

## 独立修复范围（覆盖前审第5条“复用GRU不再拟合”）

正式共同学习protocol准备时只用新train的3个480帧会话，按原有fit_predictor的固定80%会话规则：前2会话拟合，第3内部验证；seed101/30epochs/horizon8/原GRU结构和学习率固定，不按结果选择。每个会话车辆连续，fit/内部validation/newdev所有原frame区间显式核对互斥，fit/内部validation offsets与权重SHA保存。所有方法与seed共用该bundle；去预测版不使用输入。新dev只做策略32/64评价。增加一次固定预测器拟合成本，策略1536session/737280frame上限不变，正式RL启动之前写清manifest。

局部实现复核同时发现新候选构建不可把当前节点迁移net proxy复制给queued/后续不同节点；非当前候选必须清零该迁移值，仅保留自身catalog/队列风险特征。此为字段语义修正，不按得分挑参数。修复后重新软件联调，旧smoke原件保留且不得混作性能结果。
