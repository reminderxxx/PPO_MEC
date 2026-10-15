# 协同选站、机会触发与信用分配实现合同

日期：2026-10-15。前置独立审查：`joint_parameterized_results_review_20261015.md`，来源Git `a711608`。本文件是用户授权的独立实现任务；不修改前两轮结果。状态：IMPLEMENTED_VALIDATION_IN_PROGRESS。

## 公共动作与信息合同

- 环境仍使用 `joint_parameterized_control_v3`，opt-in `--coordinated_controls` 同时启用 productive 冷站条件供给、延迟服务负载和控制意图记账；旧入口/旧checkpoint路径保留。
- 注册算法文件分别提供 CoordinatedSA/MAPPO/cache PPO，公共学习实现留在 `ppo_agent.py`，checkpoint 使用 `coordinated_parameterized_ppo_v1`，不能混装旧网络。
- 活跃请求按固定排序再逐帧轮转，顺序产生实际动作。每个请求看到已声明意图及本轮先前选择的预约工作量，使用自身已知DAG估计剩余计算+input/output服务帧，不读未来轨迹或critic私有真值。
- 预约涉及的声明为 vehicle_id / execution_target / service_frames，环境按真实控制集合的JSON字节计入控制面费用，进入原 shared_state 奖励项。它是集中controller的零延迟意图总线，仍不竞争数据链路，不能当真实V2X协议实测。
- 所有方法共同获得预约特征及声明成本先验：logits扣除（其他请求预约服务帧+自身换站状态/模型费用）/(剩余帧数/四站)。这是明确的公共load-aware prior，不是隐藏选站覆盖，也不是SA独有优势。no-coordination消融不消费预约量/该队列先验。
- 冷站cache动作仍是精确条件mask，模型/状态/input/output仍共用数据链路。公共初始意图至少8帧；已执行中的节点/待送达输出保持站点，失去覆盖可恢复；节点变化允许重新选站。额外共享service_work_units已成为actor特征。
- cache候选获得各自DAG拓扑距离与对应节点GNN表示，修复只做全图池化时的候选混叠。MAPPO与其他配置拥有相同原始候选特征，SA额外编码图表示。

## 保留预测并按机会触发

- 原训练窗口GRU保持相同fit16/validation4、seed101、30epoch，增加预测绝对位置及3×训练验证RMSE的位置裕量；网络未改、不会从dev重新拟合。
- 正机会要求当前确实仍处于执行站覆盖中，3帧预测内将离开该站，目标在预计离站时覆盖且有保守几何裕量，剩余episode足够、workflow不会提前结束、confidence>=.5。
- 用自身版本/前缀/在途状态、payload、声明带宽、最低状态帧、目标缓存就绪/模型装载预算及观测竞争，估计reactive停顿与pre-copy后的增量恢复。仅可及时完成且净收益代理>0时保留机会；目标模型就绪才允许state prepare。已开始的有益前缀按剩余传输量评估，不把沉没传输重新算作新成本。
- 净收益代理=confidence×（等待帧节省×.01 + 当前节点关键路径节省比例）−额外传输/竞争/模型费用×.01。这是可检查的启发式代理，不是校准的因果价值或真实无线预测。低机会时预测特征归零并选择不prepare；预测器本身和费用始终保留。
- 无预测消融不使用预测位置/机会，也不proactive prepare；实际失去覆盖后的恢复仍按环境正常收费。ungated消融保留预测但取消机会条件mask，检查盲目prepare代价。

## 信用分配与训练

- 帧账本将每个请求实际节点送达奖励、未完成等待、车端计算、input/output/state费用归到自身；模型与共享控制费用在本帧活跃请求间等分。逐帧请求收支之和必须等于原团队奖励。它是预算平衡的信用分配，不是共享模型的因果边际贡献。
- 原始团队奖励、评估指标和费用系数不变。学习目标为 .5×请求discounted return + .5×团队discounted return/初始请求数；已完成请求自己的后续收益为0，仍保留团队tail。team-credit消融使用团队return/初始请求数。
- 三个控制阶段分别学习prefix baseline（当前critic状态+先前选择，不含本阶段action）。使用各阶段独立ratio/advantage/clip，只有实际有多选项的头参与actor与entropy平均；冻结stored行为特征、mask、head logprob可精确重算，不在执行后替换action。
- SA surrogate保留return/cache/state三输出，费用输出softplus非负，使用相同训练rollout拟合。只对合法候选计算网络，减少大量noop帧的计算。它仍非完整动力学/反事实模型；显式victim仍不学习。

## 事前冻结比较

- 同一原NGSIM+Alibaba来源manifest，20 train/8 dev、所有synthetic资源与价格不变。额外意图费用共同记账，因此不能把新旧轮奖励差当SA机制净增益。
- 七配置：`sa_ghmappo`、`mappo`、`cache_offload_drl`、`sa_no_prediction`、`sa_no_coordination`、`sa_team_credit`、`sa_ungated_prediction`；seed7/17/27，每配置128 episode，共2688 episode。
- 同update_every4、4epoch、batch512、lr3e-4、gamma.99、scale.1、clip.2、entropy.01、value.5、surrogate.2、cache周期4。每32固定checkpoint并评估8dev；主结果只用最终128，不选dev峰值、不筛负窗口。
- 三方法使用相同原始信息、公共先验/动作约束/费用和机会条件；其中MAPPO为合同匹配且有共同调度先验的first-party MAPPO，不应表述为外部论文原版复现。
- 规则是辅助参照，不作为teacher；正式统计比较在上述七配置内部。保存command/manifest/raw events/checkpoints/预算与机制计数；formal/holdout保持未评估，paper_claim_ready=false。
- 完整版胜出是待验证目标。缺少实际迁移机会时，完整和无预测可能一致；不得用额外prepare bonus制造差距。

## 验证

338项相关回归覆盖三算法概率/条件mask、变量长度候选、prefix credit更新与恢复、同轮预约、候选拓扑表示、共享费用守恒，以及覆盖重叠/已出站/不及时/费用过高/已有copy/低机会的迁移正负例。真实链4episode smoke仅验证入口；最终预算在干净commit冻结后执行。
