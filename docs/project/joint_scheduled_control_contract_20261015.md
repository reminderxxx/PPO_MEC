# 链路预约、动作信用与尺度修复：独立实现合同

日期：2026-10-15。用户授权的独立实现阶段，前置独立审查为`joint_partial_control_results_review_20261015.md`、`joint_partial_prediction_support_review_20261015.md`及`joint_feature_scale_review_20261015.md`，前阶段Git8072db1。旧结果全部保留，不改原始奖励、不筛旧checkpoint；本阶段不是同一只读审查轮内调参。

## 实现范围

1. `--scheduled_controls`启用新checkpoint合同scheduled_parameterized_ppo_v1 / scheduled_parameterized_dqn_v1；旧入口/旧checkpoint默认不变。环境新link_reservation_enabled opt-in、JointRequestControl新link_reservations字段默认空，旧事件序列化去掉新字段。当前合同只支持每RSU一数据链路槽。
2. 八帧公共意图calendar：目的站非抢占模型backlog优先保留；请求声明当前output/input/state恢复需求、即将输出的release与体积、实际选择后的模型/状态前缀。控制面JSON实际字节进入原共享费用。来源为当前私有请求声明及原有延迟cache共享，真实未来不入actor。future槽只是估计；当前槽由环境按优先级/轮转裁决grant，只有指定owner/kind可以消费。禁止cache或prepare先偷走他人的output槽，已有模型传输不强行抢占。未利用grant可以闲置，该保守代价保留。
3. 每个请求选择后更新calendar中的八帧声明，不假设所有候选站都有空闲带宽。prepare净收益重算用实际计划可用prefix量、模型装载、旧版本残余/增量恢复、观测queue及剩余deadline。只对能改变完成/等待的节省计值，不再用任意fractional completion当收益。cache/state同站占槽、执行/prepare同站在条件分布内排斥，stored mask/logprob可重算。计划仍会受延迟快照/未来竞争误差影响，不保证每次实际正收益。
4. 输入/输出/状态以声明service-frame单位log1p/log121缩放；current工作进度用完成比例，deadline slack成为公共候选特征。无dev拟合、无全数据归一化。DAG图、未来cache候选和actor输入同步，避免仅修图仍把25–50的output特征送入head。
5. 信用仍保持原团队奖励与逐请求ledger守恒，target仍为半local半team/初始请求数；新增动作条件Q critic，用实际on-policy return拟合chosen Q，actor每个头减去同一prefix下ΣπQ baseline（对当前sampled action独立、detach）。这是学习的policy-average counterfactual baseline，不是假装有精确因果Q；不添加completion bonus或替代评估reward。
6. SA保留GNN/GRU/surrogate；surrogate return已包含费用，因此移除再次扣cache/state cost的double-charge。改为相对noop/合法均值的return差、训练MSE可靠度缩放、有界residual；费用辅助target改为admitted对象声明后续传输量，而非只看首帧槽。仍非完整动力学反事实模型。
7. PPO更新增加normalized entropy、policy KL/clip/critic误差/梯度步数日志。预声明公共稳定更新：lr3e-4线性降到3e-5、minibatch KL>0.02停止本次后续更新。它控制过大更新，不承诺消除泛化回退。额外sa_constant_update仅关闭lr衰减/KL保护，原资源/预算/尺度/信用都相同，用于后期回退的独立对照。

## 其他对比算法恢复与身份

本轮主比较八配置：SA、MAPPO、SA-no-prediction、PPO、DQN、cache_offload_drl、dag_offload_drl、dt_handoff_drl。均有显式execution/cache/event参数化消费端。原仓库后三个DRL是PPO方向型实现，不是外部论文逐式复现，也不是三个独立off-policy框架；本轮不改变这个身份。

- SA：DAG message passing + 候选节点表示 + calibrated relative surrogate。
- MAPPO：共同actor输入与central critic，无SA图/surrogate。
- PPO：共同公共actor输入，critic屏蔽central RSU部分。
- cache：typed-cache context pool；DAG：剩余节点/边/服务量/完成率scalar context；DT：原始预测confidence/dwell/future target/load context（不因低net机会清空原始DT特征），均不使用SA GNN或surrogate。
- DQN：独立replay/target-network Bellman loss，execution→cache→event三micro-stage；同一物理帧内部折扣1，跨帧gamma.99，buffer4096、batch128、每4episode32gradient step、每2block同步target、epsilon.3→.05。不是PPO别名或MC regression冒充DQN。checkpoint仅供评估，不声称含完整replay的training resume。
- 其他live registry算法（QMIX/MAT/IPPO/DDQN等）仍为旧五动作/旧checkpoint合同，manifest逐项列明支持/排除原因，不偷偷映射成新算法。它们需要独立参数化适配，不能拿旧奖励拼表。

## 冻结预算与结果规则

- 三组主问题与恢复五基线全部固定128episode×seed7/17/27、20train/8dev、最终128主表；8主配置共3072训练episode。sa_constant_update为独立稳定更新support384episode，总计划3456episode。每32存checkpoint，禁止选dev峰值或按结果改预算。
- 固定原NGSIM+Alibaba来源manifest、同资源/奖励、相同GRU拟合预算/horizon（本轮统一8帧，30epoch/16hidden/8history；旧3帧视野短于约7帧状态传输，改动由前置物理预算诊断驱动，禁止与旧轮拼表）。所有学习器同一环境交互预算；DQN按自己的replay优化步数公开，不声称off-policy与PPO梯度完全相同。
- 每32额外评估固定四train窗口indices0/5/10/15，用greedy train probe对照greedy dev，避免拿随机训练回报直接诊断过拟合；记录图激活、KL/entropy/value误差轨迹。没有额外参数更新或checkpoint选择。
- 输出`artifacts/joint_scheduled_control/main/seed<seed>/`（SA/MAPPO/no-pred/PPO）、`baselines/seed<seed>/`（DQN/cache/DAG/DT）、`stability/seed<seed>/`（constant-update SA）。比较计划在开跑前保存；训练/开发/fees/checkpoint/probe/grants/raw events及command均保留。
- 完整版奖励和完成率同时领先是待验证目标，不保证成立。不打开formal/holdout、不更新canonical；论文E0_UNAVAILABLE。相邻dev/同车辆DAG不作为独立cluster。

## 实现验证

362项相关pytest通过：包含七算法真实权重更新/独立checkpoint重放、DQN replay/target不同步见证、output grant阻止更高优先cache先占槽、无空slot/无延迟节省拒绝prepare、图及candidate共同尺度、local PPO critic、action-independent基线、动态8帧训练/加载和admitted小对象最低1slot费用。修改模块py_compile、git diff --check通过。完整训练在干净source commit下另行冻结。


## 独立DQN终止信用勘误（首次主比较回收前）

先有joint_dqn_terminal_credit_review_20261015.md独立审查（2772599），再修复。DQN退出后的团队尾回报原未计入，故旧版本与共同credit目标不一致。新入口terminal_team_tail=true，仅终止transition补入观测MC团队尾标签，非终止仍为Bellman TD；旧checkpoint默认false保留。环境/共同PPO/预测/原始reward及episode生产函数不变，原初始化tensor逐位相同。三个seed各128重训，新增384episode，计划实际训练3456+384=3840。原DQN保留legacy-terminal诊断，不据结果选择新旧。原comparison_plan不改写，append-only amendment明确替换与双source共享代码hash；主比较只纳入修正版本。

2026-10-15完整回收追加：八主配置及两项独立support均完成，120checkpoint/30final首dev事件重放通过；完整版奖励高于六个baseline但未同时超过去预测和MAPPO的完成率，完整数值、回退与claim边界见`joint_scheduled_control_results_review_20261015.md`。本文件原事前/阶段描述保留历史时间含义。
