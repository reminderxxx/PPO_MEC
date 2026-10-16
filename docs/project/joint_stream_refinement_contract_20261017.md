# 持续请求 v2 有界确认合同

日期：2026-10-17。前置独立审查：[joint_stream_results_review_20261017.md](joint_stream_results_review_20261017.md)，独立审查提交9ae70d5。原run、checkpoint、raw和audit不改写。本轮是使用已消费开发条件的有界确认，不是新formal或独立holdout。

## 问题与最小修改

- 环境/两个消费auditor共同补全remote-current optional判定，所有算法共同保护当前声明依赖。
- registry、trainer、env统一joint_stream_control_v2；checkpoint为stream_parameterized_{ppo,dqn}_v2，v1文件明确拒绝。物理workload stream_v3、FIFO发布、typed LRU、原始数据和reward不变。
- 所有seed/方法共同使用同一64会话日程：24个条件按short/medium/long交错，三原始train时间会话轮转；每个前/后32会话都覆盖全部24条件，避免将初始化seed与不同课程混淆。64不能覆盖全部72个time×condition组合，仍如实保留。
- 所有算法共同选站prior加入缺失base/adapter完整加载帧、正在加载剩余帧、同站当前family闭包超容量的reload代理；state恢复按版本、pending刷新和minimum_state_frames计算。先验可被学习logit覆盖，不修改物理reward，不用未来真值。
- 所有算法可选cache候选共同要求：已发布需求、8帧内预测使用、传输先完成、当前依赖保护、完整空闲传输前缀、正的净节点交付代理。使用时间考虑已付input/工作进度/output，current输出阶段本节点不再作为未来需求；远站当前节点可以用真实迁移候选的ETA准备。目标站在预测使用时需覆盖，当前站则检查使用前驻留；容量压力折减survival权重。代理沿用当前节点交付及unfinished waiting/模型时隙成本单位，不是校准因果概率或实际边际信用。
- DT恢复独立mobility context，DAG保持公开结构domain。同动作/观察权限；auditor拒绝DAG/DT全checkpoint tensor重复，保留当前DAG强基线。
- SA的GNN、surrogate控制变量、预测器和共享状态保留。observed session team MC监督不伪称已识别动作因果边际。未新增学习型驱逐、动力学surrogate或全vehicle/RSU MARL。

## 事前固定预算与评价

8配置：SA、去预测、MAPPO、PPO、DQN、cache、DAG、DT；seeds7/17/27；64session/seed，每4session更新；总1536，最多737280训练物理帧。不同策略提前结束导致实际帧/决策不同，要报告，不能称严格等决策数。

数据、模型catalog、成本、GRU seed101/30epochs/8帧、train0/1拟合train2内部验证、dev两会话/48条件，与上轮构造完全相同。共同32固定12probe，64完整48最终；不按排名延长、不筛seed/条件、不挑峰。无新数据下载，原dev/formal/holdout封存。确认训练仅允许这一轮；失败留原件不自动重跑，负结果照常交付，若无新证据不盲目追加实施。

原旧run最大737280、实际647788；本轮上限相同，不宣称相同实际计算成本。软件验收不参与性能选参数。

## 验收

432项共同链路/回归测试通过（13.11s）；日志artifacts/joint_stream_learning/refinement_tests_20261017.log。包含remote-current反例、前后半程全条件、DT预测敏感性、v1拒绝、净收益/驻留/远站ETA准备与masked PPO/跨请求DQN/队列隐藏等。

软件全链artifacts/joint_stream_learning_refinement_smoke_20261017：32短训练会话/1024帧、8最终单元、2048事件、256收据、8checkpoint/8逐事件重放，audit pass，DAG/DT tensor不同。非性能结论。下一步清洁提交并push后在全新root冻结协议、从头运行八组；完成前不再改源码。20分钟heartbeat只查状态，不追逐奖励。

## 回收与边界

重点比较cache/真实独立DT与DAG，检查全部seed、长双family小容量负例、PPOseed17、后期共同probe、可选object使用/驻留、模型重复加载及成本/交付归一化。输出完整负例，若未领先或仍回退明确报告。仅2外层dev时间会话、long单源、合成物理成本和开发选择复用仍在，不能据此称最高算法/收敛/显著泛化。论文E0_UNAVAILABLE / Unverifiable；canonical不变。
