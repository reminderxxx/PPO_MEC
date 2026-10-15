# 持续请求共同学习合同 v1

日期：2026-10-16。依据独立前审`joint_stream_learning_review_20261016.md`，预测器划分与当前迁移值字段勘误见`joint_stream_predictor_partition_review_20261016.md`；首次软件消费端失败见`joint_stream_learning_smoke_review_20261016.md`。旧工程v3与历史delivery结果不覆盖、不拼接。

## 执行与观测

`StreamControlPPO`复用既有三头自回归PPO/MAPPO网络和独立surrogate裁剪机制；SA/MAPPO/cache/DAG/DT分别在各自算法文件直接定义Stream类并由registry导入。DQN保留Q/replay/target网络损失，另实现跨空闲/请求切换的观察区间费用，不伪装成PPO。

action=`joint_stream_control_v1`；checkpoint=`stream_parameterized_{ppo,dqn}_v1`，旧checkpoint拒绝。图覆盖active+已发布queue，总节点≤64，无截断（本池最多60）；DQN保留513候选上限，而旧257不适用。请求切换清空执行目标记忆，缓存保留由环境管理。无RNN策略隐状态，GRU预测器持续维护因果车辆历史。

actor固定20维自身/32维候选；critic36维（PPO后16维清零）。候选提供base/adapter精确字节/容量、模型ready/loading、已知队列工作量、状态恢复费用、已发布模型复用需求、工作集容量风险、预计使用前序服务时间、节点服务/输出尺度。未知远站容量只用公共catalog最大合法模型闭包字节作归一化尺度，不推断真实容量、不放开未知缓存准入；见独立feature-scale审查。身份只用于绑定，不用模型名字哈希推断未知类别。已屏蔽候选特征可为零，不影响合法动作分布。当前节点迁移净收益不能复制给不同queued/未来节点。风险/时间均为公开已知信息的近似，不冒称因果净收益。

执行站先共同采样，再重建真实意图下的必要model/input/output/state calendar，缓存/准备随后采样。当前缺失模型可逐帧反应式加载；可选预取四帧一次，完整异构closure必须放进未占用前缀。各算法共用当前依赖准入保护；未来需求只是特征，不承诺免除未来LRU外部性。预测迁移依实际离站机会、净节点收益和目的站槽开启；去预测版关闭预测特征与准备机会，其他合同相同。

## 信用和预测器

PPO/critic/surrogate标签均为观察到的session团队未来总奖励×0.1/物理8车；gamma=1。完整轨迹先回收所有帧费用，再按实际决策帧回填，含无actor空闲帧、早完成后及后续请求收益。训练标签可以使用事后结果，actor/critic看不到未发布DAG。该共同return并非已识别每个动作的因果边际贡献。

DQN中间三头discount=1；物理下一次该车决策允许跨空闲和请求切换，奖励=两次观察团队MC的差；末次为完整团队tail。不同session不能连接，checkpoint不支持恢复训练replay。

GRU固定seed101/30epochs/horizon8，在新train的前2个480帧会话拟合、第3内部验证；新dev两会话不进入fit或内部验证。记录真实原始frame/time分区与权重SHA；各算法/seed共用，准备阶段一次拟合，原预测bundle仅为历史参考。先前“复用GRU不再拟合”计划由该划分修复取代。

## 冻结科学预算与消费端

仅原train20个120帧窗口组成5个互斥480帧会话，前3train、后2新dev，原dev/formal/holdout均不打开。原256job固定short/medium前4及long唯一j729；不拼DAG或改profile。每会话24条件，train72/dev48。每seed64个预先轮换训练session，未单seed覆盖72全部，三个seed轮换偏移联合覆盖，所有方法训练日程完全相同。

8配置×seeds7/17/27×64=1536训练session，最大737280物理帧；每4session更新。与旧128×120相比每seed最大物理帧翻倍；不是同计算量比较。checkpoint32在固定12个新dev probe评价，64在完整48条件固定最终评价，主结果不选峰。保存全部48checkpoint、训练/评价逐session汇总、最终逐事件/帧费用、参数哈希、command与source fingerprints。

`prepare_joint_stream_learning.py`冻结数据/预测器/日程；`train_joint_stream_control.py`生产；`audit_joint_stream_learning.py`独立消费逐帧费用、FIFO、依赖/容量、交付、可选准入和对象使用，重载每个最终checkpoint在首个新dev case重放（24次），输出`audit.json`与`results_review.md`；`launch_joint_stream_learning.py`顺序执行有限八配置，失败保留现场，不重跑。执行清洁提交后在新root启动；最终由20分钟heartbeat回收审计、分析并先独立审查后推进。

## 局限

只有两个原始时间会话作为新dev，同workflow池、long单源、模型/链路成本合成、串行DAG、LRU和延迟共享观测仍在。64session学习与泛化是否足够未知；高预取使用比例不证明团队收益，也不能保证SA最高/收敛。formal/holdout封存，paper E0_UNAVAILABLE / Unverifiable，canonical不变。


## 启动前验证与成本

验证：`python -m pytest tests/test_stream_control_learning.py tests/test_multi_request_stream_env.py tests/test_delivery_control_learning.py tests/test_tail_control_learning.py tests/test_scheduled_control_learning.py tests/test_coordinated_control_learning.py tests/test_parameterized_control_learning.py tests/test_multi_request_target_control_env.py tests/test_verify_joint_target_control.py tests/test_joint_model_cache.py tests/test_multi_request_joint_v2_execution.py tests/test_joint_experiment_launcher.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py tests/test_tail_mobility_support.py -q`，426 passed（27.56s）；语法检查通过。

最终软件全链`artifacts/joint_stream_learning_smoke_v4`：八配置各seed7×4session×最多32frame，8个最终软件cell、2048事件/256收据/8checkpoint/8最终模型逐事件重放通过；不是性能结论。

四轮软件全链累计128个32帧上限训练session，第一轮auditor变量类型错误失败原件保留，后三轮逐步验收通过；只读字段检查与局部单测轨迹另列，不计入冻结1536科学训练session或排名。主科学运行未回收前无新SA结论。最终联调的source fingerprints与提交源码逐文件匹配；smoke因开发中dirty，不冒称clean科学运行。
