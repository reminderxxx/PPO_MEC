# 持续请求 / 多模型工程合同（v3为当前实现）

日期2026-10-16；准入缺陷前审见joint_stream_admission_review_20261016.md / 2d348e6：v2全部216已完成，但raw审计拦下c039/frame41延迟LRU驱逐自身fusion依赖，故资源保护未通过。原件/42帧复现/失败日志保留，不用其排序选参数。v3保持全部216矩阵/规则/费用/资源，增加所有方法共用的本地真实缓存准入校验，拒绝危险可选预取并明确记0模型槽/字节；显式驱逐不能跨当前依赖。新workload/protocol v3、独立root `artifacts/joint_stream_workloads_v3/`重新全跑；下文v1/v2为历史阶段。

前阶段日期2026-10-16；同帧FIFO缺陷前审见joint_stream_fifo_review_20261016.md / 58a8b03。v1完整工程在129单元后停止（另至多1未记录），与得分无关，原件/stop_record保留；下文v1阶段记录为历史。v2以发布清单顺序处理同帧同车请求，环境和必要下界一致，新workload/protocol v2、独立root `artifacts/joint_stream_workloads_v2/`，重新完整216而不拼接。action/observation字段结构保持v1，旧裸v3控制仍拒绝。新增两项逆词典序FIFO回归与独立raw启动顺序审计；资源/规则/矩阵不改。

原日期2026-10-16；独立前审见joint_stream_workload_review_20261016.md（5bc4ca6），用户授权后实现。本合同先识别环境是否提供可用预取机会，不启动SA训练、不产生新RL排名。

## 生命周期与接口

- 新`MultiRequestStreamEnv`、`joint_fifo_workload_stream_v1` / `joint_stream_control_v1` / `joint_stream_observation_v1`。物理执行保留v3。`StreamingRequestControl`绑定active request_id；cache_request_id可指本车已到达active/queued请求，未发布/过去请求及旧裸v3控制拒绝。
- 每车FIFO，外生frame发布、未发布DAG/到达时间不出现在policy观测。已发布排队DAG可见。执行状态、input/output、版本拷贝与执行目标按请求清理；模型缓存/付费在途模型传输跨请求保留。跨session/reset清空，不跨split继承。
- 无active帧仍advance缓存并计实际费用，单独frame receipt避免幽灵事件。已发布active+queued未完成每帧等待费−.01；未发布不收费。团队reward系数原样保留。stream request_rewards明确是当帧已发布pending的均分账本，不冒充边际因果信用；无请求费用归unassigned_team_reward，二者与总reward守恒。
- 旧v3仅抽取终止/等待/账本/缓存节点查找hook，并支持空帧收据；旧默认逐事件保留。旧checkpoint未适配新生命周期，不能默认为新基线。学习循环须显式适配空帧、queued缓存候选、请求身份/记忆/团队return后，从头同预算训练。

## score-blind矩阵（启动前冻结）

- 原train的前两个四窗block，每block480帧，取完整原始train时间上的8辆持续覆盖车；按几何离站/位移规则选择，无policy reward参与。源NGSIM/Alibaba及父manifest/hash验证；不使用dev/formal/holdout来选车、调资源或窗口。
- Alibaba固定前256个job前缀，short=5–9、medium=10–19、long=20–40节点，各取前4个可用DAG；不足4个如实保留（long目前仅j_729/30节点）。只重命名request/node/model身份，不删边、不拼长链、不缩输入输出。拓扑串行执行仍保留，不称并行DAG。
- 到达single_batch每车1请求frame0；queued_batch每车2请求frame0；stream相同2请求分别frame0/240。queued_batch与stream才是相同总需求的到达方式对照；single_batch是不同负载控制，不把奖励跨负载直接排名。
- 每个请求固定model族；single=一个base2048、adapter512；dual=两个base2048/3072、adapter512/1024，依赖身份唯一；全为声明合成bytes，非真实权重/无线校准。cache容量4096/8192，各依赖闭包均可放入；固定compute/链路/状态费，不以算法得分放大资源。
- 2session×3size×3arrival×2catalog×2capacity=72个scenario；每个reactive/naive_prefetch/gain_prefetch三规则，共216单元，0训练/无虚构seed。compute、IO和每车FIFO序列的乐观必要容量/截止下界全部报告；通过不等于充分可行，过载不删除。

## 控制规则与机制边界

共同sticky可达均衡选站、GRU和延迟共享观测，当前model/input/output/state预约保护；差异仅可选模型预取。reactive只准入当前模型；naive选择已发布DAG中未来未缓存模型；gain依据已知节点预计使用时刻/节点交付与原等待效用、完整base+adapter传输、预测驻留与已知working-set压力的启发式权重估计净收益。使用概率未校准，不称精确反事实/SA新算法。所有可选预取以实际异构catalog模拟依赖安全LRU，禁止驱逐同站当前团队依赖，整个非抢占span必须有空槽；未知remote占用阻止可选动作。

当前工程矩阵保留predictor/shared state，但不做状态prepare干预，专门识别模型预取；不能据此宣称预测迁移/学习型surrogate收益。对queued很远的模型，时间域外保守衰减，不能用未来轨迹或未到达请求绕过限制。

## 生产与独立消费

`python scripts/run_joint_stream_workloads.py`要求clean Git，写新timestamp目录、manifest/command/protocol.pt、216 results、全部events和独立receipts。启动前全部cases/catalog/arrivals/源码SHA已冻结。`--smoke`为9个短任务联调单元，允许记录dirty，不能进入结果结论。

`python scripts/audit_joint_stream_workloads.py <run>`独立重建已发布等待费、节点/完成、模型占用/依赖、当前依赖保护/连续span、费用/credit、对象驻留世代使用/驱逐、文件与源码SHA，并对首末scenario三规则各重放（6次完整事件一致）。输出audit.json，全保留正/负/零变化。不筛参数或按均值挑case。

## 训练前验证记录

- 408相关pytest通过：新stream10项，覆盖FIFO/cache复用、旧执行状态清理、空闲费用、queued收费、未来不可见、旧/过期控制拒绝、queued预取、reset、必要下界/拓扑映射、付费预取交付收益；原398相关回归照常通过。
- 旧scheduled 11模型完整事件重放一致。
- 修正smoke原件`artifacts/joint_stream_preflight_fixed/20261015T164030Z_engineering`：9单元/12426事件/2887收据/6重放通过。最初独立auditor的completion键索引错误已修正，原失败log保留；未据smoke均值改模型或选矩阵。
- 下一步clean source完整216工程矩阵，结果另立独立审查；算法优势/收敛仍UNVERIFIED，formal/holdout sealed，canonical不变。
