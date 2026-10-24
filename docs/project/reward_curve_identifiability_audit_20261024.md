# 长训练奖励差异的可识别性调查（2026-10-24）

用户要求调查能否得到完整长训练中的算法差异，并以真实结果验证主方法整体较优。
本轮是独立只读调查，不修改算法、奖励、训练协议、旧曲线或 checkpoint，不启动新训练。

## 调查结论

现有公平长训练可说明该固定任务下的学习轨迹与奖励平台，但不能提供用户期待的主方法整体优势。
全部基线和负结果保留，不能靠拆图、筛场景、延长同一协议的预算或改变平滑口径补出该结论。
未来共同任务可以检验算法差异，但无法预先保证所有曲线分开、主方法领先或出现渐进上升形状。

本轮新增确认的实验边界是：冻结科学 source 的 `make_episode` 未传 `cache_capacity_profile`，
真实默认 constructor 探针确认容量约束关闭、使用 legacy adapter-only profile。每个 episode
重新 clone mobility frames、RSU/cache 和 workflow，最多22步，输入窗口24帧；每cell在48个固定
窗口/workflow组合上完成32循环。多episode意味着重复训练，不等于持续缓存竞争或新增独立任务。
episodic reset 本身是正常设计；这里的限制是未检验持续容量竞争下的缓存取舍。不能把该事实推广到
另有显式容量配置的正式 benchmark，也不能仅凭此断言它解释了所有曲线重合。

最终检查点的逐步执行与回报核验再次确认，多个结构不同的基线在本验证分布产生完全相同的行为；
主方法动作不同但未转化为相对强基线的服务指标收益。该结论不等于网络/概率相同，不证明基线
实现重复，也没有通过去规则消融分配先验、门控或任务设计的因果份额。详细逐算法数值只留在本地。

另外，单独重算已有 D/E 小预检的全部4704条记录：原生奖励分项与两个服务账本逐条匹配。
该小预检验证部分无观测到的transfer/prepare兑现，训练兑现均为same-step；不能据此直接证明
提前准备的后续收益。主体切换/预测绑定、覆盖失败、观测截断与策略失效分别保留。
这部分来自不同源码与预算的预检，不混入30cell长训练，也不把它推广成完整长训练无handoff。

## 可复查证据

入口 `scripts/audit_reward_curve_identifiability.py`：

- 严格核验plan semantic hash、冻结source Git/脚本hash、原完成审计及七个用户修改的保护diff；
- 核验全部30cell训练/验证CSV hash、960个cycle的共同support和逐unit交互分配；
- 对全部990份验证trace逐step归约，与23760条原验证结果对账；全部365498条step记录纳入；
- 输出全部算法/seed/checkpoint、最终描述性配对win/tie/loss，以及逐cycle数据，不作最优点选择；
- AST检查实际producer调用，隔离子进程只构造冻结默认环境，不reset、不step；
- 小预检单独按原审计hash核验并独立归约，不删除失败或将seed重复当作独立统计cluster。

机器证据位于 `artifacts/analysis/reward_curve_identifiability_20261024_v2/`：`report.json`、
`algorithm_results.csv`、`checkpoint_results.csv`、`training_cycle_results.csv`。完整长训练来源为
`repaired_fair_reward_continuation_20261022_v1`；小预检来源为
`sagh_service_chain_audit_20261024_v1`。v1为本轮先完成的长训练核验输出，保留，不覆盖。
源数据文件、旧checkpoint与整个receipt inventory未在本轮重新hash；引用既有完成审计，不能
把本轮描述性归约升级为新的完整formal artifact审查。

## 下一轮应验收的事项（尚未实施）

1. 对显式主体/预测绑定候选完成真实provider及失败消费链验收，保留全部冻结暴露和缺失reward。
   现有585d8cb主要是合成运行时边界及旧记录保护验收，不能写成真实收益已改善。
2. 按mobility/workflow/resource特征事先声明共同负载分布，让有限缓存竞争与
   prepare→同主体transfer→后续服务具有可观测机会；不按收益或主方法排名选窗，保留无机会、
   失败与截断条件。改变分布和观察长度必须另立协议，旧新分数不可直接当作算法改进幅度。
3. 先验收网络输出能控制实际动作、监督目标可达、同一执行映射贯通训练和评估；局部梯度方向
   改善不替代真实服务收益。旧长训练使用45c8c88，未包含后续显式共同映射等独立候选。
4. 上述条件通过后再固定共同预算、所有seed/checkpoint和早期较密的验证点，执行独立长训练；
   横轴用交互次数，原生reward与完成/连续性/回传成本/实际准备兑现并列。事先定义停止规则，
   不训练到主方法赢才停止，不要求每个算法都必须有不同最终reward。

这是一条待验证路线，不是新训练授权、已完成的协议冻结或主方法优势预测。本轮不开展文献novelty
评分、正式比较或holdout，不自动合并任何候选，也不调整奖励/teacher/辅助系数。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B scripts/audit_reward_curve_identifiability.py --run-root artifacts/analysis/repaired_fair_reward_continuation_20261022_v1 --output-root artifacts/analysis/reward_curve_identifiability_20261024_v2 --service-chain-root artifacts/analysis/sagh_service_chain_audit_20261024_v1
```

上述create-only命令已经执行且PASS，不得覆盖或复用同一输出路径。新增真实环境reset/step、learner、
optimizer均为0；语法、拒绝覆盖/损坏plan和原汇总匹配的补充校验见本地`verification.json`。

reviewed_at: 2026-10-24T19:11:55.674181+08:00；literature_cutoff: 2026-09-28（未新评）；
target_venue: CSCWD2027 development；artifact_run_id: repaired_fair_reward_continuation_20261022_v1；
policy_version: tmc_review_policy_v3_20260621；scientific Git: 45c8c88e390f49fc4e66c70871c2318e18883f24；
workspace baseline Git: 068fd0ca0c42499dc20b8dd7a8e316f4b172a6e8；
evidence_level: READ_ONLY_NATIVE_CSV_TRACE_REDUCTION_FROZEN_SOURCE_AND_DEFAULT_CONSTRUCTOR_PROBE；
正式优势/策略收敛/formal/holdout: UNVERIFIED；novelty/评分: N/S。

## 2026-10-25：重复曲线合并与差异部分提取

用户进一步要求减少重复行为展示，单独看差异是否随训练扩大并趋稳。本轮新增独立消费者
`scripts/plot_reward_difference_parts.py`，只消费原完整run与上述v2只读审计的hash清单。
全部9baseline、3seed、33checkpoint与正负结果保留，未改算法/奖励、曲线原值或训练协议。

三个展示视角分别是：

- 只有当每个seed、检查点和窗口/workflow的回报都完全相同时才合并曲线；保留所有算法名。
  回报相同不等于网络/行为/实现相同，最终值相同但中途不同的曲线不合并。
- 按完整episode动作序列是否不同、原生reward是否不同分别统计覆盖率。不同动作但reward相同
  的样本保留；相同动作但reward不同也不能被动作筛选过滤掉。
- 显示全体平均差值、所有非零reward差值的条件均值，以及按初始动作差异固定的cohort曲线。
  动态非零子集保留正负两种符号，并同时导出分母、覆盖率、加权贡献与容差剩余项；未用累计
  差值、绝对值或单边筛选制造持续拉大的正向优势。空子集为null，绘图断开，不填0。

精确分解是：`全体差值 = 非零差异覆盖率 × 非零差异条件均值 + 容差内剩余项`。
动态子集成员随训练改变，其条件均值不能直接作为学习进步；初始动作cohort也是事后诊断，
不构成预注册独立评价。seed/workflow在相同窗口上的重复不充当独立统计cluster。

实际输出确认：重复曲线可以减少，动作差异与奖励差异的覆盖率显著不同；过滤持平回报不会
自动得到主方法优势。部分较弱基线的正向最终差距也未满足从初始持续扩大的描述。
详细数值、全部条件/全体曲线和21,384条配对记录仅留在本地
`artifacts/analysis/reward_difference_parts_20261025_v3/`；v1/v2为图形排版检查前版本，保留。
这仍然不能证明主方法长期优秀、策略收敛或formal优势；不通过降低基线能力或奖励动作多样性
来强行分开曲线。下一轮须按上文机制机会/可控性验收条件进入实际修复与独立比较。

专项验证：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_reward_difference_parts.py -q`
通过10项，覆盖选择分母放大、正负混合、行为差异无收益、相同行为异回报、空子集、非有限输入、
容差分解和负向差距/平台不得变成优势结论。完整导出校验990份trace/365498步记录，三张PNG
完成目视检查并同步生成PDF；完整命令、原汇总匹配、输出hash及拒绝覆盖检查见本地`verification.json`。
新增环境/learner/optimizer均0，旧artifact与七个用户修改保持。

reviewed_at: 2026-10-25T00:41:00.741578+08:00；literature_cutoff: 2026-09-28（未新评）；
target_venue: CSCWD2027 development；artifact_run_id: repaired_fair_reward_continuation_20261022_v1；
policy_version: tmc_review_policy_v3_20260621；scientific Git: 45c8c88e390f49fc4e66c70871c2318e18883f24；
workspace baseline Git: 6fb623d1c31c97dcfb1a54f7232d90ba6fb2b40f；
evidence_level: HASH_BOUND_ALL_CHECKPOINT_PAIRED_DIFFERENCE_DECOMPOSITION_ONLY；
正式优势/策略收敛/formal/holdout: UNVERIFIED；novelty/评分: N/S。
