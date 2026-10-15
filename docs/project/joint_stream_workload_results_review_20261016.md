# 持续请求 / 长短DAG / 多模型缓存：完整工程结果审查

- `reviewed_at`: `2026-10-16 (Asia/Shanghai client date)`
- `literature_cutoff`: `2026-10-15; no new literature search/novelty evaluation`
- `target_venue`: `IEEE TMC (policy target only)`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `artifact_run_id`: `artifacts/joint_stream_workloads_v3/20261015T172250Z_engineering`
- `git_commit`: `868b0e9fc3db994c7d9aaea4331368c83d71121f`
- `evidence_level`: `engineering raw audited/replayed; paper E0_UNAVAILABLE / Unverifiable`
- `audit_utc`: `2026-10-15T17:41:17.033368+00:00`

## 1. 完整性与前轮缺陷

v3 clean source完整 **216/216**，训练/GRU再拟合/新学习checkpoint均0。独立审计 **416940事件 / 71757帧收据 / 6次首末scenario完整重放一致**，FIFO发布/启动顺序、已发布等待费用、节点交付、credit守恒、缓存容量/依赖、可选当前模型保护与连续span通过。全部216单元的原始results/events JSONL、manifest、command、protocol.pt保留。后处理再次验证两个会话所有帧均属原train时间集合，彼此不重叠。

- v1/19a1cf1因同帧名字排序颠倒FIFO，在129已记录工程单元后停止，另最多1未记录；前审58a8b03与stop_record保留。
- v2/13016c7完整216，但审计发现c039/frame41延迟cache视图与实际LRU不同，可选control预取驱逐自身当前fusion模型；42帧有界重放复现。前审2d348e6及全部原件/失败audit保留。未用得分选择修复，不把v2当资源保护通过或拼入v3。
- v3/868b0e9在本地真实缓存准入端按当前已声明请求/执行目标再校验，危险可选驱逐明确拒绝且不扣模型槽/字节，显式驱逐也不能绕过。所有方法共用；不读未来位置、未发布请求或未来reward。策略仍读延迟快照；本地可行性拒绝不是向策略提供真未来oracle。
- 四轮9单元smoke共36（含消费端tuple键索引修复和FIFO阶段），clean v3 release另9；本次共记录606工程单元=129+216+216+36+9，另≤1中止未记录、42帧有界诊断及兼容重放单列，**新增RL训练0**。

## 2. Protocol / provenance

只消费原NGSIM train的两段不重叠480帧会话，八辆持续覆盖车按外生几何排序选择，4RSU；未打开dev/formal/holdout，不变更canonical。原始Alibaba固定256job前缀按节点数5–9/10–19/20–40各取前4；实际pool为short j_3/15/49/73，中 j_8/34/95/102，long只有j_729（30节点）。仅重命名请求/节点/model身份，原拓扑和input/output不缩小、不拼长链。模型字节/服务标签仍为显式合成，非真实AI权重或无线测量。

三个到达条件single_batch每车1请求frame0；queued_batch每车2请求同帧0；stream相同2请求frame0/240。未发布DAG对策略不可见；已发布queue可见。队列与当前任务都计原−.01等待费；新请求清理执行/input/output/版本拷贝，typed model cache与付费在途模型跨请求保留，session/reset/split间仍冷启动。空闲帧advance/收费，没有幽灵请求。

两model族条件：single base2048/adapter512；dual base2048/3072、adapter512/1024；capacity4096/8192，最小依赖闭包均可放下；其余compute/link/reward系数不改。每个scenario三规则共同sticky可达均衡选站、公共GRU/共享观测、当前必要资源保护；仅模型预取规则不同。本轮不做state prepare消融，不能宣称新预测迁移或学习surrogate收益。

2会话×3DAG层×3到达×2catalog×2容量=72scenario，各3规则=216。单批8任务与重复16任务不直接用总reward跨层排名；queued_batch与stream的任务集合/顺序相同，才是同总需求的到达对照。下面均为同scenario相对reactive的配对差。

72/72通过compute/IO/FIFO串行的乐观必要下界，**非充分可行性证明**：仍忽略coverage、model/state加载、局部队列/缓存抖动。实际部分scenario未全交付，不归咎于算法之前已知不可能完成的总量上界，也不声称调度一定可全完成。

## 3. 配对收益、完成与费用

每层24个匹配scenario，均保留正/负/零结果；不是24个独立原始时间窗口。没有CI/显著性或学习算法排序。

| DAG层 | 规则−reactive | 奖励差 | 完成请求差 | 节点差 | 模型链路帧差 | 奖励胜/负/平 |
| --- | --- | --- | --- | --- | --- | --- |
| short | naive_prefetch | 0.200895 | 0.000000 | 0.000000 | 5.083333 | 14/10/0 |
| short | gain_prefetch | 0.351480 | 0.000000 | 0.000000 | -1.708333 | 21/3/0 |
| medium | naive_prefetch | 0.892497 | 0.041667 | 0.708333 | 28.500000 | 12/12/0 |
| medium | gain_prefetch | 1.387206 | 0.083333 | 0.791667 | 0.458333 | 19/5/0 |
| long | naive_prefetch | 1.275728 | -0.250000 | 1.416667 | 41.666667 | 14/10/0 |
| long | gain_prefetch | 1.928182 | -0.083333 | 1.666667 | 3.041667 | 16/8/0 |

跨72个匹配scenario，gain_prefetch平均reward差 **1.222289**，56胜/16负/0平；平均完成请求差0，完成数4胜/5负/63平。naive平均reward差0.789707，40胜/32负，完成请求差−0.069444。平均增益不等于每种条件或双指标领先。

gain相对reactive在中任务平均多完成0.083333请求，但长任务少0.083333；长任务平均多交付1.666667节点。既有目标主要奖励节点交付，并不保证总节点上升时完整workflow数也上升。不能为制造完成率优势而在本轮改reward。

## 4. 预取是否被真正使用 / 是否有持续缓存价值

| 规则 | 未来对象准入 | 驻留期内计算使用 | 使用比例 | 使用前驱逐 | 窗末未用 |
| --- | --- | --- | --- | --- | --- |
| reactive | 0 | 0 | N/A | 0 | 0 |
| naive_prefetch | 2911 | 1170 | 40.19% | 1712 | 29 |
| gain_prefetch | 1079 | 1012 | 93.79% | 66 | 1 |

使用定义：本次准入的对象驻留世代，在同站被实际rsu_compute/output_ready用到，止于驱逐或会话结束；不是一般cache hit率、独立样本率或因果延迟节省。不同对象大小不同，不能用对象比例替代带宽费用。

naive尚未计算使用就被驱逐1712，gain为66；gain减少预取浪费，但“被用上”仍不证明团队净收益。旧SA1252/178是旧任务/旧policy的另一数据分布，不把它与本表直接相减冒称SA修复效果。

第一波计算使用后、同一驻留世代又被第二波请求使用的对象数：reactive **553**、naive **530**、gain **547**；只作为跨请求复用存在性，不按此绝对计数排名效率。

实际同站服务两种base的station-scenario对：reactive 67、naive 63、gain 62；base驱逐分别479/442/442，多base竞争不是只有catalog标签。车辆族分配/共同路由可能形成亲和，不声称每站每时刻都有多base竞争。

准入端实际拒绝危险可选动作：naive **9**、gain **3**，保护检查非空；raw可选当前依赖违规为0。保护限于本地实际当前声明需求，未来排队/后续模型外部性仍可能损害交付。

## 5. “大任务少导致预取失效”是否得到证明

持续请求确实提供缓存跨workflow复用；门控预取在三层都有正的平均绝对reward差。**但不能把旧失败唯一归到任务小，也不是任务越大越好。** 不同层原DAG、总节点和序列不同，绝对reward差会随任务规模改变。

| 层 | gain奖励差/提供节点 | 完成比例差 |
| --- | --- | --- |
| short | 0.004495 | 0.000000 |
| medium | 0.006165 | 0.005208 |
| long | 0.004129 | -0.005208 |

单位提供节点的平均gain reward差，medium最高、long没有高于short；归一化仍不能去除源DAG结构/请求时序混杂。long仅一个源job，不能据此推导一般任务尺度的因果规律。

## 6. 正负样例（事后诊断，非选择模型/场景）

全部72配对保留，以下为reward差两端：

- 正例c067（第二会话、long/queued_batch/dual/8192）：reactive reward400.650588、465节点、14/16；gain415.843573、477节点、15/16，模型帧94→98。增加12节点、1完整请求；不是靠免费模型加载。
- 负例c070（第二会话、long/stream/dual/4096）：reactive253.592521、313节点、8/16；gain238.116181、298节点、7/16，模型帧379→416。reward−15.476341；计算/IO乐观利用率0.55/0.50且FIFO必要界通过，不能简单用总量过载解释，也不能声称必要界通过即可全完成。

该负例说明当前启发式净收益/使用概率与团队完整轨迹仍有差距。保护当前模型只是约束，后续他者模型需求、换站/排队和执行优先级变化的机会成本尚未精确建模。未做逐动作反事实干预，不把损失唯一归给某一次预取。

## 7. 结论与下一阶段训练门槛

**环境/producer/consumer工程链通过；预取机会和持续复用确实存在；尚未证明SA优势或双指标领先。** 本轮三规则不是SA/MAPPO/cache DRL新训练结果；旧checkpoint/action控制明确拒绝直接用于stream，不拿新reward与旧48.544等跨合同比较。

下一独立实现应冻结：request-bound actor/记忆切换、已发布queue候选、异构catalog成本特征、空闲费用和动态请求团队tail return；SA/MAPPO/PPO/DQN与方向baseline统一消费，再按同一新训练预算/相同数据比较。保留predictor与共享状态；将已用/未用/驱逐和团队后续节点/完整交付的实际结果用于校准surrogate，不能只学命中率或把高使用比例当净收益。

优先处理c070这类已知负例的团队未来模型需求/排队外部性与信用；不直接增加DAG规模或训练量保证胜出。后续训练/独立开发会话和workflow pool应另立事前合同；long来源仅1、仅2原时间会话、合成成本、串行DAG、自动LRU、无真实到达校准仍未覆盖paper要求。formal/holdout sealed，canonical不变；论文证据E0_UNAVAILABLE/Unverifiable，novelty/顶刊评分N/S。

## 8. 验证与复核命令

解释器`/Users/howen/Projects/PPO_MEC/.venv/bin/python`。

- `python -m pytest tests/test_multi_request_stream_env.py tests/test_delivery_control_learning.py tests/test_tail_control_learning.py tests/test_scheduled_control_learning.py tests/test_coordinated_control_learning.py tests/test_parameterized_control_learning.py tests/test_multi_request_target_control_env.py tests/test_verify_joint_target_control.py tests/test_joint_model_cache.py tests/test_multi_request_joint_v2_execution.py tests/test_joint_experiment_launcher.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py tests/test_tail_mobility_support.py -q`：**411 passed**，`artifacts/joint_stream_v3_regression_tests.log`。
- `python scripts/verify_joint_scheduled_consumers.py`：11旧模型完整事件一致，`artifacts/joint_stream_v3_legacy_replay.log`。
- clean release `python scripts/run_joint_stream_workloads.py --smoke --output_root artifacts/joint_stream_v3_release` +对应audit：9单元/12500事件/2837收据/6重放通过。
- `python scripts/run_joint_stream_workloads.py` + `python scripts/audit_joint_stream_workloads.py artifacts/joint_stream_workloads_v3/20261015T172250Z_engineering`：216单元/416940事件/71757收据/6重放通过，`artifacts/joint_stream_v3_full_{command,audit}.log`；没有为写报告重复实验。
- `python artifacts/analyze_joint_stream_v3.py`：train时间边界/两session不重叠、配对分层/归一化、对象驻留使用与跨请求复用重建通过，输出本root `workload_analysis.json`。

- `python artifacts/verify_joint_stream_report.py`：12行分层/驻留/归一化表格、输入SHA、正负例及项目记录指针逐值核对通过，输出`report_validation.json`。

关键文件SHA256：manifest `832292df2d2ca84467bb4482ed765ccbcc41235cff48ef4cec24626c2e116c1e`；audit `a93865b7ef2a177c9d8478213ae7fcb3aede2db39ae6c5a7d88bbe87e8eb6fc7`；results `6fbcae0c20c82cd422456170dbca2d433f442e1632c7f3c9f3e6c89d1dbf0c19`。完整事件/收据SHA见manifest；报告表格逐值核验另见本root `report_validation.json`。
