# CSCWD 新原始 development 来源资格事前冻结（2026-10-11）

状态：**原始全量扫描前冻结**。负责人已授权审查本机既有 NGSIM 中此前未分配的较长 development 来源；本方案只定义结果盲来源资格与至多一次固定公共规则可达性检查。不改旧 split、不读取 formal/holdout/hidden 性能或 checkpoint、不改论文。原 `raw_ngsim_event_time_v1` 的几何、cost、reward、mask、整步门及末端截断保持原样。

## 固定输入与排除边界

- 原始 CSV SHA-256 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`；静态 NGSIM+Alibaba workload manifest SHA-256 `b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc`。仅纳入 `us_101`、`lankershim` 的 13 位毫秒 `Global_Time` 和 `100 ms` 原始采样；Peachtree 的 10 位时间单位未独立核验，I-80 属另一协议，均不参与本轮。
- 精确原始区间从 v27/v28/v71 strict split 的 `split_manifest.json` 中 `plans.*.window_ids` 提取；三个文件 SHA-256 依次为 `6e724186cfb18032b27e4f1ef44ce949e2b9298a78de24893da293ca50f41cd3`、`246fcacebb04051e5181a1b2eacf702e942532bbf69e5a4c2e6d1194e98ee322`、`2110347990464f12532a73886f6f617e461a18b4e781d940da2172b6ca3eeff5`。所有 `formal`/`hidden_holdout` 闭区间禁用；新窗口与其最近边界必须相隔至少 **24 个完整空白帧**，即相邻端点 `Global_Time` 至少差 2,500 ms。train/dev 区间仅用于标记历史暴露，不作为正式独立测试集。
- v8 strict split 与 v17/v20 的四个 future-validation manifest 使用 `window_off…t…` 匿名索引，不能从 ID 本身证明原始 segment/time。其 `max_mobility_rows` 分别不超过 10,000/50,000/200,000；本轮将原 CSV **前 200,000 数据行**在每个纳入 segment 中出现的全部原始时间范围连同上述 24 帧 embargo 整体禁用，绝不猜匿名窗口为 unused。v8 split manifest SHA-256 `c974a18443d4b6600c765bc71af486308b2fbe0f0fe6be4e094abbd37937f53a`；v17/v20 四个来源 manifest SHA-256 依次为 `449ed33ff52e8d760601c91dd7421091a4e99bf472abe91f1a93766503cfe41c`、`f0be3b4d48f9f243a233e891c10965902ed51b98e605d2e40262140d7ac84d3a`、`fe15b77e9dd4337ee259aca47ac0369b609ff92b25f5d319064d5e5cea7e34ea`、`61648d88fa295649025adf26f573ec80db738c7c1e3dd3ccf9dce7f61e37d4db`。Peachtree-only v89 与 I-80 typed-cache 范围不纳入本轮。
- 不读上述 sealed window-plan 文件或任何 sealed/performance 结果。若出现未能定位原始时间的其他历史封存/禁用来源，相关 segment/time 一律先隔离并标 `unresolved_provenance`，不当作未用数据。

## 时间、车辆与选择规则

- 36 个现有 workload 的最短/最长纯计算和为 16.298588/52.032198 s；之前一次冻结的静态工程目标 `1.10×(最大计算和 52.032198 + 最大单冷 bundle 47.79078256 + 单输入 0.04313084 + 正常状态单包 0.0356504 + 一次 fallback 8)+首帧 0.1` 向上取 0.1 s 为 **118.8 s**。本轮固定 **1,189 个 100 ms 连续帧**为来源预筛长度；它只是给多阶段工作流留出工程时间的资格门，不能证明任一算法可完成，极端 `state_scale=1,000,000` 单状态包约 131.724 s 另作接触不足控制。不因算法结果缩短/延长时长、改变成本或重复末帧。
- 以 `(Location 规范段, 数值 Vehicle_ID)` 为车身份；原始 CSV 全量按该键、`Global_Time`、`Frame_ID` 稳定排序。仅连续时间 `+100 ms`、车辆 `Frame_ID +1`、单一车帧、坐标有限的原始行组成 maximal run；任何缺帧、重复、单位异常或坐标异常都切断 run 并记录理由。先从 run 扣除完整禁用区间及 embargo，得到合法 fragment；每个 fragment 只取**最早**的固定 1,189 帧作为一个候选，余下同 fragment 后缀不再供本轮试选。不拼接片段、车辆或 segment。
- 候选排序固定为 `(source_segment_id, Global_Time_start, Vehicle_ID)`；先最多取 `lankershim` 1 个，再取 `us_101` 2 个，均要求同段新候选彼此至少相隔 24 个完整空白帧。某段不足时不从另一段补额。相同 source/time 只留数值最小 `Vehicle_ID`；所有合格/拒绝 fragment 的计数、身份、原始时间与理由保存在 create-only 机器清单。不看任何策略结果筛候选；随机种子 `0` 仅记录，排序不用随机数。
- 所选窗口与 v27/v28/v71 train/dev 原始区间的重叠数量及最小边界距离逐一记录，若重叠只标为**历史训练已暴露 development**，不宣称新独立测试或统计 cluster。formal/hidden/匿名隔离区任一冲突为硬拒绝。

## 固定公共可达性门

只有出现至少一个合格来源，才把所选每个窗口分别绑定四个冻结 workload `regression_05`（`all_ready`, ample, 1000 Mb/s）、`dev_00`（tight, 1000 Mb/s）、`dev_01`（ample, 200 Mb/s）与 `regression_04`（tight, 极端 state package, 200 Mb/s），最多 3×4=12 实例。这是静态资源和成本控制，不代表真实无线链路；初始 raw 接触长短按冻结几何/前两帧因果速度单独分层，缺少接触充足/不足或两类资源控制时如实报缺口，不补挑窗口。

五个预定方法是合法动作约束的 `fallback`、`serve`、`causal_public_immediate_rule`、`causal_public_two_step_rule` 与固定 `prepare_if_legal_else_serve`。每实例每方法至多一次，共不超过 **60 episode / 1,440 实际 step / 5,000 public preview**；动作若不在 mask 中，按预定 fallback 次序选择合法动作并记录请求/执行差异。离线真实未来 route/contact 只可作为注明 privileged 的上界诊断，不能进入 actor 标签、规则输入或公平排名。逐例保存完整 workflow、prepare/serve、成功/失败、source 时长、接触和计算/模型/状态/输入/失败阶段成本；若仍不可达，区分数据长度、几何/整步门和工作负载成本，不自动放宽门禁。

本次结果一律标 `development feasibility`。B 的四方法×五 seed、23,040 step 一次 pilot 只有来源、公共规则可达性与 B 的 public planner future-perturbation/真实 raw `_info()` 两门均独立通过后才可启动；A 不训练。旧 `clone_for_decision_model().step()` 的真实未来接触权限仍为 blocker，不能当公平公共规则。
