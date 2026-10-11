# CSCWD 原始时间与动作阶段：只读根因审计（2026-10-11）

- `reviewed_at`: `2026-10-11T00:54:20+08:00`
- `literature_cutoff`: `2026-10-11`；本轮未评文献 novelty。
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`；仅核对技术合同，不构成投稿评价。
- `artifact_run_id`: 父原件 `cscwd_raw_ngsim_event_time_20261010_v4`；只读诊断 `cscwd_raw_time_root_cause_20261011_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: 父原件运行代码 `0704937742ec9218094f380a1b541c4bd2159bcb`，诊断基线 `461527cb1174a730ed08b581de106c4341c2b4b3`。
- `evidence level`: 原始开发机制证据可复算；按顶刊政策的正式 paper-ready 档位仍为 `E1_DOCUMENTED / Unverifiable`，因为本剖面没有正式 checkpoint、formal/holdout/support 原件。评分 `N/S`。
- 总结：三窗口的**完整 workflow** 在当前 2.2 s 可执行原始时间内不可能完成，原因首先是冻结任务的计算时长下界；但“零 prepare”还受到新剖面整步接触门的额外保守限制，不能用来否定算法候选。决策 clone 使用未来真实接触，存在可复现的信息权限缺陷。此轮只诊断，**不改环境、参数、窗口或候选，不训练**。

## 1. Artifact 完整性与可重复性

输入只用已冻结 v4 的 `source_manifest.json`（SHA-256 `c0b64fd64c602ef5351c5686d2f315bbd5784e63ebbd08902b297fc5bacc883f`）、`summary.json`（`62c1f4c2c7fbb2e2763f6946c0fad29b4cd6a3705ebf01501a7d905d8b1fe30f`）及相同本地只读 NGSIM CSV（`ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`）。诊断入口 `scripts/diagnose_cscwd_raw_time_root_cause.py` 重新筛取**原三个区间内**的原始行，重新执行既有 30 个固定动作 episode/147 步，所有 episode 的步数、终止/截断、节点、失败、字节、奖励、动作计数与 v4 **零差异**；另做 56 次 native 阶段预览与一个合成未来后缀反例，新增策略 episode 为 0。逐步状态/时钟/字节守恒检查通过，既有原件、checkpoint 和原始 CSV 均未写入。

机器原件 `artifacts/analysis/cscwd_raw_time_root_cause_20261011_v2/diagnosis.json`，SHA-256 `2aba8413253a1a0be1868b62c639f9f7754f76293b4c78aed5109444ad6756fc`，含三窗口首状态 13 个合法动作的 actual/estimated-link 阶段表、30 条 v4 原始时间决策的门槛账本和 future-clone 反例；不含原始坐标。没有读取 holdout 性能，没有选择新窗口或训练模型。

## 2. 原始时间、车辆与单位

| 固定窗口 | source segment / `Global_Time` 闭区间 | 选定车辆 / `Frame_ID` | 采样与首决策剩余 | 首决策实际接触 / deadline | 旧合成序列名义长度 |
| --- | --- | --- | --- | --- | --- |
| `dev_01` | `us_101` / 1118846994600–1118846996900 | 首帧最小 ID 2 / 157–180 | 24 帧、23×0.1 s；2.2 s | 0.728088 / 99.164509 s | 16×5=80 s |
| `regression_00` | `lankershim` / 1118936085200–1118936087500 | 首帧最小 ID 456 / 4053–4076 | 24 帧、23×0.1 s；2.2 s | 2.2 / 68.178196 s | 10×5=50 s |
| `regression_05` | `us_101` / 1118847021000–1118847023300 | 首帧最小 ID 2 / 421–444 | 24 帧、23×0.1 s；2.2 s | 0.390382 / 43.578451 s | 10×5=50 s |

三段 `Global_Time` 都为毫秒，相邻差 100，所选车辆 `Frame_ID` 都严格 +1；`time_index_start/end` 与原始时间吻合、无缺帧/重复车辆帧。父 manifest 绑定的 v28 `train_window_plan.json` SHA-256 `1bc37e452f41f3b8233daf5f88c56e0d15869977c2beea02f25e6cd29a11090a` 和 `dev_window_plan.json` SHA-256 `5efcb9068bbbd712e0f7bf9a1af2c4f910f0cad145f90b5ed6f77daa62aabcfa` 均匹配本地文件：`dev_01` 是**train 原计划 index 3 留出的内部 dev 实例**，`regression_00/05` 来自 dev 原计划 index 0/5；三者在 v3 的 36 个实例中均无同 segment 原始时间重叠。这不授予独立 formal/holdout 身份。`frame_offset` 是 window plan 在按 `(segment,time)` 排序后帧列表中的索引，**不是车辆 `Frame_ID`，也不是秒**。首帧仅供速度历史，首决策在第二帧 `t=0.1 s`，末帧 `t=2.3 s`。已验证区间内未发现 ms/s 或 frame/time 错配；Peachtree 10 位时钟未进入本轮。最小首帧车辆选择是冻结规则，`regression_00` 的所选车辆在该短区间内静止，导致无预测目标、action1/4 mask 关闭；并非按算法结果换车。是否有**更长且不进入封存区间**的同车开发轨迹，本轮未判定；不以现有原始文件全量扫描挑选新 live 窗口。

旧 `rsu_sequence` 由 `freeze_calibrated_continuous_workflow_pilot.py` 按节点数和 NGSIM 窗口的 `estimated_handoff_count>0` 转为 high/low pressure，再用固定 block 长度重复三 RSU；`decision_step_index` 每动作前进一个元素，失败也前进。配置 `mobility_abstraction.decision_step_seconds=5.0` 明标 `artificial_time_scale_applied_to_trace_derived_handoff_pressure`；此前 `synthetic_elapsed_5s_atomic_v1` 另将一个元素视为 5 s，并在末尾钳位。80/50 s 与原始 2.3 s 无时间单位上的等价性；它们是不同合同，不是原始 NGSIM 接触时长。

链路公式为 `bytes × 8 / (Mbps × 1,000,000) + 0.02 s`（仅非零流付固定项），单位为**十进制 Mb/s**，不是 MB/s/MiB/s；对象容量和传输字段以 byte 储存，奖励的 `/1024^3` 仅用于 GiB 惩罚。`dev_01` 实际 200 Mb/s、公共估计 1000 Mb/s，另两窗均 1000 Mb/s；例如 1,000,000 bytes @1000 Mb/s 的网络段是 0.028 s。`node_compute_seconds` 来自本地参考 3.807234 s 与 Alibaba `duration_vec/median` 的截断比例（0.4–2.5）相乘，不直接把 `trace_duration_raw` 当秒；`fallback_seconds=8`、失败等待目标 2 s，恢复开销 0.004412 s。当前证据未发现这些转换实现的单位 bug；绝对耗时是否代表真实路侧服务仍未验证。

## 3. 首状态各合法动作的阶段账本

列内 `M`=模型网络+加载，`S`=状态网络+恢复，`I`=输入网络，`C`=当前节点计算，`F`=车辆 fallback，`X`=失败等待；`estimate/actual` 是 native step 在公共估计链路速率/实际链路速率下的总耗时，**估计 clone 在本剖面仍错误使用真实未来接触**（见第 5 节），所以不把估计列称为无泄漏决策。数字为秒，四舍五入至 0.001；`gate` 是新剖面的整步准入结果，`first` 按先到的物理边界判定。未列的阶段为零；完整精度及准备字节见机器原件。

| 窗口 | 合法 action | estimate / actual | M | S | I | C | F | X | gate / first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dev_01` | 0 | 1.523 / 1.523 | 0 | 0 | 0 | 1.523 | 0 | 0 | contact / contact |
|  | 1 | 1.909 / 2.217 | 0.694 | 0 | 0 | 1.523 | 0 | 0 | trace / contact |
|  | 2 | 9.544 / 9.551 | 0 | 0 | 0.028 | 1.523 | 8 | 0 | trace / trace |
|  | 3 | 1.523 / 1.523 | 0 | 0 | 0 | 1.523 | 0 | 0 | contact / contact |
|  | 4 | 1.934 / 2.247 | 0.694 | 0.030 | 0 | 1.523 | 0 | 0 | trace / contact |
| `regression_00` | 0 | 1.523 / 1.523 | 0 | 0 | 0 | 1.523 | 0 | 0 | admitted / none |
|  | 2 | 9.544 / 9.544 | 0 | 0 | 0.022 | 1.523 | 8 | 0 | trace / trace |
|  | 3 | 1.523 / 1.523 | 0 | 0 | 0 | 1.523 | 0 | 0 | admitted / none |
| `regression_05` | 0 | 4.193 / 4.193 | 0.386 | 0 | 0 | 3.807 | 0 | 0 | trace / contact |
|  | 1 | 2.386 / 2.386 | 0.386 | 0 | 0 | 0 | 0 | 2 | trace / contact |
|  | 2 | 11.830 / 11.830 | 0 | 0 | 0.023 | 3.807 | 8 | 0 | trace / trace |
|  | 3 | 2.000 / 2.000 | 0 | 0 | 0 | 0 | 0 | 2 | contact / contact |
|  | 4 | 2.000 / 2.000 | 0 | 0 | 0 | 0 | 0 | 2 | contact / contact |

Native action1 是“目标 bundle 准备 + 当前节点服务”；action4 是“先判定目标 bundle/state 准备预算，**当前节点完成后**才记状态传输字节/恢复并提交 prepared state”。代码的 contact 测试用 prospective state 费用，但实际账本和 commit 在计算之后；目标模型准备、当前计算、状态提交的执行位置并非同一无线占用相位。Native 对 action1 只以**目标模型网络+加载**比较当前接触，对 action4 只以**目标模型网络+加载+状态网络+恢复**比较当前接触；当前节点计算和最终总时间未由该物理门限制。新 raw profile 额外要求**整个 native step 总时长**（含本地计算、fallback/失败等待、输入/状态传输）同时不超过原始窗口剩余和当前 RSU 接触；action2 仅受窗口剩余限制。这是明确的保守增强，不能解释成生产原合同已证明整步全需占用无线。

具体阶段反例：`dev_01` 的 action1 模型准备 **0.694262 s**、action4 含状态准备 **0.724293 s**，都小于当前实际接触 **0.728088 s**；但加上当前节点计算后总计 **2.217156/2.247187 s**，被新整步门回滚。`regression_05` action1 目标模型准备 **0.385720 s** 也小于接触 **0.390382 s**，但 native 当前服务因 bundle 缺失要付 2 s 失败等待，总计 2.385720 s；新门把原本可提交的目标准备一并回滚。这证明“v4 零准备字节”不能单独归因于原始接触无法容纳准备，也不能判算法未学会准备。现有代码没有异步/部分传输/跨界远端服务相位；这些反例**不证明**应放宽为无线可行的成功 service。

## 4. 严格可达性下界与失败/截断语义

冻结 DAG 每个节点只能顺序完成一次；任一合法完整 workflow 的时长至少是所有 `compute_seconds` 之和，忽略全部正的模型、输入、状态、fallback、recompute、失败代价，故这是乐观严格下界：`dev_01=38.739782 s`、`regression_00=24.968087 s`、`regression_05=19.138455 s`。三者均大于首决策至末帧 2.2 s（甚至大于全部 2.3 s），所以**在现有“末帧截断且无后续窗口延续”的合同内，无论策略和整步接触门如何，完整 workflow 均不可能完成**。这只定位工作负载与短窗口失配；更长真实连续轨迹、跨窗口恢复或不同运行任务的可达性均为 `UNKNOWN`，本轮不得由固定策略失败外推。

30 条 raw 决策中 10 次请求受 action mask 改写，2 次完整 native step 准入并服务成功，28 次被 raw 门拒绝。v4 的 `if` 顺序先查 trace，故原报告为 trace 24 / contact 4；逐步复算发现 **8 次两者同时超限**。按“先到的物理边界”是 trace 20 / contact 8 / 无超限 2；不能把 24 与 4 当作互斥因果计数。28 次拒绝均零三类传输字节、缓存和 prepared-state 不变、DAG 不进展；每次时钟增加 `min(2 s, 剩余原始时间)`，合计 29.954212 s（跨全部固定动作 episode），无零时间加速。到末帧返回 `truncated=True, terminated=False`；这为 value bootstrap 提供正确边界标志，但本轮**未运行 learner/GAE 消费端**，不宣称端到端 bootstrap 已验。`failed_service_attempt_seconds_proxy` 在新拒绝路径累积实际截断等待，在 native 路径则按失败次数×2 s 重写；跨 profile 或混合路径同名指标语义不完全一致，后续修复应另立任务，不改本轮科学原件。

## 5. 公共决策 clone 的未来接触泄漏

`RawNGSIMEventTimeEnv` 继承 `clone_for_decision_model()`，该方法只置 `_decision_model_mode=True`；但 raw 类 `_physical_contact_budget_seconds()` 和整步 `step()` 仍无条件读取实际未来轨迹。独立合成反例保持公开首两帧、RSU 几何、DAG、配置及时间终点完全相同，仅改变后缀：两个初始 public state/行动 mask 完全一致，调用 `clone_for_decision_model().step(3)` 后，一边完成当前节点，一边因接触提前结束拒绝；真实接触预算分别为 2.2 和 0.029384 s。固定动作 v4 原件未使用对外 two-step 策略，因此这个缺陷不推翻其 147 步重放；**任何在此 raw profile 上用该 clone 做 two-step/public action 估计的比较都会越过因果权限**。这是独立的、须在后续实现任务处理的 blocker，不以事后调参或换窗口消除。

## 6. 审查归类与停止门

| 类别 | 本轮判定 | 边界 |
| --- | --- | --- |
| 数据不足 | **确认**：固定 24 帧/2.2 s 可执行区间短于三 DAG 的 compute-only 严格下界 | 不声称原始文件没有其他连续开发轨迹；未筛新窗口 |
| 时间/传输单位 bug | 三已消费区间**未见** ms/s、frame/time 或 byte/Mb/s 转换错误 | Peachtree、真实无线速率及测量外推未核验 |
| 动作合同过度保守 | **确认**：raw 整步 contact gate 强于 native prepare contact gate；有准备阶段可行、整步被回滚的具体见证 | 不直接断言放宽后的 service 物理可行 |
| 工作负载失配 | **确认**：19.14–38.74 s 计算下界对 2.2 s 轨迹，不依赖算法 | 其他任务/更长窗口 UNKNOWN |
| 因果预览权限 | **确认缺陷**：决策 clone 仍读未来真实接触 | 禁止在修复前把 two-step/raw 预览结果作公平基线 |

此轮停在只读缺陷/限制报告；不更新论文主表、不固定 SA 算法贡献、不扩窗口、不训练或晋级候选。若后续实施，应先独立修公共决策 clone 与阶段接触语义并做同前缀后缀扰动验收，再另立不看结果的长时段开发数据合同；两项属于**后续任务**，本报告不替其选策略或窗口。
