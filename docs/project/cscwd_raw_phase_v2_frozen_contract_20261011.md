# CSCWD raw phase v2 冻结物理与开发矩阵合同（2026-10-11）

## 决策与版本

研究负责人已选择方案 B：**车辆在当前 RSU 接触内完成输入上传后，该 RSU 保留本节点执行权并继续计算；节点成功后才生成状态，经有成本的 RSU 间链路传到预测目标，目标恢复并向车辆交付后才算成功迁移**。此为显式 opt-in `raw_ngsim_event_time_v2`，不改 v1 的字节、输出或历史结果。实现配置见 `configs/experiment/cscwd_raw_phase_v2_development_20261011.json`；新能力不是实测无线/RSU 网络，也不预设 SA 收益。

## 拓扑、相位与提交

- 车辆位置/时间只来自冻结 NGSIM trace；RSU 三圆几何沿用 v1（轨迹首点锚定、y 轴 12 m 间隔、半径 10 m），属模拟覆盖。当前 RSU 与预测目标在每次决策起点由**已观测前缀**决定；隐藏未来仅用于执行时检验覆盖和交付，不传给 `_info()`、规则或 learner。
- 车辆→当前 RSU 输入上传使用既有 workload `input_bytes`、instance `actual_mbps` 和配置 `link.fixed_seconds`。上传必须在当前实际接触内完成；若提前离开，只计已耗时间与固定时延后的部分字节，节点不得开始，缓存/状态不提交。fallback 不需 RSU 接触，仍服从 trace 结束截断。
- 全相位串行且不双计时间：`input_upload` → action 0 的当前模型准备或 action 1/4 的目标模型准备 → 当前节点计算（含必要的前缀重算）→ action 4 在节点成功后才有的状态生成/传输 → target restore → target delivery wait。action 3 不预备模型；action 2 走车载 fallback。action 1/4 的目标模型，以及 action 0 的当前模型，从声明的 RSU repository 经模拟 backhaul 传输；完整传输和 load 后才改变 typed cache。当前节点计算在已上传输入后可由原 RSU 跨车辆接触持续执行。只有单一 `execution_owner_rsu_id`，一个节点最多完成一次。
- backhaul 复用既有配置的 `1000 Mbps` 默认和 `200 Mbps` 不利量级及 `0.02 s` 固定时延，但**不是测量**；每传输相位最多 10 s，超时或 trace 结束时按已传部分字节计成本，模型/状态不凭空提交。共享队列暂取 `0 s`，明确记为未校准模拟假设。model load 使用既有 object catalog；state restore 使用既有 `0.004412 s`；不得新增未校准的无限带宽或零成本云端通道。
- 节点成功前没有可传的状态。action 4 只有在节点完成、目标 bundle ready、状态全量经 backhaul 到达、restore 完成、车辆在预测目标覆盖内交付，才写 `prepared_state[target]` 并令 `migration_success=True`。计算时车辆可离开源站；目标在恢复时已离开则真实失败，尚未抵达则最多等待 10 s。预测目标错误、目标不出现、链路不足、trace 结束均写明确失败/待终止原因；不向末帧以后钳位或使用隐藏未来重选目标。
- 已完成的输入上传、模型/状态传输、load、compute、wait 各自保留真实消耗；未完成传输只记 partial bytes 而不创建 resident/state。目标模型可在节点服务失败后保留，但状态不会生成。完整迁移失败不能抹去已经完成的节点；下一决策依据真实 `completed`、cache 与 `prepared_state`，不得重复执行。
- 现有五动作不变：0 当前 bundle fill 并服务；1 预测目标 prefetch 并尝试当前服务；2 vehicle fallback；3 当前 steady offload；4 预测目标模型+状态准备并服务。任何 action 的 current-node 依赖和缓存容量规则均共享 v1 的 dependency-safe 逻辑。环境合法 mask 只表示语义动作允许，不伪装隐藏实际 contact 可行性；非法请求的 executed action 显式写出。

## 公开/诊断 schema

`_info().semantic_state.time_profile`、`calibrated_context.time_contract.schema_version` 均为 `raw_ngsim_event_time_v2`。新增公开 `calibrated_context.phase_contract`，只含相位顺序、input upload 的**估计**速率、backhaul 的**估计**速率/固定时延、10 s 两类 timeout、queue 假设、模拟来源标签；不含实际未来、目标抵达时刻或 native preview。`action_mask` 与五动作顺序不变，`run_metadata` 继续原接口。

`transition` 保留 `action`、`service_completed`、`migration_success`、`model_transfer_bytes`、`state_transfer_bytes`、`input_transfer_bytes`、`transfer_seconds`、`recompute_seconds`、`service_operation_seconds`、`step_cost_seconds`、`clock_seconds_after`、`reward`、`reward_components` 等既有消费者字段；新增 `phase_events`（相位/地点/start/end/计划与实耗秒数/计划与实耗字节/status）、`execution_owner_rsu_id`、`predicted_target_rsu_id`、`backhaul_transfer_bytes/seconds`、`partial_transfer_bytes`、`delivery_wait_seconds`、`migration_status`、`failure_reason` 和 `time_profile`。partial bytes 同时进入对应三类传输字段和 backhaul/input 分项；`total_transfer_bytes` 不重计。episode summary 新增分项累计和 pending/timeout/重复执行见证。

## 截断、验收和矩阵

trace 最后时间戳是严格上界；相位耗时未尽即停止在末帧、保留已耗时间/字节和完成状态，`truncated=True`，不隐式完成剩余 compute 或恢复。达到 `max_steps` 且 workflow 未完成同样截断；若节点完成而目标 delivery 待决，先按 10 s 等待/trace 结束给出 `delivered` 或 `pending_timeout`，不把 pending 标签当迁移成功。

先做真实 `reset/step` 正负链与 v1 source SHA 不变检查：输入未完、计算跨接触、模型到达但状态传输失败、目标离开/未到、trace 耗尽、重复交付/执行权、同公共前缀不同隐藏未来、每相位秒数/bytes/缓存与节点守恒。B 独立验收真实 `_info()`、estimator、规则、SA consumer/checkpoint profile 拒绝；失败不开训。

通过后仅用已冻结的两个 118.8 s development trace，固定 `dev_01` 一个工作负载、三个条件、v1/v2 两 profile、五固定方法，共 `2×1×3×2×5=60` episode，环境实际 step≤2,880、预览≤10,000。`dev_01` 在既有诊断中已暴露并出现可在接触内完成的准备阶段，故是**机制导向开发选择**，不能伪称结果盲或独立确认；`regression_04` 的大状态仅留作预先指定的负向合成/局部验收，不追加矩阵 cell。三个条件在产出前固定：默认 backhaul 1000 Mbps；实际 200/公开估计 1000 Mbps 的不利链路；1000 Mbps 且所有必需 bundle 初始 resident 的宽裕缓存。仍保留原 workload DAG/deadline/原始位置。资源宽裕组不能和默认组差值单独解释为模型预备因果；恢复 vs 重跑在同输入的 action 4/3 两固定策略上报告，同时单独给模型准备开关的**下一轮冻结清单**，本轮不偷扩第四条件。

全量报告 workflow completion/on-time、真正 state-ready/migration、重复执行、model/state/input/backhaul 与 partial bytes、load/recompute、每阶段时间/失败、资源成本、v1/v2 对照及全部负结果。两窗口并非独立 formal cluster；本轮不是模型真实调用、真实无线、论文主表或 paper-ready 证据。
