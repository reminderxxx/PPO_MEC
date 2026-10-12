# 四 RSU、八车竞争环境工程合同（2026-10-12）

状态：**engineering replay / 非正式论文证据**。版本 `vec_competition_v1`；不替换冻结的 Protocol 2.9、旧 `VecWorkflowCoreEnv`、checkpoint 或任何 formal/holdout。

## 实现范围

`scripts/run_vec_competition.py` 从本地 NGSIM CSV 选择同一 `Location`、同一连续 100 ms 时间窗内始终存在的八辆不同车辆。坐标与速度按 NGSIM 官方单位由英尺、英尺/秒转换为米、米/秒；`Global_Time` 作为毫秒时钟。只在同一源段选择车辆，不跨 Peachtree、Lankershim、I-80、US-101 拼接。窗口选择与数据 SHA-256 写入 `scenario_manifest.json`。

八个车辆分别绑定 Alibaba 八个不同 job 的 DAG 结构。四个 RSU 沿所选轨迹的主轴放置，覆盖半径根据该窗口的轨迹范围推定。场景记录所选八车的 `Section_ID/Lane_ID/Direction` 观测转移图。**这些是受控布站点，转移图也不是完整 GIS 道路图或真实运营商 RSU 位置。** 不将 LuST 仿真路网冒充 NGSIM 的观测路网。

每个 RSU 有容量、单独的 radio/backhaul 带宽、有限计算槽及计算速率。八车同时竞争；模型权重装载有非零传输时间且占用回传与缓存预留容量。任务须等前驱完成且前驱输出收据到达目标站；跨站通过显式 `state_migration` 传输。运行中的旧站任务继续完成，后续节点可在新站执行；最终结果经过回传和 radio downlink 才计为 delivered。`prepare(vehicle_id, target_rsu_id, node_id)` 是显式对象级未来节点准备入口。账本记录 association/handoff、cache load/eviction、transfer、compute 和 delivery。

## 当前真实性边界

- NGSIM 的位置和时间为真实观测轨迹；四站布置、无线覆盖、回传拓扑和带宽是推定或受控参数。当前未做地图匹配、遮挡/干扰、车道级道路拓扑或真实 RSU 地理坐标核验。
- Alibaba 提供 job/DAG 拓扑及任务字段，**不是原生车载 AI 工作流**。`alibaba_size_unit_mb` 是显式受控换算，不是原数据的实测字节。八个 job 分别赋给八车，二者没有观测到的联合分布。
- typed base/adapter catalog 的大小、初始驻留和兼容关系是 `controlled_validation_only`。没有真实 LoRA 权重加载、GPU 内存测量、推理质量或可恢复的 KV/tensor 状态。当前输出是可追踪的任务收据与大小元数据，不是模型推理输出。
- 当前脚本是工程场景 replay；旧 agent registry、PPO/MAPPO/SA 训练动作和正式 benchmark 尚未消费 `vec_competition_v1`。不得把其完成率和成本与 Protocol 2.9 表格并列或声称方法优势。
- `event_ledger.jsonl` 与 manifest 可重放同一场景；这不替代正式多源、独立时间段、多个 seed、强基线等统计证据。旧 holdout 边界不变。

## 论文最终态尚需关闭的阻断项

1. 选择可核验的 NGSIM 道路底图和真实 RSU 候选/部署点，完成坐标系、拓扑、无线传播和回传路径校准；当前仅有观测轨迹及虚拟布站。
2. 获取许可明确的真实 AI 任务数据、与其匹配的 base/adapter 权重、任务正确性标注、输入与中间状态，测量加载、计算、状态序列化、回传与无线成本；当前 Alibaba DAG 与模型目录是结构或受控参数。
3. 冻结新的多主体 observation/action/reward 合同，把该内核接入所有训练、评估、baseline、checkpoint 与 benchmark 消费端；旧五动作和此 replay 分离，不能单边改名为正式算法。
4. 重新预注册互斥的原始时间区间和车辆/任务划分，做同训练预算、同选择规则的强基线与机制消融；旧 v70 窗口身份错段、重叠和预算不公平不能沿用。
5. 输出可审计的正式 checkpoint、manifest、command log、逐 episode 原始结果以及独立 formal/holdout/support；再按 `top_journal_review_policy.md` 逐层审查 claim。

## 工程验收

`tests/test_vec_competition_env.py` 检查八车四站、共享资源、实际跨站状态传输、完工交付、确定性与冷加载不能同刻命中。真实 CSV 入口运行一次后检查 manifest、summary、ledger 和输入 hash。通过只表示工程 replay 可执行，不自动关闭以上阻断项。
