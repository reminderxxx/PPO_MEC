# CSCWD 原始 development 窗口资格冻结方案（2026-10-11）

状态：**扫描原始连续性前冻结**。仅做 source/time/vehicle 与静态成本资格，不跑算法、不选择 checkpoint、不改变 `raw_ngsim_event_time_v1` 的几何、整步准入、成本、奖励、mask 或轨迹末端截断。先前 2.2 s 失败和方法表现不作为筛选条件。

## 输入身份与授权边界

- 父工作负载为已提交 `configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json`（SHA-256 `b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc`），只消费其中静态节点/链路/对象成本；原始 NGSIM CSV 身份仍须为 SHA-256 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`。
- 本轮**唯一已授权 source interval 集合**是 v28 strict split 的非 sealed `train_window_plan.json` 和 `dev_window_plan.json` 中的 40 个原始闭区间（各 20 个）。`formal` 和 `hidden_holdout` 只从 `split_manifest.json` 的 `plans.*.window_ids` 读取区间元数据作冲突排除，绝不打开对应结果或 sealed plan 文件。没有明示授权的其余 NGSIM 行，即使同车连续，也不得自行归入 development。
- 对每个候选保留 `plan split/ID/source segment/Global_Time/frame_offset/length` 和拒绝理由。`Global_Time` 差 100 表示 0.1 s，选第一时间戳的最小数值 `Vehicle_ID`，要求该车每帧唯一、`Frame_ID` 连续 +1；不跨窗口拼接，不钳位末端。Peachtree 10 位时间与其它段的 13 位时间单位身份不一致，在另行核验前标 `unit_ambiguous` 并排除。
- 按 `(source_segment_id, time_index_start, window_id)` 固定排序，同段同车或相邻窗口不能充作独立 cluster；若以后有合格者，最多 12 个，未通过者仍保留。来源/成本/接触资格先于任何策略运行；不基于 SA/PPO/规则结果增减窗口。

## 一次冻结的时间门槛

**必要下界**：36 个既有开发 workload 的 `sum(node.compute_seconds)` 范围为 16.298588–52.032198 s；首帧仅作历史，故即使零传输、零失败，窗口至少要有 `0.1 + 16.298588` s 才可能让其中最短 DAG 完成。该条件不保证可服务或迁移。

**工程资格目标**（只用于提议未来 source/时段，不声称真实测量）：`T_target = ceil_0.1[1.10 × (C_max + B_max + I_max + S_normal_max + F) + 0.1] = 118.8 s`，即至少 **1,189 个连续 100 ms 帧**，其中：

| 静态项 | 秒 | 来源/含义 |
| --- | ---: | --- |
| `C_max` | 52.032198 | 36 个开发 workload 最大计算和 |
| `B_max` | 47.790783 | 最大单次冷 bundle 在现有最慢 `actual_mbps=200` 下的网络+加载 |
| `I_max` | 0.043131 | 最大单节点输入的网络秒数 |
| `S_normal_max` | 0.035650 | `state_scale≤64` 最大单包状态网络+恢复 |
| `F` | 8.000000 | 已配置的单次车辆 fallback |
| 10% 与首帧余量 | 10.898238 | 工程敏感性假设，非 NGSIM 测量 |

`state_scale=1,000,000` 的最大状态包可达 3,292,500,000 bytes，在 200 Mb/s 下单次网络+恢复为约 131.724 s；本目标**不保证**这种极端状态迁移可行，必须单独作为接触不足控制记录。`118.8 s` 也不保证所有 DAG/缓存策略可完成，只是一个结果盲的统一开发来源资格目标；不因后续实验失败逐次放大。当前冻结的 24 帧窗口若不足，不降低真实成本、缩短 DAG 或放松整步门去制造成功。

## 执行门与输出

先仅验证 40 个授权区间的原始帧时间、选定车辆连续性、与 `split_manifest` formal/hidden 区间的重叠，以及窗口内可用时长。输出只含身份/计数/理由、不含原始坐标或性能。任何候选都必须同时满足来源合法、单位一致、所选车连续、与 formal/hidden 不重叠及 `T_target`；只要授权区间最长连续时长低于最短 DAG 的必要下界，就可提前停止可达性 episode，资格结果为 `NO_QUALIFIED_DEVELOPMENT_INTERVAL`。不得把未经授权的相邻/同车原始行自动延长进窗口。只有至少一个符合此门的窗口且可构成预先指定的资源宽裕/竞争与接触充足/不足控制时，另立一次不超过 12 实例、60 episode/1,440 实际步/5,000 preview 的固定公共合法规则可达性检查；未来真实路由只能标离线资格参考，不能进 actor 标签。

若不通过，仅交付机器资格清单、最小缺口与供负责人另行批准的新 development 原始区间方案；本轮不创建新 split，不使用未授权范围、不启动 B 条件训练。
