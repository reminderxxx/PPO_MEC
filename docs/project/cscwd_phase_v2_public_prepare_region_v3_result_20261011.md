# CSCWD phase-v2 公开 prepare 区域 v3：单次有界结果与缺口审查

- `reviewed_at`: 2026-10-11
- `literature_cutoff`: 2026-10-11（沿用同日既有公开近邻表；本轮未新增文献或 novelty 结论）
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `cscwd_phase_v2_public_prepare_regions_20261011_v3`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`（执行源码）: `0cade5cb19dcf8e8c1b169c6b08deb1c422455e3`（启动前已 push）
- `evidence_level`: **独立核对的有界 development 原件**；正式论文所需 formal/holdout/support 不存在，TMC-ready 证据仍不足
- `verdict`: **局部正例 1、无益高成本 6、截断不确定 4；四项论文贡献均 `UNVERIFIED`，paper-ready=`Unverifiable`**

## Artifact 与 protocol 门

原事前方案 SHA-256 `472808fc90a3af16f8fe2c4f76acb69a9921a70f50d02184baa1eb40ec9c1def`；v3 授权 manifest SHA-256 `56a417b84c4823ab0cea93ccf265582d88d870d157ba05fde3e87497e25ac36e`。Python 3.9.6 零环境步输入/选择预检 SHA-256 `76ba658bcc045fa0ab3d8089ba6351f062c83b1ae3aa944ca061957f323c22b8`。唯一 create-only root 为 `artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_v3/`；`completion_receipt.json` SHA-256 `20dd2ab1b046f7710ab9c78a3057f438074fc4710481cfeecaa3dbdf945be98c`，`summary.json` `c5591db1c39f630b7d1d87856daba0a8698aad9b3d3c6867dc9793effa79dff5`，`artifact_integrity.json` `5928581d59ab89401157b4bb379cde7e21fe578897c924fcbf48466c177f60a4`。

独立只读核对输出 `artifacts/analysis/cscwd_phase_v2_public_prepare_regions_v3_audit_20261011_v1/audit.json`，SHA-256 `1eda52da0777b3786916f05e1d3af2b097af0414fa2bf0963ae54abedadae10f`：**43/43 manifest 文件 SHA/size、11/11 前缀自校验、27/27 分支自校验、逐事件相位/字节/时钟/合法动作、指标、迁移事后归因及 11/11 配对差值/区域分类全部通过**。24 个预设格仍是原 11 入选/13 排除；全部入选来自同一 US101 车辆 1183、Frame 5368–6556、Global_Time 1118849293800–1118849412600 原始轨迹，3 格是既有 `[4]` 回归，8 格是新增覆盖状态。Lankershim 没有 action3/action4 同合法格。11 个 full state 均不同；9 个公开观测组，其中 2 组各含 2 个不同 full state，因此 v2 的全局公开 hash 去重门确实会误把不同模拟状态视作同一状态。8 个 prefix `env.step` + 199 个比较 `env.step` = **207/288**，27/36 分支；0 训练/模型调用/formal/holdout。

## 全部 11 个配对格

`C/T/F/X` 为完成/按期/服务失败次数/达到 continuation 截断；`Δ` 为 action4 − action3，字节含输入、模型、状态，秒是冻结模拟器时间。以下格是**相关开发状态**，不得按 11 个独立样本计算置信区间。

| 条件 | 前缀 | 来源 | 原首动作 | 区域 | action3 C/T/F/X | action4 C/T/F/X | Δ总字节 | Δ耗时 s | Δ重算 s | action4 首步迁移/后续目标计算 |
|---|---|---|---:|---|---|---|---:|---:|---:|---|
| default | `[]` | 新覆盖 | 4 | 截断不确定 | 0/0/2/1 | 1/1/0/0 | +9,213,791 | +23.425 | 0 | delivered / 是 |
| default | `[0]` | 新覆盖 | 0 | 截断不确定 | 0/0/2/1 | 1/1/1/0 | +9,266,068 | +12.205 | −4.316 | current_service_failed / 否 |
| default | `[1]` | 新覆盖 | 1 | 无益高成本 | 1/1/0/0 | 1/1/0/0 | +9,782,424 | +0.411 | 0 | target_departed_before_delivery / 否 |
| default | `[4]` | 旧回归 | 4 | 无益高成本 | 1/1/0/0 | 1/1/0/0 | +9,782,424 | +0.411 | 0 | target_departed_before_delivery / 否 |
| adverse_link | `[]` | 新覆盖 | 4 | 截断不确定 | 0/0/2/1 | 1/1/0/0 | +9,213,791 | +23.738 | 0 | delivered / 是 |
| adverse_link | `[0]` | 新覆盖 | 0 | 截断不确定 | 0/0/2/1 | 1/1/1/0 | +9,266,068 | +12.513 | −4.316 | current_service_failed / 否 |
| adverse_link | `[1]` | 新覆盖 | 1 | 无益高成本 | 1/1/0/0 | 1/1/0/0 | +9,782,424 | +0.724 | 0 | target_departed_before_delivery / 否 |
| adverse_link | `[4]` | 旧回归 | 4 | 无益高成本 | 1/1/0/0 | 1/1/0/0 | +9,782,424 | +0.724 | 0 | target_departed_before_delivery / 否 |
| ample_cache | `[]` | 新覆盖 | 4 | **正向机制/资源取舍** | 1/1/0/0 | 1/1/0/0 | **+140,480** | **−4.290** | **−4.316** | delivered / 是 |
| ample_cache | `[0]` | 新覆盖 | 0 | 无益高成本 | 1/1/0/0 | 1/1/0/0 | +140,480 | +0.026 | 0 | target_departed_before_delivery / 否 |
| ample_cache | `[4]` | 旧回归 | 4 | 无益高成本 | 1/1/0/0 | 1/1/0/0 | +140,480 | +0.026 | 0 | target_departed_before_delivery / 否 |

唯一正向格 `us_101:ample_cache:reset` 的两臂均完成、按期且无服务失败。action4 首步 state 交付成功，后续目标 RSU 有计算且该步不重算；相对 action3，重算由 4.315814 s 降至 0，耗时由 91.321613 s 降至 87.031334 s，同时状态/backhaul 多 140,480 bytes，总传输由 2,650,408 增至 2,790,888 bytes。它是**模拟成本下的一例机制成本取舍**，不是净收益或普遍规律。四个截断格中 action3 达 continuation cap，而 action4 完成；由于后续长度不对称，冻结规则仍标 `UNCERTAIN_CENSORED`，不得把其方向数值当确认性服务优势。六个无益格中三格属于旧回归，同源/同状态结构相关性不因条件数变成独立样本。

**在线可观测性**：`ample_cache` 的缓存状态与公开 mask/估计信息可见，真实目标交付、未来是否离开和后续目标计算仅在环境事后可知。default/adverse 的 reset 与 `[0]` 各有相同公开观测、不同 full state；其中实际 backhaul 条件不可由该公开 hash 区分。一个正向格加六个负格不足以冻结安全在线 gate，不把事后 `target_departed_before_delivery` 送入策略。

## 四项 `UNVERIFIED` 的最短缺口

| 待证主张 | 本轮证据/当前代码检查 | 最短可审查补证（本轮未执行） |
|---|---|---|
| 真实跨 RSU 净收益 | NGSIM 只提供车辆轨迹；`raw_ngsim_phase_v2_env.py` 的 RSU 几何和 backhaul 为模拟假设，`cscwd_raw_phase_v2_development_20261011.json` 明示 `queue_seconds=0`、200/1000 Mbps 假设。仓库已检查的 artifact/项目记录未见独立 RSU 主机或实测链路原件；本轮未扫描网络、用凭证或部署。 | 先取得至少两台独立 RSU 主机和链路的可审计时钟/带宽/时延/丢包/队列测量及真实模型/状态 payload；在相同 workflow 与接触条件下配对连续执行和合法重算，完整计传输、失败、完成、时延与能耗。单机多进程不算真实 RSU。 |
| typed base sharing × state migration 因果交互 | `VecWorkflowCoreEnv` 有 opt-in `workflow_state_migration.enabled`，但 `no_base_sharing` 在 formal execution 表中标为 partially available/不可执行；当前 episode reset 恢复 cache template，无跨 workflow 持久共享。`ample_cache` 改的是容量/初始 resident，**不是** base-sharing 开关。故现有公共实现不能直接做不混杂 2×2。 | 先补可执行、对称计费的 sharing OFF 与 migration OFF 合同，并在同一 DAG/任务、trace、catalog、容量、初始资源、动作权限、固定策略及预算下跑四臂 `(S,M)∈{0,1}²`。OFF-M 仍须保留合法 action4/重算路径；按每独立来源报告交互 `I=(Y11−Y10)−(Y01−Y00)`，完成率取正、耗时/字节分别报告原符号，不用 reward 加权掩盖代价；同时验证两因素机会与公平性。 |
| 三头控制优于公平基线 | v3 仅是同状态原生动作分支，不训练或评价 SA/PPO/MAPPO/强规则的匹配策略；此前 B pilot 开发结果 `MIXED` 且外层 terminal `FAIL`。 | 在统一 observation/action/mask、预测权限、物理/成本合同、训练交互数、checkpoint 选择及固定公平强规则能力下做匹配训练和多窗口/seed 对照；三头单因素消融、机制 endpoint 与失败率同报，先 development 冻结，再独立确认。 |
| 独立泛化与 paper-ready | 11 格都在同一 US101 开发轨迹，两个公开观测碰撞组亦非独立来源；无新 formal/holdout/support、独立 host/link 或多 workflow/adapter 原件。 | 先按原始 frame/time interval 与 vehicle recurrence 确认独立新来源、多个 workflow/adapter 机会和预注册统计单元；冻结方法/checkpoint/指标后再做 formal、独立 holdout、support/robustness，窗口外层 bootstrap/BCa、paired sign 与 Holm，完整 manifest/command log。不得重开已消费 holdout。 |

**Hard blockers**：真实主机/链路、可执行公平 2×2、匹配训练基线、独立 formal/holdout/support 均缺。**Major concerns**：单 US101 开发轨迹、模拟 RSU/backhaul、4 个截断配对。**Minor concerns**：完整数据全在本地忽略的 artifact 中，发表前需冻结允许共享的复现包与数据许可。**Scorecard**：TMC 七维均 `N/S`，本轮不打伪精确分。**Safe claims**：只可陈述该模拟开发环境的一例有代价的迁移机制节省、六格局部成本浪费和四格截断不确定。**Prohibited claims**：真实跨 RSU 净收益、sharing×migration 交互为正、三头优于公平基线、独立泛化、TMC-ready。**TITS/TVT fit**：缺相同核心证据，不以 venue 降级替代验证。
