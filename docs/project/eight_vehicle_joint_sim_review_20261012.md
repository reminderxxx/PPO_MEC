# 八车联合控制仿真首轮独立审查（2026-10-12）

| 字段 | 值 |
|---|---|
| `reviewed_at` | 2026-10-12 08:16 UTC |
| `literature_cutoff` | 2026-10-12；近邻定位沿用同日 `eight_vehicle_baseline_validity_review_20261012.md`，本轮在 artifact 阶段停止论文晋级审查 |
| `target_venue` | IEEE TMC（项目内部审查口径） |
| `artifact_run_id` | `20261012T080927Z_formal_sim` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| Git commit | `b97477d5111d199909904a6513b852c42d278635` |
| evidence level | 内部仿真原始 train/dev/formal 与 checkpoint 已审；论文完整包缺 holdout/support，为 `E0_UNAVAILABLE` |
| verdict | **内部仿真执行完成；主方法优势不成立。论文或 TMC-ready 判断为 `Unverifiable`。** |

本轮只读审计上述固定 run 的 manifest、checkpoint、训练更新、开发/正式结果和逐请求事件；未修改算法、成本、窗口或 checkpoint，也未开启封存 holdout。

## Evidence inventory 与协议核对

运行根目录为 `artifacts/eight_vehicle_joint_sim/20261012T080927Z_formal_sim/`。`manifest.json` 记录命令、输入 CSV SHA-256、七个原 registry 算法、seed 7/17/27、20/8/12/12 窗口划分、四站 train-only 布局及配置成本。
`audit.json` 独立核验了 21 个 checkpoint SHA-256、504 个训练 episode、168 个 dev 窗口、252 个 formal 窗口和 119,014 条 formal 逐请求事件；formal 事件数、节点完成数、八请求计数和汇总行逐项一致。每 `(agent, seed)` 恰有 24 次训练、8 个 dev 和 12 个 formal 窗口。train/dev/formal/holdout 的原始时间区间两两不重叠，且没有 holdout 评估文件。

本次所谓 `formal_sim` 仅指**内部轨迹驱动仿真协议的冻结评估阶段**。RSU 计算/链路/cache 容量与车端时长由配置给出，NGSIM 与 Alibaba 请求配对、adapter 标签均为受控合成。训练、开发、正式和封存窗口来自同一 Peachtree segment；时间互斥不等于独立道路场景。原拟 holdout offset 5400–6720 曾在开发检查中查看换站机会，已排除；当前封存区间 offset 7080–8400 未运行策略评估。已有单主车正式 Protocol 与本 run 不合并。

## 原始结果与失败机制

下表是 3 seed × 12 formal 窗口的描述性均值；`8/8` 完成窗口数按 36 个执行计，不是 36 个独立道路样本。

| 算法 | 平均完成请求 / 8 | 8/8 窗口 | 平均 formal reward | 说明 |
|---|---:|---:|---:|---|
| `sa_ghmappo` | 0.000 | 0/36 | -7.825 | 几乎始终 action 3，adapter 未就绪 |
| `ppo` | 5.333 | 24/36 | 37.910 | 有 seed 决策退化为 action 3 |
| `mappo` | 8.000 | 36/36 | 61.650 | 主要依靠 action 2 车端兜底 |
| `dqn` | 5.333 | 24/36 | 39.777 | 有 seed 决策退化为 action 3 |
| `dag_offload_drl` | 5.306 | 15/36 | 38.863 | seed 间波动大 |
| `cache_offload_drl` | 8.000 | 36/36 | 61.976 | 主要依靠车端兜底 |
| `dt_handoff_drl` | 8.000 | 36/36 | 65.378 | 使用缓存与车端兜底混合策略 |

`sa_ghmappo` 的 34,560 条 formal 决策中，action 3 为 34,518 条、action 4 为 42 条；没有 action 0，34,459 条事件为 `adapter_not_ready`。八请求在 36 个窗口均未完成。三个 seed 的 dev 窗口同样全部为 0/8，说明这不是只在 formal 窗口出现的偶然失败。训练 episode 普遍 8/8 完成，显示随机采样训练轨迹与确定性评估策略严重分离；导致策略退化的具体代码原因尚未定位，不能把推测写成已证实根因。

五动作现在确实能控制缓存拉取、车端/RSU 执行和迁移准备，但完成率由车端兜底主导的算法不能自动被解释为 cache 或 handoff 机制更优。formal 的 252 个执行中，只有 94 个在**工作流仍活动时**出现换站事件；累计 142 次换站、70 次到站已准备、2 次显式冷启动失败。这些计数跨 seed/算法复用窗口，不构成独立机会数。短 DAG、全完成天花板及不同策略的提前结束使机制收益难以辨别。

## Blocker、评分与 claim 边界

- **Artifact blocker：** 没有 holdout、prediction/system robustness、scalability、ablation 和近邻论文方案输出；单独的命令字符串与 `audit.json` 也不等于完整 paper package。故顶刊证据等级为 `E0_UNAVAILABLE`，总评分各维均为 `N/S (not scored)`。
- **Baseline 公平性 blocker：** 七个算法使用相同 episode 数，但因早停不同，训练决策数约 17,324–21,208，未严格对齐交互预算；三种方向型实现不是外部论文原算法。不能称已完成全面近邻论文比较。
- **机制/统计 blocker：** SA-GHMAPPO 零完成；部分对照靠 action 2 完成；窗口同一 segment 且 seed 重复。没有按窗口外层重采样的 CI、BCa、paired sign 与 Holm 校正，也没有独立 holdout。不能从平均 reward 排序得出主方法优势。

Scorecard：Novelty、技术正确性、baseline 公平性、实验统计、机制兑现、鲁棒性和可复现性七项均 `N/S`；缺完整论文证据且存在上述硬 blocker，不计算总分。

**可安全表述：** 该 commit 的内部八车、四逻辑 RSU 联合动作仿真已经跑通 21 个原算法 checkpoint 与 252 个正式窗口执行；原始事件可回溯；SA-GHMAPPO 在此冻结设置下确定性评估失败。**禁止表述：** 主方法优于 PPO/MAPPO/近邻论文方案、真实四站部署、八车真实请求映射、论文主表或 TMC-ready。

后续应另立实现任务，在 dev 上定位 SA-GHMAPPO 确定性策略退化，冻结等交互预算与机制机会协议，并增加忠实的外部论文对照和实测成本/独立场景。当前 formal 已消耗，不能在看过本轮结果后调参再把它当新的盲测；封存 holdout 仍不打开。TITS/TVT 适配性也不能在缺少这些证据时作积极判断。
