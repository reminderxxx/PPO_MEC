# 联合环境真实性、动作与规模预检（2026-10-14）

- `reviewed_at`: `2026-10-14 14:30 CST`
- `literature_cutoff`: `2026-10-14`；本轮仅核验与仿真有效性直接相关的 VEC-Sim，不构成完整 novelty 检索
- `target_venue`: IEEE Transactions on Mobile Computing（证据门槛，不预测录用）
- `artifact_run_id`: `20261014T035709Z_development`；新增输入预检 `artifacts/analysis/eight_vehicle_scale_preflight_20261014/preflight.json`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `39e3aa392a03a2a6b7dfe8069a4b051abb26e541`（审查时源码）
- `evidence_level`: `E0_UNAVAILABLE`（完整论文证据）；本轮输入预检及既有开发结果可复核，但 formal/holdout/support 新包缺失
- `verdict`: `Unverifiable`（主方法优于 MAPPO / 更善于 model cache）；以下只读诊断不是算法晋级

## Evidence inventory 与审查范围

- 既有 24/96 episode 的开发结果及完整性核算见 `eight_vehicle_joint_v2_budget_review_20261014.md`，本轮不重新训练、不评估 formal/holdout、不修改算法。
- 输入预检脚本及 JSON 在 `artifacts/analysis/eight_vehicle_scale_preflight_20261014/`；命令为 `PYTHONPATH=. /Users/howen/Projects/PPO_MEC/.venv/bin/python artifacts/analysis/eight_vehicle_scale_preflight_20261014/preflight.py`。只读取 NGSIM 前 800,000 行和 Alibaba 原始 CSV；用 120 帧不重叠块计算持续车辆数，另用同一批 16 车合格窗口的前 20 train / 一窗 gap / 8 dev 计算 RSU 关联切换。此预检不运行策略、缓存和迁移，也不查看 hidden holdout 的策略结果。
- 原始输入哈希沿用开发 manifest：NGSIM `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`，Alibaba `6346b0726c6e10466a585c67645af807b425b5be091caf410f5e1aff41a270bc`。预检脚本使用的原始路径见其 JSON；本轮没有重新计算整文件哈希。

## 可观察结论

1. **数据真实性是分层的。** NGSIM 提供原始车辆轨迹，Alibaba 提供 DAG 结构；两者的逐车配对和 `semantic_ai_service` adapter 标签是受控生成。四站布局、计算槽、链路槽、模型大小和传输率来自配置；`_work_required=max(1,input_size*workflow_profile_unit_bytes/workload_bytes_per_unit)`，当前配置下既有开发集节点全部一帧。没有真实模型的负载、端到端无线传输或物理 RSU 执行证据。它是轨迹驱动的联合仿真，不是现场学习。
2. **现有五动作能联调联合控制，但不能完整表达拟议决策。** action 0 在当前 RSU 请求 adapter 并执行；action 1 向预测下一站预取当前 adapter；action 2 车端执行；action 3 当前站直接执行而不请求缓存；action 4 为预测 handoff 目标预取当前 adapter 并准备状态。base 由 `JointModelCache.request_adapter()` 按依赖自动加载，驱逐是共同的自动 LRU。没有显式选择 base/adapter 对象、未来 DAG adapter、任意候选站、逐对象驱逐、链路/算力份额或迁移发起后的控制。状态传输在 v2 按字节/帧计费，但仍是仿真 payload。动作数量本身不是瓶颈；未覆盖的真实控制变量和 action 0/3 冷缓存可行性才是关键。
3. **训练不足不是唯一原因。** 同条件 24→96 episode 后，MAPPO 的 greedy 恢复，SA seed 27 仍 0/8 且每次选 action 3；采样策略仍全部成功。当前需要记录条件动作概率、cold/warm cache、完整 team credit 与 critic 质量，先修共同动作合同/任务机会，再画训练预算曲线。
4. **原始数据可支撑适度规模，但训练入口尚不能直接扩容。** 10,421 帧形成 86 个同路段不重叠 120 帧块；持续至少 8/12/16/24/32 辆车的合格块分别为 83/81/80/79/76。预检取得 32 个独立 Alibaba 工作流（310 节点、5 种 adapter 标签）。上述合格轨迹均为 Peachtree；增加车数不增加路段多样性。`select_continuous_window`、工作流 builder 和 RSU 布局函数接受可变规模，但实验 launcher 将 8 工作流、8 车、4 RSU、每站 1 计算槽/1 链路槽写死。v2 actor/critic 使用定长局部/聚合特征，PPO 另有 `/8.0` 车辆负载归一化；因此需修改并审计生产者、所有算法、checkpoint 与 benchmark 消费端，不能直接拿八车 checkpoint 推断扩容效果。
5. **规模与机制机会不等价。** 以相同 29 个持续 16 车窗口中的 8 个开发窗口作配对几何预检，4 RSU 时 8/16 车的全 120 帧关联切换为 4/23 次，前 20 帧为 0/2 次；6 RSU 时为 10/35 次，前 20 帧为 2/7 次；8 RSU 时为 15/43 次，前 20 帧为 3/9 次。它们只是轨迹关联切换，未确认发生时 DAG 仍运行、目标模型尚未就绪或状态可提前传输。每站峰值关联车数最高达 13（16 车 4 站）；若不控制算力/缓存/链路比例，只会混合拥塞和几何效应。

## Hard blockers、major concerns 与 claim 边界

- **Hard blockers**：当前模型缓存优势、预测/共享 surrogate 增益和迁移兑现均没有对应正式机制对照；模型与状态字节/时延未实测；共同动作合同会让 greedy 全选 action 3 时无人启动缓存；formal/holdout/support 新包缺失。现有方向型 registry agent 也不是最近邻论文的逐项复现。
- **Major concerns**：同一 Peachtree 路段窗口、合成到达与请求配对、五动作过度绑定缓存与执行、静态车端 adapter、常速度预测和人工参数区间；扩大车数与站数可能同时改变争用和预测难度。观察归一化与策略输入的信息边界需随规模复核。
- **Safe claims**：目前 NGSIM+Alibaba 可用于明确假设下的轨迹驱动训练；原始样本量足以构造 16 车 120 帧开发候选；开发集 96 episode 未稳定修复 SA 的 greedy 冷缓存死锁。
- **Prohibited claims**：策略已在真实 RSU 上在线学习；16 车/6 站能提高 SA 奖励；更多训练会修复 SA；当前 cache/offload/迁移动作已经覆盖论文完整控制空间；当前结果证明 SA 比 MAPPO 或 cache baseline 更好。
- **Scorecard**：完整 TMC rubric 七维均 `N/S`，因为本轮缺少冻结的新 formal/holdout/support 与规模策略运行，不输出伪精确分数。
- **TITS/TVT fit note**：本轮不评价适投性；需先完成共同合同、系统校准与正式证据。

## Required actions before re-review

1. 冻结可落地的边缘控制问题：明确每个时刻车辆/RSU 实际观测、候选执行位置、base/adapter 对象与依赖、模型来源及传输路径、缓存容量/驱逐、执行状态版本、链路/算力队列、请求到达与 DAG deadline。先用小型实测校准模型加载、adapter 切换、计算和传输成本；无法实测的参数明确做区间敏感性分析。
2. 独立实现新共同动作合同与可行 mask。建议慢时域选择候选 RSU 与需提前准备的 base/adapter/状态对象，快时域选择当前节点执行位置和必要的传输/计算资源；候选集有界，所有算法共用解码器与物理约束。为 action 0/3 的冷缓存语义和失效成本设负例；自动 LRU 保留为清晰标注的规则 baseline，而非 SA 专属贡献。
3. 构造能兑现机制的开发分层：cache 工作集高于/低于容量、可/不可复用、跨帧 DAG 节点、handoff 前可准备窗口、预测误差、共享状态延迟、链路争用、错过 deadline。每项输出 cold miss、validated hit、base/adapter transfer bytes、预取有效率、handoff 到达就绪率、状态迁移兑现、完成量和成本；先证明环境有机会，再比较策略。
4. 再做 8/16 车 × 4/6/8 站的分轴规模实验，固定或明确标注每车资源比；另变缓存容量和带宽。训练预算按环境交互量/更新次数对齐，并对 SA、MAPPO、cache 方向基线做同合同重训及预测、共享、图/DAG、cache 各自消融。记录训练曲线、greedy/sample 配对和失败 seed，冻结开发规则后再启用新的独立 formal/holdout。

与仿真方法相关的新增近邻 VEC-Sim 已记入 `literature_reference_table.md`；该论文处理 service cache，不能用作 adapter cache 机制已验证的证据。
