# 车辆预测器与冷缓存动作修复：开发集结果（2026-10-14）

- `reviewed_at`: `2026-10-14`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-14`；核验车辆预测/迁移近邻，非完整 novelty 更新
- `target_venue`: IEEE Transactions on Mobile Computing（仅用于证据门槛）
- `artifact_run_id`: `20261014T064956Z_development`；24 episode 旧合同对照 `20261013T083022Z_development`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `6865720fdaea1debac337e212651274a35339483`（新开发运行时）
- `evidence_level`: `E0_UNAVAILABLE`（论文完整证据）；本轮开发 run 的 manifest、原始事件、checkpoint 和奖励收据通过局部审计
- `verdict`: `Unverifiable`（SA-GHMAPPO 优于 MAPPO、model cache 优势及 paper-ready）；共同冷缓存死锁在 opt-in 开发合同中消失

## Evidence inventory 与 protocol/provenance

- 新 run 根：`artifacts/eight_vehicle_joint_v2_cold_mask_20261014/20261014T064956Z_development/`，含 `manifest.json`、`train.jsonl`、`updates.jsonl`、`checkpoints.jsonl`、9 个 checkpoint、`dev.jsonl`、`dev_sampled.jsonl`、`dev_events.jsonl`、`independent_audit.json` 和 `cold_mask_comparison.json`。216 train、54 update、72 greedy dev、72 sampled dev、48,295 事件、7,980 帧收据；14 项审计（含 checkpoint SHA-256、每帧奖励复算）全部通过。
- 新旧均为 8 车 4 RSU、20 train/8 dev 窗口、3 seed、24 episode/seed、每 4 episode 更新。`cold_mask_comparison.json` 核对原始 NGSIM/Alibaba SHA-256、时间窗口、工作流、adapter、RSU 布局、奖励、训练轮数和物理成本均一致；`src/` 与 `scripts/` 的 commit 间改动仅为共同 cold-cache mask 与入口字段。两轮均从头训练；旧 checkpoint 没有放进新合同评估。
- 新合同为 `semantic_discrete_5_per_request_v2_cold_cache_mask`：只有当前 adapter 在本车实时可见的关联站就绪，action 3 才合法；action 0 保持可请求模型。改变可行域的规则对全部算法相同，原 v2 默认不变。新/旧不能混写为同一策略排名。
- 预测离线证据：`artifacts/analysis/joint_predictor_qualification_20261014/qualify.py` 与 `qualification.json`。8 车 4 站使用新 run 的实际 RSU 布局和 dev 窗口；16 车 6 站是只用训练窗口建站的几何预检。该脚本未训练 LSTM、未运行策略，也没有访问 formal/holdout 结果。

## 开发集结果与 first-order 解释

三算法各 3 seed × 8 dev 窗口；数字为 greedy 均值，完成请求分母为 8。

| 算法 | 旧合同奖励 / 完成 | cold-mask 奖励 / 完成 | 新合同逐请求动作 |
|---|---:|---:|---|
| `sa_ghmappo` | `43.748 / 5.500` | `66.384 / 8.000` | 8,037 次 action 0；其他 0 |
| `mappo` | `42.396 / 5.333` | `66.384 / 8.000` | 8,037 次 action 0；其他 0 |
| `cache_offload_drl` | `66.384 / 8.000` | `66.384 / 8.000` | 8,037 次 action 0；其他 0 |

- SA 原失败 seed 27 从奖励 `−1.522`、近 `0.5/8` 完成回到奖励 `66.384`、`8/8`；MAPPO 原失败 seed 7 同样恢复。新合同下 SA 对 MAPPO 的 24 个配对 dev 行奖励差均为 0（0 胜、24 平、0 负）。缓存方向基线原本已全部完成，结果不变。
- 三者的 greedy 均完全选择 action 0，平均每窗口 `11.625` 次 cache fetch；状态提前准备、执行期关联 handoff 和车端 fallback 都为 0。共同 mask 排除了明显无效动作并解决了训练/部署分布失配的一个症状，但没有产生 SA 结构性优势，不能把新合同分数提升归因于图、层级、surrogate 或 model cache 友好性。
- 新旧方案物理成本和 reward 没变，但动作可行域变了；不能把新 `66.384` 与旧 `43.748` 当作同一任务上的 SA 净算法增益。当前成功轨迹在预见的 handoff 前已完成，比较缺乏迁移/预装机会。

## 预测器判断

- 当前 `CausalMobilityPredictor` 是一帧速度估计加未来 3 帧几何投影；它是未训练的因果基线，`prediction_confidence` 为已到期预测正确率的滚动均值，不是校准概率。
- 原 8 车 4 站的 8 个 dev 窗口仅有 4 次独立关联切换，位于窗口第 50、99、105、114 帧。未来 3 帧事件标签对应 12 个高度相关 vehicle-frame，常速度基线 12/12 命中；10 帧和 20 帧事件 F1 分别约 `0.986`、`0.976`。这些不是 12 个独立事件，也不覆盖跨路段外推。
- 16 车 6 站几何预检有更多机会：3 帧事件标签 103 个 vehicle-frame，真阳性 97、假阴性 6、假阳性 1，F1 约 `0.965`；10/20 帧 F1 约 `0.957/0.951`。同样有连续帧相关与单一路段限制。这些分数只评价 RSU 切换事件，尚未评价到站提前量、错误预取费用或策略收益。
- [2024 TMC 的 GRU+GCN 主动迁移](https://doi.org/10.1109/TMC.2023.3271655)、[2026 IFIP 的 LSTM 预测 serving BS](https://doi.org/10.23919/IFIPNetworking70592.2026.11579216)、仓库已有的 LightGBM/Mamba 近邻说明可选模型很多；其中 IFIP 摘要明确指出错误预测会拖累 RTT。当前瓶颈不支持直接以更大模型替换 predictor 来解释算法落后。候选顺序宜是因果速度、LightGBM/轻量 GRU 或 LSTM 的同 split 比较，再按提前量与下游净收益选型；复杂模型不自动获得方法 novelty。

## Claim 边界与下一实现任务

- **Hard blockers**：三算法在新开发合同下全部同动作同结果；没有已兑现的 model cache、状态迁移或预测增益；真实模型/无线成本未校准；formal/holdout/support 新包不存在。
- **Major concerns**：单一路段、同一开发窗口反复使用、交互预算可比性仍需按真实决策数复核；新动作合同尚未成为冻结正式协议。预测评估按 vehicle-frame 报告会夸大独立样本量，正式预测统计须按窗口/换站事件聚类。
- **Safe claims**：共同冷缓存动作可行性修复消除了本开发设置的贪心停滞；现有 3 帧常速度预测在这批 8 车开发轨迹的少数换站事件中已命中。
- **Prohibited claims**：SA 优于 MAPPO/cache 基线；LSTM 能提高当前奖励；更复杂预测器能保证跨 RSU workflow 连续性；新结果证明真实 model cache 优势或论文问题无新颖性风险。
- **Scorecard**：论文级七维均 `N/S`；当前只有受控开发诊断，缺少完整 formal/holdout/support 及机制兑现，不能给伪精确分数。
- **TITS/TVT fit note**：本轮不评价适投性。

下一独立实现任务应先冻结可辨识机制的开发输入：跨帧节点、模型工作集高于容量、工作流在 handoff 时仍运行、目标端有真实可用的预取提前量；分开改变车数与 RSU 数，并保持所有算法同动作/资源合同。再测“预测准确度 → 提前准备 → 到站就绪 → 完成量/传输成本”的链式证据，并在训练窗口拟合轻量预测器、以窗口/事件分组的独立开发数据评估。不能按本轮奖励挑选参数或重用已查看 formal 作为盲测。
