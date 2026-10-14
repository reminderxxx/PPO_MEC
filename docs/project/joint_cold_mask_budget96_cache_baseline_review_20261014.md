# 八车联合控制：96 episode、缓存基线与可辨识性审查（2026-10-14）

- `reviewed_at`: `2026-10-14`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-14`（本轮核验缓存/卸载直接近邻，非完整 novelty 再检索）
- `target_venue`: IEEE Transactions on Mobile Computing（内部证据门槛）
- `artifact_run_id`: `20261014T071053Z_development`；同合同 24 episode 对照 `20261014T064956Z_development`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `6aba08b5c4673f101f8fe413c37936f9b491f198`（96 episode 运行时）
- `evidence_level`: 96 episode **开发集原件已局部审计**；论文完整证据为 `E0_UNAVAILABLE`
- `verdict`: `Unverifiable`（SA-GHMAPPO 超越 MAPPO、model-cache 友好优势、paper-ready）；在当前合同中，增加训练量没有打破 SA 与 MAPPO 的行为及奖励平局。

## Evidence inventory、protocol 与完整性

- 新 run：`artifacts/eight_vehicle_joint_v2_cold_mask_budget96_20261014/20261014T071053Z_development/`；训练命令记录在 `manifest.json.command`，终端输出在上级 `training.log`。含 864 条训练、216 次更新、9 个最终 checkpoint、72 条 greedy 与 72 条 sampled 开发评价、48,275 条事件和 7,977 份独立逐帧奖励收据。
- `independent_audit.json` 的 14 项检查全部通过，包括运行结束、数量、checkpoint SHA-256、唯一行键、收据对奖励守恒及 formal/holdout 不存在。审计由 `artifacts/eight_vehicle_joint_v2_development_20261012/audit_development.py` 从原始 JSONL 重建。
- `budget_comparison.json` 验证两次开发运行的原始数据 SHA-256、工作流/adapter、RSU 布局、成本/奖励、全部 train/dev/formal/holdout 窗口、动作/观察合同、agent 选项、seed 和更新间隔逐项一致；唯一实验预算改变是每 agent/seed `24 → 96` episode，更新 `6 → 24` 次。9 个 agent/seed 的前 24 条训练与前 6 次更新逐条一致。96 episode 的实际训练决策总量：SA 61,982、MAPPO 61,993、缓存基线 61,968；相同 episode 不代表逐决策数量完全相等。
- 原始移动轨迹为 NGSIM、DAG 源为 Alibaba；两者的配对、adapter 身份、四站资源和模型/无线字节成本属于受控仿真，非真实联合请求日志或实体 RSU 测量。`development_only=true`，未评价 formal/holdout/support，也未改 canonical。

## 结果与收敛边界

3 seed × 8 个开发窗口；均为末端 checkpoint 的确定性 greedy 评价，完成请求分母为 8。

| 算法 | 24 episode 奖励 | 96 episode 奖励 | 96 episode 完成 | 96 episode greedy 动作 |
|---|---:|---:|---:|---|
| `sa_ghmappo` | 66.383740 | 66.383740 | 8.000 | action 0：8,037；其他 0 |
| `mappo` | 66.383740 | 66.383740 | 8.000 | action 0：8,037；其他 0 |
| `cache_offload_drl` | 66.383740 | 66.372424 | 8.000 | action 0：5,640；action 3：2,414 |

- SA 对 MAPPO 的 24 个配对窗口奖励差全部为 0。SA 对缓存基线的平均差为 `+0.011316`，只出现在 seed 17 的 4 个窗口；其他 20 个配对行均相同。缓存基线 seed 17 多选稳态 action 3，平均模型传输链路帧从 15.875 增到 17.000，完成时间从 55.375 增到 55.750 帧；三算法仍全部完成 70 个节点。
- 96 episode 的 sampled 开发均值：SA `66.382907`、MAPPO `66.374164`、缓存基线 `66.373747`，完成均为 `8/8`。采样与 greedy 行为不同，不能用 sampled 的微小差值声明部署策略优势。
- 每个 seed 最前与最后一组完整 20 train 窗口，SA 平均回报约 `67.528–67.533`，没有随四倍预算稳定升高；三算法末 6 次更新的 critic `explained_variance` 近 0，SA 的 `policy_entropy` 约 `0.525–0.553`。因此只能说**greedy 结果在 24 与 96 的两个端点持平**，不能据此证明 PPO 参数或 critic 已收敛。96 episode 没有中途 greedy checkpoint 曲线；需要后续固定间隔 checkpoint 才能检验训练期收敛。
- 原 v2 无 cold mask 的 96 episode 诊断曾出现 SA seed 27 贪心 action 3 死锁，而 MAPPO 改善；它与本轮动作可行域不同，不能跨协议拼接排名。本轮证据说明共同 cold mask 消除了该死锁症状，训练量本身没有形成 SA 特有的收益。

## 缓存基线到底是什么

`src/agents/cache_offload_agent.py` 中的 `CacheOffloadDRLAgent` 是本项目自建的**缓存特征增强、三 controller head、中心 critic 的 PPO 方向对照**：它与 SA/MAPPO 共用 PPO 基类和五动作环境，使用 flat semantic encoder 加 14 项缓存/目标站/预测负载标量，不使用 SA 的图编码、surrogate 融合、相关 guard 和辅助损失。它同样接收环境提供的预测信息，因此不是“无预测”对照。

文献表中的 [Xue 等 IEEE TVT 2023](https://users.sussex.ac.uk/~zs70/files/2023_TVT_Joint_caching_vehicular) 研究 service program cache 与卸载，原文采用 **DDPG** 且优化任务处理时延；本仓库算法是 **PPO**、缓存对象为 typed base/adapter、奖励和动作不同。因此 `paper_grade_independent_baseline=true` 只是仓库元数据，**不等于外部论文的独立忠实复现**。它可作为第一方领域消融/方向基线，论文中必须按此名称和边界表述。

代码中的额外特征存在可复现错配：`_occupancy()` 用 `len(cached_adapter_ids) / adapter_cache_capacity`，而 v2 真正约束是 `occupied_bytes / capacity_bytes`，包含 base、adapter 与预留传输字节。示例：两 adapter 加 base 占 `3072/4096=0.75`，旧特征却报 `2/2=1.0`；共同 joint actor feature 另有正确字节比，故不可把此单项错配解释为本轮排名的已证实唯一原因。`_demand_score()` 要求 `predictions.cache_demand`，而本轮 `CausalMobilityPredictor` 不产出该字段，两项需求特征固定为 0；`future_load` 是投影车辆数，不是实际 CPU 队列。需要在后续**实现轮**修复生产端与消费端，并给三算法相同可见信息。

## 一阶机制诊断：为何难以拉开奖励

1. 当前 action 0 同时向当前站请求所需 adapter 并继续 RSU 执行；adapter 已就绪时请求返回 `already_cached` 且不消耗链路，action 3 仅做稳态执行。冷态下共同 mask 禁用 3，热态下 0 与 3 高度接近，差别主要是队列/状态未可执行时 action 0 会额外 touch LRU。这给所有方法提供简单而强的反应式动作，弱化预测与缓存规划的可辨识性。
2. 当前五动作既不指定未来 DAG 节点的 base/adapter 对象，也不选择候选执行 RSU、传输量或驱逐对象；typed cache 的依赖装载及 LRU 驱逐是环境自动逻辑。所谓“cache/offloading”高奖励不代表策略学到了 base/adapter 放置和选站。SA/MAPPO 的 96 episode greedy 共 8,037 次 action 0，预测预取、状态准备、执行期间实际关联切换和车端 fallback 均为 0。
3. `manifest.mechanism_qualification` 记录 70 个节点的工作量全部为 1，RSU 每帧也处理 1 单位，多帧节点比例为 0。所有成功开发轨迹均在首次有效换站前完成；几何轨迹虽含少数后期换站，当前执行期没有迁移机会。模型工作集 `4608 B` 比单站容量 `4096 B` 仅高 `512 B`，存在少量真实缓存替换，却没有足以体现预测预取和迁移的压力梯度。
4. SA 的 96 episode 平均奖励可由每窗口 `70` 个节点完成减去等待 `3.268750`、模型传输 `0.158750`、共享广播 `0.188760` 精确复算。按该奖励，固定 70 节点的理论上界为 70，当前距无惩罚上界仅 `3.616260`；主要可改进空间是等待。每站仅 1 个计算槽，但动作没有替代站/资源分配能力。不能在这一开发任务里期待靠更多训练得到大的、可归因于 model cache 的分数领先。
5. 预测器是 3 帧常速度因果外推，共享状态是每帧广播、延迟 1 帧的序列化 JSON 代理。它们是环境公共输入/成本，不是已训练并证实能改善决策的 surrogate。当前没有“预测 → 预取 → 到站就绪 → 完成/成本”的有效事件链。

## 审查结论和后续独立实现任务

- **Hard blockers**：无 formal/holdout/support 新证据；SA 对 MAPPO 零差；model-cache 与 handoff 机制没有兑现。论文算法奖励最高且收敛、优于缓存基线、TMC-ready 均为 `Unverifiable`。本轮只读审查未在看到结果后调参数、改环境或挑选窗口。
- **Major concerns**：缓存基线元数据暗示外部独立复现、v2 字节容量特征错配、需求特征恒零；五动作对 cache 对象/执行站缺控制；1 帧 DAG 及零执行期换站；critic explained variance 近零；相同窗口跨 seed 重用，24 个配对行不能视为 24 个独立场景。
- **Scorecard**：按 `top_journal_review_policy.md` 七维均 `N/S (not scored)`，因为开发 run 不含正式性能与独立 holdout/support；不造论文级分数。**Safe claims** 仅限同合同开发端点及其动作/成本原件。**Prohibited claims**：SA 在这轮显著优于 MAPPO；本轮实现是 Xue 等 DDPG；缓存差异已证明 SA 的 model-cache 友好机制；训练 96 次已数学或统计收敛。
- **TITS/TVT fit note**：不依据本轮开发仿真评价适投性。

下一实现轮应依次：（1）先冻结共同 actor 信息/动作与 typed byte cache schema，使显式 base/adapter、目标站、预取时机和状态准备成为可执行决策，并将对应字段的所有 producer/consumer 同步；（2）用训练数据预定而非开发得分倒选的多帧工作量、持续到 handoff 的 DAG、缓存容量/链路竞争构造独立机会层，再单轴扩展到 16 车 6 站；（3）保持 SA、MAPPO、第一方缓存 PPO 的同输入同物理约束，并增加反应式 action-0 与外部论文忠实对照或清晰标注的领域对照；（4）固定 24/96/更高预算的中间 checkpoint、同交互量和成对窗口统计，检验 greedy 奖励、方差和预测/缓存/迁移链指标。满足开发集机制门槛后才冻结候选并运行从未查看的 formal/holdout。研究目标可以是最高回报，但排名必须由公平协议和独立证据产生。
