# CSCWD 强基线接口与独立数据可用性清单

## 身份与使用边界

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09（仅使用仓库已登记文献，本轮未作新检索）
- `target_venue`: CSCWD 2027（拟投；官方模板与页数待核）
- `artifact_run_id`: `calibrated_workflow_service_reward_learning_diagnosis_20261009_v1`；旧消费登记为 `typed_model_cache_formal_protocol_freeze_20260820_g14b_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`（项目内部审查约束）
- `git_commit_at_review`: A 文档基线 `3bdfa1ea886ca4298cd9b06b5fb023099008bf84`；B 诊断源码身份 `70a83afb5d3d3b3fb40fcaeaf825a82e4b74c416`，诊断报告/原件交付提交 `f8baf30f8d6ed37898804f570d8ebfcbeaa50ad9`；投稿 scientific commit 待冻结
- `evidence_level`: `E2_ARTIFACT_AUDITED_NONFORMAL_OFFLINE_DIAGNOSIS` 仅适用于 B 诊断自身；强基线和新独立数据为 `E0_UNAVAILABLE`
- `verdict`: `BASELINE_INTEGRATION_PENDING / CONFIRMATORY_INTERVALS_UNVERIFIED / NOT_SUBMISSION_READY`

本文只读 B 交付提交 `f8baf30` 中的文档、配置与源码，以及 A 工作树中的历史登记。B 证据路径仅在该提交及其分支可见，A 分支没有复制 B 产物。没有运行训练、读取封存性能、下载数据或创建新 split。以下数量是元数据审计，不是新实验结果。

## 唯一拟用接口及能力公平性

下一轮若实施，接口身份拟为 `calibrated_workflow_interface_v2`、`service_aligned_v1`、`independent_heads_executed_env_v2`，并整合 frozen-window identity 校验；**scientific commit 尚未冻结**。旧 v70 sparse-tail prior、预测门控或其他历史开关不得据名称推定在此版本生效。PopArt critic normalization 是已有训练稳定化技术；截至所读 B 提交仅有单变量开发 A/B 冻结设计，本稿尚无可审查的训练因果结果，不把它写成原创算法。

| 行/能力组 | 当前真实代码入口 | 必须披露的信息、状态和预算 | 强基线缺口 |
| --- | --- | --- | --- |
| SA-GHMAPPO | `src/agents/sa_ghmappo_agent.py`，`_build_agent` | 单一 controller 的 slow/fast/event 决策头；实际执行五动作之一，`executed_action_ppo_only=true`。需公开最终有效特征、loss、参数量、推理耗时和 checkpoint。 | 学术身份与 PopArt 生效版本未冻结；不得套用 v70 机制或结果。 |
| PPO | `src/agents/ppo_agent.py` | 同动作 mask、原始请求、交互/梯度更新和四次选模机会；平面语义编码的能力差异单列。 | 需同科学版本重训及完整原件。 |
| controller-level MAPPO | `src/agents/mappo_agent.py` | controller 级 cache/execution/event 头；不称 vehicle/RSU 多智能体 MAPPO。 | 需同科学版本重训及完整原件。 |
| DT handoff DRL | `src/agents/dt_handoff_agent.py`，registry 名 `dt_handoff_drl` | project-native、literature-inspired，不是指定原论文的精确复现；使用预测 RSU sequence、dwell、confidence/uncertainty、future load 和 boundary pressure。逐字段核对其是否超出 SA/PPO 可用预测信息。 | 仅 registry 和单步构建可用；service runner 的训练、dev 选模、checkpoint/hash、逐行结果和分析消费链均未接入。 |
| Popularity | `src/agents/popularity_cache_heuristic_agent.py` | 非学习、累计 adapter count，阈值预取，并使用预测 next RSU / handoff target 选择 prepare。必须明确记忆在独立 episode/窗口间如何重置，避免跨 split 状态泄漏。 | 仅 registry/单步可用；service runner 的规则结果、状态生命周期、原件和分析消费链未接入。不得复制为多个“训练 seed”增加样本量。 |
| two-step exact-transition | `src/envs/core/calibrated_continuous_workflow_env.py::TwoStepCostRule` | clone 可执行环境，枚举前两步；按 completed nodes、failed service、deadline、elapsed seconds、bytes 字典序选动作。它有 exact transition/model-based 能力，单列能力表，不称 matched-capacity learned baseline。 | 已在旧 service runner 中评估，但必须随新版本重新核对信息边界和结果原件。 |

当前 `scripts/run_calibrated_workflow_service_reward_alignment.py` 从配置读取方法，却在第 206 行硬性要求 `['sa_ghmappo','mappo','ppo']` 和旧 seeds；训练、dev checkpoint 选择、selected 保存、evaluation rows、behavior ledger 与 manifest 均围绕该链路生成。`scripts/run_calibrated_workflow_interface_repair.py::_build_agent` 本身调用通用 registry，`_evaluate_agent` 也能接收 agent；这不等于 DT/Popularity 已进入完整 runner。最小独立接入任务是：扩方法/规则的配置及身份门；给 DT 相同预算的训练、dev 选模、hash 与完整逐行 artifact；给 Popularity 明确 episode memory/reset 及单次确定性评估；统一五动作 mask/观测权限；扩 analyzer 的方法枚举、能力标签和条件时延分母；加 raw row 与 manifest/command/checkpoint 完整性门。任何 capability mismatch 先记录并停止排名。

学习方法建议采用 B 已冻结的**待执行设计**：seeds `7/17/29/43/61`，每 cell 1,440 environment steps、24 updates、每次 60 transitions、4 PPO epochs、batch 32、192 optimizer steps；在 updates `6/12/18/24` 给相同选模机会，0 方法专属超参搜索。旧服务实验按 192 episodes 训练，实际 steps 为 `1,216–1,456`，不能称交互预算相同。规则方法无训练 seed，按每个原始窗口/工作流一次确定性结果入表。

## 原始区间并集和可用性

只读输入：`artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/historical_window_usage_registry.json`（截至当时的 1,209 个计划/结果文件元数据，668 个独特历史外层区间；其中 418 个区间身份仍为 unknown）、`configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json`（36 实例）、`configs/experiment/typed_model_cache_formal_protocol_v1_20260820/{train,dev,formal,sealed_holdout}_window_plan.json`（60 个已选择/保留 I-80 窗口）和 `candidate_window_inventory.json`（579 个当时的 I-80 coverage 候选）。36 实例的 `source_segment_id` 与闭合原始 time interval 两两无交叠；但全部已进入 train/dev/regression/frozen-check，均为 development。按相同 segment 和原始 time interval 与旧登记交叉，36 个中 34 个至少与一项历史已登记区间重叠；另两个 Peachtree 项无“已登记的确定时间区间”匹配，**不能据此认定未消费**。

| 已消费 calibrated split | 数量 | 原始来源与 time/frame 概览 | 当前资格 |
| --- | ---: | --- | --- |
| train | 12 | Lankershim 2、US-101 10；各实例 24 帧；time/offset 见 manifest `source_interval` | development，不能转作新评价 |
| dev | 4 | Peachtree 1、US-101 3；各 24 帧 | 已用于 checkpoint selection |
| regression | 12 | Lankershim 4、Peachtree 1、US-101 7；各 24 帧 | 已暴露开发验证，不是 holdout |
| frozen_check | 8 | US-101 8；各 24 帧 | 已暴露开发检查，不是新 holdout |

历史 I-80 inventory 以 `i_80_run_001` 208、`run_002` 166、`run_003` 205 个互不重叠 24 帧候选构成。三 run 的**库存跨度**分别为原始 frame `12–9971` / time `1113433136100–1113434132000`，frame `27–7970` / time `1113436769600–1113437563900`，frame `1813–11628` / time `1113437746200–1113438727700`；跨度中的间隙不自动可用。四份 v1 计划已选择 train 24、dev 12、formal 12、sealed holdout 12，共 60 个 exact window ID；其余 519 个只表示**未入这四份计划**，尚未与 2026-08-20 之后所有执行/结果及 unknown 历史区间闭环，不能称“未用于方法开发”。sealed/旧 holdout 不重开。

**可证明未用于方法开发、且可立即作为新确认性 split 的范围：空集（当前认证数 train/dev/evaluation 均为 0）。**这不是断言 NGSIM 没有剩余物理帧，而是历史消费并集尚不完整。尤其旧 registry 对 418 个区间缺可靠身份，不能把未登记当未消费。I-80 库存的上述三段仅是需要进一步核验的来源范围，不是可发布的新 seal。

数据祖先：NGSIM 提供真实车辆移动轨迹，Alibaba `batch_task` 提供批处理 DAG 拓扑/时长/资源字段；二者不是共同观测的车载 AI workflow。模型/adapter 指派、状态比例、链路、车辆—DAG 配对与 deadline 为测量校准或人工假设。没有驾驶任务答案标签、真实 RSU association/无线时延，也没有真实 adapter 请求联合 trace。36-instance manifest 的 `source_files` 记录了窗口计划和 sampled DAG 的 SHA-256，但未直接给出完整 NGSIM 原始 CSV 的 hash；下一轮须把原始源 identity 补入独立 ledger。NGSIM 官方入口和 Alibaba 仓库见 `DATASET_SOURCES.md`；当前文档未取得两源原始数据再分发许可结论，后续公开数据卡需逐源复核。`LuST` 是保留 support provider，`highD` 原始源缺失；均不替换当前主线。

最低成本后续方案：先仅做元数据审计，按 source hash/segment/run 与原始 frame/time 将旧 668 记录、unknown 保守排除、2026-08-20 后全部计划/命令/结果的窗口身份、calibrated 36 项及 60 个 I-80 已选窗口去重取并集；不读封存性能。再从完整原始源按连续性/机制机会盲选不重叠候选，给 train/dev/evaluation 明确互斥区间和最小 gap，核对 Alibaba workflow 祖先、标签/许可和每个实例的 source hash。评价外层建议至少 12 个独立原始窗口以满足项目最低统计功效提示，但不能为凑数牺牲互斥或把 seed/workflow 当额外 cluster。若候选不足，直接记 `inconclusive`，先扩数据来源或缩小 claim，不复用旧 holdout。

## 主表冻结草案与否证门

学习组 SA、PPO、controller-level MAPPO、DT 同列服务主表；Popularity 单列有预测的 stateful heuristic；two-step 单列 exact-transition model-based upper-capability comparator。逐方法报告 on-time/total completion、unfinished-after-deadline、failed-service attempt 与 episode 率、连续无进展、完成条件下 elapsed seconds **及完成样本覆盖率**，再列 model/state/input transfer bytes、recompute seconds、cache miss/eviction、训练/推理时间和 reward（次要）。失败 workflow 的传输、重算仍计入总体成本；条件时延不可用行不能以 0 填入。统计以不重叠原始窗口为外层，seed/workflow 为内层，报告 paired effect、percentile/BCa CI、sign test 和 Holm family。

任一窗口 identity/interval/哈希门失败，DT 信息越权，方法预算或 checkpoint 机会不等，缺原始行/命令/manifest/checkpoint hash，或 B 的 PopArt A/B 未同时通过机制与服务否证门，就停止算法晋级。PopArt 的 A/B 只检验学习稳定性，不能代替 SA 对强基线的优势证据。最多两项论文消融在**实际生效机制和主假设冻结后**选定；现有恢复对照可做系统机制实验，第二项不预设 sparse-tail prior、门控或任何有利开关。
