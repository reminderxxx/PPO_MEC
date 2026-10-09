# CSCWD 强基线接口与独立数据可用性清单

## 身份与使用边界

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09（仅使用仓库已登记文献，本轮未作新检索）
- `target_venue`: CSCWD 2027（拟投；官方模板与页数待核）
- `artifact_run_id`: `calibrated_workflow_service_reward_learning_diagnosis_20261009_v1`、`calibrated_workflow_value_normalization_ab_launch_attempt_20261009_v1`（仅启动失败回执）；旧消费登记为 `typed_model_cache_formal_protocol_freeze_20260820_g14b_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`（项目内部审查约束）
- `git_commit_at_review`: A 文档基线 `43f288fa00ea1cb8f81035670476d6a0bc9bf65d`；B 诊断证据交付 `f8baf30f8d6ed37898804f570d8ebfcbeaa50ad9`、PopArt pre-run 失败回执 `8ebaf9ce`；可用于投稿的科学结果 commit 待冻结
- `evidence_level`: `E2_ARTIFACT_AUDITED_NONFORMAL_OFFLINE_DIAGNOSIS` 仅适用于 B 诊断自身；强基线和新独立数据为 `E0_UNAVAILABLE`
- `verdict`: `BASELINE_INTEGRATION_PENDING / CONFIRMATORY_INTERVALS_UNVERIFIED / NOT_SUBMISSION_READY`

本文只读 B 交付提交 `f8baf30` 中的文档、配置与源码、`8ebaf9ce` 的启动失败回执，以及 A 工作树中的历史登记。B 证据路径仅在对应提交及其分支可见，A 分支没有复制 B 产物。没有运行训练、读取封存性能、下载数据或创建新 split。以下数量是元数据审计，不是新实验结果。

## 唯一拟用接口及能力公平性

下一轮若实施，接口身份拟为 `calibrated_workflow_interface_v2`、`service_aligned_v1`、`independent_heads_executed_env_v2`，并整合 frozen-window identity 校验；**可用于论文结果的 scientific commit 尚未冻结**。旧 v70 sparse-tail prior、预测门控或其他历史开关不得据名称推定在此版本生效。PopArt critic normalization 是已有训练稳定化技术。B 的第一次授权启动在 run root 创建前失败，科学 steps/updates/checkpoints/评价行为均为 0；本稿没有可审查的训练因果结果，不把它写成原创算法。

| 行/能力组 | 当前真实代码入口 | 必须披露的信息、状态和预算 | 强基线缺口 |
| --- | --- | --- | --- |
| SA-GHMAPPO | `src/agents/sa_ghmappo_agent.py`，`_build_agent` | 单一 controller 的 slow/fast/event 决策头；实际执行五动作之一，`executed_action_ppo_only=true`。需公开最终有效特征、loss、参数量、推理耗时和 checkpoint。 | 学术身份与 PopArt 生效版本未冻结；不得套用 v70 机制或结果。 |
| PPO | `src/agents/ppo_agent.py` | 同动作 mask、原始请求、交互/梯度更新和四次选模机会；平面语义编码的能力差异单列。 | 需同科学版本重训及完整原件。 |
| controller-level MAPPO | `src/agents/mappo_agent.py` | controller 级 cache/execution/event 头；不称 vehicle/RSU 多智能体 MAPPO。 | 需同科学版本重训及完整原件。 |
| DT handoff DRL | `src/agents/dt_handoff_agent.py`，registry 名 `dt_handoff_drl` | project-native、literature-inspired，不是指定原论文的精确复现；使用预测 RSU sequence、dwell、confidence/uncertainty、future load 和 boundary pressure。逐字段核对其是否超出 SA/PPO 可用预测信息。 | 仅 registry 和单步构建可用；service runner 的训练、dev 选模、checkpoint/hash、逐行结果和分析消费链均未接入。 |
| Popularity | `src/agents/popularity_cache_heuristic_agent.py` | 非学习、累计 adapter count，阈值预取，并使用预测 next RSU / handoff target 选择 prepare。必须明确记忆在独立 episode/窗口间如何重置，避免跨 split 状态泄漏。 | 仅 registry/单步可用；service runner 的规则结果、状态生命周期、原件和分析消费链未接入。不得复制为多个“训练 seed”增加样本量。 |
| two-step exact-transition | `src/envs/core/calibrated_continuous_workflow_env.py::TwoStepCostRule` | clone 可执行环境，枚举前两步；按 completed nodes、failed service、deadline、elapsed seconds、bytes 字典序选动作。它有 exact transition/model-based 能力，单列能力表，不称 matched-capacity learned baseline。 | 已在旧 service runner 中评估，但必须随新版本重新核对信息边界和结果原件。 |

当前 `scripts/run_calibrated_workflow_service_reward_alignment.py` 从配置读取方法，却在第 206 行硬性要求 `['sa_ghmappo','mappo','ppo']` 和旧 seeds；训练、dev checkpoint 选择、selected 保存、evaluation rows、behavior ledger 与 manifest 均围绕该链路生成。`scripts/run_calibrated_workflow_interface_repair.py::_build_agent` 本身调用通用 registry，`_evaluate_agent` 也能接收 agent；这不等于 DT/Popularity 已进入完整 runner。最小独立接入任务是：扩方法/规则的配置及身份门；给 DT 相同预算的训练、dev 选模、hash 与完整逐行 artifact；给 Popularity 明确 episode memory/reset 及单次确定性评估；统一五动作 mask/观测权限；扩 analyzer 的方法枚举、能力标签和条件时延分母；加 raw row 与 manifest/command/checkpoint 完整性门。任何 capability mismatch 先记录并停止排名。

**给下一轮实现任务的最小 I/O 与测试入口：**

| 方法 | 输入/输出与状态合同 | 必须补的消费者 | 最小必要局部测试（尚未实现） |
| --- | --- | --- | --- |
| DT `dt_handoff_drl` | `agent.act(observation, info)` 读取公共 9 值 observation、`semantic_state.predictions` 中的 sequence/next/target、dwell、confidence/uncertainty、future load，以及车辆/RSU 边界字段和 `info.action_mask`；返回五动作内合法 `action` 与 `action_info`（训练需 log probability/value）。逐字段对照 SA/PPO 是否可取得同一因果预测，超出则能力单列。 | `run_calibrated_workflow_service_reward_alignment.py` 的方法身份门、`_collect_training_episode`、dev checkpoint/selected hash、`_evaluate_agent`、`evaluation_rows`/`behavior_ledger`/`run_manifest`/integrity；`analyze_calibrated_workflow_service_reward_alignment.py` 当前只枚举三学习方法和 two-step。 | 扩 `tests/test_calibrated_workflow_service_reward.py`：同一实例与 mask 的 DT 构建/动作合法性、预测字段缺失保守行为、固定预算 update/四次选模、save/load hash 和逐行字段；现有 `tests/test_algo_pool_contract.py` 只证明 registry/算法合同，不能代替 runner 验收。 |
| Popularity `popularity_cache_heuristic` | `act` 忽略 numeric observation，消费 `info.semantic_state` 当前节点/RSU、预测 next/target 与 mask；`_adapter_counts` 累积到阈值 2 后可预测预取，`learn` 跳过。输出合法动作/原因；必须定义每个独立 workflow/window 的计数重置还是显式连续请求流，不让 train/dev/evaluation 串状态。 | 在 service runner 加一次性确定性规则评价入口；写同 schema 的 raw rows、行为 ledger、能力/记忆元数据、命令与完整性清单；analyzer 单列 heuristic，不能要求 checkpoint 或复制 5 个 seed。 | 扩 `tests/test_calibrated_workflow_service_reward.py`：empty/current-missing/预测目标/阈值分支、mask 合法、跨 split 状态隔离、同输入复测确定性、无伪训练 seed；`tests/test_algo_pool_contract.py` 的 registry 测试只作前置。 |

训练诊断脚本当前按 SA/MAPPO/PPO 的 checkpoint 和梯度结构写成，不应为了让 Popularity 入主表而改造成通用训练平台；对 DT 只补主表所需的同预算收据和消费端。

学习方法建议采用 B 已冻结的**待执行设计**：seeds `7/17/29/43/61`，每 cell 1,440 environment steps、24 updates、每次 60 transitions、4 PPO epochs、batch 32、192 optimizer steps；在 updates `6/12/18/24` 给相同选模机会，0 方法专属超参搜索。旧服务实验按 192 episodes 训练，实际 steps 为 `1,216–1,456`，不能称交互预算相同。规则方法无训练 seed，按每个原始窗口/工作流一次确定性结果入表。

## 原始区间并集和可用性

只读输入：`artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/historical_window_usage_registry.json`（截至当时的 1,209 个计划/结果文件元数据，668 个独特历史外层区间；其中 418 个区间身份仍为 unknown）、`configs/experiment/calibrated_continuous_workflow_interface_repair_v3_manifest.json`（36 实例）、`configs/experiment/typed_model_cache_formal_protocol_v1_20260820/{train,dev,formal,sealed_holdout}_window_plan.json`（60 个已选择/保留 I-80 窗口）和 `candidate_window_inventory.json`（579 个当时的 I-80 coverage 候选）。36 实例的 `source_segment_id` 与闭合原始 time interval 两两无交叠；但全部已进入 train/dev/regression/frozen-check，均为 development。按相同 segment 和原始 time interval 与旧登记交叉，36 个中 34 个至少与一项历史已登记区间重叠；另两个 Peachtree 项无“已登记的确定时间区间”匹配，**不能据此认定未消费**。

| 已消费 calibrated split | 数量 | 原始来源与 time/frame 概览 | 当前资格 |
| --- | ---: | --- | --- |
| train | 12 | Lankershim 2、US-101 10；各实例 24 帧；time/offset 见 manifest `source_interval` | development，不能转作新评价 |
| dev | 4 | Peachtree 1、US-101 3；各 24 帧 | 已用于 checkpoint selection |
| regression | 12 | Lankershim 4、Peachtree 1、US-101 7；各 24 帧 | 已暴露开发验证，不是 holdout |
| frozen_check | 8 | US-101 8；各 24 帧 | 已暴露开发检查，不是新 holdout |

历史 I-80 inventory 以 `i_80_run_001` 208、`run_002` 166、`run_003` 205 个互不重叠 24 帧候选构成。三 run 的**库存跨度**分别为原始 frame `12–9971` / time `1113433136100–1113434132000`，frame `27–7970` / time `1113436769600–1113437563900`，frame `1813–11628` / time `1113437746200–1113438727700`；跨度中的间隙不自动可用。四份 v1 计划已选择 train 24、dev 12、formal 12、sealed holdout 12，共 60 个 exact window ID。G14R23 已把截至 2026-09-27 的后续元数据去重，确认余下 519 个只满足当时的时间/24-frame 隔离；车辆复现排除后 14 个全在同一 run，无法建多 run 独立测试。详见 `cscwd_2027_i80_source_scope_audit_20261009.md`；sealed/旧 holdout 不重开。

**可立即作为新确认性 split 的范围：空集（当前认证数 train/dev/evaluation 均为 0）。**这不是断言 NGSIM 没有剩余物理帧。旧 registry 的 418 个 unknown 全部按保守来源范围排除 Lankershim/Peachtree/US-101，**不阻断 I-80**；I-80 真正的 blocker 是车辆复现和跨 segment-run/场景覆盖不足。9 月 27 日之后的新 artifact 历史仍须做增量元数据核对，不能给 519 个时间隔离候选补发永久未消费证明。

数据祖先：NGSIM 提供真实车辆移动轨迹，Alibaba `batch_task` 提供批处理 DAG 拓扑/时长/资源字段；二者不是共同观测的车载 AI workflow。模型/adapter 指派、状态比例、链路、车辆—DAG 配对与 deadline 为测量校准或人工假设。没有驾驶任务答案标签、真实 RSU association/无线时延，也没有真实 adapter 请求联合 trace。36-instance manifest 的 `source_files` 记录了窗口计划和 sampled DAG 的 SHA-256，但未直接给出完整 NGSIM 原始 CSV 的 hash；下一轮须把原始源 identity 补入独立 ledger。NGSIM 官方入口和 Alibaba 仓库见 `DATASET_SOURCES.md`；当前文档未取得两源原始数据再分发许可结论，后续公开数据卡需逐源复核。`LuST` 是保留 support provider，`highD` 原始源缺失；均不替换当前主线。

最低成本后续方案：复用 G14R23 的全历史元数据索引，只增量核查其后可能触及 I-80 的新 artifact；不重扫旧 3,957 文件或封存性能。现有同 run 两个相关性预览窗口若被保留，在每 run 最多 4 的规则下还至少需 10 个合格窗口来自至少 3 个其他 runs；更强的跨道路/捕获会话 claim 需要新来源。新源先绑定许可、原始 hash、segment/run、frame/time、车辆身份和 workflow 祖先，再按机制机会盲选互斥 train/dev/evaluation。若没有足够独立窗口，记 `inconclusive`，不复用旧 holdout 或靠 seed/workflow 数凑样本。

## 主表冻结草案与否证门

学习组 SA、PPO、controller-level MAPPO、DT 同列服务主表；Popularity 单列有预测的 stateful heuristic；two-step 单列 exact-transition model-based upper-capability comparator。逐方法报告 on-time/total completion、unfinished-after-deadline、failed-service attempt 与 episode 率、连续无进展、完成条件下 elapsed seconds **及完成样本覆盖率**，再列 model/state/input transfer bytes、recompute seconds、cache miss/eviction、训练/推理时间和 reward（次要）。失败 workflow 的传输、重算仍计入总体成本；条件时延不可用行不能以 0 填入。统计以不重叠原始窗口为外层，seed/workflow 为内层，报告 paired effect、percentile/BCa CI、sign test 和 Holm family。

任一窗口 identity/interval/哈希门失败，DT 信息越权，方法预算或 checkpoint 机会不等，缺原始行/命令/manifest/checkpoint hash，或 B 的 PopArt A/B 未同时通过机制与服务否证门，就停止算法晋级。PopArt 的 A/B 只检验学习稳定性，不能代替 SA 对强基线的优势证据。最多两项论文消融在**实际生效机制和主假设冻结后**选定；现有恢复对照可做系统机制实验，第二项不预设 sparse-tail prior、门控或任何有利开关。
