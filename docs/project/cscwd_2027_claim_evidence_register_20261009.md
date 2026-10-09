# CSCWD 作者审阅稿：主张、原件与实验接口台账

## 身份与判定

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09（沿用仓库已有来源；本轮未作新文献检索）
- `target_venue`: CSCWD 2027（作者拟投；本轮未核对当届官方页数、日期或模板）
- `artifact_run_id`: `independent_recovery_cost_measurement_20261005_v1` + `real_cache_victim_reload_20261005_v1` + `eviction_aware_recovery_corrected_20261006_v2` + `recovery_cost_boundary_check_20261006_v1` + `calibrated_workflow_service_reward_learning_diagnosis_20261009_v1`（仅非正式诊断）；算法主比较待 B 冻结
- `policy_version`: `tmc_review_policy_v3_20260621`（内部审查规则，不是 CSCWD 的录用标准）
- `git_commit_at_review`: A 文稿基线 `3bdfa1ea886ca4298cd9b06b5fb023099008bf84`；B 诊断源码身份 `70a83afb5d3d3b3fb40fcaeaf825a82e4b74c416`、证据交付提交 `f8baf30f8d6ed37898804f570d8ebfcbeaa50ad9`；v70 独立审查 `3b5ab4f`；投稿 scientific commit 待 B 冻结
- `evidence_level`: 四个有界包为各自范围的 `E2_ARTIFACT_AUDITED`，B 学习诊断为 `E2_ARTIFACT_AUDITED_NONFORMAL_OFFLINE_DIAGNOSIS`；新算法主比较及独立评价为 `E0_UNAVAILABLE`。旧 v70 为 `E1_PLUS_RAW_PARTIAL` 且存在协议/独立性 blocker。
- `verdict`: `AUTHOR_REVIEW_DRAFT / ALGORITHM_CONTRIBUTION_UNVERIFIED / NOT_SUBMISSION_READY`

唯一当前主稿：`cscwd_2027_manuscript_working_draft.md`（作者审阅版 2026-10-09）。旧 `system_mechanism_manuscript_working_draft.md` v1.3 是机制与成本纠错阶段工作稿，不与新主稿拼成两套结果。正文避免内部轮次标签；本台账保留 artifact 和版本细节。

强基线接线、能力披露与原始区间并集见 `cscwd_2027_baseline_data_readiness_20261009.md`；I-80 的来源范围、418 unknown 排除作用和 G14R23 车辆复现边界见 `cscwd_2027_i80_source_scope_audit_20261009.md`。拟用接口为 `calibrated_workflow_interface_v2 + service_aligned_v1 + independent_heads_executed_env_v2`，但可用于论文结果的 scientific commit 未冻结。PopArt 仅为已有 critic normalization 技术的单变量开发 A/B；B 的首次启动在 run root 前失败，没有科学结果，不能写作原创算法或已改善服务。

## 原件完整性核对

本轮只读逐文件重算四个 manifest；72/72、22/22、7/7、5/5 项文件的 size 和 SHA-256 均一致。核验到两个 synthetic 终态为 `COMPLETED`，分别为 12 点/48 隔离路径与 6 点/24 隔离路径；这两包模型调用均为 0。核验结果只支持各包自身的有界完整性，不把四包合称为一个端到端真实系统实验。

| 证据层 | 原始入口 | 执行身份与状态 | 不可跨越的限制 |
| --- | --- | --- | --- |
| 两节点真实后缀与成本 | `artifacts/analysis/independent_recovery_cost_measurement_20261005_v1/{frozen_plan.json,all_measurements.json,aggregate_summary.json,terminal_receipt.json,integrity_manifest.json}` | 固定输入、24 次 generate、6 配对；6/6 决策同侧；72/72 文件核验 | 单主机技术图像；无线项为公式；任务正确性 unavailable；不能估计边界准确率 |
| 真实 adapter victim→reload | `artifacts/analysis/real_cache_victim_reload_20261005_v1/{frozen_plan.json,all_measurements.json,terminal_receipt.json,integrity_manifest.json}` | 执行提交 `f31024d`；12/12 generate；22/22 文件核验 | 两条件各一配对；权重留在本地、OS cache 未清；无远端模型传输、任务标签或统计区间 |
| 原 12 点对称纠正 | `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/{frozen_protocol.json,all_method_results.json,old_to_new_point_comparison.json,aggregate_summary.json,completion_receipt.json,integrity_manifest.json}` | 实现/执行 `800f0a1`；12 点、48 隔离路径；7/7 文件核验 | synthetic development correction；网络和状态字节为冻结建模量；不是独立真实工作负载 |
| 六点边界 | `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/{frozen_protocol.json,all_method_results.json,aggregate_summary.json,completion_receipt.json,integrity_manifest.json}` | 执行 `3069117`；6 点、24 隔离路径；5/5 文件核验 | analytic link-estimate stress，不是无线测量或现实误差分布 |
| 旧控制器奖励包 | `artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/`，审查见 `cscwd_2027_baseline_contribution_audit_20261009.md` | 数值可作归档开发描述 | 计划/实际窗口 ID 交集 0、full 177 对重叠、训练更新和选模预算不同；CI/Holm 和同预算领先主张停用 |
| 服务奖励学习诊断 | `artifacts/analysis/calibrated_workflow_service_reward_learning_diagnosis_20261009_v1/`、`docs/project/calibrated_workflow_service_reward_learning_diagnosis_20261009.md`（B 分支只读） | 360 episode/3,161 step 转移/奖励零 mismatch；九个 service learned cell 的 critic EV 近零、固定 dev V:P 梯度比 539–7,907 | 非正式离线诊断；缺原训练 minibatch 更新序列，不能推出 PopArt 因果改善或 SA 优势 |
| PopArt 启动状态 | B 提交 `8ebaf9ce` 中 `calibrated_workflow_value_normalization_ab_launch_attempt_20261009_v1/failure_receipt.json` | 授权尝试在 run root 前结束；科学 steps/updates/checkpoints/评价行均为 0 | `E1_PREFLIGHT_AND_PRE_RUN_FAILURE_RECEIPT`；不产生 A/B 效果结论，后续新 root 仍须独立冻结与审查 |

## 主张—证据—待补实验

| ID / 主稿位置 | 可写主张 | 直接原件 | 证据边界 / 待补项 |
| --- | --- | --- | --- |
| C1 / §§3–4 | 单一连续 DAG 的 typed base/adapter 依赖、容量与节点边界状态接口可执行 | production action-4 状态收据；真实后缀包；真实 victim/reload 包 | 有界同机实现；未证明真实跨 RSU 传输、跨 workflow 共享或任务语义正确性。对这项组合是否形成文献增量仍待审稿定位。 |
| C2 / §§1,4.3,6.4 | 工作流感知准备控制器是待验证候选 | 拟用三元接口和执行动作 PPO 已在 B 文档/配置定位，投稿科学提交与 PopArt A/B **待 B 冻结** | `UNVERIFIED`；需 SA/PPO/controller-level MAPPO/DT 的同信息、同动作、同训练/选模预算新比较。不得把旧 v70 sparse-tail/门控公式与新结果拼用。 |
| C3 / §§3.3,5–6 | 双路径对称事件核算与服务/成本共同报告能揭示机制、持平和估计错选 | 纠正 12 点与六点边界包；真实 adapter/后缀收据 | 已验证有界可复算性，不证明总体性能或新算法；无线/queue/独立任务分布待补。 |
| R01 / §6.1 | 恢复后缀的输入与 token 在 6 个新配对中相等，省去 prefix 计算 | `independent_recovery_cost_measurement_20261005_v1/all_measurements.json` | 6/6 同侧，不是边界分类率；2,185 B 状态与 192,757 B 输入分列，网络仍建模。 |
| R02 / §6.2 | 合法 victim 触发真实 PEFT 对象移除与后续本地重载 | `real_cache_victim_reload_20261005_v1/all_measurements.json` | ALPR 780 tensors→0→780；真实 lifecycle 两臂共同计费约 1.02 s，不能声称驱逐导致动作翻转。 |
| R03 / §6.3 | 修正后简单阈值、完整规则和正确两步在原 12 点全部持平 | `eviction_aware_recovery_corrected_20261006_v2/aggregate_summary.json` | 12/12 synthetic 开发点；旧 10/12 对 7/12 优势撤销。 |
| R04 / §6.3 | 六点边界中在线规则共同发生两次近边界估计错选 | `recovery_cost_boundary_check_20261006_v1/all_method_results.json` | 4/6 匹配离线参考；误差 0.001472/0.004626 modeled s，不能外推错误率。 |
| R04b / §6.4 | 服务对齐奖励的非正式开发训练未改善三种 learned method 的总完成，critic 尺度诊断给出单变量 A/B 动机 | `calibrated_workflow_service_reward_alignment_20261006_v2` 与 `calibrated_workflow_service_reward_learning_diagnosis_20261009_v1` 原件及 B 报告 | 36 实例已消费；诊断无新训练、无因果结果。不得把 PopArt 当原创方法或把旧开发表晋级主表。 |
| R05 / §6.4 | 控制器提高完成、连续或净成本 | **[B-RESULT PENDING]** | 目前没有可引用数字、CI、checkpoint manifest 或独立窗口统计。 |

## 研究问题与主比较、消融

| 问题 | 必要比较 | 主要结果与共同代价 | 当前状态 |
| --- | --- | --- | --- |
| RQ1：依赖、驻留与状态如何决定连续执行？ | 同一节点依赖下 continuous/restart/recovery；真实 victim 对无驱逐对照 | suffix 输入/token 保真、完成节点、action 内 load/unload、状态/模型/动态字节、local wall | 有界真实 witness；缺交通任务标签、远端链路 |
| RQ2：对称核算后恢复规则是否比正确简单规则有新增决策能力？ | 简单阈值、相同信息的两步规则、完整规则、事后微型参考 | 同完成量时的动作、modeled time、总 bytes、victim/reload、错选代价 | 原 12 点及六点已完成；目前答案为持平/共同错选，不据此主张新算法 |
| RQ3：工作流感知学习是否增进服务与成本结果？ | **学习主比较**：冻结 SA、PPO、controller-level MAPPO、project-native DT；Popularity 为有预测的 stateful heuristic，two-step 为 exact-transition model-based 能力对照，不称同容量学习基线 | on-time/total completion、failed attempt/episode、连续无进展、完成条件下时延及覆盖、model/state/input bytes、recompute、miss/eviction、reward secondary；原始区间外层配对统计 | **[B-FREEZE + B-RESULT PENDING]**；DT/Popularity 尚未进入完整 service runner，独立评价区间未核证 |
| 消融 A1（RQ1） | 禁用状态恢复并执行对称 restart；模型请求与目标 cache 初态相同 | prefix 重算、状态字节、完成与时延；不得把失败请求省下的字节当收益 | 已有技术对照；投稿版若更换 workload 须重新冻结 |
| 消融 A2（RQ3，条件项） | 科学版本冻结后仅针对确实生效、可独立关闭的机制预先定义；不得默认 v70 sparse-tail prior 或预测门控 | 对应的服务/准备机制指标、完成与成本；同预算重训 | **[B-ABLATION IDENTITY PENDING]**；若没有合法独立开关，则只保留 A1。PopArt A/B 不冒充 SA 特有消融 |

最多保留上述两项消融。A2 必须随有效方法身份先验冻结，不能从历史模块名或有利结果中挑选。两步规则使用 exact transition 与字典序目标，模型能力单列；其 outcome 不承担同能力 learned baseline 的统计归因。

## 撤销与降级清单

1. 旧驱逐规则 `10/12` 对简单阈值 `7/12`，以及所谓额外 5 个正确动作：**撤销**。对称复算四方法 12/12 同动作、同成本、同完成。
2. 旧相对简单阈值约 `17.993 s / 972,125,684 B` 优势及 d10/d11 纠正后错选解释：**撤销**。旧原件只作缺陷历史，不能进入新主稿结果表。
3. 真实 victim 已改变 restart/recovery 选择：**撤销**。两臂都卸载/重载，共同 lifecycle 自然抵消；恢复胜出来自省去 prefix。
4. 旧 v70 的 mixed/full 奖励均值：**降级为描述性开发记录**。计划 Peachtree 与 rows Lankershim 错配；full 48 窗口 177 对重叠、基线更新/选模不匹配。原 48-window BCa CI、Holm p、同预算算法领先和独立验证语句：**不得用于新主稿**。
5. 共享结构带来的 62.2% 传输下降：**仅归于固定顺序/共享配置机制对照**，不归因于 SA 策略或真实网络吞吐。
6. 真实恢复 package、PEFT unload/reload 和 synthetic 成本矩阵：**分别保留**，不合并为一个跨 RSU 端到端真实部署试验。

## 给 B 的版本选择与接口清单（保存于此，不在本任务内发消息）

在替换主稿的 `[B-*]` 栏之前，需要一次性交付并冻结：

1. 投稿候选的科学 Git commit、profile/config 及语义 hash；精确方法身份、actor/critic 输入、动作分布/合法 mask、训练 loss、推理期模块、参数量与 inference cost。说明与旧 controller、calibrated workflow 和 service reward 诊断各自的版本关系。
2. PPO/controller-level MAPPO/DT 的相同观察信息与动作权限，DT 额外预测字段逐项审计；Popularity 的计数记忆、预测准备及 reset 语义披露；two-step 的 exact-transition clone 和字典序目标单列，禁止真实未来轨迹或 outcome 泄漏。
3. 同预算训练与选择合同：学习方法建议 5 seeds、1,440 environment steps、24 updates、192 optimizer steps、四次同等 checkpoint 选择机会；记录实际数、搜索机会、各 checkpoint path/hash/manifest 和训练/推理时间。规则不复制成多训练 seed。
4. 数据/窗口身份：NGSIM/Alibaba 版本与 source hash、workflow IDs、train/dev/evaluation 原始 frame/time interval、互斥审计、窗口计划与实际 rows 一一对应；已消费 split 不冒充新 holdout。
5. 原始 per-seed/window/workflow rows 与完整命令/环境/manifest：on-time/total completion、unfinished-after-deadline、failed-service attempt/episode、连续无进展、完成条件下 delay 及缺失覆盖、model/base/adapter/state/input bytes、recompute、miss/eviction、reward；各指标单位、分母和失败处理。
6. 预注册配对统计与全部结果：外层不重叠原始窗口、内层 seed/workflow，percentile/BCa 95% CI、窗口层效应量、sign test、Holm family；负结果与不利成本完整保留。
7. A2 消融若存在，交付实际生效机制的独立 toggle 与同预算重训收据；PopArt A/B 的机制/服务否证门另行报告，不能代替 SA 优势。若 controller 公式或特征仍在变化，标为 `NOT_FROZEN`，主稿只保留系统机制贡献，不预填收益。

当届会议页数、模板、截止时间和参考文献终版 metadata 仍需从官方渠道在投稿前核对。本台账不授权新科学执行，也不改变 B 线协议或结果。
