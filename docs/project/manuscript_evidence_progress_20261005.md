# 论文成果与问题台账

更新：2026-10-06。主稿：`system_mechanism_manuscript_working_draft.md` v1.5。
本次新增 calibrated workflow 服务目标对齐奖励的两奖励匹配训练与负结果；不调用模型、不下载资源、不读取旧 holdout，
也不改写历史 artifact。

## 2026-10-06 服务目标对齐奖励闭环

- 奖励 preflight 冻结 original 与唯一 `service_aligned_v1`；7 项排序、同动作转移不变和 truncation bootstrap 均通过。
- A/B 的 SA、MAPPO、PPO 全部新训练 3 seeds×192 episodes；理论/实际 steps=`82,944/24,952`，432 updates，
  90 checkpoints 本地保留；选模不读取 reward 数值。
- 候选使 regression completion 的 SA/MAPPO/PPO 从 `.889/.972/1.000` 变为 `.806/.833/.944`；frozen development
  从 `.917/.958/1.000` 变为 `.833/.875/.958`。PPO on-time 有描述性改善，但 failed-service/invalid prepare 新增。
- SA transfer/invalid prepare 下降，但连续无进展和未完成增加；不是服务改善。two-step 行为与 completion 完全不变，
  同轨迹双公式 return 只反映换评分尺。
- 最终分类 D；不进入 event auxiliary target 重设计消融。奖励调整不构成算法创新或 SA 机制证据。
- 原件：`artifacts/analysis/calibrated_workflow_service_reward_preflight_20261006_v1/`、零步启动失败 v1、唯一科学 run
  `artifacts/benchmarks/calibrated_workflow_service_reward_alignment_20261006_v2/`；完整报告见
  `calibrated_workflow_service_reward_alignment_20261006.md`。

## 2026-10-06 SA prepare/execution 单因素闭环

- 定位 artifact：`artifacts/analysis/calibrated_workflow_prepare_balance_diagnosis_20261006_v1/`。12 个事前规则选取 dev
  状态覆盖 current ready/missing、near/far 和 target feasible/infeasible；完整 dev replay 100 状态。确认本轮 raw-policy
  rollout/eval 实际跳过 margin/sharpening，temperature 生效，auxiliary/temporal 只进入训练 loss。
- 冻结候选：只把 `auxiliary_coef=0.1→0.0`；A 精确复用，B 新 3 seeds×192 episodes，理论/实际新 steps=
  `13,824/4,214`，72 updates，无重试、扩 seed、改 reward/环境/数据/网络。
- 结果 artifact：`artifacts/benchmarks/calibrated_workflow_prepare_balance_ablation_20261006_v1/`。regression/frozen
  completion 均与 A 相同（0.944/0.958）；frozen current-missing action 4 `50.0%→57.7%`，invalid prepare `56→86`，
  action 4 `54.7%→71.6%`。失败 seed 从 17 转到 29，没有消失。
- 结论：`存在其他未定位问题`；拒绝删除整个 auxiliary 作为修复。该负结果不支持算法创新或 auxiliary 普遍有益。
- 唯一后续：若另立任务，只重新设计同时感知 current service readiness 与 target feasibility 的 event auxiliary target；本轮不实现。

## 已完成成果及可用范围

| ID | 已解决的问题 | 可引用成果 | 原件入口 | 限制 |
|---|---|---|---|---|
| E01 | 后缀恢复是否保持执行内容 | 6/6 新配对输入/token/identity 一致，24 次调用 | `independent_recovery_cost_measurement_results_20261005.md` 与对应 artifact root | 单主机、单技术输入；不是交通任务正确性 |
| E02 | 旧成本能否预测新执行方向 | 6/6 选择较低成本 recovery | 同上 `aggregate_summary.json` | 全在 recovery 一侧，不能证明边界分类性能 |
| E03 | 绝对成本估计是否准确 | 中位绝对误差 0.369–1.944 s；保留高估 | 同上逐次测量表 | prepared recovery 相对误差 45.9%–49.7% |
| E04 | 后续驱逐是否影响决策 | 合成原生状态见证；12 点中修正规则匹配参考 10 点 | `eviction_aware_recovery_results_20261005.md` 与对应 artifact root | 开发点，非真实模型 victim→reload |
| E05 | 新规则是否胜过正确两步 | 12/12 持平，未发现性能/开销优势 | 同上 | 不支持原创通用算法或 RL 必要性 |
| E06 | 是否全面改善所有指标 | 汇总时间/传输下降，但多 5 次重跑、53.641216 s 重算 | 同上 | 不可只保留正向指标 |
| E07 | 合法 victim 是否导致真实后续重载 | native 合法驱逐 ALPR 后，PEFT 780 tensors→0；后续从本地文件重载为 780 tensors 并执行 n2；两臂均通过 | `real_cache_victim_reload_results_20261005.md` 与 `artifacts/analysis/real_cache_victim_reload_20261005_v1/` | 单主机、单实例、OS cache 未清、网络模拟 |
| E08 | 无驱逐时是否确实无外部代价 | base+双 adapter 对照的当前/未来请求均 `noop_all_resident`，两臂 action load/unload=0 | 同上 | 仅一个对照，不做统计外推 |
| E09 | 驱逐是否改变 restart/recovery 相对选择 | 两臂 lifecycle 为 1.022245/1.026997 s，因节点依赖相同而共同计费；两条件仍都选 recovery | 同上 | 未跨决策边界，不支持 action reversal |
| E10 | 历史公式是否影响实际比较而非只影响文字 | 旧 action 0 未执行 target 当前模型 transaction，online scorer、实际 branch score 和 offline reference 同时受影响 | `recovery_cost_defect_impact_report_20261006.md` | 只读缺陷报告，不把修复预判为通过 |
| E11 | 原 12 点公平复算后还剩多少方法优势 | 四方法 12/12 动作、成本、完成和 reference gap 相同；旧 10/12 vs 7/12 优势完全撤销 | `recovery_cost_corrected_matrix_results_20261006.md` 与对应 artifact | synthetic development correction，非独立验证 |
| E12 | 正确规则是否新增能力或只与简单规则持平 | 新 6 点三在线方法 6/6 相同；在 2 个冻结 link 误差边界共同错选 | `recovery_cost_boundary_check_results_20261006.md` 与对应 artifact | analytic boundary check，不是现实误差分布 |
| E13 | 共享 base、当前 resident、无驱逐是否正确记账 | b02/b03/b04 全部逐事件通过；共同 lifecycle 在两路各计一次且 6/6 相同 | 同上 `all_method_results.json` | 没有 action-specific cache 差异或算法收益 |
| E14 | 纠正版小型连续 workflow 是否存在非退化序列决策 | depth-4 检查 1,183 个可达状态、4,962 个合法状态—动作；对称传输误差 0、actual link 泄漏 0、两规则 406 次分歧 | `artifacts/calibrated_continuous_workflow_non_degeneracy_20261006/` | 只证明非明显退化，不证明 RL 优势或最优策略复杂性 |
| E15 | 当前 SA-GHMAPPO 是否在公平小预算下优于强基线 | `Unverifiable`；虽观测到 SA completion 0.750、PPO/MAPPO/两 planner 为 1.000，但 executed-action likelihood、mobility、encoder 与 planner capability 不匹配 | `artifacts/calibrated_continuous_workflow_pilot_v2_20261006/` 与 `calibrated_workflow_interface_defect_report_20261006.md` | 执行结果保留为 interface diagnostic，不作方法排名 |
| E16 | DAG dependency message passing 是否有单因素贡献 | 未建立；当前 defective contract 下 full minus no-dependency completion `-0.167 [-0.250,-0.083]`，coverage `-0.120 [-0.192,-0.050]`，reward `-6.375 [-10.336,-2.596]` | `sa_ghmappo_innovation_fair_training_v2_20261006.md` 与逐行 artifact | 负向敏感性信号，不是 paper-grade 因果消融 |
| E17 | 修复接口后 SA 是否有公平相对收益 | 没有；regression / frozen-development completion 为 SA `0.944/0.958`，PPO/MAPPO/rule 均 `1.000`；1,340-row head replay 全匹配 | `calibrated_workflow_interface_repair_results_20261006.md` 与 v3 artifact | 只支持 interface recovery；新检查不是独立 holdout，event 偏置未做消融 |

## 正在解决和未解决的问题

- P01，RESOLVED / HISTORICAL CLAIM WITHDRAWN：已确认旧公式、旧 action-0 branch 和旧 offline reference 同时路径不对称；旧 artifact 保留，比较性结论撤销，并由 create-only 纠正版替代。
- P02，RESOLVED BOUNDED：合法 victim→实际 PEFT object/tensor 卸载→后续本地文件重载→n2 执行已闭环；logical resident、runtime 对象、磁盘文件和未控制 OS cache 四层分开。
- P03，PARTIAL / BOUNDED：6 点边界检查覆盖两侧并保留两个估计错选，但只来自 analytic link mismatch，不是独立真实分布；不得包装为校准性能。
- P04，OPEN：远端成本、任务标签及外部代表性。暂不建设共享队列或跨 workflow 平台；是否需要由最终主张决定。
- P05，OPEN：系统组合的新颖性及投稿定位；引用已有成本感知思想，不用润色填补增量价值缺口。
- P06，INTERFACE RESOLVED / METHOD BLOCKED：hierarchical likelihood、failure-time mobility、byte/typed feature consumption
  已在显式 v3 profile 中修复并有界重训；旧冻结循环消失。SA 仍在 current-missing 时约一半选择 action 4，且没有相对
  PPO/MAPPO/rule 收益。事件增强缺少单因素消融，新检查不是独立 holdout；论文贡献继续收紧，最终决策仍 D。

## 本轮收口结论

最小真实生命周期与对称事件核算已经得到验证，无需继续同侧重复。修正后旧方法优势为 0，正确简单阈值、完整规则与正确两步
前瞻持平。当前最强贡献收敛为 state recovery 与 adapter lifecycle 一体化实现、可复现成本审计及失效条件。若仍主张 eviction
改变动作，必须先有外部动机充分且真正 action-specific 的合法 resident 转移；不得靠不对称计费制造翻转，也不据此扩建多租户
平台、启动 RL 或追加模型测量。

## v1.5 calibrated workflow interface repair 追加记录

- 执行 source commit：`f00d212d681af2215062e0cb47e6bec0a5e49240`；唯一科学 run 为
  `calibrated_continuous_workflow_interface_repair_20261006_v1`，实际 12,478 steps、216 updates、51.70 s，无重试。
- 修复前只读证据：旧 action-4 循环由边缘概率聚合、likelihood 错位和 node-index mobility 冻结共同造成；flat/graph
  encoder 另有单位与关键字段消费缺陷。旧结果、checkpoint 与 integrity 保持不变。
- 修复后结果：SA/PPO/MAPPO/rule 在 regression 为 `0.944/1/1/1`，frozen development 为 `0.958/1/1/1`；旧状态冻结
  消失，但 SA current-missing action-4 rate 仍约 50%。不支持 SA 相对收益或单一 event 机制归因。
- review identity：`reviewed_at=2026-10-06`，`literature_cutoff=2026-10-06`，`target_venue=IEEE TMC`，
  `artifact_run_id=calibrated_continuous_workflow_interface_repair_20261006_v1`，`policy_version=tmc_review_policy_v3_20260621`，
  `evidence_level=E1_DOCUMENTED_WITH_AUDITED_NONFORMAL_DEVELOPMENT_VALIDATION`；paper-ready=`Unverifiable`。

## 后续追加格式

每次完成后在此追加：日期、问题 ID、实际改动、执行 commit、原始 artifact 路径、输入/输出身份、正负结果、已解决/未解决项及论文对应章节。先保存原件，再改总结；不覆盖失败或历史版本。此台账不等同后台监控，也不自动授权任何执行。

## v1.1 编辑维护记录（历史）

- delivery baseline：`510bc43bb4e67cf72803e54c4b391a4518b2b821`。
- reviewed_at/literature_cutoff：2026-10-05；本次未刷新文献或评价 novelty。
- target_venue：IEEE TMC（目标，不是就绪判断）。
- policy_version：`tmc_review_policy_v3_20260621`。
- artifact_run_id：无新增科学 run；证据等级为编辑性文档复核，不重新授予 E2/E3。
- 完成：主稿澄清 RQ 对应证据、全路径成本分解、冻结公式的适用边界；建立本台账。
- 未完成：P02–P05，以及 P01 的生产端核对；paper-ready 未成立。

## v1.2 真实生命周期追加记录

- execution commit：`f31024d957bd07718c4087fa55d8f7bb707e0f6e`；唯一执行 `12/12 generate`，scientific wall
  `64.309679 s`，无重试。
- artifact：`artifacts/analysis/real_cache_victim_reload_20261005_v1/`；22/22 manifest 文件独立 size/hash 复算通过。
- 输入/输出身份：同一技术图像 `83af0966…c307`；状态 `2,195 B`，原图 `192,757 B`；n0、n1、n2 的声明保真检查全通过，任务正确性 unavailable。
- 正结果：ALPR `780 tensors / 154,308,608 B` 实际卸载为 0 后，在后续节点前真实重载；Helmet 同样 `520→0`；权重文件 hash 不变。control 无 action load/unload。
- 负结果：预测仍显著偏高；victim 对两臂是共同成本，未发生决策翻转或边界跨越；网络、queue、任务标签与泛化仍未覆盖。
- claim 更新：支持真实 lifecycle mechanism；削弱并删除 recovery-only eviction penalty、真实 victim 已改变动作、算法原创性和 paper-ready 暗示。
- review identity：`reviewed_at=2026-10-05`，`literature_cutoff=2026-10-05`，`target_venue=IEEE TMC`，
  `artifact_run_id=real_cache_victim_reload_20261005_v1`，`policy_version=tmc_review_policy_v3_20260621`，
  `evidence_level=E2_ARTIFACT_AUDITED`（bounded same-host lifecycle；network assumed）。

## v1.3 成本纠错与公平复算记录

- 只读缺陷提交 `4e897bf…`；实现冻结 `800f0a1…`；原 12 点执行提交 `800f0a1…`；边界执行提交 `3069117…`。
- 原矩阵 artifact：`eviction_aware_recovery_corrected_20261006_v2`，12 点、48 隔离 path、0 model call；四方法
  12/12 动作和成本相同，历史新增正确动作 `5→0`。
- 边界 artifact：`recovery_cost_boundary_check_20261006_v1`，6 点、24 隔离 path、0 model call；三在线规则 6/6
  相同、匹配参考 4/6；两个固定 link estimate mismatch 的 gap 为 `0.001472/0.004626 s`。
- 保留：production state fidelity、真实 PEFT victim→reload、shared-base dependency safety、负面边界和估计误差风险。
- 撤销：10/12 vs 7/12、相对简单阈值 17.993 s / 972,125,684 B 优势、d10/d11 作为纠正后错选证据，以及任何
  优于正确两步前瞻的暗示。
- review identity：`reviewed_at=2026-10-06`，`literature_cutoff=2026-10-05`，`target_venue=IEEE TMC`，
  `artifact_run_id=eviction_aware_recovery_corrected_20261006_v2 + recovery_cost_boundary_check_20261006_v1`，
  `policy_version=tmc_review_policy_v3_20260621`，`evidence_level=E2_ARTIFACT_AUDITED`（bounded synthetic correction；
  network modeled）。

## v1.4 SA-GHMAPPO 创新候选与小预算训练尝试记录

- 执行 source commit `b418eb4…`，确认 `d67575b…` 为祖先；旧不对称 scorer、旧 action 0、旧 offline reference 未用于
  新结论。live remote 在执行前不可验证，cached `origin/codex/manuscript-evidence-v1=b3a00b6…` 不含纠正提交。
- 唯一候选：DAG 节点的 current/predicted/target-RSU adapter-residency feature 沿依赖边消息传递；不把 MAPPO、GNN、
  shared base、adapter prefetch 或 checkpoint recovery 当原创。
- frozen config/manifest 为 v2；train/dev/evaluation=`12/4/12`，跨 split 原始 frame interval overlap=`0/0/0`；预算上限
  main/ablation/total=`27,648/9,216/36,864`，实际=`8,154/2,873/11,027`。
- 方法：SA-GHMAPPO、PPO、controller-level MAPPO、immediate/two-step model-based planners，以及只关闭
  `use_dependency_aware` 的同架构敏感性臂；3 seeds × 128 episodes，不做搜索，不用 evaluation 选模。
- 观测：SA completion 0.750；no-dependency 0.917；PPO、MAPPO 和两 planner 均 1.000；full minus ablation
  completion `-0.167 [-0.250,-0.083]`。事后接口审计使公平排名与机制因果结论 `Unverifiable`；最终决策 D。
- 原件：`artifacts/calibrated_continuous_workflow_pilot_v2_20261006/`；日志：同名 `.log`；直接论文补丁：
  `manuscript_experiment_section_sa_ghmappo_pilot_v2_20261006.md`。
- review identity：`reviewed_at=2026-10-06`，`literature_cutoff=2026-10-06`，`target_venue=IEEE TMC`，
  `artifact_run_id=calibrated_continuous_workflow_pilot_v2_20261006`，`policy_version=tmc_review_policy_v3_20260621`，
  `evidence_level=E1_DOCUMENTED_WITH_AUDITED_NONFORMAL_PILOT`；paper-ready 仍为 `Unverifiable`。
