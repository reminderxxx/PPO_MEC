> 当前状态：2026-10-23已完成30/30单元，操作合同、原冻结完成审计（990 checkpoints）及真实曲线导出通过；文中启动/待执行段落为历史记录，以文末完成回执为准。

# 公平训练资源检查修复与续作（2026-10-22）

用户在上一轮异常退出报告后明确要求“继续”。本轮只修资源进程扫描并完成原公平训练，不改算法、奖励、窗口、seed、梯度更新规则或单元预算；不进入算法优化主线，不开启formal/holdout。

原run `repaired_fair_reward_training_20261021_v1` 在资源扫描严格UTF-8解码失败后保留，SAGH seed13只有38912步/152次调用，最后保存checkpoint144，不能精确恢复中断状态。原10个seed7完整单元经逐cell原生receipt/概率/分类/预算与33个checkpoint核验后只读引用；旧seed13 partial保留且不进入接受曲线。只对未完成的20个单元重新从零权重/optimizer/replay执行，其中seed13 SAGH也是fresh完整196608步；不拼接旧partial。中断尝试的38912步作为额外失败计算开销公开，不冒充接受单元预算。

实现仅在 `codex/fair-reward-resource-guard`。`scripts/continue_fair_reward_training.py`为独立启动层，原source commit `45c8c88e390f49fc4e66c70871c2318e18883f24`、其脚本/源码和scientific frozen_plan字节全部不改；原auditor/plotter消费接口不变。启动层仅将resource scan替换为安全字节解码和fail-closed等待，并拦截run_cell：对冻结manifest中全部完整单元只读跳过，其余调用原run_cell。新root用指向完整原cell的目录symlink，保留原文件和hash；新fresh单元由原runner直接产生。运行时替换显式记录在独立continuation_contract/execution_receipt，绝不伪称原始未修复进程仍在运行。

资源检查读取ps原始bytes，UTF-8无效字节用backslashreplace可见转义，保留训练脚本识别；进程参数引号不完整时用token fallback，不能让坏尾参数隐藏训练脚本。ps暂不可读/失败时写RESOURCE_SCAN_UNAVAILABLE并等待后重试，不视为资源空闲。只让本实验等待，不杀停/改优先级其他任务；CPU单线程、nice19、单cell顺序。

新run计划 `artifacts/analysis/repaired_fair_reward_continuation_20261022_v1`。scientific plan SHA仍为36f019938092eed36bb5bd34e23d88532156db37aae88c2704fa2f689dd4eef4，全部30cell最终各196608交互/768调用机会/32循环/33固定验证，实际Adam接受/拒绝步另报。共同24train+12validation窗口、两workflow、native reward offset0与原探索bonus/auxiliary目标不变。原完整10cell全量引用，不按reward或排名选择；seed13/29各10cell fresh执行，旧partial任何文件不写入新cell。

continuation_contract另冻结launcher commit/hash、parent_root、全部继承文件inventory与hash、保留partial inventory、pending列表及原命令/故障日志hash。prepare/run/verify均create-only或只读，任何原文件变化/重复运行/未冻结cell都拒绝；execution_lock和attempt_receipts保留，进程失败不自动重启。所有原cell与partial的hash在启动前和完成后核验。新root的frozen_plan为原文件原样复制，因此报告必须同时记录科学source/plan和操作启动层的commit/contract，不混淆两种provenance。

先做无效UTF-8、引号异常、ps失败等待、其他训练让出、继承不重训、pending精确参数转交及拒绝覆盖测试；真实预检为4算法×2seed，每512交互/2循环，先完成parent seed7再通过新启动层引用4cell并fresh完成seed13。运行原只读审计/绘图验证全部8cell。预检不用于效果或收敛结论。

全部30cell完成后先调用新启动层verify核验原现场，再用原source的audit_long_fair_reward_training.py核验全30cell、原source的plot_long_fair_reward_training.py导出真实原值/100episode trailing mean和33个不平滑验证点。figure create-only、全seed保留、mean/SD只取共同观察前缀，不平移/补尾/归一化/插值/筛选；附continuation provenance和失败说明再打包。恢复20分钟提醒，正常运行安静，仅完成/明确失败/需要处理时通知；故障保留现场，不自动修复或重跑。

reviewed_at: 2026-10-22；literature_cutoff: 2026-09-28（未新评novelty）；target_venue: CSCWD2027；artifact_run_id: repaired_fair_reward_continuation_20261022_v1；policy_version: tmc_review_policy_v3_20260621；scientific Git: 45c8c88；launcher Git由continuation_contract记录；evidence: 实现与预检待执行，完整训练UNVERIFIED。此轮matched development，非独立holdout/paper-ready，不保证SAGH上升、收敛或领先。

## 已验证并启动

322项相关测试通过；命令为`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -B -m pytest tests/test_fair_reward_continuation.py tests/test_repaired_probability_contract.py tests/test_window_selection_contract.py tests/test_env_contract.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py tests/test_long_fair_reward_training.py tests/test_fair_reward_training.py tests/test_training_reward_plots.py -q`，在独立launcher worktree执行。

真实预检root `artifacts/analysis/fair_reward_resource_guard_preflight_20261022_v1`完成4继承+4fresh共8cell，各512steps/2cycles，原audit_long_fair_reward_training.py审计PASS、24checkpoints、原plot_long_fair_reward_training.py出图PASS，SAGH双面板目视通过。证据为preflight_validation_summary.json、completion_audit.json和figures/report.json；预检不用于效果结论。

正式continuation root `artifacts/analysis/repaired_fair_reward_continuation_20261022_v1`已于2026-10-22T15:21:23.109713+08:00启动，PID81776仅作线索，实际命令和路径以execution_receipt为准。launcher Git71ca85dcac7f709bc27fc1c3e115f771f50a4cb4已push；独立操作contract hash7e94571e9dab4ef187076e6716127224bfeb6e19afe07f05ddc8717ada3e3deb，科学source45c8c88及原plan hash36f019938092eed36bb5bd34e23d88532156db37aae88c2704fa2f689dd4eef4保持原样。

正式10个继承单元已逐cell核验全部原生账本、概率、分类、budget和330个checkpoint；所有文件inventory/hash、旧partial、原故障报告/日志绑定于contract。新进程日志记录10次REUSE_COMPLETED_READ_ONLY，随后SAGH seed13从零开始。startup_check确认fresh initial与原seed13 initial各tensor相等，update_count0、optimizer空，已出现有效梯度步且初始logp误差合格，只有1个新attempt receipt。没有从旧partial checkpoint加载或从38912步续接。

提醒ppo-mec-3已重新建立，20分钟监控新root；原deleted提醒不监控旧run。两处冻结worktree的HEAD和文件均不得变动；后续完成文档只在主工作区提交。原七个用户修改保护hash复核一致。完整结果尚待20个新cell完成，不保证收敛、上升或领先。

启动命令（已执行，不可重复run）：

```bash
PPO_MEC_EXPERIMENT_PROJECT_ROOT=/Users/howen/Projects/PPO_MEC PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -B /Users/howen/.codex/worktrees/fair-reward-resource-guard/PPO_MEC/scripts/continue_fair_reward_training.py run --output-root /Users/howen/Projects/PPO_MEC/artifacts/analysis/repaired_fair_reward_continuation_20261022_v1
```

## 完成回执（2026-10-23）

本轮30/30单元完整：10个seed7原样只读继承、20个seed13/29 fresh完成。每cell196608交互、768次learner机会、32循环、33固定验证点；合计5898240交互、23040次learner机会、479461个训练episode、23760个验证unit与990个checkpoint。旧故障38912步/152调用为额外失败开销，partial保留排除，不拼接曲线。

按顺序执行冻结launcher的`verify --output-root`得到`OPERATIONAL_CONTRACT_VERIFIED`；原科学source的`audit_long_fair_reward_training.py --run-root`得到`PASS cells=30 checkpoints=990`（实际reviewed_at见completion_audit.json）；原`plot_long_fair_reward_training.py --run-root`得到`EXPORTED cells=30`。均使用主工作区`.venv/bin/python -B`和已冻结PROJECT/单线程环境；绘图仅增加原指定临时PYTHONPATH/MPLCONFIGDIR。未改两处source HEAD、原数据/plan/checkpoint/预算/奖励或七个用户修改。操作核验还确认继承inventory、旧partial及故障报告/日志hash；审计覆盖原生奖励和执行账本、概率合同、更新KL、窗口真实身份/互斥、分类及fresh初始化。

13张PNG（同步PDF）全部目视检查，原导出hash复核。保留所有原值和seed、100episode trailing mean、33个不平滑验证点；训练周期波动和负向结果均保留。28/30个seed满足事前末5点奖励平台规则；DQN seed7/29未满足，DQN seed7完成率亦低于0.8。SAGH三个seed都有网络和少量同状态执行动作变化，但固定验证reward下降，未显示验证收益，最终也未领先若干基线；平台不能解释为参数/策略收敛。完整逐seed初始/最终、正负结果、有效更新数量及局限只保存在本地报告，不将详细实验包上传公开仓库。

有效Adam步与768次learner机会分开报告。策略方法使用原始接受/拒绝/回退账本；DQN/Dueling DQN/QMIX依据native epoch日志与最终Adam状态计数交叉核验，每seed3072步。它们无KL候选拒绝/回退机制，相关字段标NA；原审计对缺字段聚合出的0不能解释为没有更新。

本地root下`completion_review.md/json`、`seed_result_summary.csv`、`algorithm_result_summary.csv`、`visual_and_postprocessing_check.json`和`delivery_manifest.json`记录完整结果与证据。交付包为`artifacts/analysis/repaired_fair_reward_continuation_20261022_v1_delivery.zip`，包含全部figures/CSV/report、操作contract/provenance及原平线/下降/资源故障诊断说明，不包含checkpoint或原始mobility/workflow数据；CRC与逐member hash核验见`delivery_receipt.json`。

reviewed_at: 2026-10-23（精确时刻见本地报告）；literature_cutoff: 2026-09-28（未新评novelty）；target_venue: CSCWD2027；artifact_run_id: repaired_fair_reward_continuation_20261022_v1；policy_version: tmc_review_policy_v3_20260621；scientific Git: 45c8c88e390f49fc4e66c70871c2318e18883f24；launcher Git: 71ca85dcac7f709bc27fc1c3e115f771f50a4cb4；evidence_level: MATCHED_DEVELOPMENT_CHECKPOINTS_NATIVE_RECEIPTS_AND_ACTION_TRACES_AUDITED。未开启独立formal/holdout，paper-ready与正式优势仍UNVERIFIED；不作新文献评分，不开展后续优化或合并算法主线。完成后移除本次轮询，保留当前聊天。
