> 2026-10-21 用户停止本轮：旧run于14:32:50+08:00 SIGINT退出，完整和未完整cell原样保留，`user_pause_receipt.json`为停止身份与文件清单证据。旧status是停止前快照，不再视为正在运行。后续独立修复及全新重训见 `fair_reward_contract_repair_20261021.md`。

# 扩大循环与预算的独立公平奖励训练

- reviewed_at: 2026-10-21；literature_cutoff: 2026-09-28（未重新评价 novelty）
- target_venue: CSCWD 2027 development；policy_version: tmc_review_policy_v3_20260621
- artifact_run_id: long_fair_reward_training_20261021_v1
- Git: 算法 clean source `8e15418`；诊断 release `c67d662`；runner/配置/observer hash 写入 frozen plan，发布commit写入 execution receipt
- evidence_level: PROTOCOL_FROZEN_AND_FOUR_ALGORITHM_PREFLIGHT_VERIFIED；正式结果尚待完整30 cell回收
- formal/holdout、算法优势与策略收敛：UNVERIFIED

## 本轮授权和诊断前置

用户本轮明确要求先查 SAGH 从初始就平线以及实际动作是否改变，然后统一增加所有算法训练量和循环轮次，
真实出图、启动后每20分钟检查，不改算法优化主线。这覆盖上一轮完成后“不追加预算”的限制；
旧实验保持原样，本轮新建独立目录，从零训练，不接旧权重或按排名延长某个方法。

前置独立诊断已完成并另存 `sagh_flat_action_diagnosis_20261021.md`：三seed各368个相同状态，
update0/12/24最终执行动作全部相同，原网络初始→最终argmax却改变258/262/269个；216条原生验证回报
逐条复现。现有 prior、机会约束、门控与 fallback 将网络变化映射回相同执行动作。
这是当前验证分布的近端原因，不证明训练没有执行或已经最优。本轮不删除/调弱这些规则来制造上升曲线。

## 先验固定的扩大预算

保持旧独立实验的全部10算法、seed7/13/29、原算法配置、24train+12validation真实非重叠窗口、
Alibaba两个workflow、同原生reward/offset0、无外部temporal shaping、CPU单线程/nice19/单cell顺序。
依旧调用 clean source `8e15418`，主工作区七个用户文件不导入、不覆盖、不提交。

| 每算法每seed | 旧独立实验 | 本轮 |
| --- | ---: | ---: |
| 完整窗口×workflow循环 | 1 | 32 |
| 真实环境交互 | 6,144 | 196,608 |
| 每次learner输入transition | 256 | 256 |
| learner调用机会 | 24 | 768 |
| 每个window×workflow每循环交互 | 128 | 128 |
| 固定验证次数（含零交互） | 13 | 33 |

每个cell执行32×48个unit，按原顺序重复访问，unit内episode结束则重置，末段真实截断明确标记
budget_cut，不延长补造回报。每个本地unit的环境随机种子固定为seed+local_unit，跨循环重复相同环境条件；
训练策略采样随机序列持续前进，不每循环重置网络/optimizer/replay。所有方法完全相同。
每6,144交互（24updates，一次完整循环）固定验证一次，包括0和196,608，总计33点；每次12×2真实episode。
相应保存initial与每循环checkpoint共33个；全部768次learn_info逐条保留，不因早停改变外部统一预算。
各算法native train_epochs、batch、KL早停和内部auxiliary loss不变，所以匹配交互与更新机会，不宣称同算力。

初始网络结构/先验和原超参数不同，初始奖励可以不同，不能强行拉齐起点。不会训练到SAGH排名满意才停止。
只报告最终点及所有固定点，不筛best checkpoint、不追加holdout、不改旧论文和优化任务。

## 动作证据和原始日志

新增 `scripts/run_long_fair_reward_training.py` 复用旧固定环境/配置生产端；旧runner原文件hash不变。
训练逐episode写 `episode_receipts.jsonl`、training.csv（含所有真实动作计数、cycle、unit_index、截断标记），
全部updates写 `updates.jsonl`。每次验证保存逐step `validation_actions_<step>.jsonl` 和原始CSV。
SAGH验证附加透明observer，不改变输出，并对全部初始验证状态在每个检查点评分，
记录 `paired_initial_states.csv`，把“网络变了”“最终动作变了”“回报变了”分开报告。
验证重建独立agent、复制当前网络和update_count、关闭dropout；保存恢复Python/NumPy/Torch RNG，不污染训练。

资源控制沿用原guard：发现其他Python训练/benchmark仅本实验等待，不杀/暂停/降优先级其他任务。
主线七文件保护diff SHA-256仍为`1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2`。

## 图形规则：参考样式，真实数据决定形状

- 训练图以实际episode为横轴、原生episode total reward为纵轴，事先固定100-episode trailing mean。
  前99点只用已有前缀，逐点导出使用的原始回报和均值；不取累计最大值、不平移/归一化/拟合/插值/补尾。
- 原始episode全部保留并在单算法图显示；所有seed完整观察曲线可见。mean/SD只在各seed都有实际数据的
  共同episode前缀上计算，不补齐较短seed，也不把不足3seed的尾部当作三seed平均。
- 总览显示10算法；固定两个workflow分别作图；每算法另给原始/移动均值及固定验证的双面板。
  相同交互预算下自然episode数可不同，episode横轴不代表交互数完全相等，公平结果另用interaction横轴核对。
- 33个固定验证点完整显示，不作平滑；图例标明mean与seed SD。不承诺出现参考截图的S形上升或SAGH领先。
- 平台规则沿用事前末5验证点relative range≤5%、relative slope/checkpoint≤1%；完成率≥0.8另报。
  即使曲线变平，也要核对初末收益和配对动作；初始就平、低性能平、未收敛和负结果都必须报告。

## 验证、执行和回收

28项专项测试通过。四算法SAGH/PPO/DQN/QMIX各512交互、2updates、两循环，验证0/256/512的预检完成；
`long_fair_reward_preflight_20261021_v1`中4cell、12checkpoints、全部原生episode receipt/循环暴露/
逐step验证回报/七文件保护的离线审计PASS，初始相同状态自比较29/29均零变化。绘图链通过，
训练总览及SAGH双面板已目视核对，无裁切/遮挡。预检不作为科学收敛结果。

```bash
.venv/bin/python -m pytest tests/test_long_fair_reward_training.py tests/test_fair_reward_training.py tests/test_training_reward_plots.py -q
.venv/bin/python scripts/run_long_fair_reward_training.py prepare --source-root /Users/howen/.codex/worktrees/fair-reward-continuation/PPO_MEC --output-root artifacts/analysis/long_fair_reward_training_20261021_v1
.venv/bin/python scripts/run_long_fair_reward_training.py run --source-root /Users/howen/.codex/worktrees/fair-reward-continuation/PPO_MEC --output-root artifacts/analysis/long_fair_reward_training_20261021_v1
.venv/bin/python scripts/audit_long_fair_reward_training.py --run-root artifacts/analysis/long_fair_reward_training_20261021_v1
PYTHONPATH=/private/tmp/ppo_mec_reward_plot_deps MPLCONFIGDIR=/private/tmp/ppo_mec_matplotlib .venv/bin/python scripts/plot_long_fair_reward_training.py --run-root artifacts/analysis/long_fair_reward_training_20261021_v1
```

完整实验计划、执行命令/进程、源码和绘图/审计脚本hash以本run的frozen_plan.json与execution_receipt.json为准。
prepare、run cell、audit、figures均create-only；失败保留现场，不自动重跑、覆盖cell或修改算法。
启动后复用当前聊天 `ppo-mec-3` 每20分钟检查，保存next_check_not_before；未到时间立即结束，不空等，
正常或让出资源保持安静，仅完成/明确失败/需处理时通知。完成后审计全部30cell、checkpoint、
原生回报、实际动作、共同暴露、数据和保护hash，再导出全部曲线/CSV/report与本地zip；不含checkpoint或原始数据。
完成或失败后暂停该提醒，不归档聊天。完整训练结果仍在运行时不得称已出收敛结论。

风险：扩大交互和循环不保证规则后动作发生变化，也不保证所有方法稳定；没有搜索或匹配梯度计算成本，
开发窗口不是untouched holdout，不进行paper-ready/novelty/正式优势判断。
