> 2026-10-21 用户暂停旧轮：14:32:50+08:00核对命令后SIGINT退出，user_pause_receipt.json保存360文件hash并复核未变；3完整cell和DQN seed7未完成单元保留，不续跑。本文件旧RUNNING叙述为历史。修复后独立新轮见fair_reward_contract_repair_20261021.md。

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

## 启动验收（2026-10-21 11:32:50 Asia/Shanghai）

完整30cell独立训练已启动；runner release=`2d7cfce`已push，plan semantic SHA-256为
`9a30d9ac0ebf841d7747bc2d5762ed6ec1ff0a3dfd36d2a9e53228070bd3fcc8`。
`parent_protocol_comparison.json`确认全部算法/seed/窗口/配置/原始数据/source/reward及unit/update粒度
与父实验一致，仅增加预算、循环及预先声明的验证间距/日志。启动进程pid=19388，经命令与output-root核对，
nice19、约一个CPU核、约408MiB RSS；启动验收已实际完成SAGH seed7的5updates/1,280交互，无Traceback。
这是启动时快照，不作为训练完成或效果证据。完整命令、进程观察和绘图/审计脚本hash见execution_receipt.json。
七文件保护diff未变。当前聊天原`ppo-mec-3`已恢复ACTIVE、20分钟间隔，最早下一次检查为11:52:50；
正常运行/资源等待不重复通知。

## 用户查询的阶段核验（2026-10-21 14:10 Asia/Shanghai）

本次用户直接询问进度、SAGH动作/奖励和预计完成时间，授权在常规轮询间隔外读取这些现有记录；
不启动新训练或改变任何计划。当前2/30cell完成（SAGH/PPO seed7），正在MAPPO seed7；
查询快照约516,864/5,898,240交互，8.76%。SAGH seed13/29尚未执行，不能推断三seed共同结果。

SAGH seed7完整完成196,608交互、768updates、32循环、15,907训练episodes，用时95.1分钟。
其全部训练receipt/循环暴露、33×24=792验证units及逐step原生回报对账通过，相关CSV hash、
plan/runner/base/observer及后处理hash一致；本轮未额外训练或重放。阶段证据在
`artifacts/analysis/long_fair_reward_training_20261021_v1/interim_status_20261021_1410/`。

- 固定验证0至98,304交互保持18.871250；首次在104,448交互观察到5/368个相同初始状态的最终动作变化，
  对应回报18.662917。之后实际动作变化增多，最终为77/368（20.92%）。
- 最终验证回报15.465000，较初始下降18.05%，完成率始终0.875；这是“动作开始改变、回报下降后趋稳”，
  不能写成训练上升收敛或学习收益。训练采样回报自身有真实波动，不是常数；与固定验证曲线分开。
- 原网络最终argmax在233/368个相同状态变化；最终执行变化更少，现有规则仍影响映射，但不再全部锁定。
  自有轨迹按相同步号比较有81个不同动作；其状态已受前序动作影响，不与严格相同状态的77个混用。
- 按已完成SAGH/PPO、当前MAPPO速度和父实验各算法用时校准，剩余训练粗估20–26小时，中心约22.4小时；
  未完整运行的长预算baseline耗时仍不确定，遇主线训练会让出资源而延后。不是完成承诺。

阶段证据等级为`PARTIAL_RUN_SAGH_SEED7_NATIVE_RECEIPTS_AND_PAIRED_ACTIONS_VERIFIED`；
reviewed_at=2026-10-21，literature_cutoff=2026-09-28未新评，target_venue=CSCWD2027 development，
policy_version=tmc_review_policy_v3_20260621，source Git=8e15418，runner release=2d7cfce。
完整30cell审计、另外两个SAGH seed、baseline比较及formal/holdout结论仍待完成或UNVERIFIED。

## 只读审查补充（2026-10-21 14:19）

用户随后要求只读查目标、映射和训练更新，保持当前run及优化主线不动并继续20分钟监控。
独立报告`sagh_reward_decline_audit_20261021.md`已定位：native探索加分与成本/teacher辅助目标差异；
option之后的最终动作仍使用前置env概率作PPO ratio；日志存在最大3.39e13的env-action loss且现有KL未拦住；
本独立runner遗漏window_class，导致原方法部分分类依赖分支以unknown运行。没有漏更新/原生reward符号错误证据。
这些属于现存算法/入口解释限制，不因未来完整性审计PASS消失。本run只能连同这些限制作为开发诊断，
不能声称完整复现原方法机制或据此给原算法的正式优劣结论。本轮未修复/未改变输入，继续冻结预算。
精确动作/奖励账本、原始日志/hash、闭式GAE检查及因果未定事项保存在
`artifacts/analysis/sagh_reward_decline_audit_20261021_v1/`，无新环境重放、act或learn。

风险：扩大交互和循环不保证规则后动作发生变化，也不保证所有方法稳定；没有搜索或匹配梯度计算成本，
开发窗口不是untouched holdout，不进行paper-ready/novelty/正式优势判断。
