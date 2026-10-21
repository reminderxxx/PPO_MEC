# 独立公平训练与奖励收敛实验

- `reviewed_at`: 2026-10-21（Asia/Shanghai）
- `literature_cutoff`: 2026-09-28，沿用既有文献记录，本轮不评价 novelty
- `target_venue`: CSCWD 2027；开发训练诊断，不替换论文、canonical 或正式主表
- `artifact_run_id`: `fair_reward_training_20261021_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: 隔离算法源码固定为 `8e15418`；新 runner 自身 SHA-256 和配置 hash 写入 frozen plan
- `evidence_level`: 冻结方案与小规模执行预检通过，完整结果以各 cell 的 `complete.json` 和导出报告为准
- `verdict`: 正式算法优势、holdout 泛化及策略参数收敛均 `UNVERIFIED`

## 授权与隔离

用户在本聊天明确要求统一所有算法的训练窗口、奖励规则和训练预算，再公平训练、真实出图，并要求不影响算法优化的大规模训练。
该要求覆盖此前“保留原训练划分”的约束。采用新独立实验，**从零初始化权重、优化器、replay 和 update schedule**，
旧 checkpoint 仅提取已经使用的结构/训练超参数，不恢复任何权重或 optimizer；不筛选新的超参数。

算法代码读取本聊天已有 worktree `/Users/howen/.codex/worktrees/fair-reward-continuation/PPO_MEC`，固定 `8e15418`，
执行前检查 `src/` 和旧 `scripts/` 相对该 commit 无变化。主工作区七个用户修改不导入、不提交、不覆盖。
所有新文件写入 `artifacts/analysis/fair_reward_training_20261021_v2/`，无修改旧训练目录、优化任务、checkpoint 或原始数据。
本轮没有开启 G14 fixed-commit continuation、formal 或 sealed holdout。

资源控制：单 cell 顺序执行、CPU only、PyTorch/BLAS 单线程、nice=19；不占 MPS/GPU。
每个 episode、更新和验证窗口前检查其他 Python 训练/benchmark；发现后只让本实验等待，不改变对方进程。
让出是协作式的，正在执行的一次更新完成后才进入检查；不能保证资源使用在任意瞬间严格为零。
启动前只读进程检查未发现其他训练。预检和完整训练也遵守相同控制。

## 固定协议

配置：`configs/experiment/fair_reward_training_v1.json`。
完整计划：`artifacts/analysis/fair_reward_training_20261021_v2/frozen_plan.json`，semantic SHA-256
`81de66f5f104ebcf582828a0dda941cf99079521eec026c434601bb21d13053b`。

- 10 个学习算法：SAGH、PPO、MAPPO、DQN、Dueling DQN、QMIX、Controller MAT、DAG/Cache/DT Offload DRL；3 seeds `[7,13,29]`。
- 所有算法共用 NGSIM 前 30,000 rows 的实际窗口、Alibaba ordered 的相同两个 workflow；24 train + 12 validation，长度 24、stride 48，按事前 offset 顺序选择，不按奖励筛窗。
- freeze 绑定实际 `window_id`、source segment、原始 time interval、frame 内容 hash、workflow 内容 hash 和两个原始文件 SHA-256；36 个窗口所有两两区间均不得重叠，执行前再核验身份。旧 ordinary loader 不改，本 runner 显式校验并拒绝静默移动/缩短窗口。
- 同一原生环境、`semantic_discrete_5`、prediction horizon=16、handoff_pressure、相同 RSU 布局、max episode steps=22，`reward_positive_offset=0`。
- 外部 temporal reward shaping 在该实验不执行，逐 episode 检查进入 learner 的 reward 之和等于原生 total reward。保留各算法内部已有网络、credit/auxiliary loss、原超参数；SAGH 原 retrospective auxiliary label 路径保留。
- 每算法每 seed **6,144 环境交互、24 次 update 调用**，每次 update 恰好 256 个真实 transition。24 windows × 2 workflows 各分配 128 steps，按相同顺序执行；完成后重置当前 unit，budget 尾段明确记录截断，不补造 reward。
- 从零交互开始，每 512 steps 固定验证一次，共13个检查点。每次相同12 windows × 2 workflows；独立重建相同配置的评估 agent，复制当前网络、保留 update schedule，隔离 runtime/cache/RNG 并关闭 dropout。无 evaluation-only 新 guard、无 best-so-far 替代原曲线。
- 所有检查点完整保留，最终点和初始点都报告；不根据结果自动延长单个算法、不改 seed、不筛 checkpoint、不为维持排名调参。

统一预算指真实交互、update 机会和 validation/selection 机会。native train_epochs、早停和模型规模仍属于原算法配置，
因此实际 gradient work/计算时间可能不同；逐 update 保存 learn_info/learning_seconds，不声称统一计算成本。

## 收敛与绘图

事前固定末5次 validation 的相对 reward range≤5%、相对 slope/checkpoint≤1% 为“描述性奖励平台”；
完成率≥0.8 单独报告，不把低性能平台称为优化成功。每个 seed 单独判断；平台不等于策略参数/分布收敛。
无论是否出现平台或 SAGH 是否领先，原样输出所有算法和所有 seed 的结果。

绘图入口 `scripts/plot_fair_reward_training.py`：只在30个 cell 都完整且预算、窗口暴露、CSV hash一致后导出。
验证图使用13个真实点，不平滑、不平移、不归一化、不插值、不补齐尾段；均值与 seed SD 透明标注。
训练图为固定256-step block的原生 reward sum / 实际 step count，明确标为 step reward；原始 episode total reward 单独完整导出。
两种纵轴不能混用。前一轮历史日志图保留，不拼接成新训练曲线。

## 验证与执行

四算法 SAGH/PPO/DQN/QMIX 真实预检各256 interactions/2 updates及初始/中间/末次固定验证完成，
预检只用两个 train + 两个 validation windows、一个 workflow；不用于论文或算法排名。
证据位于 `artifacts/analysis/fair_reward_training_preflight_20261021_v1/`；预算核验和绘图生产/消费链通过。

```bash
.venv/bin/python -m pytest tests/test_fair_reward_training.py tests/test_training_reward_plots.py -q
.venv/bin/python scripts/run_fair_reward_training.py run \
  --source-root /Users/howen/.codex/worktrees/fair-reward-continuation/PPO_MEC \
  --output-root artifacts/analysis/fair_reward_training_20261021_v2
PYTHONPATH=/private/tmp/ppo_mec_reward_plot_deps MPLCONFIGDIR=/private/tmp/ppo_mec_matplotlib \
  .venv/bin/python scripts/plot_fair_reward_training.py \
  --run-root artifacts/analysis/fair_reward_training_20261021_v2
```

测试19项通过；覆盖精确预算、无外部 shaping/旧权重恢复、资源隔离、跨真实区间重叠拒绝、其他训练检测、
plan/hash 变化及平台/业务质量/参数收敛的区分。`prepare` 和 `run_cell` 为 create-only；已存在 cell 不自动重试/覆盖。
旧 v1 是零交互准备版本，因评估隔离改进重新冻结为 v2，不作为科学结果。

未覆盖：独立 untouched holdout、正式窗口新颖性、最优超参数搜索、同算力预算、所有算法收敛保证及 TMC-ready。
本轮 validation 是开发数据，不称独立检验，不依据训练曲线自动改论文或算法优化方案。

## 用户指定的后续轮询（2026-10-21）

用户要求“开启实验后隔20min轮询，后续再优化”。本聊天已创建 heartbeat `ppo-mec-3`，每20分钟检查既有 run；
`continuation_monitor.json` 记录下次最早检查时间和通知去重。正常运行或让出资源时保持安静；只在完成、失败或需行动时反馈。
完成后只核验原件、导出真实曲线和报告并暂停提醒，不启动新训练或优化。失败保留现场、独立诊断，不自动重跑。
