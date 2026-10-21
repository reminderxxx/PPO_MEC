# 独立公平训练合同修复（2026-10-21）

## 授权与隔离

用户要求暂停旧轮并保留现场，在独立实验分支修复动作概率映射、更新稳定性和 window_class，验证通过后统一重训。旧轮 `long_fair_reward_training_20261021_v1` 已于 2026-10-21 14:32:50 +08:00 按核对过的 source/output 命令发送 SIGINT 并确认退出。`user_pause_receipt.json` 保存停止身份、360 个既有文件的 hash；3 个完整单元和 DQN seed7 的未完成单元均保留，不续跑或混入新结果。旧 `status.json` 是停止前快照，停止回执优先。

实现仅在 `codex/fair-reward-contract-repair`，worktree `/Users/howen/.codex/worktrees/fair-reward-contract-repair/PPO_MEC`，起点 `2d7cfce`，其算法与原 `8e15418` 一致。不合并到算法优化主线。主工作区七个用户修改的保护 diff SHA-256 仍须为 `1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2`。

## 修复范围及概率含义

旧代码先归一化采样 base 动作，再经 option/guard/fallback 映射，PPO 却把映射后动作在映射前分布中的概率当成行为概率。现在保存原始采样 base 动作，以及该上下文实际采样的 option，使用 `log p(base|state) + log p(option|state,base-context)` 更新。该联合采样路径定义了有效的扩展动作空间；多个路径可映射到同一最终动作。它不是最终执行动作的边际概率。原有 `env_action_probs` 仅用于映射前提议诊断，新增显式 scope 和 sampled 字段，诊断消费者同步，不能再称 final probability。

学习开始前必须逐条复算行为 log likelihood，最大绝对差不得超过 `2e-5`。平层和分层 PPO 共用该校验；DQN、Dueling DQN、QMIX 不假装提供策略密度。冻结配置未启用的 critic option improvement 不在本合同支持范围内。无有效 action mask 的历史兼容路径不得进入新实验。

数值稳定性修复覆盖完整总损失：检查有限 loss/gradient，保留 native 梯度裁剪；候选 Adam 步按原 target_kl 检查观测上下文 base categorical KL 加条件 option KL，均值不超过 target、单条不超过 8 倍 target，最多尝试 8 个固定半步缩放；不接受则回退网络和 Adam 状态并结束本次更新。该约束只覆盖当前观测上下文，不能证明全状态或最终动作策略收敛。概率比使用 log 差 [-20,20] 的数值边界，原额外 ratio barrier 改在 log-ratio 上计算，保留原系数/方向，避免指数平方爆炸。梯度 epochs、学习率、原辅助目标和奖励系数不调优。记录接受/拒绝 Adam 步、backtrack、KL、梯度范数；相同 learner 调用机会不等于相同有效梯度步。

window_class 从每个既定窗口的原始 mobility 活动统计计算，提取并复用既有 native 分类阈值函数，冻结类别和输入统计，再贯通 bundle、train/validation metadata、CSV 与原始 receipt。窗口身份、排序、帧内容及 reward 均不因类别修改；不按奖励重新选窗。

## 训练与验证方案

正式新 run 计划 `artifacts/analysis/repaired_fair_reward_training_20261021_v1`，配置 `configs/experiment/repaired_fair_reward_training_v1.json`。10 算法 × seeds 7/13/29，全部 fresh network/optimizer/replay；旧 checkpoint 仅提取配置。保留同 24 train + 12 validation 非重叠真实窗口、两个 Alibaba workflow、原生 reward offset0，无外部 reward shaping。每 cell 196608 交互、768 次 learner 机会、32 完整循环，每 unit128 steps；0 起共33固定验证点。不加载旧权重，不根据 SAGH 排名或曲线续跑。

先执行相关回归测试和全部10算法每512step/2循环的真实预检，再离线审计初始化、完整计数、分类、采样概率、native reward 和 checkpoint。预检只验证接口与数值，不用于算法效果结论。完整训练只能在预检通过并冻结 clean source 后启动。CPU单线程、nice19、单cell顺序，发现其他训练时仅本实验让出资源。

训练图保留所有真实 episode total reward，另显示预先固定100episode trailing mean；验证33点不平滑，两类图分开。所有seed保留，mean/SD只取共同已观察前缀，不平移/补尾/归一化/拟合，不保证上升、收敛或SAGH领先。完成后审计、目视核图，打包 curves/CSV/report，不含 checkpoint/原始数据。奖励平台仍按末5点 range<=5%、slope/checkpoint<=1%描述，完成率>=0.8另报，不等于学习收益。

## 状态与证据边界

本文件建立时：旧训练停止；代码回归311项通过，真实预检/新训练待完成；后续启动信息以本文件追加记录和新run execution_receipt为准。

reviewed_at: 2026-10-21；literature_cutoff: 2026-09-28（沿用，未新评novelty）；target_venue: CSCWD2027；artifact_run_id: repaired_fair_reward_training_20261021_v1；policy_version: tmc_review_policy_v3_20260621；source Git: 独立分支冻结后由 frozen_plan.source_commit 记录；evidence: 修复回归测试，完整训练 UNVERIFIED。此轮 matched development，不是 untouched holdout、formal 晋级或 paper-ready。旧审计指出的 native exploration bonus 和辅助成本目标差异保留；本轮不以改奖励或关辅助目标制造上升曲线。

## 冻结前验收追加

312项相关回归全部通过，包括多路径合并概率、七类策略初始ratio=1、非有限loss拒绝、接受步KL、空及已有Adam动量的完整回退。命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest tests/test_repaired_probability_contract.py tests/test_window_selection_contract.py tests/test_env_contract.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py tests/test_long_fair_reward_training.py tests/test_fair_reward_training.py tests/test_training_reward_plots.py -q`。

真实预检 `artifacts/analysis/repaired_fair_reward_preflight_20261021_v1`（源码 `cc0232ef9459efab494a6fa7b0242d0a735b23cd`，plan `a07556890455021b284d98ce57795c2c98b0c0d464ae30daad2d5c9a0450bd45`）完成全部10算法×seed7，各512step/2循环/2learner调用/3检查点；原生receipt与30 checkpoints离线审计PASS，严格概率消费端及create-only绘图PASS，SAGH双面板目视无截字。七类策略所有更新前logp复算误差均0；每算法都有接受梯度步，SAGH70步接受、2候选步拒绝并回退，最大接受平均KL约0.00139772。预检存在原生奖励负值和固定验证平线，不因结果更改奖励、初始值、类别或预算。小预算不能用于长训练收敛或优势结论。

预检后只补充非空Adam回退测试及只读消费端的归一化/采样概率/计数/KL核验，没有再次改动训练、映射或分类源码。正式新run将锁定最终clean commit；启动后该source worktree的HEAD、源码和plan全部保持不动，完成文档通过主工作区的scoped docs提交更新，不改变冻结源码。
