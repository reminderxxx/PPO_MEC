# Fast auxiliary 与执行动作 PPO 的冻结策略梯度探针

## 身份和边界

- `reviewed_at`: 2026-10-10 Asia/Shanghai；`literature_cutoff`: 2026-09-28（未进行新文献评价）；`target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `cscwd_fast_aux_gradient_probe_20261010_v1`；事前计划 commit `900c79e`；诊断源码来自 B commit `8176c8c7d96d3d1d360b49a928a9e29cd5ab2aaf`，其中 agent/environment 与候选实现 commit `46a68f11` 无差异，后续仅训练行为日志生产者修正。A 本报告/脚本最终 commit 以 Git 历史为准。
- `evidence_level`: 冻结开发 checkpoint 上的 **fixed-policy new probe / local diagnostic**；20 槽、1,200 新 train 环境步、0 optimizer step、0 参数改变、0 formal/holdout/support。历史训练 minibatch 的 old logprob/advantage/逐 head 梯度未保留，故**历史训练梯度冲突 UNVERIFIED**。论文效果、正式优秀基线优势和 TMC-ready 均 `Unverifiable`；原科学 verdict `MIXED_STOPPED` 不变。
- 旧/新原 run manifest SHA 分别 `48718e48dc55e6958c03b676634dca53e84fd21be251115d4809380a1441c615` / `d9f256b6277c6b358b7eafd329f118f1f56884b67d94640bfcb8775d29fa880c`；integrity SHA 分别 `b66dfb946d2fef8c8e93b3ffacd7f332c1280635c7c635ef1f58f7eb8c4e2ac1` / `fa03b3fe9eed9af66f96e10e10d90db380b13a28893716071a395a8e4eb129cc`。新原件 `artifacts/analysis/cscwd_fast_aux_gradient_probe_20261010_v1/`，`probe_manifest.json` SHA `74665bb92708c94a17978fb6f4ebb32af95002cb3b75601a282d871c001cbb20`，`aggregate_summary.json` SHA `b71c36da5d12e41fb933dd85157695866e6ba3435339e00ca40178a536e7f98e`，敏感度 sidecar SHA `4bf26b08ba45abca68780a092856dfdc3a0fb9c9bf670bce6022b65f62d5c29f`；20 个逐槽 JSON 在 manifest 中分别绑定 SHA。

## 冻结采样和梯度消费链

旧 v4 SA 对照与条件弃权候选各取 5 seed × `selected/update96`，每 checkpoint 从原 12 个 train 实例、原 seed/RNG/实例顺序出发，用训练器 `_collect_exact_update_batch` 采一批 60 个随机策略 transition。原 `gamma=.99`、GAE `.95`、episode cap 24、raw policy、原 mask、原 checkpoint；各槽互不混批。`selected/update96` checkpoint **文件 SHA 为 20 个**，但网络参数 SHA **仅 13 个**：旧 seed 7/17/43/61、候选 seed 7/17/43 两视角参数相同；重复槽不能当独立策略。采样路径仍可能在不同策略间分叉，所以“同状态”仅指**每一槽内 PPO 与 CE 用同一 batch 和同一参数**，不表示候选/对照跨臂状态逐条相同。

生产端是 `sa_ghmappo_core.py::_build_mechanism_targets`：本配置关掉 fallback/steady target 特例，当前 RSU 均存在，fast target 恒 0；head1 对应执行 action2 vehicle fallback。`_compute_auxiliary_loss` 对 confidence>1e-6 的状态按**共同 eligible 分母**平均，慢/快/event CE 原权重 `1/.5/1`、总 `auxiliary_coef=.1`；候选只在当前 bundle 缺失时将 event CE 与 temporal 一致性乘零，不重定分母。探针仅分离 CE，**temporal 一致性未纳入 event CE 梯度**。`_compute_env_action_ppo_loss` 对原聚合五动作分布与实际执行动作求 logprob，采用 rollout `env_action_log_prob`、原 batch 标准化 advantage、机制 transition weight、clip/混合系数；`executed_action_ppo_only=true`，不使用 canonical inverse head tuple。value 是原 raw critic return MSE 乘 `.5`，未调用 `learn` 或 PopArt 更新。当前配置其余 advantage 扩展与 model critic 关闭；如被打开，脚本会停止而非近似。

对同一 forward 图分别求执行动作 PPO actor、slow/fast/event CE、value 梯度；记录 raw CE 和原系数加权 CE 的 loss/范数、加权项相对 actor 的 dot/cosine/范数，分组为 shared encoder、各 actor head 与 critic。原系数为正，raw 与 weighted CE 的 cosine 相同；actor 系数 1，raw/weighted 相同；value raw 范数为记录的 weighted 范数除 `.5`。零范数的 cosine 明确写 `UNDEFINED_ZERO_NORM`。value 对全部 actor-head 参数梯度为 0，CE 对 critic 梯度为 0；fast CE 经分层条件进入 slow actor 和共享 encoder，event CE 还经 fast actor 回传，因此不能把 head 看作独立。

实际动作新旧 logprob 最大差 `5.07e-7`，20 槽参数前后 SHA 全同；20 个文件 SHA/逐槽 JSON SHA、有限梯度和 1,200 步上限均通过。每个样本对 fast_logit[1] 的 post-forward `P(action2)` +1e-3 差分均为正（1,200/1,200）。独立零环境步 sidecar 在固定公开 train reset 状态对 20 槽比较 autograd 与中央差分，20/20 action2 合法、导数均为正，绝对误差最大 `4.25e-6`。这是**固定其他输出 logits 的局部偏导**，不等于完整网络参数训练一步的行为变化。

## 分母、方向与可复核读数

| 臂 / 视角 | transition | aux eligible | event supervised | fast target=0 | action2 | fast CE 对 PPO：encoder 负点积 seed | fast_actor 负点积 seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 v4 selected | 300 | 286 | 286 | 300 | 44 | 5/5 | 5/5 |
| 旧 v4 update96 | 300 | 287 | 287 | 300 | 44 | 5/5 | 4/5 |
| 候选 selected | 300 | 288 | 215 | 300 | 55 | 3/5 | 3/5 |
| 候选 update96 | 300 | 287 | 206 | 300 | 57 | 3/5 | 2/5 |

总计 fast target 0 为 `1,200/1,200`、confidence eligible `1,148/1,200`；候选 event 监督 `421/575` eligible。candidate selected 五 seed 的 shared-encoder fast/PPO cosine 是 `+.719, −.889, −.137, +.177, −.198`（seed 顺序 7/17/29/43/61），fixed96 为 `+.719, −.889, −.087, +.177, −.919`。control selected 为 `−.634, −.031, −.531, −.416, −.649`；旧对照方向较一致，候选方向**跨 seed 混合**。以候选 selected 为例，weighted fast CE encoder 范数均值 `.214`，PPO actor encoder `1.239`，weighted value encoder `18.242`；不能把 fast CE 判为总更新主导项，亦未复原历史 clip 后更新。逐槽 raw/weighted 范数、其他 head dot/cosine 和零梯度都保存在 JSON，不能从均值替代逐 seed。

按**公开当前状态**重建 readiness：ready `856/1200`、missing `344/1200`、unknown `0`。estimated missing **resident** bytes 从当前 RSU typed resident + 公开 object catalog 得到：zero `856`、`(0,100MB]` `75`、`>100MB` `269`、unknown `0`；不是实际未来 transfer bytes。公开上一 transition clock/deadline 可用时，剩余 deadline `<=10s` `122`、`(10,40]s` `419`、`>40s` `465`，首步等 unknown `194`。公开预测 contact budget `<=5s` `576`、`(5,15]s` `624`，其它 `0`。这些均是**槽位 transition 曝光次数**，重复参数与重复状态不算独立样本。

ready/missing 分层在原**整批** advantage 标准化和 aux eligible 分母下另算局部 fast/PPO 梯度：候选 selected 两层 encoder 负点积均 `3/5` seed，fixed96 也均 `3/5`；fast_actor 的 ready/missing selected 为 `2/5`、`3/5`，fixed96 为 `2/5`、`2/5`。对照 missing encoder selected/fixed96 各 `5/5`，ready 各 `3/5`。这个分层仍不支持候选跨 seed 一致冲突。其余三种预定 bin 保留了完整**计数**，此次未保存逐 transition forward 图，无法在不超过冻结的每 checkpoint 一批/1,200 步预算下事后重算各 bin 梯度；相应**bin 内冲突方向 UNVERIFIED**，不能据计数判断局部方向。

## 机制与决定

fast CE 目标 0 对 action2 的可执行概率有实际局部压力，并非无效 head：全样本 fast target=0、fast_actor 的 weighted CE 梯度非零、action2 概率敏感度为正。然而候选的 PPO/fast CE 冲突方向并不跨 seed 稳定；又缺历史训练 minibatch、完整优化器更新和独立确认 split，**“fast 监督造成候选多加载/逾期退化”尚未证实**。旧成本审计中 selected seed7 `regression_10` 的 action2 相比 action3 额外 8.023s 且逾期，`regression_04` action2 却避免重算并按期；本轮梯度不能把这两个状态差异解释成单一全局 fast target 缺陷。

目前不修改 fast/slow/event 权重，不选新 checkpoint，不启动消融；保留 `MIXED_STOPPED`。若 B 另立并预冻结最小可证伪训练实验，唯一可隔离的变量是**仅 fast CE 权重 .5→0**，保留 event 弃权候选、slow CE、PPO/critic、原数据/预算/选模与成本指标；必须与原候选配对报告所有 seed、action2、on-time、失败、model bytes 和 deadline，并独立检验。现有 local gradient 不能预言该实验会改善，也不授权自动执行。
