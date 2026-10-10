# CSCWD per-head 梯度探针独立验收

## 审查身份与结论

- `reviewed_at`: 2026-10-10 Asia/Shanghai
- `literature_cutoff`: 2026-09-28（本轮未新增文献判断）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_fast_aux_gradient_probe_20261010_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: 验收源码基线 `8176c8c7d96d3d1d360b49a928a9e29cd5ab2aaf`；探针提交 `9376403bc3d059505054682016fdd8258dfef2e7`
- `evidence_level`: `E2_ARTIFACT_AUDITED / fixed-policy new development probe`；不是历史训练更新复现、formal、holdout 或性能证据
- `verdict`: `ACCEPTED_DIAGNOSTIC / FAST-LOSS CHANGE UNSUPPORTED`

独立核验接受 A 的冻结探针为**当前 checkpoint 上的局部梯度诊断**，但拒绝把它解释成历史训练根因或删除 fast CE 的授权。20 个 checkpoint 文件对应 13 个唯一网络参数状态；共 1,200 个新 train transition，未执行 optimizer step，所有参数前后哈希一致。历史 minibatch、当时 optimizer state 和当时 minibatch 顺序不存在，因此历史冲突保持 `UNVERIFIED`，上一轮科学结论保持 `MIXED_STOPPED`。

## 实际消费链

环境动作由 `event → slow → fast` 的优先级聚合：event=1 为 action4；否则 slow=2 为 action1、slow=1 为 action0；仅 slow=0 时 fast=1/0 分别为 action2/action3。`_hierarchical_env_action_scores` 的五项分别是这些互斥路径的 log probability；无额外 bias/mask 时指数和为 1。训练的 `_compute_env_action_ppo_loss` 对这个五动作分布中的**实际执行 action**求 log probability，不消费 `_head_targets_for_env_action` 给出的 canonical inverse tuple。

fast target 在本探针的 1,200/1,200 个 transition 中均为 0。fast=1 确实控制可执行 action2：1,200/1,200 个前向差分均显示提高 `fast_logit[1]` 会提高 `P(action2)`；独立零环境步校验在 20/20 checkpoint 槽上得到同号 autograd 和中央差分，最大绝对误差 `4.25e-6`。因此 fast 不是无效 head，fast CE 也确有局部压低 action2 的能力。

层级条件化未 detach。fast CE 会经 `slow_probs` 回传到 slow actor 和 shared encoder；event CE 会经 `slow_probs + fast_probs` 回传到 slow/fast actor 和 shared encoder。critic 与 actor head 参数隔离，但 value loss 可进入 shared encoder。因此验收按 encoder、slow/fast/event actor、critic 分组；critic 的零 actor norm 和 value 对 actor head 的零梯度不算冲突，任一零范数 cosine 均为 `UNDEFINED_ZERO_NORM`。

auxiliary 仍使用原 confidence-eligible 固定分母，slow/fast/event CE 权重为 `1/.5/1`，再乘总 `auxiliary_coef=.1`。event abstention 只把 current-missing 的 event CE 与 temporal consistency 置零，不改变 fast/slow 项，也不按 event 有效数重新归一。探针明确只把 event CE 分离出来，未把 temporal consistency 冒充 event CE。

## 独立验收读数

| 臂 / checkpoint 视角 | transition | eligible | event supervised | action2 | fast CE vs PPO encoder 负点积 seed | fast actor 负点积 seed |
|---|---:|---:|---:|---:|---:|---:|
| 旧 v4 / selected | 300 | 286 | 286 | 44 | 5/5 | 5/5 |
| 旧 v4 / update96 | 300 | 287 | 287 | 44 | 5/5 | 4/5 |
| event-abstention / selected | 300 | 288 | 215 | 55 | 3/5 | 3/5 |
| event-abstention / update96 | 300 | 287 | 206 | 57 | 3/5 | 2/5 |

候选臂的 shared-encoder 和 fast-actor 冲突均未跨 seed 稳定；按公开 readiness 分层后仍是混合方向。20 个文件 SHA 不同，但 loaded network parameter SHA 只有 13 个；重复参数槽不能当作独立策略。公开 missing bytes、deadline 和 predicted contact 仅保留曝光计数，未保存逐 transition forward graph，相应 bin 内梯度为 `UNVERIFIED`。

实际动作 old/new log probability 最大差 `5.07e-7`。20 个逐槽文件与 manifest SHA 全匹配；参数前后 hash 全相同；value 对 slow/fast/event actor 参数的梯度全为 0。独立合同测试还验证了五动作精确边缘概率、canonical action4 tuple 与真实 action4 marginal 的差异、实际动作概率的有限差分方向、原 auxiliary 分母/系数、同图梯度可重复和 shared/upstream 反传。

## 假设判定与算法决定

- **支持**：fast target 恒 0 会对 action2 施加真实、可执行的局部下压力；不是无效 head 或仅日志现象。
- **否定为优先根因**：候选臂 PPO 与 fast CE 的冲突方向跨 seed/视角不稳定，且现有 action2 有益/有害反例并存；不能据此解释服务退化。
- **仍未定位**：历史实际 minibatch 上的冲突、clip 后合成梯度、optimizer 动量影响，以及 bytes/deadline/contact 各 bin 的局部方向。
- **本轮决定**：不改 fast/slow/event 权重，不实现 loss 变体，不启动训练、不重选 checkpoint。event abstention 的 `MIXED_STOPPED` 不变。

由于证据未达到“一致实际执行概率冲突”，本轮**不冻结新的科学 A/B**。若未来获得新的明确授权、且仍要做唯一最小区分实验，最多只能把 event-abstention 候选中的 `auxiliary_fast_weight: .5 → 0` 作为唯一变量；event abstention、slow/event CE、固定分母、PPO/critic、数据、预算、选模和全部成本指标必须不变。这只是区分设计，不是当前推荐。

该设计会被以下任一结果否定：fast/PPO 冲突仍不跨 seed 稳定；action2 变化不能同时改善按期完成/服务失败；model bytes、elapsed 或 recompute 的退化抵消服务收益；任一 seed 的主要服务指标明显恶化；或改善只来自 checkpoint 选择而非固定 endpoint。不得以 reward 或 SA 排名替代这些条件。

## A 论文线的主张与证据变更说明

可新增的安全陈述只有：冻结 development checkpoint 的新 on-policy probe 证明 fast CE 能影响实际 action2 概率，但候选臂 per-head 冲突跨 seed 不稳定，未支持 loss 改动。不得写成“辅助监督导致了历史服务退化”“删除 fast CE 会改善性能”或“已形成新算法贡献”。本轮未编辑论文稿，paper-ready 继续为 `Unverifiable`。

资源边界不变：训练与探针均使用 development synthetic/calibrated 环境，bytes/time 是模型化 accounting；没有真实共享无线带宽、队列、跨 workflow cache 生命周期或独立 holdout。两步规则仍具 model-based transition clone 与词典序目标权限，不能与 learned policy 当作同权限证据。

## 产物与验证

- A 报告：`docs/project/cscwd_fast_aux_gradient_probe_20261010.md`
- manifest SHA-256：`74665bb92708c94a17978fb6f4ebb32af95002cb3b75601a282d871c001cbb20`
- aggregate summary SHA-256：`b71c36da5d12e41fb933dd85157695866e6ba3435339e00ca40178a536e7f98e`
- autograd/finite-difference sidecar SHA-256：`4bf26b08ba45abca68780a092856dfdc3a0fb9c9bf670bce6022b65f62d5c29f`
- A 产物独立核验：20/20 slot hash、1,200 steps、13 unique parameter hashes、finite/zero-norm contract、20/20 sensitivity records 全通过。
- A 邻接测试：20 passed；A smoke 与脚本 compile 通过。
- B 合同/邻接测试：19 passed；B smoke 通过。

**直接回答**：当前最值得补足的不是删除某个 auxiliary head，而是**让辅助目标在实际执行动作概率上的作用具备跨 seed、按服务状态一致的信用证据**。证据是 fast target 1,200/1,200 为 0 且确实压低 action2，但候选冲突仅为 encoder `3/5、3/5`、fast actor `3/5、2/5`，无法解释正反 action2 结果。下一轮当前不应自动改变任何算法项；若另获授权做唯一最小区分，只改变 fast CE 权重 `.5→0`，而任何跨 seed 不一致、服务指标无共同改善或成本恶化都会否定它。
