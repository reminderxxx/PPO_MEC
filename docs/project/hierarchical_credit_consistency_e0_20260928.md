# 层级五动作 PPO credit consistency E0 数学审计

- `reviewed_at`: `2026-09-28T19:08:17+08:00`
- `literature_cutoff`: `2026-09-28`（本轮未新增网页检索）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `hierarchical_credit_consistency_e0_20260928`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `c4cdf85a368170160197ac41bee35f973068063d`（审计执行时）
- `source_artifact`: `crdcm_performance_matrix_v2_20260928`
- `audit_version`: `hierarchical_credit_consistency_e0_v1`
- `scope`: `mathematical_validation_only_no_environment_no_training`
- `evidence_level`: `E2_IMPLEMENTATION_MATH_AUDITED_PLUS_CHECKPOINT_READBACK`
- `verdict`: `LEGACY_PER_HEAD_SURROGATE_NOT_EQUIVALENT_TO_MASKED_ENV_ACTION_PPO`

## 结论

当前层级 agent 的采样分布是 mask 后五动作 categorical，但主 actor loss 是 canonical slow/fast/event 三头各自 ratio、各自 clip 后的聚合。E0 使用 full-SA `head_credit_disabled` 的等权 PPO core、令 base/event advantage 同号同值并将 event reliability modifier 置为 neutral；这是实际 actor core 的一个有效受控切片。通用 PPO 一致性必须在这个切片成立，一个反例即可否定合同。E0 穷举证明两者不是同一 PPO objective：

- 31 种非空五动作 mask；
- 每个 mask 的全部合法动作，共 `80` 个 mask-action pairs；
- 4 组预先固定、有限、非退化的 old logits 与固定 perturbations；
- 正/负 advantage 两种 clip 方向；
- 合计 `640` 个 case。

legacy SA-core surrogate 在 `640/640` case 的 objective 或梯度与真实 masked environment-action PPO 不一致。该结论是实现数学正确性结论，不是性能因果结论；MAPPO 特定 head weights 和额外辅助项没有在 E0 中被冒充为性能复跑，本轮也没有修改算法、训练模型或启动环境。

## 审计对象

层级头在 mask 前定义五动作分布：

```text
p0 = P(event=keep) × P(slow=current_fill)
p1 = P(event=keep) × P(slow=prefetch)
p2 = P(event=keep) × P(slow=no_change) × P(fast=vehicle)
p3 = P(event=keep) × P(slow=no_change) × P(fast=current_rsu)
p4 = P(event=prepare)
```

正确的 behavior/update contract 应为：

```text
π_mask(a|s) = exp(score_a) / Σ_{j: mask_j=1} exp(score_j)
r(a,s) = π_mask,new(a|s) / π_mask,old(a|s)
L_actor = -min(r A, clip(r, 1-ε, 1+ε) A)
```

当前 legacy update 则将一个环境动作反映射为 canonical head labels，对每个 head 独立构造 ratio 和 clip surrogate，再按 head weights 聚合。这不仅会给未观测 latent labels credit，也无法表达 mask 条件归一化的联合梯度。

## 结果

| 检查 | 结果 |
|---|---:|
| mask 数 | 31 |
| 固定 logits probes | 4 |
| mask-action pairs / probe | 80 |
| 正负 advantage 总 cases | 640 |
| probability normalization 最大误差 | `2.22e-16` |
| invalid padded action 最大概率 | `0` |
| log-ratio vs direct probability ratio 最大误差 | `4.44e-16` |
| 六位 old log-prob 在 identity ratio 的最大误差 | `4.999e-7` |
| 单合法动作 cases | 40 |
| 单合法动作 exact gradient 最大范数 | `0` |
| 单合法动作 legacy 非零梯度 | `40/40` |
| exact 零梯度 head 仍被 legacy credit 的 cases | `410/640` |
| legacy objective/gradient mismatch | `640/640` |
| 只屏蔽零梯度 head 后仍 mismatch | `610/640` |
| ratio 落在 clip 区间外 | `250/640` |
| override actor weight=0 最大梯度 | `0` |

### 单合法动作反证

当 mask 只有一个合法动作时，条件策略概率恒为 1，不依赖任何 head logits；正确 log-prob、ratio 和 actor gradient 必须分别为 `0 / 1 / 0`。40 个 fixed cases 全部满足这个 exact 结果，但 legacy canonical head surrogate 全部产生非零 gradient。这是无需 reward、状态或训练即可复现的最小反例。

### 为什么只改 head weights 不够

E0 还使用 exact behavior log-prob 的 autograd 判断哪些 head 梯度为零，并构造 oracle 式“只保留非零梯度 heads”的 per-head surrogate。它仍在 `610/640=95.31%` cases 与正确目标不一致，原因有两层：

1. mask 后分母把多个环境动作耦合起来，head 的 exact gradient 通常不是某个 canonical label log-prob 的梯度；
2. 每个 head 独立 ratio/clip/平均，不等于一个 joint environment-action ratio 的单次 clipping。

因此 `head_credit_weights` 调整只能改变启发式 surrogate，不能使其成为真实 behavior policy 的 PPO objective。

### old log-prob 精度

当前 action info 把 `env_action_log_prob` 四舍五入为六位后进入 rollout。E0 的 identity ratio 最大误差为 `4.999e-7`，不是本次主要缺陷，但修正 contract 不应继续让日志格式决定训练精度。训练 buffer 应保留未舍入 behavior log-prob；展示/JSON trace 可单独舍入。

### padding、override 与 checkpoint

- `-1e30` mask padding 在 float64 审计中的 invalid action probability 严格为 0；非法动作未进入合法 action case。
- `actor_credit_weight=0` 的纯 actor objective 在全部 cases 梯度严格为 0；未来实现仍须明确 critic-only 是否允许通过共享 trunk 间接改变 actor outputs。
- 48 个 `update_0004/0008/0012/0016.pt` 均成功读回，update、network/residual state 与 v1 checkpoint/decision/credit contract 一致；读回前后 SHA-256 相同。
- v1 credit contract 为 `mask_external_override_actor_credit_v1`。任何 corrected implementation 必须使用新 credit/checkpoint contract，不能把历史 v1 checkpoint 重新解释为已修复模型。

## 可归因最小修复规格

本轮不实施修复。下一实现任务的最小规格冻结为：

1. 新 credit contract：`exact_masked_env_action_ppo_v2`。
2. rollout 以训练精度保存实际 sampled masked five-action behavior log-prob；rounding 仅用于日志。
3. update 用原 action mask 和当前 head logits 重建完全相同的 masked five-action categorical。
4. 主 actor objective 只使用一个 environment-action ratio 和一次 PPO clip。
5. policy entropy 也来自同一 masked five-action categorical。
6. external override transition 的 actor weight 保持 0；critic-only 与 shared parameters 的边界必须显式定义并测试。
7. canonical head labels 不再冒充 observed behavior actions；若保留 head auxiliary loss，必须单独命名、单独系数、单独消融，且不能替代主 PPO objective。
8. 严格提升 credit/checkpoint version；v1 checkpoint 只保留历史审计用途。

修复验收不是 reward 变高，而是 31 masks×全部合法动作在 probability、old/new ratio、clip objective 和 gradient 上与 exact reference 一致；单合法动作梯度严格为零，invalid actions 为零概率，override actor 梯度严格为零。

## E1 matched experiment 冻结草案

状态：`draft_not_execution_authorization`。只有新实现通过 E0 后才能另立执行任务。

| 条件 | 角色 |
|---|---|
| legacy CRDCM-SA v1 | 历史合同对照，不覆盖旧结果 |
| corrected CRDCM-SA v2 | 只改 behavior/update 数学合同 |
| corrected CRDCM-MAPPO v2 | 检查缺陷是否为共享层级路径 |
| unchanged CRDCM-PPO | 运行漂移控制 |

冻结参数：

- seeds=`1401/1402/1403`
- 每 cell 64 episodes，max_steps=20
- 固定 update16，不选最佳 checkpoint
- 训练上限 768 episodes
- learned evaluation 144 episodes；heuristic 12；总上限 924 episodes / 18,480 environment steps
- reward、temperature schedule、model size、window、workflow、action schema/mask、budget 和 checkpoint selection rule 全部不变

预先报告 completion、request failure、包含全部 failure 的 transfer/backhaul/migration 总成本、每完成 workflow 与每成功 request 成本及分母、逐 seed 动作/失败、优化一致性、wall-clock 和 peak memory。

停止规则：

- corrected contract 未通过 E0，禁止训练；
- E1 后不追加 seed、预算，不改 reward/temperature，不择 checkpoint；
- 数学正确但无可靠性能改善，停止扩张 SA superiority claim；
- 正确性修复的收益不等于 base-sharing×migration 机制收益，也不等于 graph/hierarchy 必要性。

## 问题结构隔离

| 问题 | 本轮状态 |
|---|---|
| implementation defect | 已确认：legacy surrogate 不是 masked five-action PPO |
| 修复能否改善 seed 1403 / completion | `UNVERIFIED`，属于 E1 |
| base sharing × migration 是否有独立/交互收益 | `UNVERIFIED`，属于后续机制实验 |
| graph / hierarchy 是否必要 | `UNVERIFIED`，必须在机制成立后单独消融 |
| PPO 是否已具 paper-grade 优势 | 否；旧结果只有 development、3 outer windows、无新 holdout |

## 证据与复现

机器产物：`artifacts/analysis/hierarchical_credit_consistency_e0_20260928/`

- `case_results.csv`：640 个逐 case objective/ratio/gradient 结果
- `checkpoint_readback.csv`：48 个 legacy checkpoint contract/hash 读回
- `summary.json`：汇总、最小修复规格与 E1 freeze draft
- `artifact_integrity_manifest.json`

复现：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -B -m pytest \
  tests/test_hierarchical_credit_consistency_e0.py -q

/Users/howen/Projects/PPO_MEC/.venv/bin/python -B \
  scripts/audit_hierarchical_credit_consistency_e0.py \
  --source-root artifacts/training/crdcm_performance_matrix_v2_20260928 \
  --output-dir <new-create-only-output-dir>
```

本轮验证 `4 passed`；environment steps=`0`、training updates=`0`。历史算法、checkpoint、evaluation 和受保护文件均未修改。
