# CRDCM SA 失败的一阶诊断（只读独立复核）

- `reviewed_at`: `2026-09-28T18:47:26+08:00`
- `literature_cutoff`: `2026-09-28`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `crdcm_performance_matrix_v2_20260928`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `97785936b047fb8035c2c81c8786ac1217185c06`
- `evidence_level`: `E3_REPRODUCED_OBSERVED_DATA_DEVELOPMENT_PILOT_NOT_HOLDOUT`
- `diagnosis_version`: `crdcm_sa_first_order_diagnosis_v1`
- `verdict`: 当前层级 SA 存在 actor behavior/update distribution contract 缺口；它是最可信的一阶修复对象，但其对 performance gap 的因果贡献仍为 `UNVERIFIED`。

## 结论先行

PPO 的 `27/36=75%` completion 相对 full-SA 的 `18/36=50%` 是重要反证，不能再把 SA 的低 transfer 单独包装成效率优势。逐请求 paired 复核显示：

- seed 1401 和 1402 的 SA/PPO 共 248 个请求，动作、reward、request success/failure 全部逐项相同；两者各完成 `9/12`。
- 差距全部来自 seed 1403：两者 124 个请求中只有 12 个动作相同；SA 在 112 个普通请求全部执行 `current_rsu_steady_offload`，PPO 全部执行 `current_rsu_cache_fill`。结果为 `74` 个 `PPO-only success`、`46` 个共同 success、`4` 个共同 failure、`0` 个 `SA-only success`；该 seed 的 workflow completion 为 SA `0/12`、PPO `9/12`。
- 因此这不是“PPO 在每个 seed 都普遍更强”，而是 seed 1403 发生了明确的 policy bifurcation，且 bifurcation 直接落在 cache fill 与 steady offload 的服务准备差异上。

源码与 trace 同时暴露一个优先级更高的实现合同问题：层级 agent 从精确的五动作聚合分布采样，却使用 canonical 三头标签做 PPO 更新。在全动作合法时，action 4 的真实概率只依赖 event 头，action 0/1 不依赖 fast；在真实 action mask 归一化后，更多共同因子会被约掉，例如只允许 action 0/2/3 时 event 对三者的条件概率完全无梯度。当前 actor loss 仍会训练这些 exact behavior-gradient 为零的头。flat PPO 没有该错配。

这足以判定为“相对于 on-policy PPO 行为分布的实现合同缺口”，但不能在不做新 matched retraining 的情况下声称它已经解释了全部 25 个百分点 completion gap。

## 范围与不可越界边界

本轮严格只读：

- 没有执行环境 step；
- 没有训练、续训、调参、换 seed、增加窗口或容量 stress；
- 没有改变 temperature、selection、checkpoint 或既有结果；
- 读取 12 个 run、48 个 `update_0004/0008/0012/0016.pt`、训练/评价 raw summary 与 update logs；
- replay 前后 48 个 checkpoint SHA-256 全部相同，7 个受保护文件 SHA-256 全部相同。

历史 summary 没有保存完整 `semantic_state`，所以 checkpoint replay 使用三个明确标注的、因果字段固定的 synthetic observations。它是 no-env-step sensitivity/aggregation probe，不是历史请求的 exact replay。

## 公平性、真实架构与冻结配置

每个 learned cell 都是 64 episodes、656 actual steps、16 updates，统一使用：

- learning rate `3e-4`
- clip ratio `0.2`
- entropy coefficient `0.01`
- value coefficient `0.5`
- batch size `32`
- train epochs `6`

但“训练预算一致”不等于“优化目标完全同构”：

| 条件 | encoder / critic | policy | head credit | 架构特有项 |
|---|---|---|---|---|
| full-SA / signal-off-SA | graph / centralized | slow(3)+fast(2)+event(2) | disabled，因此基础权重全为 1 | hierarchical conditioning；event temperature `1.4→0.85`/8 updates；sharpening final `2.3`；temporal consistency `0.35`；event entropy scale `3.5` |
| full-MAPPO | flat / centralized | slow+fast+event | `aggregation_reason_weighted_controller_ppo_v3` | reduced irrelevant-head weights/floors；无 temperature decay/sharpening/temporal consistency |
| full-PPO | flat / independent | 单一五动作 categorical | 不适用 | 直接对实际五动作行为分布做 PPO |

full-SA 与 signal-off-SA 的唯一区分仍是 CRDCM residual 是否进入输出；signal-off residual 为严格零。所有 checkpoint restore 的 `update_count`、feature mode 与结构均通过 strict load。

因此该矩阵满足冻结架构的 matched budget 比较，但不是 iso-objective 或 iso-inductive-bias 比较。不能把复杂架构默认超参数的后果改写为数据不公平，也不能把相同预算当作“已证明训练充分”。

## state → action → execution → reward → update 证据链

五个环境动作语义为：

| ID | 语义 |
|---:|---|
| 0 | 当前 RSU cache fill required adapter |
| 1 | predicted next RSU prefetch |
| 2 | vehicle fallback |
| 3 | 当前 RSU steady offload、无 cache mutation |
| 4 | handoff migration prepare |

层级输出先得到 `slow / fast / event` logits。五动作概率的精确 factorization 是：

```text
P(a0) = P(event=keep) × P(slow=current_fill)
P(a1) = P(event=keep) × P(slow=prefetch)
P(a2) = P(event=keep) × P(slow=no_change) × P(fast=vehicle)
P(a3) = P(event=keep) × P(slow=no_change) × P(fast=current_rsu)
P(a4) = P(event=prepare)
```

这五项严格和为 1；action mask 再将非法动作置零并归一化。评价 trace 中用保存的 head probabilities 重建 behavior log-prob，12 个 condition×seed cell 的最大绝对误差均小于 `3.1e-6`，证明保存的五动作行为概率链是自洽的。v2 中 guard/planner override 为零，所以链条是 `aggregated_policy_action == executed_action`，reward 由实际动作产生。

问题出现在 update：采样后的五动作被反映射为 canonical head targets。例如 action 4 被写成 `slow=0, fast=0, event=1`，但 `slow=0/fast=0` 并非 behavior 中被采样或观察到的潜变量。当前层级 actor loss 分别计算每个头的 PPO ratio、分别 clip、再按权重平均；它没有使用实际采样的五动作 log-prob ratio 作为主 actor objective。

full-SA 又关闭 head-credit protocol，使三头基础 actor 权重均为 1。本审查直接对每步保存 logits、mask 和实际动作做 autograd，只有当某个 head 对 exact masked behavior log-prob 的梯度范数 `<=1e-10`、但当前 actor weight 仍大于零时，才计为错配。训练 trace 中：

- full-SA：`1919/1968=97.51%` steps 对 exact behavior-gradient 为零的 head 给了非零 actor credit；
- signal-off-SA：`1894/1968=96.24%`；
- full-MAPPO：`1940/1968=98.58%`，虽使用较低权重，但仍非零；
- full-PPO：`0/1968`。

确定性 development evaluation 中，full-SA、signal-off-SA、full-MAPPO 都是 `372/372`，PPO 为 `0/372`。其中 full-SA/MAPPO 的零梯度 credit 组合为：`event|fast=224`、`event=112`、`fast|slow=24`、`fast=12`。这不是日志缺字段推测，而是由 source formula、保存的 per-step logits/mask、canonical mapping、autograd 和 checkpoint config 联合验证。

### override、critic-only 与 shared representation

CRDCM 在 external override 时把 actor credit 置零，并将这些 transition 送入 `_critic_only_update()`。该 value-only backward 会经过共享 encoder 与 CRDCM residual 的共享隐藏层，因此即使 actor loss 为零，也可能间接改变 actor logits；“没有直接 policy loss”不等于“actor output 保持不变”。

不过 source run 的 12 个训练 cell 合计 external override masked samples 为 `0`，评价 override 也为 `0`。所以这是未来合同风险，不是本次 SA/PPO gap 的 active cause。

### advantage、entropy、clipping 与终止

- GAE 使用 `gamma=0.99 / lambda=0.95`；每个 episode 独立 finalize，time-limit truncation 保留 bootstrap，真正 termination 才将 bootstrap 置零。
- 本次 768 个训练 episodes 的 endpoint `right_censored=0`，因此没有发现由 horizon truncation 造成的条件差异。
- 四条件训练 step reward 范围一致，均为 `[-4.25, 14.02]`。均值差异来自访问动作/结果不同，不能证明某架构获得了不同 reward scale。
- full-SA 对每个头独立计算和 clip ratio；event 还叠加 reliability、extra gain、temperature/sharpening 与 entropy scale。其 actor objective 与采样用的 joint five-action categorical 不同。
- `env_action_log_prob_missing_count=0`；问题不是 log-prob 缺失，而是正确的 behavior log-prob 没有成为默认层级 PPO 主 loss。

## checkpoint 4/8/12/16 固定观测 replay

三个 probe 只改变 CRDCM v1 可观测字段，base semantic state 固定，不调用环境：

1. target warm + state ready；
2. current adapter miss + target cold；
3. capacity conflict + state missing。

在第二个 probe 上，seed 1403 的 handoff-prepare probability 随 checkpoint 演化为：

| 条件 | update 4 | update 8 | update 12 | update 16 | update 16 entropy |
|---|---:|---:|---:|---:|---:|
| full-SA | 0.635 | 0.734 | 0.851 | 0.865 | 0.560 |
| full-MAPPO | 0.503 | 0.536 | 0.564 | 0.579 | 1.249 |
| full-PPO | 0.247 | 0.260 | 0.280 | 0.307 | 1.572 |
| signal-off-SA | 0.548 | 0.443 | 0.677 | 0.815 | 0.697 |

三 seed 的 signal-off-SA 在同一 checkpoint 内对三个 CRDCM-only perturbations 的 action-distribution L1 差严格为 `0`；full-SA/MAPPO/PPO 均有非零响应，说明 full residual 确实进入 logits，而 off arm 确实切断该路径。

该 replay 支持“SA 默认 schedule 形成更尖锐的 event 分布”和“full residual 对 CRDCM 字段敏感”，但不支持“这些 synthetic states 就是 seed 1403 的历史失败状态”，也不支持事后调 temperature 或选择 checkpoint。

## 效率重新解释

全 36 episode、保留 failure cost 的统计为：

| 条件 | completed workflows | successful requests | failed requests | total transfer MB | transfer / successful request | transfer / completed workflow |
|---|---:|---:|---:|---:|---:|---:|
| full-SA | 18 | 286 | 86 | 7,432 | 25.986 | 412.889 |
| full-MAPPO | 18 | 248 | 124 | 7,432 | 29.968 | 412.889 |
| full-PPO | 27 | 360 | 12 | 10,224 | 28.400 | 378.667 |
| signal-off-SA | 14 | 319 | 53 | 9,410 | 29.498 | 672.143 |

full-SA 的总 transfer 较低，但也少完成 9 个 workflow、少成功服务 74 个请求。按 successful request 归一化时 SA 数值较低；按 completed workflow 归一化并保留所有 failure cost 时 PPO 反而较低。不存在无条件效率 dominance。

common-success 是 post-outcome selected：seed 1401/1402 的 SA/PPO 各有 9 个共同成功单元，且成本逐项相同；seed 1403 没有共同成功单元。这个切片只能说明共同成功路径相同，不能删除 SA 的失败代价后宣称其更省。

## 原因分类

| 候选原因 | 结论 | 证据边界 |
|---|---|---|
| checkpoint 串线 / restore 失败 | 排除 | strict load；update 4/8/12/16；48 checkpoint 前后 hash 相同 |
| budget 不公平 | 排除 | 每 cell 同 64 episodes、656 steps、16 updates |
| action mask 单动作强迫 | 排除 | trace 中行为分布和 mask 可重建；实际输出覆盖多个动作 |
| external override credit | 本 run 不活跃 | train/eval override 均为 0 |
| aggregation implementation | 五动作概率实现正确 | factorization 和 trace log-prob 误差 < `3.1e-6` |
| multihead actor credit | **确认存在实现合同缺口** | 行为从 masked five-action distribution 采样，更新却给 exact behavior-gradient 为零的 canonical heads 非零 credit |
| exploration / initialization | 可信放大因素，未单独识别 | seed 1403 bifurcation；SA event schedule 更尖锐；仅 3 seeds |
| training budget insufficient | `UNVERIFIED`，不能默认 | update-stage变化不等于继续训练会改善；无预注册收敛判据 |
| rational but suboptimal objective | 可能 | reward 下 steady offload 可有即时正 reward，但累积 readiness/completion 较差；需新的因果验证 |
| architecture unnecessary | 当前 evidence 支持暂停复杂路线，而非永久证明 | PPO 在该 pilot 更稳，但无 formal/holdout，只有 3 outer windows |

## 唯一优先建议

暂停 capacity-stress 和性能扩展。下一步只做一个 factorization-consistency validation：对每个合法五动作，在相同固定状态上比较

1. 实际 aggregated behavior log-prob 的 PPO ratio/gradient；
2. 当前 canonical per-head surrogate 的 ratio/gradient；
3. 对 action 0/1/4 将 latent head credit 置零后的 ratio/gradient。

验收首先要求数学/自动测试层消除 behavior/update mismatch；随后才可另立实现任务，发布新 contract，在预注册的相同 seed/budget/window 上做一次 matched retraining。不得靠加 seed、延长训练、调 temperature 或择 checkpoint补救。

停止规则：若修正后仍不能消除固定观测 gradient mismatch，或一次预注册 matched retraining 不能改善 completion 且不恶化包含 failure cost 的总成本，则停止复杂 SA 路线，保留 flat PPO 作为当前 supported candidate。这个建议区分了：

- **实现缺陷**：当前层级 PPO credit 不对应实际 behavior distribution；
- **科学失败**：修复后是否仍不优于 PPO，当前尚未验证。

## 安全与禁止表述

安全表述：

- PPO 在该 development pilot 描述性领先：`27/36` vs full-SA `18/36`。
- seed 1403 的 cache-fill vs steady-offload bifurcation 是 observed gap 的直接执行层来源。
- 当前层级 actor credit 与实际五动作 behavior distribution 不一致，是优先级最高的实现修复对象。

禁止表述：

- “SA 比 PPO 更节省传输”——没有归一化口径和 completion 边界时不成立。
- “训练更久就会修好”——没有证据。
- “credit bug 已经因果解释全部性能差距”——需新 contract matched retraining。
- “PPO 已证明论文级优越”——只有 development、3 outer windows、无 formal/holdout。

## 证据文件

派生产物位于 `artifacts/analysis/crdcm_sa_first_order_diagnosis_20260928/`：

- `diagnosis.json`
- `training_config_and_credit_audit.csv`
- `training_trace_credit_summary.csv`
- `evaluation_step_credit_audit.csv`
- `evaluation_credit_summary.csv`
- `sa_vs_ppo_request_pairs.csv`
- `sa_vs_ppo_request_pair_summary.csv`
- `checkpoint_fixed_observation_replay.csv`
- `efficiency_all_episode_summary.csv`
- `common_success_selected_summary.csv`
- `artifact_integrity_manifest.json`

复现入口：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/diagnose_crdcm_sa_failure.py \
  --source-root artifacts/training/crdcm_performance_matrix_v2_20260928 \
  --output-dir artifacts/analysis/crdcm_sa_first_order_diagnosis_20260928
```
