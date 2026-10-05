# Calibrated workflow interface repair：有界重训结果（2026-10-06）

## 1. 审查身份与结论

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-06`
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `calibrated_continuous_workflow_interface_repair_20261006_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `run_source_git_commit`: `f00d212d681af2215062e0cb47e6bec0a5e49240`
- `evidence_level`: `E1_DOCUMENTED_WITH_AUDITED_NONFORMAL_DEVELOPMENT_VALIDATION`
- `paper_ready_verdict`: `Unverifiable`
- `final_decision`: **接口故障已纠正，SA-GHMAPPO 从异常崩溃恢复到接近完成，但没有相对公平对照收益；停止。**

旧 v1/v2 结果、checkpoint 和 integrity 保持原样。本轮新 run 不读取 formal/hidden/旧 holdout，不调用/下载真实模型，
不改 reward，不加 SA 消融，不按排名加 seed 或 episode。

## 2. Action-4 失败循环的判定

旧固定失败实例中，slow/fast/event raw argmax 为 `1/0/0`，本应聚合为 action 0；旧确定性路径却比较五个边缘动作概率。
event-prepare 的全部概率集中给 action 4，而 event-keep 被拆给 action 0–3，最终 action 4 以 `0.366485` 胜过 action 0 的
`0.321471`。随后发生以下链：

1. action 4 合法地只准备目标 RSU，不填当前 RSU；当前 bundle 缺失使当前节点服务失败。
2. 旧环境曾把目标 state/migration 提前记为成功；v2 已修为 current-node 成功后才提交。
3. 旧 mobility 由 `node_index` 驱动，服务失败使 node 不变，也冻结当前/目标 RSU 与预测状态。
4. 同一确定性状态反复产生 action 4，直到 step cap。旧 trace 为 `[4,0,4,4,4,4,4,4,4,4]`，只完成 2/5 nodes。
5. 旧 rollout 保存 canonical inverse head log-prob；同一步实际执行动作 log-prob 为 `-1.003798`，而优化的 head log-prob 为
   `-2.279641`，训练目标与执行动作不一致。

因此，**action 4 本身不是非法动作，也不应在当前缺模型时直接屏蔽**。旧无限式循环由“确定性边缘概率聚合偏置 +
executed/optimized likelihood 不一致 + failure-time mobility 冻结”共同造成；action 4 只准备目标是合同语义，不是 bug。

## 3. 确认缺陷与修复

| 层 | 确认缺陷 | v3 修复 | 历史兼容 |
|---|---|---|---|
| 环境 | failure 时 mobility 随 node 冻结 | 新 profile 按 decision step 推进 mobility，node 仍只按成功推进 | legacy profile 不变 |
| 环境 | 旧 action 4 可在当前服务失败时提交 state/migration | staged migration 只在 current node 成功后提交 | 旧结果不重写 |
| flat encoder | adapter count 除以 byte capacity；actor 不见 readiness/bytes/link | byte occupancy；actor/critic 同见 typed bundle、大小、estimated link、contact | 显式新 profile |
| graph encoder | 只见 adapter id，不验证 required base；忽略大小/link/state | full typed bundle readiness，并消费 bundle/state/input/link/contact | tensor shape 保留，语义不兼容 |
| action | deterministic env-marginal argmax 可覆盖 raw head argmax | independent head argmax → aggregate → mask projection | legacy contract 保留 |
| PPO | canonical heads log-prob 优化实际五动作 | SA/MAPPO 只用 executed env-action PPO；`log_prob==env_action_log_prob` | 旧 checkpoint 必须重训 |
| 审计 | canonical inverse 被标成 raw heads | 显式 `raw_head_actions_source`，另存 raw logits/probs | 新 ledger |

最小扰动测试证明 `cache_used_bytes`、typed base identity、`state_bytes`、estimated link/contact 和 current/target full-bundle
readiness 都进入对应编码。传入 ndarray observation 仍不是该 agent family 的合同输入；policy 使用 `semantic_state`，这点已显式记录。

## 4. 冻结执行与完整性

- SA-GHMAPPO、PPO、controller-MAPPO：3 seeds × 192 episodes，单 episode ≤24 steps；候选 checkpoint
  `48/96/144/192`，只用 4 个 dev instances 字典序选择。
- 理论总上限 `41,472` steps；实际 `12,478` steps、`216` updates、`51.70 s`；所有方法共享相同上限。
- 实际 steps：SA `1456/1408/1440`，PPO `1317/1308/1319`，MAPPO `1427/1389/1414`。
- selected episodes：SA `96/48/96`，PPO `144/48/192`，MAPPO `192/192/144`。
- regression 12 windows 是已暴露旧 evaluation；frozen check 8 windows 来自未用 dev-plan source indices 12–19。
  后者未使用新 template family，且不是 formal/hidden holdout，故只称开发验证。
- 训练曲线保存 1,728 rows；raw-head replay 保存 1,340 rows，action 序列与原 evaluation `0` mismatch。
- integrity：postprocess 前 59/59，最终 65/65 files 通过。checkpoint 保留本地，不提交 Git。

训练期 completion（576 episodes/method）为 SA `0.9826`、PPO `0.9983`、MAPPO `0.9913`；修复后训练与确定性评估
差距已从旧约 40 个百分点收敛到约 2–3 个百分点，但这不是方法优势证据。

## 5. 主要结果

数值为先按 window 平均 seeds，再对 source window 做 5,000 次 percentile bootstrap；seed 不作独立样本。

### 已暴露 regression

| 方法 | workflow completion | node coverage | deadline violation | service failure episode | 完成样本 elapsed / 覆盖 | transfer MB |
|---|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 0.944 [0.861,1.000] | 0.991 [0.979,1.000] | 0.500 [0.250,0.750] | 0.389 [0.222,0.556] | 74.73 / 34/36 | 717.77 |
| PPO | 1.000 [1,1] | 1.000 [1,1] | 0.778 [0.528,1.000] | 0.000 [0,0] | 101.32 / 36/36 | 760.28 |
| controller-MAPPO | 1.000 [1,1] | 1.000 [1,1] | 0.417 [0.167,0.694] | 0.194 [0.056,0.361] | 72.07 / 36/36 | 1211.82 |
| two-step rule | 1.000 [1,1] | 1.000 [1,1] | 0.167 [0,0.417] | 0.000 [0,0] | 52.30 / 12/12 | 169.03 |

旧 v2 相同 regression windows 上 SA completion 为 `0.750`；修复后为 `0.944`，描述性变化 `+0.194`。由于编码、
mobility 和 action likelihood 同时被纠正，该变化只能称 interface repair recovery，不能作单一机制消融或原创贡献。

### 新冻结开发检查

| 方法 | workflow completion | node coverage | deadline violation | service failure episode | 完成样本 elapsed / 覆盖 | transfer MB |
|---|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 0.958 [0.875,1.000] | 0.992 [0.975,1.000] | 0.833 [0.708,0.958] | 0.417 [0.333,0.542] | 106.53 / 23/24 | 670.68 |
| PPO | 1.000 [1,1] | 1.000 [1,1] | 1.000 [1,1] | 0.000 [0,0] | 107.70 / 24/24 | 676.41 |
| controller-MAPPO | 1.000 [1,1] | 1.000 [1,1] | 0.708 [0.375,1.000] | 0.167 [0.042,0.333] | 91.23 / 24/24 | 1084.48 |
| two-step rule | 1.000 [1,1] | 1.000 [1,1] | 0.375 [0.125,0.750] | 0.000 [0,0] | 51.67 / 8/8 | 65.03 |

SA 对 rule 的 completion paired delta 为 `-0.0417 [-0.125,0]`，deadline delta 为 `+0.458 [0.125,0.750]`，transfer
delta 为 `+605.66 MB [188.98,1056.35]`。未完成 episode 的 elapsed 未计作“更快”；表中单列完成样本覆盖率。

## 6. Action 4、缺模型与故障链对账

| split / 方法 | action 4 成功/尝试 | 当前模型缺失时 action 4 | 最大连续无进展 | raw→executed mismatch | log-prob 最大差 |
|---|---:|---:|---:|---:|---:|
| regression / SA | 99/174 | 44/83 = 53.0% | 5 | 11（全部 mask projection） | 0 |
| regression / MAPPO | 137/188 | 10/68 = 14.7% | 2 | 8（全部 projection） | 0 |
| frozen / SA | 53/110 | 33/66 = 50.0% | 4 | 3（全部 projection） | 0 |
| frozen / MAPPO | 78/119 | 5/52 = 9.6% | 2 | 0 | 0 |

旧 action-4 链中的“同状态永远不动”已消失：mobility、current/target RSU 与 mask 会随 step 改变；1,340-row replay
逐步复现原 action ledger，raw logits/probs、head argmax、aggregate、projection 和 executed action 全部可对账。但 SA 仍有
局部失败链：例如 frozen `seed=17/frozen_check_07` 在 current bundle missing 时，event prepare 概率约 `0.50–0.58`，连续
选择 action 4；移动会改变 RSU/ready state，故不是旧实现的 frozen-state loop，却仍造成 4-step 无进展并最终未完成。

这说明修复后剩余问题是**合法但不明智的 policy choice**。本轮保留原 SA event/temporal/auxiliary 配置，没有 event 消融，
因此只能把 event-head 偏置列为强可疑因素，不能宣称四个增强中的某一个已被因果定位。

## 7. 强规则权限与最终回答

two-step rule 不读 actual execution link，但拥有显式环境 transition clone 和 lexicographic objective；learned actors 只优化标量 reward。
这不是隐藏真值泄漏，却是更强的规划能力。规则在两组都 100% 完成，并在冻结检查给出最低 deadline violation、完成时延与传输量，
故当前 workload 下显式两步规划已经足够强，不应削弱它。

1. **action-4 循环由什么造成？** 旧循环由聚合概率偏置、executed/optimized log-prob 错位和 failure-time mobility 冻结共同造成；
   action 4 只准备目标是合法语义。修复后三者对账/推进正常，仍有短暂重复 action 4，来源转为 learned event policy 偏置。
2. **确认缺陷有哪些？** byte-capacity 单位、typed dependency/大小/link/state 字段丢失、mobility producer、raw-head 审计标注、
   deterministic 聚合和 PPO likelihood 均确认并修复；reward 未改。
3. **SA 是否有相对收益？** 没有。它从旧异常的 0.750 恢复到 regression 0.944 / frozen 0.958，是实现纠正后的正常化；PPO、
   MAPPO、rule 都为 1.000。没有证据支持 SA 原创机制优势。
4. **仍落后的证据指向什么？** 精确接口缺陷已不再解释差距；SA 在 current-missing 时仍约一半选择 action 4，支持当前 policy/
   event design 偏置。固定预算内的学习不足不能被排除，但训练 completion 已 0.983、三 seed 均完成 192 episodes，且强 rule 明显更省时省流量，
   所以证据更支持“当前方法设计仍不稳 + 对该 workload 两步模型规划已足够”，而不是继续无界加训练预算。

按冻结协议，本轮到此停止；不改 reward、不换数据、不扩大网络、不追加训练。
