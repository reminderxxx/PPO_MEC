# Event auxiliary abstention 单变量候选（2026-10-10）

## 状态与审查身份

- `reviewed_at`: `2026-10-10`（Asia/Shanghai）
- `literature_cutoff`: `2026-09-28`；本轮没有新增文献检索或 novelty 评价
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `cscwd_event_aux_abstention_ab_20261010_v1`（预留，尚未创建 scientific root）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `source_science_commit`: `f46ec72b15f534ac44768a83ef6316c1cfcb6b58`
- `implementation_commit`: `46a68f11c289ccc304b88cdfed01c34ba5f61c3d`
- `independent_gate_commit`: `e9b19f6ad2ec432487a0fa7fbdc6542dfb742dc1`
- `evidence_level`: `E2_INDEPENDENT_INTERFACE_CONTRACT_AUDITED_NO_SCIENTIFIC_RUN`
- `verdict`: `INDEPENDENT_GATE_PASS / execution_authorized=true / performance_unverified`

旧的 service-feasible hard-zero 候选保持 `MIXED_STOPPED`：不得覆盖、放宽门槛或启动其五 seed 训练。本文件冻结一个不同的、
更弱的单变量候选。当前只有实现、合成梯度验收、历史原件 hash 预检和 0-step 回执；没有候选训练、评价、formal/holdout、
性能提升或论文贡献证据。

## 为什么从 hard-zero 改为 abstention

既有 v4 事件链确认：当前完整 bundle 缺失时，SA 的 event auxiliary 经常仍给 action4 方向正标签；但 action0/2/4 冻结后缀
又发现 action4 虽牺牲当前服务，却能通过目标缓存副作用改善后续服务。故把 hard/soft target 统一重标为 0 会错误声称这些状态的
长期正确动作必为非 action4，MIXED 门禁已经否定这一强假设。

当前证据只支持更窄的干预：**在 current-missing 样本上不让短期 event 伪标签监督 event head，把长期 action4 价值继续完全
交给 PPO return；current-ready 样本维持原监督。**这不声称旧标签为错，也不把 target 改成 0。

## 唯一变量和实现边界

显式 default-off 开关为 `mechanism_aux_missing_current_event_abstention_enabled`。仅 SA-GHMAPPO 在 v4 公共观测 profile 下消费：

```text
event_supervision_weight = 1 if current_complete_bundle_ready else 0
event_loss = weight * existing_event_CE
temporal_margin_loss = weight * existing_event_soft_target_BCE
```

- current-ready 时 hard label、soft label、loss coefficient 和逐 logit 梯度与旧版完全相同；current-missing 时不重标 label，
  只令 event CE 与依赖同一 label 的 temporal consistency/margin 样本权重为 0。
- slow/fast auxiliary、confidence、PPO actor、value、entropy、reward、network、optimizer、action mask/guard 与 raw inference 不变。
- loss 仍按旧版 confidence-eligible `loss_terms` 分母取 mean；abstain 样本保留在原分母中，不按 supervised valid count 重归一化。
- readiness 只由当前公开 `semantic_state` 的 vehicle/current RSU/current node 和 `bundle_ready` 计算；篡改预测未来不改变权重。
- 开关只传给 `sa_ghmappo`；PPO/MAPPO/DT 不接收候选语义，旧 checkpoint 继续加载且初始化张量、动作概率和 mask 不变。
- checkpoint 显式记录 `missing_current_event_abstention_v1`，拒绝跨语义加载；它与旧 hard-zero 开关互斥。

局部验收覆盖 current-ready 精确梯度不变、current-missing event 梯度为零而 PPO event 梯度非零、mixed/all-missing 稳定、固定
分母、公共 readiness、旧 profile 拒绝、checkpoint fail-closed、真实 8-transition learn 路径和三种基线身份。候选与 control
均为 `165,512` 参数、60 个 state keys，seed 7 初始化张量完全相同。

## 时间/接口合同边界

现有 36 个实例的 v3/v4 `rsu_sequence` 是由 NGSIM handoff-pressure 构造的 synthetic block，不是逐 frame mobility trace。
环境移动口径为 `mobility_progression=decision_step_index`；`decision_step_seconds=5` 是人工接触尺度，deadline 则累积 modeled action
cost。既有 90 分支/463 row 与该实现合同一致，未发现确定性接线 bug；但不能据此主张真实时空外推、真实无线接触或 frame-time
realism。A 必须在本实现 clean commit 上独立确认接口、时间、SA-only、当前 ready/missing 梯度和无未来信息后，形成带 commit/tree、
report/manifest/time receipt/abstention receipt SHA-256 的 PASS。A 已在上述 implementation commit 上完成该门禁：30/30 raw-policy
forward 相同，6/6 current-ready 梯度精确相同，24/24 current-missing event 梯度为零，0 replay/training step；B 已独立复核
A commit/tree、四个文件 SHA-256 和全部 runner 消费字段。此 PASS 只授权冻结 A/B，不是性能或论文 PASS。

## 条件 A/B（尚未授权）

只有不可变 A 门禁 PASS 后，才允许一次无重试 SA-only matched run：seeds `[7,17,29,43,61]`，每 seed 5,760 environment
steps、96 updates、768 optimizer steps，总计 28,800 steps/3,840 optimizer steps；checkpoint candidates 固定
`[24,48,72,96]`。control 精确复用 v4 原 SA selected/update96 及强基线/规则 hashes，不重训、不重评价；candidate 只新增
selected 100 + fixed96 100 episodes，总新评价上限 200。全部 36 实例已暴露，只能称 development A/B。

主指标为 completion、on-time completion 和 service failure；联合报告 failed attempts、current-missing action4、state commit、
prepared-prefix reuse/stale、recompute、model/state/input bytes、共同完成 elapsed/coverage，以及直接记录的 event-supervision fraction
与 auxiliary gradient。reward 只作次要指标。

否证/停止条件保持唯一：任一视角或 split 的 completion 下降即 FAIL；selected 与 update96 必须都满足 on-time 不降、failure
不升且至少一项严格改善，并要求每视角至少 3/5 seed 改善、至多 1/5 恶化。否则为 FAIL/MIXED 并停止，不转第二候选、不改
权重/门槛、不补 seed、不扩预算、不用最终检查选 checkpoint。SA 是否排名第一不是通过条件。

## 当前答案

当前最值得补足的是 **短期 event 伪标签与 action4 长期缓存收益之间的信用分配边界**：current-missing 时不再由该伪标签强推
action4，也不反向强推非 action4，而让 PPO return 决定。证据是既有 event 标签在失败状态实际生效，同时 hard-zero 被多步严格
更优反例否定。下一轮只改变这一个 event-supervision sample weight；若机制日志未显示预期 abstention，或双视角 completion/
on-time/failure 门不通过，就否定候选并停止。
