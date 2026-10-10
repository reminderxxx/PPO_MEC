# Prepared-state v4 预测、准备与到达链只读诊断

## 审查身份和范围

- `reviewed_at`: 2026-10-10 10:38 Asia/Shanghai；`literature_cutoff`: 2026-09-28（沿用项目最近一次文献审查，本轮未作 novelty 检索）；`target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `cscwd_causal_prepared_state_visibility_matched_20261010_v1`；科学 commit `f46ec72b15f534ac44768a83ef6316c1cfcb6b58`；实现 commit `709746bc1f1ea3037f27497bdb51cf6f45c8963c`；诊断事前方案 commit `5c83367`。
- `evidence_level`: 对此 development run 的 provenance、119 文件与逐步执行链为 `E2_ARTIFACT_AUDITED`，并完成确定性记录动作重放；不是正式 comparison package 的 `E3_REPRODUCED`。formal/holdout/support 尚缺，任何 TMC-ready、canonical 或固定论文贡献判断仍为 `Unverifiable`。
- 来源：`/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1/`；本轮机器诊断：`artifacts/analysis/cscwd_prepared_state_event_chain_diagnosis_20261010_v1/`。两处均未改动原科学产物。terminal=`PASS`、completion=`complete`、训练 115,200 steps、formal/holdout reads=`0`，119/119 SHA/size 匹配。按冻结规则仅重放 new selected 与 new update96 各 400 episode，共 800 episode、5,632 step；所有逐步 service、state、bytes、clock 及 episode 指标与原记录一致。其余历史视图只作对照，未重评。

## 核心计数

以下每个 view 每种方法都是 100 个 development episode（`regression` 60、`frozen_check` 40）。同窗口五个 seed 和两个 checkpoint view 非独立样本，数字是描述性计数。
完整的 view × split × source segment × method × seed 共 160 个分层单元及预测命中、准备、复用、重算、失败、成本分母，见 `split_region_seed_rows.csv`；不得把这些单元当成 160 个独立时间窗口。

| view / 方法 | 完成 | 按期完成 | 有服务失败的 episode / 失败事件 | action4 尝试 / state commit | 非 fallback handoff 中有效复用 / 重算 / 当步服务失败 |
|---|---:|---:|---:|---:|---:|
| selected / SA | 100 | 38 | 32 / 50 | 253 / 166 | 50 / 137 / 14 |
| selected / PPO | 100 | 35 | 0 / 0 | 0 / 0 | 0 / 70 / 0 |
| selected / MAPPO | 100 | 53 | 19 / 40 | 435 / 311 | 131 / 115 / 0 |
| selected / DT | 100 | 49 | 7 / 13 | 261 / 190 | 83 / 67 / 0 |
| update96 / SA | 100 | 43 | 26 / 39 | 248 / 169 | 51 / 134 / 8 |
| update96 / PPO | 100 | 33 | 0 / 0 | 0 / 0 | 0 / 54 / 0 |
| update96 / MAPPO | 100 | 53 | 8 / 9 | 358 / 292 | 106 / 86 / 0 |
| update96 / DT | 100 | 50 | 1 / 1 | 167 / 147 | 65 / 48 / 0 |

`handoff` 记录是服务尝试，失败时 `last_execution_rsu` 不更新，可能在同一到达位置重复计数。车辆 fallback 的 `state_ready=True` 只表示无需 state；绝不计入有效复用。selected SA 的 319 次 handoff 尝试中，118 次是 fallback、201 次非 fallback；后者正好分成上表 50/137/14。update96 SA 相应为 113 次 fallback、193 次非 fallback，分成 51/134/8。修正口径在 `handoff_interpretation_rows.csv`。

冻结 `frozen_check` 的 SA：v3 historical selected→v4 new selected，完成 `39/40→40/40`、按期 `5/40→9/40`、有失败 `8/40→13/40`；v3 historical fixed96→v4 new fixed96，完成 `39/40→40/40`、按期 `11/40→11/40`、有失败 `9/40→9/40`。两视图未共同兑现更低失败/更高按期，不能仅凭 selected 的改善晋级。selected 新方法横比按期 SA `9/40`、PPO `8/40`、MAPPO `15/40`、DT `13/40`，且后两者分别有 `10/40`、`3/40` 失败 episode；需保留完整多指标权衡。

## 第一个断点与完整链

1. 两 view 的 SA 首次服务失败分别为 `32/100` 与 `26/100` episode，**全部**是当前 bundle 缺失时选择不会填充当前 bundle 的合法动作：selected `action4=28, action1=3, action3=1`，update96 `action4=23, action1=3`。没有出现“bundle 已 ready 仍报失败”或 action0 准入失败。失败事件总量分别为 50、39，其中 action4 占 46、36。MAPPO/DT 的首次失败也同属这一类；这是共有 action contract 下的策略选择风险，不能称 SA 独有实现 bug。
2. action4 首先在**预测目标**调用 `_admit_bundle`；即使其 model admission/contact 可行，也只有当前节点服务成功后才 commit prepared state。selected SA 46 个失败 action4 中，42 个显示 `staged_pending_current_node_completion` 且目标准备 preview 可行，仍因当前 bundle 缺失而无法推进节点/提交 state；另 4 个目标准备不可行。update96 对应 `32/36` 与 `4/36`。目标模型 cache 可能变化、model bytes 可能计费；不应把这些行写成“state 迁移成功”。
3. selected SA 的非 fallback 到达中，未 ready 的 151 次尝试依首个链路断点为：未向实际目标发起 prepare 93、曾向实际目标 prepare 但当前服务失败无 state commit 31、prepare admission/contact 失败 18、曾 commit 后随节点进展 prefix 过期 5、准备了其它预测目标 4。update96 的 142 次为 92、23、19、7、1。`recompute_seconds>0` 只在服务可执行的未 ready 行发生；14/8 个当步服务失败没有同一步重算。这些是尝试级多标签时间链，不能解释为独立 handoff 事件数。
4. 预测错目标并非主要观测断点：selected SA 的 253 次 action4 中，248 次预测目标与后验 horizon 内首次 handoff RSU 一致；update96 为 243/248。这只是所消费窗口的后验审计，不能证明预测器跨独立 trace 泛化。当前模型可用性与 state prefix 新鲜度是不同条件；已准备成功后继续完成节点仍可使 prefix stale。

正例：selected、`frozen_check_00`、seed 17，step 3 针对 `rsu_2` 的 action4 成功完成当前节点并提交 state；step 4 在 `rsu_2` 以 action3 服务，`state_ready=True`、重算 `0 s`。负例：selected、`regression_02`、seed 7，step 1、2 当前 bundle 不可用而连续选择 action4，均服务失败、state commit=false、各增加约 2 s；step 3 到达 `rsu_1` 后有效执行但未有可复用前缀，发生 `4.316 s` 重算。该负例没有证明改选 action0 的完整策略净收益；原路径已足以判定失败链，事前允许的可选 action 分支执行 `0/24`。

## 辅助目标与实现判断

- SA 训练的 `auxiliary_coef=0.1` 确实生效：本 run `optimizer_step_records.csv` 中 SA 的 `3840/3840` optimizer step 辅助损失及加权辅助梯度范数均非零；三个 baseline 对应为零。代码最终生效的 `_build_mechanism_targets`：目标 adapter 已缓存且时机分数过阈值时给 `event_target=1`，**没有以当前 bundle ready 或 action4 后能否完成当前节点作前置条件**。event cross-entropy 和时序 margin 项会在训练中鼓励此类状态的 event head。
- 按事前排序有界探测 24 条 SA 首次失败记录（仅 7 个不同的公开状态、14 个不含 view 的 episode key；不得当 24 个独立样本），24/24 当前 bundle 缺失；22/24 目标 bundle 已 ready、伪 `event_target=1`，21/24 实际 action4；22/24 的记录 action4 概率大于 action0。24 次只重建已记录的失败前缀并计算伪标签，**无新 policy forward、无 action 分支**。原策略的 raw head/probability 来自冻结 ledger，逐行见 `auxiliary_probe_rows.csv`。
- 评估时 `policy_evaluation_mode=raw_policy`，`_forward_policy` 跳过 `_apply_policy_adjustments`；训练形成的权重仍受辅助损失影响，但运行时没有再加机制 logit bias。上述标签不含当前可服务性，与错误 action 的方向吻合；没有消融反事实，不能把 SA 与 baseline 差距单独归因于辅助损失。
- 环境执行代码与重放一致：action4 是“预测目标 model/state 准备 + 当前节点服务”的组合动作，当前节点失败不提交 state。现有证据支持合法但次优的策略选择以及 label/feasibility 语义缺口；**未确认环境或 ledger 的执行实现 bug**。`migration_success` 是 calibrated simulation 的 state commit，不是 production technical execution-authority 迁移。

## 判定与后续任务边界

本轮只读诊断不自动改算法、mask、reward、guard 或结果筛选。下一轮独立实现任务应先冻结 action4 的训练标签是否要求当前可服务、以及当前 bundle 缺失时 0/2/4 的成本比较语义；随后以相同信息权限和预算、预注册双视图门检验，保留 PPO/MAPPO/DT 与失败率、按期完成、transfer、recompute 全指标。不能用已经消费的 `frozen_check` 反复挑选方案，也不能把此 development run 写成正式贡献或 paper-ready 证据。
