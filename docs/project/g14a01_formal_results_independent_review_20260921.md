# G14A01 正式结果独立复核与论文主张冻结

- `reviewed_at`: `2026-09-21`
- `literature_cutoff`: `2026-09-21`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_post_ablation_20260921_g14e07_pending`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `repository_git_commit`: `62f432c68b05b8dcd148fde454dd71d1d9621d68`
- `scientific_commit`: `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`
- `executor_commit`: `1770401ccc2f4ce15b5d5adcabdecb386fd90a05`
- `evidence_level`: `below E2; targeted raw-artifact audit of the formal-only package, without a full 29,909-file independent rehash and without holdout evidence`
- `verdict`: `Unverifiable for paper-ready; formal-only completeness is complete, performance superiority is not established`

## Executive verdict

G14E07 已合法完成不含 holdout 的正式执行闭环。`formal_gate.json` 是 completeness-only、outcome-blind gate；
它确认矩阵、ledger、provenance 和输出完整，不是性能通过门禁。`complete_without_holdout / completed /
return_code=0` 可以引用为执行事实，不能翻译成算法优势或 paper-ready。

主统计的安全读法是：SA-GHMAPPO 在极少数 formal 窗口出现微小的 service-ready / continuity 增益，但在大多数
窗口增加传输；完成 workflow 的延迟在可比较子集中完全相同。当前证据不支持“全面更优”“降低传输”或
“显著优于所有 baseline”。

## Evidence inventory

- 主统计：`statistics/paired_statistics.json`，SHA-256
  `0fbba996b27e3a18121a9a7c0f3b22cc4b896bb9c0909ddfc3eccae00cdaa0f7`。
- 完整性 gate：`formal_gate.json`，SHA-256
  `20a688c4cf5b3a0e269cfbae86bf65d3f6b6cbf6cc1c6f8dacfbe1d33e0cf36a`。
- evaluation-only 初始化状态：`evaluation_only_state.json`，SHA-256
  `11604009b220028a3ff557acad58f21c3957d79e64bb70ccd68450e7857f8f28`。
- manifest 声明 `29,909` 个内容文件、0 个重复路径；本轮未对全部内容文件逐个独立重算 SHA-256。
- 实际目录含 `29,912` 个文件：manifest 覆盖的 29,909 个文件，加上 manifest 自身、后续追加的
  `phase_state.jsonl` 和 `formal_gate.json`。该差值与 manifest 的 live-file 排除时序一致。
- `phase_state.jsonl` 确认 `complete_without_holdout` 于 `2026-09-21T11:34:09.487652+00:00`
  完成，`return_code=0`。
- formal 主结果来自 3 个 capacity controller CSV；每档 2,700 行，共 8,100 行、540 个
  `(capacity, seed, window, workflow)` 配对单元，每单元 15 个 agent。
- formal window plan 有 12 个原始窗口；按 `source_segment_run_id + raw frame interval` 复核为 0 重叠，
  同一 segment-run 的最小 frame gap 为 24。
- holdout 仍 `sealed=true / opened=false / consumed_permanently=false`。G14E07 execution contract 明确
  `holdout_capability=false`、`holdout_commands=0`。

## 1. 统计方向与 claim map

`paired_statistics.json` 已把所有指标转换为统一方向：`signed_positive_favors_candidate=true`。
因此每行只应按 signed CI 判定：

- `ci95_low > 0`：candidate-favorable；
- `ci95_high < 0`：candidate-adverse；
- 否则：mixed；
- 不得再按 `higher_is_better` / `lower_is_better` 二次翻转。

当前 gate 的 `_claim_evidence_rows()` 对 lower-is-better 指标做了第二次翻转。由此产生的
`12 supported / 72 mixed` 中，12 个 `supported` 全部是 `transfer_mb_per_request` 的负向 signed CI，
实际均为 candidate-adverse。按冻结 signed 语义纠正后应为：

- `0 supported`；
- `72 mixed`；
- `12 contradicted`。

对 LRU，raw candidate-minus-baseline transfer delta 为 `+2.440697 MB/request`；signed delta 为
`-2.440697`，BCa 95% CI `[-4.951683, -1.307530]`。这表示 SA-GHMAPPO 多传输，不是节省传输。

## 2. 统计单位与 sign test

BCa / percentile CI 的实现使用 `window_id` 为外层 cluster，`seed + workflow_id` 为窗口内层；这一部分
没有把 540 行当作 540 个外层独立窗口。

但 `sign_test_pvalue` 的实现直接在全部 paired-row delta 上计数 wins/losses，未使用 12 个窗口均值。
因此现有 sign test 和基于它的 Holm p-value 存在行级伪重复，不能作为论文显著性证据。

只读复算先在每个原始窗口内平均 capacity、seed 和 workflow，再以 12 个窗口效应做 exact sign test：

| 对 LRU 的 endpoint | window wins / ties / losses | exact p | 84 项 Holm p |
|---|---:|---:|---:|
| full-service-ready byte hit | 2 / 10 / 0 | 0.500000 | 1.000000 |
| joint base-adapter hit | 2 / 10 / 0 | 0.500000 | 1.000000 |
| full-service-ready request | 2 / 10 / 0 | 0.500000 | 1.000000 |
| workflow continuity | 2 / 10 / 0 | 0.500000 | 1.000000 |
| transfer MB/request | 0 / 3 / 9 | 0.003906 | 0.328125 |
| completed-workflow delay | 0 / 9 / 0 | 1.000000 | 1.000000 |

84 项 family 内没有 window-level Holm-adjusted sign test 通过 0.05。CI 与 sign test 回答的问题不同：
12 个 transfer CI 可判为 candidate-adverse，但不能继续引用当前行级 Holm p-value 宣称多重校正后的显著性。

## 3. 延迟缺失机制

`end_to_end_workflow_delay` 只在 workflow 完整、未 censor、全部 exposed requests 成功时为 finite；失败或未完成
workflow 为 null。formal 主表中每个 agent 都是 `220/540` finite，`320/540` unavailable：

- 可用率 `40.74%`；缺失率 `59.26%`；
- 只覆盖 9/12 个外层窗口；
- 15 个 agent 的 availability mask 完全相同；
- 220 个共同 finite pair 的差值全部为 0。

这消除了 agent 间差异性缺失这一特定问题，但没有消除 survivor-conditioned estimand：延迟只描述成功完成的
workflow，不处罚 59.26% 的失败/未完成样本。当前延迟 endpoint 对策略没有区分力。论文必须同时报告 completion /
continuity，并把 delay 写成“conditional on completed workflows”；不能写“端到端延迟更低”。

## 4. 三容量结果

以下为 SA-GHMAPPO 相对 LRU 的 signed mean；transfer 为负表示 SA-GHMAPPO 多传输：

| Capacity | ready / continuity | joint hit | transfer MB/request | 非零外层窗口 |
|---|---:|---:|---:|---|
| 288 MB | +0.000327 | +0.001634 | -0.155265 | gain 1/12；transfer adverse 4/12 |
| 576 MB | +0.001961 | +0.003268 | -3.583413 | gain 2/12；transfer adverse 9/12 |
| 864 MB | +0.001961 | +0.003268 | -3.583413 | gain 2/12；transfer adverse 9/12 |

576 MB 与 864 MB 的六个 primary endpoints、reward 和 episode success 在全部 2,700 个对应行上完全相同；
只有 capacity occupancy 一类分母相关字段变化。因此当前证据只能说明 288 MB 与较大容量 regime 不同，不能证明
576→864 MB 带来可观测扩展收益，也不能把三档写成三份独立样本。

## 5. 消融与 support

冻结 typed-semantics matrix 有 6 个预注册 level，但只有 `typed_full` 和 `no_prediction` 可执行；
`legacy_adapter_only`、`no_base_sharing`、`no_workflow_state_migration`、`fixed_no_eviction` 均为
`unavailable_pre_execution`。因此 typed base sharing、workflow-state migration 和 fixed-no-eviction 的独立贡献仍
不可验证。

在 576 MB、同 12 窗口、5 seeds、3 workflows 的 SA-GHMAPPO 配对中，`typed_full - no_prediction` 为：

| 指标 | mean delta | window wins / ties / losses |
|---|---:|---:|
| ready / continuity | +0.003268 | 2 / 10 / 0 |
| joint hit | +0.002614 | 2 / 10 / 0 |
| handoff-ready ratio | +0.033333 | 2 / 10 / 0 |
| handoff-failure rate | -0.033333 raw，候选有利 | 2 / 10 / 0 |
| transfer MB/request | +3.656616 raw，候选不利 | 0 / 3 / 9 |
| backhaul traffic cost | +34.266667 raw，候选不利 | 0 / 3 / 9 |
| total reward | +1.051500 | 9 / 3 / 0 |
| episode success / completed delay | 0 / 0 | 全 ties |

这支持“prediction-related mechanism 以额外 transfer/backhaul 换取稀疏的 ready/handoff 改善”的探索性解释，
不支持 Pareto dominance。reward 在 9/12 窗口改善，但 primary service endpoint 只在 2/12 窗口改善，说明 reward
优势不能代替机制 endpoint，应单独审查 reward component alignment。

Prediction-boundary support 的 G12 supervised predictor 均为 disabled，
`causal_predictor_snapshot_accepted_step_count=0`。这些结果只能评价 baseline predictor 条件，不能支撑
“supervised causal predictor 已验证”。五个 reactive eviction policies 的 primary endpoints 在全部 540 个配对单元
完全相同；内部 eviction/churn 仅少量行不同，故 eviction-policy 性能差异当前未兑现到主 endpoint。

## 6. Oracle 与 scalability 边界

3 个 scalability cell 均对 H=1/3/6/12 返回 `optimal`，只说明相应具体 replay 的 exact rolling enumeration
正常结束。每个 replay 只有 9 个请求；访问状态为 20、36、54、63。三档 state limit 是 1,000 / 10,000 /
100,000，但不是实际访问规模。所有 `baseline_oracle_gap.json` 均因
`observed_baseline_outcome_not_provided` 而 unavailable，`formal_benchmark_executed=false`。

安全表述仅限“这些小型具体求解单元达到 solver optimal status”。禁止表述“接近 oracle”“oracle gap 已验证”
或“大规模 exact solving 已证明”。

## Claim map freeze

### Safe claims

- 不含 holdout 的 formal execution 完整性已通过，且早于 2026-09-25 完成。
- 在 formal 12-window 汇总中，SA-GHMAPPO 的 ready/continuity point estimate 略高，但增益集中在 2 个窗口，
  BCa 下界为 0，window-level sign test 不显著。
- SA-GHMAPPO 相对 12/14 baselines 的 transfer signed CI 完全位于候选不利方向；相对 PPO、Dueling-DQN 跨 0。
- typed-full/no-prediction 消融呈现明确的 readiness/handoff 与 transfer/backhaul trade-off。
- completed-workflow delay 在共同完成子集无差异；大多数 workflow 对该 endpoint 不可用。

### Prohibited claims

- “SA-GHMAPPO 全面优于所有 baseline”或“12 项算法优势”。
- “负向 transfer signed delta 表示节省传输”。
- “现有 Holm sign test 已按窗口完成多重校正显著性验证”。
- “端到端 workflow delay 更低”或把 null workflow 当作零延迟。
- “三容量均证明扩展收益”或把 576/864 当独立重复样本。
- “完整 typed semantics、base sharing、state migration 消融已闭合”。
- “G12 supervised predictor 已被 support 验证”。
- “接近 oracle”或“大规模 oracle 求解能力已验证”。

## Holdout 主张与分析规则冻结

在打开 holdout 前冻结以下规则，不因 holdout 结果调整：

1. 候选、checkpoint manifest、三容量、14 baselines、6 primary endpoints 和 84 项 Holm family 保持不变。
2. 外层统计单位是原始 non-overlapping window；seed、workflow、capacity 只在窗口内聚合，不增加独立 n。
3. sign test 必须基于每个 outer window 的 paired mean，而非行级 wins/losses；Holm 使用窗口级 p-value。
4. claim map 直接消费 `signed_positive_favors_candidate=true` 的 CI，不再按 lower-is-better 二次翻转。
5. 三容量分别报告；576/864 若继续相同，必须如实报告 saturation / non-discrimination。
6. delay 仅作为 completed-workflow conditional endpoint，同时报告 availability、completion/continuity 和 censoring；
   不做零值或 reward 插补。
7. typed-full/no-prediction、prediction-boundary、sensitivity 和 oracle 均为预注册 exploratory/support 分析；
   不按 holdout 效果选择子组或设置。
8. confirmatory 论文主张冻结为 trade-off，而非 dominance：检验稀疏 readiness/continuity gain 是否可复现，
   同时完整报告 transfer/backhaul cost；任何一侧不得隐去。
9. holdout 一次开启后永久 consumed；不得因结果不理想调参、换 checkpoint、改窗口或改 family。

当前 G14E07 runner 无 holdout capability，也没有 holdout command。启动 holdout 前必须另立、审查并批准独立
holdout execution contract、append-only opening record、candidate/checkpoint hash 绑定、原始 formal/holdout interval
互斥复核和预计耗时。formal cache-policy + controller 历史 wall-clock 约 5 小时 12 分，统计/gate 另需约 15 分钟；
这只能作为同规模 workload 的粗略下界，不包含新合同实现、审查、失败恢复或 support 复跑，不能据此承诺日期。

## Hard blockers

- holdout 未开启且当前 entrypoint 明确无 holdout 执行能力；paper-ready 仍为 `Unverifiable`。
- claim-map signed direction bug 和 row-level sign-test pseudoreplication 未修复；现有 supported/Holm 结论不可引用。
- delay endpoint survivor-conditioned 且无策略区分力。
- 4/6 typed-semantics ablations unavailable；多个机制 claim 缺独立消融。

## Major concerns

- 主 endpoint 增益集中在 2/12 窗口；transfer/backhaul 代价覆盖 9/12 窗口。
- 576/864 primary outcomes 完全相同，capacity discrimination 有限。
- reactive eviction policies 的 primary outcomes 完全相同，eviction-policy claim 缺效果层证据。
- G12 supervised predictor 未启用；oracle gap unavailable；数据仍是 NGSIM + Alibaba + controlled typed catalog。
- 本轮未独立重算 manifest 内全部 29,909 个文件，也未做 E3 独立重建。

## Minor concerns

- `evaluation_only_state.json` 是初始化快照，保留 `formal_execution_started=false`；最终执行状态应以 append-only
  phase ledger 和 `formal_gate.json` 为准，论文 provenance 需解释状态文件职责，避免表面冲突。
- `formal_performance_evidence` 和 `paper_claims_permitted` 在当前统计/gate 中为 null，不能解释为 true。

## Scorecard

本轮不做伪精确总分。Novelty、完整 holdout 统计、泛化和最终可复现性均缺少完成态证据，记为 `N/S`。
formal-only artifact 足以冻结上述安全/禁止主张，但不足以给出 TMC-ready candidate 分数。

## Required actions before re-review

1. 独立修复并验证 signed claim-map 与 window-level sign test；不得改动 G14E07 原始 artifact。
2. 生成纠正后的只读统计附录，保留原始与修正值的 provenance。
3. 冻结并审查独立 holdout opening/execution contract，先验证 raw interval 互斥和一次性消费记录。
4. holdout 结束后按本文件冻结规则复算，不做结果导向筛选。
5. 若论文保留 typed base sharing、state migration、eviction-policy 或 supervised predictor claim，必须补匹配消融；
   否则收缩 claim。

## TITS/TVT fit note

当前负结果和 trade-off 并不使研究失去价值，但在 holdout、统计修正和机制消融闭合前，不应通过降低目标期刊标准
来绕过证据缺口。TITS/TVT 适配只能在同一证据边界下另行讨论。
