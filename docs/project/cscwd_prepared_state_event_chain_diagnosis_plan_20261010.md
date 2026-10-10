# Prepared-state v4 预测→准备→复用链：事前只读诊断规则

- 冻结日期：2026-10-10（Asia/Shanghai）。来源只读 run：`/Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/artifacts/experiments/cscwd_causal_prepared_state_visibility_matched_20261010_v1`，科学 commit `f46ec72b15f534ac44768a83ef6316c1cfcb6b58`。必须先校验 terminal PASS、completion、run manifest 和实际读取文件的 SHA/size。A 代码工作树与 B 原件均不作为写入目标。
- 范围：new selected 与 new update96 各 400 episode，共最多 800 已记录动作轨迹，仅重放一次；与原 evaluation/ledger 每一步及 episode 指标核对。历史 selected/update96 只作语义或配对参考，不重评、不读旧 holdout。不得把 selected 与 fixed96 重复 checkpoint 或同窗口多 seed 当独立样本。
- 每 episode 记录首次 `service_completed=false` 决策、失败次数、action、current bundle、target forecast、prepare feasibility、目标 admission/驱逐、model/state bytes、`migration_success`、是否最终完成；`service_failure_rate` 只视为 episode 指示，`service_failures` 才是次数。正向样本完整保留。
- 每次执行 handoff 建链：到达前最后一次同源 RSU 的公开预测及 contact、current/target bundle、准备动作与模型 admission/状态 commit；中间节点进展和 prepared prefix 有效性；实际到达时 `state_ready`、重算和首次服务。记录 model prepare、state prepare 与 calibrated simulation 的 execution-authority 概念分别；不与 production technical export/import 混写。已记录的 `actual_next_rsu_id`、实际 link 和后来发生的事件只用于审计，绝不喂给在线 policy。

## 多标签事实与互斥主因

先生成所有事实标签；主因只是可复核的**首个链路断点**，不宣称单动作反事实收益。对首次 service failure 依次判：

1. 当前 bundle 缺失且选了不修复当前 bundle 的动作 1/3/4（合法动作不等于高效动作）；细分 `action4_target_prepare` 是否真的改目标 model，当前服务失败、状态未提交。
2. action0 试图填 current bundle，但 admission 因容量/依赖拒绝；若未能判明具体原因记 `admission_unknown`。
3. current bundle 已 ready 却失败，或其他不匹配：`execution_inconsistency_or_unknown`，须逐代码核对，不能默认称 bug。

对到达 handoff 后的 state 未就绪/重算，按时间线上最早可观测断点判：已针对实际到达 RSU 发起 action4 但其 current 服务失败而状态未提交；已发起但目标模型准入或 contact 失败；此前已成功提交状态但节点继续进展使 prefix stale；曾准备其它目标而未准备实际到达目标（forecast mismatch）；尚无针对实际目标的 action4（`not_initiated`，另标有无公开机会）；其他/unknown。若存在多次历史准备，按到达前最后一次相关目标 prepare，保留全部历史计数；若链条信息不足保留 unknown。

同时逐 split/method/seed 报告预测目标正确与不正确、action4 的成功/失败、无准备/有效准备/过期、handoff ready/recompute、service failure 与总完成覆盖。对同公开状态的 SA、PPO、DT 首次动作分叉，只报告分叉时同状态及其后各自完整轨迹；分叉后不作同状态单动作因果比较。正例包括 SA 提前 prepare 后实际复用，负例保留所有方法。

## 有界可选局部分支

只有原轨迹无法判定“在首次失败状态改选 action0/2/4 的净成本与后续复用”时，才对**按 `(split,design_id,seed,checkpoint_view,step_index)` 排序的前 24 个 SA 首次失败状态**做 clone 分支；每状态最多五个合法 action、最多 24 步后缀。继续使用该 episode 已记录的后续动作序列作为显式 open-loop 敏感性，不重新调用 policy，不将实际未来作为决策特征。每分支保留当步恢复、完成/按期、recompute、model/state/input bytes 和后续 prepared reuse 的全路径对照；不同后续状态的 open-loop 结果不当作策略净收益证明。若轨迹足够且不开分支，明确报 `0/24`。

报告先确认 artifact/provenance，再给分母、互斥主因和多标签事实、正负例、代码路径/aux 是否在本 raw-policy 实际生效、候选与反例、无法识别项。无确定实现缺陷时只称策略次优或 unknown；不修改算法、reward、mask、guard、网络、B runner、论文及论文表。
