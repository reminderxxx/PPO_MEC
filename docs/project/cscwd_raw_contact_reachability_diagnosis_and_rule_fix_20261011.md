# CSCWD Raw contact 可达性诊断与 public rule v2 修复（2026-10-11）

## 审查身份与结论

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（本轮未检索新文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_public_reachability_coverage_20261011_v1` / `cscwd_public_rule_feasibility_fix_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: ledger diagnosis `5c96998a6ce1ddcaf83725462f9b1c0d655ff7fe`；rule v2 fix
  `c9eb64f4f7ada882eb76ca2f0223ea92dec4b054`
- `evidence_level`: `E2_ARTIFACT_AUDITED / bounded ledger + exact public-state regression`
- `verdict`: `RULE CONSUMER DEFECT FIXED / MIGRATION MECHANISM UNREACHABLE / TRAINING STOP`

本轮先完成独立只读原因分类，再按授权只修 public rule consumer。没有修改 Raw 环境、action mask、reward、学习方法、
物理 contact 语义、数据窗口、训练预算或旧 artifact。

## 合并原因分类

### 1. 生产端、mask 与执行动作

- 旧 estimator 已正确生产 `raw_full_step_contact_fit`；prepare auxiliary label 在其为 `no` 时会 abstain。
- public action mask 表示语义动作是否存在，不表示持续 contact 是否足够；Raw 无覆盖时另行收窄为 fallback-only。
- 320 个 ledger step 中，producer 在调用环境前显式把非法固定规则动作投影到合法 fallback，形成 96 个
  requested/executed mismatch；public immediate/two-step 的 mismatch 为 0。
- producer 逐步断言 `transition.action == mask-legalized action`，Raw executor rewrite 为 0。因此 mismatch 不是环境隐藏改写，
  也不是 PPO action-probability 口径错误。

### 2. 可学习覆盖与 action 4 阶段

| 项目 | 结果 |
|---|---:|
| bounded episodes / real steps | 40 / 320 |
| workflow completed / on-time | 40 / 4 |
| contact rejections | 68 |
| action 4 executions / successes | 12 / 0 |
| action 4 planned full-step > public contact | 12 / 12 |
| action 4 planned full-step > actual contact | 12 / 12 |
| migration success / model bytes / state bytes | 0 / 0 / 0 |

12 条 action 4 的 estimator 分类为：current service `yes/no=7/5`，target prepare `yes/unknown=7/5`，
raw full-step fit `no/unknown=6/6`，cost known/unknown=`6/6`，deadline yes/unknown=`6/6`。实际阶段探针显示：

- target model network + load + state network + restore 可在 actual contact 内为 8/12；
- native preview 中完成当前 service 并提交 migration 为 4/12，但完整 step cost 仍跨出 current contact；
- 另 4/12 虽可完成 prepare，但当前 service 失败；其余 4/12 连 prepare 都超出 contact；
- Raw 原子 gate 对 12/12 完整回滚，因此真实 model/state bytes 为 0，符合 success-only commit 合同。

当前 Raw gate 把 model network、model load、current-node compute、prefix recompute、成功后的 state network、state restore
以及适用的 input transfer 全部计入 current-RSU contact。实现与冻结合同一致，但这种物理合同是否符合目标应用尚未建立，
不能因为 4 条 native preview 能内部完成迁移就直接放宽 gate。

### 3. 独立 public rule consumer 缺陷

旧 public immediate/two-step 都用 `_score` 排序，而 `_score` 不消费已经生产的 `raw_full_step_contact_fit`。因此两规则在
同一个 US-101 / `dev_01` / step 0 exact 公共状态上都选择 action 4，尽管 action 4 已明确为 `no`；Raw 随后拒绝。
该缺陷只覆盖 public 规则 126 次决策中的 2 次，修复只能避免已知不可行动作，不能创造任何 migration success。

## 最小修复：`causal_public_rule_selection_v2`

Raw 合同下统一候选分类：

- 非 fallback 动作只有 `raw_full_step_contact_fit=yes` 才进入 scorer；
- `no` 分类为 `excluded_explicit_raw_contact_infeasible`；
- `unknown` 分类为 `excluded_unknown_raw_contact_feasibility`，不自动当作 yes；
- 合法 action 2 分类为 `eligible_contact_independent_fallback`，作为 no/unknown 的明确退路；
- 非 Raw profile 保留历史候选面；two-step 的第一步和投影后第二步使用同一分类。

这不屏蔽 SA 或 PPO 的 action 4，不改变 mask；只纠正两个 deterministic public rules 的消费逻辑。能力 profile 仍为
`public_causal_semantic_state_only_v1`，objective profile 升为
`raw_feasible_then_lexicographic_service_deadline_known_readiness_cost_v2`。

Exact 公共状态 SHA=`049de83dba0218565999c821d7b053391f9ae1f32f45e4c01079358e12028fa8`；外部纯公共 snapshot
SHA=`4036254c97db4bad48936d8b5f8276931ca2985523250a6385e697b1ab8e8f98`。修复前 immediate/two-step 均选 4；
修复后 action 0/1/3/4 全部明确 `no` 并被排除，两规则均选 2，输入 hash 不变，未读取隐藏未来。

## 若后续需要机制可达：仅冻结语义候选，不实现

当前问题不是继续调 PPO，而是明确跨边界执行状态机。最小候选是 phase-split asynchronous prepare：

1. target model transfer/load 是否走 RSU 间 backhaul，是否需要车辆仍在 current-RSU contact，必须显式建模；
2. current node 若在无线离开后仍由原 RSU 完成，需要 `in_flight_service` 状态、结果路由和失败语义；
3. state snapshot 只能在当前 node 成功后形成，再经有容量/时延约束的 RSU 间链路提交到 target；
4. handoff 后 target 仅在 model 与 state 两阶段都 committed 时可复用，否则回退/重算。

应用依据是跨 RSU 长时 AI 服务连续性，而不是为了制造 SA 优势。该方案必须被实际部署假设或公开系统文献支持，并冻结新的
observation/action/transition/resource contract。若应用要求计算全程维持当前无线接触，或没有可用 inter-RSU state path，则该方案
被否定，应如实认定当前任务对 migration 主要不可达，而不是改 reward 或换窗口。

## 验证与证据

- pre-fix ledger diagnosis SHA：`a1d1c6928418771a8ad2a0e6bfbdb041819144e1db6c1e69f83f2a085360a5b5`
- pre-fix manifest SHA：`8e15b2242e47e9b782efceb1f9a0a8927c76267a97b07ca413772ad9e1509c6d`
- post-fix exact-state audit SHA：`dd49f9442d857373f780455fb99c97e74fc09f8d49039dc975b61d68ed4223b0`
- post-fix manifest SHA：`eeb3d8f6c291ada3590d373ea45d6ee26c80884daaa4ed43828211ab54e721ea`
- A action4 phase audit SHA：`b3f0634ad0c305fdd9f0cb5780c3b0120c94aa8b4e3600ce175f5fd601d44554`
- tests：43 passed；smoke：6/6 toy DAG nodes completed
- training / checkpoint selection / new policy evaluation：`0 / false / 0`

训练门继续为 STOP；候选仍为 `UNTESTED`，paper-ready 仍为 `Unverifiable`。
