# CSCWD action mask 与公共估计器—执行端一致性审查（2026-10-11）

> 2026-10-11 后续实现：本报告定位的 phase-accounting 缺陷已由
> `cscwd_public_estimator_phase_accounting_fix_20261011.md` 在 commit `ce20727e…` 最小修复，并在 `3afeeac…` 修正实际 raw
> `time_profile` 接线后通过 canonical v2 conformance；
> 本报告的 v4 历史账本和“候选未测试”边界不变。

## 审查身份与结论

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（未检索新文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_raw_ngsim_event_time_20261010_v4`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: raw executor `0704937742ec9218094f380a1b541c4bd2159bcb`；public estimator
  `5f88785ae4c2951e3826a468d3c17e389365f39c`
- `evidence_level`: paper claim 为 `E1_DOCUMENTED / Unverifiable`；本轮源码、v4 JSON/hash 与最小反例为本地 contract evidence，
  没有训练、checkpoint、formal/holdout 或算法效果证据
- `verdict`: `ESTIMATOR_PHASE_ACCOUNTING_DEFECT / CANDIDATE_UNTESTED / TRAINING_UNAUTHORIZED`

最优先问题不是“窗口内没有 prepare，所以算法无效”，而是公共 estimator 尚未和 native/raw executor 的**条件阶段成本与
contact admission** fail-closed 对齐。三个 2.2 s 窗口本身不足以完成 workflow；fixed-action 的 mask 改写又使请求动作不等于
执行动作。这些事实只能说明本轮无执行资格，不能否定学习候选，也不能直接推出扩窗。

## v4 身份与真实消费

v4 `source_manifest.json` SHA-256=
`c0b64fd64c602ef5351c5686d2f315bbd5784e63ebbd08902b297fc5bacc883f`；`summary.json` SHA-256=
`62c1f4c2c7fbb2e2763f6946c0fad29b4cd6a3705ebf01501a7d905d8b1fe30f`。入口先核验父 config/manifest 和 raw CSV
身份，再只读扫描冻结三窗口；命令为：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/audit_cscwd_raw_ngsim_event_time.py \
  --raw-csv-path '/Users/howen/Projects/PPO_MEC/data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv' \
  --output-root artifacts/analysis/cscwd_raw_ngsim_event_time_20261010_v4
```

这是固定动作接口探针，不含 learned policy、PPO minibatch 或 logprob；`logprob identity=N/A`。真实 learned consumer 的 masked
distribution 合同已有相邻测试，但 v4 没有 learned rollout，不能用 fixed-action 改写推断策略概率或更新方向。

补充只读根因账本为
`artifacts/analysis/cscwd_raw_time_root_cause_20261011_v2/diagnosis.json`，SHA-256=
`2aba8413253a1a0be1868b62c639f9f7754f76293b4c78aed5109444ad6756fc`。它重放 v4 已存的 `30 episodes/147 real
steps`，给 raw 子集形成 30 条逐决策 ledger，并执行 56 个无策略更新 native preview；`summary_mismatches=0`、
`additional_policy_episodes=0`。它没有扩窗、换 workload 或生成新 learned 轨迹。

## 30 个 raw 决策的 v4 账本

每行是一个冻结实例×请求动作，两次决策；v4 只保存该二步 episode 的 executed counts，而不保存逐步 mask/reason 顺序。

| design | requested | decisions | mask rewrite | executed counts | rejection counts | service | migration |
|---|---:|---:|---:|---|---|---:|---:|
| `dev_01` | 0 | 2 | 0 | `0:2` | contact 1, trace-end 1 | 0 | 0 |
| `dev_01` | 1 | 2 | 1 | `1:1,2:1` | trace-end 2 | 0 | 0 |
| `dev_01` | 2 | 2 | 0 | `2:2` | trace-end 2 | 0 | 0 |
| `dev_01` | 3 | 2 | 0 | `3:2` | contact 1, trace-end 1 | 0 | 0 |
| `dev_01` | 4 | 2 | 1 | `4:1,2:1` | trace-end 2 | 0 | 0 |
| `regression_00` | 0 | 2 | 0 | `0:2` | trace-end 1 | 1 | 0 |
| `regression_00` | 1 | 2 | 2 | `2:2` | trace-end 2 | 0 | 0 |
| `regression_00` | 2 | 2 | 0 | `2:2` | trace-end 2 | 0 | 0 |
| `regression_00` | 3 | 2 | 0 | `3:2` | trace-end 1 | 1 | 0 |
| `regression_00` | 4 | 2 | 2 | `2:2` | trace-end 2 | 0 | 0 |
| `regression_05` | 0 | 2 | 1 | `0:1,2:1` | trace-end 2 | 0 | 0 |
| `regression_05` | 1 | 2 | 1 | `1:1,2:1` | trace-end 2 | 0 | 0 |
| `regression_05` | 2 | 2 | 0 | `2:2` | trace-end 2 | 0 | 0 |
| `regression_05` | 3 | 2 | 1 | `3:1,2:1` | contact 1, trace-end 1 | 0 | 0 |
| `regression_05` | 4 | 2 | 1 | `4:1,2:1` | contact 1, trace-end 1 | 0 | 0 |

合计：requested 每动作各 6 次；executed 为 action0/1/2/3/4=`5/2/16/5/2`。action2 的 16 次中只有 6 次是请求，另 10 次
来自 mask fallback。请求动作的 rewrite 数为 action0/1/2/3/4=`1/4/0/1/4`。按 v4 优先级标签，28 次拒绝为
trace-end 24 与 current-contact 4；其中 8 次同时超过两者，若按最早物理边界分解则为 trace-end 20、contact-end 8、
accepted 2。没有 deadline、model-missing、dependency rejection 类别。后两类可能影响 native phase，但 outer raw gate
先拒绝时不会进入相应提交结果，v4 不能据零计数断言它们不存在。

`RawNGSIMEventTimeEnv` 在 invalid request 时统一改为 action2；这是 fixed probe 的安全回退，不是 learned policy 选择。
`ActionMaskBuilder` 本身只按当前 node、distinct predicted target 和 target adapter readiness 屏蔽 1/4；current bundle missing、
deadline、contact 和 dependency 不会统一屏蔽 action4。raw profile 只有无当前覆盖时额外变为 `[0,0,1,0,0]`。所以没有证据表明
“有可行动作却被统一算法 guard 禁止”；但 v4 未保存逐步 `invalid_reasons`，不能从汇总反推每次 rewrite 的精确原因。

## action 4 语义：当前服务是合同的一部分

源码与最小执行见证否定“native action4 仅准备 target”的假设：action4 先在目标 RSU admission model bundle，并暂存 state
package；随后和 action1/3 一样尝试在当前 RSU 完成当前 node。只有 current service 成功，才增加 `completed_nodes`、提交
state bytes/restore、写入 `prepared_state` 并令 `migration_success=true`。current bundle missing 时，目标 model cache 仍可能
提交，但当前 node 失败、state transfer 为 0、migration 不成功。

因此 `causal_public_prepare_advantage_v1` 要求 action4 current service 可行这一方向是正确的；问题在成本阶段如何条件聚合，
不是把该条件删除。

## estimator 与 executor 逐项一致性

| 项目 | 状态 | 证据与边界 |
|---|---|---|
| byte / Mbps 单位 | 一致 | 双方均以 bytes，网络时间为 `bytes*8/(Mbps*1e6)+fixed_seconds` |
| typed bundle / prepared prefix | 基本一致 | estimator 只读 public residents/catalog/prefix；需要私有 LRU `last_used` 时 admission 返回 unknown |
| action mask | 一致消费 | estimator 接收同一五位 mask；masked action 不生成候选标签 |
| action4 当前服务 | 一致 | native 确实完成当前 node 后才 commit state；estimator 检查 current bundle |
| 实际未来 route/contact | estimator 不读取 | estimator 不 clone/step/读 instance；future-field 扰动测试通过 |
| prediction | 有风险 | v4 target 为 prefix constant-velocity，confidence=`0.5` 且明确未校准；estimator 不使用 provenance/confidence，不能把 target 当 ground truth |
| unknown phase 聚合 | **缺陷** | target eviction/prepare 为 unknown 时，`estimated_total_seconds` 仍可能排除该阶段后给出有限值和 deadline fit |
| current-service failure | **缺陷** | estimator 可计入 action4 state bytes/restore；native 失败路径只保留已提交 model phase，state bytes/restore 为 0 |
| contact gate | **错配** | estimator 只用 contact 判断 target prepare；raw executor 对整个 planned step（含当前 compute/recompute）施加 current-contact gate |
| trace end | 有意不可见 | raw executor 用私有 source endpoint 做物理准入；公共 estimator 不应读取，故不能精确预测 24 次 trace-end rejection |
| decision clone | 权限风险 | public rules 不 clone；raw exact clone 若走 `_physical_contact_budget_seconds()` 会读取真实未来轨迹，只能列 privileged reference |

最小数值反例：目标 eviction 顺序不可公开识别时，action4 `target_prepare=unknown`，但 estimator 仍返回 `3.798119 s`；同状态
native step 实际为 `4.208268112 s`。current bundle missing 的合成状态中，estimator 报 state transfer `2,000,000 bytes` 和
总时长 `2.81 s`；native 合同在当前服务失败时提交 state bytes=`0`。这些是 estimator contract 缺陷，但没有参与 v4 fixed
policy 执行，因此不是 28 次 raw rejection 的成因。

另一个直接执行反例把“准备阶段”和“整步原子提交”分开：`dev_01` 首决策的实际 current contact 为 `0.728088 s`，action1/
action4 的 prepare 分别为 `0.694262/0.724293 s`，均可在接触内完成；但加上当前 node compute 后整步为
`2.217156/2.247187 s`，被 raw gate 拒绝。`regression_05` action1 的前瞻准备为 `0.385720 s < 0.390382 s`
current contact，但当前服务失败使整步为 `2.385720 s`，raw profile 仍回滚。故“0 成功 prepare”不是 public estimator
候选的反例，而是当前 raw atomic-admission 语义没有给准备阶段独立提交机会。

## 原因优先级与唯一下一改动

1. **输入时长不足（已证 execution blocker）**：首决策后每窗仅 2.2 s；三个 DAG 仅 compute 总和下界分别为
   `38.739782/24.968087/19.138455 s`，无需策略或传输成本就已超过窗口。它不否定算法。
2. **保守 atomic admission（已证行为主因）**：v4 优先标签是 trace 24/contact 4，按最早物理边界则是 trace 20/contact 8；
   整步 gate 会回滚接触内本可完成的 prepare，不支持部分提交或跨界继续服务。这是明确的环境能力边界，不是 estimator 造成。
3. **mask/固定探针混杂（已证）**：10/30 请求被改写为 fallback；v4 fixed rows 不能当作原 requested action 的结果或 learned
   policy 行为。没有 PPO logprob 可核对。
4. **公共 estimator 错配（已证实现缺陷，但未作用于 v4）**：unknown phase 被漏出有限 partial total、失败路径 state phase
   条件错误、raw total-contact criterion 未表示、未校准 target 未进入 uncertainty 语义。

另有独立权限风险：合成 suffix 反例保持公开 state 与首两帧相同，仅改变未来轨迹，
`clone_for_decision_model().step(3)` 一侧成功、另一侧以 current contact 拒绝；`_decision_model_mode` 没有阻止 raw
`_physical_contact_budget_seconds()` 读取未来后缀。public estimator/rule 不走该路径，因此不影响上面的候选，但 raw exact clone
只能继续作为 privileged reference，不能与 public rule 做同权限排名。

下一实施轮只改一个变量：**把 public estimator 改为 fail-closed phase accounting contract**。每个 action 显式列出实际会执行/
提交的阶段；任何必需阶段 unknown 时 full total/deadline/contact fit 均为 unknown；action4 state phase 仅在 current service 成功时
计入，并以 full coupled step 检查公开 current-contact fit。先做 estimator↔native 合成 conformance，不改 raw 环境、窗口、标签
阈值、reward 或训练。该修复通过后再独立决定数据资格；当前条件训练授权继续无效。

## 验证与未覆盖

- `tests/test_cscwd_public_estimator_executor_diagnosis.py`：4 个最小见证通过。
- estimator/candidate 相邻测试合计：`13 passed`；上游 raw 账本与环境相邻测试另为 `7 passed`。
- 未扫描冻结区间之外的 raw 数据；未新增策略 episode、训练 step、optimizer step、checkpoint 或 formal/holdout 消费。
- v4 没有逐步 mask invalid reason、public estimator label 或 policy logprob；这些字段仍不可恢复，不以推测补位。
- 算法效果、SA 稳定领先、真实无线外推与 paper-ready 均为 `Unverifiable`。
