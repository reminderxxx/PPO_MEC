# CSCWD public estimator fail-closed 阶段核算修复（2026-10-11）

## 身份与裁决

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（本轮未检索新文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_public_estimator_phase_conformance_20261011_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: phase fix `ce20727ee29cfbb02f992204065c5da853fef5ae`；raw profile wiring correction
  `3afeeac678329d4e885954f2cf3d172de1fac8c2`
- `evidence_level`: `E1_DOCUMENTED / synthetic contract evidence`；不是 formal/holdout、算法效果或 paper-ready 证据
- `verdict`: `IMPLEMENTATION_DEFECT_FIXED / SYNTHETIC_CONFORMANCE_PASS / CANDIDATE_UNTESTED`

本轮只修 2026-10-11 独立审查确认的 public estimator 阶段核算缺陷。没有改 raw executor、action mask、reward、网络、
auxiliary loss、数据窗口、训练预算或 checkpoint 选择，也没有读取实际未来 route/contact。

## 修复内容

`causal_public_action_estimator_v2` 将 action4 明确分成三个顺序阶段：

1. 目标 model bundle staging；
2. 当前 node service；
3. 仅在当前 service 成功后提交 state network/restore。

具体合同变化：

- 任一必需阶段不可识别时，`estimated_total_seconds`、`estimated_conditional_seconds` 与 `deadline_fit` fail-closed 为
  `None/unknown`，不再把已知局部和伪装成完整 action cost。
- current service 失败时保留已经提交的 target model staging 和 failure wait，但 `state_transfer_bytes=0`，state network/restore
  不计入总时长；与 native executor 的实际顺序一致。
- 新增 `phase_status`、`estimated_phase_seconds`、`cost_status` 与 `cost_basis`，区分条件估计、unknown 和 online 不可得的实际成本；
  `actual_executed_seconds` 固定为 `None`，防止把 estimator 输出误写成 realized cost。
- 对 raw profile 输出 `target_prepare_contact_fit` 与 `raw_full_step_contact_fit` 两个不同字段。后者只用公共预测的 current-contact
  budget 表达现有 full-step gate；真实 trace 终点继续不可见，`raw_trace_fit=unknown`。public label 在“prepare 可行但公开
  full-step contact 不足”时 abstain，不使用实际未来真值。
- raw 合同以公开 `time_profile=raw_ngsim_event_time_v1` 识别，同时兼容同名 `interface_profile`；保留 prepared-state
  `interface_profile`，不再由审计脚本人工覆盖。A 独立用实际 `RawNGSIMEventTimeEnv._info()` 发现并复现了旧接线缺口，修复后
  跨分支实测 action0/1/3/4 均产生 raw full-step fit、action2 保持 not-applicable、合法动作 trace fit 均为 unknown。
- estimator schema 从 `causal_public_action_estimator_v1` 升为 `v2`；候选的 prepare/serve/abstain 原则不变，但 checkpoint
  语义递增为 `causal_public_prepare_advantage_v2`，旧 v1 候选 checkpoint 必须 fail-closed 拒绝加载。

## 六个合成 conformance 见证

| case | estimator | native / 边界 | 结果 |
|---|---|---|---|
| current ready | model→service→state，`12.610111768 s` | native 同时长、model/state bytes 一致、migration success | PASS |
| current missing | model→2 s failure，state skipped，`10.787563208 s` | native model bytes 保留、state bytes=0、migration false | PASS |
| prepare contact rollback | target prepare=no，model/state bytes=0，仅 current service `3.798119 s` | native rollback 后仍完成当前 node，同成本 | PASS |
| prepare 可行、full-step 不足 | prepare=yes，但 `raw_full_step_contact_fit=no` | native prepare 合同可完成；raw full-step 语义单列为拒绝条件 | PASS |
| private eviction unknown | required target phase unknown，完整 total/deadline/contact 均 unknown | 不使用私有 LRU 顺序补值 | PASS |
| hidden trace end | 只把隐藏 trace remaining 从 100 改为 0 | 两份 estimator 输出完全相同，`raw_trace_fit=unknown` | PASS |

这里的 native 对账使用实际/估计链路速率相同的合成 fixture，只验证阶段顺序与计费；不能写成真实 NGSIM 性能或算法收益。

## Artifact 与验证

- root: `artifacts/analysis/cscwd_public_estimator_phase_conformance_20261011_v2/`
- `manifest.json` SHA-256: `2679d4ddc4d31b8a9a802655d1eea8c2a1310d471cbe66bd3fcaae09c16dc267`
- `conformance.json` SHA-256: `9d5a1dc0c92fcf5e7ed477f6f7495eed496b89ef478b6c8c087f92ed0dcd99f0`
- manifest source commit: `3afeeac678329d4e885954f2cf3d172de1fac8c2`
- counts: 6 synthetic cases；raw rows/training steps/optimizer steps/evaluation episodes 均为 0
- v1 原件保留为接线失败见证：其审计脚本人工把 `interface_profile` 改成 raw，未覆盖实际 raw env 的
  prepared-state interface + raw time-profile 组合，不再作为 canonical conformance。
- 相邻回归：`46 passed`；`scripts/smoke_test.py` 完成 6/6 toy DAG nodes。

## 训练状态与剩余 blocker

条件训练没有启动。A 对 v28 非 sealed train/dev 计划的 40 个既有区间完成结果盲资格检查：40/40 均为 24 帧、2.3 s，
最长授权连续 span 仍为 2.3 s，而最短冻结 DAG compute-only 下界为 16.298588 s，合格区间为 0。因此数据资格门失败，
训练授权不生效；候选保持 `UNTESTED`，不扩预算、换窗口或称算法失败。formal/hidden 只用于 interval metadata 排除交叉，
未读取 sealed 性能或计划内容。独立 eligibility manifest SHA-256=
`e0e3dfdbeb35da5f99a4aa00f985983273efe364c5a014dafd6bd310e3cc5daa`。

raw `clone_for_decision_model().step()` 读取实际未来 contact 的权限缺陷不在本修复范围；exact clone 继续只能列为 privileged
reference。public estimator/rule 不调用 clone/step，本修复没有引入该泄漏。
