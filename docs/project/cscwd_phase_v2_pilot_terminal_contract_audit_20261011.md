# CSCWD phase-v2 pilot 终态合同独立审查（2026-10-11）

## 审查身份与两层结论

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（本轮未检索文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_phase_v2_causal_public_pilot_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `executor_git_commit`: `5a79ffd018d4c260bc7a8417229a440c7d591fe4`
- `evidence_level`: `E2_COMPLETE_DEVELOPMENT_ARTIFACT / two exposed development traces; no independent test`
- `scientific_artifact_verdict`: `VERIFIED COMPLETE DEVELOPMENT ARTIFACT`
- `original_supervisor_verdict`: `FAIL / CHILD_EXECUTION_FAILURE_NO_RETRY`（原件保留，不覆盖）
- `algorithm_verdict`: `MIXED / no promotion / paper-ready=false`

这三个结论不矛盾：scientific child 的训练、评价和分析原件完整；原 supervisor 的 receipt schema 合同未满足；算法结果按预注册
门为 `MIXED`。本审查只读，不改旧 receipt/manifest/integrity，不补造旧字段，不重跑训练。

## 终态矛盾的精确根因

原 `completion_receipt.json` 含 `status=complete`、完整预算计数和 commit，但没有
`scientific_execution_complete` 字段。共享 supervisor 在
`scripts/launch_calibrated_workflow_value_normalization_ab.py:391-396` 将 scientific complete 定义为：

```text
bool(child_receipt.get("scientific_execution_complete")
     and child_receipt.get("status") == "complete")
```

因此缺字段被严格解释为 False。虽然 exit sidecar 的 `return_code=0`、`expected_exit_code=0`、`timed_out=false`，outer receipt
仍按既有代码写成 `status=FAIL`、`classification=CHILD_EXECUTION_FAILURE_NO_RETRY`。这是 child receipt producer 与公共
supervisor consumer 的 schema 不匹配，不是训练进程异常、timeout、partial output、optimizer 中断或分析失败。

原 terminal receipt SHA-256=`2fbf478e3286cc6f8bf8fc031f81ca17904c19f23df60226329f3765d3879a1e`；
supervisor integrity SHA-256=`d5a25f95e6796a6376b69638dd38b12ed4dfde19002a8eda45615631a378cedd`。两者不得改写。

## 科学原件完整性

| 项目 | 预注册 | 只读复核 | 结论 |
|---|---:|---:|---|
| learned cells | 20 | 20 | VERIFIED |
| environment steps | 23,040 | 23,040 | VERIFIED |
| optimizer steps | 3,840 | 3,840 | VERIFIED |
| checkpoint-selection episodes | 80 | 80 | VERIFIED |
| learned final evaluation rows | 320 | 320 | VERIFIED |
| rule evaluation rows | 18 | 18 | VERIFIED |
| update rows | 480 | 480 | VERIFIED |
| training signal rows | 23,040 | 23,040 | VERIFIED |
| checkpoint files | 100 | 100 | VERIFIED |
| automatic retry | false | false | VERIFIED |
| formal/holdout reads | 0 | 0 | VERIFIED |

- run integrity：113/113 文件逐个重算 SHA-256，0 mismatch；run integrity SHA-256=
  `1ecea21b8daddbd79c908edf40a5adfbe38b8e411e9e30ffbc48c8974b95acb2`。
- analysis integrity：3/3 文件逐个重算，0 mismatch；analysis integrity SHA-256=
  `063431ff08c1f33b84aa343e73dc821ef825038828bee86d76d9693be77d03b7`。
- supervisor integrity：11/11 文件逐个重算，0 mismatch；child/supervisor stderr 均为空。
- 进程已退出；dispatch/ACK/exit/terminal 只有一个 create-only supervisor root，`automatic_retry=false`，没有第二次 scientific launch。

## Checkpoint 与选择身份

- 20 cells 各有 updates `6/12/18/24` 四个候选和一个 selected，共 80 candidate rows、100 个 checkpoint 文件。
- 80/80 candidate 文件 hash、20/20 selected hash、20/20 fixed-update24 hash 与记录一致。
- 逐 cell 重算 lexicographic score 后，20/20 `selected_update` 一致；selected checkpoint 与获选 candidate 的
  `network_state_dict` 20/20 张量完全一致。
- SA control/candidate 仅候选 flag 不同：candidate 5/5 checkpoint 记录
  `causal_public_prepare_advantage_v3`，SA control、PPO、MAPPO 共 15/15 保持 `legacy_target_timing_v1`。
- selected updates：SA control/candidate/MAPPO 均为 5/5 update6；PPO 为 2 个 update6、3 个 update12。固定 update24 独立保留，
  最终评价没有参与选模。

## 结果与 reward 分项边界

320 learned rows 的 episode reward 与六项分量之和最大残差 `1.78e-15`，0 行超过 `1e-8`。selected 视角 SA mean：

| arm | reward | node | workflow | time | transfer | failure | deadline |
|---|---:|---:|---:|---:|---:|---:|---:|
| SA control | 10.142597 | 16.0 | 4.0 | -7.477813 | -0.004590 | -1.875 | -0.5 |
| SA candidate | 10.236331 | 16.0 | 4.0 | -7.384042 | -0.004626 | -1.875 | -0.5 |

但预注册 primary service 指标在 selected 与 fixed-update24 两视角完全相同：completion `1.0→1.0`、on-time
`0.875→0.875`、failure episode rate `0.25→0.25`。candidate transfer `+0.077264 MB`、recompute
`-1.186849 s`；5/5 seed 均为 service-neutral。故 `strict_service_improvement=false`、favorable/unfavorable=`0/0`，结论只能是
`MIXED`，不能因 reward `+0.093734` 或 recompute 下降而晋级。

## 可用性与未验证项

- 可用：本 run 可作为完整、hash-verified 的 development artifact，并支持“该唯一 auxiliary 候选未产生严格服务改善”的
  描述性结论。
- 不可用：outer supervisor receipt 不能写成 PASS；必须保留原 `FAIL` 并单列 schema mismatch。
- 未验证：独立 test、formal/holdout、统计显著性、跨来源泛化、paper-ready、SA 稳定领先均未建立。
- 修复边界：后续只允许为 child producer 与 supervisor consumer 定义显式 receipt schema，并用合成 fixtures 验证；不得据此
  修改本次原件、重跑训练或改变 `MIXED` 裁决。
