# Frozen v2 评价、失败归因与 Service-Readiness Guard v3

## 审查元数据

| 字段 | 值 |
|---|---|
| `reviewed_at` | `2026-09-28T11:14:07+08:00` |
| `literature_cutoff` | `2026-09-28` |
| `target_venue` | `IEEE Transactions on Mobile Computing (TMC)`；实际投稿目录未指定 |
| `artifact_run_id` | `mechanism_algorithm_retraining_v2_freeze_20260928`；`mechanism_frozen_evaluation_v1_20260928_v2`；`mechanism_training_failure_diagnosis_v1_20260928`；v3 训练根为 `mechanism_algorithm_retraining_v3_20260928` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| `implementation_git_commit` | `0d461befb3f2175bcc20da7bae27c4ec0143f77a` |
| `evidence_level` | `E2_ARTIFACT_AUDITED_OBSERVED_DATA_PILOT_NOT_HOLDOUT` |

## v2 冻结基准

v2 训练 `completion_receipt.json` 为 `SUCCEEDED`，两个 child 均返回 0。冻结不根据已观察表现从 4 个 candidate
中择优，而是对两个算法共同使用固定预算终点 `update_0016.pt`。该规则是在训练统计已被查看之后才确定，明确不是
预注册。冻结脚本只读取 checkpoint，加载前后文件 SHA-256 不变，不覆盖或复制原文件。

| agent | checkpoint | size | file SHA-256 | tensor-state SHA-256 |
|---|---|---:|---|---|
| SA-GHMAPPO | `update_0016.pt` | 2,090,999 | `fb8f3d141f0de6bccff96474cbbc063b3dae0b21a3a59f907e53ce73e77e4b99` | `2978048c2a316f7bdd02187e0eb9fc55f410e3e3f3cb75ec51f8a8619b895da7` |
| MAPPO | `update_0016.pt` | 537,655 | `e569ca52ee9555a098a2ebb5891cd1a5d958250c261bc41f096284418f3c8835` | `ba1596de93911456faeac00993a15ae7f4c5f2672ca3026122678aa330c9240a` |

freeze manifest SHA-256 为 `0b5babad01e8be10c161e8cb22a983ff68936f5bf5e34343e4bb1dde2e94ab84`，
同时绑定训练 commit/tree、配置、预算、Python identity、数据源、window plan、训练 CSV/summary 和 12 个 matched
request-exposure fingerprints。`latest.pt` 的文件 SHA 与 `update_0016.pt` 不同，因此冻结消费端只能使用明确的
`update_0016.pt` 路径和 SHA，不能回退到 `latest.pt`。

## 固定评价关系与执行勘误

评价计划在执行前冻结于 `configs/experiment/mechanism_frozen_evaluation_v1.yaml`，控制器为两个 frozen learned
checkpoint、popularity、reactive greedy 和 handoff-first feasibility。五者使用同一 ON/ON 机制、相同 semantic
state、action mask、资源状态和 request exposure。handoff-first 只是可完成性诊断规则，不是 paper baseline。

评价复用了训练时相同的 3 个 NGSIM 窗口和 4 个 Alibaba workflow，是 resubstitution pilot；没有 train/dev/test
划分，也不是独立评价或 holdout。旧 consumed holdout 未打开。

首次命令在 import 阶段因缺仓库根路径失败，0 episode；第二次在完成 SA 的 12 个 episode 后发现 evaluator 把含
controller 名称的 identity-bearing fingerprint 当成 controller-neutral workload fingerprint，partial root 已标记无效，
且结果未读取或用于修正选择。最小修复把 `(window, workflow)` evaluation unit 设为 controller-neutral；新 root
完成 60 个 episode，并验证每个 matched unit 的 request exposure 一致。两次失败均完整保留，未删除负结果。

## Frozen v2 多目标评价

canonical artifact 为 `artifacts/analysis/mechanism_frozen_evaluation_v1_20260928_v2/`。每个控制器只有 12 个
训练内单元，因此以下只是描述性 point estimates。

| controller | completion | continuity / ready | handoff failure | MB/request | backhaul | migration cost | delay coverage | conditional delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 2/12 | 0.748529 | 0.833333 | 31.379630 | 280.833333 | 1.250 | 2/12 | 1601.0 |
| MAPPO | 5/12 | 0.788617 | 0.000000 | 41.741503 | 324.833333 | 0.225 | 5/12 | 1061.0 |
| popularity | 4/12 | 0.778431 | 0.625000 | 33.133224 | 294.166667 | 0.840 | 4/12 | 1251.0 |
| reactive greedy | 2/12 | 0.748529 | 0.833333 | 26.006645 | 235.833333 | 1.250 | 2/12 | 1601.0 |
| handoff-first feasibility | 3/12 | 0.683987 | 0.458333 | 34.032353 | 293.833333 | 0.635 | 3/12 | 1367.666667 |

没有 controller 在全部目标上占优：MAPPO 的 completion、continuity 和 handoff failure 优于 frozen SA，但付出更高
transfer/backhaul；reactive greedy 成本最低但 completion 低。`continuity` 与 `full_service_ready_request_rate` 在本表
数值完全相同，不得重复计算为两个独立优势。delay 只在完成 workflow 上可用，覆盖仅 2–5/12，不能脱离 coverage
比较。没有外部成本偏好，因此不定义事后 weighted reward、容忍阈值或综合排名。

## First-order 失败归因

诊断直接读取 128 个训练 episode 的 request/event trace：

| source | request failures | adapter miss | base miss | handoff unprepared | state not ready | capacity rejection | invalid/precondition |
|---|---:|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 225 | 216 | 45 | 12 | 11 | 0 | 0 |
| MAPPO | 273 | 272 | 48 | 2 | 2 | 0 | 0 |

事件类别可重叠，但一阶主因明确是当前服务路径缺 required adapter/dependency，而不是容量拒绝或非法动作。SA 的
225 个失败动作分布为 action 0/1/2/3/4=`20/7/61/78/59`；MAPPO 为 `11/3/82/95/82`。大量 action 2/3/4
在当前 adapter 未就绪时继续执行或迁移，导致立即失败。

所有 episode 的 external denominator、request trace 和 exposure count 一致。外生 replay 在首次 request 失败后仍
继续暴露固定后续请求：SA 有 58/64、MAPPO 有 61/64 episode 如此。这些后续请求仍属于同一 workflow 分母，不是
新的成功/失败 workflow 样本。固定 64-episode budget 不能证明训练收敛或训练不足；该项保持不可识别。

## 唯一 v3 修正候选

唯一修正为 `mechanism_cache_readiness_guard_current_only_v1`：当当前 associated RSU 缺 required adapter 时，
把 controller heads 投影为 current-RSU cache fill，再允许后续执行；当前 adapter 已就绪时不改变动作。该 guard：

- 对 SA-GHMAPPO 与 MAPPO 对称启用；
- 只读取两者已有的当前 semantic state，不增加 future/oracle 信息；
- 不启用 target-prefetch 分支，不修改 reward、环境、action schema、request exposure、窗口或 workload；
- v2 checkpoint 和 artifact 保持冻结，新 v3 从头训练，不 warm-start；
- 预算、seed、3 windows、4 workflows、ON/ON runtime 与 v2 完全一致，总上限仍为 128 episodes / 2,560 steps。

两算法各 2-episode 集成 smoke 已通过并产生 checkpoint；guard 对两者在相同两个 episode 中均触发 3 次和 4 次，
证明配置被对称消费。v3 只能作为 observed-data paired development experiment；即使结果变好，也不是独立泛化、
算法优越或 paper-ready 证据。

## Claim boundary

安全表述：v2 已按共同 fixed-budget endpoint 冻结；一次训练内评价显示明显多目标 trade-off；训练事件把低 completion
的一阶根因定位为 adapter/dependency readiness；据此定义了一个对称、current-only 的最小 guard 候选。

禁止表述：v2 或 v3 优于全部 baseline、独立 test/holdout、统计显著、收敛、全方位优势、delay 普遍改善，或把
resubstitution 结果扩展到其他 mobility/workflow/seed。v3 完成后必须完整报告正负结果，不因表现不佳追加调参。
