# CSCWD 公开基线公平性与主张变更说明（2026-10-11）

## 审查身份

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（本轮未检索新文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_public_rule_raw_fairness_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: `6bf5d0b0ba9af2460cdc736f14909b0692126a48`
- `evidence_level`: `E2_ARTIFACT_AUDITED / bounded actual-environment contract audit`
- `verdict`: `PUBLIC INPUT FAIRNESS PASS / PERFORMANCE UNTESTED / PAPER-READY UNVERIFIABLE`

本说明只改变 baseline 的能力标签和可用主张，不修改 A 论文主稿，不修改旧实验、checkpoint、reward、mask、环境
转移或训练定义。

## 已确认的能力边界

| 方法 | 信息权限 | 决策目标 | 公平性用途 |
|---|---|---|---|
| `causal_public_immediate_rule` | 仅公共 `semantic_state`、公共 causal prediction、公共 mask | service → deadline → known estimate → readiness gain → time/bytes | 可作为公开一阶规则候选 |
| `causal_public_two_step_rule` | 同上；第二步只用冻结 `next_rsu_sequence` 投影，未知 contact 保持 unknown | 同一公开 lexicographic score 的两步组合 | 可作为公开两步规则候选 |
| `immediate_cost_rule` | `clone_for_decision_model().step()`，可读隐藏实际 contact/transition | completed nodes → failures → deadline → elapsed → bytes | 只能作为 privileged reference |
| `two_step_cost_rule` | 两层 exact clone/step，可读隐藏实际 contact/transition | 同上 | 只能作为 privileged reference |

公开规则与学习方法实现了输入权限匹配，但目标函数不完全相同：规则是显式 lexicographic service/deadline/readiness/cost，
学习方法优化冻结 reward。后续结果必须把这种 objective mismatch 单列，不能把“公共信息相同”写成“优化目标完全相同”。

为保护旧 evidence identity，没有改写 `calibrated_continuous_workflow_env.py` 中历史类的字节或既有协议 hash；新的
`PRIVILEGED_REFERENCE_PROFILES` 是 canonical capability registry。旧类 docstring 中的 “information-matched” 不再可作
论文或报告依据。

## 最小反例与数值核对

审计使用 A 线真实 `RawNGSIMEventTimeEnv._info()` 和同一 `dev_01` 实例，构造两条仅隐藏后缀不同、前两帧完全相同的
合成轨迹：

- 两份完整公共 `_info()`、action mask、PPO 输入三元组完全相同；
- public immediate/two-step 在两份轨迹均选 action 0；选择前后 cache residents、prepared state、clock、step、metrics、
  RNG state 均不变；
- 实际执行没有 mask rewrite：requested/executed action 均为 0；慢后缀完成服务，快后缀因
  `current_rsu_contact_expires_before_commit` 被拒绝，说明规则没有偷读结果；
- 冻结 seed 7、0 update 的 PPO 数值概率在两份输入完全相同：
  `[0.313354, 0.0, 0.3354, 0.351246, 0.0]`，action 0 log-prob 均为 `-1.1604218483`；
- privileged immediate/two-step 在慢/快后缀分别选 0/2；其 action 3 exact preview 分别成功与 contact-expiry，构成权限对照。

PPO 数值核对使用既有冻结解释器 `/Users/howen/Projects/PPO_MEC/.venv/bin/python`，`sys.prefix` 指向同一 `.venv`，
PyTorch=`2.8.0`。未安装依赖、未加载 checkpoint、未更新参数。

## Artifact 与主张变更

- root: `artifacts/analysis/cscwd_public_rule_raw_fairness_20261011_v1/`
- `manifest.json` SHA-256: `21ffc41ac51e16fa58f4e6e5dd0c0ecc5bb9f8bf5f802399499e5230232992b5`
- `audit.json` SHA-256: `45f398ee16708bcae256be7fb8a300d380a9933b839243d69260ab70767e0e7b`
- raw environment source commit: `14fa246d4ece3cb35d5e38100e3024c96e42d2bc`
- training/evaluation/checkpoint selection/raw-data export: `0/0/false/false`

A 论文线可作的变更仅为：

1. 删除或禁止“旧 two-step / immediate 与 learned methods 信息匹配”的表述；
2. 若保留旧规则，名称必须带 `privileged exact-transition reference`；
3. public causal rules 可列为下一轮公平基线候选，但当前没有性能结果，不得写成强基线已通过或 SA 已改进；
4. paper-ready 仍为 `Unverifiable`。

## 启动门

公平性合同门已通过，但 A 线随后完成的固定预算可达性门失败，因此训练门保持关闭：2 个 result-blind 长窗口 × 4 个
workload × 5 个冻结规则方法共 40 episode、320 个真实 step，40/40 workflow 完成但仅 4/40 按期；model/state
transfer bytes 与 migration success 均为 0，12 次实际 action 4 全部因 contact expiry，累计 68 次 contact rejection。
reachability manifest SHA-256=`481dd42489dbb9bab30f297237b4373fee9b1b3823082031ed308532f2332e0c`，step ledger
SHA-256=`679942ac74cc618127142abc89a5eaa0cd109ff38cad213dfd77dd850a67bacb`。这表明当前冻结几何和 full-step contact 下
迁移机制不可达；按停止条件不启动 4 方法 × 5 seed pilot，不换窗口、不改几何、不延长预算。
