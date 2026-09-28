# Frozen v2 评价、失败归因与 Service-Readiness Guard v3

## 审查元数据

| 字段 | 值 |
|---|---|
| `reviewed_at` | `2026-09-28T11:28:08+08:00` |
| `literature_cutoff` | `2026-09-28` |
| `target_venue` | `IEEE Transactions on Mobile Computing (TMC)`；实际投稿目录未指定 |
| `artifact_run_id` | v2 freeze/evaluation/diagnosis；`mechanism_algorithm_retraining_v3_20260928`；`mechanism_algorithm_retraining_v3_freeze_20260928`；`mechanism_frozen_evaluation_v3_20260928`；`mechanism_training_failure_diagnosis_v3_20260928` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| `implementation_git_commit` | guard=`0d461befb3f2175bcc20da7bae27c4ec0143f77a`；frozen plan=`4221903`；MAPPO inference restore=`702f606` |
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

## v3 完成、冻结与唯一一次有效评价

v3 completion receipt 为 `SUCCEEDED`，两算法各完成 64 episodes、16 updates、656 个实际环境步；总计正好
128 episodes。每个算法保存 updates `4/8/12/16`，按事先固定的共同终点冻结 `update_0016.pt`，没有按性能择点。
freeze manifest SHA-256 为
`8c06a75a85030672923aa7fa775a77c214fbbdf9aff60d5ffd900d5e0e7b04ee`。

| agent | checkpoint size | file SHA-256 | tensor-state SHA-256 |
|---|---:|---|---|
| SA-GHMAPPO | 2,091,191 | `67c9c940fffb4c189ab57bc6541cef632831ff18055c73337723fbb40d31ffeb` | `c23b853dbf56f53e833dbb4eb31429a246b2d4df17ff01bcf9a133bbdfabe1bf` |
| MAPPO | 537,783 | `a8d3a9dd2f26a5f3ed9a7eadccb0084d112855c1498d803678b1725ebc82d627` | `2eb4c22bf6dced61b2198fec5e74d8b98d3ed997e614231503cd1f33f5d2497d` |

评价计划先以 commit `4221903` 冻结，保持同一 3 windows × 4 workflows、同一 exposure、指标和分母。计划含
SA/MAPPO guard-on、同 checkpoint guard-off 归因设置及 3 个固定规则，共 7×12=84 episodes。第一次启动在
0 episode 的 guard preflight 阻断：MAPPO 推理 allowlist 漏恢复两个 checkpoint guard 字段。失败根单独保存，
没有读取 performance；`702f606` 只修复推理配置恢复，不改模型参数、计划或数据。随后唯一一次有效执行完成，
四个 learned 设置的实际 guard on/off 状态均通过实例级核验，评价前后 checkpoint SHA 完全相同。
该 guard 位于两个 learned controller 共享的 agent core，并非环境全局系统层；因此对 SA/MAPPO 对称启用，
固定规则保持其原生确定性策略，不额外包裹 learned-controller guard。三类规则与 v2 逐项复现，公平口径未漂移。

| setting | completion | continuity / ready | handoff failure | MB/request | backhaul | migration | delay coverage | conditional delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SA guard-on | 8/12 | 0.879630 | 0.083333 | 38.397930 | 324.500000 | 0.293333 | 8/12 | 1076.0 |
| SA guard-off | 0/12 | 0.372876 | 0.000000 | 26.744880 | 177.000000 | 0.225000 | 0/12 | unavailable |
| MAPPO guard-on | 8/12 | 0.879630 | 0.083333 | 38.397930 | 324.500000 | 0.293333 | 8/12 | 1076.0 |
| MAPPO guard-off | 0/12 | 0.372876 | 0.000000 | 26.744880 | 177.000000 | 0.225000 | 0/12 | unavailable |
| popularity | 4/12 | 0.778431 | 0.625000 | 33.133224 | 294.166667 | 0.840000 | 4/12 | 1251.0 |
| reactive greedy | 2/12 | 0.748529 | 0.833333 | 26.006645 | 235.833333 | 1.250000 | 2/12 | 1601.0 |
| handoff-first | 3/12 | 0.683987 | 0.458333 | 34.032353 | 293.833333 | 0.635000 | 3/12 | 1367.666667 |

三个固定规则与 v2 数值逐项相同，支持本次协议/环境口径未漂移。guard-on 相对同 checkpoint guard-off 的共同
描述性差值为 completion `+8/12`、continuity `+0.506754`、request failures `-67`，代价是
`+11.653050 MB/request`、backhaul `+147.5`，handoff failure 也从 `0` 升至 `0.083333`；不存在全面支配。
guard-off 没有完成 workflow，因此 conditional delay 无共同可比样本。

## v2→v3 分解与动作/信用诊断

- v2→v3 guard-on 是“重新训练 + guard”的总变化，不是纯 guard 因果效应。SA completion/continuity 分别
  `+0.500000/+0.131101`，transfer/backhaul `+7.018300/+43.666667`；MAPPO 分别
  `+0.250000/+0.091013`，transfer/backhaul `-3.343573/-0.333333`。delay 的完成者集合改变，差值只作描述。
- 同一 v3 checkpoint 的 guard-on/off 是本矩阵内较直接的推理期 override 归因。两算法 guard-on 都触发并改变
  `38/123=30.8943%` 的最终动作；训练期均触发 `204/656=31.0976%`，SA/MAPPO 实际改变
  `182/656=27.7439%`、`183/656=27.8963%`。收益明显由共享系统 guard 主导。
- SA 与 MAPPO 在 guard-on 的全部聚合指标、失败数和最终动作计数完全相同，guard-off 也完全相同；所以本矩阵的
  residual algorithm gap 为 0，不支持 SA 优于 MAPPO，也不支持两者机制相同的一般结论。
- guard-off 时两个 deterministic policy 都只输出 action 3/4，计数均为 `86/37`，5-action support 仅覆盖
  2 类，且 action projection、其他 guard 与 final guard delta 均为 0，因此这直接反映其 deterministic policy
  输出；这是动作覆盖塌缩的诊断信号。训练期仍有更广动作覆盖，SA/MAPPO 最终动作分别为
  `0/1/2/3/4=256/3/39/251/107` 与 `252/0/49/230/125`。
- guard 改写后代码会重新计算所执行 head actions 的 log-prob；16 个 update 均有记录，累计
  `env_action_log_prob_missing_count=0`。现有证据不显示 log-prob 缺失/错配，但 SA 的 head-credit disabled、
  MAPPO enabled 仍未产生可分辨 deterministic policy；该现象只诊断，不在本轮修复或调参。

v3 训练失败对账仍以 cache dependency 为主：SA/MAPPO request failures=`95/103`，其中 adapter miss=`85/97`、
base miss=`46/48`、handoff-unprepared=`11/6`、capacity/invalid=`0/0`，denominator mismatch=`0/0`。guard 缓解但
未消除 readiness 失败；首次失败后的外生请求继续属于同一 workflow 分母。

## Claim boundary

安全表述：v2/v3 都按共同 fixed-budget endpoint 冻结；训练内评价显示 current-only guard 在固定矩阵上显著改变
约 31% 动作并改善 completion/continuity，同时增加 transfer/backhaul，且两个 learned controller 结果不可分辨。
训练事件把剩余失败继续定位到 adapter/dependency readiness。

禁止表述：把 v2→v3 总变化称为纯 guard 因果效应，把 guard-on 结果称为 learned algorithm 优势，或宣称独立
test/holdout、统计显著、收敛、全方位优势、delay 普遍改善或跨 mobility/workflow/seed 泛化。本轮到此停止，
不追加训练、调参、seed、窗口或 holdout。
