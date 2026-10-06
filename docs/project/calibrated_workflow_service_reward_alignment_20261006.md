# Calibrated workflow 服务目标对齐奖励：冻结、匹配训练与停止决定

## 1. 审查身份与最终决定

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-06`；本轮未新增网页或文献检索
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `calibrated_workflow_service_reward_alignment_20261006_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `base_git_commit`: `cc1ecb44d998465c86350ebf4c216e13600f52e2`
- `interface_profile`: `calibrated_workflow_interface_v2`
- `action_contract`: `independent_heads_executed_env_v2`
- `evidence_level`: `E2_ARTIFACT_AUDITED_NONFORMAL_DEVELOPMENT_EXPERIMENT`
- `paper_ready_verdict`: `Unverifiable`
- `decision_category`: **D — 候选奖励引入新的服务权衡和退化**
- `auxiliary_target_ablation_decision`: **不进入下一轮**

候选奖励通过了公式级目标排序，但在同预算学习中没有改善 SA 服务质量，并使 SA、MAPPO、PPO 的总完成率都下降。
MAPPO/PPO 的 failed-service proxy、invalid prepare 和连续无进展稳定增加；SA 虽减少 action 4、invalid prepare 和传输，
却增加连续无进展并降低完成率。该结果不是“reward 变大即改善”，也不支持 SA 独有机制收益。因此保留完整负结果，
不执行原 event auxiliary target 与机制一致 target 的下一轮消融。

## 2. 先验语义审计

环境 episode 表示一个完整 workflow 的连续执行尝试。`terminated=true` 只表示全部节点完成；`truncated=true` 表示到达
环境/训练时间步上限但 workflow 尚未完成。truncation 不是任务失败，PPO 在非 terminated 截断上保留 value bootstrap。
deadline 是按累计 modeled clock 定义的服务目标，可能早于、等于或晚于 24-step 外部上限；workflow 可以逾期后继续完成。

现有字段含义如下：

- `service_failures` 是当前节点因 required bundle 不在当前 RSU 而无法执行的**失败服务尝试数**，不是 episode 失败数，
  也不是实测中断时长。
- `handoff_failures` 是 handoff 后没有可用 prepared state、因而触发 prefix recompute/input 路径的次数；普通 handoff
  不算失败，handoff failure 也不另称服务中断。
- 真实服务中断时长当前不可测。本轮只报告可追溯代理
  `failed_service_attempt_seconds_proxy = service_failures × failed_service_seconds`；当前 `failed_service_seconds=2 s` 是
  modeled delay，不是无线或生产实测。
- `action 4`、prepare 调用数、migration log 和普通 handoff 均无直接奖励。

H2 历史只读审计确认旧公式的实现重算误差仅 `2.84e-14`，但 deadline 只在完成时检查，且外部截断的未完成路径没有
终局目标项。4/5 未完成、低传输案例可得 `6.4`，高于有两次 failed attempt 的完整服务 `3.0`。本轮没有回退或重复修复
旧 `728040c…` 接口；其根因报告只作为奖励缺口历史证据。

## 3. 两套冻结奖励

### A. 原奖励

对每一步：

\[
r_A = 2I_{node} + 4I_{workflow} - 0.08\Delta t
      -0.5\frac{B}{\mathrm{GiB}} -3I_{failed}
      -4I_{workflow\land late}.
\]

这里 `Δt` 是完整 step cost，已含传输、模型加载、恢复、重算和 modeled failed-service delay；原公式因此保留其历史
多目标重复计费特征，不作追溯改写。

### B. 服务目标对齐候选 `service_aligned_v1`

\[
r_B = I_{node} + 100I_{workflow} -25I_{first\ deadline\ miss}
      -2I_{failed} -0.02t_{op}
      -0.25\frac{B}{\mathrm{GiB}} -0.05t_{recompute}.
\]

`t_op` 只含 node compute 或 vehicle fallback、model load 和 state restore；明确排除 network transfer time、recompute 和
modeled failed-service delay。`B` 只计实际 model/state/input transfer bytes 一次；recompute 和 failed attempt 各自只进一个
分项。deadline penalty 在 modeled clock 首次超过 deadline 时发放一次；逾期后仍可完成。外部截断本身没有 penalty，
也不改变 `terminated`，因此 bootstrap 语义保持正确。

权重不是按 SA 排名搜索得到：`100` 是 workflow 服务主项，显著高于冻结 workload 最多 12 个节点的稠密进度总和；
`25` 表示 deadline SLA 次级于最终完成但高于单节点进度；`2/failed attempt` 使用已定义失败事件；其余权重按 seconds 和
GiB 的单位缩放，使资源项只做同完成质量下的次级区分。候选只有这一套，没有按结果改系数。

## 4. 训练前单元验收

| 配对案例 | A return | B return | B 的期望排序 |
|---|---:|---:|---|
| 按期完整完成 | 9.000 | 103.900 | 高于同条件逾期完成 |
| 同条件逾期完整完成 | 2.600 | 78.900 | 高于接近完成但最终未完成 |
| 同完成质量、更少/更多 operation time | 9.400 / 8.600 | 104.100 / 103.700 | 更少时间更高 |
| 同完成质量、更少/更多 transfer | 9.500 / 8.500 | 104.150 / 103.650 | 更少传输更高 |
| 4/5 且超 deadline 未完成 | 1.100 | -21.730 | 低于完整服务 |
| 重复 prepare、不推进、6 次 failed attempt | -19.035 | -12.037 | 负分；无 prepare 奖励 |
| 超 deadline 后未完成并截断 | 1.600 | -21.480 | deadline miss 明确受罚 |
| deadline 前外部截断 | 6.400 | 3.600 | 无外部截断终局罚；继续 bootstrap |

全部 7 项排序检查通过。同一实例、同一动作序列在 A/B 下比较 5 个有效 step，状态、动作 mask、cache/mobility/
workflow 转移完全相同。外部截断单步测试 return=`1+0.9×2=2.8`。奖励、接口和辅助系数测试共 15 项通过后才启动训练。
原件：`artifacts/analysis/calibrated_workflow_service_reward_preflight_20261006_v1/`。

## 5. 冻结训练身份与执行

- learned methods：SA-GHMAPPO（`auxiliary_coef=0.1`）、controller-level MAPPO、PPO；规则保留原
  `two_step_cost_rule`。
- 每 reward/method 固定 seeds `7/17/29` × 192 episodes，单 episode ≤24 steps；candidate checkpoints
  `48/96/144/192`。
- 理论上限 `82,944` environment steps；实际 `24,952` steps、432 updates、107.96 s。A/B 实际 steps 分别
  `12,478/12,474`；相同 episode 上限没有被误报为相同实际交互量。
- checkpoint 只使用 4 个 dev instances 选择；顺序固定为 on-time completion、total completion、unfinished-after-deadline、
  failed-service episode、handoff failure、完成样本时延、transfer、recompute、invalid prepare。不同 reward 的 return
  完全不参与选模。
- 原奖励没有复用历史结果；A/B 都由当前实验代码重跑。90 个 candidate/selected checkpoints 本地保留，不提交。
- regression 是已暴露旧检查；frozen check 仍只是开发验证，不是独立 holdout。
- two-step rule 不读取 actual execution link，但拥有 exact transition clone 和字典序目标；其能力强于 learned scalar actor，
  结果保留但不宣称 matched model capacity。

第一次 shell-background 启动 `v1` 在第一个 episode 前被宿主回收，`0` step、`0` checkpoint、空日志；现场以
`STARTUP_FAILED_ZERO_STEP` 保留。唯一科学执行是随后由持久会话启动的 `v2`，没有自动重试、补 seed 或扩预算。

## 6. 全量服务结果

### 已暴露 regression

| reward / 方法 | on-time | completion | failed-service episode | proxy s | transfer MB | recompute s | 完成 elapsed / 覆盖 | current-missing action 4 | invalid prepare | max no-progress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A / SA | .417 | .889 | .611 | 5.11 | 714.58 | 30.57 | 69.79 / 32/36 | .780 | 3.78 | 1.39 |
| B / SA | .389 | .806 | .556 | 7.33 | 277.47 | 27.18 | 68.97 / 29/36 | .448 | 2.56 | 2.42 |
| A / MAPPO | .556 | .972 | .389 | 2.44 | 1066.23 | 23.05 | 69.01 / 35/36 | .329 | 1.78 | .69 |
| B / MAPPO | .417 | .833 | .694 | 6.94 | 854.63 | 27.64 | 63.67 / 30/36 | .799 | 4.33 | 1.64 |
| A / PPO | .222 | 1.000 | .000 | .00 | 760.28 | 42.22 | 101.32 / 36/36 | .000 | .00 | .00 |
| B / PPO | .417 | .944 | .306 | 2.56 | 950.98 | 38.74 | 82.80 / 34/36 | .337 | 1.39 | .64 |
| A/B / two-step | .833 | 1.000 | .000 | .00 | 169.03 | 4.02 | 52.30 / 12/12 | .000 | .00 | .00 |

### Frozen development check

| reward / 方法 | on-time | completion | failed-service episode | proxy s | transfer MB | recompute s | 完成 elapsed / 覆盖 | current-missing action 4 | invalid prepare | max no-progress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A / SA | .167 | .917 | .708 | 5.83 | 573.57 | 47.63 | 95.83 / 22/24 | .774 | 4.46 | 1.58 |
| B / SA | .167 | .833 | .750 | 9.17 | 177.40 | 42.03 | 93.49 / 20/24 | .424 | 3.42 | 3.29 |
| A / MAPPO | .292 | .958 | .333 | 2.58 | 896.40 | 41.83 | 91.42 / 23/24 | .317 | 2.21 | .75 |
| B / MAPPO | .250 | .875 | .833 | 8.00 | 566.14 | 40.70 | 87.17 / 21/24 | .828 | 5.29 | 1.96 |
| A / PPO | .000 | 1.000 | .000 | .00 | 676.41 | 46.44 | 107.70 / 24/24 | .000 | .00 | .00 |
| B / PPO | .125 | .958 | .333 | 2.83 | 894.79 | 61.15 | 111.88 / 23/24 | .308 | 1.58 | .71 |
| A/B / two-step | .625 | 1.000 | .000 | .00 | 65.03 | 2.43 | 51.67 / 8/8 | .000 | .00 | .00 |

完成 elapsed 只统计完成样本，覆盖率单列；候选失败样本不能被条件时延均值隐藏。全部 seeds、instances 和 41 个
completion 下降 strata 均保留在 artifact 中。

## 7. 配对区间与行为解释

source-window outer bootstrap（5,000 draws）的 B−A：

- SA completion：regression `-0.083 [-0.222,0.056]`，frozen `-0.083 [-0.250,0]`；on-time 分别
  `-0.028 [-0.194,0.111]` 与 `0 [-0.125,0.125]`。没有稳定服务收益。
- MAPPO completion：regression `-0.139 [-0.333,0]`，frozen `-0.083 [-0.250,0]`；failed-service episode
  `+0.306 [0.111,0.528]` / `+0.500 [0.250,0.708]`，invalid prepare
  `+2.556 [0.722,4.861]` / `+3.083 [1.458,5.208]`。
- PPO on-time 在 regression 改善 `+0.194 [0.056,0.361]`，但 completion `-0.056 [-0.139,0]`，failed-service
  episode `+0.306 [0.194,0.417]`，invalid prepare `+1.389 [0.806,1.972]`；frozen 同样 completion 下降且
  failed-service/invalid prepare 增加。因此它是权衡，不是服务质量共同改善。
- two-step 的行为与全部服务指标严格不变；只是在同一轨迹上换评分尺。

行为上，B 使 SA action-4 占比由 regression/frozen `.801/.771` 降到 `.458/.474`，transfer 和 invalid prepare 下降；
但 action 3/2 增多，步骤数、连续无进展和未完成样本增加。B 反而使 MAPPO action 4 升到 `.782/.792`，PPO 从近乎不用
action 4 升到 `.443/.391`，并引入服务失败。相同标量目标对三类参数化策略产生不同退化，不能解释为 SA 独有收益。

同轨迹只读重评分区分了“换尺”和“行为”：例如 original-SA regression 轨迹按 A/B 计为 `0.871/73.566`，数值尺度
完全不同，不能直接比较；candidate-SA 轨迹按 A/B 计为 `-2.969/65.422`。two-step A/B 行为完全一致而 return 改变，
直接证明 return 变大不等于行为改善。

## 8. 论文边界与下一步决定

已验证的系统机制仍是 state recovery、dependency-safe typed cache 和真实 adapter lifecycle；本轮没有改变这些机制。
奖励目标对齐在单元案例上成立，但学习行为出现退化，只能报告为负向目标设计实验。SA 独有机制没有获得支持；候选没有使
SA 稳定改善，也没有形成多数算法共同改善。奖励调整不是算法创新，且本轮不能建立新颖性或泛化。

最终决定为 **D**，不是 A/B/C/E：实现和预算均完成，return 尺度变化之外观察到真实行为变化，但该变化包含完成率下降、
失败尝试和无进展上升。故**不进入 auxiliary-target 消融**。此前 no-auxiliary 负结果继续保留；不得把删除失败消融、
选择有利 seed/stratum 或继续搜 reward 权重写成机制证据。

## 9. Artifact 入口

- 奖励冻结与单元验收：`artifacts/analysis/calibrated_workflow_service_reward_preflight_20261006_v1/`
- 零步启动失败：`artifacts/benchmarks/calibrated_workflow_service_reward_alignment_20261006_v1/`
- 唯一科学 run：`artifacts/benchmarks/calibrated_workflow_service_reward_alignment_20261006_v2/`
- 学习曲线：`training_curves.json/csv`、两份 `training_completion_curve_*.svg`
- checkpoint 选择：`checkpoint_selection.json`、`training_summary.json`
- 全量结果：`evaluation_rows.json/csv`、`behavior_ledger.csv`、`seed_summary.csv`、`stratified_results.csv`
- 统计与负结果：`paired_reward_arm_deltas.json`、`negative_regions.csv`、`reward_decomposition.csv`
- 论文表：`paper_table.md`
- provenance：`run_manifest.json`、`execution_code_identity.json`、`completion_receipt.json`、
  `analysis_receipt.json`、`artifact_integrity.json`
- checkpoint：90 个本地文件，hash 纳入 integrity；不提交或上传。
