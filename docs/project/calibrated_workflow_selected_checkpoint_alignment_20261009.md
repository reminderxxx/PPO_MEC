# Calibrated workflow selected-checkpoint 机制—服务对齐复核

## 身份与裁决

- `reviewed_at`: `2026-10-09`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-09`；本轮未检索新文献
- `target_venue`: `CSCWD 2027`；同时保留 TMC 审查边界
- `artifact_run_id`: `calibrated_workflow_selected_checkpoint_alignment_20261009_v1`
- `source_run_id`: `calibrated_workflow_value_normalization_ab_20261009_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scientific_execution_commit`: `858bc797e23b4e56051663f28d6fd7681f5ee77d`
- `state_freeze_commit`: `1c85bea470f775182a8f2af7312b7ff760b35d0e`
- `evidence_level`: `E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY_OFFLINE_FORWARD`
- `verdict`: `POPART_CRITIC_MECHANISM_CONFIRMED_AT_SELECTED_CHECKPOINTS; SERVICE_SUFFICIENCY_FALSIFIED`

本轮关闭了原报告中的 consumer identity 缺口：使用实际被 service consumer 评价的 30 个 selected checkpoint，
在 checkpoint 之前结果盲冻结的 48 个公共 dev 状态上统一前向。PopArt selected checkpoint 对固定行为完整
continuation return 的绝对 value error 在 `720/720` 个逐状态配对中下降，但 action 4 概率方向不一致：
SA `+.0341`、MAPPO `-.0690`、PPO `+.0268`。这与既有 SA 局部改善、MAPPO 中性、PPO on-time
`.29→.19` 相容，确认的是“critic 尺度机制兑现但不足以给出稳定服务策略”，不是 PopArt 服务优势。

因此 raw critic 继续是 canonical 参照；PopArt 仅保留为非原创的训练稳定性设置。SA 的局部正结果不构成整体晋级，
也没有 SA 稳定领先证据。本轮没有训练、选模、参数更新、holdout、模型生成、下载或 gate 修改。

## 冻结状态与证据边界

状态集只来自既有四个 dev instance。预声明规则为：每个 instance 从空前缀开始，按 action `0..4` 的合法动作
做 breadth-first prefix 枚举，去除公共状态 hash 重复，取前 12 个，prefix depth 最多 4。清单在读取任何 checkpoint、
训练/评价误差或 advantage 前独立生成；48 个状态均唯一。序列化字段只含 observation、`semantic_state`、action mask、
deterministic flag 和 raw-policy metadata；没有 `actual_mbps` 或其他未来实际链路字段。

每个状态用固定优先级 `[2,0,3,1,4]` 延续到既有 24-step episode 边界，48/48 都得到完整 discounted return。
该量的严格名称是 `complete_fixed_behavior_discounted_return_not_vpi`：它是公共、合法且可复现的固定行为 probe，
**不是**无偏 `V^π`、不是原训练 minibatch target，也不能用于 GAE 或精确 update 重现。它只能检验原单位 value 是否追踪
一个共同、闭合的回报尺度。

前向总量为 `30 checkpoints × 48 states = 1,440`，control/candidate 状态配对为 720。30 个 checkpoint 的文件
SHA-256 与 source `training_summary.json` 一致；network、optimizer、PopArt state 和 update count 的前后 hash 全部不变。
28/30 selected checkpoint 早于 update 24。15 个 method×seed arm-pair 中只有 6 对选择相同 update，因此 selected-checkpoint
表描述的是“训练臂 + 预注册 dev selection”的最终联合结果，不能冒充同一 update 的纯 normalization treatment effect；
update-24 机制表仍作为共同训练终点证据保留。

## 结果

### 方法汇总

| method | fixed-behavior RMSE raw→PopArt | mean P(action 4) raw→PopArt | probability argmax-4 raw→PopArt | executed action-4 raw→PopArt | 既有 completion Δ | 既有 on-time Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SA-GHMAPPO | `83.60→18.08` | `.503→.537` | `1.000→.846` | `.650→.575` | `+.13` | `+.02` |
| controller-MAPPO | `85.86→69.16` | `.577→.508` | `1.000→.800` | `.800→.721` | `.00` | `.00` |
| PPO | `85.02→67.74` | `.245→.272` | `.467→.200` | `.467→.200` | `+.04` | `-.10` |

三种 action 统计不可互换：`probability argmax` 是 action distribution 的最大项；`raw_env_action` 是层级 head 的
deterministic aggregation；`projected_env_action` 是 action-mask 投影后的动作；本次 raw-policy forward 中
`executed_action=projected_env_action`。例如 SA raw 的 probability argmax-4 为 1.0，但 raw head aggregation action-4
仅 `.529`，投影后执行率 `.650`。因此“概率下降”“argmax 翻转”和“执行动作改变”必须分别报告。

### 全 method × seed selected identity 对齐

`selected C→P` 是 control/PopArt 的 dev-selected update；RMSE target 仍是固定行为 probe，不是 `V^π`。

| method | seed | selected C→P | RMSE C→P | P(action 4) C→P | prob argmax-4 C→P | executed-4 C→P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sa_ghmappo | 7 | 6→24 | 85.29→13.90 | .353→.498 | 1.000→.542 | .604→.458 |
| sa_ghmappo | 17 | 18→24 | 82.30→14.08 | .442→.616 | 1.000→.688 | .000→.646 |
| sa_ghmappo | 29 | 12→18 | 83.72→27.06 | .613→.472 | 1.000→1.000 | 1.000→.458 |
| sa_ghmappo | 43 | 12→18 | 83.85→16.11 | .538→.578 | 1.000→1.000 | .646→.708 |
| sa_ghmappo | 61 | 18→18 | 82.80→15.84 | .571→.522 | 1.000→1.000 | 1.000→.604 |
| mappo | 7 | 6→6 | 86.56→76.49 | .580→.594 | 1.000→1.000 | 1.000→1.000 |
| mappo | 17 | 6→6 | 86.29→72.91 | .593→.392 | 1.000→1.000 | 1.000→.604 |
| mappo | 29 | 12→6 | 83.53→74.47 | .628→.562 | 1.000→1.000 | 1.000→1.000 |
| mappo | 43 | 6→6 | 86.51→70.61 | .681→.771 | 1.000→1.000 | 1.000→1.000 |
| mappo | 61 | 6→12 | 86.36→47.09 | .404→.221 | 1.000→.000 | .000→.000 |
| ppo | 7 | 6→6 | 86.49→73.27 | .301→.299 | 1.000→1.000 | 1.000→1.000 |
| ppo | 17 | 12→6 | 83.11→70.56 | .307→.334 | .729→.000 | .729→.000 |
| ppo | 29 | 6→6 | 86.43→74.07 | .265→.271 | .000→.000 | .000→.000 |
| ppo | 43 | 18→6 | 82.65→72.47 | .099→.187 | .000→.000 | .000→.000 |
| ppo | 61 | 6→12 | 86.31→43.09 | .254→.269 | .604→.000 | .604→.000 |

SA、MAPPO、PPO 的固定行为 MAE 分别在 `240/240` 状态配对下降；这排除了“机制改善只存在于未被评价的 update 24”
这一解释。它没有排除 value 对各自 policy continuation 排序错误，因为本轮依法没有生成新的 policy rollout target。
策略侧则明显异质：SA/PPO 的 mean probability 上升，但 argmax/执行率下降；MAPPO 三者总体下降。概率质量移动、head
aggregation 和 mask projection 共同决定执行动作，单看 action-4 次数仍不足以解释完成率。

## 与原四项诊断的合并裁决

1. termination/truncation/bootstrap：仍未发现当前实现错误；本轮没有新增证据支持终局罚。
2. critic scale：selected checkpoint 上仍强支持 PopArt 跟踪回报尺度；但 probe 不是 `V^π`，claim 限于 scale tracking。
3. 无效动作：critic 改善没有使 action-4 probability 跨方法同向下降，说明最接近服务退化的剩余环节是 actor 的局部
   信用方向，而不是 critic 数值尺度本身。既有见证中，失败、无进展且未提交 migration 的 action 4 仍可取得正 raw
   advantage；本轮不把它夸大为原 minibatch 的精确更新方向。
4. 选模：共同 dev 字典序没有读最终评价，但 28/30 早选且 9/15 arm-pair 的 selected update 不同。该事实解释了为何
   update-24 与 service consumer 不能逐 cell 直接相连；本轮已按实际 consumer 身份补齐，而未另选 checkpoint。

## 唯一下一轮候选（仅冻结，不实现）

优先候选从 critic normalization 转为 **non-committing failed-action advantage cap**：仅当训练 transition 同时满足
`executed_action==4`、`service_completed=false`、完成节点数未增加且 `migration_success=false` 时，actor 使用的 normalized
advantage 取 `min(A_t, 0)`；critic return/value target、reward、PopArt、auxiliary、网络、mask、观察、动作权限和其他 transition
全部不变。该条件只使用训练后可观测 transition，不向 policy 提供实际未来链路或额外推理权限，并对 SA/MAPPO/PPO 对称。

机制依据是：这种 action 4 在当前环境中既未服务当前节点，也未提交迁移，延迟 completion credit 不应把它局部正向强化；
PopArt 已证明单纯修 value 尺度不会稳定修正其概率。它是待证的 credit-assignment 约束，不是算法创新，也不是
auxiliary-target 消融。

下一轮若获独立授权，A 固定为本轮 PopArt；B 只增加上述 advantage cap。沿用 5 seeds、每 cell 1,440 steps、24 updates、
192 optimizer steps、同一四 dev checkpoint selection 和既有暴露 evaluation splits。主要指标为 on-time/total completion、
service-failure attempt、连续无进展；机制指标为冻结 common states 上的 action-4 probability/margin/argmax，并单列 cap
触发覆盖率。出现任一条件即否定并停止：cap 触发样本为零或身份/预算不等；相关状态 P(action 4)/margin 不降；任一方法
on-time 或 total completion 下降；failure/no-progress 不改善；或收益只来自改变 action mask/奖励/额外信息。不得补 seed、调 cap、
切 checkpoint 或转入 auxiliary sweep。

## 产物

- 冻结状态：`artifacts/analysis/calibrated_workflow_selected_checkpoint_states_20261009_v1/`
- 对齐表与图：`artifacts/analysis/calibrated_workflow_selected_checkpoint_alignment_20261009_v1/`
- 主表：`method_seed_summary.csv`、`method_summary.csv`
- 逐状态：`fixed_state_forward_rows.csv`、`fixed_state_pair_rows.csv`
- 身份：`checkpoint_identity.csv`、`alignment_summary.json`、`artifact_integrity.json`
- 论文机制图：`selected_checkpoint_alignment_mechanism.svg`

最终回答：**当前最值得补足的是 actor 对“失败且未提交迁移的 action 4”的局部信用分配。** selected checkpoint 证据表明
PopArt 已把共同固定行为 probe 的 value error 在 720/720 配对中降低，却没有让 action-4 probability 或服务指标共同改善；
下一轮只加入上述 advantage cap。只要相关概率/margin 不降，或任一方法的 on-time/total completion 受损，即否定该改进。
