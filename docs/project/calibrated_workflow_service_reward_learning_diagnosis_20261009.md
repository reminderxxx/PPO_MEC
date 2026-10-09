# Calibrated workflow 服务奖励学习环节诊断

## 审查身份与结论

- `reviewed_at`: `2026-10-09`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-09`；本轮未检索或评价新文献
- `target_venue`: `IEEE TMC`（同时向 CSCWD 2027 A 线提供边界说明）
- `artifact_run_id`: `calibrated_workflow_service_reward_learning_diagnosis_20261009_v1`
- `source_run_id`: `calibrated_workflow_service_reward_alignment_20261006_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_diagnosis`: `70a83afb5d3d3b3fb40fcaeaf825a82e4b74c416`
- `source_execution_commit`: `cc1ecb44d998465c86350ebf4c216e13600f52e2`
- `evidence_level`: `E2_ARTIFACT_AUDITED_NONFORMAL_OFFLINE_DIAGNOSIS`
- `verdict`: `UNVERIFIED_FOR_ALGORITHM_OR_PAPER_CLAIM`

本轮没有确认当前执行版本存在 termination、reset observation、跨 episode GAE 或 executed-action PPO 接线错误，
因此不进入“纠错修复”。最值得补足的是 **service-aligned 大回报下 critic 的尺度适配和信用分配**：冻结开发样本上，
critic 梯度压倒 actor 并触发近乎全局裁剪，critic 本身又没有拟合该尺度；这比 auxiliary 冲突或 truncation 更一致地覆盖
SA、MAPPO、PPO。下一轮只冻结一个候选：对 value target/output 使用 PopArt running mean/std，bootstrap 始终使用反归一化
value；reward、actor、advantage normalization、value coefficient、auxiliary、网络、动作、数据和裁剪阈值全部不变。

这是一项学习稳定性候选，不是算法创新，也不是 SA 优势证据。本轮训练、评价、holdout、下载和参数更新均为 0。

## 证据边界与复现

诊断脚本先核验 6 个 agent/trainer 文件与执行 commit 一致、6 个关键实验文件与原 run identity 一致，再核验 90 个本地
checkpoint（18 selected）与原 artifact integrity 和 training summary 的 SHA-256。随后按已提交
`behavior_ledger.csv` 的 executed action 重放 360 个 episode、3,161 steps；状态转移/奖励 mismatch=`0`。checkpoint 未复制、
未上传，旧 evidence 未修改。

直接记录可提供每 episode 训练统计、全部 dev checkpoint 分数、selected evaluation 行为账本和每 cell 最后一次 update 的
loss/KL/clip/entropy/explained variance。离线重算提供 selected-checkpoint value/GAE、状态分组和 episode-48 固定 dev 样本
full-batch ratio=1 梯度。缺失原训练 transition ledger、前 23 次 update 详情、minibatch 顺序、瞬时 old policy 和原 update
裁剪前后梯度，因此下文**不声称精确重现原训练更新**。

复现命令：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python \
  scripts/diagnose_calibrated_workflow_service_reward_learning.py \
  --checkpoint-root \
  /Users/howen/.codex/worktrees/calibrated-workflow-training/PPO_MEC/artifacts/benchmarks/calibrated_workflow_service_reward_alignment_20261006_v2
```

输出：`artifacts/analysis/calibrated_workflow_service_reward_learning_diagnosis_20261009_v1/`。该目录 create-only；
`diagnosis_receipt.json`、`code_identity.json`、`checkpoint_identity.json`、`replay_receipt.json` 和
`artifact_integrity.json` 给出身份与零新增实验回执。

## 四项诊断

### 1. 终止、截断与 bootstrap

- workflow 完成才令 `terminated=true`；failed service 是可恢复 step event；instance `max_steps` 到达而未完成时
  `truncated=true`。runner 另有 24-step cap，冻结实例均满足 `max_steps<=24`。
- trainer 在 step 后的最终、未 reset observation 上估值；仅 terminated 置 bootstrap 为 0。每个 episode 独立
  `PPORolloutBuffer.finalize()` 后才拼接 rows，GAE 不跨 episode。
- 训练把 non-terminal truncation 视为可继续并 bootstrap，而 dev/evaluation 在同一有限预算把未完成计 0 completion；这是
  **目标语义错配**，但不是已证根因。18 个训练 cell 的 truncation incidence 仅 `0%–3.65%`；selected-ledger 有 31 个
  truncated episode，平均 bootstrap value=`2.963`，首步 target 增量均值=`1.119`、范围 `[-0.176,3.968]`。
  预算外真实 return 未观测，不能断言未完成轨迹被系统性高估。
- 当前 workflow 语义允许 deadline 后继续完成，但不允许无限越过 instance/评价预算；因此不预设 truncation 终局罚。
  若未来任务定义改为硬 24-step 服务预算，应同时修改训练目标和评价定义，不能只事后加罚。

### 2. 奖励尺度和 PPO 信号

service-aligned selected 轨迹的 workflow completion 分项均值约 `80.56–95.83`（regression）和
`83.33–95.83`（frozen check），但成本/失败/deadline 分项仍使 return 区分真实行为；不能因 `+100` 单独判根因。
最后一次 update 的直接日志显示：原奖励 value loss=`8.24–77.72`、explained variance=`-0.288–0.580`；service-aligned
value loss=`3,123–4,591`、explained variance=`0.000001–0.000439`。离线 selected value prediction 多在 `2.44–6.96`，
而 target mean=`38.35–73.89`。

在事前固定的 episode-48 四个 dev 实例上，service-aligned 的全参数 value:policy gradient norm ratio 为
`539–7,907`，估算 global clip scale=`0.00116–0.00147`；原奖励对应 `30–184` 与 `0.00914–0.08726`。
仅把同一 value residual 除以 target variance 的诊断版本可把 service ratio 降到 `0.93–26.51`。该操作没有更新参数，
但表明可干预的是 critic scale 与共享裁剪的耦合，而非单看奖励绝对值。

对逾期完成 episode，configured `γ=.99, λ=.95` 的首步 target 在 service-aligned 九个 cell 均值为
`43.19–54.77`；改 `λ=1` 会再增加 `17.83–21.24`，完全不折扣会增加 `22.27–26.84`。因此当前折扣已经显著衰减迟到
完成奖励；把 `λ` 调大不是尺度修复候选。已完成 episode 不 bootstrap，相关 cohort 的 bootstrap delta 为 0。

### 3. 无效行为保留

完整账本按 current bundle ready/missing、target prepare feasible/infeasible、deadline 前/后、progress/no-progress 和
连续无进展分组。service-aligned 中“不明智 action 4”（current missing 或 target infeasible）在 7/9 cell 可见，raw
advantage 为正比例=`60.3%–100%`；按完整 cell 做离线 normalization 后为正比例=`16.2%–79.3%`。这说明最终完成信用可令
早期失败动作的 raw advantage 为正，但也说明不能把 raw 符号直接当成原 minibatch 的实际更新方向。

固定状态、四个 candidate checkpoint 的只前向轨迹显示，current-missing action-4 probability 经常下降，但 margin 不足以
改变 argmax：SA seed 7/29 分别 `.517→.423` / `.695→.386`，argmax rate 仍为 `1.0` / `.889`；MAPPO seed 7
`.590→.400`，argmax rate 仍 `.889`。PPO seed 17 `.250→.115` 且 argmax 始终 0。故现有证据更符合“部分方向正确但
critic/共享裁剪使更新不足或不稳定”，而不是所有无效动作都收到同方向正反馈。

SA auxiliary 与 policy 的全 dev 梯度 cosine 为 `-0.683/0.088/0.485`，current-missing subset 为
`-0.693/-0.696/0.343`，没有跨 seed 一致冲突。既有 no-aux 消融也未修复退化，故不进入 auxiliary-target 消融。

### 4. 学习曲线和 checkpoint 选择

service-aligned 9 个 learned cell 中 8 个选 episode 48，只有 SA seed 17 选 episode 144；原奖励选择分布为
episode 48/96/144/192=`4/2/1/2`。选模顺序以 on-time、total completion 开始，reward 和成本项不在服务项之前，故没有
“按成本牺牲完成量”的直接证据。但 dev 只有 4 个实例，on-time 的粗粒度字典序使早期 checkpoint 占优；例如多个 later
checkpoint 已降低 current-missing action 4，却因 on-time 首项较低而不会被选。没有现成 later-checkpoint evaluation，
本轮没有补跑，也不声称 later checkpoint 更好。

全部 cell 都有 24 updates，但实际 exposure 为 `1,216–1,456` steps；固定 episode 数不等于固定交互量。该差异和小 dev
选模是下一轮共同协议需要消除的混杂，但它们本身不能解释三种方法共同的 critic 尺度现象。

## 两奖励 × 三方法 × 三 seed 诊断表

`V tgt/pred/RMSE`、`bad4 +raw/+norm` 和 `V:P` 为离线重算；`Vloss/EV`、`KL/clip` 为原 cell 最后一次 update 直接日志；
`clip scale` 是固定 dev full-batch 梯度的估算值。空白表示该 cell 没有不明智 action 4，不把它记为 0% positive。

| reward | method | seed | steps / updates | selected | V tgt / pred / RMSE | Vloss / EV | KL / clip | V:P / clip scale | bad4 n / +raw / +norm |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| original | SA | 7 | 1456 / 24 | 96 | 1.13 / .18 / 6.05 | 35.33 / .295 | .0080 / .057 | 158 / .00914 | 26 / .192 / .154 |
| original | SA | 17 | 1408 / 24 | 48 | -3.19 / -.17 / 8.62 | 17.04 / -.288 | -.0042 / .009 | 184 / .01352 | 101 / .149 / .218 |
| original | SA | 29 | 1440 / 24 | 48 | -2.38 / .68 / 8.04 | 18.63 / .580 | .0031 / .040 | 176 / .02561 | 116 / .129 / .259 |
| original | MAPPO | 7 | 1427 / 24 | 192 | 4.14 / .95 / 6.16 | 51.87 / .392 | .0017 / 0 | 31 / .06167 | 34 / .559 / .382 |
| original | MAPPO | 17 | 1389 / 24 | 96 | 3.19 / -.10 / 6.12 | 19.67 / .163 | .0042 / 0 | 66 / .07844 | 43 / .535 / .279 |
| original | MAPPO | 29 | 1414 / 24 | 48 | -1.23 / .10 / 8.75 | 77.72 / .226 | .0032 / 0 | 63 / .03797 | 40 / .250 / .325 |
| original | PPO | 7 | 1317 / 24 | 144 | 1.95 / .46 / 4.12 | 32.13 / .306 | .0005 / 0 | 48 / .06388 | 0 / — / — |
| original | PPO | 17 | 1308 / 24 | 48 | 1.42 / -.06 / 4.49 | 8.24 / -.099 | -.0010 / 0 | 45 / .08380 | 0 / — / — |
| original | PPO | 29 | 1319 / 24 | 192 | 3.10 / .64 / 4.32 | 17.27 / .261 | -.0014 / 0 | 30 / .08726 | 0 / — / — |
| service | SA | 7 | 1437 / 24 | 48 | 49.14 / 3.88 / 59.65 | 3975.44 / .000002 | -.0002 / 0 | 539 / .00116 | 0 / — / — |
| service | SA | 17 | 1443 / 24 | 144 | 38.35 / 6.96 / 47.87 | 4525.90 / .000001 | .0049 / 0 | 7249 / .00121 | 58 / .862 / .793 |
| service | SA | 29 | 1443 / 24 | 48 | 46.08 / 3.85 / 58.45 | 4519.76 / .000005 | .0024 / 0 | 7907 / .00119 | 116 / .603 / .457 |
| service | MAPPO | 7 | 1456 / 24 | 48 | 45.73 / 3.02 / 58.98 | 3708.95 / .000214 | -.0008 / 0 | 5753 / .00137 | 116 / .603 / .457 |
| service | MAPPO | 17 | 1353 / 24 | 48 | 45.67 / 2.89 / 59.08 | 4439.65 / .000304 | .0024 / .015 | 5933 / .00143 | 116 / .603 / .457 |
| service | MAPPO | 29 | 1455 / 24 | 48 | 45.00 / 2.65 / 58.72 | 3123.14 / .000206 | .0016 / .057 | 859 / .00144 | 51 / .686 / .510 |
| service | PPO | 7 | 1345 / 24 | 48 | 73.89 / 2.44 / 73.36 | 3790.20 / .000408 | .0004 / 0 | 1570 / .00131 | 37 / 1.000 / .162 |
| service | PPO | 17 | 1216 / 24 | 48 | 69.35 / 3.17 / 68.01 | 4591.20 / .000439 | .0007 / 0 | 2508 / .00137 | 0 / — / — |
| service | PPO | 29 | 1326 / 24 | 48 | 44.92 / 2.46 / 58.85 | 4463.33 / .000339 | -.0003 / 0 | 1292 / .00147 | 51 / .686 / .510 |

hierarchical SA/MAPPO 最后日志中的 `actor_loss=0` 不是缺陷：当前 contract 只优化 executed environment action，实际 policy
loss 在 `env_action_ppo_loss`，表中已使用该字段。

## 最小反例

选择规则事先固定：只在 service arm 中按 SA→MAPPO→PPO、seed、split、design、step 排序，从最终完成 episode 里取首个
current bundle missing 或 target prepare infeasible 的 action 4；advantage 不参与选择。见 `minimal_witness.json`：

- SA seed 17，`regression_02`，step 2；current bundle missing、target feasible、deadline 前；action 4 probability=`.524632`。
- 当步 reward=`-2`，service failed、未进展且形成连续无进展；该 episode 最终逾期完成。
- selected critic `V=6.9618`，configured target=`32.0331`，raw advantage=`+25.0713`；完整 cell 离线归一化后
  advantage=`-0.1748`。

这证明迟到 completion credit 可以让局部失败动作获得正 raw advantage，也同时否定“该样本必然在原 PPO minibatch 中被
正向强化”的更强说法。

## 假设裁决与唯一候选

| 假设 | 裁决 | 依据 |
| --- | --- | --- |
| 当前 termination/GAE 实现错误 | 否定 | final non-reset obs、episode-local finalize、hash-matched 当前代码与 0 mismatch replay |
| truncation bootstrap 是主因 | 未支持 | 存在目标语义错配，但训练 incidence 低；预算外 return 不可观测 |
| `+100` 本身就是根因 | 否定其充分性 | 分项目标排序正确；真正一致异常是 critic 拟合和梯度/裁剪比例 |
| auxiliary 梯度冲突是优先项 | 否定 | cosine 跨 seed 不一致；no-aux 既有负结果 |
| 无效动作始终获正更新 | 否定 | raw advantage 常为正，但 cell normalization 可翻转；缺原 minibatch/old policy |
| critic scale / shared clipping 是可干预原因 | 支持为首选候选，尚非因果确认 | 三方法九 seed cell 同向的 value loss、EV、V:P ratio 和 clip scale；离线尺度归一化可显著降 ratio |
| 选模或曝光差异独自解释退化 | 未定位 | 8/9 早选且 steps 不等，但没有 later-checkpoint evaluation，不可事后挑选 |

唯一候选为 `critic_target_popart_normalization`：value head 预测标准化 target；running mean/std 更新时只调整最后 value
affine 层以保持反归一化预测不变；GAE/bootstrap 使用反归一化 value。方案冻结于
`configs/experiment/calibrated_workflow_value_normalization_ab_v1.json`，`execution_authorized=false`。

## 下一轮唯一 A/B 与否证条件

A 为当前 raw critic target，B 只启用上述 PopArt critic contract。两臂固定 `service_aligned_v1`、SA/MAPPO/PPO、
seeds `7/17/29/43/61`、每 cell 1,440 environment steps、24 update opportunities、每次 60 transitions、4 PPO epochs、
batch 32、共 192 optimizer steps；checkpoint 只在 updates `6/12/18/24` 产生。相同初始化 seed、实例顺序、reward、
actor/auxiliary、网络、value coef、clip 和选模规则；0 hyperparameter search。当前 36-instance manifest 已全部暴露，只能作
development A/B；confirmatory 前必须另冻原始 interval 互斥的新数据并整合窗口身份修复 `6efdde1` 或其审计后继。

主要服务指标为 on-time completion、total completion、unfinished-after-deadline、failed-service attempt/episode 和连续无进展；
完成时延必须同时给覆盖率。机制指标为 denormalized critic RMSE/EV、policy/value/aux gradient norms、clip scale/fraction，
以及 current-missing action-4 probability/argmax。

以下任一结果否定或停止该改进：PopArt statistic update 不能在 `1e-6` 内保持反归一化预测；两臂预算不等或 identity/
interval gate 失败；paired median V:P ratio 不降且 EV 不升；无效 action 概率不降并伴随失败/无进展上升；total/on-time
completion 下降；或 completion 与 failed-service 指标均无改善。失败后停止，不转入 reward 搜索、auxiliary 消融或追加 seed。

最终回答：**当前最值得补足的是 critic 对服务完成回报尺度的适配，以及由此造成的共享梯度裁剪下信用传递不足。**证据是
九个 service cell 一致的近零 EV、千量级 value loss、`539–7,907` 的 value:policy 梯度比和约 `.0013` 的 clip scale，
并有固定状态上“概率下降但 argmax 未翻转”的行为见证。下一轮只改变 critic target/output 的 PopArt normalization；上述
机制门和服务门任何一个失败，都否定该候选。
