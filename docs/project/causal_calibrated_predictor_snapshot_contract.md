# Causal Calibrated Predictor Snapshot Contract

状态：`frozen`

冻结日期：`2026-08-19`

snapshot contract：`1.0.0`

calibration artifact contract：`causal_predictor_calibration_artifact_v1.0.0`

decision observation trace：`1.0.0`

## 1. 范围与结论边界

G12 为现有 `supervised_handoff_predictor_v1` 冻结可审计的概率校准、因果时间、慢更新、staleness、abstention 和安全消费合同。它不改变 `semantic_discrete_5`、reward、RL loss 或 policy architecture，不执行 G13，不自动进入 canonical profile。合同和非正式诊断不能解释为 policy benefit，也不能解释为完整 Digital Twin。

当前 runtime feature flag 为 `causal_calibrated_snapshot_enabled=false`。只有显式配置 `predictor_kind=supervised`、checkpoint、calibration artifact 和 fallback 后才启用。fallback 仅允许 `mask_only` 或 `no_prediction`；两者都不切换 oracle。

## 2. Snapshot schema

JSON snapshot 固定为四段：

- `identity`：snapshot ID、contract version、predictor kind/model/checkpoint、config hash、calibration method/version/artifact hash、source dataset/window-plan、Git commit、vehicle ID 和独立 oracle flag。
- `causal_time`：`generated_at`、`observation_as_of`、label horizon、valid interval、`consumed_at`、age/staleness、update interval K、source frame/time interval、history start/end 和 causal cutoff。
- `predictions`：current RSU、next-RSU distribution/class、handoff probability/decision、target distribution/class、ETA point、可识别时的 interval/uncertainty、可选且有来源的 demand belief、raw logits、calibrated probabilities、confidence/uncertainty、abstention reasons 和 availability mask。
- `audit`：feature availability、unseen handling、normalization version、validator status、oracle flag、leakage fields 和 fallback。

未知值必须为 JSON `null`；禁止 NaN/inf。`0.0` 是已知概率零，不表示 unavailable。ETA interval 当前不可识别，保持 `null`；binary classification confidence 不得充当 ETA uncertainty。

## 3. 因果时间

`observation_as_of <= generated_at <= consumed_at` 同时约束 step 与 time。history/source interval 末端不得超过 as-of；future label 只存在于离线 train/calibration/evaluation reducer，runtime snapshot 禁止 label、reward、service result、oracle action。`age_steps=consumed_at_step-generated_at_step`，过期条件是 `consumed_at_step > valid_until_step`。

slow predictor 每 K step真正运行一次；fast actor 每 step读取已生成 snapshot。`prediction_delay_steps=D` 统一解释为只选择 `generated_at_step <= current_step-D` 的历史对象，禁止在当前时刻重算再伪装为延迟结果。window/episode reset 清空 feature history、prediction history 和 snapshot history；开局历史不足记录 `insufficient_history`。

oracle 使用独立 `predictor_kind=oracle` 和 `oracle=true`；supervised snapshot 强制 `oracle=false`。validator 对未来 as-of、generation/consumption倒序、source/history越界、stale未标记、identity冲突、非法概率和 leakage field fail-fast。

## 4. 三段 split 与 provenance

冻结输入为 v71 非 hidden `train_window_plan.json` 和 `dev_window_plan.json`。在查看 G12 calibration metric 前固定规则：train-plan 顺序索引 `index % 4 == 3` 的窗口进入 calibration，其余进入 predictor-train；原 dev 只作 evaluation/dev。结果为：

- predictor-train：15 windows / 29,067 quality rows；
- calibration：5 windows / 11,992 rows；
- evaluation/dev：20 windows / 33,260 rows。

原始 frame、time 和 segment-frame interval 跨 split 零冲突；manifest SHA-256 为 `931a226b039d27501f84b10b33917cdb9bf55a54da17fa633b3bd217af1a7253`。vehicle ID 跨 split 重现被记录为相邻轨迹依赖风险；独立性依据是原始区间隔离，不是 window ID。formal、holdout、hidden 均拒绝。calibration fit 和 threshold selection 不读取 evaluation label 或 RL reward。

## 5. Calibration 与指标

pure reducers 位于 `src/predictors/calibration.py`。binary handoff 报告 count/prevalence、Brier、NLL、ECE、MCE、固定 reliability bins、AUROC，以及独立的 threshold-dependent precision/recall/F1。multiclass reducer报告 top-1/top-k、macro/weighted F1、Brier、NLL、top-label/classwise ECE、confusion、unseen coverage 和 simplex。ETA reducer报告 MAE/RMSE/median absolute error、interval coverage 和固定 ETA buckets。零分母与 missing 为 `null`。

reliability bins 固定为 `[0,.1), ... [.9,1.0]`，最后一段含 1；每 bin 记录 count、mean confidence、empirical frequency/accuracy、gap、empty/small-bin，ECE 按样本数加权。不得按结果重分箱。

候选方法只含 identity 与 deterministic temperature scaling。optimizer 在 log-temperature `[-4,4]` 上执行固定 96 次 golden-section；seed记录但无随机搜索。选择顺序固定为 calibration NLL、Brier、再优先 identity。v112 binary handoff 选择 temperature `1.0815913534284083`。evaluation/dev：

- Brier：`0.0414765275 -> 0.0412310750`
- NLL：`0.1496381084 -> 0.1480084669`
- ECE：`0.0111714768 -> 0.0056475371`
- MCE：`0.0954854545 -> 0.0572283667`
- AUROC：`0.8729678455`，排序不变。

历史 quality rows 未保留 next-RSU/target 全量 logits，因此真实 multiclass Brier/NLL/ECE 不能可靠重建，明确为 unavailable；不能用 hard accuracy冒充 calibration。hard-only dev next-RSU accuracy为 `0.9578828970`；handoff-positive eligible target top-1 为 `0.0018226002`，暴露当前 target head 与旧 all-row accuracy 的严重口径差异。ETA 在 1,646 个 handoff-positive rows 上 MAE `0.6821579496`、RMSE `0.8192647841`、median absolute error `0.9687415`，interval coverage unavailable。

## 6. Selective prediction

confidence 定义为 `max(p_handoff, 1-p_handoff)`。threshold 固定从 calibration split 的 `0.00,0.05,...,1.00` 网格选择：在 minimum coverage `0.5` 下最小化 accepted-set Brier，tie 时提高 coverage、再选更低 threshold；不读取 reward。选择结果 `0.95`，calibration coverage `0.648015`、accepted Brier `0.005196`。evaluation coverage `0.743235`、abstention `0.256765`、accepted accuracy `0.992435`、accepted Brier `0.007450`、selective ECE `0.000951`。这是偏向高置信多数类的安全 gate 诊断，不代表 handoff recall 或 policy收益。

支持 reasons：`confidence_below_threshold`、`calibration_unavailable`、`snapshot_stale`、`prediction_expired`、`unseen_rsu_or_class`、`invalid_probability_simplex`、`insufficient_history`、`target_not_distinct`、`predictor_unavailable`，以及多头不一致时的 `handoff_target_unavailable`。abstained/unavailable 输出 mask=0 和 null legacy target；已接受的“无 handoff”保持 mask=1、handoff probability可见、target为 null，二者不混淆。

## 7. Staleness 与 runtime

默认 K=3、valid-for=3。固定诊断 K=`1/3/6/12`；最大模拟 age 分别 `0/2/5/11`，reuse count分别 `0/4/10/22`（诊断样本规模随 K 保证至少两个更新周期）。现有最小 runtime trace没有逐历史 snapshot 对齐的未来实现标签，因此 drift、按 staleness calibration、handoff/target/ETA error均为 `null`，不能伪造为零，也不运行 RL performance comparison。

启用时只有 accepted 且未过期 snapshot投影回旧 prediction字段；abstained/unavailable 使用明确 mask。snapshot ID/version/generated/as-of/age进入 step trace，episode summary收集 IDs/version/accepted steps，benchmark row保留同一 provenance。旧 feature flag关闭时不添加 causal snapshot字段，保持旧消费路径。

## 8. Decision Observation Trace

`decision_observation_trace_version=1.0.0` 在 `GymVecEnv.step()` 调用 core action之前记录：G08-alignable request ID、semantic map、raw/flat vector、index map、normalization、availability、snapshot provenance、actor/controller/critic visibility和action mask。selected action只在独立 `post_decision_outcome` 子段，未混入 semantic/flat pre-action字段；禁止 future label、service result、reward和oracle action。G12 最小 NGSIM trace含 5 个 request records。

## 9. G11 与 predictor-policy 风险

BurstGPT/Azure只可为未来 arrival/token profile提供外生来源；Qwen-Bailian/Mooncake只可为 KV/prefix reuse提供来源；HF qwen/cbow/bert只提供 size metadata。它们都不能校准 NGSIM handoff/next-RSU，本轮未下载 payload、未跨源输入 predictor，也不称 joint real trace。

v112 曾导致 policy `mean_total_reward=16.366`、mechanism realization `0.0`。原因边界包括 class imbalance、target eligible性能、旧 dev threshold复用、confidence语义不严和 planner/predictor消费错配。G12关闭默认入口并建立 gate，但没有证明 collapse 已被 policy层消除。

## 10. 稳定入口、artifact 与 G13 接口

运行：

```bash
.venv/bin/python scripts/audit_predictor_calibration.py \
  --run_id causal_predictor_snapshot_validation_20260819_g12_v1
```

输出：`artifacts/analysis/causal_predictor_snapshot_validation_20260819_g12_v1/`。包含 split/provenance、before/after metrics、reliability、risk-coverage、staleness、snapshot examples/validation、real pre-action trace、command log和integrity manifest。

G13 如启动，只能显式读取 `availability_mask=1` 的 snapshot，并必须另行冻结 actor/critic visibility、objective、reward和policy协议；G12 本身未实现或授权这些变化。
