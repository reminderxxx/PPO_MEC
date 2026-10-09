# Calibrated workflow critic PopArt 开发 A/B 执行协议（2026-10-09）

## 范围与证据边界

本轮承接 `calibrated_workflow_service_reward_learning_diagnosis_20261009.md` 的只读结论，只检验一个可证伪候选：对 SA-GHMAPPO、controller-level MAPPO 和 PPO 的共享 PPO critic 同等启用 PopArt target/output normalization。它是对 value 学习尺度的工程纠正候选，不是 SA 专属技巧，也不作为算法创新。

当前 36 个实例全部是 development data。`regression` 和 `frozen_check` 仅作为暴露的开发 falsification split，不是 formal 或 holdout；不得读取或复用旧 holdout。checkpoint 只保存在本地 run 目录，不提交、不上传。

## 唯一变量

- control：raw critic return target；
- candidate：`(raw_return - running_mean) / running_std`；
- running statistics 只由当前训练 batch 的 raw return 更新；dev、regression、frozen_check 评价均不更新；
- bootstrap、GAE、actor advantage 保持原始 reward 单位；只有 critic loss 使用 normalized target/output；
- 更新统计时只补偿最后一个 scalar value affine head，使所有探针的 denormalized prediction 在 `1e-6` 内不变；
- Adam 的 `exp_avg`、`exp_avg_sq` 和 AMSGrad `max_exp_avg_sq` 随 affine scale 同步变换；
- value clipping 明确关闭；actor、auxiliary、reward、observation/action contract、network topology、value coefficient 和 global gradient clipping 均不变；
- 非有限 target 或绝对值超过 `1e20` 时 fail-fast；`min_std=1.0`；
- checkpoint 同时保存网络、optimizer 和 PopArt moments；旧 disabled checkpoint 继续按旧语义加载，enabled agent 不允许静默加载缺少 PopArt state 的旧 checkpoint。

## 固定预算与选模

- 2 arms × 3 methods × 5 seeds（`7,17,29,43,61`）=`30` cells；
- 每 cell `1,440` environment steps、`24` update opportunities、每次 `60` transitions；
- 每 update `4` PPO epochs、minibatch `32`，固定 `8` optimizer steps；
- 总计 `43,200` environment steps、`720` updates、`5,760` optimizer steps；
- checkpoint candidates 固定为 update `6/12/18/24`；只用 4 个 dev 实例按预注册服务指标字典序选择，reward 不参与；
- 不扫描超参数、不追加 seed、不重试、不补跑；科学进程 wall-clock cap 为 `7,200 s`。

训练 batch 可以在 update 边界切分尚未完成的 episode，但每个 segment 独立 finalize，并用该 segment 最后一个真实 next observation 的 denormalized value bootstrap；environment termination 使用零 bootstrap。episode reset 后的 observation 不得进入前一 segment，GAE 不跨 episode 串接。

## 记录与门槛

逐 transition 记录 raw reward、raw value target、denormalized old value、raw advantage、old policy/env-action log probability 和 action probabilities。逐 optimizer step 记录全部 loss、KL、clip fraction、entropy、policy/value/auxiliary gradient norm、pre-clip total norm 和 clip scale。逐 update 记录 raw/normalized target、raw/normalized advantage、denormalized critic RMSE、explained variance、PopArt moments 与 update identity。

候选只有同时满足下列 development falsification gates 才能保留：

1. paired median value-to-policy gradient ratio 下降且 explained variance 上升；
2. current-model-missing 条件下 action 4 的 mean probability 与 raw argmax rate 都下降，且 service-failure attempt/episode rate 与连续无进展不增加；
3. 三种方法各自的 completion 和 on-time completion 均不下降，并且至少一个 completion 或 failure 指标严格改善。

任一 gate 不通过即结论为 `FALSIFIED_OR_NOT_PROMOTED`，不得自动进入 auxiliary、decision-transformer、更多 seed 或延长预算。

## 身份与执行入口

- frozen design：`configs/experiment/calibrated_workflow_value_normalization_ab_v1.json`；
- 授权 record：`configs/experiment/calibrated_workflow_value_normalization_ab_authorized_v1.json`；
- runner：`scripts/run_calibrated_workflow_value_normalization_ab.py`；
- create-only preflight：`scripts/preflight_calibrated_workflow_value_normalization_ab.py`；
- source interval 必须含 `source_segment_id/time_index_start/time_index_end/frame_offset/window_length`，36 个 raw interval 必须两两不重叠；
- 实际 Git commit、配置 SHA-256、Python/Torch/NumPy identity 和真实预算由 run manifest 固化。

预检通过后仅允许一次后台科学启动。若启动后仍在运行，执行方只交付 PID、日志和 create-only run root，不持续轮询；若已结束，只读一次 completion/failure receipt。
