# Calibrated workflow critic PopArt 开发 A/B 结果（2026-10-09）

## 审查身份

- `reviewed_at`: `2026-10-09`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-09`；本轮未新增文献检索
- `target_venue`: `CSCWD 2027`
- `artifact_run_id`: `calibrated_workflow_value_normalization_ab_20261009_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scientific_git_commit`: `858bc797e23b4e56051663f28d6fd7681f5ee77d`
- `evidence_level`: `E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY`
- `verdict`: `FALSIFIED_OR_NOT_PROMOTED`

本报告只读回收固定产物，不重训、不补 seed、不修改门槛。当前 36 个实例已全部暴露并用于 train/dev/regression/frozen-check；5 个 seed 不是 5 个独立真实 workload，`regression` / `frozen_check` 也不是 holdout。

## 结论

PopArt 明确改善了 critic 的原单位拟合和共享梯度尺度，但**不足以稳定改善服务**：

- raw-unit critic RMSE 在三种方法的 24-update 均值中分别下降 `27.43 / 29.13 / 28.33`；这不是用 normalized loss 变小冒充改善；
- value:policy gradient ratio 从 `740–974` 降至 `13–15`，mean global clip scale 从约 `.0016` 提高到 `.158–.213`；
- actor 的 KL/clip fraction 确实增大、entropy 下降，说明策略更新被放大，而不只是 critic 内部数值改变；
- SA 的 completion 与失败指标改善，但多数 completed 样本更慢且 recompute 增加；MAPPO 服务量基本不变；PPO completion/失败改善但 on-time completion 从 `.29` 降至 `.19`，触发方法级否决。

因此本轮否定的是“修正 critic 尺度足以共同改善服务”这一假设，不是否定 PopArt 对 critic 稳定性的作用，也不是证明 raw critic 更优。决策为：**PopArt 只保留为有限的训练稳定性设置，不宣称服务优势；raw critic 继续作为 canonical 参照。** 不自动进入 auxiliary、reward 或新结构搜索。

## 完整性、预算与身份

- supervisor terminal=`SCIENTIFIC_CHILD_COMPLETE`，return code `0`，wall=`293.159 s`；
- 30/30 cells：每 cell `1,440` steps、`24` updates、`192` optimizer steps；总计 `43,200 / 720 / 5,760`；
- evaluation rows=`600`，behavior rows=`5,293`；formal/holdout/download/model-generate/hyperparameter-search 均为 `0`；
- source artifact integrity 登记 166 个文件，逐文件 size/SHA-256 复核失败数 `0`；
- 120 个 checkpoint candidate 坐标唯一，等于 30 cells × updates `6/12/18/24`；candidate hash 全匹配；
- `reward_used=false`、`evaluation_or_holdout_used=false`，checkpoint 只用 4 个 dev 实例按预注册服务字典序选择；
- selected update 分布：`6:16`、`12:6`、`18:6`、`24:2`；checkpoint 和权重只保存在本地，未提交或上传；
- 36 个 raw source interval 字段齐全且两两不重叠。

两臂使用同一 service-aligned reward 公式、相同初始化 seed、instance order、交互/update/optimizer 预算。唯一科学变量是共享 critic 的 PopArt target/output normalization。

## 预注册否证门

| 门 | 实现结果 | 关键数值 | 解释 |
|---|---:|---|---|
| mechanism | PASS | paired median `Δ value:policy=-778.582`；`Δ EV=+0.002285` | critic scale 机制按实现口径兑现 |
| behavior | FAIL | current-missing action-4 probability `.444792→.446411`；raw argmax `.679167→.558491`；executed `.695833→.597484` | argmax/执行率下降，但预注册要求 probability 与 argmax 同时下降；probability 反而 `+0.001619` |
| behavior safety companions | 改善 | failure attempt `Δ=-.080909`；failure episode `Δ=-.070`；no-progress `Δ=-.106667` | 不能覆盖 probability 子门失败 |
| service / SA | PASS | completion `.87→1.00`；on-time `.29→.31` | SA 没有方法级 completion harm |
| service / MAPPO | PASS | completion `.88→.88`；on-time `.31→.31` | 主完成指标中性；failure attempt/no-progress 略差 |
| service / PPO | FAIL | completion `.93→.97`；on-time `.29→.19` | on-time `-0.10`，任一方法 no-harm 规则否决 |
| overall | FAIL | `FALSIFIED_OR_NOT_PROMOTED` | mechanism 通过不抵消 behavior 和 PPO service 失败 |

这里须区分两种结论：整体不是“所有方法都无改善”。SA 有明显服务改善，PPO 的 total completion/失败也改善；但共同候选按预注册规则由 PPO on-time harm 和 behavior probability 子门否决。

## 全方法主表

`C→P` 表示 raw-control → PopArt。时延、流量和 recompute 只在 completed samples 上比较，并与 coverage 同列；正 delta 表示 PopArt 代价更高。

| 方法 | completion | on-time | failure attempt | no-progress episode | completed coverage | completed elapsed Δs | completed MB Δ | completed recompute Δs | reward Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | `.87→1.00` | `.29→.31` | `.319→.170` | `.50→.31` | `.87→1.00` | `+11.60` | `-119.27` | `+7.59` | `+14.84` |
| controller-MAPPO | `.88→.88` | `.31→.31` | `.305→.307` | `.48→.49` | `.88→.88` | `+2.51` | `+30.74` | `+1.92` | `-0.13` |
| PPO | `.93→.97` | `.29→.19` | `.192→.097` | `.26→.12` | `.93→.97` | `+9.26` | `-136.54` | `+8.17` | `+2.57` |

不能把较低的 unconditional 流量或时延当成优势：两臂 completion coverage 不同。上表的 elapsed/MB/recompute 已限制到完成样本；即使如此，SA/PPO 的 pooled completed elapsed 和 recompute 仍上升。

## Method × seed 成对服务结果

failure 为 service-failure attempt rate；elapsed/MB/recompute 是 completed-only candidate-minus-control。

| method | seed | completion C→P | on-time C→P | failure C→P | no-progress C→P | coverage C→P | elapsed Δs | MB Δ | recompute Δs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sa_ghmappo | 7 | 0.85→1.00 | 0.35→0.30 | 0.36→0.11 | 0.65→0.10 | 0.85→1.00 | +14.62 | -157.96 | +11.43 |
| sa_ghmappo | 17 | 1.00→1.00 | 0.15→0.30 | 0.00→0.18 | 0.00→0.35 | 1.00→1.00 | -15.14 | -457.84 | -12.98 |
| sa_ghmappo | 29 | 0.75→1.00 | 0.35→0.35 | 0.39→0.16 | 0.60→0.35 | 0.75→1.00 | +26.61 | +149.97 | +19.06 |
| sa_ghmappo | 43 | 0.90→1.00 | 0.25→0.30 | 0.38→0.20 | 0.65→0.40 | 0.90→1.00 | +10.71 | +118.79 | +3.06 |
| sa_ghmappo | 61 | 0.85→1.00 | 0.35→0.30 | 0.36→0.18 | 0.60→0.35 | 0.85→1.00 | +28.69 | -190.12 | +23.59 |
| mappo | 7 | 0.85→0.85 | 0.35→0.35 | 0.36→0.36 | 0.60→0.60 | 0.85→0.85 | +0.00 | +0.00 | +0.00 |
| mappo | 17 | 0.85→0.85 | 0.35→0.35 | 0.36→0.36 | 0.60→0.65 | 0.85→0.85 | +12.97 | +159.12 | +9.95 |
| mappo | 29 | 0.85→0.85 | 0.35→0.35 | 0.36→0.36 | 0.60→0.60 | 0.85→0.85 | +0.00 | +0.00 | +0.00 |
| mappo | 43 | 0.85→0.85 | 0.35→0.35 | 0.36→0.36 | 0.60→0.60 | 0.85→0.85 | +0.00 | +0.00 | +0.00 |
| mappo | 61 | 1.00→1.00 | 0.15→0.15 | 0.00→0.00 | 0.00→0.00 | 1.00→1.00 | +0.00 | +0.00 | +0.00 |
| ppo | 7 | 0.90→0.85 | 0.35→0.35 | 0.35→0.36 | 0.60→0.60 | 0.90→0.85 | -2.32 | +13.28 | +1.56 |
| ppo | 17 | 1.00→1.00 | 0.45→0.15 | 0.05→0.00 | 0.05→0.00 | 1.00→1.00 | +31.18 | -435.01 | +35.37 |
| ppo | 29 | 1.00→1.00 | 0.15→0.15 | 0.00→0.00 | 0.00→0.00 | 1.00→1.00 | +0.00 | +0.00 | +0.00 |
| ppo | 43 | 1.00→1.00 | 0.15→0.15 | 0.00→0.00 | 0.00→0.00 | 1.00→1.00 | -15.87 | -512.22 | -23.85 |
| ppo | 61 | 0.75→1.00 | 0.35→0.15 | 0.39→0.00 | 0.65→0.00 | 0.75→1.00 | +33.74 | +324.62 | +27.62 |

跨 seed 方向并不统一：SA completion 为 `4 improve / 1 neutral`，但 on-time 为 `2 improve / 1 neutral / 2 worsen`；PPO completion 为 `1 improve / 3 neutral / 1 worsen`，on-time 为 `0 / 3 / 2`。MAPPO 五个 seed 的 completion/on-time 全部中性。seed 只能表示初始化敏感性，不能当独立 workload cluster。

## Critic 与实际 PPO 更新

以下均为 24 updates × 5 seeds 的 optimizer/update 记录均值；RMSE 和 EV 使用反归一化原始 reward 单位。

| 方法 | raw RMSE C→P | EV C→P | value:policy ratio C→P | mean clip scale C→P | globally clipped fraction C→P | actual KL C→P | policy clip fraction C→P | entropy C→P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | `62.76→35.33` | `.00004→.00876` | `740.41→13.45` | `.00156→.21278` | `1.000→.956` | `.00462→.00577` | `.0349→.0680` | `1.252→1.185` |
| controller-MAPPO | `66.61→37.48` | `.00119→.00868` | `973.78→15.30` | `.00161→.20948` | `1.000→.960` | `.00172→.00406` | `.0079→.0253` | `1.026→.996` |
| PPO | `65.10→36.77` | `.00108→.00530` | `832.95→14.52` | `.00162→.15785` | `1.000→.983` | `.00123→.00140` | `.0010→.0128` | `1.398→1.318` |

机制方向很清楚：15/15 cells 的 final raw RMSE 下降，15/15 的 value:policy ratio 下降，14/15 的 EV 上升；global clipping 仍常见，但压缩程度显著减轻。actor KL/clip 上升且 entropy 下降，说明 critic scale 修正改变了实际策略更新强度。服务并未共同跟随：这正是“critic 改善但不足以保证服务改善”的证据。

不能使用 normalized value loss 从约 `4,000` 降到 `2–3` 作跨臂拟合比较，因为单位不同；本报告只用 raw RMSE/EV 支持 critic 拟合改善。

## 无效行为门的细分

| 方法 | current-missing decisions C→P | action-4 probability C→P | raw argmax C→P | executed C→P | probability seed direction（改善/中性/恶化） |
|---|---:|---:|---:|---:|---:|
| SA-GHMAPPO | `351→224` | `.4841→.5323` | `.6353→.6250` | `.7265→.6250` | `2/0/3` |
| controller-MAPPO | `343→345` | `.5654→.5134` | `.8980→.6696` | `.8513→.7594` | `3/0/2` |
| PPO | `266→226` | `.2375→.2589` | `.4549→.3230` | `.4549→.3230` | `1/0/4` |

MAPPO 的 probability、argmax、executed 三项都下降；SA/PPO 的 argmax 与执行率下降，但 mean probability 上升。全局 probability 仅上升 `.001619`，却足以按冻结的 conjunction gate 否决。由于两臂访问到的 current-missing 状态数为 `960` 与 `795`，这还是 endogenous visitation 比较，不能解释成同一固定状态上的概率因果效应。

## Producer / consumer 口径问题

实现完整性没有 checkpoint、预算或 hash 缺失，但存在一个统计消费口径不对齐：

- service/behavior 使用 dev 选出的 checkpoint；30 cells 中 28 个 selected update 早于 24；
- mechanism gate 却固定消费每个 cell 的 **update 24 training batch**，而不是 selected checkpoint，也不是 selection split 上的冻结 probe；
- 因此可确认“训练终点的 critic 机制门通过”，但不能把它与被评价 checkpoint 的服务结果逐 cell 串成严格因果链。

该问题不改变 `FALSIFIED_OR_NOT_PROMOTED`：behavior 与 PPO service 已独立失败，修正 consumer 也不能据此自动晋级。原结果完整保留，本轮不修改 gate、不重算后补跑。若未来要判断为什么 PPO on-time 下降，唯一需要的新诊断证据是：对**实际 selected checkpoint**使用预先冻结、两臂共同的 current-missing states，联合记录 denormalized value error、action-4 probability/margin 和 advantage；在这条证据出现前，不提出新的算法候选。

## 决策与论文边界

1. PopArt 可作为 shared-critic 的有限稳定性设置记录；不能写成服务性能改进、SA 专属机制或原创贡献。
2. raw critic 保持 canonical 参照；本开发结果不替换正式 baseline，不进入 paper-ready 主表。
3. 不自动进行 auxiliary-target、reward、更多 seed 或结构搜索。
4. 既有 simple/two-step planner 的 model-based 能力边界、旧系统机制负结果和“SA 未稳定领先”结论均未被推翻。
5. 本结果是 exposed development evidence；没有独立 holdout、formal/support 或外层真实 workload 统计，paper-ready 仍为 `Unverifiable`。

机器可读复核包位于 `artifacts/analysis/calibrated_workflow_value_normalization_ab_review_20261009_v3/`，包含 method×seed 服务、行为、机制、checkpoint selection、逐门审查和独立完整性清单。
