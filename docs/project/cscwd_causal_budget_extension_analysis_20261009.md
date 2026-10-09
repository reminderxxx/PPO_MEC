# 因果强基线统一预算延长分析（2026-10-09）

## 结论

统一 4× 预算支持“短预算限制了部分 learned policy 的完成学习”，但不支持“SA 已稳定领先”或“继续加预算即可解决
服务质量”。SA 在全部已暴露 development 实例上的 completion 从 `87/100` 增至 `99/100`，五 seed 为
`3 positive / 2 unchanged / 0 negative`；同时 frozen on-time 仍为 `5/40→5/40`，共同完成样本的 frozen latency
增加 `13.39 s`，combined recompute 增加 `8.10 s`。PPO、MAPPO、DT 也获得不同收益，因此不是 SA 专属证据。

当前最值得补足的不是另一个 loss 或 auxiliary，而是 checkpoint-selection 的可识别性：现有 dev 只有 4 个已消费实例，
selected SA 的 dev on-time 全为 `.75`，但 frozen development 只有 `0--.25`。在没有新的 source-disjoint selection evidence
前，不冻结新算法候选，不用 regression/frozen 反选 checkpoint。

## 身份、恢复与边界

- scientific commit：`d25ebcded6b43b69b82bae825b035adc1d6f19c4`；analysis commit：
  `1ab26cfbb3485ae5242d46c000d6e7e3eeed8d0e`。
- scientific run：20 cells、115,200 environment steps、1,920 updates、15,360 optimizer steps、400 learned
  evaluation rows；119-file 原 integrity 全通过。
- supervisor 的历史 `FAIL` 与 `analysis_failure_receipt.json` 原样保留。失败点仅为 Python 3.9 不支持
  `zip(strict=True)`；不是训练失败。修复后输出到新 create-only analysis root，没有覆盖科学 run 或部分分析。
- 新分析执行 0 training、0 evaluation、0 checkpoint selection、0 holdout/formal；Popularity/two-step 的 40 行只按原
  SHA-256 引用，没有重评。
- `reviewed_at=2026-10-09T09:22:10Z`，`literature_cutoff=2026-10-09`，`target_venue=IEEE TMC`，
  `artifact_run_id=cscwd_causal_strong_baselines_budget_extension_analysis_20261009_v2`，
  `policy_version=docs/project/top_journal_review_policy.md@d25ebcd`，
  `evidence_level=L2_complete_development_artifact_no_independent_test`。

## 短预算到长预算

以下为五 seed、regression+frozen 合并的等权描述均值。seed 重用同一组 source windows，不能把 100 行当作 100 个独立
cluster，也不做显著性或泛化声明。

| method | completion | on-time | service failure | return Δ | transfer Δ MB | recompute Δ s | no-progress Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SA | .87→.99 | .28→.36 | .43→.17 | +5.491 | +10.18 | +10.32 | -1.22 |
| PPO | .95→1.00 | .25→.27 | .14→0 | +3.741 | -294.88 | -19.46 | -.37 |
| MAPPO | 1.00→1.00 | .26→.43 | .12→.10 | +2.259 | +264.03 | -19.29 | -.03 |
| DT | 1.00→1.00 | .41→.45 | .08→.03 | +.806 | -174.19 | +3.59 | -.08 |

分 split 的 SA 结果为：regression completion `.883→1.0`、on-time `.383→.517`、failure `.367→.15`；frozen
completion `.85→.975`、on-time `.125→.125`、failure `.525→.20`。因此 completion 改善跨 split，但 frozen on-time
没有改善。

跨 seed 稳定性与 tradeoff：

- SA completion 为 `3+/2=/0-`，是本轮最一致的新增信号；on-time 为 `3+/1=/1-`，failure 为
  `3 lower/1 same/1 higher`。seed 7 的 completion 不变，但 failure `.25→.45`、return `-4.85`，构成明确负例。
- PPO completion/failure 分别为 `1+/4=` 和 `1 lower/4 same`；on-time 为 `1+/3=/1-`。它达到 completion=1、
  failure=0，但 on-time=.27，不能称全面服务最优。
- MAPPO on-time 为 `3+/2=`，但 failure 有两 seed 上升，transfer 平均增加 264 MB。
- DT completion 始终为 1；on-time `3+/0=/2-`，failure 也有一 seed 上升。

## 共同完成实例与幸存者边界

全 400 行结果保存在 `paired_all_instance_diagnostics.csv`；共同完成子集另存 382 行，并始终与 coverage 一起解释：

| method | common completed | elapsed Δ s | transfer Δ MB | recompute Δ s |
|---|---:|---:|---:|---:|
| SA | 87 | +4.23 | -44.77 | +8.10 |
| PPO | 95 | -13.22 | -262.56 | -19.61 |
| MAPPO | 100 | -17.15 | +264.03 | -19.29 |
| DT | 100 | -1.44 | -174.19 | +3.59 |

SA frozen common subset为 34 行，elapsed `+13.39 s`、recompute `+17.65 s`；regression 53 行 elapsed `-1.65 s`。
新增完成的 12 个 SA 实例不在 common-completed latency 均值中，所以不能只展示该子集。

## 学习前缀、固定端点与选模

- 20/20 个 update-24 checkpoint 的文件 SHA-256 完全相同。短/长 run 前 1,440 步的 action、episode segment、
  termination/truncation、prediction provenance 全相同；CSV 数值最大差仅 `3.55e-15`，属于十进制浮点表示，且不影响
  checkpoint bytes。因此没有证据表明 dev 调用/RNG 时序改变了训练前缀。
- 固定 dev update 24→96 的 20 个 cell completion 全部不变。SA on-time 五 seed 平均 `+.15`，但 failure 平均 `+.10`；
  MAPPO/PPO/DT 的 on-time 分别 `+.40/+.15/+.10`。固定端点表与 selected 表分离。
- 长预算 selected updates：SA `[48,72,48,48,24]`、MAPPO `[48,96,72,48,72]`、PPO
  `[72,24,72,24,72]`、DT `[48,96,72,96,72]`。短预算候选是 `[6,12,18,24]`，故主结果只能解释为
  “预算 + 等比例 selection schedule”的联合效应。
- selected SA 的 dev completion 都为 1、on-time 都为 .75；对应 frozen on-time 仅 `0/.25/.125/.25/0`。
  regression/frozen 没有参与选择，也没有据此重选 checkpoint。

## 优化尺度与实际成本

SA 的五-seed mean value loss 从 update 24 的 `23.26` 降至 update 96 的 `7.40`，entropy `1.23→.63`，auxiliary loss
`1.82→1.20`；说明优化仍在推进，不是完全停滞。但 update 96 的固定 dev failure 比 update 24 平均高 `.10`，所以 loss
下降不能推出服务改善。

每方法均为 480 updates、3,840 optimizer steps。参数量/五 seed 累计训练耗时为 SA `165,320 / 437.42 s`、
MAPPO `39,176 / 127.44 s`、PPO `22,406 / 79.56 s`、DT `44,808 / 174.77 s`。

## 假设裁决与唯一后续优先级

- 支持：短训练曝光是 SA/PPO completion 不足的一个可干预因素；长预算也改善 MAPPO/DT 的部分 on-time/cost 指标。
- 否定：没有 SA 稳定领先；没有“更多预算普遍改善所有服务指标”；没有 auxiliary 或新 critic 改动的新增优先证据。
- 未定位：seed 7 SA 的 failure/return 退化，以及小 dev selector 对 frozen on-time 的巨大乐观差，仍不能区分 policy
  stochasticity、状态覆盖不足和 selector sampling noise。

唯一后续优先级是先建立新的 source-disjoint、预冻结 checkpoint-selection development cohort，并保持训练、候选集合和
最终检查完全不变，检验选模稳定性；停止条件是新 selection cohort 与现有 4-window dev 对候选排序不一致或 selected
checkpoint 在预先声明的服务指标上不能跨 seed 保持方向。该任务是证据/协议补足，不是新算法训练；本轮不自动启动。
