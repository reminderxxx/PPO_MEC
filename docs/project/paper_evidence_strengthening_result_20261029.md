# 两项强化任务的固定结果与证据边界（2026-10-29）

两项任务均按[事前合同](paper_evidence_strengthening_contract_20261029.md)单次完成，自动消费者和全部六次预声明重放完成。
完整initial补证修正了对学习增量的外推；head等权候选未通过效益门，保留原main，不追加训练、换seed或选中间峰值。
本页是结果摘要，私有原件、checkpoint、完整审查和图件不进入Git；整体`paper_goal_complete=false`。

- reviewed_at: 2026-10-29
- literature_cutoff: 2026-10-29；复用已核12篇一手实验段矩阵
- target_venue: CSCWD2027；TMC/ToN及顶会仅内部证据对标
- policy_version: `tmc_review_policy_v3_20260621`
- git_commit: `fc3f910c18797197b3048d758492925b3308ff3f`
- artifact_run_id: `paper_evidence_strengthening_20261029_v1`、`paper_head_balance_campaign_20261029_v1`
- evidence_level: 局部原件审计及指定六集重放；整篇formal/holdout/support仍为`UNVERIFIED`

## A：完整12条件的初始/最终归因

只新增缺少的45条initial评价与3条初始重放，旧15条probe、60条final和12条gain仅只读引用。
实际新增17,224物理帧、562.372秒，训练/拟合为0，三重放逐帧奖励与控制摘要匹配。

| 全部五seed×12条件等权均值 | 原initial | 固定ep96 | 训练增量 |
|---|---:|---:|---:|
| native reward | 190.616180 | 189.499082 | −1.117099 |
| 完成请求数 | 15.366667 | 15.300000 | −0.066667 |
| 状态传输字节 | 25,543.483 | 27,672.117 | +2,128.633 |
| state-transfer link frames | 27.700000 | 29.816667 | +2.116667 |
| handoff失败事件数 | 6.800000 | 7.066667 | +0.266667 |

4/5 seed的12条件奖励均值回退，38/60奖励配对及8/60完成请求配对为负。
先前固定三个probe的`+2.041432`仍对原probe池成立，不能外推全12条件，也不能称稳定学习收敛。
较高最终分数、先验/支持差异与实际训练收益必须分开；当前数据没有识别唯一的回退原因。

## B：head等权候选拒绝

五seed全部完成96训练episode、48评价和13checkpoint，共480训练、240评价、65checkpoint、120次update。
保留固定ep96及全部负seed，三条final独立重放的奖励与控制摘要匹配。

| 固定全12条件，五seed等权 | 原main最终 | head等权候选最终 | 候选−原main |
|---|---:|---:|---:|
| native reward | 189.499082 | 188.050572 | −1.448509 |
| 完成请求数 | 15.300000 | 15.133333 | −0.166667 |
| 状态传输字节 | 27,672.117 | 32,513.933 | +4,841.817 |
| state-transfer link frames | 29.816667 | 35.583333 | +5.766667 |
| handoff失败事件数 | 7.066667 | 10.383333 | +3.316667 |

相对原main，seed7/17分别−5.184823/−4.500870，seed27/37/47分别+0.976464/+0.848843/+0.617840。
只有3/5 seed不退步，低于事前4/5门；六项效益检查全部失败。完整性通过不改变拒绝结论。
这里的handoff指标是v3 `service_target_changed`且状态为`state_restore/state_link_queue`的重复事件，不能改称独立失败请求率。

相对A完整initial参照，候选最终奖励差为−2.565608。这是共享初始策略矩阵的描述性参照：
update-only干预、相同初始化与已测15个initial probe匹配支持使用该参照，未再执行候选缺少的45条initial，
不能声称候选全12初始行为已另获独立实测。

训练183,362帧、评价90,553帧、重放1,147帧，共275,062；A+B共292,286，小于370,080硬上限。
候选parent历时6,408.705秒。相同外生会话不等于相同实际帧、决策、梯度或compute，主与surrogate优化器分别计费。

## 图件与复用规则

本地原件根目录：

```text
artifacts/analysis/paper_evidence_strengthening_20261029_v1/
  initial_execution/audit/{audit,summary,complete}.json
  initial_execution/audit/attribution.csv
  initial_result_review.md
  head_balance_result_review.md
  head_balance_delivery_verification.json
  head_balance_attribution.csv
artifacts/analysis/paper_head_balance_campaign_20261029_v1/collection/
  audit.json
  complete.json
  paired_reward_learning_curve.{png,pdf,svg}
  paired_curve_points.csv
  final60_paired.csv
  resource_bill.csv
  caption.txt
```

图件、CSV、caption与既有audit的SHA逐项一致；PNG已视觉检查。
图中两方法使用相同三个已消费probe、全部五seed、均值±1样本标准差（不是CI）、不平滑。
候选probe最终223.621885略高于原223.365277，不能代替全12条件的拒绝结论。
复用现存audit/complete，不再次运行collector、原训练、旧audit或任何重放。

## 继续补证的范围

已冻结followup queue的四项准备分别交付：

- [严格stream统计](paper_stream_statistics_contract_20261029.md)：v2的8项检查通过，闭区间原生tick、全局配对seed重采样、整数状态费用、非均匀权重及单outer退化边界已核。
  原v1虽功能检查通过，两轮物化64实例超过32上限，预算失败原样保留；v2另立8实例合同，累计72。合成实现资格不是实际独立性或95%覆盖保证。
- [原生provider/几何合同](paper_tgsim_native_provider_geometry_contract_20261029.md)：16项保存元数据一致性检查通过，新增轨迹/几何行读取为0。
  480个0.1秒样本只有479个真实区间；27容量窗不是27独立采集，未实现provider或配置真实RSU。
- [本地成本/质量准备](paper_local_cost_quality_support_contract_20261029.md)：40项合同核对完成，没有模型或传输调用。
  驾驶任务质量缺适用、获许可且带独立groundtruth的固定图像；技术suffix不能补位。
- [公平领域对照映射](paper_fair_domain_comparator_contract_20261029.md)：v2通过独立只读审查，明确最终autoregressive mask、cache/state先后和原生score；没有实现、训练或声称外部论文忠实复现。

本轮主方法优化分支已经结束。这些准备不授予新训练/环境评价/拟合/下载或hidden预算。
依据用户自主补证授权，另立[本地组件成本软件任务](paper_local_adapter_cost_implementation_20261029.md)，
先审查并验收生产/独立消费端，再以clean source、当前硬件及精确plan/grant界定一次技术成本测量。
该后续任务与候选失败分开，不能追加候选、改费用或把技术token保真写成真实任务质量。

418未知历史继续隔离，独立formal/hidden认证仍为0；27个TGSIM容量候选不是27独立采集。
独立时间/DAG、公平外部对照、模块必要性、真实任务质量及无线/RSU成本仍缺。
奖励图、技术adapter恢复和轨迹RMSE不能补位，不晋级canonical、不改论文主表或老师固定稿位。
