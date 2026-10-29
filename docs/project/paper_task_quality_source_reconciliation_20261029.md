# 任务质量来源补充核对（2026-10-29）

- `reviewed_at`: 2026-10-29，Asia/Shanghai。
- `literature_cutoff`: 本轮不评价novelty；旧ALPR审查截止2026-10-04，adapter卡为本地冻结版本。
- `target_venue`: CSCWD2027；TMC仅内部证据规范。
- `artifact_run_id`: `quality_source_reconciliation_20261029_v1`。
- `policy_version`: `tmc_review_policy_v3_20260621`。
- `git_commit`: `6a972a215f0dd3cec2b7251f3353e9340b132885`（本轮核对起点）。
- `evidence_level`: 保存的既有局部审查复用，本轮不重验原始推理；不新增E2/E3任务质量认证。
- `verdict`: 有已消费ALPR描述性负结果；新增ALPR/Helmet质量任务仍缺合格来源。

## 已补入的历史证据

用户回复“继续补”后，定向核对已有来源登记，发现此前质量准备未列入2026-10-04的ALPR公开样本对照。
因此不能笼统说“从未有任何任务质量观察”；准确说法是：**已有旧ALPR负结果，没有满足新Q合同的完整独立质量证据。**
本轮没有读取私有标签、图像或原始预测，没有重新推理、运行旧审计或重新选样。

| 已消费分组 | adapter exact | base exact | adapter/base micro CER |
|---|---:|---:|---:|
| 旧development，4例 | 0/4 | 3/4 | 0.178571 / 0.035714 |
| 旧locked_check，8例 | 4/8 | 4/8 | 0.245283 / 0.245283 |

旧`paired_comparison_redacted.json`明确`exploratory=true`、`new_independent_test=false`。
`locked_check`是原包字段名，不能改称当前strict holdout；两组均为已消费旧图，不与新样本相加制造独立性。
旧独立审查的决定`DO_NOT_PROMOTE_THIS_ADAPTER_FOR_CACHE_BENEFIT_STUDY`保持，不支持adapter优于base。
以上是复用保存结果，不是本轮重算或新确认模型实际输出。

## 来源、许可与覆盖缺口

旧登记包含100张图、四地区各25张，已选12例；车辆/视频身份、训练重叠均未核验，去重只提供proxy。
来源声明`CC-BY-NC-ND-4.0`、本地非商业分析且不再分发，但旧接受记录明确底层图像权利未独立建立。
本轮不作新的法律许可认定，也不改变旧原件、授权或失败记录。

两份本地冻结adapter README均写训练集未知，结果列表为空；模型名和Apache模型许可不能认证图像权利、训练独立性或驾驶质量。
公开固定revision页面本轮访问未成功；没有将失败访问记为新来源认证，也没有下载图像、模型或其他资源。

[原Q合同](paper_local_cost_quality_support_contract_20261029.md)要求ALPR三清晰牌+三无牌、Helmet三戴盔+三未戴盔且指定人员ROI，
标签在模型输出前冻结。现有记录没有覆盖无牌/Helmet样本及相应独立标签；不能把技术图表suffix或DriveLM QA补位。
旧ALPR的GT还未视觉重标，训练重叠、独立active-tensor指纹等限制均原样保留。

## 下一步和停止边界

当前具体所缺是**本地图像与标签的路径，或可核验数据来源名称**，以及允许该研究用途的条款/标签来源。
用户“继续补”确认了推进意图，没有提供这些数据身份；本轮不据此推导新增下载、模型评价或训练预算。
得到来源后先核许可适用性、图像/标签绑定、标注独立性、任务覆盖和历史暴露，再冻结精确Q软件与执行合同；
不要求结果必须为正，不更换困难样本，不把旧12图当新独立测试。小样本Q即便通过仍只支持有限诊断。
如本地没有来源，应先另立具体公开数据获取方案，明确来源、许可、样本与字节上限，再进入数据准备。
不恢复空闲提醒、不生成占位实验，也不追加主方法候选。

## 本轮交付与核对

私有包：`/Users/howen/Projects/PPO_MEC/artifacts/analysis/paper_evidence_strengthening_20261029_v1/quality_source_reconciliation_20261029_v1/`。
其中保存回复、事前上限、来源SHA、独立复核、条件性结果和文档验证；旧ALPR原包不改写。
独立复核只读5份指定小文件共21,155B，无图像、标签或checkpoint读取，无审计重跑。
验证仅为保存摘要之间的数值/标志一致性、两份冻结卡的字段、文档本地链接、`git diff --check`及主目录七文件保护核对。
新增训练/拟合/模型调用/环境评价/真实数据下载均0；不冒充真实质量测试完成。
原main、head候选拒绝、418未知隔离及formal/hidden认证0不变，`paper_goal_complete=false`。

既有证据指针：

- `M/artifacts/analysis/alpr_public_sample_20261004_v1/{design,data_acceptance}.json`。
- `M/artifacts/analysis/alpr_public_base_comparison_20261004_v1/paired_comparison_redacted.json`。
- `M/artifacts/analysis/alpr_public_pair_review_20261004_v1/review_redacted.json`。
- [既有2026-10-05结果与负例](mechanism_evidence_closure_results_20261005.md)。

`M=/Users/howen/Projects/PPO_MEC`。报告未包含牌照标签、图像或私有推理原件。
