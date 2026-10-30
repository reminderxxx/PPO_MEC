# Helmet6 paired quality pilot：完整执行、质量门失败（2026-10-30）

reviewed_at=2026-10-30；literature_cutoff=2026-10-29既有矩阵（本轮无新检索）；target_venue=CSCWD2027（TMC仅内部证据规范）；policy_version=tmc_review_policy_v3_20260621；evidence_level=E2_ARTIFACT_AUDITED，仅本固定fixture。
artifact_run_id=`paper_evidence_strengthening_20261029_v1/helmet_quality_pilot_20261030_v1/run`；实际执行Git=`0292c959f5e14eae6a2523c229b29372bc806da8`。
计划SHA=`93cc89313480e417a3949d3a4136853bd380a38d49e885b37fb677bc427bc5d1`。

## 实际结果

按事前固定的6张不同原图、Helmet/NoHelmet各3个原生ROI，共执行12次真实生成。
两臂实际视觉tensor、prompt、generation配置配对一致；Helmet真实激活及全部LoRA参数与原文件逐tensor匹配。
自动独立audit通过，完整保留12份input/returned/new-token IDs、原始文本及判分结果；审计未重跑。

| 方法 | 严格正确/总数 | 格式错 | 本固定fixture质量门 |
|---|---:|---:|---|
| base | 0/6 | 6/6 | FAIL |
| base + Helmet adapter | 0/6 | 6/6 | FAIL |

主分只strip外层空白后精确匹配两个标签；句号、小写及拼写错误都按原规则判错，未做事后清洗救分。
Helmet的6份原始回答均是带句号的NO_HELMET内容，其中包含3个Helmet正例，故还存在本批正例识别缺口。
两个0分只表示双方均未通过这个严格门；parsed_label全null也会令解析后的disagreement为0，不能由此推断原始生成行为相同。
全12原始文本及逐例类别/错误见私有CSV/JSON。没有换样本、改prompt、补测或挑选正例。

## 完整账单与交付

2个新模型进程，2processor load、2base load、1adapter load、1activation、12generate；每次最多32新token。
parent含执行及自动audit约217.449秒，claim至consumer结束约217.134秒；两worker各约107.31秒，consumer约2.08秒。
这是一轮质量worker时间（含载入/核验/预处理/生成/退出），不能当重复的组件延迟或无线/RSU开销估计。
新HTTP、训练、拟合、环境episode、ALPR、hidden均0；128MiB/16MiB上限核验通过。
全部旧科学、旧T和输入失败原件保留，292286旧科学帧等账单未重置。

私有包在主目录`artifacts/analysis/paper_evidence_strengthening_20261029_v1/helmet_quality_pilot_20261030_v1/`：
`audit.json`与`run/terminal.json`匹配；`helmet_quality_cases.csv`、`helmet_quality_raw_results.json`、`helmet_quality_resources.csv`、`delivery_verification.json`保存实际结果和资源。
实际audit SHA=`f4a3569b22488d49cf91db1438af79d07b409590ac26989e1ec23c3114de7699`。
真实图片、模型权重、逐例原件及私有review不进入Git。

## 证据边界与下一步

本轮补上了“真实视觉输入、外部原生类别、base/adapter成对任务输出”的可追溯负结果，尚未提供任务质量收益。
严格格式失败和分类内容错误需要分别解释；不能通过事后改parser、改prompt或排除负例把本轮改为通过。
只凭n6无法估计总体准确率、显著性、跨场景泛化或训练独立性；原生COCO标签并不认证模型未见过这些图片。
当前框是外部GT，检测/定位质量没有测量。裁剪分类是否匹配该adapter原训练任务，须依据可验证的任务/训练说明另立资格；缺说明时明确UNVERIFIED。
后续独立测试集/ALPR文字与NONE样本、原任务/输出协议、训练重叠来源和公平外部对照仍需补齐。
保留该adapter作为未通过质量资格的技术资源，不把其技术加载/恢复成功解释为真实任务收益；不晋级canonical、不改论文或老师稿位。
`quality_fixture_qualified=false`，`paper_goal_complete=false`。

独立交付复核已接受audit/terminal/parent/CSV/raw及12份receipt+claim匹配，0模型/HTTP/读图/旧audit重跑。固定base README声明通用VQA/caption用途；Helmet README的dataset为unknown、results为空，未提供原始prompt、标签输出格式或ROI训练分布。下一项真实缺口是原任务推理协议和训练数据谱系，不能在这6个已消费例上筛选prompt来替代。独立原件为私有`independent_result_review.json`，不上传。
