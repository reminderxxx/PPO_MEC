# 驾驶任务派生工作流：小样转换与防泄漏边界

- reviewed_at / literature_cutoff: 2026-09-29
- target_venue: IEEE TMC（目标，非达标判断）
- policy_version: tmc_review_policy_v3_20260621
- git_base: 4d0e93e
- artifact_run_id: drivelm_workflow_preview_20260929
- evidence_level: 小样转换与接口测试直接证据；AI任务质量、性能及新数据贡献 UNVERIFIED

## 已实现且实际运行

在 `scripts/audit_drivelm_workflow_sample.py` 增加显式 `--preview` 模式；默认审计模式不变。
该模式仅输出 scene/frame/category/index 引用与派生模板，不输出原问答或执行模型。
输入身份见 `drivelm_sample_qualification_20260929.md`；真实402384-byte demo重新取回并验证同一SHA。
内存运行公共CLI产出91897-byte预览，stdout SHA：
`97f217e82189845fb6e8deab18f680adbda644281445e01c7027dfa8a1327ce1`。
原始标注及完整预览均未落盘。简要机器记录位于
`artifacts/analysis/drivelm_workflow_preview_20260929/validation_summary.json`。

- 全部9帧均保留，没有基于答案、策略表现或完成率筛选；9个完整四阶段模板。
- 共36个节点、790条问题引用；每帧采用所有先前阶段上下文，产生6条边，共54条。
- 边统一标为 `researcher_defined_all_prior_stage_context_not_observed`；不是发现的原始DAG。
- 36次实际输入组装检查使用明确的synthetic测试输出；下游接收到先前输出，不读取参考A或key_object_infos。
- 模型、adapter、latency、state bytes保持null；图像binding unresolved、runtime_eligible=false。
- 未改现有WorkflowNode、Alibaba provider、环境或训练消费端；它是独立preview，不可强行用于正式矩阵。

## 节点的拟议输入输出

| 阶段 | 输入 | 输出 |
|---|---|---|
| perception | 合法图像binding＋本阶段问题 | 模型生成的感知回答 |
| prediction | 同帧图像/问题＋生成的感知回答 | 模型生成的预测回答 |
| planning | 同帧图像/问题＋生成的感知与预测回答 | 模型生成的规划解释 |
| behavior | 同帧图像/问题＋前三阶段生成回答 | 模型生成的行为解释 |

每节点当前代表一组问题，不代表一次等成本模型调用。真实执行须记录每条问题调用/批次、token、耗时、
I/O与失败。behavior仅为离线解释任务，不向真实车辆发送控制指令。

`build_stage_input`按精确源hash和scene/frame核验，要求前驱完整且无未来节点；检查生成回答数量和类型。
它只能证明输入组装，不认证调用者输出是否真正来自模型。未来模型runner必须另记录原始输出及调用身份。
原问题本身可能包含对象提示或上下文；不复制A不等于已证明所有标注信息无泄漏，仍需人工样本审查。

## 必须做的价值检验，不预设串联获胜

下一步先完成公开模型/图像的许可、精确版本、下载量及隔离环境选择，再进行有界推理。
采用同一图像和问题，对比独立回答与带生成上游上下文回答，记录质量、错误传播、token和时间。
不能把参考答案作为正常上游输入；若另做oracle诊断，必须单独标注且不得与正常结果合并。
必须保留任务失败、无收益和额外上下文成本；若依赖没有价值，缩减模板，而非选择有利样本制造优势。

共享base、真实adapter、KV和workflow-state仍需要实际绑定与测量。prompt不同不等于adapter不同；
此预览不新增模型共享或迁移收益证据。跨DriveLM/NGSIM关联只能标为半合成，不能伪称同源车辆记录。
九帧来自两个场景，不能称九个独立实验cluster，也不能恢复已消费holdout。

## 验证与限制

- `.venv/bin/python -B -m pytest tests/test_drivelm_workflow_sample_audit.py tests/test_drivelm_workflow_preview.py -q`：19 passed。
- `.venv/bin/python -B scripts/smoke_test.py`：6/6节点完成。
- 真demo通过公共 `--input - --preview`，全部36个stage输入组装通过。
- `git diff --check`通过；未跑全仓、图像解析、模型推理、训练或正式实验。
- 新数据集尚未release；来源许可、语义有效性、成本校准与独立代表性仍未闭环。
