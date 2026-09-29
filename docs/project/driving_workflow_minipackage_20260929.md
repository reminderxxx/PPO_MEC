# 首个驾驶 AI 工作流引用包：复用检索、实现与边界

- reviewed_at / literature_cutoff: 2026-09-29
- target_venue: IEEE TMC（目标，不是资格）
- policy_version: tmc_review_policy_v3_20260621
- git_base: 352835b
- artifact_run_id: driving_workflow_reference_20260929_v1
- evidence_level: E2_ARTIFACT_AUDITED（仅引用包与接口）；性能 E0_UNAVAILABLE
- status: REFERENCE_PACKAGE_NOT_INFERENCE_VALIDATED
- dataset_release=false; novelty=UNVERIFIED; actual_model_calls=0

## 公开资源检索结论

不能声称网上没有。此次按公开关键词查看作者/官方入口，而不是上传项目材料：

| 资源 | 已确认可参考内容 | 本任务决策 |
|---|---|---|
| [DriveLM](https://github.com/OpenDriveLab/DriveLM) | 驾驶图QA、数据及基线入口 | 首选其已核验demo；本地demo关系字段为空，不冒充原始执行DAG |
| [DriveVLM](https://tsinghua-mars-lab.github.io/DriveVLM/) | 场景描述、分析和层级规划，官方页标CoRL 2024 | 参考任务划分，不声称SUP-AD已下载或有缓存迁移实测 |
| [V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq) | 车路协同感知与预测序列及示例下载 | 后续空间协同来源候选；本轮不换源或大包下载 |
| [V2X-QA](https://github.com/junwei0001/V2X-QA) | 车端/路端/协同视角问答；原图另按上游获取 | 后续候选，未取得payload或核验正式venue |
| [DriveBench](https://drive-bench.github.io/) | 视觉可靠性、文本线索与评价偏差诊断 | 扩展前须检查视觉依据，非空文本不等于正确 |

本次有界检索未确认可直接复用的“驾驶任务执行依赖+模型缓存+状态迁移成本”联合包；
这不是穷尽检索或新颖性证明。其余新发现的车路数据候选已登记文献表，未下载验收。

## 已生成什么

入口 `scripts/build_driving_workflow_package.py` 复用既有源审计、计划及输入consumer，
不新增科学运行平台。固定DriveLM revision、JSON size/SHA；只在内存读取官方标注，
先固定scene/frame字典序及各类Q0，再形成一个场景的四阶段、两臂、8调用计划。
六相机绑定官方allowlist，记录Git blob身份；图像SHA256与尺寸必须等实际下载后测量。

本地包：`artifacts/datasets/driving_workflow_reference_20260929_v1/`

- `workflow.json`：场景/问题引用、六相机引用、8任务、6条派生上下文边、候选模型身份。
- `data_card.md`：来源、派生、许可、用途及未覆盖边界。
- `interface_validation.json`：8次synthetic输入传递，model_calls=0。
- `integrity_manifest.json`：前三文件exact membership、size/SHA；自排除manifest。

不包含原图、原问题/答案文本、生成输出或模型权重。consumer从精确source重建包，
拒绝包改动、跨场景task、缺失/多余/跨臂前驱。输入不带参考A/key_object_infos；
原Q可能含标注线索，不能因此宣布信息泄漏已完全排除。
依赖边是研究者定义的假设，不是观测到的因果关系。无adapter、移动映射、迁移成本实测。

## 已完成的资源与验证

- 模型下载现在完成：13 files、1,019,893,574 bytes；重新逐文件散列，mismatch=0。
- 下载回执：`data/raw/ai_workflow_pilot/model_a7da5b9_20260929/download_receipt.json`。
- 56项包/plan/preview/source测试通过，真实demo公共构建成功，发布后inventory重读通过。
- `scripts/smoke_test.py`完成6/6 toy节点；新增脚本/测试AST解析、`git diff --check`通过。
- 图像nuScenes条款接受仍待用户确认，本轮再次发问；未代接受、未下载图像。
- 模型没有实载推理；不能用下载完成替代工作流运行成功。

## 成功后才扩展

1. 按许可取得六张图，保留原相机语义/坐标；独立环境实载候选模型。
2. 先最多2次兼容性调用，再本包8次真实调用，greedy、64新token上限；失败留存，不挑题追求胜出。
3. 保存实际输出、token、单调耗时、截断/错误；参考答案独立于输入。
4. 盲化臂标签核查每个回答是否响应问题、有图像依据，以及前序错误是否传播。
   单场景只能做可行性诊断，不能检验总体准确率或系统优势；顺序/冷启动混杂需另控。
5. 若质量不可用或前序依赖无意义，停止扩容并诊断；若可用，再测输出状态序列化/恢复等价性与成本。
6. 再扩至已有两场景计划并加入文本线索/视觉依赖对照；预先定义扩展质量标准，不依算法胜负筛场景。
7. 最后接移动/RSU与缓存迁移机制；NGSIM与DriveLM为半合成关联，不是同车同刻联合观测。

复用原数据及统一格式本身不是新数据集贡献。若未来完成实测校准、代表性与独立验证、
可复现生成器和许可清晰的发布包，可再审查数据贡献；否则作为实验工具。

## 命令与验证范围

```bash
.venv/bin/python -B scripts/build_driving_workflow_package.py \
  --fetch-official-sample --output artifacts/datasets/driving_workflow_reference_20260929_v1
.venv/bin/python -B -m pytest tests/test_driving_workflow_package.py tests/test_drivelm_pilot_plan.py tests/test_drivelm_workflow_preview.py tests/test_drivelm_workflow_sample_audit.py -q
```

已存在包不可覆盖；重建测试使用新的输出路径。本轮未运行全仓测试、科学推理、训练、
formal或旧holdout。模型完整性不构成输入许可或驾驶质量证明。
