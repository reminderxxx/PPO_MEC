# DriveLM 小样本资格核验

- reviewed_at: 2026-09-29
- literature_cutoff: 2026-09-29
- target_venue: IEEE TMC（研究目标，不代表资格）
- policy_version: tmc_review_policy_v3_20260621
- Git baseline: f3b92f2
- artifact_run_id: drivelm_demo_annotation_audit_20260929
- evidence_level: E2_SAMPLE_STRUCTURE_AUDITED；仅公开样本结构，不是性能或完整数据集验证

## 实际取得的证据

官方 demo URL：<https://raw.githubusercontent.com/OpenDriveLab/DriveLM/main/challenge/data/train_sample.json>。
本轮仅在内存读取公开 JSON，单次上限 2,000,000 bytes；没有下载图像、权重或完整数据包，没有保存原始问答。
URL 的 main 可变，未建立源 commit pin；以下字节 SHA 是本次可复查身份，两次读取一致。

| 项目 | 实际值 |
|---|---|
| bytes | 402384 |
| SHA-256 | fc306fb911516275513d34c5c4b0dca308b7edea30906b1278a598a7e319658e |
| scenes / frames / QA | 2 / 9 / 790 |
| perception / prediction / planning / behavior | 337 / 271 / 173 / 9 |
| C、con_up、con_down、cluster、layer | 每个字段均 present=790、null=790、missing=0、nonnull=0 |
| 推断的边 | 0（工具不推断） |

结论：**驾驶任务语义可复用；本样本不能直接提供观测执行 DAG。**
关系字段为空不等于任务不存在逻辑依赖，更不证明完整 DriveLM 所有版本均无关系。
[官方 v1.0 准备说明](https://raw.githubusercontent.com/OpenDriveLab/DriveLM/main/docs/data_prep_nus.md)
也将这些 context 字段描述为占位。不能把类别顺序直接冒充原始执行依赖。

## 数据包定位与缺口

[官方 challenge](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge)提供 demo 和模型基线说明。
[官方 Hugging Face 文件列表](https://huggingface.co/datasets/OpenDriveLab/DriveLM/tree/main)
本次 metadata 核验列出 v1_1_train_nus.json（192,961,829 bytes）、v1_1_val_nus_q_only.json
（9,860,626 bytes）、训练图像包（3,483,205,396 bytes）、验证图像包（704,864,335 bytes）。
当前列表未见旧文档链接的 v1_0_train_nus.json；不推断其历史不存在。以上大文件均未下载。

| 可作为输入 | 仍需独立补齐 |
|---|---|
| 场景/帧标识与驾驶问答类别 | 明确可执行任务模板、节点输入输出与依赖合法性 |
| 感知、预测、规划等语义 | 模型/adapter 实际绑定及计算、传输、状态大小测量 |
| 与原图关联的标注 | 合法图像使用、车辆轨迹与 RSU 映射及跨域映射误差 |
| 开源研究入口 | 派生包许可证及再分发资格，不默认继承代码许可证 |

## 下一步设计边界

优先形成“小型驾驶语义派生工作负载”，不是重新编造全套驾驶数据：

1. 选择有清晰输入输出的任务模板；逐条区分 source annotation、derived dependency、measured cost、assumed scenario。
2. 若按感知→预测→规划连边，明确这是研究者定义的执行模板；验证下游实际消费上游输出，不能仅改标签。
3. 先验证少量真实模型调用和成本，再扩展 workload。未实测的 base/adapter/KV/state 不填成观测事实。
4. DriveLM 与 NGSIM 不是同一车辆联合记录；组合只能称可追溯的半合成场景，不能宣称实采联合数据。
5. 质量验收独立于算法排名；保留无共享、无迁移及失败样本，不以 SA 优势决定数据取舍。

当前尚无新数据集 release、训练或 rollout；数据贡献仍为候选。
论文定位及来源池沿用 `vec_ai_workload_dataset_design_20260928.md` 的 D13，
与 `vec_ai_public_resource_recovery_20260928.md` 联合阅读。

## 复查入口与验证

`scripts/audit_drivelm_workflow_sample.py --input <local_sample.json>`（或 `--input -`）
只读、默认 2 MB 上限，输出源字节 hash、计数和关系字段覆盖；不联网、不写数据、不构图。
真实公开样本经 stdin 送入该 CLI，返回码 0，计数如上。

- `.venv/bin/python -B -m pytest tests/test_drivelm_workflow_sample_audit.py -q`：8 passed。
- `.venv/bin/python -B scripts/smoke_test.py`：6/6 toy 节点完成。
- 未运行全仓回归、图像解析、模型基线、完整语料或正式实验。
- 样本很小；关系、许可、任务成本和代表性未由本次核验闭环。
