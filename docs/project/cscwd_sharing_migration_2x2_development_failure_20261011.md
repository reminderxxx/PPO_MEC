# CSCWD sharing × migration 首次四臂开发执行失败回执

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11；本轮未检索新论文或评定 novelty
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `cscwd_sharing_migration_2x2_development_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`（唯一启动版本）: `f6d89bee6ba6d65430aba2b61ce301cb69a559c7`，启动前已 push
- `evidence_level`: `E1_DOCUMENTED` 用于论文主张；工程失败有本地 command log 与 create-only receipt
- `verdict`: **`PRESTEP_FAILED_NO_RETRY`；0/96 环境步、0/4 四臂，无机制或因果结果**

唯一启动命令与 Python traceback 保存在 `artifacts/analysis/cscwd_sharing_migration_2x2_development_20261011_v1/command_log.txt`；`failure_receipt.json` SHA-256 `939bb277c8387c6060849286be01f84eaf9fa65bc6fd427fb723325141c66315`，`artifact_integrity.json` SHA-256 `c0c05817d4b4810379c502e3e1f1c978cb83ade4a7c4e281b0c6d6b35094a159`。启动 gate 原件 SHA-256 `17f6403e6afd923b415d45bd9c064fb84a3001ed0eeae94125ab7812dce1b765`，执行器 SHA-256 `13c74a81340e6be761002d81714f7bc916083263a044ec110eef855dc7798b86`。

启动前已确认 Git 干净、代码/配置/hash 和 create-only root 身份冻结；原始 CSV 与来源 manifest SHA 均匹配，来源重载检查 1189 连续帧。按事前首步公开 mask 规则，Lankershim action 4 不合法，唯一选 US101 车辆 1183；其 `Global_Time` 1118849293800–1118849412600 与来源保护区间重叠 0，历史 train/dev 暴露计数 2，因此即使后续成功也只能是已暴露 development。Peachtree/I-80、sealed performance 未读。

失败发生在 `validate_gate()` 返回前：共用 `load_traces()` 返回的第二项是 `{"raw_rows_scanned": ..., "window_identities": {window_id: identity}}`，执行器却用 `raw_identities[eligible[0]]` 直接索引，得到 `KeyError: 'cscwd_newdev_us_101_v1183_t1118849293800_1118849412600'`。这是执行器对生产端返回 schema 的消费错误，**不是**轨迹缺失或因素效果。`run()` 尚未调用，原 root 也未由执行器创建；随后只为保存这次失败创建同名 root 的失败回执与命令记录。0 环境步、0 arm、0 训练/模型调用/formal/holdout；无 `summary.json`、四臂原始 ledger 或可计算交互。

按照单次失败即停授权，本轮不改执行器、不重试、不补跑或换来源。机器 gate 已关闭为 `execution_authorized=false`、`PRESTEP_FAILED_NO_RETRY`，原 root 保留。后续若另立工程任务，应先让消费者按 `raw_identities["window_identities"][window_id]` 读取并加入零步 schema 验收，再为任何新科学尝试另定 commit、root 与授权；本报告不授予该执行权。

**Hard blockers**：四臂原始结果不存在；真实 RSU/链路、正式 checkpoint/manifest/command log/formal/holdout/support 仍缺。**Major concerns**：唯一候选 US101 已作历史 development 暴露。**Minor concerns**：失败回执为本地 artifact，公开复现包尚未整理。**Scorecard**：TMC 七维 `N/S`。**Safe claims**：只可说来源/公开选择预检通过而执行器在环境步前失败。**Prohibited claims**：sharing、migration 主效应或交互方向、优秀基线排名、独立泛化或 paper-ready。四项论文主张继续 `UNVERIFIED`；TITS/TVT 亦不能放宽该证据门。
