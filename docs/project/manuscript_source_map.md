# 论文当前版本与证据来源

稿件指针核对：2026-11-21 Asia/Shanghai；本地实验证据快照核对：2026-11-18。本页负责快速定位；确认依据与逐处变更仍以[修订台账](manuscript_revision_ledger.md)为准。

## 当前版本

| 用途 | 位置与状态 |
| --- | --- |
| 已确认正式修订底稿 | `artifacts/manuscript/r22_20261030/` 中完整题名加 `_marked.docx`；SHA-256 `78e38c83d7005e445164dfbb3dfdd62db254bb5ea68e486f15d61f967fda083b` |
| 最新待确认候选 | `artifacts/manuscript/full_notation_20261121/` 中完整题名加 `_marked_candidate.docx`；SHA-256 `9218b8a2d43e15c090d1b0c499fadd0ea7f22200dfe90c97f1498c4dae1ba3a8`，同目录 `figures.pptx` 为两图可编辑源件 |
| 阅读预览 | 同目录完整题名 `.docx` 和 `_reading_preview.pdf` 为交老师的阅读稿；标注候选另存 |
| 历史候选 | `rl_alignment_review_20261030` v4、`r23_20261111`、`teacher_merge_review_20261115`、`narrative_closure_review_20261115`；保留追溯，不重新并行改写 |

本轮全文核查从 11-21 图表候选继续，完整交付指针见上表。11-21 图表候选承接 11-20 版，含 III-B 数学排版校正；上一轮 11-20 图表候选承接 11-15 名称纠偏稿，六页阅读预览与七页标注渲染均已核验。11-15 候选已包含老师返修、v4、八种子实验和四处叙事优化，并恢复固定小节名称。11 月 18 日重新计算 R22 和 11-15 名称纠偏稿的哈希，与当时台账一致。已有 `validation.json` 记录阅读 6 页、修订 7 页及 65 条批注的检查，11-18 没有重新渲染或追认正式 R23。

## 工作区职责

| 内容 | 唯一定位方式 |
| --- | --- |
| 长期规则、需求与修订状态 | 本仓库 `/Users/howen/Projects/PPO_MEC`；`AGENTS.md`、`agent.md`、修订台账和[需求清单](requirements_backlog.md) |
| 八车实验科学代码 | `/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC`，本次读取 HEAD 为 `4bc39b95ca55bf14b72625fbbe1e59b7ac509702`；不等同 main 当前算法，也不据此证明整轮训练 provenance |
| 完整八种子原始包 | `/Users/howen/Documents/deepseek-harness/default-workspace/runs/seed_ext8/`，协议为同级 `seed_ext8_protocol/`；原件原位保留 |
| 本地可复算快照 | `artifacts/analysis/manuscript_support_20261118_v1/`；见下文，非完整训练包 |
| DOCX 与未公开逐处对照 | 本仓库 `artifacts/manuscript/`；Git 忽略，不随长期文档推送 |

主工作区原有七个代码/测试修改与本轮文档维护分开，不能用 dirty main 替代冻结科学代码。此前附件中另一环境的 Git 报错不代表本仓库损坏；本次 `git status`、HEAD 读取及 `git fsck --connectivity-only --no-dangling` 均通过，未重建 index 或修改 pack。

## 数字与措辞的来源顺序

1. 原始 `dev.jsonl` 加协议、最终 episode、agent/seed/case 选择条件是重算入口；不得用展示文本取代原始记录。
2. `artifacts/manuscript/teacher_merge_review_20261115/evidence_audit.json` 是 11 月 15 日针对均值、种子符号检验与七对照 Holm 的独立核对记录，快照内保留其同字节副本。
3. `deliverables/experiment_section_rewrite_spec_20261111.md` 是历史改写规格，其“唯一数字与措辞来源”仅适用于当轮，不能覆盖后续老师修改、名称冻结和统计更正。已复制到快照 `historical_specs/`；外部原文未改。
4. [八种子结果记录](seed_extension_8_results_20261110.md)、[缺口评估](cscwd2027_readiness_gap_and_fill_order_20261109.md)的历史段落按日期解释；候选确认状态以修订台账为准。

`209.45764472961426` 显示为 `209.458` 或 `209.46` 是精度差异；旧三种子 `210.40` 属于另一统计口径，不可混表。三种子对象生命周期收据 `757/1078` 继续单列历史来源，未在本轮重算为八种子，不把它写成八种子机制统计。

继承审查已注明：PPO/DQN 的未校正符号检验 `p=0.0078125`，七对照族 Holm 值均为 `0.0546875`；MAPPO 为 7/8 同向，cache 为 4/8。旧材料的直接“显著优于 PPO/DQN”措辞不能直接复用。当前仍为固定开发面板证据；bootstrap、窗口独立性与完整训练 provenance 不因本次复制或复算获得晋级。

## 本地快照与归档边界

run_id：`manuscript_support_20261118_v1`。共 45 个复制文件、84,864,358 字节，包含：

- 八个控制器的 `dev.jsonl`、`training.jsonl`、`command.json`、`manifest.json`；协议目录四个文件。
- 原运行日志、统计 JSON、Fig. 3 曲线 JSON；两份历史改写材料及 11 月 15 日审查 JSON。
- `snapshot_manifest.json`：原路径、相对路径、大小、SHA-256，另列完整原运行包的文件名和大小。
- `snapshot_validation.json`：复制文件/原文件哈希一致、3072 条最终开发行、均值/符号检验/Holm 对照，以及原七个 dirty 文件和两份论文的保护检查。

`events.jsonl`、`receipts.jsonl` 与 checkpoint 留在原运行包；清单中对这些排除文件只记路径和大小，没有完整哈希验证。此快照不能独立承担全训练复现或机制收据审计。完整归档待办见 `REQ-009`，不把“部分复制”登记为“全部迁移”。`scripts/export_paper_artifacts.py` 消费另一套 benchmark aggregate schema，不能用于本轮 `dev.jsonl` 导出。

本页记录来源管理与继承核对，不重新评价论文 readiness、novelty 或算法效果。既有独立审查 ID 为 `teacher_merge_review_20261115/evidence_audit.json`；本次未新增文献检索、训练或实验晋级。
