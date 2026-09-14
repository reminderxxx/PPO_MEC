# P01 写作进度与交接

## 当前完成

- 建立论文专用证据总账，固定 scientific/executor commit、protocol、model source、evaluation run 和证据等级。
- 完成现有 manuscript inventory：仓库没有成熟完整 `.tex` 主稿；历史 method report 保留不覆盖。
- 完成 IEEE TMC 导向的英文 v0.1：摘要、引言、Related Work 占位、详细 System/Problem、详细 Method、详细 Experimental Methodology、Results 占位、Discussion、Reproducibility、Conclusion。
- 完成 research question、contribution、claim、table、figure 和 result sentence 的证据映射。
- 把 current formal 状态写入稿件边界：6/22 cells committed；ablation failed；support/scalability/statistics/gate/completion 未开始；holdout sealed。
- 明确 legacy prose 与 frozen checkpoint 的冲突，正文只保留 actual enabled mechanisms。
- 未读取性能数字形成结论，未改代码/配置/脚本，未运行训练、评测、恢复或 holdout，未触碰 held lock。

## 文件入口

- `paper/manuscript_v0_1.md`：英文主稿骨架及详细核心章节。
- `paper/evidence_ledger_p01.md`：证据总账、审查元数据、实现开关、产物索引和 claim 边界。
- `paper/claim_evidence_matrix_p01.md`：RQ/贡献/claim 对照、表图计划和结果句式模板。
- `paper/progress_handoff_p01.md`：本文件。

## 等待中央实验任务

下列工作不属于 P01，也没有被本分支授权：

- 修复 executor/scientific checkout 的 active-bundle root mismatch；
- recovery/retry/lock cleanup；
- formal ablation/support/scalability/statistics/gate/completion；
- holdout 开启或消费；
- canonical/paper-ready 晋级。

中央实验任务完成后，写作端的最小接入顺序应为：

1. 先审 complete manifest、command log、cell/phase ledger 和 Git/model identities；
2. 核验 formal/holdout 原始 frame/time intervals 互斥；
3. 核验 12 个 outer windows 的 paired availability 和 nested seed/workflow 统计；
4. 只采用 gate-assigned claim statuses；
5. 按 `claim_evidence_matrix_p01.md` 填表图和 Results；
6. 最后才改 Abstract、Introduction contributions 和 Conclusion 的结果句。

## 建议的 P02 任务

P02 可与中央实验修复并行，范围限定为文献与定位：

- 从一手 publisher 页面刷新 2026-06-21 之后的 TMC/TC/TPDS/TSC/INFOCOM/NSDI/MLSys 等近邻工作；
- 核验 `docs/project/literature_reference_table.md` 中仍标 `待核验` 的 venue/DOI/录用状态；
- 围绕三条最近邻线构建 Related Work：dependent DAG + caching/offloading、predictive handoff/migration、AI model/adapter serving；
- 给出逐篇 closest-neighbor difference，不使用“first”或“state of the art”绝对口径；
- 生成可复核 BibTeX，并在表中记录 primary-source URL 与访问日期；
- 提交一页 novelty-risk matrix，特别审查 “multi-agent”、learned predictor、digital twin、adapter cache 和 oracle wording。

## 真正需要作者决策的事项

1. 是否保留 `SA-GHMAPPO` 作为论文名。若保留，建议把正文扩展改成 “surrogate-assisted graph and hierarchical controller-level PPO”，避免 “multi-agent” 与实际 topology 冲突。
2. 最终叙事主轴选哪一个：`typed model caching for continuous DAG service`（建议）还是更宽的 `joint workflow/cache/handoff control`。前者更聚焦，也更符合当前可核验机制。
3. 结果未通过 gate 前，是否先转 IEEE TMC LaTeX。建议等 P02 完成 title/terminology/related-work 定位后再转，避免在模板层反复改名。

除这三项外，P02 和实验等待期间不需要额外作者决策。

## 已知未覆盖风险

- P01 未联网刷新文献；novelty cutoff 仍按最近一次有审计记录的 2026-06-21。
- 未重新计算 checkpoint 参数量、推理延迟或复杂度常数；Method 保留明确 TODO。
- 未验证数据集当前许可/下载页面；Reproducibility 保留 TODO。
- 未做数值结果审查；当前 raw aggregates 只记录文件入口。
- 独立写作分支基于当前 worktree 基线，中央 main 的实验进度文档可能继续变化；合并时应优先保留中央任务产生的最新科学状态。

## 停止线

P01 在完成文档、校验、commit 和 push 后停止。除非用户另行授权，不继续修实验、不续跑、不改 artifact、不做 holdout、不宣布 paper-ready。
