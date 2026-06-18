# Top Journal Review Policy

- `policy_version`: `tmc_review_policy_v1_20260618`
- `updated_at`: `2026-06-18`
- `primary_target`: `IEEE Transactions on Mobile Computing (TMC)`
- `secondary_fit_targets`: `IEEE TITS / IEEE TVT`

用途：为后续 AI reviewer 提供稳定、可重复、可追溯的顶刊研究与实验审查规则。本规范评价的是“是否具备 TMC-ready candidate 条件”，不预测编辑或审稿人的最终录用决定。

## 规范来源与边界

外部规范只提供定性要求，不提供录用分数线：

- [IEEE Reviewer Best Practices](https://ieeeaccess.ieee.org/reviewers/reviewer-best-practices/)：技术深度、相关工作、benchmark、验证、数据支撑、可复现性和建设性审稿。
- [IEEE Research Reproducibility](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)：方法、数据、代码和复现实验的透明性。
- [IEEE Ethical Requirements](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/ethical-requirements/)：准确、完整地报告数据，不得捏造、篡改或选择性隐瞒结果。
- [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-badging-current)：artifact 应 documented、consistent、complete、exercisable；可复用性高于最低可运行要求。
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)：训练细节、数据划分、hyperparameter、compute、error bar / CI 和统计变异来源透明。

下文分数和阈值是 PPO_MEC 内部 operational rubric，不是 IEEE/TMC 官方录用线。引用本规范时不得写成“满足 TMC 官方量化标准”。

## 强制触发条件

涉及以下任一事项时必须执行本规范：

- 判断算法或实验是否达到顶刊、paper-ready 或 canonical 晋级条件；
- 更新论文主表、claim、abstract/result statement 或投稿建议；
- 评价新 baseline、消融、robustness、scalability 或预测机制；
- 评价相关论文的水平、近邻性、novelty 风险或 baseline 必要性。

审查与算法修改默认分轮进行。审查轮只读 artifact 和源码，不为消除 blocker 自动调参、修改算法或重跑筛选实验；如需修复，先输出审查结论和独立改进任务。

## 证据等级

1. `E0_UNAVAILABLE`：关键 artifact 不存在；只能判定 `Unverifiable`。
2. `E1_DOCUMENTED`：只有 README / PROGRESS / ARTIFACT_RECORDS 摘要；可做条件性 desk review，不能确认数值 claim。
3. `E2_ARTIFACT_AUDITED`：原始 JSON/CSV、manifest、command log、formal/holdout/support 输出齐全并交叉一致。
4. `E3_REPRODUCED`：在独立临时目录重建 comparison report，关键结果与原包一致；checkpoint/path/hash 可追溯。

正式 `TMC-ready candidate` 至少要求 `E2_ARTIFACT_AUDITED`。涉及“可复现”表述时要求 `E3_REPRODUCED`。

Artifact 审查必须覆盖：

- run root、run ID、文件清单、SHA-256 或等价完整性证据；
- seed、训练预算、数据窗口、checkpoint provenance、protocol version 和命令记录；
- manifest 中所有外部路径的存在性；
- formal、holdout、prediction robustness、system robustness、scalability、ablation 和 comparison package；
- 生产端 JSON/CSV 与论文表格、README、CONTEXT、BUGS、ARTIFACT_RECORDS 的一致性。

缺失正式 checkpoint、manifest、command log、formal/holdout/support 任一关键证据时，不得用文档摘要补位。

## 硬性 Blocker

任一科学性 blocker 成立即不得判为 `TMC-ready candidate`：

- 与最近邻顶刊工作无法形成可验证的 problem、method 或 system-level 区分；
- baseline contract 不匹配、训练预算明显不公平、重复 trace/策略未排除，或把 diagnostic baseline 当 paper-grade baseline；
- 主 claim 只依赖单 seed、point estimate，缺少变异来源、CI/error bar 或适当统计检验；
- training / checkpoint selection / formal / holdout 存在数据泄漏，或没有独立 holdout；
- 主优势依赖 reward shaping、evaluation-only bias 或 cherry-pick，且缺少真实机制兑现指标；
- 对 cache、handoff、migration、prediction、DAG 或 multi-agent 的 claim 没有对应可观测指标与消融证据；
- 结论超出数据范围，例如把 controller-level baseline 写成 vehicle/RSU-level full MARL，或把 NGSIM + Alibaba 写成普适真实部署；
- canonical artifact 缺失、不可执行、无法追溯或与文档数值冲突；
- 负向结果、失败 split、oracle/no-prediction 边界或已知 limitation 被隐瞒。

下列情况默认是 major concern，是否升级为 blocker 取决于主 claim：只有一个 mobility/workflow 数据组合、仅 3 个 seed、未报告多重比较风险、缺少 wall-clock/compute 成本、strong heuristic gap 极小、机制在部分 split 无独立正 CI。

## 100 分内部评分

| 维度 | 分值 | 核心检查 |
|---|---:|---|
| Novelty 与近邻文献定位 | 20 | 最近邻矩阵、问题交集、贡献不可被已有 DAG/cache/migration 工作直接覆盖 |
| 技术正确性与问题建模 | 15 | observation/action/reward contract、假设、复杂度、因果链和实现一致 |
| Baseline 公平性与独立性 | 15 | 方向匹配、预算、输入信息、checkpoint provenance、独立 trace |
| 实验设计、统计与 holdout | 20 | 多 seed、formal/holdout、paired unit、CI、selection bias、多重比较说明 |
| 机制兑现与系统指标 | 10 | validated hit、realized prepare、handoff ready、continuity、failure、backhaul |
| 鲁棒性、泛化与可扩展性 | 10 | prediction/system stress、规模变化、弱 split、外推边界 |
| 可复现性与 claim 完整性 | 10 | artifact、命令、环境、数据、compute、限制和负结果可追溯 |

逐维必须给出“得分 / 满分、证据路径、扣分原因”。没有原始证据时不打伪精确分数，统一记 `N/S (not scored)`。

## Verdict

- `TMC-ready candidate`：无 blocker，总分 `>= 85`，且每个维度得分率 `>= 60%`。
- `Major revision`：无致命 blocker但总分 `75–84`，或存在可修复的重要缺口。
- `Not TMC-ready`：总分 `< 75`，或存在科学性 blocker。
- `Unverifiable`：关键 artifact/provenance/原始统计缺失；优先级高于基于摘要的乐观判断。

AI 不得输出“会被 TMC 接收”“达到 TMC 官方分数线”。TITS/TVT fit 只能作为次级适配建议，不能降低 TMC 主审 rubric 后再宣称 TMC-ready。

## 固定审查流程

1. 记录元数据：`reviewed_at`、`literature_cutoff`、`target_venue`、`artifact_run_id`、`policy_version`、Git commit。
2. 核验 artifact 存在性和完整性，确定 `E0–E3`；证据不足立即标 `Unverifiable`。
3. 检索最近 5 年一手出版页面，按标题/DOI/arXiv ID 查重，并更新 `literature_reference_table.md`。
4. 建立最近邻矩阵：问题、控制对象、动作粒度、cache 语义、mobility/handoff、数据、算法和指标。
5. 审查数据划分、seed、budget、checkpoint selection、baseline contract、统计单位与 CI。
6. 将每个 claim 映射到 artifact 字段；reward shaping 不能代替机制兑现。
7. 检查 prediction、robustness、scalability、ablation、weak split、negative result 和 limitation。
8. 输出 blocker、major concern、minor concern、安全 claim、禁止表述、评分/未评分理由与投稿建议。
9. 实质结论同步到 `PROGRESS.md`；长期风险同步到 `BUGS.md`。审查报告和规范均更新日期。

## 文献质量等级

- `A-Core`：TMC 或直接支撑主 claim 的同级 Transactions 最近邻工作。
- `A-Adjacent`：TITS、ToN、TSC、TNSM 等高水平近邻工作。
- `B-Supporting`：TVT、其他可靠期刊或非核心但有用的机制工作。
- `C-Context`：预印本、辅助期刊、只用于 Discussion/reviewer response 的材料。
- `Unverified`：venue、年份、DOI、页码或正式录用状态尚未由一手页面核验。

Venue 等级不等于论文质量结论；AI 仍需检查问题近邻性、实验深度和可复现性。service cache、model/checkpoint cache、KV cache 与 adapter cache 只能类比，不能等同。

## 固定输出模板

```text
reviewed_at:
literature_cutoff:
target_venue:
artifact_run_id:
policy_version:
git_commit:
evidence_level:
verdict:

Evidence inventory:
Hard blockers:
Major concerns:
Minor concerns:
Scorecard:
Safe claims:
Prohibited claims:
Required actions before re-review:
TITS/TVT fit note:
```

## 保密与维护

- 不向外部网站、公开 AI 服务或搜索引擎上传未公开 manuscript、artifact、checkpoint、真实数据或审稿材料；网页检索只使用公开关键词和公开论文。
- 每次审查使用实际日期；规范发生评分、blocker、证据等级或 verdict 变化时递增 `policy_version`。
- 文献检索截止日期超过 90 天、目标 venue 变化或 canonical run 更新时，必须重新做最近邻检索。
