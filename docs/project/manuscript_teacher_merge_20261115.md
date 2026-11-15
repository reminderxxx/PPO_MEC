# 老师返修稿定位与审阅合并记录

日期：2026-11-15 Asia/Shanghai。状态：`pending_author_confirmation`。本文件只记录版本、流程与验证，不公开未发表正文或逐段差异。

## 来源与授权

用户本轮明确提供老师修改的同题名 DOCX，要求定位版本、用审阅模式融入现有修改，并学习记录老师的修改及原因。这是本轮采用新老师稿起点的依据，不追认旧候选，也不晋升正式 R 版本。

- 老师原稿 SHA-256：`d02be20cdd97408f03039f85117244aa89d536349eb5275cbde2fe2e5565eff2`。
- 修订作者为 `Zhu Qinghua`，53 处插入、38 处删除，涉及 17 个正文段落；另有研究问句/贡献、表题/图例两处拆段。
- 撤销老师文字修订后，忽略空白与上述拆段的正文与 R20 逐字一致。因此定位为 R20 基础上的老师返修，非 R22 或八种子候选返修。日期及元数据只作辅助，不作为确认依据。
- 本地原件、详细前后对照、原因推断、合并脚本及候选均在 `artifacts/manuscript/teacher_merge_review_20261115/`，不随 GitHub 上传。

## 合并与学习

原件逐字节保留；仅在工作副本中接受老师既有文字/格式修订，再以独立身份记录本轮真实 Word 修订。合并 R22 内容、10 月 30 日 v4 的问题链/相关工作/引用映射/算法章标题，以及 11 月八种子候选的主表和图 3；25 条参考文献。历史批注留在原历史稿；本轮重新定位批注，不伪称已继承旧批注。

老师修改体现的通用写作方向：引言先交代服务需求与问题，不提前堆结果；研究问句与贡献分开；章节导读职责清楚；图表题注精简；缩写在适当的首次位置展开；用具体主语、主动动词和领域术语。以上修改意图均为编辑分析，未经老师逐条解释，不作老师原话。

保留老师对建模章短标题、标准实验章名和图 1 图注精简的修改。与后续重写重叠的引言、相关工作和实验内容以显式修订供作者审阅，冲突处理及 17 处完整前后对照见本地 `teacher_revision_learning.md` 和 `revision_change_log.json`。

八种子主表与旧三种子机制统计分开标明。正文将继承的符号检验标为未校正，不把其直接写为全面或校正后显著优势；本轮没有新训练、调参、baseline 晋级或文献检索。

## 核验与边界

- 输入哈希、DOCX ZIP、版本逐字对照、接受/拒绝投影、批注关系及媒体关系检查。
- bundled `render_docx.py --emit_pdf --dpi 144` 分别检查接受后阅读视图与显示修订视图；最终页数、哈希和目检结果追加于修订台账。
- 原始最终开发结果 8×8×48=3072 行独立重算，均值与既有表一致；重新计算种子符号检验及七对照族的 Holm 值。继承的 bootstrap 区间和旧三种子对象收据未在本轮完整重审，不能视为 E3。
- 作者与基金仍为占位；用户确认、独立窗口和公开方法公平复现未完成。本轮候选不覆盖已确认 R22，不赋下一正式 R 号。

审查元数据：`reviewed_at=2026-11-15 Asia/Shanghai`；`literature_cutoff=2026-10-30 inherited (no new search)`；`target_venue=CSCWD 2027`；`artifact_run_id=runs/seed_ext8; teacher_merge_review_20261115`；`policy_version=tmc_review_policy_v3_20260621`；工作起点 Git `ddc85fbbfe1e24e94a02c5f9951b697bb72759b9`；`evidence_level=TARGETED_RAW_DEVELOPMENT_RECOMPUTATION_AND_DOCX_MERGE_VALIDATION`；formal/holdout/support 未审计，readiness `Unverifiable`，不作评分或晋级。

## 后续四处叙事闭环

用户随后要求自主完成四处优化。最新候选为本地 `artifacts/manuscript/narrative_closure_review_20261115/sagh_narrative_closure_tracked_candidate.docx`，SHA-256 `ef9a0f4a88c9347815c695d54952406f04f0ae52612d4043f377503595af2df1`，尚未正式确认。旧合并稿与老师原件不变；仅在工作副本接受旧建议，保留54条批注后，以本轮独立修订标出九个目标段落/标题和九条新批注。

四项均已完成：引言与工程比较形成问题发现—验证呼应；方法明确三项难点与现有机制的对应；贡献、指标交叉引用与实验小节标题对齐；未来工作从独立验证和组件归因逐步延伸到更广服务条件，世界模型仅作后续可选资源/服务预测。复读三篇已登记本地论文的引言段落功能，未新增文献、实验或性能断言。

接受/拒绝投影、63条批注锚点、修订ID、限定九处差异核验通过；图表媒体、数值及25条引用不变。`render_docx.py --emit_pdf --dpi 144`渲染接受后6页、显示修订7页，13页逐页目检通过。完整前后对照和验收只存本地 `editorial_closure_record.md`、`revision_change_log.json`、`validation.json`。既有待确认、占位信息、统计与独立窗口限制未改变，不构成正式晋版或paper-ready判定。
