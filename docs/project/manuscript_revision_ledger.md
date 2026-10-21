# 论文逐轮修订要求与位置记录

本文件是 CSCWD 2027 论文的长期修订台账。每次开始论文修改，先核对本文件；完成一轮后追加该轮的目标位置、修改前后内容、版面验证和已固定位置。只记录可追溯的要求与状态，不把未核验的差异写成事实。

## 持续生效的修订规则

1. 每轮以明确的文稿版本为基线，记录文件名、SHA-256、来源和轮次；优先使用章节标题、段落首句、表/图/公式编号等稳定锚点定位，不单靠页码。
2. 只改该轮要求指向的位置。文献重排、简称统一等要求带来的必要交叉引用变更，须逐处列明；其余内容和已固定位置保持不动。
3. 一轮通过内容核对及逐页排版检查后，将已验收的位置列入“已固定位置”。下一轮不得覆盖，除非用户的新要求与之冲突；冲突时以最新要求为准，并注明被覆盖的旧要求及具体位置。
4. 每轮交付说明“哪条要求、改了何处、如何验证、仍缺什么”，保留可复核的前后对照。没有找到原稿、没有用户确认或没有验证时，状态只能写“待核验/待确认/未落实”。
5. 用户要求修改具体细节时，只处理指定位置与必需的编号、引用、术语同步，不附带增加文献、实验、结论或其他论述。用户明确要求结合新实验、改写说法或补充说明时，扩展仅限该次要求覆盖的内容；新增内容及其依据须在当轮台账中列明。

## 来源与当前基线（2026-10-18）

- 老师版：用户提供的 `CSCWD2027何卓文.docx`；本机微信临时附件 SHA-256 `2e85afae2afcb6b0aa6cb0df0a0af8f95149a0a34ccaacb2e9b0508930eff490`。同名的微信消息文件、拖拽文件和临时打开文件哈希一致。它是 R1 只读核对基线，原件未被覆盖。
- R1 工作副本：`artifacts/manuscript/r1_20261018/CSCWD2027_He_Zhu_R1_English_Table_I.docx`，SHA-256 `1ed4567eb912533e3a11a48634fafef5e2a53503490200411e9862ffe62204a6`。论文文件在 Git 忽略的 `artifacts/` 内，不随台账提交或推送。
- **当前 R2 工作副本**：`artifacts/manuscript/r2_20261018/CSCWD2027_He_Zhu_R2_25_References.docx`，SHA-256 `0937991dbc68a0f0a6985ed16b92595157ef277d4bdf750e8738fa624e4985eb`。R2 仅纠正 R1 额外增加参考文献的问题；R1 文件保留作可追溯历史版本，后续以 R2 为基线。
- 组会 PPT：用户提供的 `mobile_edge_workflow_jury_report_compact_20261011(2).pptx`；SHA-256 `0b2173cc4d1f874dfe2b3698bcaacce4df13646f8e0506195fb2652b72e121c9`。候选替换表位于第 3 页“横向对比”，列为“任务依赖、模型复用、车辆移动、状态续接、联合决策”，末行为“本研究关注”，图例为“● 直接涉及、◐ 部分相关、— 非核心”。
- 初版论文：最近两次 ChatGPT 聊天“修改论文初稿”（ID `6ad3933d-d320-83ec-a70f-55e648132428` 与 `6ad393b7-0498-83ec-8f93-f8cb3097c381`）分别记录旧稿 `SA_GHMAPPO_Manuscript_He_Zhu.docx` 和 10 月 8 日版本 `SA_GHMAPPO_He_Zhu_EmailUpdated.docx`，路径为当时聊天的 `sandbox:/workspace/scratch/...` 临时空间。本轮在本机及仓库未找到这两份旧稿的可读取副本；下方差异是**先前聊天的文件比较记录**，本轮尚不能重新逐段验算。
- 分享聊天：`https://chatgpt.com/share/6ad42427-4eb0-83ec-9774-a29fed55e4f8`；网页读取未成功，但已通过最近聊天记录读取第一轮用户原话及先前比较结果。当前聊天所贴原话与对应 ChatGPT 聊天一致。

## 第一轮要求 R1：老师意见与用户指令

状态：**R1 是历史交付版本，参考文献数量与表内新增引用已按用户最新意见在 R2 纠正；初版原文件仍待找回，投稿页数与作者占位仍待核定。** 用户于 2026-10-18 确认 PPT 第 3 页的五维横向对比表为表 I 的替换目标，并补充要求：先把表内中文译成英文，再放入英文论文（R1-09）。

| ID | 要求 | 精准目标位置与当前状态 |
| --- | --- | --- |
| R1-01 | 删除将自有 GitHub 源程序列为正式文献的第 `[8]` 条；把源码 GitHub 链接作为普通说明放在摘要后或正文末尾。 | **已落实于工作副本**：删原第 `[8]` 条及五处自引；在第 VI 节末、`References` 前加普通源码说明。 |
| R1-02 | 随删除文献 8 重排后续文献及所有正文引用；表 I 换表后再按正文首次出现顺序核定。 | **已落实于工作副本**：为新表补 7 篇公开论文，按首次引用重排为 `[1]`–`[32]`；程序核查首次出现顺序、连续编号和对应条目。 |
| R1-03 | 全文采用老师改定的 `SAGH` 简称，覆盖标题、摘要、关键词、正文、图注、表格、算法框及图内文字。 | **已落实于工作副本**：11 处正文/表格旧名改为 `SAGH`，原自引条目随 R1-01 删除；图 3 从表 III 数值重绘，仅将方法名变更为 `SAGH`，其他数值保持一致。 |
| R1-04 | 先找到并展示组会 PPT 的“已有研究与本研究”横向比较表，待用户确认后替换论文表 I。 | **已落实于工作副本**：原三列表改成 11 行×6 列五维矩阵；保留原 PPT 每个 `●/◐/—` 判断及三档图例，为行名加论文引用；说明末行是研究范围而非全部维度的实验结论。 |
| R1-05 | 第 III 节清楚写出研究问题和实际调度优化目标，以及决策、约束、目标与奖励的关系。 | **已落实于工作副本**：在 `III-C` 的成本扩展式前补五类动作、实际目标和可行性条件，明确实际控制器按 `V-A` 奖励最大化折扣回报，(3) 只是另行分析的成本扩展。 |
| R1-06 | 明确解释公式 (1) 的 `C_r`，包括所指容量层级与单位，并与实现核对。 | **已落实于工作副本**：公式 (1) 后定义为每个 RSU 在该式所抽象的单一模型驻留缓存层上的容量，单位 MiB；不把主机、加速器、车辆或不同 RSU 容量相加；注明该成本扩展不能证明每项控制器实验都启用了字节容量约束。与项目 MB 容量 contract 的单位核对完成。 |
| R1-07 | 告知初版与老师版不同处、老师版有哪些实际标记，并逐轮报告修改位置与未完成项。 | 老师版差异及红字/黄高亮记录见下；**初版原文件仍不可读取**，逐段独立复核未完成。不能把老师版颜色等同于作者或老师的修订记录。 |
| R1-08 | 正文缩略语首次出现时写出全称；保留老师已改的 LoRA，继续检查后续缩略语。 | **已处理明确定位项**：摘要中的 SAGH、ETSI、KV、GAE、ETA、KL、NGSIM、LRU；保留老师对 LoRA/PPO/DAG/RSU 的已有展开。投稿前仍需做全篇缩写审校，特别是图内或文献标题中的缩写。 |
| R1-09 | 用户确认 PPT 比较表后，先把对应中文译成英文，才放入英文论文。 | **已落实于工作副本**：表头译为 task dependencies、model reuse、vehicle mobility、state continuity、joint decisions；“本研究关注”译为 `This study`，图例译为 `Directly addressed / Partly related / Not central`；P4 行保留论文作者名，不保留中文“迁移”。 |

## 已核实的老师版现状

- 标题使用 `Multi-agent SAGH reinforcement learning`，但摘要、关键词、表 I 后说明、图注、实验、表 III、结论等仍出现旧简称。可编辑文字中旧简称 12 处，新简称 4 处；Fig. 3 图像内仍有旧简称。
- 表 I 为 3 列、6 个已有研究条目的比较表；PPT 第 3 页是五维矩阵，两者内容和结构不同。
- 第 III 节 (1) 的 `C_r` 被黄色高亮；`illustrative advisory workflow`、`b(a)`/`x_{o,r}` 定义句、`eviction`、命题中的 `MiB` 也有黄色高亮。引言及表 I 等有红字。没有结构化 Word 批注、`w:ins` 或 `w:del` 修订。
- 老师版存在第三、第四作者与邮箱占位、`Grant xxx`，后续投稿前需由用户核定；这不是本轮擅自补写的授权。

## 初版对照的待复核记录

以下来自上述两次 ChatGPT 聊天所保存的旧稿与老师版文件对照，不是本轮重新读取旧稿后的结论：标题由以 `SA-GHMAPPO` 开头改为以 `Multi-agent SAGH reinforcement learning` 开头；LoRA、DAG、RSU 首次出现的全称和 LoRA 解释得到补充；算法定义、章节导读、Algorithm 1 标题改用 `SAGH`/`SAGH-based PPO`；第 III 节对象定义句有调整；III-A 一处 `[8]` 被移除；作者/邮箱和基金说明有增补；部分表达如 `later services`→`successive services`、`will be used next`→`can be used next` 有润色，多处数学字符有替换；五张图片宽度据报由 2600 像素变成 744 像素；摘要、关键词、表 I–III、26 条参考文献和 Algorithm 1 步骤据报未改。找到初版后须逐项重新比较并标注真实页、段落和图片对象。

## R1 交付核对与版面限制

- 以原老师版为只读基线，仅在新工作副本改动摘要/简称、II 节表 I 及其说明、III 节问题目标与式 (1) 解释、文献及交叉引用、必要的首次缩写说明、图 3 方法标签和正文末的源码说明。
- 程序验证：DOCX ZIP 可读；4 个表仍在，表 I 为 11×6；5 张图片仍在；文献 `[1]`–`[32]` 连续且首次引用次序一致；表 I 无中文，所有可编辑文字无旧算法名；原老师版 SHA-256 未变。
- 逐页渲染并目视检查 7 页，无图丢失、表格溢出或文字遮挡。老师版是 6 页，新增横向表、说明和 7 篇文献使工作副本变为 7 页；表 I 在双栏中字号偏小，若投稿限 6 页或要求较大字号，下一轮需处理版式，但不得擅删已确认的研究行与符号。
- 7 篇新表文献 DOI 已据出版社页面或论文作者页核对；表中的三档归类沿用用户确认的 PPT，不应误称为独立实验证明。
- 原老师版仍有作者/邮箱和基金编号占位，不属于 R1 代填范围。

## R2 用户纠正：撤回额外增加的 7 篇文献

- 用户指出 R1 增加过多参考文献，并确认“第八个不再是我的 GitHub 项目”这一点已改好。最新要求覆盖 R1-02 中“为新表补 7 篇”的实现决定，以及 R1-04 中给新表前 7 行添加引用编号的决定；不覆盖已确认的英文译名、比较符号和其他 R1 修改。
- 精准改动：删除 R1 中因 COHCTO、LDMCO、Ly-COMA、Fan et al.、MCMA、MATM、AWTO 而额外加入的 7 条文献及表 I 对应 7 个编号；仅对后续原有文献及正文引用做必要顺移。文献现为连续的 `[1]`–`[25]`；第 `[8]` 条是 PagedAttention，源码 GitHub 链接仍是正文末的普通说明。
- 机械核对：25 条文献连续且按首次引用次序排列，5 张图的媒体字节与 R1 完全相同，表 I 各行的五维符号与 R1 完全相同，表格尺寸和其他段落内容未因本次纠正而重写。DOCX 完整可读，逐页渲染检查 7 页。
- 新表前 7 个研究行当前只列算法/作者名称，没有正式参考文献条目。这是本次遵循用户“不要额外增加文献”的结果；投稿前仍需单独核定比较表的引文要求，未经用户新要求不得擅自把那 7 条加回。
- R2 仍为 7 页，故页数问题不能再归因于“新增 7 篇文献”；新的横向表、正文说明及其他 R1 内容仍需在目标会议页数规则明确后做版式处理。

## 已固定位置

R1 已完成内容与逐页检查，R2 按最新要求覆盖参考文献数量及新表前 7 行的新增引用。下一轮默认冻结：摘要中的 SAGH 全称和术语、II 节表 I 的英文译名与 10 行符号、表后图例与解释、III-C 实际调度目标与成本扩展边界、式 (1) 后 `C_r` 定义、第 VI 节末普通 GitHub 源码说明、图 3 方法名及 **R2 的 25 篇参考文献映射**。仅当用户指出问题或新要求与这些内容冲突时，按最新要求精确修改相应位置并记下覆盖关系。初版对照和投稿版式不在已固定范围。

## 本轮范围约束补充（2026-10-18）

- 用户说明：新增内容可以在明确授权的优化中使用；当要求是修改某一细节时，不要在要求外顺带增添内容，避免一边纠错一边引入新错误。结合新实验、改写说法或补充说明属于可扩展的明确要求，但仍须限定在该轮指定范围。
- 本轮仅更新长期规则与台账，没有修改论文文件。当前工作副本仍为上述 R2 版本，25 篇参考文献及已固定位置保持原状；这条范围说明不构成恢复 R1 额外 7 篇文献的要求。

## 下一轮启动条件

- 找到或取得初版原文件后，补做初版与老师版的独立逐段差异；若仍缺失，继续显式保留该限制。
- 核定目标会议篇幅和表 I 的双栏字号后，若需调整，只动表格版式及其必需的分页位置，不重写已确定的比较判断。
- 后续新要求先与 R1-01 至 R1-09 及“已固定位置”比对；有矛盾时以最新要求覆盖，并记录被覆盖的条目和位置。

## R2 与老师版核对及标注副本（2026-10-18）

- 本轮要求：对照当前 R2、老师版和初稿，说明改动及黄标状态，并在改过的位置作标注。用户澄清“出版”指初稿。本轮**不修改论文正文内容**，只制作核对副本并更新本台账；R2 继续作为唯一修订基线。
- 核对来源：老师版 SHA-256 `2e85afae2afcb6b0aa6cb0df0a0af8f95149a0a34ccaacb2e9b0508930eff490`；R2 SHA-256 `0937991dbc68a0f0a6985ed16b92595157ef277d4bdf750e8738fa624e4985eb`。初稿 DOCX 本轮仍不可读取，因此“初稿→老师版”只能引用上方“初版对照的待复核记录”，不能声称已经完成新的三方逐段核验。
- 标注副本：`artifacts/manuscript/r2_20261018/CSCWD2027_He_Zhu_R2_annotated_teacher_comparison.docx`，SHA-256 `ff1eb08a40a770e82809f956aa03d030d6896c338811f8b7ef940640bf9855a9`。**青色**标示相对老师版的 R2 新增或改写文字；表 I 表题、表头、末行及批注标示整表替换；图 3 的图片对象变化以图注批注说明；参考文献编号变化以对应编号及 `References` 批注说明。副本含 16 条 Word 批注。**黄色**沿用老师版原有高亮，不是完成/未完成状态码。两种颜色均只用于核对，不构成新的论文内容。

| 老师版黄标位置 | R2 文字状态 | 本轮判断 |
| --- | --- | --- |
| III-A `illustrative advisory workflow` | 原短语及黄标保留 | 未见针对此处的明确文字意见，不能判定为“已完成”或“必须改”；待老师原意核对。 |
| III-B `b(a)`、`x_{o,r}` 定义句 | 定义文字和黄标保留；所在段引用编号依删除原 `[8]` 而顺移 | 定义句字面未改；黄标原意待核对。 |
| 公式 (1) `C_r` | 公式符号和黄标保留；公式后新增容量层级及 MiB 单位说明 | 用户明确要求的 `C_r` 解释已写入 R2；黄标未被清除，不代表该解释未落实。 |
| III-B `eviction` | 原词、定义和黄标保留 | 字面未改；黄标原意待核对。 |
| Proposition 1 首处 `MiB` | 原词和黄标保留；R2 在新增 `C_r` 说明中也写明 MiB | 该标记本身未改；不能仅因旁处解释就推断老师原意已满足。 |

老师版没有结构化 Word 批注或追踪修订；红字/黄标只能说明存在视觉标记，不能判定操作者、修改先后或完成状态。

老师版→R2 的可核对正文变动位置：摘要和关键词统一 `SAGH` 并在摘要首次展开；引言、II 节、Fig. 2 图注、III 至 V 节若干首次缩略语补全及旧方法名替换；II 节表 I 改为用户确认的英文五维矩阵并改表后说明；III-B 公式 (1) 后解释 `C_r`；III-C 补实际调度目标并将式 (3) 限定为分析扩展；Fig. 3 内方法名改为 `SAGH`；结论后加入普通 GitHub 源码说明；删除原参考文献 `[8]`，原 `[9]`–`[26]` 顺移为 `[8]`–`[25]`，删除正文原自引。R1 曾额外增加的 7 篇已在 R2 撤回，不列为当前版本的新增文献。表 II 动作含义、表 III 结果数值及其余四张图片媒体字节未因本轮标注改变。

副本验证：R2 与标注副本的全部段落文字、表格单元格文字和 5 张媒体内容相同；DOCX 包完整可读。副本有老师版遗留的 24 个黄色 run 标记和新增的 52 个青色 run 标记，逐页渲染为 7 页并目视检查，没有发现文字遮挡、裁切或图片丢失。标注副本仅供审阅；不得把其中的青色高亮与批注当作投稿正文。

## R3 算法问题链与 Fig. 2 重构（2026-10-19）

- 本轮要求：根据用户提供的 Fig. 2 反馈，把“从什么问题来、解决什么、三部分如何连接”讲清楚；先画问题推演草图，再制作按该思路推进的算法框图。文献 `[8]` 的旧 GitHub 自引问题在 R2 已解决，本轮不改参考文献。
- 基线：R2 `artifacts/manuscript/r2_20261018/CSCWD2027_He_Zhu_R2_25_References.docx`，SHA-256 `0937991dbc68a0f0a6985ed16b92595157ef277d4bdf750e8738fa624e4985eb`。
- 工作副本：`artifacts/manuscript/r3_20261019/CSCWD2027_He_Zhu_R3_Algorithm_Logic.docx`，SHA-256 `9939044a246933e0d5af5e3a6f1084537cb92dd4850c6d5acf1bfd50a1dcdb79`。该文件位于 Git 忽略的 `artifacts/`，不随台账提交或推送。问题草图、宽版架构图、论文 Fig. 2 窄版图及其 SVG 均保存在同目录，文件名前缀分别为 `sagh_problem_sketch_20261019`、`sagh_architecture_20261019`、`sagh_architecture_paper_20261019`。
- 目标位置与前后对照：IV-B 标题后原来直接进入十特征 DAG 编码；现在先写“未完结任务仍需模型、车辆可能切换 RSU、过晚准备导致目标未就绪、过早准备导致驱逐或浪费”，再引出依赖识别、准备时机、合法动作三个问题。式 (6) 后原文只顺列 slow/fast/event 三头；现在分别解释缓存选择对后续依赖、当前服务路径、handoff 窗口的作用，并明确三头是单控制器同一步中的条件化角色分数。式 (7) 后原文只陈述解码优先级；现在写明角色分数经五动作映射、mask、option 与执行检查后只产生一个环境动作，以及结果如何回流到信用分配。Fig. 2 图片及图注随之重构，以问题冲突→状态编码→预测门控→角色分数→合法动作→结果反馈排列。
- 范围与冲突：本轮没有覆盖 R2 已固定的摘要术语、表 I、III-C 目标、`C_r` 定义、图 3、源码说明或 25 篇文献映射。既有标题和引言中的 `multi-agent` 术语也未在本轮顺带改动；其与 controller-level 实现身份的关系仍需后续单独审校。
- 验证：R3 可正常打开并渲染为 7 页；逐页目视未见图像遮挡、裁切或丢失。相对 R2，仅新增一段并改动 IV-B 的两段文字、Fig. 2 图片和图注；参考文献段落文字一致，Word 数学对象均为 67 个，其余五张原图的媒体 SHA-256 均相同。图示用于解释现有机制，不新增算法收益或论文创新性结论。

## R4 全文问题 方法 证据链修订（2026-10-21）

- 用户要求把整篇论文的逻辑串成“问题从哪来、如何发现、解决什么、怎么做、得到什么结果”，并在正文清晰呈现。该要求覆盖 R2 已固定的摘要措辞和 R3 算法问题段的叙事范围，也覆盖老师版标题中 `multi-agent` 的既有写法；不覆盖表 I 的比较判断、III-C 的实际目标与成本扩展边界、式 (1) 后 `C_r` 定义、图 3 数值、普通源码说明和 R2 的 25 篇文献映射。
- 基线：R3 `artifacts/manuscript/r3_20261019/CSCWD2027_He_Zhu_R3_Algorithm_Logic.docx`，SHA-256 `9939044a246933e0d5af5e3a6f1084537cb92dd4850c6d5acf1bfd50a1dcdb79`。新工作副本：`artifacts/manuscript/r4_20261021/CSCWD2027_He_Zhu_R4_Problem_Method_Evidence.docx`，SHA-256 `6266496763a78678546ea99c453277bb9dcd8d743446c96ca774e0e5ede879bd`；Fig. 2 的新窄版 PNG/SVG 保存在同目录。该目录被 Git 忽略，不随台账推送。
- 核心前后对照：标题由 `Multi-agent SAGH reinforcement learning...` 改为 `SAGH Control...`，与单控制器五动作合同一致；摘要与引言原先先列适配器/方法再报优势，现在先写“未完结 DAG 在 RSU 切换时产生准备过早或过晚的冲突”，再写可观察状态、五动作问题、SAGH 的三类角色及独立成本分析。相关工作末段把已有机制与本研究的具体准备时机问题相接，不把图编码、PPO、预测缓存单独写成新贡献。IV-B 明确五动作接口不能直接选择任意未来 DAG 节点或 adapter；Fig. 2 将“下一步需要什么”精化为“哪些依赖有风险”，并把 `migration` 改为实际语义范围内的 `preparation`。训练信用句同步使用 handoff-preparation outcome。
- 实验与结论前后对照：原摘要、V-B、图 3/4 图注及结论把旧奖励排名和原窗口重采样区间放在优势叙事中；现在仅保留既有数值作为历史开发配置的描述性观察，明示计划窗口与执行行源段不一致、原始区间重叠、基线训练/选择不匹配，原 Fig. 4 whiskers 不可解释为独立窗口的有效 95% CI。V-C 的共享/次序和过早加载数值仍是固定策略构造例，明确不归因于 SAGH。结论对应区分“模型内时机冲突已展示”和“算法独立优势、完成率/传输改善尚未证实”。不新增实验、文献、数值或投稿可用性判断。
- 审查元数据与边界：`reviewed_at=2026-10-21 Asia/Shanghai`；`literature_cutoff=2026-09-28`（沿用既有近邻审查，本轮没有新检索或文献水平评价）；`target_venue=CSCWD 2027`；`artifact_run_id=top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730`；`policy_version=tmc_review_policy_v3_20260621`（仅作内部严谨性约束）；修订起点 Git `4208494d5dc4eeae0b007e11cfa53493e1eaa1e7`；`evidence_level=E1_PLUS_RAW_PARTIAL`，核对 manifest、Mixed/Full aggregate、paired statistics 的既有 SHA 与 2026-10-09 独立审查相符，但本轮没有完成 E2 全包审计或新的独立评价。Confirmatory algorithm advantage 仍为 `UNVERIFIED`；审查结论沿用既有 blocker，没有晋级或改变规范。
- 验证：R4 DOCX ZIP 完整可读；相对 R3 参考文献 25 条和全部四张表文字不变，Word 数学对象均为 67 个，原六张媒体字节保留，Fig. 2 另嵌新图。`render_docx.py` 渲染为 8 页，逐页检查未见裁切、遮挡或图片缺失。页数由 R3 的 7 页增至 8 页；CSCWD 目标版面上限尚未核定，不能声称已满足投稿格式。作者、邮箱和基金占位及初稿缺失问题仍待用户核定。

## R5 Fig. 2 强化学习与车联网图式优化（2026-10-21）

- 用户要求针对 R4 图 2 优化图式：参考强化学习和车联网论文的表达，加入强化学习交互与训练要素，改善箭头和整体形式。本轮覆盖 R4 的图 2 图片和图注；不改正文、表格、公式、参考文献或实验结论。
- 基线：R4 `artifacts/manuscript/r4_20261021/CSCWD2027_He_Zhu_R4_Problem_Method_Evidence.docx`，SHA-256 `6266496763a78678546ea99c453277bb9dcd8d743446c96ca774e0e5ede879bd`。新工作副本：`artifacts/manuscript/r5_20261021/CSCWD2027_He_Zhu_R5_RL_VEC_Figure.docx`，SHA-256 `638d80c71e6d9b8b65382b40cfb126ffbfcf0e19cad0b0ef7483e75f0a0ca767`。宽版与论文窄版的 SVG/PNG 同目录保存；该目录被 Git 忽略，不随台账推送。
- 图式参考：核对了已列入 `literature_reference_table.md` 的 MAPPO 原论文与 mobility-aware VEC migration 论文公开页面，仅参考 actor–critic、环境交互闭环与车/RSU 场景的常见呈现方式，未复制原图、改写技术结果或新增文献。
- 前后对照：R4 图 2 为线性的问题→编码→三类分数→动作→训练列表；R5 图 2 先显示车辆跨 RSU 时未完结 DAG 的准备时机冲突，再以实线连接观察、单控制器 actor、合法动作、VEC 环境结果和下一观察，以青色虚线区分 rollout、集中价值估计、GAE/结果信用和 clipped PPO 参数更新。三头仍是同一步的条件角色分数，环境仍只执行一个语义动作；图注同步为交互与学习闭环说明。宽版细分 actor、critic、rollout 和更新，窄版用于双栏论文。
- 验证：R4→R5 的 DOCX ZIP 成员仅 `word/document.xml` 和图 2 的 `word/media/image7.png` 不同；仅第 76 个段落的图注文字变化，其他六张媒体字节、四张表文字及 67 个 Word 数学对象不变。`render_docx.py` 渲染为 8 页；逐页缩略图与第 4 页原尺寸检查未见裁切、遮挡或图片缺失。图示不构成算法性能或实验晋级证据。投稿页数上限、占位作者/基金与初稿缺失问题沿用 R4 的待核验状态。

## R6 按已发表算法图重新划分图 2（2026-10-21）

- 用户质疑 R5 是否符合实际算法论文图式。经公开论文图题与方法段落核对，R5 仍将场景、在线控制和训练压在一条主流程，critic/rollout 关系表达过于简略。本轮仅覆盖 Fig. 2 图片和图注；R5 其他正文和结果冻结。
- 基线：R5 `artifacts/manuscript/r5_20261021/CSCWD2027_He_Zhu_R5_RL_VEC_Figure.docx`，SHA-256 `638d80c71e6d9b8b65382b40cfb126ffbfcf0e19cad0b0ef7483e75f0a0ca767`。新工作副本：`artifacts/manuscript/r6_20261021/CSCWD2027_He_Zhu_R6_Algorithm_Figure.docx`，SHA-256 `f580353e6000ea6a452a9ddcb1faab42e2fa12d9f42b1b9a647301dda656ab8c`。宽版与论文窄版 SVG/PNG 同目录保存；产物目录被 Git 忽略。
- 前后对照：R5 的场景图占图首、控制器与 PPO 训练较笼统；R6 将 Fig. 2 明确分为 (a) 跨 RSU 的未完成 DAG 场景、(b) 观察→编码/预测门控→融合→三头角色分数→五动作合法解码→环境反馈、(c) rollout→central critic→GAE/结果信用→clipped PPO。三头同属一个 actor，训练虚线与在线实线分开；既不表示多实体 agent，也不把价值网络画成直接执行动作的控制器。图注按三分图同步。
- 图式对照来源为本轮新增至 `literature_reference_table.md` 的公开 VEC/PPO/DAG 论文。只借鉴场景/算法分图和 actor–critic 数据流的表达形式，没有复制原图，也没有把这几篇增加到文稿的 25 篇正式参考文献。
- 验证：R5→R6 的 DOCX ZIP 成员仅 `word/document.xml` 与 `word/media/image7.png` 变化；仅第 76 个段落图注文字变化，其他六张媒体字节、四张表文字和 67 个 Word 数学对象不变。`render_docx.py` 输出 8 页，逐页缩略图及第 4 页原尺寸检查未发现裁切、遮挡或图片缺失。图示不提供算法性能新证据；R4/R5 的实验与投稿限制继续有效。

## R7 全文主线与状态—策略—反馈图（2026-10-21）

- 本轮用户要求：图 2 清楚体现状态表征、策略决策、环境反馈与强化学习更新；全文围绕一个真实问题连续论证“问题从哪来—如何建模—为何这样设计—证据得到什么”，并对照同方向顶刊的叙事方式。该新要求覆盖 R4 的摘要/引言与结果叙事和 R6 已固定的图 2；不覆盖四张表、公式、Fig. 3/4 数值、R2 的 25 篇正式参考文献映射或源码说明。
- 基线：R6 `artifacts/manuscript/r6_20261021/CSCWD2027_He_Zhu_R6_Algorithm_Figure.docx`，SHA-256 `f580353e6000ea6a452a9ddcb1faab42e2fa12d9f42b1b9a647301dda656ab8c`。新工作副本：`artifacts/manuscript/r7_20261021/CSCWD2027_He_Zhu_R7_Logic_and_RL_Figure.docx`，SHA-256 `6f6e8ac771b9239f3252e17a19bb14854829aeb2fc11fdeacd1282d434f5fbf8`。图 PNG 和独立证据审查分别为同目录的 `sagh_algorithm_r7.png`、`independent_evidence_review.md`；均在 Git 忽略的 `artifacts/` 内，不随台账推送。
- 先审查、后修文：依 `top_journal_review_policy.md` 按 artifact、provenance、baseline、机制、claim、近邻文献的顺序写成独立只读报告；未调参、未改算法、未启动或干预实验。审查元数据：`reviewed_at=2026-10-21 Asia/Shanghai`、`literature_cutoff=2026-10-21`、`target_venue=CSCWD 2027`、`artifact_run_id=top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730; fair_reward_training_20261021_v2; long_fair_reward_training_20261021_v1 (partial)`、`policy_version=tmc_review_policy_v3_20260621`、起点 Git `76340cc9ea3333bc522a359e415af5687ef12103`、`evidence_level=E1_DOCUMENTED plus targeted raw artifact presence`。正式独立算法优势为 `Unverifiable`；本轮未完成 E2 全包审计。最近邻的两篇 IEEE TMC 论文已在 `literature_reference_table.md`，仅核对其出版摘要及论证顺序，没有复制图或文字，也没有增加正式参考文献。
- 前后对照：R6 引言贡献分为控制器、成本分析、经验描述，但读者仍要自行拼出同一个决策；R7 第 17–20 段把“未完成 workflow 跨 RSU 时何时准备”作为唯一研究问句，并把问题建模、控制器、证据边界按章节相接。II 节末段从既有 DAG/cache/RL 工具推进到具体的联合准备时机缺口；IV-B 从可观察依赖和联系窗口推进到预测门控、合法动作及反馈。Fig. 2 删除与 Fig. 1 重复的 VEC 场景小图，明确三路状态表征、单 actor 的 slow/fast/event 条件角色、五动作 mask/option/guards、环境返回与 critic/GAE/PPO 训练路径。V-B 和结论新增 `fair_reward_training_20261021_v2` 的开发集诊断：SAGH 三 seed 验证回报从初始至最终保持 18.871250，完成率 0.875，且与多个 baseline 同值；同时明示该 run 非 untouched holdout，长期独立 runner 存在概率记账及 `window_class` 缺漏。旧表 III/图 3–4 数值原样保留，只收紧解释。摘要和 IV-B 的信用措辞对应收紧，不宣称已证明学习收益。
- 精准修改的 Word 段落索引：11、17–20、25、60、74、76、82、104、109、120–121。其他段落文字、四张表、25 条正式文献和全部 67 个 Word 数学对象不变；DOCX ZIP 仅 `word/document.xml` 和 `word/media/image7.png` 改变。`render_docx.py` 重新渲染为 8 页，逐页缩略图和第 4 页全尺寸检查未见图文裁切、遮挡或丢失。该检查不证明 CSCWD 页数符合投稿要求；作者/邮箱/基金占位、初版原稿缺失、行动概率问题与正式 holdout 缺口仍待后续独立任务处理。

## R8 八车四站持续请求场景重写（2026-10-21）

- 用户新要求：把先稿改成八车、四站、持续请求场景。此要求与 R7 单车关联工作流、单控制器五语义动作、旧分数表和对应算法图直接冲突；本轮以新场景为准，覆盖原摘要、引言、系统模型、方法、实验、结论、图及表的场景和数值，不把旧奖励 18/32 与新奖励 210.40 拼接。R2 已定的 25 篇正式参考文献条目及编号保持原样；未额外增加文献。
- 基线：R7 `artifacts/manuscript/r7_20261021/CSCWD2027_He_Zhu_R7_Logic_and_RL_Figure.docx`，SHA-256 `6f6e8ac771b9239f3252e17a19bb14854829aeb2fc11fdeacd1282d434f5fbf8`。新工作副本：`artifacts/manuscript/r8_20261021/CSCWD2027_He_Zhu_R8_Eight_Vehicles_Four_RSUs_Stream.docx`，SHA-256 `e7e34fb0035fee93c168c31ebe66a833b88a7372af10de1d00828ac575cf4907`。同目录保存 `sagh_stream_algorithm_r8.png` 和先行独立证据审查 `independent_evidence_review.md`；`artifacts/` 被 Git 忽略，台账提交不会上传未公开论文或原始实验。
- 审查先于改稿：`reviewed_at=2026-10-21 Asia/Shanghai`，`literature_cutoff=2026-10-17`（沿用已有公开文献审查，本轮未重评 novelty），`target_venue=CSCWD 2027`，`artifact_run_id=joint_stream_learning/20261016T162100Z_confirmation; learning_effectiveness_stage1_20261017`，`policy_version=tmc_review_policy_v3_20260621`，修订起点主仓 Git `058fea10da994cdaad204a968baa814af8885781`，仿真实验冻结源码 `35975cb3d73c84ddbabf01f6749bbc2d60c826de`，论文证据级别 `E0_UNAVAILABLE / Unverifiable`（开发原始包可抽查，但缺正式/holdout/support）。本轮只读核对原始确认包的 manifest、audit、检查点、开发输出及实现合同；没有修改算法、选择检查点或新训。
- 前后对照：R7 的单请求状态及五动作控制 → R8 的每车 FIFO 请求释放、跨请求常驻缓存、四 RSU 共享资源、已释放队列可见而未释放请求隐藏，动作依次为执行站点、已知 DAG 节点的 base/adapter 加载、handoff 准备与优先级；R7 的直线式算法图 → R8 的环境释放、状态表征、三段条件策略、环境反馈和 PPO/critic 更新双闭环。R7 的历史旧奖惩表/图及构造案例 → R8 的本合同开发比较表和初始化/32/64 检查点诊断；旧图与 67 个旧数学对象未复制到不兼容合同中。论文标题、摘要至结论均围绕“多请求持续到达下的联合准备时机与共享资源竞争”重写。
- 新结果边界：固定最终 checkpoint 在 3 seed×48 开发条件下均值 210.399715 reward、88.8889% 请求完成率、96.9375 KiB 模型传输/会话；DAG 对照传输更低；SAGH seed 17 从初始化至最终退化；DQN 对照有已诊断 replay 缺陷。两个开发时间会话相邻且属训练 partition，资源成本是合成配置，缺正式/holdout/support，正文仅作开发性描述，不主张独立显著性、普适优势或真实无线开销。源数据为 NGSIM 移动轨迹与 Alibaba DAG 样本，DAG 串行执行，未写成并行任务调度。
- 验证：DOCX ZIP `testzip=None`；1 张算法图、1 张 5 行比较表、25 条原编号参考文献；关键场景/结果/限制词逐项核对；`render_docx.py --emit_pdf` 得 4 页，逐页及图表所在页全尺寸目视检查无裁切、遮挡或表列丢失。该工作稿不是 CSCWD 投稿模板验收或正式实验晋级。第三、四作者/邮箱、基金信息和真实源码地址仍待核定；旧稿中的 `reminderxxx` 占位网址已在正文改写为待核验说明。R7 原稿保留，可回溯。
- 交付文件名调整：按用户要求，另存同字节副本 `artifacts/manuscript/r8_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks.docx` 作为以论文标题命名的交付文件；SHA-256 与 R8 工作副本一致，正文、图表和版式均未改动。原 R8 文件保留作版本追溯。

## R9 全文论证、车联网图与算法闭环核对（2026-10-21）

- 用户要求：两张图逐项对齐八车四站持续请求的原论文及冻结实现；按既往修订要求审查全文，对照同方向顶刊/顶会的一手近邻，把问题来源、工程发现、研究缺口、方法顺序、环境反馈、实验证据与结论串成一条真实主线。该广范围新要求覆盖 R8 的摘要、引言、相关工作表、系统/方法段、图和实验叙事，以及 R2 对表 I 行内容的历史冻结；保留老师要求的五个比较维度和 R2 的 25 条正式参考文献原文及编号，不恢复 R1 额外 7 篇。无新实验、算法或配置改动。
- 基线：R8 按论文标题命名的交付件 `artifacts/manuscript/r8_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks.docx`，SHA-256 `e7e34fb0035fee93c168c31ebe66a833b88a7372af10de1d00828ac575cf4907`。新工作稿：`artifacts/manuscript/r9_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R9.docx`，SHA-256 `64ec9900402f998e9baf4c1d30e71799aedb1ac4f4a1f966106a92f144789875`。同目录有 `vec_system_architecture_r9.png`、`sagh_algorithm_architecture_r9.png` 和先行的 `independent_structure_and_evidence_review.md`；`artifacts/` 被 Git 忽略，论文与实验原件不会随本台账推送。
- 先审后改：独立报告按 artifact→protocol/provenance→统计与 baseline→机制→claim→文献近邻的顺序审查，不在同轮修算法/挑 checkpoint。`reviewed_at=2026-10-21 Asia/Shanghai`；`literature_cutoff=2026-10-21`；`target_venue=CSCWD 2027`，以 TMC/ToN/TC/MobiHoc 作为结构参照；`artifact_run_id=joint_stream_workloads_v3/20261015T172250Z_engineering; joint_stream_learning/20261016T162100Z_confirmation; learning_effectiveness_stage1_20261017`；`policy_version=tmc_review_policy_v3_20260621`；修订起点 Git `5a91bfbf12e953aaf19199584ab223a3b419891f`，stream 冻结源码 `35975cb3d73c84ddbabf01f6749bbc2d60c826de`；正式证据级别 `E0_UNAVAILABLE`，独立算法优势 `Unverifiable`。公开近邻均已在 `literature_reference_table.md`，本轮没有新增论文或向网站上传未公开稿件。
- 结构前后对照：R8 先列组件、再报开发均值 → R9 从工程 grid 的准备过早/过晚双向案例（固定规则 56/72 胜、16/72 负）引出单一联合时机问题，再列已有研究分别覆盖的任务、模型、车辆、连续状态与联合决策轴；明确空缺是本仿真合同的交集，不作绝对首创宣称。系统模型明示每车 FIFO、已发布请求可见/未发布不可见、串行拓扑执行、跨请求 RSU 缓存持久、站内容量 `C_r` 与共享链路；目标是环境原生团队回报而非实测时延/能耗。方法按可行站点→重建共享预约日历→模型对象 admission→受限 handoff/优先级推进，图和正文同序。训练分支单列集中 critic、GAE、观测型 surrogate 与 clipped PPO；冻结源码中的 surrogate-to-action-logit 调整为零，故不声称 surrogate 直接改变在线动作或独立提高性能。
- 图表前后对照：R8 仅一张简化算法流程图 → R9 Fig. 1 单独呈现 8 车、4 个逻辑 RSU、站内 cache/link、移动与两种请求发布模式；Fig. 2 呈现决策时状态、图/候选表征、条件动作、环境返回和训练反馈的双闭环，实线为在线、虚线为训练。R8 简表 → R9 Table I 保留五维逻辑并仅比较当前 25 篇文献中可回溯的近邻；Table II 报 7 个有效内部控制，DQN 因 replay 缺陷不参加排名。25 条 R8 参考文献文字和编号逐项一致；图、表、数值没有把旧单请求实验与新持续请求实验拼接。
- 结果与边界：三 seed×48 重复开发条件的固定最终 checkpoint 均值为 210.40 reward、88.89% 请求完成和 96.94 KiB/会话模型传输；同时列出 seed 17 退化、DAG 对照模型传输 92.58 KiB 更低、757/1078 个可选对象被用且 321.5/880.5 KiB 可选字节未用。两个开发时间会话位于父训练分区，正式/holdout/support 不可用，故结论只确认模拟合同与实现闭环及描述性开发表现；独立统计优势、真实无线成本、模块因果收益仍未证实。
- 验证：用 bundled Python 重建 R9；`render_docx.py --emit_pdf` 渲染得 5 页，逐页原尺寸目视检查，没有裁切、遮挡、图表丢失或跨页拆散 Table I；`pdfinfo` 确认 5 页；DOCX ZIP `testzip=None`，2 张嵌入图、2 张表、25 条参考文献与 R8 完全一致，场景、数值、负例和结论边界逐项检查通过。当前为单栏可读工作稿，不是 CSCWD 最终模板/页数验收；源码公开地址、额外作者和基金细节待核，正式独立测试、强基线修复及消融仍是独立后续研究任务。

## R10 两图统一与旧版车形复用（2026-10-21）

- 用户要求：将当前车联网场景图的车改为“两站两车”旧版图中的画法，并按顶刊算法总览图的分区、线条和版式核对算法逻辑图，使两图风格一致。本轮仅覆盖 R9 Fig. 1/Fig. 2 媒体和必要图注，不改摘要、正文、公式、表格、实验数值或参考文献。基线为 R9 `artifacts/manuscript/r9_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R9.docx`，SHA-256 `64ec9900402f998e9baf4c1d30e71799aedb1ac4f4a1f966106a92f144789875`；新稿为 `artifacts/manuscript/r10_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R10.docx`，SHA-256 `e45b0c1dd5f070d6883b308630635a4ea89da0f32985b4003f8e2a337136937b`。
- 图源与覆盖：两站两车旧图来自 R2 DOCX 的 `word/media/image1.png`，SHA-256 `50dcbd22b230128b4f3c1c3f9f685c3e705e08c929b5de883c19a1aabc0fa312`；按其中浅底、深蓝灰细轮廓轿车重绘八辆，仍保持 R9 的四 RSU、每车 FIFO DAG、两种到达模式和跨请求缓存。R9 的文字方框车辆被替换，图内车数、站数与释放合同未变。R10 Fig. 1 文件为 `vec_system_architecture_r10.png`，SHA-256 `07f70d3aa7b4f3ede82561ad74b2370f1b5596609de5e43db7ae152ca43412a4`。
- 算法图按已列入文献表的 Tang et al., *IEEE/ACM Transactions on Networking* 2024 的公开 Fig. 1 场景/Fig. 2 总览的表达层次作视觉对照，只借鉴“场景与算法分图、输入/表征/策略/学习分区、统一细线与短直角连接”的格式原则，不复制其容器迁移图或算法模块。R9 纵向大色块 → R10 与 Fig. 1 同尺寸比例、同 Times 字体、白/浅蓝灰底、统一蓝灰线宽与箭头；在线部分明确 state→representation→单共享 actor 的站点/预约/模型/切换顺序→environment→feedback，训练部分用虚线分出 rollout→central critic→GAE/observational surrogate→clipped PPO→actor 参数回流。新图 `sagh_algorithm_architecture_r10.png`，SHA-256 `a83d0b59fa83be7ee1023bc488e33123ab94b8ed1f8942205e49a4e3cf10e90f`。图注仅同步 (a)/(b) 与实线/虚线语义；不暗示 surrogate 在执行期直接改动作 logits。
- 验证：R9→R10 DOCX ZIP 仅 `word/document.xml` 和两张媒体改变；全部其他段落、两张表、25 条参考文献与 R9 完全一致，只有 Fig. 1/2 两段图注文字变化。DOCX `testzip=None`；`render_docx.py --emit_pdf` 得 5 页，`pdfinfo` 核对页数并逐页目视检查图文、车形、箭头、图注及跨页位置，未见裁切或遮挡。此为图式修订，不改变 R9 的 `E0_UNAVAILABLE / Unverifiable` 论文证据边界，也不构成顶刊投稿就绪判断。

## R11 按老师版格式排版（2026-10-21）

- 用户要求：仅以提供的 `CSCWD2027何卓文.docx` 对照**格式**，修改当前最新版 R10；不引入样稿的任何文字、数据、作者占位、基金占位、红字或黄标；参考文献统一为 8 pt。格式要求覆盖 R9/R10 的单栏工作稿版式，不覆盖其已固定的论述、图表内容、数值或 25 篇参考文献映射。
- 基线：R10 `artifacts/manuscript/r10_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R10.docx`，SHA-256 `e45b0c1dd5f070d6883b308630635a4ea89da0f32985b4003f8e2a337136937b`。只读格式样稿 SHA-256 `2e85afae2afcb6b0aa6cb0df0a0af8f95149a0a34ccaacb2e9b0508930eff490`。新稿 `artifacts/manuscript/r11_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R11.docx`，SHA-256 `c5a9d8676c0230e8670436eb3e2fd1afe46c450fbe2b3d15ea6045b04b2e9275`；论文在 Git 忽略的 `artifacts/` 内，台账提交不上传稿件。
- 格式对照与实改：Letter 纸、0.75/1.0 英寸上下页边距、约 0.625 英寸左右页边距；通栏居中标题和作者信息，其下双栏正文、0.25 英寸栏距；Times New Roman 24 pt 标题、9 pt 作者/摘要/关键词、10 pt 正文、居中一级标题与斜体二级标题、8 pt 图表标题与表格文字、8 pt 悬挂缩进参考文献。两张原图按样稿的单栏图式等比缩放，表 I/II 置于各栏内；图像字节和图文顺序不变。未复制样稿正文或样稿标注。
- 前后核对：R10 与 R11 的 71 个段落文字、两张表全部单元格文字、两张嵌入图片字节及 `[1]`–`[25]` 文献文字完全一致；仅页面设置、段落/字符样式、图表尺寸和布局变化。R11 所有参考文献文字 run 均为 Times New Roman 8 pt；DOCX ZIP 可读。`render_docx.py --emit_pdf` 输出 Letter 4 页，逐页检查标题、双栏顺序、完整 Fig. 1/2、表 I/II 与文献，无裁切或遮挡。该轮只验证格式及内容未变，不构成投稿模板官方验收或论文证据晋级。

## R12 算法总览图前置（2026-10-21）

- 用户指出算法结构图在第四部分的位置不理想。定位为 R11 第 IV 节中 Fig. 2 排在 IV-A 首段之后；本轮仅将 Fig. 2 及其图注移到 `IV. SAGH DECISION AND LEARNING LOOP` 标题之后、`A. Why the controls are ordered` 之前，使进入方法节时先见状态—策略—环境—学习总览，再读设计理由与模块说明。第 III 节仍承载系统与问题定义，算法内容仍归第 IV 节。该新要求仅覆盖 R11 的 Fig. 2 位置，不改图文、实验或文献。
- 基线：R11 `artifacts/manuscript/r11_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R11.docx`，SHA-256 `c5a9d8676c0230e8670436eb3e2fd1afe46c450fbe2b3d15ea6045b04b2e9275`。新稿：`artifacts/manuscript/r12_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R12.docx`，SHA-256 `f2e33a6eb4e1a06cd4eb425099e35b451b25ab2bfa7fb10476acbaa58ea5292b`；稿件在 Git 忽略的 `artifacts/` 内。
- 核对：除 Fig. 2 与图注的顺序移动外，R11/R12 的 71 个段落文字、两张表所有单元格文字、两张图媒体字节均一致；25 条参考文献仍为 Times New Roman 8 pt。DOCX ZIP 可读，`render_docx.py --emit_pdf` 得 Letter 4 页，逐页及图 2 所在页原尺寸检查无图像裁切、文字遮挡或异常留白。没有修改算法实现或论文证据结论。

## R13 Fig. 2 归入算法解析小节（2026-10-21）

- 用户进一步明确：算法结构图应放在具体的算法解析部分，而非仅紧贴第 IV 节标题。此最新要求覆盖 R12 把图直接放到章标题后的决定；本轮将 IV-A 标题由 `A. Why the controls are ordered` 精确改为 `A. Algorithm rationale and architecture`，保持原 IV-A 首段解释“为何先选站点、再重建预约、再决定模型与切换”的文字，再将 Fig. 2 及图注放在该段之后、IV-B 之前。第 III 节系统/问题定义、IV-B/C 解释、实验和文献不改。
- 基线：R12 `artifacts/manuscript/r12_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R12.docx`，SHA-256 `f2e33a6eb4e1a06cd4eb425099e35b451b25ab2bfa7fb10476acbaa58ea5292b`。新稿：`artifacts/manuscript/r13_20261021/SAGH Control for Continuous DAG Requests in Vehicular Edge Networks R13.docx`，SHA-256 `5dde382c3c6265892168b33bbd5c736fe9326c7be190bbad0fc775b978d0fba8`；稿件仍在 Git 忽略的 `artifacts/` 内。
- 核对：除 IV-A 标题措辞和 Fig. 2/图注的顺序外，71 段文字、两张表单元格、两张图媒体字节均与 R12 一致；25 条参考文献仍为 Times New Roman 8 pt。DOCX ZIP 可读，`render_docx.py --emit_pdf` 得 Letter 4 页，逐页及图所在页原尺寸检查无裁切、遮挡或异常留白。未修改算法实现、实验数值和证据结论。
