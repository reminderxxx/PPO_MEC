# 车联网 AI 工作流与缓存数据集：提出依据、设计与证据池

- reviewed_at / literature_cutoff: 2026-09-28
- target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621
- git_base: 40f9f22；artifact_run_id: none_new_dataset_design_only
- evidence_level: E1_DOCUMENTED（新数据集设计）；既有证据分别标注，不继承为新数据真实性
- status: DESIGN_SPECIFICATION；dataset_release=false；novelty=UNVERIFIED
- 内部候选名称：`vec_ai_workload_v0_1`。不是已经发布的数据集、live schema 或正式评估版本。

第一阶段启动记录：`vec_ai_workload_stage_one_20260928.md`。本地资格核查已完成，模型校准待输入；
新数据集尚未生成，D11/D12状态不因设计文档完成而晋级。

## 1. 范围与提出过程

范围固定为车辆关联的 AI workflow 在 RSU 间连续执行，以及模型/adapter、输入或中间数据、执行状态的
缓存与准备。不是通用云端 LLM benchmark，也不直接宣称自动驾驶安全控制能力。

提出过程保留以下可追溯链：

1. P01/P02：已有双依赖服务缓存研究与多adapter服务系统说明，任务依赖与模型共享必须显式表示。
2. P03：车辆移动使资源准备的地点和时机重要；已有迁移研究不是本项目的创新空白。
3. 本地缺口：历史formal只出现单adapter/base对，非绑定容量档重复；旧映射不足以检验共享机制。
4. 当前解析器从真实Alibaba DAG规则生成AI标签、尺寸与统一base；这使实验可运行，但并未校准AI语义。
5. 因此提出可校准的合成工作负载，将来源、兼容关系、移动过程与机制机会分开记录，支持可反驳的实验。
6. 数据集贡献成立与否由真实性、可复用性、近邻区别和验证结果决定，而不是本算法是否获胜。

本地缺口不推出“世界上没有此类数据集”。新颖性仍需全文和benchmark级比较；禁止 first/unique。
问题ID沿用 `problem_literature_traceability_20260928.md`，新增P07表示工作负载真实性缺口。

## 2. 数据贡献候选及论文关系

新增候选贡献C-D：一个面向移动RSU场景、区分共享模型依赖和可迁移执行状态、具有来源追溯与校准验证的
AI工作流/缓存工作负载数据集及生成器，支持机制隔离、负对照与分布变化测试。

它与C-M（联合准备机制）和C-A（机制驱动控制算法）分别验收。数据集有效不证明算法领先；算法领先不证明
生成器真实。C-D尚未实现/发布，论文不得写成“We release a validated dataset”。若最终仅完成规则映射，
降为实验工具，不单列核心数据贡献。格式统一和样本量增加本身不足以构成创新。

## 3. 对象与应用边界

| 对象 | 必须表达的语义 | 禁止替代 |
|---|---|---|
| base/model weights | 模型家族、架构、revision、精度、实际存储与运行内存 | 权重文件大小不等于运行显存/内存 |
| adapter | compatible base revision、格式、rank（适用时）、加载/切换成本 | 不同模态/架构不可因标签相似而共享base |
| input/intermediate data | producer、consumer、内容版本、size、有效期、可复用条件 | 数据缓存命中不等于模型命中 |
| workflow state | 状态内容、版本、owner/session、恢复条件、序列化/传输/恢复成本 | 不自动等同KV cache，也不等同模型checkpoint |

首版优先选择一个可实测的车辆关联应用族：例如非安全关键的路侧辅助事件理解或车载信息服务。
这是待选候选，不是已验证应用。必须先确定节点真实功能与模型可执行性，再绑定移动轨迹。
感知—融合—规划可以作为异构工作流候选，但不能预设都用同一个base/LoRA；闭环控制、循环/动态分支若
转换为DAG，必须记录展开规则及截断，不能声称覆盖全部agentic workflow语义。
首版主轴保持model/adapter + workflow state；输入/中间数据只有完成producer/consumer校验后才晋级live缓存对象。

## 4. 来源及证据池

论文ID L01–L10见追溯表。每项未来数据记录都应绑定证据ID、source revision/hash与转换规则。

| 证据ID | 来源/入口 | 可支持内容 | 不能支持内容与待补项 |
|---|---|---|---|
| D01 | [NGSIM官方入口](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data)；`DATASET_SOURCES.md` | 原始移动轨迹参考；现有provider接入 | 无AI请求、真实RSU association或无线链路；RSU布局/链路为派生或假设；发布许可待复核 |
| D02 | [Alibaba v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018)；同上 | 批处理任务依赖和资源字段 | 不是车载AI节点、adapter ID、状态字节；许可证及可再分发范围待复核 |
| D03 | L04 SLoRA、L05 Punica | 兼容adapter共享base的系统依据 | 不提供本场景请求联合分布，也不提供本机校准值 |
| D04 | L01 TMC、L09 TSC | execution/service双依赖问题及直接近邻 | 不能以两者未在摘要提到某功能证明创新 |
| D05 | L07 MATM、L08 AWTO、L06 POLAR | 迁移、AI workflow、adapter cache/routing近邻 | 场景不同，参数不可无条件移植；全文假设待核 |
| D06 | L10 WfCommons | 真实实例→生成器→真实性验证的方法参考 | 不提供真实车载LoRA工作负载 |
| D07 | `src/data/workflow/alibaba_dag_parser.py` | 现有统一base、task_type/拓扑adapter映射、CPU/memory派生尺寸 | 当前映射是规则，不是AI测量；主仓源码核对，历史每次run仍按其commit/profile判定 |
| D08 | `g14s01_innovation_algorithm_diagnosis_20260928.md` | 历史共享机会和容量非绑定缺口的归档审查 | 不推出所有场景无价值；本轮未重散列原formal全量 |
| D09 | `/Users/howen/.codex/worktrees/1aed/PPO_MEC/docs/project/mechanism_factorial_completion_and_launch_repair_20260928.md` | 48-episode机制pilot描述性主效应及零交互报告 | 3窗口，不是独立泛化；本轮引用报告，未独立重算全部pilot |
| D10 | `artifacts/analysis/research_problem_evidence_20260928/failure_inclusive_cost_audit.json` | 既有开发结果失败成本重聚合 | 不能用成本/赢家反向选择生成器分布；不是新数据集结果 |
| D11 | 待生成：模型/adapter/状态实测profile | 应提供同一设备上的加载、执行、恢复成本及重复测量分布 | 当前UNAVAILABLE，禁止填文献数值伪装本地实测 |
| D12 | 待生成：独立真实性与许可审查 | 应提供留出校准验证、可复现性与许可结论 | 当前UNAVAILABLE，不能发布READY数据集 |

本池是来源索引，不是新artifact integrity manifest；D11/D12不存在不应造占位结果文件。
文献参数逐项附页码/表号、硬件、单位和适用范围；尚未全文提取的只作定性依据。

2026-09-28来源池增补见 `vec_ai_public_resource_recovery_20260928.md`：D13为DriveLM驾驶图QA语义候选，
D14为V2X-Seq车路序列候选，D15为旧G11 BurstGPT/Mooncake/Qwen-Bailian请求与prefix复用候选。
均未下载验收；优先复用而非自拟全部语义。D13/D14论文已登记主文献表，D15原文献/来源保留。

## 5. 生成与校准路线（待实现）

1. 应用选型：确定节点功能、可运行模型及兼容族；冻结不依赖策略胜负的选择理由。
2. 分层提取：移动从D01、结构参考从D02、AI资源来自实际catalog/profile；不改写原件。
3. 小规模profiling：固定硬件、软件、精度、输入shape/token长度，测冷/热加载、推理、adapter切换、
   state序列化/恢复；分别记原始样本与不确定性。不能仅以下载bytes推导执行时间。
4. 语义装配：按任务功能与I/O类型匹配模型，不再以拓扑位置直接宣称真实AI功能。
   保留输入大小—计算时间—状态大小等条件关系；若无证据，标为assumed并设置敏感性范围。
5. 时空融合：声明车辆与workflow到达的关联规则、时间单位及RSU映射；独立随机拼接仅是一个假设场景。
   request到达与外生轨迹不依赖策略；动作影响实际完成/请求执行，不能据此改写外生需求。
6. 生成并冻结：生成器commit、参数来源、seed、source hashes、所有筛选与排除原因、split固定后再评估。
7. 发布验证：从现有package producer到loader、environment、CacheEvent、CSV、statistics验证；
   不旁路既有完整性要求，不把新schema只写在producer。适配先局部测试，旧正式路径保持不变。

每个字段至少带：`provenance_kind` = measured / literature_supported / derived / assumed；
`source_ref`、`transform_rule`、`unit`、`uncertainty`、`applicability`。literature_supported不等于实测。
这些是设计字段，尚未冻结为runtime接口；实施时逐项映射现有包，优先复用，避免建立平行执行平台。

## 6. 最小数据包结构建议

| 逻辑部分 | 必需内容 |
|---|---|
| dataset card / source manifest | 使用范围、版本、来源hash、许可/隐私、已知偏差、不可用场景 |
| workflow templates | 节点语义、I/O类型、DAG/展开规则、所需模型、应用时限来源或仅观察时域 |
| resource profiles | model/adapter兼容图；weights、runtime memory、数据/状态bytes分开；实测条件 |
| scenario records | 原始移动区间、RSU/网络假设、到达与模型需求、资源档、负对照标签 |
| generator specification | 确定性seed、条件分布与依赖、参数证据、排除规则及commit |
| split and evaluation spec | 开发/校准验证/测试来源、共享原始祖先、统计单位、预算与指标 |
| validation report | 语义/物理约束、真实性、重建复现、负对照；无结果不得填PASS |

禁止把模型权重、真实原始数据、私有grant上传作为默认发布步骤。许可证未闭环时仅提供生成代码、
允许发布的metadata和本地用户供给数据的重建说明；分支push不等于dataset公开发布。

## 7. 三类场景与数据集验收

| 场景族 | 设计用途 | 必须保留的边界 |
|---|---|---|
| calibration-aligned | 测量支持范围内的典型负载 | 不冒充实际车辆联合观测；拟合数据与验证数据分离 |
| controlled stress | 改变共享率、handoff、资源紧张、状态成本和预测误差 | 范围有来源/明确假设；不得按candidate胜负删档 |
| negative control | 无共享、无handoff、容量充足、昂贵迁移、不兼容模型 | 可出现零收益或负收益；不强制任何算法获胜 |

数据质量与方法性能分开验收：

- 语义正确：DAG可执行，边两端I/O兼容，adapter/base兼容，数据有效期与状态恢复满足约束。
- 物理正确：单位一致，无负时间/bytes，共享对象不重复计费，非共享复制真实占用；迁移与重算公平记账。
- 真实性：以未用于拟合的profile检查加载/执行/状态成本分布及条件相关性，报告误差和外推失败。
  样本量、误差容限在profiling设计阶段预先确定，本文件不虚构阈值或完成数量。
- 可辨识：记录模型ready但state不ready及反向失配、等待、实际迁移、错误预取、completion与全部失败成本。
- 可复现：相同输入/seed重建身份一致；不同seed不能伪称新增独立原始轨迹。
- 独立性：按原始segment/run/time及模板/模型家族标记共享祖先，区分同分布生成测试与跨场景测试。
  旧consumed holdout禁重开；合成样本增加不修复真实trace独立性不足，也不增加原始窗口cluster数。
- 公平性：任何policy不参与场景选择；baseline同可见信息/动作/训练预算，未来信息只供明确oracle。

## 8. 对论文的新增证据要求

| 候选主张 | 需要的结果 | 当前状态 |
|---|---|---|
| 数据集比旧规则映射更可信 | 实测校准、未参与拟合的验证、语义兼容及来源链 | UNVERIFIED |
| 能揭示车联网特有准备矛盾 | 同一工作流移动/非移动对照，模型与状态失配及代价 | UNVERIFIED |
| 联合决策有额外价值 | 独立/顺序/联合控制，同权同预算；不同场景收益及失败 | UNVERIFIED |
| 新算法有效 | 强基线、模块消融、独立评价、计算成本和统计 | UNVERIFIED |

下一个实施交付应是“一个可运行应用模板＋测量协议＋旧新字段对应＋不含性能筛选的生成器小样”，
不是先跑大训练矩阵。未完成兼容性与校准，不扩大到多应用/多模型以制造样本量。
本设计不改动算法、旧实验或执行授权；无新数据发布、科学rollout和性能结论。
