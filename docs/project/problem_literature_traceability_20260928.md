# 研究问题—论文—补证追溯索引

- reviewed_at / literature_cutoff: 2026-09-28
- target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621
- repository_base: 46fdf722140496c4534babc63a2adcc211aad1fa
- artifact_run_id: research_problem_evidence_20260928
- evidence_level: 官方发表页面/摘要核验，不是全文复现或性能证据
- verdict: 机制及算法创新仍UNVERIFIED

## 等级与维护口径

核心论证优先正式发表的领域顶刊/顶会；发表出处与目录等级分别记录。
内部A-Core/A-Adjacent不是CCF A、JCR Q1或中科院一区。用户未指定机构目录版本，暂按CCF口径核验
“A会”，不能保证等同学校目录。CCF分类页本次无法读取，未完成2026最新目录逐项核验。

TMC按本项目顶刊目标纳入核心；NeurIPS正式论文纳入高水平算法依据。CCF官方专委会2023简报明确
NeurIPS为推荐A类会议，但这是历史证据，不冒充2026目录或具体track认定：
https://tc.ccf.org.cn/upload/resources/file/2023/03/28/de8f23c2a54dbaf03da495399515a4da.pdf

MLSys正式发表已核验，但不自动写为CCF A。TVT/FGCS保留直接近邻身份，不统一抬为顶刊。
POLAR以本轮可验证的arXiv页为依据；作者页accepted不代替正式出版。
不能为满足等级要求删除直接相关的低级别/预印本近邻，否则创新性审查不完整。

每个问题与文献保留固定ID。未来实验引用问题ID、假设和文献ID，追加结果与失败边界，不覆盖旧结论。
摘要不能证明未提及的机制不存在；全文页码/公式未核验时不编造。论文编号仅用于内部管理，投稿使用正式引用。

## 文献登记

| ID | 论文/持久入口 | 出处与级别状态 | 核验范围与用途 |
|---|---|---|---|
| L01 | [Dual Dependency-Aware Collaborative Service Caching and Task Offloading in Vehicular Edge Computing](https://doi.org/10.1109/TMC.2025.3573379)；[IEEE原页](https://ieeexplore.ieee.org/document/11014496/) | TMC 24(10),10963–10977,2025；核心顶刊 | 官方摘要确认双依赖、GGRN、分层缓存与PPO；具体公式与实验细节待全文提取 |
| L02 | [RUDDER: Return Decomposition for Delayed Rewards](https://papers.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | NeurIPS 2019正式论文；A类历史证据见上 | 回报分解与reward redistribution；不能据此声称本项目已实现或具有策略不变保证 |
| L03 | [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html) | NeurIPS 2022 **Datasets and Benchmarks Track**，不省略track | 官方摘要支持强基线与实现敏感性；不是我们场景性能的证据 |
| L04 | [SLoRA: Scalable Serving of Thousands of LoRA Adapters](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html) | MLSys 2024正式论文；不标未经核验的CCF A | 共享base、多adapter与内存管理；GPU paging不等同RSU workflow迁移 |
| L05 | [Punica: Multi-Tenant LoRA Serving](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html) | MLSys 2024正式论文；不标未经核验的CCF A | 单份base服务多LoRA、批处理与GPU调度；支撑对象模型现实性 |
| L06 | [POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving](https://arxiv.org/abs/2604.16583) | 2026预印本；正式出版状态本轮未闭环 | 缓存/路由双时间尺度；adapter+双时间尺度不能单独算创新 |
| L07 | [Mobility-Aware Assisted Deep Reinforcement Learning for Collaborative Task Migration and Resource Allocation in Vehicular Edge Computing](https://doi.org/10.1109/TVT.2026.3660321) | TVT 2026；直接相关补充 | 预测迁移近邻；保留既有表记录，本轮未重做全文核验 |
| L08 | [AWTO: A latency-optimized task offloading scheme for LLM-driven agentic workflows on heterogeneous edge](https://doi.org/10.1016/j.future.2026.108415) | FGCS 2026；直接相关补充 | workflow/model-loading近邻；本轮出版社直读403，保留既有记录，不伪称全文完成 |

## 问题—机制—实验对应

2026-09-28补充：数据集提出依据、D01–D12来源池、对象边界、校准与验收设计统一见
`vec_ai_workload_dataset_design_20260928.md`。新数据贡献为候选，未生成/发布。

| 新增ID | 论文/持久入口 | 出处与核验 | 用途与限制 |
|---|---|---|---|
| L09 | [Intelligent Cooperative Computation Offloading and Resource Allocation for Dual-Dependency Tasks in Edge Computing](https://doi.org/10.1109/TSC.2026.3709905) | TSC 19(4):2843–2856,2026；官方摘要/出版信息已核验 | 双依赖联合优化直接近邻；全文差异待核，不作为新生成数据真实性证明 |
| L10 | [WfCommons: A framework for enabling scientific workflow research and development](https://doi.org/10.1016/j.future.2021.09.043)；[官方项目](https://wfcommons.org/publications) | FGCS 128:16–27,2022；出版社/官方项目；不标A会或顶刊 | 真实实例与合成生成器方法参考；不提供车载AI联合分布 |

| 问题ID | 已找到的问题 | 文献 | 待证明的区别 | 补证与失败判据 |
|---|---|---|---|---|
| P01 | cache hit不保证依赖workflow完成 | L01、L08 | typed共享依赖与执行状态如何共同影响剩余DAG | 共享×迁移四臂；报告completion及failure cost，只有hit增长不成立 |
| P02 | 多adapter共享改变边际空间/传输 | L04、L05、L06 | 动态RSU决策增益，不是发明共享base | 无共享/非绑定容量负对照；仅节省重复存储不足算法创新 |
| P03 | 准备先付成本，收益跨handoff才兑现 | L02、L07 | 场景特定归因是否改善准备决策 | 先确认真实延迟收益链；与同信息PPO及去归因项比较，不给actor future truth |
| P04 | 多头行为采样与更新可能不对应 | L03只支持强基线动机 | 正确性修复不是创新 | 31种非空mask及合法动作的概率/ratio/gradient；本地源码证明，不能借文献背书 |
| P05 | 五动作尚不能任意选择后续依赖bundle | L01、L06 | 说明是准备时机控制还是对象级优化 | 状态—动作—执行表，对照相同权限，不用设计文字替代实现 |
| P06 | 总传输低可能因失败多；3窗口复用不足泛化 | L03支持公平对照；事实来自本地数据 | 保留失败成本和独立评价 | 成本/完成/coverage同报；独立数据不足只能称development |
| P07 | 批处理DAG上的AI标签及资源映射尚未校准 | L04/L05兼容共享、L08应用近邻、L10生成方法；D01–D12来源池 | 车联网AI工作负载的语义与测量校准，不冒充联合实测 | 数据集质量独立于算法胜负；校准验证、负对照、来源祖先与许可审查 |

## 本地原件

- 设计与停止规则：`docs/project/research_problem_and_evidence_plan_20260928.md`。
- 主文献表：`docs/project/literature_reference_table.md`；按标题/DOI去重。
- 中央复算：`artifacts/analysis/research_problem_evidence_20260928/failure_inclusive_cost_audit.json`。
- 原矩阵：`/Users/howen/.codex/worktrees/1aed/PPO_MEC/artifacts/training/crdcm_performance_matrix_v2_20260928`。
- 矩阵审查：`/Users/howen/.codex/worktrees/1aed/PPO_MEC/docs/project/crdcm_performance_matrix_v2_independent_review_20260928.md`。
- 首因诊断：`/Users/howen/.codex/worktrees/1aed/PPO_MEC/docs/project/crdcm_sa_first_order_diagnosis_20260928.md`。

后续全文阅读追加页码、公式、实验节与自己的推论，区分原论文结论/本项目推论/未验证猜想。
本轮只改文献与追溯记录，未执行训练、rollout或holdout，不产生新的paper-ready判断。
