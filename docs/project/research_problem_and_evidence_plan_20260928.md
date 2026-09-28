# 研究问题结构与最小补证路线

- reviewed_at / literature_cutoff: 2026-09-28
- target_venue: IEEE TMC（质量参照，不承诺投稿或录用）
- policy_version: tmc_review_policy_v3_20260621
- planning_git_base: 40759a8b934a3321e5e6f51c4238432a5e344d9b
- source_experiment_commit: 4dc5adb6f6e436983745a0cf485f5221ba4caf0a
- artifact_run_id: crdcm_performance_matrix_v2_20260928
- supplemental_artifact: artifacts/analysis/research_problem_evidence_20260928/failure_inclusive_cost_audit.json
- evidence_level: E3_REPRODUCED，仅指开发记录成本/完成数重新聚合；不是独立泛化
- verdict: 核心创新与算法优势仍 UNVERIFIED；本文件是分析与补证设计，不是正式实验授权包

## 1. 一个主问题，三个子问题

主问题：车辆跨 RSU 执行依赖型 AI workflow 时，如何在有限模型缓存及传输资源下联合准备
base/adapter 与执行状态，提高端到端完成可靠性，避免把局部命中和提前准备本身当作最终收益？

研究对象是 trace-driven VEC simulation，不是已部署的 LLM serving 集群。
NGSIM 移动、Alibaba DAG、受控模型映射必须分别说明来源与假设。workflow state 不等同 KV cache；
没有真实 deadline 合同前只报告有限观测时域的完成及截断，不能改称 deadline satisfaction。

| 子问题 | 因果链 | 决策与可观察量 | 所需反证 |
|---|---|---|---|
| P1 缓存什么 | 共享依赖→边际占用/重复传输→可用资源→完成 | base/adapter 驻留、依赖闭包、淘汰、失败成本 | 无共享机会时收益应消失或减弱；容量非绑定不能支持竞争收益 |
| P2 何时/在哪里准备 | 移动→目标RSU模型/状态不齐→中断→未完成 | 因果预测、目标准备、state readiness、无效预取与迁移MB | 无handoff、错误预测、昂贵迁移下，提前准备可能有害，必须保留 |
| P3 如何联合控制 | 局部动作→未来完成与资源代价 | 相同可见信息和执行权下比较策略 | 强启发式或flat PPO足够时，不保留复杂结构优越的主张 |

目标应先报告完整向量：completion、failure、transfer、实际migration、conditional delay及coverage。
未来约束形式可为 maximize E[completed workflows] subject to cache capacity / transfer budget；
预算由应用要求或与策略结果无关的资源标定确定。当前未实现硬传输预算，因此不能宣称已求解该约束问题。
不事后增加权重制造赢家；单位成功产出的成本是补充描述，不替代完成率和失败成本。

## 2. 候选贡献与文献边界

2026-09-28更新：研究机制/算法候选保留两项，另新增独立的数据集贡献候选；三者均需分别兑现：

1. 共享模型依赖与跨RSU workflow状态的联合准备价值建模及决策；必须比独立模块相加提供可识别增益。
2. 仅当有必要性证据时，保留结构化controller作为求解方法；三头、图网络和PPO名字本身不是贡献。
3. C-D：经测量校准、可复现且支持机制隔离的车联网AI工作流/缓存合成数据集与生成器。
   尚未生成或发布，novelty与真实性UNVERIFIED；不能只靠重新标注Alibaba节点成立。
   统一提出依据、文献及证据池见 `vec_ai_workload_dataset_design_20260928.md`。

| 已有工作 | 已覆盖的内容 | 我们必须具体证明的区别 |
|---|---|---|
| [TMC dual-dependency caching](https://doi.org/10.1109/TMC.2025.3573379) | 任务/服务依赖、缓存卸载、PPO | 不只是dependency-aware caching；需要跨RSU执行状态与typed共享依赖的明确数学耦合 |
| [S-LoRA, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html) | 多adapter共享base与内存调度 | 共享base是系统基础，不主张发明LoRA共享；研究移动连续执行决策 |
| [POLAR, 2026 preprint](https://arxiv.org/abs/2604.16583) | adapter缓存/路由、双时间尺度学习 | 不能把adapter+双时间尺度写成新颖点；需说明DAG状态与handoff如何改变决策 |
| [MATM, TVT](https://doi.org/10.1109/TVT.2026.3660321) | 预测辅助迁移与资源分配 | 预测迁移不是新颖点；需typed依赖和workflow状态交互的可验证区别 |
| [AWTO, FGCS](https://doi.org/10.1016/j.future.2026.108415) | 边缘AI workflow、模型加载及调度 | 必须展示移动带来的额外问题，不能只换应用名 |

以上均已在 literature_reference_table.md 登记，本轮未重复新增。公开摘要/出版页面支持近邻性，
不支持穷尽式“没有前人”结论；精确创新仍需全文逐项矩阵，未完成前禁用 first/unique。

## 3. 设计—实现差距

机制分支的 design 文档提出 future critical bundle/object 和目标RSU选择，但
crdcm_decision_contract_and_diagnostic_20260928.md 明确实现仍为五动作加28维信息。
可以选择 current fill / predicted prefetch / vehicle fallback / steady offload / handoff prepare，
不能自由选择任意后续关键任务对象或联合分配多个RSU。当前不能写成完整对象级联合优化算法。

先证明这些状态改变了有价值的决策，再决定是否需要扩展动作；不为了符合论文故事直接大改架构。
若真实研究假设必须依靠对象级选择，应单独冻结新的动作合同并对所有对照给予相同执行能力。

## 4. 本轮新增的原始结果复算

独立标准库脚本只读取JSON和完整性清单，不加载模型，不执行环境。
1,056 source files hash/size一致；156 raw episodes重算；3 outer windows。
所有失败episode的传输都计入分子；不删除失败后再比较效率。

| 条件 | 完成/评价 | 总传输MB | MB/完成workflow（含全部失败成本） | MB/成功request（含全部失败成本） |
|---|---:|---:|---:|---:|
| full SA | 18/36 | 7432 | 412.889 | 25.986 |
| signal-off SA | 14/36 | 9410 | 672.143 | 29.498 |
| full MAPPO | 18/36 | 7432 | 412.889 | 29.968 |
| full PPO | 27/36 | 10224 | 378.667 | 28.400 |
| critical-path heuristic | 6/12 | 3268 | 544.667 | 27.695 |

SA/PPO的逐seed完成数分别为9/9/0与9/9/9。PPO按完成workflow口径成本更低，而SA按成功request
口径较低；不能选择有利分母宣称无条件效率领先。这是开发阶段事后诊断，不是新预注册端点。
仅3个外层窗口、历史数据开发复用，不能对这些描述性数字作确认性显著/泛化结论。

现有首因诊断材料还指出 hierarchical behavior distribution 与 canonical per-head surrogate 不一致。
该诊断的性能因果解释尚未验证；中央本轮只独立复算成本与完成数，不把其梯度审计冒充本轮独立重跑。
文件：机制分支 docs/project/crdcm_sa_first_order_diagnosis_20260928.md。

## 5. 按信息增益排序的补证实验

### E0：先验证算法正确性（立即优先，不训练）

在所有31种非空五动作mask、各合法动作、多个固定非退化head logits上比较：
精确聚合分布的 log probability、旧/新ratio、clip objective 与自动梯度。
检查单合法动作零policy梯度、mask归一化消去无关头、概率和为1、存储old logprob一致、
padding/override不冒充actor样本，以及checkpoint版本读回。

当前公式（未mask前）：p0=e0*s1，p1=e0*s2，p2=e0*s0*f1，p3=e0*s0*f0，p4=e1。
mask后的归一化必须包含在采样和更新两边，不能只手工把部分head权重设零就宣称正确。
验收是数学一致性，不是reward变高。零训练、零环境step；失败则先出报告，另轮最小修复。

### E1：一次匹配修复验证（E0通过并单独冻结实现后）

建议上限：legacy SA、corrected SA、corrected MAPPO、unchanged PPO × seeds1401/1402/1403 ×
64 episodes = 768 training episodes，max_steps20；固定update16，不选最好checkpoint。
评价沿用原12 paired units，每模型12，共144 learned episodes；启发式12，共156。
总上限924 episodes / 18480 environment steps。新目录，旧结果只读。

只改明确的behavior/update数学合同；不同时改reward、temperature、模型大小、窗口或训练预算。
baseline重跑用于检测运行漂移；旧模型作为历史对照，不覆盖。暴露seed1403的结果已知，因此这是
有针对性的开发修复验证，不是独立测试。技术正确性不保证性能改善。

读出：completion、包含failure成本的总MB及单位产出成本、逐seed动作/失败、优化一致性、时间/内存。
若正确性未通过，禁止训练；若修复正确但无可靠收益，不自动加seed或预算追胜，停止扩张SA主张。
若只有修复收益而没有新机制收益，写成正确性修复，不能写成研究创新。

### E2：共享base × 状态迁移的机制因果证据（先工作负载资格，后四臂）

选择规则只能看外生request/DAG/catalog/mobility，不看算法胜负。
必须出现多个adapter共享base、跨RSU未完成frontier、实际内存竞争；用对象依赖工作集标定容量，
包括绑定、过渡、充足三档，不能沿用没有竞争事件的场景名称假装有效。

先用共同固定的因果控制规则跑2×2，隔离物理机制主效应与交互；所有分支同一曝光、成本记账、
初始驻留公平且按物理共享/不共享语义构造。再分别同预算训练四臂，区分适应后策略收益。
不得用评估期开关替代四臂训练来声称学习机制的因果效果。

预先报告 sharing与migration的主效应及difference-in-differences交互；正交互不是强行必须为正，
若仅有可加的独立收益，不主张新协同。报告完成、传输、失败、错误预取和真实迁移字节。
负对照至少覆盖无共享、无handoff和充足资源；预测误差/迁移代价敏感性不得按胜负删档。
具体运行量、输入manifest和资源上限尚待E0/E1及资格结果冻结；本文件不授权无限矩阵。

### E3：最简充分方法与泛化（机制成立后）

相同可见信息、action/mask、预算下比较candidate、flat PPO及dependency-aware强启发式。
要声称graph或hierarchy贡献，必须分别移除模块，不用“SA vs PPO”一项同时归因所有差异。
同时报告推理/训练代价与不利场景。强启发式使用等价评价单位，不复制确定性seed制造样本量。

确认性评价必须另建可靠未参与开发的数据；12个互不重叠外层窗口仅为内部最低审查规则，
还需多run覆盖、相关性与功效分析，不把12当统计或期刊录用保证。
旧consumed holdout永久禁用。若没有新独立数据，只能给development结论并如实缩小论文范围。

## 6. 停止、交付与论文对应

- E0对应方法正确性；E1对应修复必要性；E2对应核心机制；E3对应算法必要性与泛化。
- 任一工程异常保留失败现场，不自动重试；科学负结果完整保留，不以失败审计理由排除不利数据。
- 长任务交后台一次启动，写终态回执；不以持续AI轮询等待，不承诺未知wall-clock上界。
- 主问题和候选贡献可进入论文草稿；本文件各阶段编号只用于内部管理，不写入论文研究叙事。
- 当前不能承诺十月前得到正向结论。可以尽快完成有停止规则的诊断、机制判断和诚实初稿。
- 论文全文评分N/S；novelty、独立泛化、核心机制公平归因尚缺，不是paper-ready。

## 7. 本轮复现

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -B -m pytest tests/test_research_problem_evidence.py -q
/Users/howen/Projects/PPO_MEC/.venv/bin/python -B scripts/audit_research_problem_evidence.py --source-root /Users/howen/.codex/worktrees/1aed/PPO_MEC/artifacts/training/crdcm_performance_matrix_v2_20260928 --output-path artifacts/analysis/research_problem_evidence_20260928/failure_inclusive_cost_audit.json
```

输出create-only；复算时另取新输出路径。验证4 passed，源文件1,056项通过；未执行新训练、rollout、holdout。
完整性清单验证非完整仓库回归；本轮没有独立重跑训练、梯度审计或最近邻论文算法。
