# 主方法优化候选：机制状态预测与短视野决策

更新日期：2026-10-26。状态：方法筛选与零交互检查完成，候选尚未实现或训练。

## 本轮结论

优先改进 SA-GHMAPPO 的学习式预测分支：让它区分具体 adapter/cache/loading/DAG 状态，然后检验有限短视野决策。只优化主方法是可行的研究范围；共同业务奖励、物理规则、测试输入和 baseline 的合理信息权限应保持可比。不能保证任一候选一定得到奖励优势。

源码中的多时域反事实规划、残差、专家蒸馏与不确定性 ensemble 已存在，不能把再次启用这些组件说成新实现或已验证提升。本轮不改算法、奖励、baseline、checkpoint、论文和正式协议。

## 已定位的输入限制

`sa_ghmappo_core.py` 初始化 `UncertaintyTransitionEnsemble(observation_dim=9)`。`marl_on_policy_trainer.py` 将包装层 observation 送入该分支；`gym_vec_env.py` 的九维原始投影包含缓存数量，却不包含具体缓存对象身份。当前 `predict_learned_transition_targets()` 消费的是一步 TD 预测，不应直接描述成已验证的多步世界模型。

构造两个其他字段相同、缓存分别为所需 `adapter_a` 和其他 `adapter_b` 的状态，调用真实 `_build_raw_observation()` 得到完全相同的九维向量。该见证为零环境交互、零训练；只证明该投影无法分辨缓存身份，不证明实际运行 profile 已启用该分支、真实 reward 已发生差异或这是收益不足的唯一原因。graph actor 另有 semantic-state 输入，不能把此限制推广到整个策略。

本地证据：`artifacts/analysis/main_method_optimization_review_20261026_v1/feature_alias_probe.json`。

## 方法取舍

| 顺序 | 候选 | 项目中的具体用途 | 本轮判断 |
|---|---|---|---|
| 1 | 机制状态条件化预测 | 当前/目标 RSU 的对象驻留、加载剩余量、合法动作、已释放 DAG 的后继需求及预测置信度进入独立预测分支 | 有明确输入缺口见证，先做单因素候选；收益待验证 |
| 2 | 保守短视野规划 | 在同一当前状态比较候选动作的预期原生回报与模型风险，覆盖加载完成等事件 | 借鉴 MBPO 控制模型误差的原则；已有规划需先核对，不能直接套用其 off-policy 算法或保证 |
| 3 | 可中断时间抽象 | 加载期间减少无效切换，在完成、失败、真实 handoff 或预定超时后重决策 | 独立请求级原型有碎片化记录；修改时间/action contract 时须另做生产消费端闭环，不直接改名为原 SA-GHMAPPO 优势 |
| 4 | 辅助预测与专家初始化 | 学习下一次对象需求或完成状态，用训练期因果专家提供初始经验 | UNREAL/DAgger/GOODRL 有方法基础；项目已有类似组件，保留为后续消融，不能一次堆叠 |
| 5 | Potential-based reward shaping | 仅在确认延迟信用问题后，为原生业务回报提供训练信号 | 暂不优先；终止势、截断和 discount 必须正确，评价始终使用共同原生奖励 |

以上是适配建议，不是新算法 novelty、效果排名或顶刊质量判断。

## 下一轮独立实现任务

1. 先绑定唯一 SA-GHMAPPO baseline commit、profile、窗口与预算，记录其是否启用学习式预测；不能把不同分支的结果混合归因。
2. 单独建立版本化模型输入：至少包含请求/节点所需对象、当前与目标驻留/加载状态、剩余服务量、当前可观测负载、已发布 DAG 的依赖与后继对象需求。历史轨迹预测可用；实际未来位置、未发布到达及评价标签不可作为在线输入。DAG 的已知后继结构与未来轨迹真值分开处理。
3. 首轮仅比较旧输入与新输入，保持一步目标、模型容量控制、其余损失和决策路径一致。额外数据/参数/训练成本单列。用真实状态复查输入区分、身份绑定、合法性及模型目标；没有对照收益不进入长训练。
4. 输入候选合格后，单独比较一步与短视野模型使用。视野按物理加载时长事先确定；没有预测时间和置信度校准时不声称多步可靠。模型轨迹不得混作 on-policy PPO 实际 rollout；采样/执行重映射必须匹配实际执行概率。
5. 新 checkpoint/schema 同步检查训练写出、load/read-back、评估、benchmark、manifest；默认 registry 与历史 checkpoint 不覆盖。

建议先预登记三 seed 的固定小预算 development pilot。对照包括原主方法、新输入主方法与冻结强启发式；其他 baseline 不调参或削弱。训练与评估保留全部成功/失败窗口，计入全部专家/模型查询和计算成本，不按 reward 筛选或延长到赢。三 seed pilot 只用于决定是否继续，不能支持显著性或收敛声明。

若后续要证明相对 baseline 的算法机制优势，需要同信息的机制消融；若主方法额外使用预测数据/推理预算，应将该资源差异独立报告。用于 formal/holdout 的窗口按原始 frame/time interval 验证互斥；开发集不改名作独立 holdout。

## 验收与停止条件

- 全体固定窗口的共同原生回报，同时报告 workflow 完成、等待/失败、加载与传输成本。训练辅助奖励不用于跨算法排名。
- 预测须同时验收误差/校准与实际准备兑现；零 handoff 的窗口不能检验预测迁移增益。已有四个开发验证窗口的这项限制来自旧实验，不外推到全部 benchmark。
- 若模型能区分新状态但动作/服务结果未改善，报告负结果并停止该候选扩展；不据此自动添加更多组件。
- 增益若只来自初始化、启发式覆盖或模型在线搜索，应分别归因，不能全部写为 PPO 自学习或图结构贡献。

## 来源与审查边界

- [MBPO，NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html)：模型误差与短 rollout 使用的基础；不是本项目收益保证。
- [UNREAL，作者预印本](https://arxiv.org/abs/1611.05397)：共享表征与辅助学习的基础；本轮正式 venue 未核验。
- [DAgger，AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)：诱导状态分布上的专家数据聚合；不同于普通一次性 BC。
- [GOODRL，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html)：动态 workflow 的图表征与 offline-online 训练；不是移动 adapter-cache baseline 复现。
- [Potential-based shaping，作者全文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)：策略不变性只在定理及目标条件内成立。

`reviewed_at=2026-10-26T02:25:16+08:00`；`literature_cutoff=2026-10-26`；`target_venue=IEEE TMC methodology / development only`；`artifact_run_id=main_method_optimization_review_20261026_v1`；`policy_version=tmc_review_policy_v3_20260621`；`git_commit=e63a4f9`，工作区有先存修改；`evidence_level=E1_DOCUMENTED_WITH_TARGETED_SOURCE_AND_RAW_JSON_INSPECTION`；formal readiness=`Unverifiable`，score=`N/S`。本轮未重审全部历史 manifest/checkpoint，未形成 E2/E3 或新训练结论。本地 `review.json` 保存读取来源的 SHA-256。
