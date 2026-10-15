# 持续工作负载与预取机会：修改前独立审查

- reviewed_at: 2026-10-16（Asia/Shanghai，客户端日期）
- literature_cutoff: 2026-10-15（无新增检索，不评价novelty）
- target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621
- Git source: b323355d966abc74693c658c6e69dbec5a404f47；artifact_run_id: artifacts/joint_delivery_control_v2
- evidence_level: 开发原件/事件只读核验；论文E0_UNAVAILABLE，Unverifiable

## 原件与问题

十组固定最终开发结果已独立回收，见joint_delivery_v2_results_review_20261016.md。workload为8个固定DAG，节点数9/17/5/10/7/5/8/9，总70、均8.75；1个base和5个adapter，每车frame0一个workflow，结束后不再到达，窗口120帧且窗口间清空缓存。该合同无法检验持续请求的跨workflow缓存复用，也没有多base竞争。

原48主单元中，full未来节点新准入对象1252个，178个在驱逐/窗末前被同站实际计算使用；833个使用前被驱逐、241个窗末未用。cache相应627/79/416/132。计算使用按rsu_compute/output_ready、实际对象/站及驻留世代追踪，不是严格cache hit或因果交付增益；对象计数非独立迁移样本。证据post_training_review/prefetch_lifetime.json。低使用比例不能单独归因“大任务少”，也不能从full略高的使用比例推导其成本效率更高。

## 修改前结论

需要补工作负载覆盖，并保留预取错误、容量过载和无净收益场景。简单增加DAG长度可能加重既有瓶颈；原tight7/8已违反全交付必要容量上界。机制识别须同时报告可行性/负载、对象首次使用前驱逐、团队节点/请求、延迟和全部链路成本。

本审查不修改算法、旧运行或奖励，不声称新环境必使SA领先。formal/holdout未开启、canonical不变。用户已授权据此另立实现任务。

## 独立实现合同

1. 新opt-in持续请求环境：外生frame到达、每车FIFO、未到达任务对策略不可见；已到达排队可见。当前工作流完成后才启动下一请求，清理该车input/output/版本/拷贝/执行记忆，模型缓存与付费在途传输跨请求保留；reset与不同split间仍清空，禁止跨split带缓存。
2. 单独action/observation contract和带request_id的控制；旧裸v3控制拒绝，不能把旧checkpoint作为新环境的有效算法比较。每帧包含空闲/无active时的真实费用收据；排队等待收费、无请求时不产生幽灵任务，截止保留所有未完成请求。
3. Alibaba原DAG按固定节点区间分层，节点拓扑/输入输出不缩小、不拼造长链。两个base族及异构base/adapter字节为显式合成catalog，adapter依赖身份唯一；manifest标明非实测，并报告完整working-set/最小闭包与容量比。
4. 请求到达/重复model族需求在任何policy运行前固定。单批和持续请求、短/中/长DAG、不同容量按预定义全矩阵运行，不按policy得分筛环境。仅使用原train mobility区间；先报告单规则容量下界与无训练reactive/naive/gain-aware预取对照，不启动RL排名或formal/holdout。
5. gain-aware规则只消费当前/已发布DAG与已到达queue，估计首次使用时刻、驻留/驱逐风险和完整模型闭包成本，保留团队当前资源保护；预测/状态共享保留。诊断规则不冒称SA改进或未来真值oracle。
6. 验证producer→stream→raw事件→summary/独立账本与对象生命周期consumer；历史v3逐事件重放通过后，clean commit再执行完整预声明工程矩阵。正负结果全部保留。可行机制通过后另冻结新训练预算/策略适配，不直接复用本轮分数。
