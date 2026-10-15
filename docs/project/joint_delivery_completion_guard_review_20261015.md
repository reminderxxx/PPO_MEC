# Delivery v1：离站前完成的错误准备估值独立审查

- reviewed_at：2026-10-15T10:56:29.841342+00:00
- literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：artifacts/joint_delivery_control；Git：cad65092e8805e601834db254e0cc38014e0df26。
- evidence level：bounded implementation witness + incomplete dev records；paper E0_UNAVAILABLE / Unverifiable。

## Artifact 完整性

十组原计划各128×3，source clean cad6509，开跑前已冻结；本次已停止。stop_record记录已完成并写盘训练316episode，最多10个正在进行/未记录episode另列，不将未完成者计成完整预算。部分32回合checkpoint文件已写但manifest尚未完成本次evaluation更新，不能以manifest checkpoint数0宣称没有文件。原始计划、manifest、部分checkpoint/训练/评价和命令日志全部保留，不复用进入v2主表。尚无完整主审计/全seed最终模型，不作算法排序。

## Protocol / provenance

同一40train/16dev、公共GRU及物理费用复制自已验证tail包；本问题是新planner估值错误，不是原始数据/物理reward变化。停止依据下述源代码反例，未依据策略奖励排名或选择seed/checkpoint。

- witness SHA256：8c2d6e797ebc966a05e4e4df2fa7134daa96d8006784d71bd897ea98ca229e53
- stop_record SHA256：8df6ccf7e582fb1423483a9b4e1ea4d14da0ee8b9f441ec4a8b73a80054bfc7d
- planner SHA256：2208941318ebd891d4265af89663dd06fc7daa5738d7ebc63bf6d1a08eae0d9f

## 机制兑现与具体缺陷

既有forecast_state fixture剩余仅最后节点，原站服务投影第4帧完成，预测第6帧离站（source模型已就绪、无竞争、当前计算2帧+输入/输出各1）。新planner使用eta+max(0,t−eta)把本应在离站前完成的节点强行推迟到eta，再给reactive/prepare恢复延迟；state_net错误为+.04，返回b/c/d准备候选。

由此，时间提前发生于工作流已经结束之后，不能获得节点/等待收益，却仍被计成正收益。旧partial路径有整图离站前完成排除，新独立geometry路径遗漏这一条件；节点分段计价也没有保持已交付prefix不变。该有界反例证明实现逻辑错误，不证明当前真实开发窗口出现次数，也不把它写成唯一算法落后原因。

## 公平性与claim边界

v1终止的316个已记录回合和工程16回合单列，最多10个未完成交互不冒充完成记录。全部失败/未完成数据保留，不纳入固定128主对比；停止旧强缓存重放同样保留未完成原件。不得从该不完整trial宣称新机制收益、收敛、全baseline领先或论文晋级。

## 独立后续实现任务

v2须保持离站前预计已交付节点的时间/收益不变；若整个请求可在离站前结束，禁止预测准备。仅对离站后剩余节点施加恢复延迟，纳入可观测原站/目的站当前节点队列；当前adapter若预计在离站前已用完，不赋予其未来预取收益。保留GRU/状态预测和此前资源保护。新增模型contract v2以拒绝v1语义混用，重新从头固定同128×3十组；物理场景/成本/GRU与主panel权重不变，v1不resume或拼接预算。另立实现合同并验证后才开跑。

评分N/S：本次是有界实现审查，不评价论文水平/novelty，未新检索文献；formal/holdout封存、canonical不变。
