# 紧覆盖压力场景：独立容量审查

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：joint_tail_control/protocol；训练源1667da3；审计父Git1d0bfb8；evidence level：策略无关必要上界诊断，论文E0_UNAVAILABLE / Unverifiable。

## Artifact / provenance

原件`artifacts/joint_tail_control/capacity_diagnostic.json`，命令`python scripts/audit_joint_tail_capacity.py`及verified_command.log。公共protocol/轨迹/成本与冻结主实验一致，不训练、不读取算法分数来选择窗口。诊断只处理已有8dev区间的base/tight；formal/holdout不评估。

## 必要容量上界

八个原始Alibaba DAG总计最低217个计算槽、193个输入/输出链路槽。对每个请求子集及每帧，用其可达站并集数量和子集请求数的较小值累计资源容量上界；忽略DAG依赖、模型加载、状态恢复与阶段耦合，因此是乐观上界，超过它仍不能完成则全8完成不可能。车端adapter未声明、每RSU计算/链路槽均1已核验；不把缺陷成本加进来制造不可行。

- base：0/8窗口被这一必要条件排除；这不证明全部存在可行调度。
- tight：7/8窗口被排除全8完成。offset3600八请求最低计算需求217，但全部可用计算容量上界仅120，差97；其他受限请求子集也有7–71槽缺口。忽略传输也不可能全完成。
- 与当前已产生1501条原始dev评价交叉检查，被上界排除的场景没有出现8/8完成，未发现反例。

## 解释与边界

这个压力panel主要引入可达计算/链路过载，并不自动创造更多有效预测迁移收益；仅检查“至少一个站可达”和几何离站计数不足以证明研究场景可解。此前准备阶段缺少这一容量检查，场景设计验证不充分。

本轮冻结主表/权重/预算继续保留，过载仍可比较相对完成/队列鲁棒性，但不能把其低完成率全部归因训练不足或拿来证明预测无效；也不能看到结果后更改主表权重。该轮为开发探索，非论文主表。

## 独立后续实现任务

追加一个探索性支持协议，保持原覆盖布局1.2、费用、模型与最终128checkpoint，固定原8dev时间区间，从真实持续车辆中选全程至少2个站可达的高迁移cohort（覆盖离开、位移、ID排序），不得据算法分数筛选。先验证必要容量条件，再对所有八主方法与两support共同评价；不额外训练、不替换冻结base/tight主表，追加计划必须在首次支持集得分前保存。

这个条件检验协同选择/预测准备在有可选资源时的机会，不构成新独立时间样本，也不保证实际有可行8/8调度或主方法领先。如果选择条件无法满足，报告Unavailable，不为获胜搜索覆盖参数。文献novelty未评估。
