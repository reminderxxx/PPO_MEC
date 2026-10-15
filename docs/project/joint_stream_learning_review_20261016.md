# 持续请求学习适配前审与冻结实施范围

- reviewed_at: 2026-10-16 Asia/Shanghai
- literature_cutoff: 2026-10-15；本轮不新增文献或novelty评价
- target_venue: IEEE TMC（政策目标）
- artifact_run_id: joint_stream_workloads_v3/20261015T172250Z_engineering
- policy_version: tmc_review_policy_v3_20260621
- Git: 6efd8dc61e8a6cd1340aa388b510bb8c1103581f（工程源码868b0e9）
- evidence level: audited engineering diagnostic；paper E0_UNAVAILABLE / Unverifiable

## 审查（只读）

完整性：216/216、416940事件、71757收据、6次重放及输入哈希已通过，原表保留全部正负例。协议：仅两个原train会话、单个long源DAG，模型成本合成；没有formal/holdout。公平性：三条规则共同路由/物理/保护；不是SA算法比较。

机制：gain对象实际使用1012/1079而完整请求平均差0，c070少15节点/1请求。证明已用比例不足以代理团队收益；不能唯一归因任务小或训练少。现有DeliveryControlPPO只接受裸v3控制、缓存候选仅当前DAG、固定120帧尺度和初始活跃请求数；旧episode假设每帧有事件、request_rewards用vehicle键。直接换StreamEnv将遗漏空闲费用、错误索引request ID、丢失queued候选与请求切换边界。旧DQN固定257cache候选，不能容纳两个30节点DAG。

claim：工程通过不等于SA/cache优势；收敛和双指标目标仍OPEN，不能通过修改reward/放宽SA专属mask/选峰值满足目标。novelty N/S。

## 后续独立实现与预算（用户授权自主推进）

1. 新joint_stream_control_v1学习合同，旧checkpoint明确拒绝。各算法同样收到active+已发布FIFO queue图（总节点≤64）、公共异构catalog字节与共享快照；未发布DAG不可见。vehicle记忆在request变化时清除，站位控制两阶段协同，必要链路与当前依赖保护沿用公共v3。预测准备依实际机会与净收益开启，去预测版只关闭预测输入/机会。
2. 所有动作的return为从该物理帧至session末的团队奖励/10/固定8车；包括无actor空闲期、后续已发布/尚未发布请求的事后收益，训练label可用未来，actor/critic禁止使用未发布请求。surrogate监督同单位，仍为控制变量而非附加logit奖励。DQN跨空闲/请求切换的下一次该车决策连接，并用观察到的区间团队费用/尾部MC结算。
3. 原train的5个不重叠480帧block：0/4/8起始为新train，12/16为新development；不使用原dev/formal/holdout。旧被诊断的两个会话只在新train中。源workflow固定原256job池和short/medium/long选取规则不变；long单源限制保留。
4. 每session 3size×queued_batch/stream×1/2family×4096/8192共24条件；train72、dev48。训练日程按预先固定轮换均衡覆盖，不按结果排序。对照8配置：sa_ghmappo、sa_no_prediction、mappo、ppo、dqn、cache_offload_drl、dag_offload_drl、dt_handoff_drl，各seeds7/17/27、64session、每4session更新；总1536训练session、最大737280物理frames。相较旧128×120，单seed预算上限翻倍，以实际env frames/决策数另记账，不声称等计算量。
5. 32/64checkpoint均保存；32只在固定12个development工程probe上观察曲线，64完整48条件固定最终对比，不按峰值挑选。GRU复用原train因果bundle不再拟合；真实费用、reward不改。不以已见结果调整场景/预算。
6. 先局部合同测试和所有方法短联调，再clean commit冻结protocol，启动一次独立后台有限作业，训练后自动完成原始事件/收据/参数变化/checkpoint重放/配对和曲线分析。每20分钟仅状态检查，不逐回合轮询。任何失败保留原件，先另出审查再修复；不自动续训或重跑。
