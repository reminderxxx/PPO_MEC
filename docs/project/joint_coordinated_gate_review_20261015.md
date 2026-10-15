# 严格完成门控的独立边界审查

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_coordinated_control/seed{7,17,27}/20261015T015625Z_development`；Git：84028c1；evidence level：论文E0_UNAVAILABLE，七配置完整比较尚未回收，整体排序Unverifiable。

## 独立审查

1. artifact：三seed的SA/MAPPO最终128 checkpoint与dev已存在，全部七配置仍在执行。不得用部分表提前晋级。
2. provenance：固定20train/8dev，同资源与预算，无formal/holdout。support_plan.json在full最终回收前声明固定权重预测开关，额外训练为0。
3. 公平性：SA均值58.218/7.125，MAPPO58.071/6.958，仅暂列已完成组的描述值；开发反复使用，无独立性/显著性结论。
4. 机制：固定权重预测开/关24个dev的完整事件流一致，gate候选为0。独立于gate的forecast覆盖影子核验却发现9个离站请求帧：均在offset3120、peachtree:1689、frame46/47/48，三seed各3帧，ETA3/2/1，而状态约6424–6426B需7帧。证据 `artifacts/joint_coordinated_audit_20261015/coverage_shadow.json`。因此必须说“没有通过严格完成条件的候选”，不能说不存在真实迁移或可获利部分预复制。
5. claim：9帧仍只对应3seed下同一个外生事件，不能充当9个独立迁移。尚未证明每个部分复制有正实际收益，机会代理需计增量版本、残余恢复、模型与状态共享目的站链路、竞争和剩余期限。
6. novelty：未做新文献检索/novelty判断。

## 独立实现任务

保留84028c1整轮及旧默认行为；新opt-in partial-preparation版本以相同费用/预算评估：允许预测到站前只推进部分状态、到站后继续版本化恢复；cache-only模型预取和state prepare分开估值，不要求全部内容在ETA前完成。相同条件下从头训练full SA、MAPPO、SA-no-prediction，各128×3seed。它是用户已授权修复的第二阶段，不修改第一阶段gate以追认旧结果，主表按新阶段内部比较。
