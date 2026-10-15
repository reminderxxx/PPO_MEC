# Scheduled预测/预约机制的固定权重独立核验

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_scheduled_control/main/seed{7,17,27}/20261015T053901Z_development`；Git：aa8676a；evidence level：开发固定权重/原始事件support，论文E0_UNAVAILABLE / Unverifiable；八主方法完整排序仍待回收。

## Artifact / provenance

先于训练声明`support_plan_joint_scheduled_control_main.json`，固定最终128三个full checkpoint、全部8dev，切换预测消费而不更新任何参数。checkpoint/内存tensor hash前后一致。8帧GRU供给/控制费用条件保持相同，仅预测使用与准备机会关闭，不把额外训练或峰值选择混入此支持核验。

## 统计与公平性

24个seed-window单元，开启−关闭预测奖励平均+0.0834215，完成请求+0.0833333；1奖励胜、1负、22平，合计多完成2请求。不是新训练的no-prediction消融，不能替代主三组比较。来源相邻时间片/相同DAG，不按24独立cluster推断显著性。

## 机制与费用

- seed27、offset3120：奖励+2.0121328125，完成请求多2，节点66对64；6次prepare各获1实际链路槽并传输1024B，总state帧14对13。准备目的站曾在C/D之间变化，后续队列与其他控制也变化，收益是整个预测分支干预效果，不能全部归到某一条state动作。
- seed17、offset3120：1次prepare实际推进1024B，节点/请求不变，奖励−0.0100166016；保留这个负例，正net proxy不保证真实净收益。
- seed7、offset3120：有候选但无prepare，物理结果相同（诊断候选字段变化不算控制变化）。
- 三seed共7次prepare全部推进，共7168B/7槽；这直接验证本轮动作获得了实际传输资源。与旧轮比率不能作单独预约机制的因果增益，因为预测horizon、尺度、信用和策略同时更新。
- 真实/预测覆盖影子分别24/21请求帧，实际只有1个去重外生离站事件；真实未来只在审计中使用，不进入actor。

## 表示与claim边界

三个最终full在首dev初始70个已知节点的graph encoder饱和（|tanh|>.99）比例均0；缩放后input/output最大0.398171/0.196210。它验证这个表示范围问题已改变，不代表整个rollout/网络从未饱和，也不能单独证明奖励改进由归一化导致。

预测局部收益和预约执行已有证据，但仍有负净收益和极少迁移事件。八算法最终性能、完成率、稳定更新及后期回退须等完整回收；不在本审查轮改算法、加预算或选择窗口。formal/holdout继续封存，不更新canonical。无新文献检索，不作novelty结论。

原始支持：`artifacts/joint_coordinated_audit_20261015/frozen_prediction_probe_joint_scheduled_control_main.json`、`joint_scheduled_control_main_prediction_effects.json`、`joint_scheduled_control_main_opportunities.json`、`joint_scheduled_control_main_feature_scale.json`；对应只读probe/effects/opportunities/feature_scale脚本已可执行。

2026-10-15完整回收追加：八主配置及两项独立support均完成，120checkpoint/30final首dev事件重放通过；完整版奖励高于六个baseline但未同时超过去预测和MAPPO的完成率，完整数值、回退与claim边界见`joint_scheduled_control_results_review_20261015.md`。本文件原事前/阶段描述保留历史时间含义。
