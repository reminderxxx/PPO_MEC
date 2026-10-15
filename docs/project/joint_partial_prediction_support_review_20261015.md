# 部分准备预测机制：固定权重独立核验

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_coordinated_partial_control_v2/seed{7,17,27}/20261015T030133Z_development`；Git：e96f6ac；evidence level：开发固定权重support，论文E0_UNAVAILABLE / Unverifiable；主三组完整排序仍待回收。

## 1. Artifact

三个full最终128模型与24条最终dev已回收后核验，其他比较算法仍训练中。support计划在本轮训练开始前保存在`artifacts/joint_coordinated_audit_20261015/support_plan_joint_coordinated_partial_control_v2.json`；实际结果`frozen_prediction_probe_joint_coordinated_partial_control_v2.json`及off完整事件保留。未更新任何权重、未选checkpoint、未执行formal/holdout。

## 2. Protocol与provenance

同一full权重、环境/费用/GRU供给，仅关闭预测消费/机会准备。三个seed全部八dev，最终128，无额外训练。原文件/内存tensor hash核对不变。后续完整三配置主表必须独立报告，不把此干预当重新训练的无预测消融。

## 3. 公平性与统计

平均奖励开−关为+0.0295703，完成请求+0.0416667；24单元1奖励胜、1负、22平，合计多完成1请求。不构成独立cluster、显著性或跨场景预测优势证据。full训练32/64/96/128的dev奖励均值58.562567/57.902758/58.021530/57.660698，完成7.291667/6.916667/7.041667/6.708333；仅四个点、未证明收敛，也不支持当前失败只因预算不足。主结果保留128，不追选32峰值。

## 4. 机制核验

- seed7、offset3120：奖励+0.7196875、完成7对6、节点67对66。离站前frame46向rsu_c实际传1024B，随后两帧被链路队列阻挡；迁移请求完成109对115，同时其他请求的共享队列顺序也变化。总state链路11对10，不能把总干预收益全部分解为单条state动作的因果贡献。
- seed7、offset3360：三条prepare仅一帧传1024B；完成/送达时间不变，多耗一帧state，奖励−0.01。这是代理净收益为正但实际净收益为负的明确反例，须保留。
- seed17、offset3120：三次prepare全被目的站link_queue挡住，最终奖励/完成相等。
- seed27、offset3120：有机会但没有prepare，最终奖励/完成相等；诊断候选变化不能当实际执行变化。
- 因此部分复制路径已经兑现、预测分支不再必然完全无效，但“共同服务量预约”没有解决未来数据链路时隙分配，net proxy也尚未校准为真实边际收益。后续需要目的站model/state/input/output的可用时隙估计、prepare与实际到站选择一致性、版本增长及局部信用的反事实校准。此审查轮不再改算法或追选结果。

核验产物：`joint_coordinated_partial_control_v2_prediction_effects.json`（逐项原始费用和完成时间）、`joint_coordinated_partial_control_v2_opportunities.json`（真实/预测覆盖，只读真实未来，不输入控制器）。full最终dev未使用车端计算兜底。

## 5. Claim边界

保留预测并按机会/期望净收益选择准备的方向有局部可执行证据，当前不足以声称完整版总体最高奖励或稳定超过MAPPO/去预测。机会来自当前小场景、外生事件少、价格合成；不能通过强制prepare或改奖励制造优势。

## 6. 文献

本轮没有新文献检索，不作novelty判断。
