# 图特征尺度的独立诊断

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_coordinated_partial_control_v2/seed{7,17,27}/20261015T030133Z_development`；Git：e96f6ac；evidence level：开发初始图表示witness，论文E0_UNAVAILABLE / Unverifiable。主三组尚未全部完成。

## 审查顺序

1. artifact：只读三个full最终128 checkpoint、首dev初始已知70 DAG节点；不重新训练、不改变环境或当前主比较。
2. provenance：`python scripts/audit_joint_feature_scale.py joint_coordinated_partial_control_v2`，输出`artifacts/joint_coordinated_audit_20261015/joint_coordinated_partial_control_v2_feature_scale.json`。
3. 公平性：不是额外获胜试验、没有选择最优checkpoint/窗口，不是整个rollout分布或baseline优势检验。
4. 机制：graph输入大小/32范围0.25–0.71875，输出大小/32范围25–50；图编码tanh绝对值>0.99比例seed7/17/27分别0.547321/0.839286/0.905357。存在可复查的尺度不匹配和大量近饱和激活，可能削弱completed/current等小幅特征；尚未证明其导致排序失败。
5. claim：不得以此宣称归一化后必然胜出，也不能在当前结果回收中改表示而仍沿用同一协议。后续独立实现可将工作量/IO大小改为声明service-frame单位或仅train拟合的稳健尺度，对SA/MAPPO/cache/各消融共同应用并更新checkpoint合同；同时做饱和、概率、重放及同预算比较。当前不执行该实现，不筛选结果。
6. novelty：没有新检索，不作novelty评价。
