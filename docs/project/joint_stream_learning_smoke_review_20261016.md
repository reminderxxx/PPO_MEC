# 持续请求学习首次软件联调审查

reviewed_at: 2026-10-16 Asia/Shanghai；literature_cutoff: 2026-10-15；target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621；Git: e753892（实现dirty）；evidence_level: smoke only / paper E0_UNAVAILABLE；artifact_run_id: artifacts/joint_stream_learning_smoke。

八配置各seed7×4session×最多32frame的联调训练均完成（32软件训练session），但审计在checkpoint重放初始化失败：`audit_joint_stream_learning.py:124 KeyError: 0`。账本消费段将`cases`由list改为case_id→record字典，后续重放仍用`cases[0]`。属于auditor生产/消费局部变量类型冲突，不能记为完整联调通过；失败原件保留，不报告算法排名。修复时恢复typed protocol case list，仅修消费端，不替换或重用本轮RL checkpoint。新clean-source科学运行尚未启动。

另有独立预测器划分和迁移值字段审查，见`joint_stream_predictor_partition_review_20261016.md`。修复后在新smoke root重新有限联调一次，固定预算不变；不得把重复smoke算正式学习结果或据其奖励调整参数。
