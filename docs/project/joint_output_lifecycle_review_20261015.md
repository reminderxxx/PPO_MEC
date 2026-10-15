# 输出阶段控制合同独立审查

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_coordinated_partial_control/seed{7,17,27}`；Git：58796cc；evidence level：论文 E0_UNAVAILABLE，Unverifiable。

1. artifact：三 seed partial 开发试跑尚未完整结束，保留 checkpoint/原始结果；终止原因是确定性合同缺陷，不依据排序筛选。
2. provenance：20 train/8 dev，正式/holdout未执行。独立 witness 见 `artifacts/joint_coordinated_audit_20261015/output_lifecycle_diagnostic.json`。
3. 公平性：缺陷为三算法共用；旧 partial 轮不得混入修复后的完整比较，须全部三组同预算从头训练。
4. 机制：env允许已计算结果在无adapter的站点交付，控制器却拒绝模型no-op并强制冷启动；机会成本对output_pending已按零模型需求计算，两端矛盾。在途模型选站成本亦仍按全量对象计，忽略已付传输。
5. claim：没有完整三组最终结果，不作算法优劣/收敛判断。修复不改变原始奖励和资源。
6. novelty：未检索新文献、未作novelty评价。

独立后续实现：增加默认关闭的生命周期合同选项，保留旧checkpoint行为；输出阶段允许模型no-op，选站/机会费用使用观测到的在途剩余模型字节。新partial v3轮独立目录，三组各128 episode × 3 seed；旧试跑额外成本单列。
