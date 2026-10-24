# 持续工作流独立优化执行索引

2026-10-25用户授权计划后直接执行。实现位于独立分支`codex/persistent-workflow-learning`，科学试训版本`c9c98b46e3c71e60be686c7e9be643e104257f96`，不合并覆盖主工作区已有修改。

本轮实现有限typed cache、跨已释放FIFO请求的驻留、模拟加载就绪与主体保护，以及独立命名的最小直接图PPO变体。默认主线、旧曲线、固定文稿与live registry不变；成本和AI模型标签为受控假设，不能称真实部署测量。

89项匹配测试通过；真实资格验证72组/6840步完成。修正首个pilot入口遗留旧策略样本的实现错误后，在新root完成4方法×3seed×3840训练交互，总46080步/180次learn、240条固定开发验证结果。失败现场和保守预算预留保留，不复用旧checkpoint或补预算。完整学习曲线已生成，未启动大规模长训练或访问holdout。

详细结果审查、奖励分解、全量CSV/图及checkpoint只保存在本地Git忽略的`artifacts/analysis/persistent_workflow_learning_20261025_v1/`。入口为`implementation_and_pilot_results.md`、`figures_v1/report.json`和`service_gain_decomposition.json`。候选的稳定服务收益、结构增量及长期收敛仍未证实；下一轮需要匹配消融与动作可控性合同，不能按奖励排名筛选窗口或搜索权重直到获胜。

验证命令（在独立源码目录使用主项目Python）：`python -B -m pytest tests/test_persistent_workflow_env.py tests/test_sagh_subject_binding_runtime.py tests/test_env_contract.py tests/test_cache_capacity_mb.py -q -p no:cacheprovider`；结果89 passed。本地出图核验命令为`PYTHONPATH=/private/tmp/ppo_mec_reward_plot_deps MPLCONFIGDIR=/private/tmp/ppo_mec_matplotlib .venv/bin/python -B artifacts/analysis/persistent_workflow_learning_20261025_v1/plot_and_verify_pilot.py`，create-only输出已完成，勿覆盖重跑。9项全量账本检查及图页查看通过。

reviewed_at=2026-10-25；literature_cutoff=2026-10-25；target_venue=CSCWD2027 development / IEEE TMC methodology；artifact_run_id=persistent_workflow_learning_20261025_v1；policy_version=tmc_review_policy_v3_20260621；workspace_base_git=52adfc9；scientific_git=c9c98b4；evidence_level=RAW_DEVELOPMENT_TRAINING_RECEIPTS_AND_FIXED_VALIDATION_ONLY；formal/holdout/策略收敛=UNVERIFIED，TMC-ready=Unverifiable，score=N/S。
