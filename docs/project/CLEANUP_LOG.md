# Cleanup Log

## 2026-04-22: Artifacts 与旧文档整理

清理目标：

- 只保留 frozen paper protocol 和当前主线 `NGSIM + Alibaba` 可引用记录。
- 删除旧阶段文档、模板目录、toy/tmp/quickcheck/单次 dry-run/偏离主线 stage 产物。
- 将 artifacts 中的关键报告整理到 `ARTIFACT_RECORDS.md`。

保留记录：

- `docs/project/CONTEXT.md`
- `docs/project/PROGRESS.md`
- `docs/project/BUGS.md`
- `docs/project/ARTIFACT_RECORDS.md`
- `docs/project/CLEANUP_LOG.md`

保留 artifact 族：

- `artifacts/paper/paper_protocol_v1_20260409_rerun_20260415_ngsim_v2/`
- `artifacts/benchmarks/main_results/main_results_mixed_informative_20260415_154627_405291/`
- `artifacts/benchmarks/main_results/main_results_full_stratified_20260415_154815_801060/`
- `artifacts/benchmarks/ablation/ablation_mixed_informative_20260416_120513_376004/`
- `artifacts/benchmarks/ablation/ablation_full_stratified_20260416_103353_408954/`
- `artifacts/benchmarks/prediction_robustness/prediction_robustness_20260409_140350_672221/`
- `artifacts/benchmarks/robustness/robustness_20260406_194918/`
- `artifacts/benchmarks/scalability/scalability_20260415_155814/`

删除原则：

- 不删除 `src/`、`scripts/`、`configs/`、`data/`。
- 递归删除前必须确认所有目标路径在 `D:\PPO_MEC` 下。
- 删除后用目录扫描验证旧文档和 deprecated artifact families 是否还存在。

实际结果：

- 删除目标数：122
- 删除文件数：4898
- 删除大小：约 1524.96 MB
- 路径安全检查：0 个目标在工作区外
- 清理后 `docs/` 只保留 `docs/project/`
- 清理后 `artifacts/` 只保留 `benchmarks/`、`paper/`、`training/` 和 `.gitkeep`


