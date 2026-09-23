# G14R22-A 真实 non-holdout benchmark 消费链验收

- `reviewed_at`: `2026-09-23`
- `literature_cutoff`: `2026-09-21`（本轮不评价 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_g14r22a_real_non_holdout_acceptance_20260923_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `evidence_level`: `E1_IMPLEMENTED_PENDING_REAL_LOCAL_COMPLETION`
- `holdout_opened`: `false`

## 只读核对结论

G14R22 的 `scripts/g14r22_non_holdout_scientific_child.py` 直接按公式生成 30 rows/capacity，未加载 checkpoint、
未构建真实 agent，也未执行 policy rollout。原验收保留，身份更正为 `synthetic transaction acceptance`；不得
追认为真实科学消费者证据。

真实 `benchmark_main_results.py` 的 generated-resource consumer 在出现
`--evaluation-model-source-reference-path` 时，强制要求以下三个文件同目录且身份闭合：

1. `resolved_execution_context.json`
2. `evaluation_model_source_reference.json`
3. `evaluation_execution_contract.json`

原 dedicated runner 只生产前两项；原 builder 虽在内存构造第三项，但未写入 command package，runner 也未落盘。
因此旧包可通过 synthetic child，却会在真实 checkpoint resource resolution 前 fail-closed。该缺口不允许以空文件、
复制旧合同或 consumer fallback 绕过。

## 最小修复与真实验收范围

- command package 现在绑定完整 `evaluation_execution_contract`、context 与 source reference；production package
  缺少任一项即拒绝。
- runner 在 atomic publication 前把三项 create-only 写入同一 root，并在 opening 前执行合同/source/context 的
  live commit/tree/hash/path 复验。
- 每个 benchmark producer manifest 在发布前逐文件复算 size/SHA-256，并检查 `benchmark_rows.csv` 已登记；
  row count 必须与 package 声明一致。
- 真实 non-holdout 验收使用既有 formal public split，显式 `--non-formal-rehearsal`、1 seed、1 workflow、
  1 step、12 frozen windows；每个 capacity 运行 `reactive_lru` 与 `sa_ghmappo`。预计 3 个真实 checkpoint load、
  36 learned policy rollouts、36 reactive rollouts、72 raw rows 与 6-row corrected statistics family。
- 结果固定标记 `test_only=true`、`formal_statistics_eligible=false`，不进入正式统计或论文结论。

长验收只能从最终 clean executor commit 启动。完成回执由
`scripts/run_g14r22a_real_non_holdout_acceptance.py` 写入独立 artifact root；本合同不签 grant/token，不提供 holdout
capability，也不读取 sealed holdout performance。
