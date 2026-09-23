# G14R22-A 真实 non-holdout benchmark 消费链验收

- `reviewed_at`: `2026-09-23`
- `literature_cutoff`: `2026-09-21`（本轮不评价 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `failed_artifact_run_id`: `typed_model_cache_g14r22a_real_non_holdout_acceptance_20260923_v1`
- `failed_executor_commit`: `ce5e9e02c92e9e3a723a8bf45bead531518dfe4c`
- `failed_executor_git_tree`: `389076e110194d21e80f857cf6d3400cd9409bc3`
- `next_artifact_run_id`: `typed_model_cache_g14r22a_real_non_holdout_acceptance_20260923_v2`
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
- 真实 non-holdout 验收使用既有 formal public split，显式 `--non-formal-rehearsal`、1 seed、冻结的 3 workflows / 22 steps、
  12 frozen windows；每个 capacity 运行 `reactive_lru` 与 `sa_ghmappo`。预计 3 个不同 checkpoint、108 次 learned
  policy rollout、108 次 reactive rollout、216 raw rows 与 6-row corrected statistics family。缩减统计矩阵不声明
  正式 15-agent order identity；生产 84 项 family 不变。
- 结果固定标记 `test_only=true`、`formal_statistics_eligible=false`，不进入正式统计或论文结论。

长验收只能从最终 clean executor commit 启动。完成回执由
`scripts/run_g14r22a_real_non_holdout_acceptance.py` 写入独立 artifact root；本合同不签 grant/token，不提供 holdout
capability，也不读取 sealed holdout performance。

基于首次 executor 生成、现已因后续配置修复而 superseded 的 unsigned holdout request 位于
`artifacts/analysis/typed_model_cache_g14r22_holdout_execution_contract_20260923_v2/`：request SHA-256=
`91f9568b93a590eb975afb0a2a281e5c3334d95ec9fadb18a3474ad348499069`，command package canonical SHA-256=
`e4b924282274d9aa8ec75374ae44bfea5404355900134bfa84a6333b93c2fdf3`。v1 原包保留并显式 superseded；v2 仍
`grant_signed=false`、`execution_authorized=false`、`holdout_opened=false`。下一份 unsigned 包必须绑定修复后的
新 executor commit/tree，不沿用 v2 hash。

首次真实验收 root `typed_model_cache_g14r22a_real_non_holdout_acceptance_20260923_v1` 在第一容量科学 child 前
因测试命令 `max_workflows=1` 与冻结 fairness manifest 的 `3` 不符而失败，`return_code=1`；`max_steps=1`
同样与冻结值 `22` 不符。该 root 的失败回执、child stderr 和 ledger 保留，不恢复或重试。此失败未执行 checkpoint
load、rollout 或统计；正式 holdout root 仍不存在。下一次验收使用新的 test-only root。
