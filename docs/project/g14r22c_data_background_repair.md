# G14R22-C 数据源候选与后台状态定向修复

- `reviewed_at`: `2026-09-23`
- `literature_cutoff`: `2026-09-21`（本轮不评价 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_g14r22c_data_background_acceptance_20260923_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `base_git_commit`: `33d8aa8fba2361e4b0598fe193dda7ceae4816f5`
- `executor_git_commit`: `cc6edfc43f826dd252443897cffe01d9e7508fa1`
- `executor_git_tree`: `aa45e7fed6faf5981e0791b8264172bd3a458cb5`
- `evidence_level`: `E2_TARGETED_IMPLEMENTATION_AND_REAL_RESOURCE_PREFLIGHT; REAL_PUBLIC_CHAIN_PENDING_SINGLE_BACKGROUND_LAUNCH`
- `grant_signed`: `false`
- `token_issued`: `false`
- `holdout_opened`: `false`

## 1. G14R22-B 失败候选的实际身份

失败 child 为
`artifacts/analysis/typed_model_cache_g14r22b_background_acceptance_20260923_v1/real_non_holdout/public_entry_output/staging/scientific_constrained_288mb/child_stderr.log`。
它在首容量 benchmark 的 fairness manifest 文件校验阶段停止，尚未进入 checkpoint load 或 rollout。

| 数据 | logical role | 合法 explicit/data-root 对象 | size / SHA-256 | executor worktree 同名对象 | size / SHA-256 |
|---|---|---|---|---|---|
| NGSIM | `mobility_dataset` / `dataset.mobility.ngsim.vehicle_trajectories` | `/Users/howen/Projects/PPO_MEC/data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv` | `2118175938` / `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c` | `/Users/howen/.codex/worktrees/g14r22b-executor/PPO_MEC/data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv` | `135` / `48c89cbaccd30352e25d7f29e2f2029cd91e95d84aa0821ba1399fdc00e0d054` |
| Alibaba | `workflow_dataset` / `dataset.workflow.alibaba2018.batch_task` | `/Users/howen/Projects/PPO_MEC/data/raw/workflow/alibaba2018/batch_task.csv` | `802261444` / `6346b0726c6e10466a585c67645af807b425b5be091caf410f5e1aff41a270bc` | `/Users/howen/.codex/worktrees/g14r22b-executor/PPO_MEC/data/raw/workflow/alibaba2018/batch_task.csv` | `134` / `14d63f5b68fcba83931563b8bff35d6e22df043b470604e233b1811c04027a3f` |

两个 worktree 对象均是 Git LFS pointer；其 pointer 内的 OID/size 指向冻结真实内容，但 pointer 文件自身不是数据集。
portable registry 对这两个数据资源只允许 `explicit_path`、`data_root`，并已将真实路径解析为 compatible；随后
fairness validator 又独立把 `executor ROOT_DIR / logical_path` 加入候选，因此把不具备 dataset role 的 LFS pointer
作为另一内容候选，触发正确的内容冲突拒绝，但候选资格来源本身错误。

## 2. 最小修复与拒绝边界

- `benchmark_main_results.py` 将已通过 portable registry 的 mobility/workflow resolution 按 logical role 传入
  fairness validator；重复 role、缺 role 或 registry 非 pass 继续拒绝。
- fairness validator 只消费 registry 已允许的 eligible observations，逐个重新读取 size/hash，核对 frozen logical
  resource ID、role、schema、revision、semantic fingerprint、selected path 与 audit identity；显式路径在 compatible
  时保持第一优先，选择方法、候选和 hash 写入 validation report。
- eligible candidates 内容不同、role 不同、selected path 不属于 compatible set、audit hash/size 漂移仍 fail-closed。
  内容相同的合法重复定位允许按冻结 precedence 消歧。不删除真实数据或 LFS pointer，不关闭冲突检测，不增加 fallback。
- benchmark、formal support、dev selection 和 registry-aware cache-policy wrapper 均消费同一 portable dataset audit；
  oracle/audit/analyze 等无 registry 的消费者只新增严格的 Git LFS pointer 分类：pointer 的 OID/size 必须匹配冻结
  dataset identity，且只作为 locator metadata 排除，真实文件之间的内容冲突继续拒绝。未作全局路径替换。

真实 G14R22-B 命令重建的只读 preflight 已对两份完整数据重新计算 SHA-256：portable resolution 与 fairness
validation 均为 `pass`，两份 selected method 均为 `explicit_path`，eligible/compatible candidate count 均为 `1/1`。

## 3. 后台进程身份

G14R22-B 启动快照保存 venv invocation path；macOS Python 启动后 `ps command/comm` 可显示 framework 内实际
Python executable，原始 command string hash 因 argv0 改写产生 false-negative。本轮 host v2 分别保存：

1. frozen Python launch path；
2. resolved launch path；
3. `ps comm` 的 actual process executable；
4. process start time/token；
5. frozen launch-command hash 与 argv-tail hash；
6. live argv0、live argv-tail hash 和原始 live command/hash。

`RUNNING` 要求 PID 对应的 start token、actual executable、允许的 argv0 identity 和 argv tail 同时匹配。仅 PID
存在不足以成功；缺任一证据、PID reuse、birth/actual executable/argv mismatch 均为
`INTERRUPTED_OR_UNKNOWN`。terminal receipt 仍是 `SUCCEEDED`/`FAILED` 的唯一终态来源，UNKNOWN 不触发 retry、
resume、reopen 或第二次 launch。

## 4. 回归、保护和历史统计边界

- 定向 resolver/background/capacity/producer-integrity 回归：`122 passed`。
- 独立只读复审另行执行的定向组合为 `111 passed`，相关 generated-resource/formal-execution/integrity
  为 `98 passed`；复审 verdict 为 `GO_AFTER_FINAL_COMMIT_AND_CLEAN-CHECKOUT_REGRESSION`，不等于 scientific READY。
- 最终 commit 的 clean detached executor 合并回归（含原先由 dirty-checkout 保护拒绝的 13 个 live-consumer 用例）：
  `136 passed`。
- dedicated publication/statistics/environment 回归：`33 passed`。
- toy smoke：通过。
- G14R22-B 失败 root 未修改；旧 opening/execution/terminal receipts 继续声明永久消费，禁止 retry/resume/reopen。
- 旧 G14R22-B capacity-aware 复算原件未覆盖。`g14r21_old_new_capacity_comparison.json` 实际导出：
  `ci95_high` 改变 59/84、`ci95_low` 改变 14/84、inner cluster 改变 84/84；claim transition 为
  `mixed->mixed 72`、`contradicted->contradicted 12`，claim changed 0。
- `independent_recalculation.json` 为 `passed=true`、`mismatch_count=0`、Holm<0.05 为 0；它独立复核 raw/signed mean、
  nullable coverage、W/T/L、exact sign p、fixed-family Holm 和 effect，明确不独立复算 bootstrap CI。因此只能说
  非 CI 复算字段与 production 输出一致，不能把 bootstrap CI 描述为独立双实现复算。

## 5. 单次真实验收与当前 verdict

新 root 必须是
`artifacts/analysis/typed_model_cache_g14r22c_data_background_acceptance_20260923_v1/`，由最终 clean executor
commit 生成 create-only request/scientific/job package。只允许执行 `freeze_receipt.json` 的一次 launch；启动后本轮
不轮询、不创建监控任务、不修复或重试。用户后续回来后再只读集中核对三容量真实 CSV、checkpoint/rollout counts、
capacity-aware statistics、publication integrity 与 terminal receipt。

当前 provisional verdict：`IMPLEMENTATION_AND_PREFLIGHT_PASS / REAL_CHAIN_NOT_YET_VERIFIED / NOT READY`。
只有新 job 的 terminal receipt 为 `SUCCEEDED`，且科学 receipt、三容量 CSV、statistics 和完整性逐文件复核均通过，
才可在后续独立审查中升级；否则准确保留失败或 UNKNOWN，绝不宣称 READY。

冻结身份：request SHA-256 为 `307397b24d99a96b980c78539f96a76931809230767b48c6a029f67c4117ddd2`，
scientific package SHA-256 为 `8e590f8473b7657c090adb72a05bf371e248b183b32afb042e28194aaddaa73c`，
background job canonical SHA-256 为 `a9e532a4761494e946a53eb52398cfdad9f1d58faa6a84b7792c7d2750d83e74`。
