# G14R20-I5 最小受限恢复合同

- `contract_version`: `1.0.0`
- `scope`: `one original run / formal_ablation / two ordered cells`
- `status_ceiling`: `READY_FOR_RESTRICTED_RECOVERY_AUTHORIZATION`
- `recovery_grant_issued`: `false`
- `real_recovery_started`: `false`
- `holdout_opened`: `false`

## 固定身份与只读边界

实现只识别原 run `typed_model_cache_evaluation_only_20260913_g14r20_i3_pending`。原 request、project grant、
supplemental authorization、source reference、phase/cell ledger 完整前缀、6 个 committed cell、失败 attempt 1、
staging、held-lock inode/owner、150 个冻结模型及其 selection/freeze/provenance 均为外部不可变对象。每次 dispatch
前重新执行原生 ledger、producer manifest、transaction inventory、marker、150-model source validator；失败只停止，
不改写或修复原件。PID 可见性只记录为观察值，不能据此证明 quiescence 或删除旧锁。

I4 声明的 `artifacts/analysis/g14r20_i4_bundle_root_recovery_20260914/` 在 I5 开始时未出现在文件系统；I5 不创建、
补写或冒充该旧包，而是直接复算底层原件，并在新的验收包中保留这一输入缺口。

## 新 recovery identity

unsigned request 绑定原 request/grant/source 的文件与 canonical SHA-256、两份 ledger 的完整字节前缀、6 个 external
committed cell 的 ID/path/producer/transaction 双层完整性、150 个模型坐标/path/size/SHA-256、新 executor
commit/tree、独立 resolved context、两条命令和精确 phase/cell scope。新旧 executor 身份始终分别校验；旧 grant
不符合 recovery grant schema，不能被复用。

新 run root 必须不存在、与原 evaluation run 和 v16 模型源完全分离，且所有 staging/committed output 都位于新
root。symlink、路径逃逸、同名异内容、原 root/staging 目标、context/commit/tree 混用均在 dispatch 前拒绝。

## 固定执行顺序与冻结失败

唯一顺序为：

1. `formal_ablation-3e9322fac172fcae01f2cc58`：原 attempt 1 保持 `failed_terminal`；新 recovery ledger 记录
   `recovery_attempt=2`。
2. 仅当前项成功 committed 后，`formal_ablation-e40a9c86c8fdbf3b7962a689` 以首次 attempt 1 执行。

两个 process 共享新 initialization marker、cell ledger 和 recovery phase hash chain。任一 child、publication、
handoff 或进程中断失败都会使新 recovery terminal；`automatic_retry_count=0`，不提供 finalize/retry/resume。
已 committed cell 不能再次 dispatch。

## 外部结果与交接

6 个旧 cell 不复制、不 symlink、不重新发布，只以 `origin=external_original_run` 的 hash-bound reference 消费；2 个
新 cell 使用 `origin=new_recovery_execution`。交接 manifest 的逻辑主键是 `(phase, cell_id)`，精确要求 6+2=8 个
唯一对象，禁止重复计数和按性能筛选。

完成两 cell 后只 create-only 输出 `ablation_recovery_handoff.json` 与
`unsigned_followup_request.json`。显式 resolver 可取得 3 个原 formal-controller `benchmark_rows.csv`，供后续
statistics consumer 独立授权后使用；本轮不运行 statistics，也不把 split-root reference 伪装成 legacy single-root
结果。

## 科学不变性

旧/新 formal-ablation 命令仅允许 executor entrypoint、executor mirror 绝对前缀、recovery output、resolved context
和 source-reference copy 路径改变。runtime/fairness mirror 必须保持相同 logical path、size 和 SHA-256；其余解析参数
逐项相同。request 另绑定 Protocol、split/window/catalog、agent order、5 seeds、576 MB capacity、150-model source、
exposure context、nullable metric contract 和未改变统计规则。不得依据已暴露结果调整 endpoint、setting、cell 或矩阵。

## 授权边界

公共 prepare/validate 入口只生成或核验 unsigned request，不创建 recovery root/ledger/lock/staging/grant。公共 execute
入口要求全新的 exact recovery grant 与独立 review；旧 evaluation grant 会被 schema 拒绝。测试 adapter 只能由源码
测试 driver 注入并要求 synthetic fixture/grant，生产 CLI、环境变量和 request 均无开启开关。

`formal_support`、`formal_scalability`、`formal_statistics`、`formal_gate`、completion、training、selection、freeze、
holdout/hidden 均不在本合同权限内。消融交接不自动启动下一阶段，也不构成性能或论文结论。
