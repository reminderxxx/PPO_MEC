# Formal Resolved Execution Context Contract

## 状态与边界

- 合同版本：`resolved_formal_execution_context_version=1.0.0`
- 首个消费者：`typed_model_cache_formal_protocol_version=1.5.0`
- Readiness：G14R5 验收通过后才可标记 `READY_FOR_G14C_V6_CLEAN_TRAIN_AND_FORMAL`
- 本合同只修复 execution context 的生产、传递、复验和审计；不改变数据、split/window、catalog、capacity、agent、seed、训练预算、endpoint、support/statistics 或 claim 科学变量。
- 宿主机绝对路径进入 runtime audit 与 full context hash，但不是科学比较身份。
- 普通 formal runner 没有 holdout/hidden 能力；G14R5 不训练、不产 checkpoint、不运行 performance evaluation。

## 唯一 producer

`scripts/run_typed_model_cache_formal_protocol.py` 是唯一 producer。它必须先完成 environment resolution，再一次性解析：

- 显式绝对 Python 与 execution environment manifest；
- clean detached worktree、observed execution commit 与项目 import origin；
- durable run root、protocol path、repository/data/checkpoint/protocol-artifact roots；
- portable registry/resources 与 environment/dependency identity；
- 全部 phase、matrix cell、argv、expected output、timeout、retry 与 resume phase。

v1.5 禁止从 cwd、环境变量、相对 `.venv/bin/python`、子进程 `sys.executable` 或协议候选列表补猜缺失字段。外层未显式提供 Python 或 environment manifest 时，active execution 必须在创建 run artifact 前失败。

## Immutable artifact

外层在 fresh run 中以 atomic create-only 方式写入：

`<durable_run_root>/resolved_execution_context.json`

artifact 至少绑定：execution commit、Protocol ID/version/semantic SHA-256、portable registry、split、window contract、catalog、typed runtime identities、environment/dependency fingerprint、resolved Python、clean/repository/data/checkpoint/protocol roots、resolved command-matrix SHA-256、created-for-run identity 与 canonical context SHA-256。

JSON 使用 UTF-8、sorted-key compact canonical hashing，并拒绝 NaN/Infinity、未解析 `{placeholder}` 与 `/ABSOLUTE/` sentinel。科学身份和 host runtime location 分区记录。context SHA 与文件 SHA 同时进入 phase input hash、run identity 和 phase ledger；最终 integrity scan必须包含该文件。

## Nested consumers

- preflight validator 必须从 `--resolved-execution-context-path` 加载同一 artifact，再展开全部 command templates；不得读取未覆盖的 `default_expansion_context`。
- dev selection、formal ablation/support/scalability 和 formal statistics 的嵌套 Python 启动必须来自该 artifact，并验证当前子进程就是外层选定解释器。
- checkpoint freeze、formal cache policy/controller/gate 没有独立 runtime resolver；其命令由外层已解析 context 展开或显式传入。
- preflight 必须逐项覆盖 phase order、matrix coordinates、argv、expected outputs、timeout、retry、resume phase，并要求 nested expansion SHA-256 等于 outer expansion SHA-256。

`default_expansion_context` 仅保留 portable template/audit 角色。v1.0–v1.4 artifact仍可读取和回归，但 runner 拒绝用旧版本开始新的正式 execution。

## Resume / finalize-only

fresh run 只允许创建一次 context。后续同 run `--resume` 或 `--finalize-phase-only` 必须加载已有 artifact，并复验：

- context/file SHA、run root、protocol semantic identity；
- observed Git root/commit、environment/dependency fingerprint；
- resolved Python与全部 runtime paths；
- outer command matrix SHA 与当前重新展开结果。

缺失、篡改、跨 run、跨 commit、跨 environment、Python/path drift 均 fail-fast。所有 G14C v1–v5 invalid run 的 root、ledger、marker、checkpoint 和 artifact 禁止作为新 execution 输入；G14C v5 尤其禁止 resume、finalize-only 和 checkpoint salvage。

## G14R5 acceptance boundary

Readiness v7 需要 detached Git-clean、无本地 `.venv` 的 Commit A6 候选快照使用共享绝对 Python完成：全命令 dry-run，以及真实非正式 `preflight → tests`。preflight 必须验证 NGSIM `11,850,526` raw rows 对应的 60/60 frozen-window reachability和 outer/nested expansion equality；tests phase必须形成合法 phase-ledger v3 terminal chain。rehearsal root必须标记 `formal=false`、`training=false`、`performance_evidence=false`、`holdout_opened=false`。

达到 Readiness v7 只授权未来独立 G14C v6 task 创建新 run；不表示正式训练、formal、holdout、G14D、G15 或 paper-ready 已完成。
