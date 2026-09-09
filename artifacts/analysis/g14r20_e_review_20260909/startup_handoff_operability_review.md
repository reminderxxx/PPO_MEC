# startup_handoff_operability_review

## 结论

`startup_handoff_operability: unsupported`

受审代码的底层同进程机制可验证，但现有公共入口没有一条受支持、可操作的路径完成“输出当前挑战 → authority 签回执 → 同一进程/同一 context 读取 → qualification/execute”。这是公共启动组件缺失的代码问题，不是实际 owner 尚未任命造成的外部事项。

## 固定源码调用链

`TrustContext.__init__` 每次在 `production_trust.py:118-121` 生成随机 nonce 并绑定当前 PID；`startup_request()` 在 123-131 行输出 challenge；首次 `_startup()` 在 133-145 行立即从固定路径读取、验签并要求 receipt 与当前 challenge 完全一致。公共 CLI 在 `execute_fixed_commit_continuation.py:22` 只支持 `compatibility/qualification/execute`；qualification/execute 在 52 行首次调用 `verify_approval`，production route 才在 `production_trust.py:288-293` 懒创建 singleton context，并在同一次调用内进入 receipt 读取。CLI 没有 challenge 输出、等待/IPC 或保持该 context 后继续 qualification 的模式。

因此：

1. 现有公共入口不支持该交接流程。
2. 未来安装必须新增或改造一个同进程长生命周期宿主/启动组件：创建 production context、公开只读 challenge、等待固定来源 receipt、在同一对象内验证，再进入 qualification/execute。两条独立 CLI 命令不能替代该组件。
3. 该组件是授权边界的一部分，必须纳入新的完整 executor identity、代码审查和独立验收；不能只安装 key/pin 后沿用当前 22-file identity。

## 隔离探针

test-only Ed25519 探针没有 production key、真实批准或真实 run 写入：

- 同一 `TrustContext`：challenge 被 test custodian 签回后，`approval_verified=true` 且 `real_execution_authorized=false`，证明底层同进程调用可行。
- 两条独立命令：第一条只输出 challenge 并退出；用该 nonce/PID 正确签署的 receipt 被第二条新进程以 `startup challenge mismatch` 拒绝。
- 缺失 receipt：拒绝且不 bootstrap。
- stale/timeout receipt：以 `startup receipt stale` 拒绝。
- 公共 CLI `--check startup`：argparse 以 invalid choice 拒绝；源码无 `startup_request` 引用。
- 旧 nonce/PID、fork、rollback、新高序号撤销后回退、错误签名等其余边界由本轮 89 项 production-trust 测试及 21 项 native retry 复现通过。

## “测试可行、公共入口未覆盖”差距

正例测试 `TestTrustFixture.restart()` 直接持有 context，调用 `context.startup_request()`、写签名 receipt、再调用同一个 context 验证；这不是公共入口路径。现有 `test_public_cli_refuses_test_trust` 只证明公共入口 fail-closed，没有公共入口正向测试。故不能用直接函数成功替代 operability，也不能把 production installation 仍为 `None` 当作本差距的唯一原因。

## 最小独立修复任务（本轮不实现）

新增最小同进程启动宿主及其公共 contract，且不得接受 CLI/env/proposal/approval 提供 trust root。独立验收至少覆盖：公共入口的真实 challenge/receipt 正例；两命令、退出、fork、旧/错 PID/nonce、timeout/missing/stale/bad-signature receipt 拒绝；并发 startup receipt 竞争；授权前零 science import/lock/run write；qualification 与 execute 复用同一 context；下一阶段与 cold finalize 重新启动/准入；组件及配置纳入新 executor commit/tree/files/digest。测试仍须使用 synthetic key，但必须走真实公共入口和真实 `verify_approval`，不得常量成功或 monkeypatch 授权结果。
