# G14R20-F public startup handoff 隔离验收

## 结论

`PASS_ISOLATED_IMPLEMENTATION_ONLY`。实现提交 `abe92a11d8d5bbd0a3313cbffbf6077f29c14cef`
在 D 实现提交上补齐了受支持的公共同进程启动路径；没有合并 main，也没有改变科学协议或运行真实 v16。
这不是生产批准：production trust 仍未安装，continuation approval 未签发，
`real_execution_authorized=false`。

## 根因与最小修复

E 复现的根因是公共 CLI 直到首次 `verify_approval` 回调内部才构造 `TrustContext`。它没有把 challenge 输出给
独立 custodian，没有有限等待 receipt，也无法证明 qualification 与 execute 复用同一个 context。原正例只能由
测试代码直接持有内部对象完成，因而公共 startup handoff 为 unsupported。

修复只扩展公共启动层：先校验静态输入和固定 executor identity，再创建并保留一个 context；以 flushed JSONL
输出 challenge 和等待状态；在单调时钟上有限等待固定 receipt；用 receipt 到达后的实际可信当前时间执行
`verify_approval`；同一 context 继续 qualification 和本次 execute。独立 qualification 命令退出后不会产生可供
下一条 execute 命令复用的 context，下一进程必须发布新 challenge、使用新 nonce/PID 并取得新 receipt。

## 冻结的公共输出合同

每行是 version `1.0.0` JSON，事件次序为 `challenge` → `waiting` → `qualification` → `handoff` →
`terminal`。等待状态区分 `receipt_pending` 与 `legacy_receipt_ignored`；拒绝和终结都带稳定 status/reason。
成功 terminal 仍明确标记 synthetic 或 production authorization；host handoff 只标记
`unverified_host_report` 和 `custodian_verification_required=true`。

等待默认 30 秒、上限 300 秒，使用 monotonic deadline；等待参数不进入或延长 receipt/approval 有效期。固定位置
启动前已有的 receipt 只记录为 legacy 并忽略。新到达候选只验证一次；schema、签名、authority、checkpoint、
nonce、PID 或时间任一不匹配均 fail-closed，不换 nonce/context、不无限重试，也不删除 continuity 来制造成功。

## 并发与准入边界

startup lock 是安装时预置、固定 inode 的独立锁；宿主从 challenge 前持有到 execute/handoff 完成。它不复用
SingleWriter 或 continuity 内部锁，也不会按 PID/超时删除重建。第二宿主得到 `STARTUP_HOST_BUSY`，不修改第一
宿主的 receipt/continuity；authority 发布 receipt 不需要等待该锁。

准入前禁止科学模块导入、原 run writer lock、原 run/phase/cell ledger/checkpoint 写入。允许的动作仅是已安装
scope 内的 startup lock 获取、receipt 观察、签名 revocation/continuity 复核与必要的防回滚状态推进。本轮这些
动作只发生在 `/private/tmp/synthetic_g14r20_f_public_abe92a1`。高序号已认证撤销状态可在最终拒绝前持久化，故合同
没有作错误的“授权前零锁/零写”承诺。

## checkpoint 交接

宿主 terminal 输出的 checkpoint 只是待核验材料。独立 custodian 使用上一完整 state、签名 revocation 与当前
continuity 独立复算 checkpoint；公共流程不读取 `context.expected` 充当证据，也不允许执行器签自己的 receipt 或
建立 trust root。正常、拒绝或异常结束后，下一宿主和 cold finalize 都必须重新挑战；旧 receipt 因 nonce/PID/
checkpoint 身份不匹配被拒绝。

## 本轮真实执行证据

公共正例由两个真实进程完成：host PID `90007` 输出 challenge；custodian PID `90004` 独立签名并原子发布
receipt；host 的真实 `verify_approval` 后完成 qualification 与一次 synthetic execute。报告还覆盖 missing/legacy
receipt、7 类错误新 receipt、SIGTERM、SIGKILL、两个并发宿主和锁释放后的新宿主，以及 3 次独立 custody
checkpoint。结果为 synthetic dispatch 1；scientific rollout、real v16 dispatch/write、holdout consumption 均为 0。

相邻复验为 targeted `174 passed`，覆盖 startup、production trust、executor boundary、撤销累积/防回滚、锁后
复核、阶段租约、原生 rc=75、retry/finalization、false/missing gate。全仓为 `1534 passed, 0 failed, 0 skipped`；
16 条 warning 是既有 xunit2 `record_property` 兼容提示。smoke、全 Python compile/import 和 `git diff --check`
均通过。

D 的昂贵八阶段完整 synthetic chain 没有无差别重跑。本轮对受影响公共路径执行了真实双进程验收和 174 个相邻
回归；D evidence `35bec632...` 中的完整链和 E `f53f5f0...` 的审计只作为沿用证据，不冒充本轮重现。

## 保护与身份

新 executor identity 覆盖 25 个文件，digest 为
`1c4be453fe6d1dbe14c7cc57c1bc4e7a88685d39f68b6bd13c91d46f3529769d`，绑定实现 commit/tree
`abe92a1...` / `7511ddd...`。这替代 D 的旧 22-file identity，但没有被签发为 production contract。

原科学 worktree 仍是 commit/tree `a6d1fd8...` / `eb8e83c...` 且 clean。原 run 保护复算覆盖 107,320 个对象，
零 mismatch、零 addition；phase ledger 15 行（5 个 completed，末端 checkpoint_freeze/completed），cell ledger
348 行（174 committed）。原 proposal、两个 ledger 和七个用户文件 SHA-256 均与 D 的保护基线一致。

## 保留的真实状态

- origin launch：仅授权至 `checkpoint_freeze`
- 历史独立 release attestation：`unavailable`
- 现时 release 资格方案：未采纳
- production trust：`not_installed`
- continuation approval：`not_issued`
- real execution：`false`

后续人工清单见 `unapproved_next_steps.md`；本实现不自行指定 owner/verifier/issuer，不采纳 release 资格规则，
也不因外部 pending 扩大实现范围。
