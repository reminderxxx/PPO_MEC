# G14R20-F Public Startup Handoff Contract 1.0.0

日期：2026-09-09。状态：实现与 synthetic-only 隔离验收；不是生产安装或真实 continuation 批准。

## 公共操作序列

`scripts/execute_fixed_commit_continuation.py` 的 `qualification`/`execute` 在同一进程内固定执行：静态
proposal/contract/executor identity 与源码内 installation 身份校验；创建一个 `TrustContext`；取得安装预置
`startup_lock_path` 的 fixed-inode 非阻塞独占锁；输出并 flush challenge；以单调时钟有界等待固定
`startup_receipt_path` 的一次新发布；按等待结束时的可信 UTC 调用真实 `verify_approval`；再 qualification；本次
请求为 `execute` 时继续执行。首次资格校验、锁前/后及执行器内的全部 authorization callback 复用该 context。

`compatibility` 保持既有只读行为，不建立生产信任。两条独立命令会创建不同 nonce/PID/context；前一命令回执
在新命令中只作为固定位置遗留 receipt 忽略，新命令必须取得新回执。等待默认 30 秒，允许 0.05–300 秒；等待
参数只控制单调 deadline，不修改 approval、receipt、revocation 的 issued/expiry 或 freshness。

## 并发与文件边界

`startup_lock_path` 是 installation 预置且固定 inode 的空锁文件；宿主不创建、删除、按 PID/时间替换该文件。
锁覆盖 challenge、等待、qualification、execute 和 handoff material 输出，且与 `SingleWriter` 及
`continuity_path + '.lock'` 分离。第二宿主以 `STARTUP_HOST_BUSY` fail-closed，不发布 challenge、不改 receipt、
continuity、原 run 或第一宿主。authority 原子发布 receipt 不获取 startup host 锁。

宿主不删除 fixed-position 旧 receipt。challenge 输出后先记录旧文件 identity；只接受随后一次 create/replace/
content change。新候选一旦出现即只验证一次；schema、签名、authority、checkpoint、nonce、PID、时间或 scope
错误均终结本 context，不换 nonce、不重试。timeout/SIGINT/SIGTERM 在准入前不写 continuity 或原 run；进程
崩溃由 kernel 释放 startup lock，下一宿主仍须新 challenge/receipt。

准入前允许且仅允许：读取静态输入、executor identity、installation record，打开预置 startup lock，观察固定
receipt。真实 verify 可在最终拒绝前持久化已认证的高序号累计撤销 continuity。准入前不导入科学执行模块，
不触碰原 run writer lock，不写原 run、phase/cell ledger、checkpoint 或正式 failed terminal。本轮所有 startup/
continuity 写入只发生在 `/private/tmp` synthetic fixture。

## JSON Lines 输出 1.0.0

每行是 strict JSON，立即 flush，公共字段为 `version=1.0.0`、`event`、`status`：

- `challenge/published`：`request` 精确含 installation、scope SHA、nonce、PID，并给出固定 receipt path。
- `waiting/receipt_pending|legacy_receipt_ignored`：记录单调 timeout 与旧 receipt 非可信 identity。
- `qualification/accepted`：真实 `verify_approval` 结果；不等于生产已安装。
- `handoff/verification_required`：仅是 `trust=unverified_host_report` 的待核验材料。
- `terminal/qualified|completed|rejected|timed_out|interrupted`：唯一终结状态。拒绝码冻结为
  `STARTUP_RECEIPT_TIMEOUT`、`STARTUP_INTERRUPTED`、`STARTUP_HOST_BUSY`、
  `STARTUP_RECEIPT_REJECTED`、`AUTHORIZATION_REJECTED`、`STATIC_OR_OPERATION_REJECTED`。

`qualification`/`execute` 成功码为 0；一般拒绝 2；等待超时 124；SIGINT 130；SIGTERM 143。未进入 startup 的
静态拒绝只有 terminal 行。`compatibility` 为向后兼容继续输出单个既有 JSON 对象。

## checkpoint custody

宿主结束输出 context/nonce/PID/installation scope 与 `checkpoint_sha256`，明确标记为
`unverified_host_report`。它不能签 receipt、不能建立 trust root，主机自报 hash 也不能成为下一进程依据。
独立 custodian 调用受支持的 `verify_handoff_for_custody`，输入其独立保管的完整上一 continuity state、公共
challenge 和 handoff material；函数从固定 installation 重新读取 continuity 与签名 revocation，验证 scope、
序号、累计集合、时间、内容和主机报告后独立复算 checkpoint。custodian 保管返回的完整 state/checkpoint，
再用 `atomic_publish_receipt` 为下一宿主的新 challenge 签署并发布 receipt。

正常完成或授权拒绝后，如 continuity 已推进，均需核验交接；timeout/准入前取消通常没有新 checkpoint，但
不得据此删除 receipt/state。异常崩溃无可信 terminal，custodian 必须直接核验固定状态；失去上一可信 state
时不得签发下一回执。下一阶段、独立 qualification、cold finalize 和 crash 后重启均必须新挑战；旧回执拒绝，
新回执只绑定同一新宿主的 context/nonce/PID/installation scope。

## synthetic-only 验收绑定

`run_fixed_commit_continuation_acceptance.py` 内置源码受控的 `SyntheticAcceptanceBinding`，只能用于新建
`synthetic_*` fixture/run；它通过同一公共 parser/main/startup/`verify_approval` 路径执行一个明确计数的
synthetic dispatch，且固定报告 real v16/scientific rollout/holdout 为 0。绑定不是生产 CLI 参数，不能从 env、
proposal 或 approval 建立 production trust。公共宿主与 test custodian 为不同真实 PID；custodian 从输出和固定
文件复算 checkpoint，不读取 `context.expected`。该绑定和验收入口均进入完整 executor identity。

生产状态保持：`PRODUCTION_INSTALLATION=None`、release unavailable、continuation not issued、
`real_execution_authorized=false`。本合同不修改科学 Protocol 2.9、A proposal 或原 v16。
