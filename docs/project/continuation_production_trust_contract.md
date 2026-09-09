# Continuation production trust contract 2.0.0 — G14R20-D

日期：2026-09-09。仅版本化 continuation 授权接口；不是科学 Protocol 2.9 的修改或真实批准。

| C 项 | 软件验证器 | 独立核验者 / 签发者 | 不能由代码证明及真实状态 |
|---|---|---|---|
| PT1 | 只接受安装对象内固定公钥、主体、范围、撤销源 pin；生产安装默认空 | trust owner 认证主体及离线安装记录，独立接受新 executor 后安装 | 身份组织资格不能由自签声明证明；未安装 |
| PT2 | 逐字节核对原件及签名认证记录、作者/原时间、核验者/核验时间/依据、run/commit/release/scope 和覆盖职责 | 核验者阅读原件并验证来源与语义；批准签发者担保核验者资格、原件保管及职责覆盖 | 签名不证明内容真实；launch 仅至 checkpoint_freeze，release unavailable，continuation 未签发 |
| PT3 | 校验固定位置 Ed25519 撤销快照、authority/signer/source/scope/序号/时间/内容摘要；锁保护 continuity 状态并拒绝回滚 | 撤销 authority 及时发布；trust owner 保管启动 continuity checkpoint 与可信时钟 | 新鲜快照不等于全球最新；无法联机证明尚未获知的撤销 |
| PT4 | 锁前及锁后阶段准入复核；阶段内原生事务、一次合法 rc=75 重试及 finalization 可结束；下一阶段/冷启动 finalize 重新准入 | 批准主体接受可能很长的阶段租约 | 不追溯撤销 committed 账本 |
| PT5 | 合同绑定完整新 executor identity、command plan 和安装 identity | 独立验收者接受新精确 commit/tree/files；签发者据此另签 | B 身份不能继承；本轮不签发 |
| PT6 | owner 原件摘要和 quiescence 认证记录纳入批准；恢复仍需 kernel lock 和原 owner identity 消失 | 运维核验 host/namespace/process tree/no live descendants 并保存原件，独立核验者认证 | 声明合法不证明无 descendants；真实恢复证据 pending |

生产安装仅允许由受审源码内固定 installation 对象建立；CLI、环境、proposal、approval 均不能建立根。
对象包含 installation_id/record identity、信任 owner、按职责固定主体与 Ed25519 公钥、精确 scope、
撤销 authority/key/source_id/固定绝对文件位置、max_age_seconds、coordination 状态位置及固定启动回执来源。
安装记录由 trust owner 独立保管、核验并纳入新源码 identity；每次变更必须重新冻结及验收 executor。
本轮仅 test-only fixture 生成临时 Ed25519 key；生产对象为 None，不保存生产私钥。

撤销 producer 发布签名 envelope，内容身份为 canonical message SHA-256。message 包含 version、
installation_id、authority、source_id、scope、sequence、issued_at、expires_at、revoked_ids、untrusted_signers。
序号必须非负整数；同序号不同内容、时间倒退、未来签发、过期、超过新鲜度均拒绝。
撤销集合累积；更高序号也不得移除已知撤销 ID 或不可信 signer，恢复身份必须另走受审安装。
未知/缺失/不可读/签名错误/主体及范围不符均拒绝。撤销所有证据核验 signer 与批准 signer 均检查。批准必须携带非空字符串撤销 ID；null、空白或非字符串均拒绝。

continuity 文件和 startup receipt 只能位于 coordination_root；初次安装由 trust owner 显式准备 checkpoint，验证器不自动初始化。
`TrustContext.startup_request()` 只读输出每次 context 独有的随机 nonce、PID、installation_id 和 scope 摘要。
固定撤销/continuity authority 在独立保管的当前 checkpoint 基础上签署 startup receipt，另含 authority、
checkpoint_sha256、issued_at、expires_at；由固定路径读取并用固定公钥认证。旧进程回执不能匹配新挑战，
fork 继承的 context 也拒绝。测试 producer 只为 synthetic scope 签署回执；生产接口是
`production_context().startup_request()`，未安装时仍拒绝，不能通过回执建立新根。
每次成功读取持久化序号、内容摘要、可信时间，文件 fsync + replace + directory fsync；同 inode 单独锁串行。
运行中保留预期 checkpoint 摘要，文件丢失/修改即拒绝。重启要求独立保管的精确启动 checkpoint 摘要，
不能从当前本地文件自举。每次重启都必须取得绑定新 nonce 的独立签名回执；仅保留旧回执时，即使本地
状态完整回滚到 bootstrap，也拒绝。可信 authority 必须保管已知最高 checkpoint，不能照抄主机提交的摘要；
若失去该依据则不签发回执。新回执绑定 checkpoint 后，本地较新或较旧内容均拒绝。本轮用测试侧独立保管的 checkpoint 和经 Ed25519 认证的新挑战回执模拟交接。
可信 authority 错误认证回退后的外部 checkpoint、可信时钟/管理员被攻破超出保证；必须人工停用并重新核验。
绝不把格式正确的本地文件称为权威最新状态。可信 UTC 由受管主机提供；倒退拒绝，前跳导致过期拒绝。

证据 producer 提供认证记录 envelope：record_id、verifier_id、verified_at、basis、original（path/sha256/
size_bytes/author/recorded_at/source）、scope、roles、coverage。核验者签名覆盖所有字段；批准签名覆盖
完整认证 envelope。original 字节必须存在且匹配；basis/source/coverage 必须非空且由独立核验者负责真实性。
允许同一原件覆盖多个职责，roles 与 coverage 显式说明；不要求三份不同 hash。
所有 scope 精确绑定 run_id/run_root/scientific_commit/release_identity/executor_identity_sha256/phases。
launch 原件范围单独固定为截至 checkpoint_freeze；continuation 范围不能从 launch 推导。
owner/quiescence 额外 scope 绑定 recovery_owner_sha256，owner 原件字节必须对应该 canonical owner 摘要。

未批准交接：未来 release authority 可提出“当前日期独立核验历史发布事实”的方案，必须保留原时间和
当前核验时间，不追认历史批准。本轮不采用该方案，不改变任何真实资格。
