# owner_decision_packet

## 用户需要决定什么

1. 是否采纳 `current_dated_release_verification_proposal` 作为**本次** release 资格规则；不采纳则 release prerequisite 保持 unavailable。
2. 指定实际责任主体及允许的兼任关系：技术独立验收、launch/release/continuation 原件认证、trust/key 与 continuity checkpoint 保管、revocation 发布/保管、最终 continuation 授权、必要时 owner/quiescence 现场核验。这里按职责而非角色名计，不要求六个不同的人。
3. 是否另立实现任务修复公共 startup handoff，再设计 production installation；当前 D 不应直接安装后投入运行。
4. 新代码/安装独立验收完成后，是否签发一个有 scope/expiry/revocation ID 的 continuation grant。本轮不代替该决定。

## 拟采用但尚未批准的 release 路径

由有权者先采纳“当前日期独立核验历史发布事实”规则；独立核验者按 proposal 生成现时、有签名且精确绑定 `a6d1fd8`/run/Protocol/scope 的 release certification；release authority 明确接受该记录作为本次资格，而不声称历史时点已有 attestation。随后才可进入 trust installation、新 executor 验收和独立 continuation 决策。

## 信任职责

| 职责 | 必须完成的决定/保管 |
|---|---|
| 技术独立验收 | 审查新的完整 executor 身份与公共启动路径；不因自己写实现而自批 |
| 原件认证 | 阅读 launch/release/continuation 原件，确认字节、来源、主体、原时间、范围、语义覆盖并签认证记录 |
| trust owner / key custody | 决定并保管固定公钥 pin、installation record、可信 UTC、外部 continuity checkpoint；私钥不得进入仓库/approval/CLI |
| revocation custody | 固定 authority/source，发布 fresh signed cumulative state，保管最高可信 checkpoint；不能照抄本地主机状态 |
| continuation issuer | 在 release 资格、安装和技术验收均通过后，独立决定是否签发 bounded grant |
| recovery/quiescence | 如需恢复，核验 exact owner、host/namespace/process tree/no live descendants 并保存可认证原件 |

## 安装会改变什么、必须怎样验收

真实 installation pins/record 和新增 startup host 都会改变 source files、Git commit/tree、executor file set/digest，当前 `876c369` / `00f4c3e` / `8d64e529...e2dd` 不能继承。必须冻结新的完整 executor identity，并重新验收：public startup 正向交接；所有 receipt/restart/fork/replay/timeout/并发负例；Ed25519 原件与撤销；锁前后准入；phase lease/next-phase/cold-finalize；owner/quiescence；test-to-production 隔离；false/missing gate；authorization 前零真实写入；固定科学 commit/worktree/command plan/protected inventory。验收仍不得用直接函数或恒成功 authorization 代替公共入口。

## 必须修复的代码问题与外部事项

- **必须修复的代码问题**：公共入口缺少同进程 startup challenge/receipt handoff，当前只存在内部对象 API 和 test fixture 正例。因此 `startup_handoff_operability=unsupported`，D 的 production-startup 技术验收范围为 fail。
- **待明确授权的外部事项，不是软件缺陷**：是否采纳现时 release 资格规则；实际 owner/verifier/issuer 及兼任；真实 key/checkpoint/trusted-clock/revocation source 保管；原件现实真实性；是否签 continuation grant；真实 owner/quiescence。缺少这些不会被本报告自动重分类为新 bug。

## 本轮有限交接状态

- `technical_acceptance`: `fail`（仅针对 D 所宣称的 production startup handoff / public operational path；共享密码学、撤销、事务与 gate 边界在 synthetic scope 独立通过）
- `startup_handoff_operability`: `unsupported`
- `origin_launch_evidence`: `verified historical task evidence; exact scope ends at checkpoint_freeze; principal legal identity not independently authenticated`
- `origin_release_evidence`: `unavailable as historical independent release attestation; historical Git/push facts verified`
- `proposed_release_qualification_path`: `current-dated independent verification of historical release facts; unapproved proposal`
- `production_trust_installation`: `not_installed`
- `continuation_approval`: `not_issued`
- `real_execution_authorized`: `false`
- `scientific_performance_and_tmc_paper_ready`: `not_evaluated`
