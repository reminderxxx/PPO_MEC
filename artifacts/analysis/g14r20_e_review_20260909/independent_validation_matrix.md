# independent_validation_matrix

| 风险边界 | E 独立结果 | 证据等级与说明 |
|---|---|---|
| Ed25519 approval/record/revocation signature | pass | E3；production-shared `TrustContext` test-only fixture 正负例 |
| 原件字节、认证 record、author/run/commit/release/scope/coverage/verifier/time | pass | E3；篡改、缺失、伪 state/hash 均拒绝 |
| 撤销缺失/不可读/陈旧/未来/过期/错签名/错 authority/source/scope | pass | E3；fail-closed |
| 撤销累积、低序号回滚、同序号 equivocation、重启 | pass | E3；已知 revoked IDs/untrusted signers 不能被高序号擦除 |
| startup nonce/PID、旧/缺失/stale receipt、bootstrap rollback、fork | pass（机制） | E3；但公共可操作性另判 unsupported |
| 测试信任进入 production | rejected | E3；public CLI synthetic/production 组合与内存升级均拒绝 |
| 锁前/锁后重新准入 | pass | E3；两边界撤销均不能进入 writer |
| 阶段内租约、rc=75 retry、下一阶段拒绝 | pass | E3；native transaction 测试；签发者仍须接受 phase 可能很长 |
| cold finalize 与 owner/quiescence | pass（合成） | E3；合法 candidate recovery、撤销 cold finalize 拒绝、owner/certificate/namespace/false quiescence 篡改拒绝 |
| false/missing gate 阻止 completion | pass | E3；无 completion ledger/child dispatch |
| 公共入口正向 startup handoff | fail | E3 source+probe；无受支持路径，形成 `unsupported` |
| authority 独立 checkpoint custody | contract 可表达；真实责任未验证 | 测试 supervisor 通过独立 process pipe 保留 checkpoint；签名不证明来源真实/全球最新/无 live descendants |
| D full pytest 1520 | pass（仅审计） | E2；JUnit 完整性复算，未在 E 重跑全仓 |
| D 八阶段合成链/25 child | pass（仅审计） | E2；未重跑 2719 秒完整合成 |
| 原 run 150 checkpoints、44 active、6 generated | pass（仅审计） | E2；E 仅重算保护字节与账本状态，未重跑语义 loader |
| 科学性能、TMC/paper-ready | 未评估 | N/S |

## 本轮测试结果

- `tests/test_continuation_production_trust.py` 首轮受限环境：89 collected，73 passed，16 failed；16 项均为 native subprocess 内 `ps` 被沙箱拒绝而导致的环境性失败，原始失败 JUnit 保留。
- 仅对 native transaction parameter group 在可读取本机进程身份的隔离环境重跑：21 passed、68 deselected；覆盖首轮 16 项环境失败及该组其余用例。
- `tests/test_continuation_executor_boundaries.py`：71 passed。
- `review_checks.py`：identity/evidence、107,320 项保护、startup probe 均 pass。

不能把 73+21+71 简单相加为独立用例总数，因为 21 项是首轮 89 项中的参数组重跑。按用例集合计，本轮覆盖 production-trust 89 项和 adjacent executor-boundary 71 项，共 160 个不同测试 case；每个 case 最终均在满足其原生进程前置条件的环境通过。

## 未覆盖风险

没有 production installation、真实 signer/receipt/revocation source/trusted-clock service、真实 owner/quiescence 或 production public-entry 正向演练；没有证明签名内容的现实真实性、撤销快照全局最新性或无 live descendants；没有重跑完整合成、全仓测试、150-checkpoint 语义加载或科学 rollout。所有这些均不得从密码学 pass 推导。
