# G14R20-F requirement/evidence matrix

| 要求 | 本轮证据 | 结果 |
|---|---|---|
| 公共 parser/main 同进程 challenge→verify→qualification→execute | `public_startup_acceptance.json` positive events；独立 host/custodian PID | PASS |
| qualification/execute 同一 context | `tests/test_continuation_startup_handoff.py` callback identity；公共 terminal request identity | PASS |
| 两条独立命令不伪装交接 | 新进程 nonce/PID negative；post-release 新挑战 | PASS |
| missing/legacy receipt | `missing_receipt_events` / `old_receipt_events` 均 rc=124 | PASS |
| 错 schema/signature/authority/checkpoint/nonce/PID/expired | `invalid_receipt_events` 7 类真实公共宿主 | PASS |
| timeout/SIGINT/SIGTERM/exit/crash | 公共 timeout、SIGTERM、SIGKILL；单元测试覆盖 SIGINT/退出映射 | PASS |
| fork 与旧 PID/nonce | targeted JUnit 的 startup/fork cases | PASS |
| 两宿主并发与释放后新宿主 | `concurrent_*_events`、`post_release_events` | PASS |
| startup lock 独立、预置 fixed inode、无删除重建 | startup 单元测试与 public concurrent events | PASS |
| receipt 发布不等待 startup lock | 第一宿主持锁时独立 custodian 成功发布 | PASS |
| checkpoint 不能由 host 自报升级 | 3 个 `custody_checkpoints` 均标明 independent recomputation | PASS |
| 下一宿主/cold finalize 旧 receipt 拒绝、新 receipt 通过 | checkpoint/nonce/PID negative + post-release host；targeted finalization | PASS |
| 撤销、防回滚、锁后复核、阶段租约、rc=75/finalization | targeted 174 tests | PASS |
| false/missing gate 阻止 completion | executor-boundary targeted cases | PASS |
| 生产入口拒绝测试 trust | production-trust targeted cases；`PRODUCTION_INSTALLATION is None` | PASS |
| 准入前保护计数 | 公共结果 scientific import/writer-lock/original-write 全为 0 | PASS |
| 原 run/科学身份/七文件保护 | `protection_recomputed.json`、`scope_and_protection.json` | PASS |
| 新完整 executor identity | 25-file `executor_identity.json` + `executor_verification.json` | PASS |
| 全仓、smoke、compile/import、diff check | `full_pytest.xml`、`command_log.json` | PASS |
| 不执行真实科学或 holdout | synthetic=1；science/real dispatch/write/holdout=0 | PASS |
