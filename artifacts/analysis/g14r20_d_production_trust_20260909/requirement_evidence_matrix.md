# G14R20-D 需求与证据矩阵

实际通过与计数以 `acceptance_summary.json`、`full_pytest.xml`、`targeted_cases_from_full_pytest.json` 和完整链报告为准；本表是验收索引，不是生产批准。

| 要求 | 实现 / 合同 | 验收证据 |
|---|---|---|
| PT1 有限信任安装 | production_trust.TrustContext、PRODUCTION_INSTALLATION=None；完整固定 installation schema | installation_boundary、public_cli_refuses_test_trust、scope_cannot_become_real_capability；unsigned_handoff.json |
| PT2 原件认证闭环 | 每项认证记录签名覆盖记录 ID、核验者、basis、原件 bytes/author/time/source、scope、roles/coverage | certification_negative、valid_shared_original_certificates；serialized_test_only 认证原件与公钥 |
| PT3 认证撤销及失联拒绝 | 固定 source_id/path/authority/key；时间、序号、scope、累计撤销集；coordination continuity | signed_approval_requires_revokeable_identifier、revocation_fail_closed、revoked_state_cannot_roll_back_to_allowed、same_sequence_equivocation、higher_sequence_cannot_erase_known_revocation |
| 首读/重启/可信时间 | 新 nonce/PID/scope 的 authority-signed startup receipt；独立保管当前 checkpoint；不从本地自举 | startup_challenge_blocks_replayed_checkpoint、fresh_custodian_receipt_detects_local_bootstrap_rollback、missing_startup_receipt_never_bootstraps、restart_requires_external_checkpoint、continuity_and_clock_fail_closed |
| PT4 阶段准入语义 | authorize 在 execute_phase 与 SingleWriter 锁前/锁后；阶段内保留原事务/一次 rc=75 retry/finalization | signed_revocation_at_lock_boundaries；原生 revoke_during_phase/revoke_retry/expire_during_phase/candidate_crash/candidate_crash_revoked |
| PT5 新精确身份 | 完整 22-file identity、commit/tree、新 contract 2.0.0、原冻结 command plan | executor_identity.json、executor_verification.json、frozen_command_plans.json、unsigned_execution_contract.json、各命令前后 clean identity |
| PT6 owner/quiescence | 认证原件内绑定精确 owner、host、namespace、观察时间和运维依据；仍检 kernel lock/旧进程消失 | quiescence_original_certification；原生 publication_crash/candidate_crash；B 原并发/恢复边界回归 |
| 八阶段合成/原 consumer | synthetic_chain 使用共享 v2 Ed25519 core；原科学加载/事务/statistics/gate 保留 | synthetic_acceptance_report.json 的 phase_results、consumer_cases、command_mapping、origins、monitor |
| 缺失/false gate | 原 validate_complete_without_holdout_gate | 原生 gate_missing/gate_false 及完整合成 gate 结果 |
| 全仓验证 | 精确 clean executor 上完整 pytest、smoke、全仓 compile/核心 import | full_pytest.xml、full_pytest_result.json、smoke_result.json、compile_import_report.json |
| 严格格式与完整性 | JSON duplicate/nonfinite 拒绝；JSONL 逐行、XML parse；所有交付文件 SHA-256 | format_validation.json、artifact_integrity.json、verify_delivery.py |
| 原科学/账本/七文件保护 | 只读核对原 A 保护 inventory、原科学 commit/tree、目录新增、proposal、七文件及账本摘要 | protected_before.json、protected_after.json、fixed_input_provenance.json、readonly_stdout.log |
| 真实资格保留 | launch 已核验只至 checkpoint_freeze；release unavailable；生产 trust 未安装；continuation 未签发 | unsigned_handoff.json；real_execution_authorized=false；未产生生产 key、批准或真实执行命令 |

签名验收通过不担保外部事实真实或权威状态全球最新；相关职责和限制见正式合同及交接方案。本轮没有安装真实 signer、追认历史 release 或恢复真实 v16。
