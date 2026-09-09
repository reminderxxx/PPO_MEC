# G14R20-B 技术验收报告

验收日期：2026-09-09T03:05:18.119957+00:00。实现提交：`e6a73357d08220d34ddc2d6537d140131c60520f`。原科学提交：`a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`。
证据记录使用后续独立 Git 提交，不把该提交的自身 hash 写回文件；完整 hash 由交付记录给出。

独立 executor 的实现与隔离合成技术验收通过。真实 v16 未恢复，独立批准未签发，原始启动/发布证据仍 unavailable；v16-B 尚未获执行许可。

## 三层证据

1. 实际新 CLI：八个 phase 的缺批准拒绝、越权 phase、固定身份漂移、生产入口拒绝 synthetic trust；71 项定向测试均通过。完整用例见 `fault_and_boundary_cases.json` 与 `targeted_pytest.xml`。
2. 原生产/消费入口：原 artifact main 实际调用两次，生成 synthetic selection/freeze/companion；原 benchmark main 的正例及 13 个负例均经过真实 loader/gate，在 rollout 前停止。另有 150 个合成 checkpoint gate 全覆盖。每例实际调用和模块来源见 `synthetic_acceptance_report.json`。
3. 原事务完整链：五类 cell 共 22 个，后接原 statistics、integrity/formal gate 和 completion。实际 25 次 child 调度对应 25 份子进程监测；scientific rollout / real v16 dispatch / real v16 write 都为 0。

| phase | 原冻结命令数 | 合成结果 |
|---|---:|---|
| formal_cache_policy | 3 | completed |
| formal_controller | 3 | completed |
| formal_ablation | 2 | completed |
| formal_support | 11 | completed |
| formal_scalability | 3 | completed |
| formal_statistics | 1 | completed |
| formal_gate | 1 | completed |
| complete_without_holdout | 0 | completed |

`compatibility_details.json` 保存原 15-phase / 186-command 完整展开、八阶段 argv/order/coordinates/output/hash、cwd/env 与原 context 的一致性。
定向测试直接求值原固定源码中的 identity 和 staging 表达式，逐项比较；没有调用原 public main 或以恒成功 validator 替换检查。
合成前五类 cell 使用明确的 test payload adapter；其命令替换与原科学命令分别记录，不把这些替换称为科学 argv 等价。

合成授权在落盘后重新读取并核验 proposal 原始字节 SHA-256、command plan digest 和批准签名；四个序列化正负例通过。

## 恢复与隔离

实际覆盖 exit 75 同一原命令的一次重试、非 75 终止、缺失/损坏 payload、descriptor/provenance 错配、publication 后 append 前进程崩溃、candidate finalize-only、重复 committed ID、并发单写者、锁 owner 崩溃与精确批准恢复、重复启动幂等、阶段中撤销/过期、UTC 回拨、乱序/分叉/截断/跨账本、immutable payload 损坏及 false/missing gate。
实际生产 CLI 的两个 test-trust 拒绝例见 `cli_trust_refusal_report.json`，均在写入/dispatch 前拒绝；独立原生 staging-root 错配记录见 `staging_refusal_report.json`。
原账本前缀保持不变；registry freeze anchor `bd9b84b23dfe050cd5d40e8301162f15c8784a8cea63816f889f2c086842c5bf` 与最终 tip 不同，仍经原 consumer 验证。
来源验收覆盖旧 src/scripts __file__/hash、实际 Python/cwd/sys.path、5 个关键依赖来源、关闭 user-site、无 editable install，以及当前 main、外部 src、错误解释器/依赖污染拒绝。

## 真实只读与保护

真实只读复核：150 checkpoints、44 active resources、6 generated resources、174 committed payloads、167 immutable evidence identities；15 phase / 348 cell records。
原保护清单校验：107,320 条、38,300,782,534 bytes；检查旧 run/worktree 新增文件，七个用户文件完整 SHA-256 起止一致。
原 proposal 与原 inventory 不改写。没有真实锁、running record、staging 或 child；没有 G14D/G15。

## 验证与使用边界

- `python -m pytest tests -q`：1431 passed，0 failed/error/skip。
- `python -m pytest tests/test_continuation_executor_boundaries.py -q`：71 passed，0 failed/error/skip。
- `python scripts/smoke_test.py` 与 compile/import：通过。Smoke 是单独的仓库 toy 检查，不合并进 continuation monitor 的计数。
- 原测试的 V2.4 临时 context builder 不再要求测试分支等于 origin/main；生产发布 gate 保持不变。
- 保留 16 条既有 pytest record_property/xunit2 warning；不把 warning 或未执行项写成通过。
- 命令、stdout/stderr、JUnit、运行前后 clean Git 身份与实际耗时均保留。格式/hash 检查见 `format_validation.json` 与 `artifact_integrity.json`。

当前生产 signer map 为空。独立原始启动/发布凭据、可信批准及其信任根仍需另行核验；仅提供 JSON 或测试密钥不会开放执行。
未来批准必须绑定经过验收的精确 executor 身份、未改写的 A proposal、原科学来源和固定八阶段合同。若安装生产信任导致代码身份变化，必须对新精确身份重新验收。
本报告不作论文效果、统计优势或 paper-ready 判断。完整独立材料清单见 `acceptance_summary.json`。
完整合成夹具（含 test-only checkpoint/输入表）保留在本机验收目录；Git 只保存报告、命令、JUnit、验证脚本和 hash 清单。`versioned_evidence_manifest.json` 定义随提交交付的文件子集，`artifact_integrity.json` 覆盖本机完整验收夹具；不提交真实数据、checkpoint 或旧 run 内容。
