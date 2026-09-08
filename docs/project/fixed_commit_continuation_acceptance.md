# G14R20-A 验收记录（2026-09-08）

固定提交 continuation 合同与只读预检已实现；执行器未实现，独立批准未签发，v16-B 未获执行许可。
真实资格为 `unavailable`，不等于 run 失效，也不要求重训。

- 新增 continuation 回归：56 passed；含既有 checkpoint/bundle 回归的定向集：125 passed。
- 全仓：1362 passed、2 skipped；两项要求 finalized Git-clean active bundle，当前七个受保护用户修改不动。
- smoke、无字节码 compile/import、git diff --check、严格 JSON、schema 语义检查和产物 hash 检查通过。
- 原工作树、旧 run、原 G14R18 包和所列 Protocol 2.9 配置：107,320 个保护条目、38,300,782,534 bytes 核验一致。
- 174 committed cells、105,588 条 cell inventory、185 条 phase outputs 与 marker/hash 一致。
- 150/150 冻结 checkpoint 的原 benchmark gate、44 active resources、6 generated resources 通过；nullable/top-level/nested/envelope/Git/window/runtime/capacity/agent/seed 检查保留。
- 原 public gate 当前 `HEAD != origin/main` 失败，单独报告；固定提交候选检查不产生恢复授权。
- 原始启动批准/发布证明在所搜索 artifacts 中未得到独立核验；executor 和 continuation 批准 pending。
- train/dev/formal/holdout dispatch 全部 0，未分析 dev 分数决定批准。未覆盖 8-phase 实际执行/发布、单写者、故障恢复或正式评估。

验收目录：`artifacts/analysis/g14r20_a_continuation_20260908/`。
完整命令与退出状态：`command_log.json`；真实结果：`real_authorization_check.json`；保护：`protected_before.json.gz`、`user_file_protection.json`；
原声明勘误：`correction_record.json`；产物：`artifact_integrity.json`。
合同与 B 接口见 `fixed_commit_continuation_contract.md`。
