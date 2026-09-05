# Formal Cell Artifact Publication Contract

状态：`1.0.0`，由 Formal Protocol `2.6.0` 激活。Protocol 2.5 及更早版本只供 historical/audit。

## 目的与边界

本合同关闭 formal support producer 与 cell consumer 的路径分叉，并统一正式入口与 non-formal rehearsal 的
事务实现。它只规定执行完整性、恢复和完成判定，不改变科学配置，不提供 formal performance、算法优势、
holdout 或 paper-ready 证据。

## Cell 身份与目录

logical cell identity 绑定 protocol/run/phase/cell/setting/command。每次 attempt 使用独立 staging root；child
必须通过结构化 descriptor 返回唯一 validated artifact root。descriptor 与输出必须位于该 attempt 内，并精确
匹配预期 run、cell、phase、setting。禁止 mtime“最新目录”、模糊 glob、任意 rglob、symlink、路径逃逸、
cross-cell/run 内容以及多份冲突输出。

## 验证、发布与消费

executor 在 child 成功后先验证 required payload、provenance、hash/size 与完整 inventory，再将 validated root
原子发布到唯一 committed destination。committed marker 记录 identity、inventory 与 publication order，随后
append cell committed terminal。只有可由 cell ledger、marker 和 inventory 交叉验证的 committed destination
可被 selection、statistics、integrity 或 gate 消费；staging 和 failed attempt 永不具备消费资格。文本/JSON
中的内部 staging 路径在发布时重绑定为 committed path，保证发布后仍可读取。

## 中断与恢复

- child 输出后、publish 前中断：旧 attempt 保持 incomplete，新 attempt 可重跑命令，不产生 committed 记录。
- atomic publish 后、terminal append 前中断：以 committed marker + inventory 恢复 terminal，不重复执行 child。
- 已 committed cell：重验 identity/marker/inventory 后 skip，不重复提交。
- freeze 已 committed、generated registry 缺失：只从同 run 的 immutable freeze terminal create-only 重建。
- registry 已存在：仅 exact identity/content match 返回幂等成功；同名异 hash、跨 run 或缺 freeze 证据拒绝。

## Gate 与完成

完整性 gate 从 cell ledger 与 committed marker/inventory 计算 exact counts；禁止文件数量 fallback。missing output、
count mismatch、registry/provenance/inventory 失败都会令 `passed=false` 且命令非零退出。performance 高低不参与
该 gate，合法 nullable metric 可保持 unavailable。`complete_without_holdout` 必须验证同 run 的合法 passed gate。

## Readiness 边界

G14R15 clean detached non-formal rehearsal 使用同一 executor 完成 13/13 phases 和 52 unique committed cells。
Readiness v18=`READY_FOR_G14C_V14_CLEAN_TRAIN_AND_FORMAL` 仅授权未来另立任务；G14R15 的 formal
training/checkpoint/performance 均为 0，holdout sealed/unopened/unconsumed，未创建或消费 G14C v14。
