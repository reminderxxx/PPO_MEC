# Typed Model-Cache Formal Transaction and Resume Contract

- `formal_phase_ledger_version`: `3.0.0`
- `formal_cell_ledger_version`: `1.0.0`
- `formal_phase_transaction_version`: `1.0.0`
- Protocol：`1.4.0`

## Phase timing and completion

Phase duration 只使用 monotonic clock；UTC 仅作 audit timestamp。Ledger 分别记录 phase elapsed、child
duration、finalization/I/O duration 与 `wall_clock_adjustment_seconds`。forward/backward clock jump、时区和
舍入差异会被标记，不因 UTC 与 monotonic 不等而误杀长 phase；跨进程不比较 monotonic 绝对 origin。

Phase commit 顺序为 `running` → `completion_candidate` → output/integrity validation → immutable terminal
commit。若所有 commands 已结束但 terminal append 失败，同一新协议 run 可用 `--finalize-phase-only` 重验
candidate 与全部 output hashes 后追加 terminal record，不重跑 commands。无 candidate、output drift、旧协议
或重复异 hash finalize 均失败；相同 duplicate finalize 幂等返回。

## Cell transaction and resume

Train、dev candidate、formal cache/controller、ablation/support/scalability 使用 stable cell/episode ID、唯一
attempt staging、append-only hash chain 和 committed marker。Summary/checkpoint/log/metadata 或
summary/row/event/audit 必须作为一个 inventory 验证；只有 committed cell 可进入 selection、aggregate 和
statistics，staging/partial attempt 永不进入消费者。

Resume 必须匹配 run root、execution binding、Protocol/resource/environment/split/window/catalog/runtime 与
command matrix hash。Committed cell 重验后 skip；running 未提交先标 incomplete，再从 cell 起点启动新
attempt；retryable 同命令最多一次；terminal failure、跨 run marker、旧 v4 checkpoint/ledger、同名异 hash
均拒绝。当前不实现单 cell 内部 checkpoint 续训。

no-.venv rehearsal 在 8/16 后中断，同 run resume 精确 skip 8 个 committed cells 并完成剩余 8 个；
75/150 模拟精确 skip 75 个并最终 commit 150 个，无 duplicate cell/episode。另模拟 150/150 后 train terminal
append 失败，finalize-only 零重跑成功。真实 tiny chain 完成 dev selection、checkpoint freeze、formal-like、
support、statistics、integrity 与 `complete_without_holdout`；全部为非正式证据。
