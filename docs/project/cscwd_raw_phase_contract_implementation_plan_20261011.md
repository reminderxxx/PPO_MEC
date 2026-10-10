# CSCWD raw 跨接触逐相位合同实施方案（待决策，2026-10-11）

## 状态与证据入口

`status=PROPOSAL_NOT_AUTHORIZED_FOR_IMPLEMENTATION`。本方案只根据冻结两窗 40 episode/320 step、68 次接触拒绝及 12 次 action 4 阶段诊断拟定下一实现门，不改现有 raw 环境或已消费实验。证据和 SHA 见 [阶段报告](cscwd_new_development_contact_phase_diagnosis_20261011.md)。当前公开规则 consumer 的 `raw_full_step_contact_fit` 漏用由 B 独立最小修复；它不能解决 raw 环境内 0 成功迁移。

## 必须先冻结的物理选择

当前 `RawNGSIMEventTimeEnv.step` 先在 clone 中完成 native 整步，再比较整步秒数与**当前 RSU**接触剩余；失败时全部状态/字节回滚并收取 2 秒失败等待。native action 4 却先按接触预算检查目标模型与状态准备、再执行当前节点，最后记录状态传输及目标 restore。固定诊断中有 8/12 次准备可在接触内完成、4/12 次 native 预览会迁移成功，但完整当前节点计算跨越接触。若仅删除整步 gate，便会在无服务归属说明下允许旧 RSU 在车辆离开后继续执行。

实施前在独立协议中明确且版本化以下二选一的服务语义，不由本次结果倒推选择：

| 方案 | 当前节点跨接触时 | 对 action 4 的含义 | 需要的新增证据 |
|---|---|---|---|
| A：当前接触内原子服务 | 计算和必要输入必须在当前圆内结束，否则失败；已完成模型准备是否保留须明确 | 本两窗的多数 action 4 仍不可达；需事前重新定义来源资格或工作负载，但不可按结果补选 | RSU 覆盖/计算预算与事前选窗规则 |
| B：远端连续服务与有序状态交接（推荐作为下一合同候选） | 输入在当前接触内上传后，旧 RSU 可继续计算；结果/状态经声明的 RSU 间链路到目标，车辆在目标恢复服务 | 把模型准备、当前节点计算、状态生成与传输、目标 restore 拆为带时间戳的事件；只有目标状态真正可用才算 migration success | RSU 间链路吞吐/延迟、队列/保留成本、车辆目标接触或交付路径、同一节点输出正确性 |

推荐 B 仅是**待核验建模方向**，不是已获授权的真值或实验收益。若缺目标 RSU 交付链路与状态正确性依据，继续采用 A 的保守失败，不以成功率代替物理证据。

## 候选 v2 的最小实现顺序

1. 冻结 `raw_ngsim_event_time_v2` 的 topology、链路、coverage、时间与阶段定义，连同 producer/consumer schema；v1 源码和旧 run hash 只读保留。B 的公开 estimator 与所有 learned 方法必须消费同一公共前缀字段，真实未来位置/接触只在执行器和诊断里可见。
2. 为每个 action 显式生成有序事件：当前输入上传、目标模型准备、当前节点计算、节点完成后的状态生成/传输、目标 restore。每相位记录 `start/end`、执行地点、接触或 backhaul 链路、bytes、deadline 与 failure reason。`state_bytes` 不得在节点完成前记为已传输；成本与模拟 clock 一次入账。
3. 定义相位 commit：模型若已完整到达目标可单独保留并收费；未完成的传输按事前指定的原子/可续传规则处理；状态与 `migration_success` 仅在节点结果和目标 restore 均完成后提交。回滚只能撤销未完成相位，不得把已耗时间/网络字节抹去。若当前节点服务失败，明确模型缓存与 pending state 生命周期。
4. 定义事件时间里的 RSU 归属：原始轨迹插值给出当前/目标覆盖进入和离开时刻；跨接触继续计算时须记旧 RSU 占用与结果交付路径，不得将目标 RSU 可用性从 hidden trace 放进 actor/规则。相同公开前缀、不同未来后缀须给出相同选择输入，执行结果可以不同。
5. 用事前四类合成反例验收：准备本身超接触；准备成功但计算跨接触；计算完成但目标 restore 未完成；相同公开前缀而隐藏后缀不同。每类对五动作验证 cache/state/clock/bytes 守恒、mask、终止/截断、失败回滚与 public leakage。v1 replay 作为不变对照，不能以 v2 自动覆写 v1 结果。
6. 仅在合同测试和 B 公开规则 consumer 修复均独立验收后，另立**结果盲**来源与匹配训练方案，冻结预算/seed/checkpoint/正式 split 消费边界。优先检查至少一个可重复的真实 migration/state-ready 机制事件，再讨论强基线学习；0 migration 时继续停机，不用 fallback 完成率晋级。

## 决策门与禁行

- 尚缺真实或公开可校准的 RSU 间链路、队列、target delivery、跨界服务归属和状态正确性证据；上述方案的任何成功数均为 `UNVERIFIED`。
- 不允许在现有两窗结果上调半径、改变 deadline、挑 workload、放宽 gate 后直接宣称新算法优势，也不重开 formal/hidden/sealed。
- 本文是一份实施规范草案；没有修改算法/环境/训练器，没有新性能结果。下一任务首先应由研究负责人冻结 A 或 B 的物理假设与校准来源，然后再写 v2 和对称基线验收。
