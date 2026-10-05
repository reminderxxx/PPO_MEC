# Measurement-calibrated semi-synthetic VEC workload v0.1

## 审查元数据与边界

- `frozen_at`: `2026-10-05T00:00:00+08:00`
- `target_venue`: `IEEE TMC`（只使用 claim-boundary policy）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `literature_cutoff`: `2026-09-30`（本轮不评价 novelty，不新增文献）
- `status`: `FROZEN_BEFORE_COMPARISON`
- `evidence_level`: `UNVERIFIED_BEFORE_EXECUTION`

该工作负载只用于第一轮机制诊断。它不是新的真实数据集，不使用旧 holdout，不训练 RL，不下载数据或权重。
公开价值、代表性、许可和独立复现均未完成。三类 typed-model ID 是抽象对象，不能写成三个真实任务模型；两节点 DAG
是 technical workflow，不能写成真实交通业务流程。

## 一页研究问题—机制—假设—指标映射

| RQ | 机制 | 事前可检验假设 | 主要指标 | 反证/无收益保留 |
|---|---|---|---|---|
| RQ1 | immutable base 在兼容 adapter 间共享；LRU 容量竞争 | 同完成量下，`A→B→A` 顺序只有在 base identity 兼容且容量未迫使 base 重载时减少 base 传输；adapter 仍单独计费 | 完成 workflow、base/adapter 分解传输、prepare 次数、完成时间 | sharing-off、tight 容量下的 base 重载必须保留 |
| RQ2 | 节点边界导出状态；目标模型就绪后校验并只执行后缀 | measured 低恢复成本应优于重算；synthetic 高恢复压力点应使重算更优；目标模型缺失时两者都先承担模型准备，状态包不能替代权重 | 输入/状态/模型分解传输，serialize/save、restore/validate、input rebuild、重复计算、端到端时间 | high-restore 和 missing-model 负例必须保留 |
| RQ3 | 当前可观察量上的局部增量成本比较 | 若决策相互独立，局部规则应与同可行集的离线枚举相同；若出现差距，只能据可解释条件再提一次最小改进 | 完成量、deadline violation、端到端时间、总传输、重复计算、决策开销、local-to-exact gap | 不要求局部规则、当前规则或 exact 必须领先 |

共同目标按完成 workflow 最大化、deadline violation 最小化、总完成时间、总传输、重复计算的固定字典序执行。若这些指标
互相冲突，报告 Pareto 交换，不再调权重合成单分数。`offline_exact_enumeration` 可读完整冻结实例，只是离线参考，不能当
信息公平在线 baseline。

## 数据卡与字段来源表

配置：`configs/experiment/measurement_calibrated_vec_workload_v0_1.json`。生成器：
`src/data/workflow/measurement_calibrated_vec_workload.py`。执行入口：
`scripts/run_measurement_calibrated_vec_workload.py`。

| 字段组 | 数值/范围 | 来源类型 | 来源与限制 |
|---|---:|---|---|
| base file / network directory bytes | 1,015,025,832 / 1,019,895,869 B | measured | 既有固定 SmolVLM 本地资源；文件大小与网络目录字节分开 |
| adapter file / network directory bytes | 154,423,432 / 154,428,630 B | measured | 既有固定 adapter；不代表生成的抽象 adapter 是多个真实模型 |
| single-process peak runtime memory | 2,379,005,952 B | measured | 旧本地进程观测；不能与文件或网络字节混用 |
| input / dynamic package / intermediate record | 192,757 / 2,040 / 169 B | measured | 既有两节点 technical workflow；2,040 B 不含 KV/tensor production state |
| prefix / suffix compute | 11.035304 / 3.827448 s | measured | 单次 CPU/float32 目标进程；无 CI |
| target model load | 1.173114 s | measured | 本地预存资源加载；不是网络传输 |
| serialize+save / restore validate / input rebuild | 0.000736 / 0.002763 / 0.002652 s | measured | 既有 technical package；只校准 low 恢复点 |
| high restore | 14.0 s，冻结范围 11.5--16.0 s | synthetic | 明确保留恢复不划算对照，高于 measured prefix 重算；不是测量 |
| arrival jitter | 0--2 s；seed 17/29/43 | synthetic | 唯一随机过程；三方法使用共同随机数 |
| request order | `A0→B0→A1` | synthetic | 激活复用、驱逐和重访；不是生产请求 trace |
| target cache capacity | 1.2 / 2.4 GB | synthetic | 分别近似只能保留一组与可保留两组 base+adapter 的覆盖点 |
| access handoff / compute migration | source→target / prefix 后迁移 | synthetic | 两者显式分列；不是 trace-derived 真实移动 |
| link | 100 Mbps + 20 ms | synthetic | 只代入字节公式；不得称真实无线迁移 |
| deadline | arrival 后 60 s | synthetic | 只用于机制压力诊断；无业务 SLA 主张 |
| wireless、排队、丢包、任务质量、KV/tensor 状态 | `unavailable` | unavailable | 不以 0 代替 |

本轮没有 trace-derived 字段。旧 observation/holdout 内容均不参与生成、筛选或 seed 选择。

## 冻结覆盖设计、方法与预算

设计为平衡 `2^(4-1)` 半分数 8 点，覆盖 sharing on/off、tight/ample、初始 target model ready/missing、low/high
restore；每点 3 个事前 seed，共 24 workload instances。每实例 3 个两节点 workflow，三方法面对相同 arrival、模型、
移动和链路事件。全部 72 方法行、失败行和分解字段必须输出。

- A `current_restart_rule`：中断后重建输入并重算 prefix。
- B `mechanism_aware_local_incremental_cost`：只用当前 model/cache ready、状态字节和冻结成本，比较完整 recovery 与 restart
  增量成本。
- C `offline_exact_enumeration`：对每实例 `2^3=8` 个 restart/recover 组合穷举；完整未来只用于离线下界。

半合成矩阵计划真实模型调用 `0`；production action 4 的 technical workflow 正/负例计划 `6` 次 generate，硬上限
`12`，无自动重试。先运行 action 4 最小贯通，再一次性运行冻结矩阵。结果产生前不得修改点位、seed、目标顺序或截止期。

## Action 4 接线合同

默认 `workflow_state_migration.enabled=false`，旧五动作行为不变。显式启用后，action 4 仅在 source 当前节点真实完成后
写 create-only package；package 记录 workflow/nodes/edges/order、completed/remaining/next-node、节点输出、输入和
模型 identity、source/target RSU 以及执行权状态。target 必须先证明 base+adapter 就绪，再校验 package；只有全部一致
才提交 `execution_right_transferred=true` 并执行当前合法后缀。缺模型、hash/identity/next-node/prefix 冲突均不提交迁移，
且不推进 DAG。模型准备、状态传输、恢复校验、输入重建、后缀计算和等待分别记录。

真实 technical runner 与正常 `reset/action/step` 共用 `src/runtime/production_action4_state.py`。本地文件读写和链路公式
不得称真实跨 RSU 无线迁移。

## 结果占位（执行后更新）

本节在固定 implementation commit 上完成实际执行后更新；未执行前不得写入数值结论或 paper-ready 判断。
