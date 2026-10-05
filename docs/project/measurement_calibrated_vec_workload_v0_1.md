# Measurement-calibrated semi-synthetic VEC workload v0.1

## 审查元数据与边界

- `frozen_at`: `2026-10-05T00:00:00+08:00`
- `target_venue`: `IEEE TMC`（只使用 claim-boundary policy）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `literature_cutoff`: `2026-09-30`（本轮不评价 novelty，不新增文献）
- `reviewed_at`: `2026-10-05`
- `status`: `COMPLETE_BOUNDED_MECHANISM_DIAGNOSTIC`
- `artifact_run_id`: `production_action4_workflow_20261005_v2` + `measurement_calibrated_vec_workload_20261005_v3`
- `git_commit`: `94600ded7752a548de74384f17ab4e0ebabf916c`
- `evidence_level`: `E2_BOUNDED_TECHNICAL_AND_SEMI_SYNTHETIC_ARTIFACT_AUDITED`
- `verdict`: `LOCAL_RULE_SUFFICIENT_IN_FROZEN_MATRIX / NOT_PAPER_READY`

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

## Production action 4 真实 technical workflow 结果

最终 run `production_action4_workflow_20261005_v2` 在 clean commit `94600de...` 完成 6/6 generate、四个独立进程，
wall 58.825 s，无自动重试。continuous/source/restart/target 的 model、adapter、processor、输入和 greedy generation
identity 相同；连续、重跑和恢复的 `n1` prompt/rendered/input IDs/hash/token IDs 全部相同。source 只执行 `n0`；target
先完成 model load，再调用 production import，只执行 `n1`，图片访问为 0。执行权只在完整校验后提交。

| 路径 | 中断后节点 | target/model load s | 动态字节 | restore validate s | input rebuild s | suffix compute s | child wall s | queue wait |
|---|---|---:|---:|---:|---:|---:|---:|---|
| restart | `n0,n1` | 1.148209 | input 192,757 B | n/a | n/a | 3.924471 | 32.858409 | unavailable |
| action-4 recovery | `n1` | 1.186145 | state 2,185 B | 0.002865 | 0.002528 | 3.888702 | 21.651422 | unavailable |

恢复路径比重跑少 11.206987 s child wall，主要来自不重算 target `n0`（11.066554 s）；单次同机结果无 CI。v2
production package 比矩阵事前采用的旧 measured 2,040 B 大 145 B（7.11%）。这是 schema 元数据增量，不改写冻结矩阵；
低/高恢复决策边界不因 145 B 改变。网络时延仍只可按字节公式计算，不能称真实无线迁移。

必要负例在 target model 未就绪时返回 `BLOCKED_MISSING_TARGET_MODEL`：`migration_success=false`、
`execution_right_transferred=false`、restore validation 0、generate 0。状态包未被读取为可执行后缀，也未推进 DAG。
正常 `GymVecEnv.reset/action/step` 的正例完成 `[n0,n1]`，负例保持 completed prefix `[n0]`；同一能力对所有输出 action 4
的方法开放，默认未启用 profile 的行为不变。

真实单进程 runner 不含 RSU queue model，因此 waiting 明确为 `unavailable`；半合成矩阵另行记录 synthetic serial queue
wait。模型准备、状态传输、恢复校验、输入重建、后缀计算和 waiting 均未混写成同一“迁移时间”。

## 全部方法、全部设计点结果

最终矩阵 `measurement_calibrated_vec_workload_20261005_v3` 在相同 commit 上生成 24 instances、72 方法行，wall
0.0083 s，真实模型调用 0。下表时间是三个固定 seed 的均值；每个 seed 的完成量、deadline count、动作与字节相同。
`A/B/C` 分别为 current restart、local incremental cost、offline exact。每个单元格式为
`决策×3 / 完成workflow / deadline违约 / makespan秒 / 总传输字节`。

| 点 | sharing / capacity / initial model / restore | A | B | C |
|---|---|---|---|---|
| d01 | on / tight / ready / low | restart / 3 / 2 / 105.634342 / 309,435,531 | recover / 3 / 1 / 72.501111 / 308,863,380 | recover / 3 / 1 / 72.501111 / 308,863,380 |
| d02 | on / tight / missing / high | restart / 3 / 3 / 200.773416 / 1,483,760,030 | restart / 3 / 3 / 200.773416 / 1,483,760,030 | restart / 3 / 3 / 200.773416 / 1,483,760,030 |
| d03 | on / ample / ready / high | restart / 3 / 2 / 105.634342 / 309,435,531 | restart / 3 / 2 / 105.634342 / 309,435,531 | restart / 3 / 2 / 105.634342 / 309,435,531 |
| d04 | on / ample / missing / low | restart / 3 / 3 / 200.773416 / 1,483,760,030 | recover / 3 / 3 / 167.640185 / 1,483,187,879 | recover / 3 / 3 / 167.640185 / 1,483,187,879 |
| d05 | off / tight / ready / high | restart / 3 / 2 / 268.817681 / 2,349,227,269 | restart / 3 / 2 / 268.817681 / 2,349,227,269 | restart / 3 / 2 / 268.817681 / 2,349,227,269 |
| d06 | off / tight / missing / low | restart / 3 / 3 / 363.956755 / 3,523,551,768 | recover / 3 / 3 / 330.823524 / 3,522,979,617 | recover / 3 / 3 / 330.823524 / 3,522,979,617 |
| d07 | off / ample / ready / low | restart / 3 / 2 / 187.226011 / 1,329,331,400 | recover / 3 / 2 / 154.092780 / 1,328,759,249 | recover / 3 / 2 / 154.092780 / 1,328,759,249 |
| d08 | off / ample / missing / high | restart / 3 / 3 / 282.365085 / 2,503,655,899 | restart / 3 / 3 / 282.365085 / 2,503,655,899 | restart / 3 / 3 / 282.365085 / 2,503,655,899 |

全部方法均完成 3/3 workflow、6/6 nodes，service failure=0；因此没有以少服务换节省。low restore 的 B/C 在
d01/d04/d06/d07 均选择三次 recovery，相对 A 每实例少 33.133231 s makespan、66.266462 s total end-to-end、
33.133231 s synthetic queue wait、572,151 B dynamic transfer 和 33.105912 s 重复计算；只有 d01 的 deadline
violation 从 2 降到 1，其余 low 点 deadline 不变。high restore 的 B/C 都选择 restart，与 A 完全一致。

决策开销均值/最大值：A `0.080/0.167 µs`，B `1.273/2.625 µs`，C `74.203/106.209 µs`。C 读取完整冻结实例并
枚举 8 个组合，只是 offline reference；B 不读取未来实现值。

## 收益区、无收益区与代价交换区

- **共享收益区**：sharing-on 时 ready/missing 分别传 0/1,019,895,869 B base；sharing-off 的 tight ready/missing
  分别传 2,039,791,738/3,059,687,607 B，ample 分别传 1,019,895,869/2,039,791,738 B。`A→B→A` 中，
  sharing 避免不兼容 base 切换；sharing-off 时 ample 相对 tight 恰少一次 1,019,895,869 B base 重载。
- **无共享收益区**：adapter 不共享。initial ready 时仍传两个 adapter（308,857,260 B），missing 时传三个
  （463,285,890 B）；sharing 和容量不消除这些字节。
- **恢复收益区**：low measured-calibrated 恢复严格降低时间、动态传输和重复计算，完成量不变；target model missing
  增加共同的 model prepare，未改变 recover/restart 边界，但在冻结 60 s deadline 下常把全部 workflow 推入违约。
- **恢复无收益区**：high synthetic restore 下局部规则和 exact 均选 restart；因此不存在“恢复必优”的结论。
- **代价交换区**：若强制 high recovery，每实例仍可少 572,151 B dynamic bytes，却会多 8.848316 s makespan；固定
  目标优先时间时选择 restart。该 counterfactual 来自冻结成本模型，不是额外筛选点或实测系统结果。

## 能支持与不能支持的表述

安全表述：在该三请求、两节点、100 Mbps 公式和固定测量校准下，base sharing 与容量共同决定重复 base transfer；
validated suffix recovery 只在完整增量成本低于 restart 时有益；目标模型缺失是共同前置成本，状态包不能替代模型；
当前可观察信息的局部规则在全部 24 instances 上与 offline exact 的决策和主指标相同。

禁止表述：RL 必要或优越；SA/任何 learned 方法领先；真实无线迁移收益；任务质量保持；production KV/tensor state 已测；
新真实数据集贡献；跨 trace、跨模型、跨带宽或 paper-ready 泛化；8 点和 3 seed 是独立现实样本或最终统计。

## 执行偏差与预算账本

- real workflow v1 在 commit `6c07005...` 成功 6/6 generate，但 receipt 未显式列 queue wait；保留为非最终 run。
- workload v1 的 `decision_overhead_seconds` 错包 episode replay；v2 修复计时但未显式列 queue wait；两者均保留为
  非最终代码结果，不用于上表。
- v2 首次命令因手工写错 expected full SHA 在 output 创建前 fail-fast，调用/结果为 0；随后读取 Git 实际 SHA 启动。
- final real v2 再执行 6/6 generate；三次 workload run 均为 0-call。真实模型新增调用总计 `12/12`，无自动重试、
  无下载、训练、formal 或 holdout。

两个 final integrity manifest 已独立复算：real v2 为 15/15 files，workload v3 为 5/5 files，集合、size 和 SHA-256
全通过。主工作区仍是指定 main commit 和原七个 modified 文件；本分支未修改、stash、reset 或提交它们。

## 唯一下一步决定

**局部规则已足够：继续系统机制与外部有效性验证。** 本轮不启动 RL，也不提出算法改进。下一轮只应在新的公开许可
工作负载/移动来源和独立网络或 RSU 测量上验证同一冻结局部规则，补 production KV/tensor 状态、排队/丢包和任务质量；
不得重开旧 holdout。若外部验证出现可解释且信息公平的 local-to-exact gap，再另立一次最小方法任务。

最终机器证据：

- `artifacts/analysis/production_action4_workflow_20261005_v2/`
- `artifacts/analysis/measurement_calibrated_vec_workload_20261005_v3/`
- 非最终但保留：`production_action4_workflow_20261005_v1/`、`measurement_calibrated_vec_workload_20261005_v1/`、
  `measurement_calibrated_vec_workload_20261005_v2/`
