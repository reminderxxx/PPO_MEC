# 最小机制证据闭环结果

## 审查元数据

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-09-30`（本轮不评价 novelty；ALPR 适配性复核截止 `2026-10-04`）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `mechanism_evidence_closure_20261005_v1` + `two_node_workflow_reexecution_comparison_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scientific_git_commit`: `f323fa2c4a098a6e73b054ed7fc6b4946bec2147`
- `evidence_level`: `E2_BOUNDED_NATIVE_AND_TECHNICAL_WORKFLOW_ARTIFACT_AUDITED`
- `verdict`: `PASS_MECHANISM_DIAGNOSTIC_ONLY / NOT_PAPER_READY`

本轮没有训练、formal/holdout、权重或数据下载。A/C 用时 34.193 秒、真实模型调用 0；B 用时 59.168 秒、
`generate=6/6`，低于硬上限 8，无自动重试。两个 artifact 的文件集合、字节数和 SHA-256 已独立复算通过。

## 已完成 / 未完成

| 实验 | 状态 | 证据等级 | 未完成边界 |
|---|---|---|---|
| A 共享缓存四配置 | 完成 | 既有原生仿真账本 + 单列分析成本模型 | 无真实网络、RSU 加载或推理时钟；request replay 无 workflow completion 字段 |
| B 连续 / 重跑 / 后缀恢复 | 完成 | 同机真实 base+adapter、四独立进程、固定输入与 greedy 配置 | 无交通任务标签、真实无线链路、重复统计；production action 4 未调用该状态接口 |
| C 当前规则 / 两步 / 完整枚举 | 完成 | 12 个事前冻结 synthetic design points，全部结果保留 | 不是独立现实样本；状态成本两档均非 production 完整状态测量 |
| ALPR 适配性复核 | 增量复用完成 | 12 个旧公开样本的 redacted artifact audit | 不是新独立测试；训练重叠、图像权利和视觉重标注未验证 |

## 实验 A：相同完成量下的共享缓存收益与代价

下表只列 candidate `sequential_dependency_recompute_lru_v1`，四臂均完成 72/72 请求、拒绝 0。`load`、`repeat`
和 `evict` 是原生 synthetic 账本；等待与总完成时间是 `100 Mbps + 每成功请求 30 ms` 的串行分析模型，不是实测。

| 顺序 / sharing | base load 次数/MiB | adapter load 次数/MiB | base/adapter 重复次数（MiB） | base/adapter 驱逐数（MiB） | 总传输 MiB | 模拟加载等待 s | 模拟总完成 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocked / on | 24 / 2,688 | 72 / 576 | 22/66（2,464/528） | 23/71（2,560/568） | 3,264 | 273.804 | 275.964 |
| interleaved / on | 72 / 8,064 | 72 / 576 | 70/66（7,840/528） | 71/71（7,936/568） | 8,640 | 724.776 | 726.936 |
| blocked / off | 72 / 8,064 | 72 / 576 | 66/66（7,392/528） | 71/71（7,936/568） | 8,640 | 724.776 | 726.936 |
| interleaved / off | 72 / 8,064 | 72 / 576 | 66/66（7,392/528） | 71/71（7,936/568） | 8,640 | 724.776 | 726.936 |

在相同 72 完成量下，`blocked + sharing on` 相比 `interleaved + sharing on` 少传 5,376 MiB base，分析模型少
450.972 秒。收益来自同 family 连续段带来的 48 次 request-level base reuse；interleaved/on 与 sharing-off 两臂
都没有 base reuse。10/100/1000 Mbps 敏感性均保留在原始 JSON，符号不变。

旧语义不能放进相同完成量排名：sharing-on 只完成 36、拒绝 36，传 96/24 MiB base/adapter；sharing-off 只完成
12、拒绝 60，传 96/8 MiB。其低传输来自少服务，不是节省。原四配置没有 handoff/state request，状态成本是
`not_applicable` 而不是 0。

## 实验 B：恢复保真与真实同机成本

四个进程 PID 不同且严格串行。continuous、source、restart、recovery 使用相同 base/adapter/processor、输入、prompt、
CPU/float32 和 greedy 参数。continuous/source/restart 的 `n0` 文本与 token 完全相等；continuous/restart/recovery
的 `n1` prompt、rendered prompt、input IDs、input hash 和输出 token 完全相等。三条比较路径均完成 2/2 节点；
recovery target 的调用数为 `n0=0,n1=1`，图片访问为 0。

| 路径 | 总 generate（含共享中断前缀） | 中断后执行 | 进程内路径时间 s | supervisor child wall s | 远端目标所需动态字节（公式输入） |
|---|---:|---|---:|---:|---:|
| 连续 | 2 | `n0,n1` | 18.763 | 19.465 | 0 |
| 中断后从头重跑 | 3 | `n0,n1` | 30.951 | 32.504 | 原图 192,757 B |
| 中断后状态恢复 | 2 | `n1` | 19.831 | 21.370 | 状态包 2,040 B |

单次同机见证中，恢复路径比从头重跑少 11.120 秒进程时间、11.134 秒 child wall；该差异主要来自不再执行 target
端 `n0`（重跑为 11.035 秒）。它不是跨 RSU 网络时延结论，也没有统计区间。

成本分解：source 序列化/持久保存为 0.000060/0.000676 秒；recovery target 静态资源 hash 校验 0.408783 秒，
状态读取与校验 0.002763 秒，输入重建 0.002652 秒，processor/base/adapter 加载合计 1.173114 秒，`n1` 推理
3.862783 秒。restart target 加载合计 1.199118 秒，`n0/n1` 推理 11.035304/3.827448 秒。

字节严格分层：动态 payload/state file/manifest/package 为 1,090/1,703/337/2,040 B；`n0` 中间结果记录 169 B；
原图 192,757 B；base/adapter 目录为 1,019,895,869/154,428,630 B。目标预存模型是实测路径，网络传输为 0；
目标缺模型时，两条中断路径都需额外 1,174,324,499 B，只做
`bytes*8/link_rate + fixed_latency` 的 10/100/1000 Mbps 敏感性，不声称实测网络改善。

技术工作流的恢复保真和成本通过，但任务正确性仍为 `unavailable`。production action 4 只记录 prepare history，
未实际调用本轮 export/import；因此不能写成 action 4 已实现真实状态迁移。

## 实验 C：全部场景与盈亏边界

`C/T/F` 分别是当前请求规则、既有两步规则、完整四步后缀枚举。每场景完整枚举 625 条候选，其中短尾 247 条、
长尾 209 条动态合法。所有方法使用 action 0--4 的同一合法 mask；下表的时间仍是 A 的分析模型。

| 场景 | sharing / 容量 / 尾部 / 状态成本 | C actions / 完成 / 失败 / MiB / s | T actions / 完成 / 失败 / MiB / s | F actions / 完成 / 失败 / MiB / s |
|---|---|---|---|---|
| s01 | on / 136 / short / 2,040 B | 0-0-0 / 2/2 / 1 / 240 / 20.192659 | 同 C | 0-4 / 2/2 / 0 / 240.001945 / 20.192822 |
| s02 | on / 136 / long / 2,040 B | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 344.001945 / 28.946975 |
| s03 | on / 320 / short / 2,040 B | 0-0-0 / 2/2 / 1 / 240 / 20.192659 | 同 C | 0-4 / 2/2 / 0 / 240.001945 / 20.192822 |
| s04 | on / 320 / long / 2,040 B | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 344.001945 / 28.946975 |
| s05 | off / 136 / short / 2,040 B | 0-0-0 / 2/2 / 1 / 240 / 20.192659 | 同 C | 0-4 / 2/2 / 0 / 240.001945 / 20.192822 |
| s06 | off / 136 / long / 2,040 B | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 344.001945 / 28.946975 |
| s07 | off / 320 / short / 2,040 B | 0-0-0 / 2/2 / 1 / 240 / 20.192659 | 同 C | 0-4 / 2/2 / 0 / 240.001945 / 20.192822 |
| s08 | off / 320 / long / 2,040 B | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 344.001945 / 28.946975 |
| s09 | on / 136 / long / 160 MiB | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 504 / 42.368584 |
| s10 | on / 320 / long / 160 MiB | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 504 / 42.368584 |
| s11 | off / 136 / long / 160 MiB | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 504 / 42.368584 |
| s12 | off / 320 / long / 160 MiB | 0-0-0-0 / 3/3 / 1 / 344 / 28.946812 | 同 C | 0-4-0 / 3/3 / 0 / 504 / 42.368584 |

所有 36 行 completion/deadline 结果相同：短尾 2/2、长尾 3/3、deadline violation=0。完整枚举以 action 4 换掉
一次服务失败；低状态成本多 2,040 B，高状态成本多 160 MiB。因此它是明确 Pareto 交换，不是统一算法优势：若事前约束
要求零服务失败，F 是可行解；若允许一次重试并优先字节/分析时间，C/T 更低成本。sharing 与容量在这 12 格中均没有
改变结果，属于机制未激活区域，不能外推为两者普遍无用。

信息价值要单列。完整枚举与 current-only exhaustive projection 在 12 格的首动作全部相同（均为 0）；差异出现在第二步
是否用 action 4。既有两步规则只在“下一节点复用同一 adapter”时选 action 1，因此没有覆盖“交接前准备当前节点”这一
合法局部条件。结合旧 4 个配对实例中“两步与完整剩余序列首动作全部相同”，当前没有证据需要更长未来信息或 RL；
新差距更符合 action-rule coverage 缺口，而不是复杂模型能力缺口。

## ALPR 负结果与适配性

最新 redacted 复核保持不变：development 上 adapter/base exact 为 0/4 与 3/4，micro CER 为 0.1786 与 0.0357；
locked-check 上二者都是 4/8、micro CER 都为 0.2453。12 图是同一旧观察集合，base 对照是 exploratory paired，
不是新独立测试。结论仍是 `DO_NOT_PROMOTE_THIS_ADAPTER_FOR_CACHE_BENEFIT_STUDY`；技术加载成功不能替代任务准确性。

## 论文候选主张—证据—限制

| 候选主张 | 原始证据 | 安全边界 |
|---|---|---|
| 同完成量下，共享底座只有与可复用顺序配合才降低传输 | A JSON 的 288 candidate requests；blocked/on 少 5,376 MiB base | synthetic 96/128/8 MiB；时间为公式；不是部署吞吐 |
| 节点边界显式状态可在独立进程恢复并保持后缀输入/输出 | B 四 receipt、state package、terminal checks | 固定两节点技术任务、单次 CPU/float32；不证明交通语义 |
| 本地预存模型时，恢复比从头重跑少一次前缀推理 | B target/restart receipt，单次 wall 差 11.134 s | 单次同机观察；无 CI、无无线链路 |
| action 4 可在有界设计点减少一次服务失败 | C 全部 12 格，F failures 0 vs C/T 1 | 代价是 2,040 B 或 160 MiB 设计点；production state 未接线 |
| 当前无需启动 RL | 旧配对两步=完整首动作；新矩阵差距由确定性 action 4 局部条件解释 | 只支持“本轮不晋级”；不是一般性否定 RL |
| production action 4 已完成状态迁移 | 无 | `UNVERIFIED / PROHIBITED`：当前只 prepare，无 export/import 调用 |
| ALPR adapter 优于 base | ALPR 对照为负/持平 | `PROHIBITED`：不得重命名为独立测试或任务优势 |

## 唯一下一步

不启动 RL。唯一最值得做的是：先将已验证的 state package 以显式 opt-in 方式接入 action 4 的 producer/consumer，随后在
新的、事前冻结且与本轮 12 格不重用的两节点技术实例上，只比较“当前规则”与一个最小局部规则：当交接目标缺少当前
节点模型且经验证状态成本低于一次失败/重跑边界时选择 action 4。必须使用真实状态字节与真实传输路径，保持任务保真、
完成量和失败约束，设计集与验证集分开。若该独立实验没有稳定跨越盈亏边界，则停止算法扩展。

## 证据路径与保护

- `artifacts/analysis/mechanism_evidence_closure_20261005_v1/`
- `artifacts/analysis/two_node_workflow_reexecution_comparison_20261005_v1/`
- `/Users/howen/Projects/PPO_MEC/artifacts/analysis/alpr_public_pair_review_20261004_v1/`
- `configs/experiment/mechanism_evidence_closure_v1.json`
- `configs/acceptance/two_node_workflow_reexecution_comparison_v1.json`

主工作区七个用户修改文件的逐文件 SHA-256 与既有保护快照完全一致；combined binary diff SHA-256 仍为
`1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2`。没有 stash、reset、覆盖或纳入本分支提交。
