# 真实两节点 AI 工作流前缀保存与独立进程后缀恢复验收

## 审查元数据

- `reviewed_at`: `2026-10-03`
- `literature_cutoff`: `2026-09-30`（本轮不做 novelty 评价，未刷新外部文献）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`（仅使用 claim-boundary policy）
- `artifact_run_id`: `two_node_workflow_suffix_recovery_20261003_v3`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `01c9d972243cff0c58d1d597f8d029fc4cc76707`
- `plan_sha256`: `74b428386f785e4942cde2ebcaf1642077ea43f4f6f4743a907b0095120c208c`
- `evidence_level`: `E2_ARTIFACT_AUDITED`，仅限本 technical workflow calibration
- `verdict`: `PASS_WITH_STRICT_SCOPE`

结论：本轮已证明 target 独立进程实际读取 source 保存的 `n0` 描述并据此构造 `n1` 输入，只执行后缀节点；它没有
重新读取图片或重跑视觉前缀。continuous 与恢复路径的 `n1` 实际输入及 token IDs 严格一致。该结论只覆盖本地
CPU/float32、单 adapter、固定两节点技术链；任务正确性仍为 `unavailable`。

## 1. 基线、冻结与旧证据保护

本轮从 adapter/state 分支提交 `4689403…` 新建隔离分支，不带入 main 的七个用户修改。第一阶段版本化执行器提交
`1f809cf…`；直接脚本启动 bootstrap 修复提交 `01c9d97…` 是最终科学执行 commit，两者均已 push 到
`origin/codex/workflow-suffix-recovery`。冻结方案见
`two_node_workflow_suffix_recovery_plan_20261003.md` 和机器配置；科学执行没有修改 DAG、prompt、输入、生成参数、
判据或预算。

主工作区起止均为 `73051ab…` 和同一 7 个 modified 文件，逐文件 SHA-256 完全相等。旧 10/3 artifact 的四个关键
文件 hash 也保持不变，原 `terminal_receipt.json.status=FAIL` 未改写。旧证据仍只证明：双 adapter 加载与
A→B→A 技术兼容性通过、独立进程模型重载和相同输入复现通过；该旧 run 自身未证明工作流中间状态恢复，任务正确性
仍 unavailable。本轮是新的独立证据，不回填旧回执。

模型执行前有两次被保留的启动失败：v1 在 Python import 前因 `src` 路径缺失退出；v2 因传入错误完整 commit 被
preflight 拒绝。两者均未创建 run root、未加载模型、generate=0；随后使用新 committed executor 和 Git 实际完整
commit 只启动一次科学 run。记录见 companion review artifact 的 `launch_attempts.json`。

## 2. 三个进程与实际数据流

| 路径 | PID | 节点顺序 | 调用数 n0/n1 | generate | exit |
|---|---:|---|---:|---:|---:|
| continuous | 95812 | `n0,n1` | 1/1 | 2 | 0 |
| source | 95837 | `n0` | 1/0 | 1 | 0 |
| target | 95847 | `n1` | 0/1 | 1 | 0 |

source 子进程结束时间 `1791039450.262079`，target 启动时间 `1791039450.2628422`；三个 PID 不同。target 读取的
状态 payload SHA-256 为 `6abd23d4…d22f`，其中保存的 `n0` 原始输出是
` The image is a line graph depicting the`，token IDs 为
`[378,2443,314,253,1761,3670,17691,260]`。

target 由该字符串实际构造：

```text
Upstream visual description:  The image is a line graph depicting the
Summarize only that description in one short phrase.
```

continuous/target 的 prompt、rendered prompt、34 个 input IDs 与 canonical input SHA-256
`9a036ce8…7d14` 全部相等；`n1` token IDs 均为 `[216,34,32,33,32,731,216,34]`。输出文本为
` 2010 - 2`，语义质量不佳且没有标签，因此只用于恢复保真性，不能作为任务成功。

target 的 `n0=0,n1=1`；application-level 记录的 source-image access=0。目标读取白名单仅含冻结方案/执行器、
`state.json`、`manifest.json`、预存 base/processor 和单 adapter；状态包显式声明不含模型权重或最终 `n1` 答案。

## 3. 状态字段与负例

状态声明 schema/workflow、DAG nodes/edges、completed/remaining、next node、`n0` 文本/token IDs、后缀消费文本、
两节点 generation config、base/adapter/processor identity、payload hash，以及 RNG/tensor 不需要的原因。greedy
节点之间传递显式文本，不是 token continuation，因此没有保存 RNG continuation、KV cache 或 tensor。

模型加载前 6/6 检查通过、generate=0：缺少中间结果拒绝；内容被篡改而 hash 未更新拒绝；next node、completed
nodes、base identity 冲突分别拒绝；两份合法的 `a line graph` / `a road intersection` 状态产生不同 `n1` prompt
与 hash。旧 PEFT 回执的汇总回归读取 `available_adapters`，字段缺失及与 legacy `loaded_adapters` 冲突均拒绝。

## 4. 单次见证成本分层

| 层 | 内容 | 单次观察 |
|---|---|---:|
| target 预存静态资源 | base 目录 / base weight | 1,019,895,869 / 1,015,025,832 B |
| target 预存静态资源 | adapter 目录 / adapter weight | 154,428,630 / 154,423,432 B |
| target 缺失才需传输 | base + adapter 目录 | 仅字节盘点；本路径未传输，时延 `null` |
| workflow 动态状态 | canonical payload | 1,090 B |
| workflow 动态状态 | `state.json` / manifest / 整包 | 1,703 / 337 / 2,040 B |

| 计时项 | 秒 |
|---|---:|
| serialize / durable save | 0.000248 / 0.000826 |
| target 静态 hash 校验 / 状态读取与恢复校验 | 0.394852 / 0.000202 |
| target 后缀输入重建 | 0.002563 |
| target processor / base / adapter load | 0.072237 / 0.261007 / 0.813699 |
| target 后缀推理 | 3.805719 |
| target Python/导入等启动开销 / process 内部 / supervisor child wall | 0.612508 / 6.252111 / 6.864620 |
| 三科学进程 supervisor 总墙钟 | 41.606533 |

这些是同机单次见证，不是均值、性能 benchmark、RSU 实测或网络时延。没有将 2,040 B 与旧 synthetic 104 MiB
相减，也没有净收益结论。

## 5. 完整性、已证明与未证明

科学 artifact 的 manifest 覆盖 11 个非 manifest 文件、33,290 bytes；独立重算文件集合、size 与 SHA-256 全部通过。
计划内 generate 4/4，硬上限 6，每进程小于 300 秒，累计小于 900 秒，无科学自动重试。三进程命令、stdout/stderr、
退出码、启动/结束时间、模块来源与全部输入输出见 artifact 原始 JSON。

已证明：在本冻结技术链上，动态中间状态可被独立 target 进程验证并实际消费，target 只执行 `n1`，连续/恢复后缀输入
及输出 token 完全相等；保存的不同中间描述会改变后缀输入。

未证明：ALPR/Helmet 或其他真实车联网任务正确性；真实 RSU/无线传输、排队、故障恢复；KV/tensor 恢复；多 adapter
workflow；production action 4 原生状态迁移；训练、算法优势、formal/holdout、论文净收益或可泛化性能。下一阶段若要
推进，具体缺口是带许可与标签的真实车联网输入、production state producer/consumer 接线、独立网络测量及多次独立
统计；本轮不自动进入这些工作。

## 6. 证据路径

- `artifacts/analysis/two_node_workflow_suffix_recovery_20261003_v3/`
- `artifacts/analysis/two_node_workflow_suffix_recovery_20261003_review_v1/`
- `configs/acceptance/two_node_workflow_suffix_recovery_v1.json`
- `scripts/run_two_node_workflow_suffix_recovery.py`
- `src/runtime/workflow_suffix_recovery.py`
- `tests/test_workflow_suffix_recovery.py`
