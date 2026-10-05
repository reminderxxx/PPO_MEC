# 驱逐代价感知恢复规则：最小验证结果

## 审查身份与结论

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-09-30`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `eviction_aware_recovery_validation_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 基线提交：`1a999287c172b8f34d64680aaaa3d8eb6be23508`
- 冻结执行提交：`f5033c1b10f8c323f2247a09b1f5ab73a8621387`
- `evidence_level`: `E2_ARTIFACT_AUDITED`（仅 bounded synthetic native transition）
- `verdict`: `IMPLEMENTATION_CORRECTION_WITH_BOUNDED_POSITIVE_DIAGNOSTIC / NOT_PAPER_READY`

一次 clean run 完成 12 个事前冻结 synthetic validation instances、24 条 native branch；wall `0.635429 s`，
RL/model call/download/old holdout operation 均为 0。30 个 manifest 文件的 size/SHA-256 独立复算一致。

新规则在 12/12 点与同信息、正确计入 victim cost 的两步前瞻动作和估计成本完全相同；两者均在 10/12 点匹配事后离线参考。
原简单阈值为 7/12。新规则没有优于两步前瞻，且纯函数 median-of-instance-medians 决策开销为 `7.875 µs`，
两步为 `7.896 µs`，差异只有 `0.021 µs`，不构成效率优势。当前可支持价值是：把原阈值遗漏的当前 model preparation 和
victim-induced reload 写成可审计的显式成本，并直接给出触发对象；不是新算法或在线最优性证明。

## 方法身份与成本边界

完整公式、字段、复杂度和伪代码见 `eviction_aware_recovery_plan_20261005.md`。规则只读当前 target residents、typed
dependencies、capacity、原生 dependency-safe LRU victim preview、已声明下一节点和事前成本估计。它不读未来实际链路、
hit、服务结果或完成时间。缺输入、action 4 非法、native preview 不可行，或新增 reload 不能由 victim plan 解释时均返回
action 0；相应负例已由局部测试覆盖。

技术分类如下：

- 已有技术：typed base/adapter dependency、sequential dependency-safe LRU、action 4 状态包、简单增量成本阈值；
- 本文组合/实现修正：把当前 model preparation 和已声明近期节点的 victim-induced reload 纳入两路完整增量成本；
- 待验证贡献：这种修正在真实/独立 workflow、真实链路或跨 workflow persistent cache 中是否带来稳定收益；
- 非贡献：两步前瞻、offline oracle、共享 base、LRU、状态恢复本身。

原 A/B/C 只提供对象大小、状态转移和测量来源，不作为新规则独立验证。12 点不是新增真实观测，严格称为未参与结果筛选的
合成验证实例。

## 全部方法结果

所有方法均完成 24/24 节点且 deadline violation 为 0，因而下面才比较成本。时间是冻结的 event-based model，
不是无线 wall-clock。

| 方法 | 匹配离线参考 | 恢复决策 | 失败/重跑 | 完成时间 s | 传输 B | base B | adapter B | state B | rerun input B | 重算 s | 决策开销中位数 µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原简单阈值 | 7/12 | 11 | 1 | 174.096793 | 2,038,648,604 | 1,879,048,192 | 159,383,552 | 24,103 | 192,757 | 11.035304 | 2.438 |
| 驱逐代价感知 | 10/12 | 6 | 6 | 156.103518 | 1,066,522,920 | 973,078,528 | 92,274,688 | 13,162 | 1,156,542 | 64.676520 | 7.875 |
| 两步前瞻 | 10/12 | 6 | 6 | 156.103518 | 1,066,522,920 | 973,078,528 | 92,274,688 | 13,162 | 1,156,542 | 64.676520 | 7.896 |
| 离线参考 | 12/12 | 6 | 6 | 144.675011 | 923,916,580 | 838,860,800 | 83,886,080 | 13,158 | 1,156,542 | 64.676520 | post-hoc；不在线可比 |

新规则相对原阈值少 `972,125,684 B`、少 `17.993274 s`，但多 5 次失败/重跑和 `53.641216 s` 重算；这只是异质
合成点的等权和，不是均值、置信区间或一般收益。结果来自在时间优先目标下用更多重算换取避免大 model transfer，不能写成
所有指标全面改善。相对离线参考，新规则多 `142,606,340 B`、多 `11.428507 s`；差距全部来自两项冻结估计误差。

## 逐事件解释（全部 12 点）

| 点 | 当前恢复 preview / 近期依赖变化 | 原阈值 / 新规则 / 两步 / 离线 | 解释 |
|---|---|---|---|
| d01 | 无 victim；before=after=空 | 4 / 4 / 4 / 4 | 共享 b0 base 已 resident，当前只传 8 MiB adapter；无外部代价。 |
| d02 | 当前无 victim；b1 在两路下一步都缺失 | 4 / 4 / 4 / 4 | b1 136 MiB 是共同成本；保留在两路总成本并抵消。 |
| d03 | 240 MiB 恰容纳 b0+b1；无 victim | 4 / 4 / 4 / 4 | 与 d02 同决策，但验证无驱逐的 exact-fit 状态。 |
| d04 | victim=`adapter:b1.a0, base:b1`；新增 reload 136 MiB | 4 / 0 / 0 / 0 | 原阈值漏掉 b1 reload，恢复 `20.258985 s`，重跑 `11.130725 s`。 |
| d05 | victim=`adapter:b0.a1`；共享 base 不驱逐，reload 8 MiB | 4 / 4 / 4 / 4 | 对象去重和 dependency protection 生效；恢复仍为 `1.468504 s`。 |
| d06 | victim=`adapter:b0.a0, base:b0`；新增 reload 104 MiB | 4 / 0 / 0 / 0 | 反向 family 顺序同样由对象身份驱动，不是硬编码 b1。 |
| d07 | 无 victim；70 Mbps 下当前 b0 prepare 使恢复本身更贵 | 4 / 0 / 0 / 0 | 原规则把 model prepare 当共同项；新规则把 `28.887342` 降到 `27.455200 s`，但仍有一次重跑。 |
| d08 | 无 victim；14 s high-restore stress | 0 / 0 / 0 / 0 | 三个在线方法都拒绝恢复；验证恢复本身不划算的控制。 |
| d09 | 当前/下一 adapter 与共享 base 全 resident；无 cache mutation | 4 / 4 / 4 / 4 | 只传约 2.1 KiB state，恢复 `0.086326 s`。 |
| d10 | 无 victim；recompute estimate=真实值的 0.5 | 4 / 0 / 0 / 4 | **科学负结果**：新规则因低估重算而错误拒绝恢复，损失 `2.300247 s`，但少传 `108,861,322 B`。 |
| d11 | victim=完整 b1；recompute estimate=真实值的 2.0 | 4 / 4 / 4 / 0 | **科学负结果**：所有在线方法错误恢复；新规则虽计 reload，仍被高重算估计翻转，损失 `9.128260 s`。 |
| d12 | victim=完整 b0；200 Mbps、9.5 s、reload 低估 10% | 4 / 0 / 0 / 0 | 误差没有翻转新规则；保留为“并非任何误差都会失败”的负对照。 |

完整每一步 pre/post residents、victim/admission、service、state bytes 和 completed nodes 位于
`artifacts/analysis/eviction_aware_recovery_validation_20261005_v1/event_explanations.json`。实际 state package 因
instance ID 不同为 `2,175--2,215 B`，在线规则只用冻结事前估计 `2,185 B`，没有读取事后实际包大小。

## 主张—实验—原件—限制

| 主张 | 实验 | 原始原件 | 限制 |
|---|---|---|---|
| 当前恢复可通过合法 victim plan 改变下一节点 reload | d04/d06 的两条 native branch | `all_method_results.json`、`event_explanations.json` | 同一两节点 workflow；synthetic catalog；非跨 workflow。 |
| 共享 base 将完整 bundle reload 降为 adapter-only reload | d05 对照 d04/d06 | 同上，及 `native_transaction_preview` 字段 | 仅一个 96 MiB base/8 MiB adapter family；不能外推真实 LoRA 服务。 |
| 原阈值遗漏外部代价会选错 | d04/d06/d07 | `all_method_results.csv` | 设计点不是独立现实抽样；只证明构造内反例。 |
| 新规则与正确两步前瞻等价 | 全部 12 点 | `aggregate_summary.json` 的 12/12 action/cost agreement | 只覆盖两动作、下一节点 horizon；不证明更长 DAG 等价。 |
| 新规则对事前成本误差敏感 | d10/d11；d12 负对照 | `decision_inputs`、两 branch summary | 误差倍率是假设，无真实校准分布或置信区间。 |
| 新规则优于两步前瞻 | 无 | 两者所有结果相同 | **不能支持**；开销差异 0.021 µs 不具统计/系统意义。 |
| 新规则具备真实无线或跨 workflow 收益 | 无 | 无 shared link/queue/persistent workflow cache artifact | **不能支持**。 |

## 当前最强可支持贡献与不能支持项

最强可支持贡献是一个小而明确的**实现修正/组合**：在现有 native typed-cache 合法 victim preview 上，把当前 model
preparation 与已声明下一节点的对象级 reload 外部代价加入恢复/重跑完整增量成本；集合去重可正确处理共享 base。在这 12 个
冻结合成实例上，该规则把原阈值的离线参考匹配从 7/12 提高到 10/12，并与同信息两步前瞻完全持平。

不能支持：原创通用算法、优于正确两步前瞻、在线最优、真实无线收益、跨 workflow 共享调度、共享带宽/计算队列、真实任务质量、
统计泛化、full MARL 或 paper-ready/TMC-ready。

## 最多三个投稿关键缺口

1. 缺少未参与设计的真实/独立 workload 与成本校准分布；当前 12 点不能提供统计泛化或外部有效性。
2. 缺少跨 workflow persistent cache、真实链路/队列和任务质量执行；当前只能证明单 episode 内的状态转移与公式成本。
3. 缺少比正确两步前瞻更强的增量价值：当前 12/12 持平，计算开销也近似相同；若论文以算法创新为主，这一缺口是 blocker。

## 顶刊政策审查摘要

- Evidence inventory：冻结 config/commit、48 行方法结果、24 条 native branch、逐事件证据、command、completion receipt
  与 30-file integrity manifest 齐全；没有 formal/holdout/support、独立数据或真实网络原件。
- Hard blockers：正好对应上节三个缺口；因此当前不是 `TMC-ready candidate`。
- Major concerns：异质合成点等权汇总、字典序时间优先会用更多重算换 model bytes、decision overhead 无独立重复。
- Minor concerns：实际 state package 因 ID 有 40 B 范围，而规则使用固定事前值；当前未改变符号但应在未来校准中建模。
- Scorecard：`N/S (not scored)`；本轮不是完整投稿包审查，不能用 bounded matrix 给伪精确 100 分。
- Safe claims / prohibited claims：见上节“当前最强可支持贡献与不能支持项”。
- Required actions before re-review：仅为上节三个投稿缺口，不扩展为本轮后续实验。
- TITS/TVT fit note：目前同样不具备投稿就绪证据；降低目标 venue 不消除独立数据、两步等价和系统真实性缺口。

## 验证与失败分类

- 代码/合同验证：`python -m py_compile ...` 通过；局部 pytest `26 passed`。
- 执行验证：冻结 commit clean-run 单次完成；artifact integrity 30/30 通过。
- 工程失败：仅首次在隔离工作区调用裸 `python` 得到 `command not found`，随后使用项目虚拟环境；没有环境 step、输出或科学预算被消耗。
- 科学负结果：d10/d11 估计误差选错；新规则与两步前瞻完全持平；均保留且未调参。
