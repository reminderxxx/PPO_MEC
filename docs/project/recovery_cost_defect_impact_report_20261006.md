# Restart / recovery 成本不对称缺陷影响报告（只读审查）

## 审查身份

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-05`（本轮未做新增文献检索或 novelty 复评）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `eviction_aware_recovery_validation_20261005_v1`、`real_cache_victim_reload_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `b3a00b6141039c7ab223eaef8d03cf6c3efdc221`
- `evidence_level`: `E2_ARTIFACT_AUDITED`（真实 lifecycle 原件 22/22 byte-exact；旧 synthetic CSV 在 Git checkout 中发生已解释的 CRLF→LF 规范化，JSON 原件及其余 29/30 文件可核验）
- `verdict`: `CONFIRMED_PATH_ASYMMETRY / HISTORICAL_COMPARATIVE_CLAIMS_WITHDRAWN_PENDING_CORRECTED_RERUN`

本报告是修复前的独立只读审查。它不修改算法、配置、旧 artifact 或旧论文结论，也不把后续修复预先判为通过。

## 1. 首因与三条链路

### 1.1 决策时成本估计

旧冻结公式从同一个初始目标缓存 `C` 出发，只在 recovery 一侧显式加入当前依赖准备：

```text
J_rerun   = recompute_est + transfer(input) + transfer(U - C)
J_recover = transfer(D0 - C) + transfer(state) + restore_est + transfer(U - C')
C'        = (C - recovery_victims) union D0
```

`src/runtime/eviction_aware_recovery.py::_full_incremental_costs()` 与上述文字一致。它没有构造 restart 自己的当前模型准备、victim plan、准备后 resident set 和后续 reload；`two_step_lookahead_decision()` 复用了同一个 `_full_incremental_costs()`，因此不是独立正确基线。两者 12/12 相同只能证明共享实现相同，不能证明公平。

`original_simple_threshold` 同样不显式计模型准备，但它把模型准备声明为共同项。这个简化只有在两路实际执行相同模型生命周期并经事件核验后才可抵消；旧 runner 没有提供该核验。

### 1.2 实际合法动作、缓存状态和执行成本

旧 runner 为 action 0 和 action 4 各建独立环境，但两条分支执行的并不是对称生命周期：

- action 4 在目标 RSU 为当前 `D0` 执行 native typed-cache transaction，可能驱逐对象，再执行下一节点；
- action 0 没有在目标 RSU 执行当前 `D0` 准备。它保留 `C` 进入后续节点，再用一次 service failure 表示重算，并在之后准备下一节点；
- 因而 action 0 与 action 4 的 victim、resident 轨迹、model bytes 和后续 reload 从定义上不同。旧离线参考虽分别运行了两条 native branch，却比较了两个生命周期不匹配的分支。

这不是单纯的公式或文案问题。例如 d04 的 action 4 合法执行 `b1 victim → b0 prepare → b0 victim → b1 reload`，而 action 0 从未执行 `b0 prepare`，其目标 cache 一直保留 b1。于是旧评分把 240 MiB 的两次 model preparation 全部放在 recovery 路径，把 restart 记为 0 model bytes。d06 为反向 family 的同一问题；d05 只向 recovery 记 8 MiB 当前 adapter 和 8 MiB 后续 adapter reload。

不能据此断言 restart 在任何真实实现中不需要当前模型。真实三节点 runner 已证明相反：restart 和 recovery 都准备 Helmet，并都发生 ALPR victim、Helmet load、Helmet victim、ALPR reload。也不能从该单一真实实例进一步推断所有历史 synthetic 点的两路终态必然相同；纠正版必须逐分支执行并记录状态转移。

### 1.3 结果统计与汇总

`_resummarize_branch()` 按每条旧 branch 的实际 cache events 汇总 model bytes，再加 state/input、重算与服务时间。因此算术没有重复求和，但输入的两条 branch 生命周期已经不公平。`offline_reference` 事后选择这两条不对称 branch，故也受影响；它不是可用于修正在线方法的独立真值。

汇总中的 `10/12 vs 7/12`、`-17.993274 s`、`-972,125,684 B`、`+5 reruns` 和 `+53.641216 s recomputation` 都依赖该不公平比较，必须撤回为历史结果，等待对称复算。旧 artifact 保留，不覆盖。

## 2. 逐点结构影响

下表的“遗漏当前准备”是旧 restart 决策/执行未建立的子项，按旧事前 link 参数计算。它不是总成本预测误差，也不能直接加到旧总分后冒充纠正结果，因为加入当前准备还会改变 victim、resident 和后续 reload。

| 点 | restart 遗漏当前对象字节 | 对应事前传输子项 s | 旧 action：简单 / 代价感知 / 两步 / 离线 | 影响判断 |
|---|---:|---:|---|---|
| d01 | 8,388,608 | 0.691089 | 4 / 4 / 4 / 4 | 时间、bytes、restart resident 未公平；动作结论待复算 |
| d02 | 109,051,904 | 8.744152 | 4 / 4 / 4 / 4 | 两路当前准备与后续 b1 准备需逐事件去重 |
| d03 | 109,051,904 | 8.744152 | 4 / 4 / 4 / 4 | exact-fit 终态不能由旧 action 0 代表 |
| d04 | 109,051,904 | 8.744152 | 4 / 0 / 0 / 0 | 原“victim 使恢复翻转”直接受缺陷影响，撤回 |
| d05 | 8,388,608 | 0.691089 | 4 / 4 / 4 / 4 | shared-base 合法 victim 机制仍有效；比较收益待复算 |
| d06 | 142,606,336 | 11.428507 | 4 / 0 / 0 / 0 | 原反向 family 翻转直接受缺陷影响，撤回 |
| d07 | 109,051,904 | 12.483075 | 4 / 0 / 0 / 0 | 原“当前准备使 recovery 不划算”是单边收费结果，撤回 |
| d08 | 8,388,608 | 0.691089 | 0 / 0 / 0 / 0 | high-restore 负例方向可能保留，但完整成本需复算 |
| d09 | 0 | 0 | 4 / 4 / 4 / 4 | 当前与下一模型均 resident；是唯一没有该当前准备遗漏的点 |
| d10 | 109,051,904 | 8.744152 | 4 / 0 / 0 / 4 | 原“recompute 低估导致选错”与单边模型费耦合，作为历史负结果保留但不再作纠正后证据 |
| d11 | 109,051,904 | 8.744152 | 4 / 4 / 4 / 0 | 原“recompute 高估导致选错”及参考动作均受不对称评分影响 |
| d12 | 142,606,336 | 5.724253 | 4 / 0 / 0 / 0 | 原 near-boundary 动作受单边当前准备影响，待复算 |

旧公式改变了代价感知规则和伪两步基线在 d04、d06、d07、d10、d12 的选择；旧 offline reference 的分支评分除 d09 外都缺少 restart 当前准备事件。即使某点旧动作最终不变，也不能据此保留其相对成本或 gap 数值。

## 3. 影响分类

| 层级 | 是否受影响 | 结论 |
|---|---|---|
| 文字公式 | 是 | `J_rerun` 不是一般全路径公式；v1.2 已承认限制，但旧计划/结果仍把它用于比较 |
| 策略选择 | 是 | 代价感知规则与“two-step”共享同一不对称 scorer；5 个点相对简单阈值发生动作变化 |
| 执行成本记账 | 是 | 旧 action 0 没有目标端当前模型 transaction；model bytes、fixed latency 和 completion 均少记 |
| 后续缓存与重载 | 是 | action 0 未经当前准备，victim、resident 及下一节点 reload 轨迹不是对应 restart 生命周期 |
| 汇总表与论文主张 | 是 | 方法匹配率、相对参考 gap、总时间/bytes 优势和 d10/d11 归因均不可继续引用为纠正后结果 |

## 4. 总成本误差与子项误差必须分开

旧 synthetic runner 的 branch score 和公式使用同一组模拟参数，不能把二者的一致说成经验性“预测准确”。其中 `T(D0-C)` 的遗漏是结构子项错误；总成本还取决于该准备造成的 victim、fixed-latency 事件数、后续 reload、重算和服务执行，必须由纠正版分支重新执行后才能报告。

真实 `real_cache_victim_reload_20261005_v1` 提供了独立的两类误差，二者不能混写：

- 总成本预测误差（预测减 scored 实测）：victim restart/recovery 为 `+8.258994/+4.353426 s`，control 为 `+8.091637/+4.467135 s`；
- lifecycle 子项误差：冻结 victim lifecycle 增量估计 `0.868896 s`，restart/recovery 实测 `1.022245/1.026997 s`，即分别低估 `0.153349/0.158101 s`。

真实 scored 时间是 local action wall 加冻结的模拟动态网络项；100 Mbps、0.02 s 不是实测网络时间。local adapter load bytes 也不是无线传输 bytes。

## 5. 仍有效的证据

以下证据不依赖旧 restart/recovery 相对评分，继续有效：

1. production action 4 的状态包校验、后缀恢复和输入/token fidelity；
2. recovery 分支中的 native dependency-safe victim transaction 真实存在：d04/d06 的完整 bundle victim、d05 的 shared-base adapter-only victim；
3. 非法 victim、依赖破坏、容量不足时的原子拒绝和无 orphan 合同；
4. `real_cache_victim_reload_20261005_v1` 的真实 PEFT ALPR `780→0→780`、Helmet `520→0`、文件 hash 不变及 control 零 action load/unload；
5. 同一真实实例中 restart/recovery 的约 1.02 s lifecycle 是共同成本，未形成 action reversal；
6. 网络、queue、任务正确性、跨 workflow persistent cache、统计泛化和优于正确两步前瞻仍未验证。

## 6. Artifact 完整性与停止边界

- `real_cache_victim_reload_20261005_v1`: manifest 22/22 文件 size/SHA-256 byte-exact。
- `eviction_aware_recovery_validation_20261005_v1`: 当前 Git checkout 中 29/30 byte-exact；唯一差异是 `all_method_results.csv` 被 `.gitattributes` 规范为 LF。工作树文件为 8,529 B / `fc3efee6…`；恢复 CRLF 后为 manifest 记录的 8,578 B / `0f1a1624…`。JSON 原件、逐事件记录和数值内容可核验，但历史“30/30 当前 checkout byte-exact”表述不成立。
- 本报告不判定后续修复通过。纠正版必须另用 create-only artifact root，分离估计决策与实际评分，保留全部 12 点和 d10/d11 历史对照，并逐分支执行合法 cache state transition。

当前安全结论是：真实状态恢复与 adapter lifecycle 接线成立；旧 synthetic 方法优势及其离线参考公平性不成立。最强贡献是否还能包含成本决策收益，必须由后续对称复算决定。
