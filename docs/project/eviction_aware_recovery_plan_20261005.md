# 驱逐代价感知恢复规则：冻结计划

## 基线、范围与停止规则

- 解析基线：`codex/shared-cache-coupling` 最终提交
  `1a999287c172b8f34d64680aaaa3d8eb6be23508`。
- 本轮只研究同一连续两节点 AI workflow 中，当前状态恢复/目标模型准备如何改变下一已声明节点的缓存和加载成本。
- 不训练或运行 RL，不下载模型/数据，不开启旧 holdout，不新增动作，不声称跨 workflow 共享、共享资源队列、真实无线或 full MARL。
- 12 个设计点、参数、方法与目标在任何新输出产生前冻结；负结果、持平和选错均保留，不因结果改参数。
- 12 点是未参与结果筛选的 synthetic validation instances。对象大小和合法状态转移来自现有 native typed-cache；
  prefix/input/state 取既有技术工作流测量；70/100/200 Mbps、14 s restore、9.5 s recompute 和误差倍率均显式标为假设。

## 方法公式

设当前目标 RSU resident set 为 (C)，当前节点依赖为 (D_0)，原生 dependency-safe victim plan 为 (V)，
恢复后的 resident set 为 (C'=(C\setminus V)\cup D_0)。已声明近期节点的依赖并集为
(U=\bigcup_{v\in H}D_v)，按 object ID 去重。(T(B)) 表示集合 (B) 中唯一对象的估计传输时间，正字节传输
包含一次冻结 fixed latency。

完整两路增量成本为：

\[
J_{rerun}=\hat C_{recompute}+T(input)+T(U\setminus C),
\]

\[
J_{recover}=T(D_0\setminus C)+T(state)+\hat C_{restore}+T(U\setminus C').
\]

显式驱逐外部代价是

\[
\Delta_{future}=T(U\setminus C')-T(U\setminus C).
\]

其中新增 reload 对象为 ((U\setminus C')\setminus(U\setminus C))，必须能由合法 (V) 解释。共享 base 因集合语义只计一次。
两路相同的未来 missing objects 在两个完整总成本中均保留并在比较时抵消；相同完成量下的成功服务时间也相同，故不进入
二选一差值。选择 action 4 当且仅当 opt-in 已打开、action 4 合法、所有必要输入有效、native preview 可行且
(J_{recover}<J_{rerun})。任何必要输入缺失、非法或出现无法由 victim plan 解释的新增 missing object 时保守选择 action 0。

## 输入字段与信息边界

- 当前可观察：action mask、target residents、capacity、typed catalog 的 dependency/size、native LRU policy state。
- 已声明任务图：当前 adapter 和下一节点 adapter；本轮 `near_term_horizon_nodes=1`。
- 事前成本：link rate、fixed latency、input/state bytes、restore/recompute estimate，以及冻结的误差倍率。
- native read-only preview：当前 placement 的 missing objects、required free capacity、dependency-safe ordered victims。
- 禁止输入：未来实际链路、未来 cache hit、未来服务结果、事后 completion time。离线参考单独明确使用这些未来结果。

## 复杂度与伪代码

规则本身对当前依赖、victims 和去重后的近期依赖各扫描一次，时间
(O(|D_0|+|V|+|U|))，额外空间 (O(|U|+|C|))。现有
`sequential_dependency_recompute_lru_v1` 每轮重算合法候选，最坏 (O(|C|^2))；这不是新平台或新增搜索空间。
两步前瞻显式算 rerun/recover 两条路径，在本轮固定两动作、两步下为常数分支，但仍执行两组集合/成本计算。

```text
if opt_in is false:
    return original_simple_threshold(inputs)
if action4 illegal or native_preview infeasible:
    return rerun
V = native_dependency_safe_victim_plan(C, D0, capacity)
C_after = (C - V) union D0
U = unique_objects(dependencies(declared_near_term_nodes))
before = U - C
after = U - C_after
if required cost missing or (after - before) is not explained by V:
    return rerun
J_rerun = recompute_est + transfer(input) + transfer(before)
J_recover = transfer(D0 - C) + transfer(state) + restore_est + transfer(after)
return recover if J_recover < J_rerun else rerun
```

## 冻结比较与设计点

四方法共享 action 0/4 可行集。`original_simple_threshold` 保持 workload-v0.1 经纠正后的语义：model preparation
被视为两路共同项，未计 victim externality。`eviction_aware_recovery` 使用上式。`two_step_lookahead` 与新规则使用完全相同
的信息和估计，显式评分 `rerun→next` 与 `recover→next`。`offline_reference` 在两条完整 native branch 实现后选择，具有明确
未来信息优势，不是在线 baseline。

12 点覆盖：4 个无驱逐外部代价、2 个完整 bundle reload、1 个共享 base 下 adapter-only reload、2 个恢复本身不划算、
3 个成本估计误差。完整参数见 `configs/experiment/eviction_aware_recovery_v1.json`。原 A/B/C 只作为对象、状态转移和测量来源，
不计为新方法独立验证。

## 输出与执行

冻结实现通过局部测试后先提交。随后从 clean freeze commit 只执行一次：

```bash
python scripts/run_eviction_aware_recovery_validation.py \
  --config configs/experiment/eviction_aware_recovery_v1.json \
  --output-root artifacts/analysis/eviction_aware_recovery_validation_20261005_v1 \
  --expected-git-commit ${FREEZE_COMMIT}
```

输出必须包含 `frozen_protocol.json`、`all_method_results.json/csv`、`aggregate_summary.json`、
`event_explanations.json`、`completion_receipt.json`、`integrity_manifest.json` 和 create-only 状态包。结果出来后只做复算、
论文与长期文档绑定，不修改冻结方法或设计。
