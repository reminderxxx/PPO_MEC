# Public recompute observability 缺陷定位

- evidence root: `artifacts/analysis/cscwd_lankershim_v668_positive_prepare_coverage_probe_20261011_v1/`
- reviewed commit: `ccef09eb992ead8e8debef3f66be63c42d6c26f5`
- mode: read-only saved-state recompute；0 新 env step / optimizer / parameter update

192 个 actual-consumer 状态中有 46 个 `decisive_public_comparison_unavailable`。它们全部满足 action4 legal、
current service=yes、target prepare=yes、future readiness gain=`target_bundle_and_state`，且至少一个普通服务动作可服务；唯一
阻断是 action4 current-service phase/recompute unknown，继而 total/deadline fit unknown。

源码对账确认 executor 仅在 `current_rsu != last_execution_rsu` 时执行 `_recompute_seconds(node)`；但 public semantic state
只给 current RSU 和 prepared-state prefix，没有给已发生的 `last_execution_rsu` 或当前 handoff boolean。
`_known_recompute_seconds` 因而在任何 completed prefix 且 current prepared prefix invalid 时都返回
`handoff_and_missing_predecessor_identity_unknown`，无法区分“根本没有 handoff”与“handoff 后需重算”。46 行中 30 行当前节点
为无前驱 node 3/4，仍被该全局 prefix 条件压成 unknown。

这是 public causal execution-history 接口缺失，不是 future-truth 权限需求。最小纠错是公开 last execution RSU/current handoff，
并用与 executor 相同的已完成 predecessor closure 计算 handoff recompute；无 handoff 时 recompute=0。该字段不进入 actor feature
向量，不改变 network/action mask/reward/执行器/基线权限；只使 public estimator 与 auxiliary label 从不必要 unknown 恢复为
可核验值。修复后必须用全新 create-only probe root 复核 direct label 与真实 SA consumer，一旦仍无非零干预即停止。

