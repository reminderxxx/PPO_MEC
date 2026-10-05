# 对称生命周期成本纠正版：原 12 点复算结果

## 身份、执行和结论

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-05`（未新增检索）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `eviction_aware_recovery_corrected_20261006_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 缺陷审查提交：`4e897bf152422beb072fd021c40560e6229faeae`
- 冻结实现/执行提交：`800f0a12f13e34d7de19aee375124813b8817e3a`
- `evidence_level`: `E2_ARTIFACT_AUDITED`（bounded synthetic development correction；非独立验证）
- `verdict`: `HISTORICAL_METHOD_ADVANTAGE_WITHDRAWN / CORRECT_SIMPLE_BASELINE_TIES_FULL_RULE_AND_TWO_STEP`

唯一一次执行完成原 12 个设计点、每点 4 个相互隔离的 native cache path（两条在线 preview、两条实际评分），共 48 条；
wall `0.502246 s`，0 RL、0 model call、0 download、0 old holdout、无重试。7/7 manifest 文件 size/SHA-256
byte-exact。

修正后四方法动作、完成、成本和相对参考 gap 在 12/12 点全部相同：除 d08 high-restore 选择 restart 外，其余 11 点均
选择 recovery。旧驱逐代价感知规则相对简单阈值的 `10/12 vs 7/12` 优势完全撤销；相对正确两步前瞻仍无新增能力。

## 对称核算合同

每条路径从相同初始 target cache 建立独立环境，按 `current adapter → declared next adapter` 顺序各执行两次 native typed-cache
transaction。restart 与 recovery 都必须准备当前模型；各自的 victim、admission、resident 变化和下一节点 reload 由本路径状态
决定，不能从另一分支复制。每个正字节 model admission、input/state transfer、restore、recompute 和共同服务时间只计一次。

在线三方法只读另两条 detached preview 与事前 estimate；实际评分另建两条环境。微型 offline reference 只在两条合法评分
分支完成后读 modeled outcome。所有点均记录 `separate_environment_count=4`、`online_realized_score_used=false`。

时间是冻结 Mbps/fixed-latency 参数上的 event-based modeled completion，不是实测网络时间。模型字节是 native admission 对象的
transfer size；state 使用原配置的冻结决策/评分参考 `2,185 B`，不冒充本轮真实序列化测量。旧 v1 分支曾因 instance metadata
产生 `2,175--2,215 B` 包，因此旧→新表中部分 recovery 行有 8--26 B 的非机制差异；动作和核心结论不依赖这一区别。

## 方法汇总

| 方法 | 匹配纠正后离线参考 | recover | 完成节点 | service failure | deadline violation | model B | dynamic B | 总 B | 完成时间 s | 重算 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原简单阈值 | 12/12 | 11 | 24 | 0 | 0 | 2,046,820,352 | 216,792 | 2,047,037,144 | 174.787877 | 11.035304 |
| 驱逐代价感知 | 12/12 | 11 | 24 | 0 | 0 | 2,046,820,352 | 216,792 | 2,047,037,144 | 174.787877 | 11.035304 |
| 正确信息权限两步 | 12/12 | 11 | 24 | 0 | 0 | 2,046,820,352 | 216,792 | 2,047,037,144 | 174.787877 | 11.035304 |
| 微型离线参考 | 12/12 | 11 | 24 | 0 | 0 | 2,046,820,352 | 216,792 | 2,047,037,144 | 174.787877 | 11.035304 |

原 12 点不是独立验证；总和也不是统计均值或泛化证据。相同完成量已先核验，才能比较成本。修正后 service failure 为 0：
restart 的 prefix recompute 是主动执行成本，不再用一次请求失败代替。

## 旧→新逐点对照（驱逐代价感知规则）

完整四方法 48 行逐点表位于 `old_to_new_point_comparison.{json,csv}`；它逐行包含 old/new action、victim、terminal residents、
总 bytes、完成时间、完成量、failure、recompute 和相对 reference gap。下表不筛点，展示原主张方法的全部 12 点；`A/B`
表示 current/future 两次 transaction 的 victim。

| 点 | action 旧→新 | victim 旧→新 | terminal residents 旧→新 | 总 B 旧→新 | 完成 s 旧→新 | 完成 | fail 旧→新 | 重算 s 旧→新 | gap s 旧→新 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| d01 | 4→4 | `∅/∅→∅/∅` | `b0+a1+a0→b0+a1+a0` | 8,390,801→8,390,793 | 0.777415→0.777414 | 2→2 | 0→0 | 0→0 | 0→0 |
| d02 | 4→4 | `∅/b0→∅/b0` | `b1+a0→b1+a0` | 251,660,443→251,660,425 | 20.258986→20.258985 | 2→2 | 0→0 | 0→0 | 0→0 |
| d03 | 4→4 | `∅/∅→∅/∅` | `b0+a0+b1+a0→same` | 251,660,451→251,660,425 | 20.258987→20.258985 | 2→2 | 0→0 | 0→0 | 0→0 |
| d04 | 0→4 | `∅/∅→b1 bundle/b0 bundle` | `b1+a0→b1+a0` | 192,757→251,660,425 | 11.130725→20.258985 | 2→2 | 1→0 | 11.035304→0 | 0→0 |
| d05 | 4→4 | `b0.a1/b0.a0→same` | `b0+a1→same` | 16,779,407→16,779,401 | 1.468504→1.468503 | 2→2 | 0→0 | 0→0 | 0→0 |
| d06 | 0→4 | `∅/∅→b0 bundle/b1 bundle` | `b0+a0→same` | 192,757→251,660,425 | 11.130725→20.258985 | 2→2 | 1→0 | 11.035304→0 | 0→0 |
| d07 | 0→4 | `∅/∅→∅/∅` | `b1+a0→b0+a0+b1+a0` | 142,799,093→251,660,425 | 27.455200→28.887342 | 2→2 | 1→0 | 11.035304→0 | 0→0 |
| d08 | 0→0 | `∅/∅→∅/∅` | `b0+a1→b0+a1+a0` | 192,757→8,581,365 | 11.130725→11.821813 | 2→2 | 1→0 | 11.035304→11.035304 | 0→0 |
| d09 | 4→4 | `∅/∅→∅/∅` | `b0+a0+a1→same` | 2,185→2,185 | 0.086326→0.086326 | 2→2 | 0→0 | 0→0 | 0→0 |
| d10 | 0→4 | `∅/∅→∅/∅` | `b1+a0→b0+a0+b1+a0` | 142,799,093→251,660,425 | 22.559231→20.258985 | 2→2 | 1→0 | 11.035304→0 | 2.300247→0 |
| d11 | 4→4 | `b1 bundle/b0 bundle→same` | `b1+a0→same` | 251,660,419→251,660,425 | 20.258985→20.258985 | 2→2 | 0→0 | 0→0 | 9.128260→0 |
| d12 | 0→4 | `∅/∅→b0 bundle/b1 bundle` | `b0+a0→same` | 192,757→251,660,425 | 9.587710→10.192568 | 2→2 | 1→0 | 9.5→0 | 0→0 |

这里的 `bundle` 均为对应 base+adapter，完整 object IDs 保存在 CSV。d04/d06/d07/d10/d12 的旧 action 变化证明旧收益不是
稳健机制收益，而是 restart 未执行当前准备造成的结果。d10/d11 历史负结果没有删除：旧 gap `2.300247/9.128260 s` 仍在
对照原件中；纠正后它们不再是错选点，因为同一模型生命周期在双方对称出现，50%/2× recompute 误差仍未跨越新的差值边界。

## 逐事件机制与 claim 影响

- d04/d06：两路都执行完整 bundle swap-and-reload；该成本是真实路径共同项，不能用于惩罚 recovery。
- d05：依赖安全保留 shared base，仅两个 adapter 依次交换；机制证据保留，但不产生方法优势。
- d08：14 s high-restore 仍使 restart 更优，是修正后保留的负边界。
- d09：模型全 resident 时只比较 input+recompute 与 state+restore，简单阈值已经充分。
- d10/d11：历史 wrong-choice 归因撤销，不代表估计误差普遍无害；只说明原误差倍率在纠正后的这些点未越过边界。

因此历史优势保留为 0/5 个额外正确动作，撤销 5/5。可保留的是 native state recovery、依赖安全 victim plan、真实 adapter
lifecycle 接线和完整事件核算；不可保留的是相对正确简单阈值或正确两步前瞻的算法收益。

## Artifact

- `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/frozen_protocol.json`
- `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/all_method_results.{json,csv}`
- `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/old_to_new_point_comparison.{json,csv}`
- `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/aggregate_summary.json`
- `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/completion_receipt.json`
- `artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2/integrity_manifest.json`
