# 对称恢复成本机制边界检查：冻结方案与全量结果

## 身份与范围

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-05`（未新增检索）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `recovery_cost_boundary_check_20261006_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 方案/实现冻结提交：`800f0a12f13e34d7de19aee375124813b8817e3a`
- 执行提交：`30691177bcac540ec1650d14494fe48326805a8c`
- `evidence_level`: `E2_ARTIFACT_AUDITED`（bounded synthetic mechanism boundary check）
- `verdict`: `ONLINE_RULES_IDENTICAL / TWO_FROZEN_ESTIMATION_FAILURES / NO_NEW_CAPABILITY`

本组只验证修复与机制边界，不是独立真实 workload、无线测量或统计泛化证据。0 generate、0 RL、0 download、0 old
holdout。第一次命令在创建 output root 前因手工填写的 expected commit 错误而 fail-fast，没有启动科学执行；修正为 Git 实际
HEAD 后唯一一次科学执行完成 6 点、24 条隔离 native path，wall `0.232873 s`，无重试或点位搜索。5/5 manifest 文件
size/SHA-256 byte-exact。

## 事前冻结方案

完整机器配置为 `configs/experiment/recovery_cost_boundary_check_v1.json`，在任何结果产生前已随 `800f0a1…` 提交。一次性
冻结内容如下：

- 场景来源：沿用原 v1 native catalog、96/128 MiB base、8 MiB adapter、2,185 B state、192,757 B input 和同一
  dependency-safe sequential LRU；
- 方法权限：简单阈值只读 state/input 与 restore/recompute estimate；代价感知与两步只读初始 resident、声明的 current/next
  依赖、native preview 和同一事前 estimate；离线参考在两条实际分支完成后读真值；
- 主要指标：动作、victim、resident trace、model/dynamic bytes、modeled completion、完成量、failure、recompute 和 reference
  gap；
- 停止条件：6 点全部只执行一次，保留持平、错选和失败，不按输出继续搜索；
- 生成规则：1 个完整 bundle 共同 prepare/reload、1 个当前模型已 resident exact-fit、1 个 shared-base adapter-only swap、
  1 个容量宽裕无驱逐、2 个 200/1000 Mbps 对换的 analytic break-even estimate mismatch。

现有 production/cache 能力可合法表达全部五类覆盖；没有标记为“未覆盖”的请求，也没有新增跨 workflow 或共享队列机制。

## 全量结果

四方法均完成 12/12 节点、0 service failure、0 deadline violation。三种在线规则 6/6 动作相同；前四点与离线参考相同，
后两点由冻结 link estimate error 产生相反方向错选。加载/驻留生命周期在 restart/recovery 6/6 完全相同。

| 点 | 覆盖 | 在线动作 / 参考 | victims（current / future） | terminal residents | model B | online dynamic B | online time s | complete/fail/recompute | online gap s |
|---|---|---|---|---|---:|---:|---:|---|---:|
| b01 | 双方共同准备/重载 | 4 / 4 | `b1 bundle / b0 bundle` | `b1+a0` | 251,658,240 | 2,185 | 20.258985 | 2/0/0 | 0 |
| b02 | 当前模型已驻留 | 4 / 4 | `∅ / ∅` | `b0+a0+b1+a0` | 142,606,336 | 2,185 | 11.514833 | 2/0/0 | 0 |
| b03 | shared base 依赖保留 | 4 / 4 | `b0.a1 / b0.a0` | `b0+a1` | 16,777,216 | 2,185 | 1.468503 | 2/0/0 | 0 |
| b04 | 容量宽裕无驱逐 | 4 / 4 | `∅ / ∅` | `b0+a0+b1+a0` | 251,658,240 | 2,185 | 20.258985 | 2/0/0 | 0 |
| b05 | actual 200 / estimate 1000 | 0 / 4 | `∅ / ∅` | `b0+a0+a1` | 0 | 192,757 | 0.087710 | 2/0/0 | +0.001472 |
| b06 | actual 1000 / estimate 200 | 4 / 0 | `∅ / ∅` | `b0+a0+a1` | 0 | 2,185 | 0.086168 | 2/0/0 | +0.004626 |

`bundle` 为相应 base+adapter；完整 object IDs 和每个事件 pre/post resident 位于 `all_method_results.json/csv`。表中时间由
冻结链路参数计算，不是实测网络 wall time。b05/b06 的 prefix recompute 冻结为 0，以隔离 input/state bytes 与 link estimate
的盈亏边界；这两个点只证明估计误差可以翻转选择，不代表现实误差分布。

## 方法汇总与解释

| 方法 | 匹配参考 | recover | modeled time s | total B | 完成/fail | 相对参考额外 s |
|---|---:|---:|---:|---:|---|---:|
| 原简单阈值 | 4/6 | 5 | 53.675185 | 662,903,714 | 12/0 | 0.006098 |
| 驱逐代价感知 | 4/6 | 5 | 53.675185 | 662,903,714 | 12/0 | 0.006098 |
| 正确两步前瞻 | 4/6 | 5 | 53.675185 | 662,903,714 | 12/0 | 0.006098 |
| 微型离线参考 | 6/6 | 5 | 53.669086 | 662,903,714 | 12/0 | 0 |

规则与正确两步前瞻完全相同，也与正确简单阈值完全相同；不使用微秒噪声包装效率优势。新规则没有新增动作能力。修复后的
正面结果是工程/审计正确性：共同成本被双方各计一次，cache 状态相互隔离，shared base 依赖安全，且 decision estimate 不读
实际评分。负面结果是：更完整的 ledger 没有带来更优选择，边界误差仍由所有在线方法共同承受。

## Artifact

- `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/frozen_protocol.json`
- `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/all_method_results.{json,csv}`
- `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/aggregate_summary.json`
- `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/completion_receipt.json`
- `artifacts/analysis/recovery_cost_boundary_check_20261006_v1/integrity_manifest.json`
