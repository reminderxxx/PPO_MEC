# CRDCM performance matrix v2 独立派生审查

- `reviewed_at`: `2026-09-28T17:52:48.665343+08:00`
- `literature_cutoff`: `2026-09-28`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `crdcm_performance_matrix_v2_20260928`
- `policy_version`: `tmc_review_policy_v3_20260621`
- source Git commit: `4dc5adb6f6e436983745a0cf485f5221ba4caf0a`
- source Git tree: `2ecb51d8d075d671e3a4d3bed3eb042aad262c08`
- `evidence_level`: `E3_REPRODUCED_OBSERVED_DATA_DEVELOPMENT_PILOT_NOT_HOLDOUT`
- `verdict`: `Not TMC-ready`

## 审查边界与结论

本轮只读 `artifacts/training/crdcm_performance_matrix_v2_20260928/`，没有新增训练、调参、checkpoint 选择、
rollout 或 holdout 开启。独立脚本从 156 个原始评价 episode 重建 CSV 与 aggregate，producer CSV 和 aggregate
均为 `0` mismatch；完整重算同时覆盖 12 个训练 cell、768 training episodes、7,872 actual training steps、
192 updates、12 个 endpoint checkpoint 和 1,612 个评价 request。

矩阵提供了有价值的反证与 trade-off 信息，但不能支持“CRDCM 或 SA-GHMAPPO 优于 baseline”。主要原因不是单一
均值，而是：只有 3 个 raw window 外层独立单元、评价仍是 observed-data development/resubstitution、没有
formal/holdout、seed 交互很强，并且预期的 capacity contrast 在实际 trace 中完全塌缩。

## Evidence inventory

- source artifact integrity：self-excluding manifest 的 1,056 个文件全部重新计算大小和 SHA-256，`0` mismatch。
- execution/protocol：completion、state、evaluation 均 `SUCCEEDED`；12/12 child return code 为 0；14 个 stderr
  均为空；automatic retry 为 0。
- data identity：NGSIM 与 Alibaba 实体文件大小和 SHA-256 与 command manifest 完全一致；config、runtime config、
  window plan 的 SHA-256 也一致。
- budget：实际 924 episodes、9,484 steps；冻结 cap 为 924 episodes、18,480 steps。训练实际 7,872/
  15,360 steps，评价实际 1,612/3,120 steps。
- checkpoint：12/12 为 episode 64、update 16；参数 before/after hash 均变化且 finite；评价前后 exact
  `latest.pt` SHA-256 不变。
- endpoint：`latest.pt` 与 `update_0016.pt` 是两次独立序列化，12/12 byte SHA 和 size 不相同，但加载后的完整
  payload 递归相等，且都记录 `update_count=16`。评价身份必须引用 checkpoint manifest 中 `latest.pt` 的 exact
  byte SHA，不能用 `update_0016.pt` SHA 替换。
- raw rows：156/156 request event、external denominator、exposure request count 一致且 alignment 为 `pass`；
  共 1,612 requests。
- delay：只有 completed workflow 有 delay；83/156 episode 可用，恰好等于 83 个 completed episode。delay 只能作
  conditional-on-completion 比较，不能与 completion coverage 分开隐藏。
- independent derived artifact：`artifacts/analysis/crdcm_performance_matrix_v2_independent_review_20260928/`。

## 独立主汇总

下表所有 learned condition 是 3 seed × 4 scenario × 3 raw window/workflow 的 36 个 episode 描述性汇总；
heuristic 只有 12 个 episode，没有三 seed 复制。`episode mean` 与 `request weighted` 是不同 estimand。

| condition | completed | completion | continuity episode mean | continuity request weighted | transfer episode mean (MB/request) | transfer request weighted | migration overhead episode mean | delay coverage | conditional delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full SA | 18/36 | 0.500000 | 0.764343 | 0.768817 | 24.063036 | 19.494624 | 0.256667 | 18/36 | 1067.666667 |
| signal-off SA | 14/36 | 0.388889 | 0.846006 | 0.857527 | 29.209368 | 24.973118 | 0.393333 | 14/36 | 1201.000000 |
| full MAPPO | 18/36 | 0.500000 | 0.664198 | 0.666667 | 24.063036 | 19.494624 | 0.256667 | 18/36 | 1067.666667 |
| full PPO | 27/36 | 0.750000 | 0.940741 | 0.967742 | 31.613072 | 27.000000 | 0.256667 | 27/36 | 1067.666667 |
| critical-path heuristic | 6/12 | 0.500000 | 0.929630 | 0.951613 | 29.724183 | 26.354839 | 0.530000 | 6/12 | 1201.000000 |

重要的双向结论：full SA 相对 signal-off SA 的 completion 增加 `4/36`，即 `+11.1111` 个百分点；但 continuity
episode mean 同时降低 `0.081663`。因此只能说 signal 改变了 trade-off，不能只报告 completion 正向一侧，也不能
宣称稳定因果优势。

## Seed、动作和 failure trace

| condition | seed 1401 | seed 1402 | seed 1403 |
|---|---:|---:|---:|
| full SA | 9/12 | 9/12 | 0/12 |
| signal-off SA | 4/12 | 1/12 | 9/12 |
| full MAPPO | 9/12 | 9/12 | 0/12 |
| full PPO | 9/12 | 9/12 | 9/12 |

full SA 与 full MAPPO 的 36 个 paired unit 中，24 个动作序列相同；全部 12 个差异都集中在 seed 1403。seed
1403 时：

- full SA：action `3` 为 112/124、action `4` 为 12/124；completion `0/12`，request failures `78`；
- full MAPPO：action `2` 为 112/124、action `4` 为 12/124；completion `0/12`，request failures `116`；
- full PPO：action `0` 为 112/124、action `4` 为 12/124；completion `9/12`，request failures `4`。

full SA/MAPPO 的 aggregate completion、handoff failure、transfer 和 migration overhead 相同，但 continuity 与
request failure 不同；不能简化为“所有结果都相同”。12 个差异 paired row 的 primary failure 类别仍相同：10 个
`cache_dependency_miss`、2 个 `handoff_unprepared`，差异来自失败 request 数量和 deterministic action mode。

所有 seed 都完成 64 episodes、656 actual steps、16 updates，参数确实变化且 finite。seed 1403 的 SA/MAPPO/PPO
checkpoint 和 update logs 都存在；这些记录能定位“最终 deterministic policy 选择了不同动作”，但不能单凭最终
loss/KL/clip fraction 把现象归因为训练不足、未收敛或架构缺陷。没有预注册 convergence diagnostic，也没有额外预算
实验，本轮不作此因果归因。

signal-off SA 的 `4/12, 1/12, 9/12` 同样说明强 seed 交互。尤其 seed 1403 的 signal-off 为 9/12，而 full SA 为
0/12，因此 `18/36 vs 14/36` 的 aggregate completion 不能包装成跨 seed 稳定的 signal benefit。

## 场景名称与真实语义

### `capacity_competition` vs `shared_adapter_reuse_prepare`

两个场景都使用 repeated-reuse binding、migration enabled 和同一组三个 raw window/workflow；唯一冻结配置差异是
capacity `280 MB` vs `360 MB`。归一化 request schedule 在 39/39 paired controller-unit 中相同，exposure fingerprint
因 scenario/capacity 身份而不同；但动作序列和全部主 metric 在 39/39 中完全相同。

280 MB 场景最大 occupancy 为 `0.985714`，360 MB 场景为 `0.766667`，说明容量配置确实进入环境；然而两场景均为
`0` capacity rejection、`0` eviction，未产生 outcome contrast。因此可把它们保留为同一 raw unit 上的两个 paired
treatment 记录，但不能说获得了两组独立支持，也不能把 `capacity_competition` 写成已兑现的容量竞争 robustness。

### `low_reuse_no_migration_negative`

该场景 39 个 episode 中有 26 个 step 的 `adapter_state_migration_overhead > 0`；同时：

- `workflow_state_migration_enabled=false`；
- actual `state_migration_size_mb=0`；
- realized prepare/migration step 为 0；
- 22 个 prepare request 被 suppress；
- 26 个非零 overhead step 都是 failed handoff + cross-RSU cold start。

源码和 trace 一致表明该字段是跨 RSU transfer 在没有有效 `prepare/migrate` 时的 reward penalty，不是实际 workflow
state migration bytes。因此表头应写 `migration overhead penalty`，不能把非零值解释成迁移真的发生，也不能把场景名
理解为该 metric 必然为零。

## Checkpoint 与 trace provenance

checkpoint manifest、paired CSV 和 evaluation receipt 的链闭合：每个 condition/seed 对应一个 exact `latest.pt`
SHA；144 个 learned rows 的 CSV 均引用该 SHA；evaluation before/after SHA 全部相等；独立加载又验证 latest payload 与
update 16 endpoint payload 逻辑相等。

但 144 个 learned raw episode JSON 的 `run_info.checkpoint_metadata` 都是 in-memory override placeholder：path 为空、
episodes/update_count 为 0。当前可以通过 `raw condition_id+seed -> paired CSV checkpoint_sha256 -> checkpoint
manifest -> evaluation before/after hash` 间接完整对账，但 raw JSON 不是 self-contained checkpoint provenance
envelope。这是可复现性 major concern，不影响本轮数值复算，却会阻止把单个 raw JSON 脱离包后独立引用。

## 窗口相关性与统计边界

三个冻结 NGSIM raw window 的 frame/time interval 两两不重叠，interval audit 通过。但独立抽样单元仍只有 3 个 raw
window：seed 是同一 window 上的训练/评价重复，scenario 是同一 window 上的 paired treatment，每个 window 固定绑定
一个 workflow。不得把 156 行、36 learned rows、seed、scenario 或 workflow 当成独立 cluster；也不适合事后发明
显著性阈值、CI 门槛或多目标权重。这里只有 point estimate、paired trace 和 Pareto 描述，外层统计功效很低。

## Pareto trade-off

按预先已有的方向：maximize completion/continuity，minimize handoff failure/transfer/migration overhead；不构造 post-hoc
weighted score：

- learned frontier：`full SA`、`signal-off SA`、`full PPO`；
- `full MAPPO` 被 `full SA` weakly dominated：completion/cost/handoff 相同，full SA continuity 更高；
- full PPO completion/continuity 最高，但 episode-mean transfer 为 `31.613072`，高于 full SA 的 `24.063036`；
- signal-off SA continuity 高于 full SA，但 completion 更低且 transfer/migration/handoff 更差；
- heuristic 只有 12 个固定 episode，单独报告，不并入 36-episode learned frontier，也不冒充三 seed independent sample。

full PPO 在这个 development matrix 更强，不构成改名主算法、择优 seed、换 checkpoint 或把 CRDCM 主线改为 PPO 的授权。

## Hard blockers

1. 没有 formal 或 hidden holdout；全部评价都是 observed-data development resubstitution。
2. 只有 3 个 raw window 外层独立单元，无法支持独立 cluster 推断或 paper-ready statistics。
3. capacity contrast outcome 完全塌缩，不能作为容量 robustness/mechanism realization 的正证据。
4. seed 交互强：full SA/MAPPO 为 `9,9,0`，signal-off SA 为 `4,1,9`；稳定优势 claim 不成立。

## Major concerns

- 144 个 learned raw episode 不含 self-contained checkpoint identity，只能借助包内 CSV/manifest 间接对账。
- strong heuristic 仅 12 episodes，不能当成 36 个独立样本或三 seed baseline。
- delay coverage 为 83/156，只能在同一有效样本/coverage 定义下比较。
- `latest.pt` 与 `update_0016.pt` byte 不同但 payload 相等；引用时必须使用实际被评价的 latest SHA。

## Minor concerns

- `low_reuse_no_migration_negative` 的名字容易让读者误以为 migration-overhead metric 应为零，需要在表注中解释其
  reward penalty 语义。
- episode mean 与 request-weighted 结果不同；必须显式标注 aggregation unit，不能混用。

## Scorecard

本轮是单一 development artifact 审查，不对完整论文打伪精确总分：Novelty、technical/modeling、baseline
independence、formal statistics/holdout、mechanism realization、robustness/generalization/scalability 和 paper-level
reproducibility 均记 `N/S`。矩阵自身达到独立派生重算意义上的 E3，但不能替代缺失的 formal/holdout/robustness 和
novelty 证据，因此 verdict 固定为 `Not TMC-ready`。

## Safe claims

- 冻结 observed-data development matrix 完成，156 个 raw evaluation episode 可独立重建 producer CSV/aggregate，
  mismatch 为 0。
- full SA 相对 signal-off SA 的描述性 completion 为 `18/36 vs 14/36`，同时 continuity 为 `0.764343 vs
  0.846006`；这是双向 trade-off 且有强 seed interaction。
- full PPO 在该固定 development matrix 的 completion 为 `27/36`，但 transfer 高于 full SA；只能作 Pareto 描述。
- full SA 在本 artifact 的 episode-mean vector weakly dominates full MAPPO；范围仅限该 observed-data pilot。

## Prohibited claims

- CRDCM、SA-GHMAPPO、MAPPO 或 PPO 已获得 formal、独立泛化或 paper-ready superiority。
- signal-on 对 signal-off 存在稳定、跨 seed 的因果优势。
- capacity scenario 已验证容量竞争鲁棒性，或两场景提供两个独立样本。
- seed 1403 失败证明训练不足、收敛失败或架构缺陷。
- heuristic 的 12 个 episode 等价于 36 个独立 baseline observation。
- no-migration 场景非零 overhead 表明真实 state migration 已发生。

## 唯一 first-order 后续建议

先不要调算法。冻结一个 outcome-blind capacity-stress evaluation，使 trace 真实触发 capacity rejection、eviction 或可观测
的容量结果差异；随后只用当前已冻结 checkpoints，在更多两两不重叠的 outer raw windows 上评价。不得用本轮结果择 seed、
换 checkpoint、重命名主算法或回看结果后设置多目标权重。完成新的独立审查前停止晋级。

## 复算入口

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/audit_crdcm_performance_matrix.py \
  --source-root artifacts/training/crdcm_performance_matrix_v2_20260928 \
  --output-dir <new-create-only-audit-directory>
```

派生表：`independent_episode_rows.csv`、`condition_aggregate.csv`、`condition_seed_aggregate.csv`、
`condition_scenario_aggregate.csv`、`action_failure_summary.csv`、`training_checkpoint_audit.csv`；结构化审查为
`independent_review.json` 和 `pairwise_trace_audit.json`。
