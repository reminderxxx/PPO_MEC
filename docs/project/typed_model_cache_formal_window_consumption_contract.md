# Formal Window Consumption and Phase Ledger Contract

## Review identity

- `reviewed_at`: `2026-08-20T18:23:21.124778+08:00`
- `literature_cutoff`: `2026-08-20`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_formal_window_repair_20260820_g14r2_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- implementation baseline：`89049c92b41054d78294893643f241926181645a`
- evidence level：`E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE`

## Failure boundary

G14C v2 run `typed_model_cache_formal_20260820_164251_g14c_v2` 使用 Protocol v1.1 semantic SHA-256
`b8bbb53d6af47d111b840efbb53d3389485535d66c8de19b747e2a5727786629`。训练模板没有显式传递
mobility source range，因此入口回落到 `max_mobility_rows=1500`，在首个 frozen train window 的
provider offset 处失败。Failure audit SHA-256 为
`5da5e20395e5c1e48bf2e267ce757248d024246bdc121d4d2b33ca4f8c6c594b`，旧 phase ledger SHA-256 为
`78ac969b024f205da8dbdda5541527b01a5746bc4e5b8d3f12a7a0ed73574e79`。

该 run 永久为 `invalid_before_performance_execution / INVALID_PROTOCOL_OR_IMPLEMENTATION`：0/150
training、0 checkpoint、0 formal episode/result，holdout 未开启。Return code 1 分类为
`data_window_unreachable`，不是 infrastructure retry，禁止 resume、覆盖、删除或改写旧 ledger。

## Loader identity and source range

冻结 split 的 `frame_offset` 是 NGSIM loader 对 raw rows 完成数值解析后，按
`(normalized Location, Global_Time)` 形成 provider frame、再按 `(segment, time)` 排序所得的全局
provider-frame index。它不是 raw CSV row、raw `Frame_ID`、segment-local rank 或 window rank。
`max_mobility_rows` 由 `read_csv(nrows=...)` 在清洗、segment grouping 和 vehicle materialization 前截断。

完整 source 有 11,850,526 raw rows、2,118,175,938 bytes，SHA-256 为
`ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`，解析为 73,871 provider
frames。60 个冻结 window 的 requested offset 范围为 1,736--27,786。NGSIM 文件按 vehicle 组织；为
保留任一目标 I-80 frame 之前的全部 I-80 provider identity，并保留目标 frame 的全部 vehicles，四个
split 的最大 minimum-safe prefix 都落到最后一条 raw row。因此冻结范围为：

- `start_row_inclusive=0`
- `end_row_exclusive=11,850,526`
- `source_row_count=11,850,526`
- `margin_rows=0`
- 超出 source：reject

这个数字来自对完整 source、所有 frozen raw-time keys 和每个 I-80 time 的最大 raw row index 的实际
扫描，不是人为估算或 safety buffer。仅增大默认值仍不足以冻结身份；正式消费者还必须显式绑定
window plan、segment/time identity、loader/preprocessing 和 fingerprint。

## Window consumption contract

`formal_window_consumption_contract_version=1.0.0`，semantic SHA-256 为
`ec475799b3fba4a3af3e4372e7c25781c6565a88ec814322b4cd4d447fef2771`。每个 evaluation unit 冻结
split/window、segment/run、raw frame/time、provider local/run-local、requested offset、length、sampling、
source hash、loader/preprocessing/vehicle-selection/RSU identities、minimum required prefix、observed raw-row
interval 与 expected fingerprint。

正式 loader 扫描完整冻结 prefix，但只 materialize contract 中的 segment/time frames；它保留同一 raw
frame 的全部 vehicles，再构造 `RealMobilityBundle`。Training 与 benchmark 通过同一入口加载相同 bundle，
不得各自重新做 window rank selection。Source path/size、range、ordered selector、24-frame length、
`auto_dominant_tight` RSU layout、allowed vehicle selection 或 plan hash 任一漂移都在 episode/checkpoint
写入前 fail-fast。

## Reachability and commands

真实 `--validate-window-plan-only` preflight 扫描 source 并验证 24 train、12 dev、12 formal 和 12 sealed
holdout windows。60/60 的 raw frame interval、raw time interval、provider interval、window length、vehicle
coverage 与 observed/expected fingerprint 全部一致。Holdout 行固定 `metadata_only=true`，validator 不
构建 agent、不执行 episode、不读取 performance fields。

Protocol v1.2 command audit 对 150 个 training cells 逐条解析并验证 agent/seed/capacity、typed runtime、
train plan、11,850,526-row range、cadence=4、SA config、唯一 output 与 holdout=false；150/150 pass。
另外 30 条 dev/formal/support commands 全部解析并绑定 dev 或 formal plan。所有 mobility templates 都显式
携带 source/path、range、plan、selector、length、vehicle selection 和 RSU layout；普通 runner 不展开
holdout command，CLI 重复/覆盖冻结字段被拒绝。

## Phase ledger 2.0.0

每个 phase 先 append `running` record，结束时 append `completed` 或 `failed` terminal record。共同字段为
phase、sequence、status、started/completed、wall-clock、input/output hashes、expanded commands/identity、
return/retry、failure class/message reference 和 previous/current record hashes。Running 可暂缺 completion；
terminal 必须有 timezone-aware completion，且 system-time delta 与 monotonic wall-clock 在 2 秒容差内。
Terminal phase 不可再写；resume 只允许 append，整条 SHA-256 chain 每次写入前后重验。

冻结 failure enum：`infrastructure_retryable`、`infrastructure_terminal`、`protocol_mismatch`、
`implementation_error`、`data_window_unreachable`、`test_failure`、`training_failure`、
`artifact_integrity_failure`、`user_interruption`。Return code 75 只允许
`infrastructure_retryable`，且至多原命令 retry 一次。

## Rehearsal, protocol, and boundary

Non-formal rehearsal 使用 frozen train loader，覆盖 SA-GHMAPPO、PPO、MAPPO、Cache-Offload-DRL，seeds
7/13 与 288/576 MB，共 16 个 tiny training cells；只保存 update 4，restore/provenance 通过。两个 train
boundary windows 的 tiny evaluation 在两档 capacity 各产生 36 rows，summary/row/metrics 完整。Dev/
formal 的 min/max boundary 只做 loader identity rehearsal；holdout min/max 只做 metadata reachability。

Protocol v1.2 semantic SHA-256 为
`718c0f78aabd5d01012df31267626eab74a51b2b621aaa67a535c5b60e655ca9`；split semantic SHA-256 保持
`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`。Agent、seed、budget、
capacity、endpoint、support、statistics、claim、checkpoint cadence 与 SA coefficient 均未变化。

Readiness v4 为 `READY_FOR_G14C_V3_CLEAN_TRAIN_AND_FORMAL`，只授权未来从 Commit A3 clean worktree
另立执行任务。本轮正式 checkpoint/episode/performance count 为 0；holdout
`sealed=true/opened=false/consumed_permanently=false`；未运行 formal/holdout/hidden/G15，不是 G14 或
paper-ready 完成。
