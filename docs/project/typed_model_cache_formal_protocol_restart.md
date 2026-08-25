# G14R Formal Execution Contract Repair and Protocol Restart

## G14R2 outcome

G14C v2 `typed_model_cache_formal_20260820_164251_g14c_v2` 在首个 training cell 前失败：Protocol v1.1
train command 未传 `max_mobility_rows`，共享入口使用 1500 raw-row default，仅形成 1,151 provider
frames，无法定位 frozen train plan。该 run 永久为
`INVALID_PROTOCOL_OR_IMPLEMENTATION / invalid_before_performance_execution`，0/150 training、0
checkpoint、0 formal、holdout unopened；return code 1 分类为 `data_window_unreachable`，不可 retry 或
resume，旧 artifact 不覆盖/删除/改写。

G14R2 冻结 window consumption contract `1.0.0`，把原 60 windows 绑定到 11,850,526 raw-row prefix、
73,871 provider frames、segment/raw frame/raw time/provider offset 与 fingerprint。60/60 reachability、
150/150 training commands、30 条 dev/formal/support commands、ledger `2.0.0` append chain 和 16-cell
non-formal rehearsal 均通过。Protocol v1.2 semantic SHA-256 为
`718c0f78aabd5d01012df31267626eab74a51b2b621aaa67a535c5b60e655ca9`；split hash 保持
`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`。Readiness v4 为
`READY_FOR_G14C_V3_CLEAN_TRAIN_AND_FORMAL`。

本轮没有训练正式 checkpoint，没有运行 formal/holdout/hidden，没有观察性能结果，没有启动 G14C v3
或 G15。Holdout 仍 `sealed=true/opened=false/consumed_permanently=false`。详细合同与审计见
`typed_model_cache_formal_window_consumption_contract.md` 及
`artifacts/analysis/typed_model_cache_formal_window_repair_20260820_g14r2_v1/`。

## G14R historical review identity

- `reviewed_at`: `2026-08-20`
- `literature_cutoff`: `2026-08-20`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_formal_protocol_restart_20260820_g14r_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- implementation baseline: `351fdb8a309614a751cedb180ecaccf2a681db2d`
- evidence level: `E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE`

## Outcome and immutable boundary

G14C v1 protocol `1.0.0`（semantic SHA-256
`41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4`）与 run
`typed_model_cache_formal_20260820_g14c_351fdb8_v1` 永久状态为
`INVALID_PROTOCOL_OR_IMPLEMENTATION`。Phase-0 在测试和训练前发现六个 critical blockers，故旧 run
不 resume、不覆盖、不删除，也不能作为性能证据；其 training/checkpoint/formal result count 都是 0。

G14R 没有运行 G14C v2、formal、holdout、hidden 或 G15，没有训练正式 checkpoint，也没有改变科学
问题或 primary comparisons。修复只把已声明语义变为可机械执行合同。

## Closed blockers

1. checkpoint cadence：共享训练入口新增正整数 cadence；formal 固定 4，legacy omission 保持 1；
   candidate indices 为 `[4,8,12,16,20,24,28,32]`，resume 校验一致频率。
2. SA binding：manifest-bound agent companion 将 `auxiliary_coef=0.06` 传入共享 builder，并对实例、
   summary、checkpoint metadata 三处审计；SA loss 和用户 dirty SA 文件未修改。
3. primary endpoint producer：metrics 1.2 实现 byte-ready 与 typed transfer/request pure reducer，写入
   episode summary、row 和 nullable aggregate，stored reducer 与 raw event 不一致即失败。
4. support values：capacity、object-size、transfer-cost、handoff、reuse、base sharing、ablation、prediction
   boundary 与 oracle state limit 均有 fixed level identity；缺少安全 runtime transformer 的 level 明确
   `unavailable_pre_execution`。
5. typed support runner：统一消费 typed MB runtime、catalog fingerprint、fairness 1.1、checkpoint
   provenance、protocol/split、seed 与 setting ID；禁止 typed slot、legacy fallback、自由 CLI override、
   G12 supervised、KV 和 HF metadata profile。
6. execution assets：持久化三档 runtime、formal/dev/support fairness manifests、agent config、完整命令
   模板、dev selection、checkpoint freeze、statistics/integrity wrappers 与 append-only phase runner。

## Endpoint formulas

`full_service_ready_byte_hit_rate` 是 full-ready request 的 unique base+adapter dependency resident bytes
之和除以全部 eligible typed request 的同口径 bytes。共享 base 只在单 request 内去重；不同 request
各自计一次需求。Partial readiness 的整请求 numerator 为 0；缺 size 时值为 null/partial 并报告
coverage；零分母为 null；legacy unavailable。

`transfer_mb_per_request` 的 numerator 固定为 base transfer + adapter transfer + workflow-state migration；
其他 typed transfer 单列且排除。分母为 typed request event count，单位 decimal MB/request。

## Frozen support and scalability

- capacity：`[288,576,864] MB`，baseline 576。
- object size：`[0.75,1.0,1.25]`；transfer cost：`[0.5,1.0,1.5]`；reuse：
  `[low,medium,high]`；base sharing：`[1,3,6] adapters/base`。这些 level 已冻结，但当前因缺少
  fingerprint-safe runtime consumer 而 `unavailable_pre_execution`，不能支撑对应 sensitivity claim。
- handoff：`stable_first` 与 `handoff_pressure`，由独立冻结 fairness manifest 绑定。
- ablation：typed full/no prediction 可执行；legacy-only、no sharing、no state migration、fixed no
  eviction 显式 unavailable。
- prediction boundary：baseline、no prediction、noise 0.2、confidence 0.7、delay 2、drop 0.3。
- scalability：RSU `[2,3,4]`、active vehicles `[4,8,16]`、DAG nodes `[5,10,20]`、typed objects
  `[4,8,10]` 当前 unavailable；oracle state limits `[1000,10000,100000]` 可执行。Wall-clock 固定 3
  repetitions，memory 为 Python `tracemalloc` peak increment；不推断 RSS。

## Commands, phases, and rehearsal

Protocol template 展开全部 10 learned agents × 5 seeds × 3 capacities 的 150 个 training cells，以及
dev checkpoint selection/freeze、formal cache-policy/controller、ablation、robustness/prediction、
scalability、statistics、integrity 和 completeness-only gate。13 阶段 runner 使用 append-only JSONL；
hash mismatch、顺序违规、失败覆盖、formal 后训练、output-root 冲突与 holdout 访问均拒绝。

非正式 rehearsal 使用 G07 controlled non-hidden smoke window，覆盖 SA-GHMAPPO、PPO、MAPPO、
Cache-Offload-DRL，2 seeds 与 288/576 MB。16 个 training cells 均只保存 update 4，checkpoint
restore/provenance 通过，SA resolved auxiliary 为 0.06；36 个 raw summaries 与 rows 的六 primary
endpoint 对账通过；另运行一个 ablation、一个 prediction robustness 与一个 oracle scalability
setting，并将 phase simulation 推进至 `complete_without_holdout`。该 rehearsal 不进入论文结论。

## Protocol restart, split, seal, and readiness

- new version：`1.1.0`
- new semantic SHA-256：`b8bbb53d6af47d111b840efbb53d3389485535d66c8de19b747e2a5727786629`
- split semantic SHA-256：`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`
- split change：false；60-window semantics 未重写，仅新增 companion metadata
- holdout：`sealed=true`、`opened=false`、`consumed_permanently=false`
- Readiness v3：`READY_FOR_G14C_V2_CLEAN_TRAIN_AND_FORMAL`

Readiness 只表示 execution contract 可从 Commit A2 的 clean worktree 启动；protocol v1.1 仍不等于
formal completed、paper-ready 或 G14 completed。完成 G14R 后必须返回计划窗口，不自动启动 G14C v2。

## 2026-08-24 G14R3 restart boundary

后续 Protocol v1.2/G14C v3 在 150/150 training cells 与 1,200 candidates 之后、首次 dev performance
之前因 clean-worktree workflow path 身份错绑失败，永久状态为
`invalid_before_dev_performance_execution`。Dev performance/selection/formal 均为 0，holdout 未开启；旧 run
禁止 resume 或 checkpoint salvage。

G14R3 冻结 Protocol v1.3、portable identity 1.0.0 与 Readiness v5。成功 exact non-formal rehearsal 使用
全新 root、真实 dev selector/freeze，完成 13 phase 至 `complete_without_holdout`；没有正式数值或 claim。
Protocol semantic SHA-256 为 `1525b7cb...ac17`，Readiness 为
`READY_FOR_G14C_V4_CLEAN_TRAIN_AND_FORMAL`。下一步仍必须由独立 G14C v4 任务在最终 A4 clean worktree
从头训练，不自动启动。

## 2026-08-25 G14R4+ restart boundary

G14C v4 Run A 在 150/150 training cells 后因 ledger terminalization timing contract 失败；Run B 在首个
冻结子命令前因相对 `.venv/bin/python` 失败。两者均为永久 `INVALID_PROTOCOL_OR_IMPLEMENTATION`，不能
合并、resume、finalize、复制 checkpoint 或作为新 run 的 ledger/cell marker 来源。

G14R4+ 冻结 Protocol v1.4（semantic `4429531d...4155d`）、portable execution environment `1.0.0`、
phase ledger `3.0.0`、cell ledger `1.0.0` 和 Readiness v6。no-.venv clean snapshot 的非正式链通过
4 agents × 2 seeds × 2 capacities、8/16 same-run resume、75/150 interruption、150/150 后 simulated
terminal append failure 与 finalize-only，并完成真实 dev/freeze/tiny formal-like/support/statistics/integrity。
本轮正式训练/formal/holdout/hidden/G15 计数均为 0；下一步只能由独立任务从 pushed Commit A5 新建
G14C v5 run。
