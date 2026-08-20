# G14R Formal Execution Contract Repair and Protocol Restart

## Review identity

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
