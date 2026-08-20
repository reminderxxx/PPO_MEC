# Typed Model-Cache Split Exclusion Audit

## 审查身份与结论

- `reviewed_at`: `2026-08-20T12:00:00+08:00`
- `literature_cutoff`: `2026-08-20`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_formal_protocol_freeze_20260820_g14b_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- baseline Git commit：`5a14d6e4a8ace19b4e2c1612bd3c6989deeda633`
- evidence level：`E2_PROTOCOL_AND_CONTRACT_VALIDATED_NO_PERFORMANCE_DATA`

结论：历史账本、保守排除、完整 NGSIM metadata inventory、result-blind split builder 和 pairwise
独立性审计均通过。冻结 split 为 train/dev/formal/sealed-holdout=`24/12/12/12`，所有 60 个 outer
windows 在 raw frame、raw time 和 segment-run identity 上互斥，且至少相隔 24 frames。Formal 与
sealed holdout 各有 12 个独立 outer windows。没有读取 cache opportunity、handoff outcome、reward、
oracle gap、typed hit rate 或 agent performance 来构造 split。

## Historical usage registry 1.0.0

账本 semantic SHA-256 为
`09ee3109f6789cada996870bcb6a3dac9496a4d98f8373c11cb99f61f55beaae`。Metadata-only parser 扫描
1,209 个带 `selected_window_plan` 的历史 config/artifact JSON，只解析窗口 identity/interval 字段，
不解码 performance outcome。

- raw window references：34,661
- unique outer intervals：668
- known raw intervals：250
- unknown unique intervals：418
- unknown references：25,050
- duplicate references collapsed：33,993
- mixed/full outer deduplications：74
- unique historical run IDs：931
- parse failures：0

覆盖的 purpose 包括 train、dev、calibration、formal、holdout、hidden、future validation、robustness、
scalability、ablation、checkpoint selection、window scanning、opportunity/predictor validation、
rehearsal/smoke 和 benchmark observation。曾参与设计、选择或结果观察的 interval 永久 consumed；
降级或不再引用不恢复 sealed 资格。

无法唯一恢复 raw identity 的 418 个 interval 形成 conservative exclusion。它们覆盖
`lankershim`、`peachtree` 和 `us_101` 三个 segment；因此 G14B 不从这些 segment 选 formal 或
holdout。`i_80` 不在该 unknown scope，且全历史 registry 未发现其 raw interval consumption。

## NGSIM available interval inventory

输入文件：
`data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv`
（2,118,175,938 bytes，11,850,526 rows），SHA-256：
`ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`。

扫描只使用 Location、Frame_ID、Global_Time、vehicle availability 和 coordinate validity。完整
runner scope 包含 `i_80`、`lankershim`、`peachtree`、`us_101`，共 73,871 个可映射 frame、10 个
continuous runs、1 个显式 discontinuity；14,645 个 conflicting raw-frame identities 被排除在
continuous runs 外。5,000,000-row legacy prefix 会产生 46,843 个跨边界 partial frames，已按三个
连续 raw-time ranges 记录并保守判无效。

时长账本（原始时间单位）：

| 项目 | duration |
| --- | ---: |
| continuous available total | 7,387,100 |
| known historical union（gross） | 455,800 |
| unknown-scope conservative exclusion | 4,599,800 |
| known consumed outside conservative scope | 0 |
| minimum-gap exclusion outside上述范围 | 0 |
| quality exclusion | 9,500 |
| remaining eligible | 2,777,800 |

Known historical union 全部落在已保守排除的三个 segment 内，所以只扣除一次。剩余 eligible 数据
位于 I-80；按 24-frame window 与 24-frame minimum gap，可形成 579 个不重叠候选：
`i_80_run_001=208`、`i_80_run_002=166`、`i_80_run_003=205`。因此 formal 12 与 sealed holdout 12
均可行，无需降低数量或复用历史 formal/hidden。

## Split protocol 1.0.0

- protocol：`typed_model_cache_split_protocol_version=1.0.0`
- semantic SHA-256：`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`
- split generation seed：1401
- window length：24 frames
- minimum gap：24 frames
- minimum vehicles：2
- allowed segment：`i_80`
- source scope：完整 11,850,526 rows
- tie-break：`SHA-256(seed, segment-run, raw-frame, raw-time)`，再按 `window_id`

Frozen counts：train 24、dev 12、formal 12、sealed holdout 12。四个 plan 位于
`configs/experiment/typed_model_cache_formal_protocol_v1_20260820/`。Holdout plan 可由 validator
读取 identity/interval，但普通 runner 不得开启或读取其结果。

Candidate builder 的 allowed inputs 仅为 raw continuity、minimum vehicle availability、coordinate
validity、historical exclusion 与 minimum gap。它显式拒绝 reward、cache hit/byte hit、cache
opportunity、estimated handoff、mechanism score、oracle gap、transfer outcome 和 agent performance。
重复构建得到相同 semantic hash；修改任一语义字段、CLI 覆盖或 formal/holdout 少于 12 均 fail-fast。

## Pairwise independence

完整矩阵包含 `C(60,2)=1,770` 对，全部分类为 `safe`，conflict=0。审计同时检查：

- raw frame overlap
- raw time overlap
- segment-frame / segment-run identity overlap
- split 内和 split 间 minimum gap
- historical registry exact overlap、insufficient gap 与 unknown conservative conflict
- duplicate、nested、same interval different ID、mixed/full 同 interval
- seed/workflow 伪重复和 outer cluster count

Outer cluster counts 固定为 train 24、dev 12、formal 12、sealed holdout 12；seed/workflow 重复和
mixed/full 投影均不增加 outer count。任何 formal/holdout conflict 会阻止 protocol 生成。

## 证据与限制

机器证据根目录：
`artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/`。关键文件为
`historical_window_usage_registry.json`、`available_interval_inventory.json`、
`historical_exclusion_audit.json`、`candidate_window_inventory.json`、`split_manifest.json`、
`split_overlap_matrix.json` 和 `split_independence_audit.json`。

本审计证明 interval exclusion 与 split protocol 可进入 clean-worktree execution；它没有运行正式
episode、训练 checkpoint、选择 checkpoint、执行 formal/holdout/hidden 或产生性能证据。NGSIM +
Alibaba + controlled typed catalog 是跨源研究原型，不得写成真实联合 model-cache trace。
