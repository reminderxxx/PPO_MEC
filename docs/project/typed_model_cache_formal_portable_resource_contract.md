# Typed Model Cache Formal Portable Resource Contract

版本：`portable_resource_identity_contract = 1.0.0`
Resolver：`resource_resolver_version = 1.0.0`
Protocol：`typed_model_cache_formal_protocol_version = 1.3.0`
状态：`READY_FOR_G14C_V4_CLEAN_TRAIN_AND_FORMAL`（只表示可启动下一独立任务，不表示已启动）

## 修复对象与永久边界

G14C v3 run
`/private/tmp/ppo_mec_g14c_v3_a7c9e8e/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260820_203430_g14c_v3`
永久标记为 `invalid_before_dev_performance_execution`。Failure audit SHA-256 为
`476cfc3f57312263da7dff388a89c088e4716d43b1949eb121598c86dc5ac3af`。该 run 完成 150/150 training
cells 并生成 1,200 candidate checkpoints，但 dev performance、selected checkpoint、formal count 均为 0，
holdout 未开启。旧 checkpoint 只能审计存在、hash 与训练阶段事实；禁止 resume、复制、dev/formal 消费或
进入新 checkpoint manifest。

G14R3 不运行正式训练、formal、holdout、hidden、G14C v4 或 G15，不生成 paper claim。未来 G14C v4
必须使用新 run root，从冻结矩阵重新训练。

## 科学身份与位置

科学身份不等于主机绝对路径。每个外部资源的科学身份固定包含：

- logical resource ID 与 resource role；
- content SHA-256 与 byte size；
- schema/parser version 与 revision fingerprint；
- expected logical relative path；
- required、allowed resolver、provenance 与 relocation policy。

`semantic_identity_fingerprint` 只对上述字段做 canonical JSON SHA-256。本机 absolute path、resolution root、
resolved/original location、symlink lexical path、validation time 和 output root 只进入 runtime audit，不进入
Protocol、fairness 或 checkpoint 科学身份。

## 解析与 fail-fast

共享入口为 `src/runtime/portable_resource_identity.py`。显式允许的解析器只有 `explicit_path`、`data_root`、
`worktree_root`、`manifest_relative`、`protocol_artifact_root` 与 `checkpoint_root`。禁止根据 cwd、文件名、
环境猜测或网络下载。候选文件必须逐个校验 role、schema、size 与 content hash；同一 logical ID 出现不同
content 的多个本地候选时直接拒绝。内容相同的跨 worktree/data-root relocation 允许，并记录 symlink 与
resolved path audit。

以下情况必须拒绝：content/size/schema 漂移、logical ID 不存在、mobility/workflow role swap、同名异 hash、
多候选冲突、缺失必需文件、CLI content override、无 checkpoint provenance，以及任何指向无效 G14C v3
run 的 checkpoint。

## Fairness、window 与 checkpoint

历史 fairness 1.1 manifest 不重写。`fairness_portable_identity_companion.json` 将 legacy absolute path 解释为
非语义位置；validator 同时检查 legacy location 与 current root logical path，只接受 size/hash 相同的迁移。
CLI mobility、workflow、catalog 与 window plan 以内容身份对账。

Window consumption contract semantic SHA-256 保持
`ec475799b3fba4a3af3e4372e7c25781c6565a88ec814322b4cd4d447fef2771`。v1.3 只加入被 semantic projection
排除的 `source.runtime_resolution` companion，使共享 resolver 已验证的同内容 mobility location 可被消费；
旧 v1.2 contract 继续保持 exact-path 兼容诊断。

Checkpoint freeze 的科学身份绑定 checkpoint hash、agent、seed、capacity、execution commit、Protocol、
catalog/runtime/split/window identity、dev selection values 与 update index。Artifact location 单独记录，迁移后
必须重验 hash；绝对路径不参与 selection/freeze hash。Formal/support 只能消费本 run freeze 产生的 portable
seed/provenance manifests。

## 命令绑定与验证

Train/dev/formal/support templates 显式绑定 mobility、workflow、catalog、window plan、window contract、
Protocol、fairness、runtime 与 checkpoint logical IDs/root。Dev selector 现在强制显式 workflow path，并将其
传给每个 benchmark child。Support runner 也显式绑定 workflow，并真正附加冻结 setting 对应的 predictor
flags。

G14R3 展开验证 186 条命令，其中 150 条 training、24 条 formal/support；main 与 clean worktree 的 186/186
command shapes 在 `<REPOSITORY_ROOT>`/`<DATA_ROOT>` 归一化后完全一致。12 个路径负例全部按合同通过。

## 非正式 exact phase-chain rehearsal

成功 run 为 `rehearsal_runs/g14r3_clean_c3b99bc`，execution snapshot 为 clean commit `c3b99bc`。它只消费
G07 public controlled window，覆盖 SA-GHMAPPO、PPO、MAPPO、Cache-Offload-DRL，seeds 7/13，capacities
288/576 MB，共 16 个 tiny training cells；只保存 update 4，SA coefficient 为 0.06。

真实 dev selector 评估 16 candidates 并选择 16；真实 freeze 冻结 16，freeze SHA-256 为
`5ef988d07ca1907867ff1c219f1b1dd3f4cf38db87a3339993948a89fd83267c`。随后完成 cache policy、controller、
ablation、robustness、scalability、statistics、artifact integrity、non-formal completeness gate 与
`complete_without_holdout`。Ledger 为 26 records/13 terminal phases，last record hash
`817b0dd40488dc72e9103efd3a3f2e5912dde1fdaf80597e8156146a2e13da29`。正式训练/评估计数均为 0，
holdout/hidden 未用，paper claims 为空，旧 run checkpoint reuse 为 false。

前三个 fresh attempts 均在 training 前停止并保留审计：一次解释器 symlink 配置错误、一次外层会话中断、
一次 `hidden` 保守禁词护栏触发。成功 run 未 resume 这些 root。

## 冻结 hashes 与 readiness

- Protocol v1.3 semantic SHA-256：`1525b7cbfaea123b360ffbedd06ef9177b9f8996987d0e49dfd67cafb411ac17`
- Portable registry semantic SHA-256：`810f0fa987202da0e018f309509a286d811bfe867ab62acdf1f476a987c086d7`
- Split semantic SHA-256：`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`
- Catalog fingerprint：`89c548980b63df733553d748e8db3ca622965b63abcd08ebd4c231790b40a9d6`
- Readiness v5 file SHA-256：`bea010290f326604983b32ed1a16a8b06f66e9afa1f60d7aaaaff6e5a68bd985`
- Top-level artifact integrity identity：`4beccd3299905a139f6a0d43895603b0f1bb565129ddc8931abdc962d3ee4bf8`

Readiness v5 的 12 项检查全部为 true，唯一 verdict 为
`READY_FOR_G14C_V4_CLEAN_TRAIN_AND_FORMAL`。该 verdict 不授权本任务自动执行下一阶段；必须在 push 成功后
由独立用户任务启动 G14C v4，且以最终 A4 为唯一 execution commit。
