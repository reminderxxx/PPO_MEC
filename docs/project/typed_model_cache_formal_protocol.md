# Typed Model-Cache Formal Protocol

## G14R2 当前审查身份

- `reviewed_at`: `2026-08-20T18:23:21.124778+08:00`
- `literature_cutoff`: `2026-08-20`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_formal_window_repair_20260820_g14r2_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 实现基线 Git commit：`89049c92b41054d78294893643f241926181645a`
- execution commit：Commit A3（包含本协议 exact semantic hash 的提交）
- evidence level：`E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE`

Protocol v1.2 supersede v1.1。v1.1/G14C v2 因 train command 未传显式 NGSIM source range、回落到
1500 raw rows 而在首个 training cell 前发生 `data_window_unreachable`，永久状态为
`invalid_before_performance_execution / INVALID_PROTOCOL_OR_IMPLEMENTATION`。旧 run 记录 0/150
training、0 checkpoint、0 formal、holdout unopened，禁止 resume。

v1.2 semantic SHA-256 为
`718c0f78aabd5d01012df31267626eab74a51b2b621aaa67a535c5b60e655ca9`；split semantic SHA-256 仍为
`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`。本次只修复
window/source/command/ledger execution contract，没有按结果改 split、agent、seed、预算、capacity、
endpoint、support、统计、claim、checkpoint cadence 或 SA coefficient。

机器事实源：

- `configs/experiment/typed_model_cache_formal_protocol_v1_2_20260820/protocol_v1_2_manifest.json`
- `configs/experiment/typed_model_cache_formal_protocol_v1_2_20260820/formal_window_consumption_contract.json`
- `artifacts/analysis/typed_model_cache_formal_window_repair_20260820_g14r2_v1/`

Readiness v4=`READY_FOR_G14C_V3_CLEAN_TRAIN_AND_FORMAL`，只表示未来可从 Commit A3 clean worktree
另立 G14C v3；正式 checkpoint/episode/performance 仍为 0，holdout sealed/unopened，不是 formal、G14
或 paper-ready 完成。窗口消费细节见 `typed_model_cache_formal_window_consumption_contract.md`。

## G14R/G14B 历史审查身份

- `reviewed_at`: `2026-08-20T16:20:00+08:00`
- `literature_cutoff`: `2026-08-20`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_formal_protocol_restart_20260820_g14r_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 实现基线 Git commit：`351fdb8a309614a751cedb180ecaccf2a681db2d`
- execution commit：Commit A2（本协议 exact semantic hash 的提交）
- evidence level：`E2_EXECUTION_CONTRACT_VALIDATED_NO_FORMAL_PERFORMANCE`

## 状态与边界

G14R 已用 `typed_model_cache_formal_protocol_version=1.1.0` supersede v1.0。v1.0 与其 G14C v1
run 永久为 `invalid_before_execution` / `INVALID_PROTOCOL_OR_IMPLEMENTATION`；Phase-0 后没有运行测试、
训练、checkpoint selection 或 formal，因此没有观察正式性能。v1.1 semantic SHA-256 为
`b8bbb53d6af47d111b840efbb53d3389485535d66c8de19b747e2a5727786629`，Readiness v3 为
`READY_FOR_G14C_V2_CLEAN_TRAIN_AND_FORMAL`。机器入口位于
`configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/`，修复审计见
`docs/project/typed_model_cache_formal_protocol_restart.md`。

以下 G14B v1 内容保留为历史冻结背景。G14B 当时的 readiness verdict 为
`READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL`。它只表示 split、runtime、agent、预算、统计和
holdout 执行合同已通过运行前检查，不表示 formal 已完成、holdout 已打开、已有正式 checkpoint、
存在性能结果或达到 paper-ready。

本协议的数据边界是 NGSIM mobility、Alibaba 2018 batch DAG 与 repository-controlled typed
catalog 的跨源受控组合；它不是真实联合 model-cache request trace。HF metadata、KV/prefix cache
和 G12 supervised predictor 均不进入本协议。

机器事实源位于：

- `artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/formal_protocol_manifest.json`
- `artifacts/analysis/typed_model_cache_formal_protocol_freeze_20260820_g14b_v1/protocol_hashes.json`
- `configs/experiment/typed_model_cache_formal_protocol_v1_20260820/protocol_index.json`

G14R v1.1 机器事实源：

- `configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/protocol_v1_1_manifest.json`
- `configs/experiment/typed_model_cache_formal_protocol_v1_1_20260820/protocol_index.json`
- `artifacts/analysis/typed_model_cache_formal_protocol_restart_20260820_g14r_v1/`

## v1.1 executable repair

- 共享训练入口机械消费 protocol、agent config 与 `checkpoint_every_updates=4`；`latest.pt` 只供
  resume，只有 updates `[4,8,...,32]` 进入 dev candidate set。
- SA-GHMAPPO 经共享 registry 实例化并审计 `auxiliary_coef=0.06`；其他 agent 不接收该字段。
- metrics 1.2 从 raw CacheEvent 1.3 重算两个新增 primary endpoint，并强制 summary/row 对账。
- support/scalability 每个 level 有 stable ID、数值、单位、baseline、seed、资源与 artifact 身份；
  现有机制不能安全实例化的 level 显式 `unavailable_pre_execution`，不能静默回退。
- command templates 展开 150 个 train cells，并覆盖 dev selection、checkpoint freeze、cache-policy、
  controller、ablation、robustness/prediction、scalability、statistics、integrity 与 gate。
- phase runner固定 13 阶段 append-only ledger；只有 input/output hash 完全一致的 completed phase
  可 skip，失败 terminal，exit 75 最多原命令重试一次，formal 开始后禁止训练，普通 runner 不具备
  holdout token 或执行接口。

Split semantic SHA-256 仍为
`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`：没有 formal 结果、没有
holdout opening、没有按结果改窗口，因此只新增 companion metadata，不重写 60 个窗口语义。

## Identity 与不可变性

- protocol ID：`typed_model_cache_formal_protocol_v1`
- protocol semantic SHA-256：`41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4`
- split semantic SHA-256：`aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`
- historical registry semantic SHA-256：`09ee3109f6789cada996870bcb6a3dac9496a4d98f8373c11cb99f61f55beaae`
- catalog fingerprint：`89c548980b63df733553d748e8db3ca622965b63abcd08ebd4c231790b40a9d6`
- NGSIM SHA-256：`ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`
- Alibaba SHA-256：`6346b0726c6e10466a585c67645af807b425b5be091caf410f5e1aff41a270bc`
- typed catalog file SHA-256：`0be81a904f09ca0e8976926caf064f095f6d9fb4173cfa1f0a8eed7042f62b2c`

Canonical JSON 使用 UTF-8、compact sorted keys，拒绝 NaN/Infinity。`created_at`、输出路径和
审查时间等非语义字段从 semantic hash 排除；split、agent、seed、预算、capacity、endpoint、
statistics 或 claim 的改变都会改变 protocol semantic hash。CLI 不能覆盖语义字段。若需修改，
必须升级 version、run ID 和 Git commit，旧协议不得覆盖。

后续执行 commit 绑定规则为：Commit A 必须包含与上述 semantic hash 完全一致的冻结协议，且
G14C 从该 commit 的 clean worktree 启动。

## Agent matrix

Controller table 冻结为：

- learned paper-grade：`sa_ghmappo`、`ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、
  `controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`
- matched heuristic：`popularity_cache_heuristic`
- cache-policy isolation：`reactive_lru`、`reactive_fifo`、`reactive_lfu`、
  `reactive_aging_lfu`、`reactive_random`
- exact-only cells：`exact_oracle_h1`、`exact_oracle_h3`、`exact_oracle_h6`、`exact_oracle_h12`

所有 learned controller 使用 clean typed checkpoint per seed/capacity 和相同最大环境交互预算；
heuristic/reactive 不伪装 learned checkpoint；oracle 只允许 exact 状态。MAPPO、QMIX 和
SA-GHMAPPO 仍是 controller-level contract，不声明 vehicle/RSU-level MARL。

## Seeds 与训练预算

- seeds：`[7, 13, 29, 43, 71]`
- 每个 learned agent/seed/capacity：256 episodes
- update interval：8 episodes；expected updates：32
- batch size：64；max steps：22
- train outer windows：24；workflows：3（`j_3`、`j_8`、`j_15`）
- 最大环境交互：5,632 / learned agent / seed / capacity
- optimizer：算法原生 Adam；device：CPU
- checkpoint frequency：每 4 updates
- early stop：禁用；固定交互预算
- infrastructure retry：同命令一次；不得 reseed 或事后替换
- 资源上限：12 wall-clock hours / learned agent / seed / capacity，合计 2,500 CPU-hours

学习率、entropy/value/auxiliary coefficient 和算法固有的 on-policy、replay、CTDE/value-decomposition
差异均在 manifest 中逐项冻结。Checkpoint 只在 dev 依据预注册 endpoint 顺序选一次；formal 或
holdout 不能参与选择。

## Typed catalog 与 capacity

容量单位为 repository contract 的 decimal MB。依赖 bundle 为 atomic base+adapter；workflow state
不计入长期 capacity；KV disabled；oversized bundle 必须 `rolled_back_no_mutation`，不能部分接纳。

| stratum | capacity | 预注册推导 |
| --- | ---: | --- |
| constrained | 288 MB | `ceil(max(initial resident, largest atomic bundle)/32)*32` |
| medium | 576 MB | `ceil(mean(constrained, relaxed)/32)*32` |
| relaxed | 864 MB | `ceil(total capacity-counting catalog resident MB/32)*32` |

输入量为 max initial RSU resident 276 MB、largest atomic dependency bundle 280 MB、catalog total
856 MB。容量不得按正式结果调整。

## Workload 与 endpoints

正式 workload 冻结 NGSIM、Alibaba DAG、24-frame window、22 steps、
`auto_dominant_tight` RSU layout，以及同 request replay/fingerprint 的跨 agent 核对。

Primary endpoints：

1. `full_service_ready_byte_hit_rate`
2. `joint_base_adapter_hit_rate`
3. `full_service_ready_request_rate`
4. `transfer_mb_per_request`
5. `workflow_continuity_rate`
6. `end_to_end_workflow_delay`（现有抽象单位）

Secondary endpoints 包含 base/adapter hit、base sharing、avoided duplicate transfer、pollution、
churn、future-reuse proxy、occupancy/saturation、rejection、backhaul、handoff failure、migration、
reward、wall-clock、memory、oracle state/solve time；`latency saved` 保持 unavailable。

Primary comparisons 为 SA-GHMAPPO 对 dev 一次性选出的 strongest matched non-oracle baseline、typed
full 对 legacy adapter-only、best compatible learned 对 best reactive，以及 reactive 对 exact oracle
的 exact cell。Strongest baseline 不得由 formal/holdout 选择。

## Ablation、support 与统计

Ablation/support 预注册 typed full、legacy adapter-only、no base sharing、no workflow-state migration、
fixed/no eviction、no prediction、三档 capacity、object-size/transfer-cost/handoff-pressure/reuse/
base-sharing sensitivity、RSU/vehicle/DAG/object scalability、predictor boundary 和 oracle state limit。

统计外层单位为 raw-time mobility window；seed/workflow 是内层重复，不增加 outer count。冻结
10,000 次 hierarchical bootstrap（seed 1401）、percentile 95% 与 BCa 95% CI、exact paired sign
test、effect size、win/tie/loss、`1e-9` tie tolerance 和 missing/failed-run policy。Primary Holm family
为全部预注册 primary comparisons × 六个 primary endpoints；secondary 使用独立 exploratory Holm
family，不使用 confirmatory 表述。Mixed/full 同一时间区间只作为一个 outer cluster。

## Claim 模板

训练前已预注册 typed base sharing、byte hit、transfer overhead、workflow continuity、capacity
pressure、eviction policy、oracle opportunity、controller comparison、predictor boundary 和 data
realism boundary。每项结果只能取 `supported`、`mixed`、`unsupported`、`contradicted` 或
`unavailable`。G14B 不为任何 claim 填入结果状态。

## Holdout seal 与 readiness

Sealed holdout 当前 `sealed=true`、`opened=false`、`consumed_permanently=false`，一次性 token 状态为
`not_issued_in_G14B`。执行前只允许 validator 读取 identity/interval；opening gate 只能检查训练与
eval 完整性、protocol/checkpoint provenance、fairness、artifact integrity 和基础设施健康，禁止
使用 formal performance gate。成功开启最多一次，执行记录 append-only，并绑定正式 checkpoint
provenance hash、执行 commit、命令、时间和输出 run ID。

Readiness v2 为 `READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL`，但 `formal_completed=false`、
`paper_ready=false`、typed checkpoint count=0、formal episode count=0、holdout unopened。G14C 仍须
独立任务从 Commit A clean worktree 开始；本文件不授权自动执行 G14C、formal、holdout、hidden 或 G15。
