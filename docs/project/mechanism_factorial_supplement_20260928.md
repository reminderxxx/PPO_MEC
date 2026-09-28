# Base Sharing × Workflow-State Migration 受控补充

## 审查元数据

| 字段 | 值 |
|---|---|
| `reviewed_at` | `2026-09-28T08:00:17+08:00` |
| `literature_cutoff` | `2026-09-28` |
| `target_venue` | `IEEE Transactions on Mobile Computing (TMC)`；实际投稿目录未指定 |
| `artifact_run_id` | `mechanism_factorial_pilot_v1_20260928_v2`；后台训练为 `mechanism_algorithm_retraining_v1_20260928` |
| `policy_version` | `tmc_review_policy_v3_20260621` |
| `implementation_git_commit` | `97e24a6d6d2541a0a5fde54cdbc69a789d4dfde0` |
| `evidence_level` | `E1_CONTROLLED_OBSERVED_DATA_PILOT_NOT_HOLDOUT` |

## 结论

本轮建立了真实 NGSIM mobility、真实 Alibaba DAG 结构上的四臂受控补充。工作流节点使用冻结的
`semantic_ai_service` 映射，使同一 episode 实际请求 3–4 个 adapter；共享臂使用一个 180 MB base，非共享臂为
每个 adapter 建立同尺寸、同 family 的物理 base replica。三台 RSU 的初始逻辑服务均为
`base + adapter_perception = 220 MB`，容量均为 360 MB。迁移关闭只压制 `prepare/migrate`，其余 cache/offload
动作不变，并以一请求 cold restart 作为冻结 fallback。

修正版小表显示两条机制均真正进入执行路径：base sharing 保持 readiness/continuity 不变，同时把平均
`transfer_mb_per_request` 从 `67.972223` 降至 `19.083334`，主效应为 `-48.888889 MB/request`；migration 使
ready/continuity 从 `0.902778` 升至 `0.944444`，主效应为 `+0.041666`，同时增加 `0.833333 MB/request` 和
每 episode 平均 `10 MB` workflow-state transfer。8 个 episode 的 completion 全为 0，因此机制的端到端完成收益
仍为 `UNVERIFIED`，当前证据只能支持“机制可激活且存在 readiness/cost trade-off”。

## 四臂小表

| base sharing | state migration | episodes | completion | ready | continuity | MB/request | backhaul | evictions/episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| on | on | 2 | 0.000000 | 0.944444 | 0.944444 | 19.500000 | 192.0 | 1.0 |
| off | on | 2 | 0.000000 | 0.944444 | 0.944444 | 68.388889 | 672.0 | 6.0 |
| on | off | 2 | 0.000000 | 0.902778 | 0.902778 | 18.666667 | 182.0 | 1.0 |
| off | off | 2 | 0.000000 | 0.902778 | 0.902778 | 67.555556 | 662.0 | 6.0 |

原始 episode、请求级事件和小表位于
`artifacts/analysis/mechanism_factorial_pilot_v1_20260928_v2/`。首轮目录
`mechanism_factorial_pilot_v1_20260928/` 已由 `invalidity_receipt.json` 标记为工程无效：非共享 base 从共享
catalog 继承了 `pinned`，导致 exclusive adapter/base 无法成组淘汰并人为产生容量拒绝。修复后只在显式
`base_sharing_enabled=false` 的新 profile 中把 exclusive pair 当作原子 eviction unit；历史路径不变。修正版
事件审计确认每次切换按 `adapter + exclusive base` 成组提交，容量拒绝为 0。

## 冻结训练筛查

由于机制产生实质资源效应，允许启动一个受限、非正式、重新训练的算法筛查；旧 checkpoint 不可复用。冻结合同：

- algorithms：`sa_ghmappo`、`mappo`；均为 controller-level，不扩写为 vehicle/RSU-level MARL；
- mechanism arm：仅 `sharing_on_migration_on`；360 MB、LRU、相同初始 220 MB residency；
- workload：预先冻结的 3 个互不重叠 NGSIM mechanism-activating windows、4 个 Alibaba DAG、
  `semantic_ai_service`；窗口在 learned outcome 前按 mobility covariates 选定；
- budget：seed `1401`，每算法 64 episodes、每 episode 最多 12 steps、每 4 episodes 更新、batch 32；总上限
  128 episodes / 1,536 environment steps；
- request exposure：训练前冻结、policy-neutral replay；reward positive offset 为 0；
- selection：只报告 fixed budget 的 `latest.pt` 与完整训练轨迹，不按训练结果换 seed、窗口、预算或 checkpoint；
- status：后台启动状态、固定 argv、stdout/stderr、PID 与 exit receipt 以
  `artifacts/training/mechanism_algorithm_retraining_v1_20260928/` 为唯一权威。

该筛查只有一个 seed，没有独立 test split，也没有 algorithm evaluation；即使成功完成，也不能形成算法优越性或
论文主结果。后续只有在新增的、与本轮开发数据独立的 window/run 可证明可用后，才能冻结多 seed 训练和 matched
evaluation。旧 G14R22D holdout 不得重开。

## 最近邻边界

- TMC 2025 的 dual-dependency service caching/offloading 已覆盖任务/服务依赖、层级 active/passive cache 和 PPO；
  本项目不能把 dependency-aware cache 或 PPO 本身写成创新。
- TVT 2026 的 MATM 已覆盖 mobility prediction、RSU 间 task migration、资源分配和 SAC；本项目只能检验连续
  DAG frontier、adapter/base residency 与 workflow state 的联合代价。
- Punica/S-LoRA 已证明多 adapter 共享 base 与 adapter paging 是真实 serving 对象；当前 180/40–100 MB 是受控
  catalog 值，不是真实 GPU paging latency。
- ICDCS 2026 EdgeFlow 已把 geo-distributed edge LLM 的 KV state migration 做成 transmission/recomputation
  trade-off；PPO_MEC 当前 workflow state 仅是 20 MB 受控 payload，不能等同 KV cache 系统实现。
- IFIP Networking 2026 的 proactive service migration 显示错误预测会显著恶化 RTT；本项目 supervised predictor
  仍未在本补充中启用，不能把 prediction gain 合并进机制结果。

## Claim boundary

允许表述：实现了可执行的 base-sharing × workflow-state-migration 2×2 受控合同；在一个 observed-data
mechanism-activating window 中观察到显著量级的传输节省和小幅 readiness trade-off。

禁止表述：算法优越、端到端 completion 改善、独立泛化、正式 holdout、真实 LoRA paging 性能、统计显著或
paper-ready。单窗口/两 workflow 不是独立 cluster 统计，未运行 Holm 检验。
