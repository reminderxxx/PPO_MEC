# 实测校准连续工作流匹配训练（2026-10-06）

## 审查元数据与边界

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-05`（复用项目上一轮最近邻检索，本轮不新增 novelty claim）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `calibrated_continuous_workflow_pilot_20261006_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `0dbb0c1c91e0fac6740bb6e4220a92d014f1fd9d`
- `evidence_level`: `E1_DOCUMENTED_WITH_AUDITED_NONFORMAL_PILOT`；artifact 内部完整性已审计，但缺 formal/holdout/support，
  不满足 policy-level `E2_ARTIFACT_AUDITED`。
- `verdict`: `Unverifiable`（针对 paper-ready / 算法优势）；本轮小规模负向 pilot 本身可报告。

本轮只检验现有 SA-GHMAPPO 在一个小型、共同机制、结果盲冻结的连续工作流问题上是否呈现优势。未修改
SA-GHMAPPO、PPO、MAPPO 的算法实现，未读取 formal/hidden holdout，未在训练 step 调用真实模型，也未根据结果追加
场景、调参或重跑。成本纠错基线使用 `d67575b…` 的对称生命周期版本；旧 artifact、checkpoint 和失败结果不覆盖。

## 共同实验基础

四种方法使用同一个 `CalibratedContinuousWorkflowEnv`、同一五动作 mask、同一依赖安全 typed cache、同一状态恢复、
同一合法 victim/rollback 和同一执行能力。两步规则通过 clone 做两步枚举，但只使用其他方法同样收到的完整 DAG、预测
RSU 序列、cache resident、对象目录、link 和成本参数；不读已实现分支结果。训练和评估均为校准仿真。

| 来源类别 | 本轮使用内容 | 解释边界 |
|---|---|---|
| 实测 | base `1,015,025,832 B`；ALPR adapter `154,423,432 B`；Helmet adapter `9,641,944 B`；本地 load `0.281637/0.711175/0.288584 s`；state `2,195 B`；input `192,757 B`；node `3.807234 s`；prefix recompute `10.789533 s`；restore overhead `0.004412 s` | 同机/本地测量，不是无线或真实 RSU 部署测量 |
| 真实轨迹提取 | NGSIM 冻结窗口的 handoff pressure；Alibaba DAG 的节点、边、duration、plan memory | 轨迹与 workflow 来源真实，但二者配对不是现实请求记录 |
| 文献参数 | 无 | 本轮没有用文献值填补运行参数 |
| 人工假设 | 1 Gbps + 20 ms link、5 s 决策尺度、adapter 映射、第二 base family、状态倍率、deadline、轨迹—workflow 配对 | 明确为 stress/design factor，不得称实测 |

冻结 manifest 含 train/dev/evaluation=`12/4/12` 个实例，Alibaba workflow 为 5–12 节点。八个分层模板覆盖共享/不同
base、紧张/宽裕 cache、低/高 handoff、`1/64/1,000,000×` state、competitor/all-ready/empty 初始 target。
DAG predecessor closure 决定恢复失败后的 prefix recompute；父节点数决定状态包；contact budget 决定 prepare 是否能在
handoff 前完成；模型需求和容量决定 admission、victim 与 bundle reload，因此这些量都进入真实状态转移，不只是 encoder 输入。
矩阵保留 all-ready 简单规则区、大状态恢复不可完成/不划算区、不同 base 无共享收益区。

## 训练与统计协议

- 方法：SA-GHMAPPO、PPO、controller-level MAPPO、确定性 two-step cost rule。
- 学习预算：每个学习方法 3 seeds（7/17/29），每 seed 192 episodes，8 episodes/update，共 24 updates；相同
  learning rate、clip、entropy、value、GAE 与 batch contract。
- 评估：冻结 12 个 dev-plan 窗口，每个已训练 seed 都跑一次确定性 raw policy；规则每窗口一次。
- 统计：先在每个窗口内平均 3 seeds，再以 12 个窗口为外层单元做 5,000 次 percentile bootstrap 95% CI。
- 限制：不是 formal/holdout；未做 BCa、paired sign test 或 Holm；仅 3 seeds、单一 NGSIM+Alibaba 组合；未记录完整
  wall-clock/能耗；没有共享带宽、计算 queue、多租户或跨 episode persistent cache。

## 论文实验表

| 方法 | Workflow completion ↑ | Elapsed-to-terminal/truncation (s) ↓ | Transfer (MB) ↓ | Prefix recompute (s) ↓ | Handoff failure rate ↓ | Return ↑ |
|---|---:|---:|---:|---:|---:|---:|
| SA-GHMAPPO | 0.528 [0.278, 0.750] | 56.202 [40.574, 73.939] | 589.799 [174.324, 1087.161] | 20.001 [9.470, 31.113] | 0.319 [0.153, 0.528] | -10.299 [-20.895, -0.239] |
| PPO | 1.000 [1.000, 1.000] | 97.069 [67.527, 133.150] | 825.278 [240.491, 1526.933] | 60.035 [35.393, 87.809] | 1.000 [1.000, 1.000] | 6.517 [3.633, 9.705] |
| Controller-level MAPPO | 1.000 [1.000, 1.000] | 97.069 [67.301, 131.913] | 825.278 [227.689, 1514.824] | 60.035 [35.295, 87.743] | 1.000 [1.000, 1.000] | 6.517 [3.612, 9.687] |
| **Two-step cost rule** | **1.000 [1.000, 1.000]** | **45.655 [34.329, 60.829]** | **279.151 [57.473, 557.995]** | **3.297 [1.319, 5.455]** | **0.302 [0.094, 0.532]** | **13.218 [11.262, 15.109]** |

数值为窗口均值和 percentile bootstrap 95% CI。`Elapsed-to-terminal/truncation` 对未完成 episode 是截断前累计时间，
因此 SA-GHMAPPO 的较小点估计不能解释为完成更快。完整逐行数据、aggregate 和 checkpoint hash 位于
`artifacts/benchmarks/calibrated_continuous_workflow_pilot_20261006_v1/`。

## 结果解释

结果不支持 SA-GHMAPPO 优势。相对两步规则，SA-GHMAPPO 的 completion 差为 `-0.472`
（95% CI `[-0.722,-0.250]`），return 差为 `-23.516`（`[-34.553,-13.620]`）；不能以截断后的 elapsed 值掩盖完成率下降。
其 36 个 seed-window run 共选择 action 4 `243` 次并产生 `191` 次 service failure，表现为 handoff-prepare 过度激活后
未先恢复当前 RSU bundle。PPO 的确定性评估全部选择 action 0；MAPPO 选择 action 0/4=`230/16`，二者均完成，但每次
handoff 都走 prefix recompute，未兑现状态迁移机制。

强两步规则选择 action 0/2/4=`55/15/12`，全部 12 个窗口完成，并在完成量相同时显著减少 modeled elapsed、传输和
重算。PPO/MAPPO 相对规则的 elapsed 差均为 `+51.414 s`，transfer 差 `+546.127 MB`，recompute 差
`+56.738 s`；对应 CI 均不跨 0。该结果只说明这个冻结 pilot 中显式两步成本规则更稳，不证明它在真实无线、未知动态或
更大规模上占优。

## Claim 边界

Safe claims:

- 已把 A 线的真实对象字节、同机 load/state/recompute 时间和合法 cache/state transition 转为所有方法共享的仿真基础。
- 冻结小型结果盲分层 workload 上，强两步规则优于本轮三种已训练策略；SA-GHMAPPO 未通过优势检验。
- DAG、contact、model identity、cache capacity 和 state volume 均改变执行状态，而非装饰性特征。

Prohibited claims:

- 不得称 SA-GHMAPPO 优于 PPO、MAPPO 或强成本规则；不得把本轮写进支持算法优势的主表。
- 不得称真实 RSU、真实无线、生产部署、formal、holdout、canonical、TMC-ready 或统计泛化结果。
- 不得把 controller-level MAPPO 写成 vehicle/RSU-level full MARL，也不得把 NGSIM+Alibaba 配对写成真实 adapter request trace。

Required actions before re-review:

- 先独立分析 SA-GHMAPPO action-4 collapse；任何 reward/guard/预算调整都必须作为新实验任务和新 run ID，不能覆盖本轮。
- 若继续论文算法 claim，需预先冻结修复/训练方案，再完成独立 formal/holdout/support、BCa/paired/Holm、robustness、
  scalability、ablation 和 compute accounting；本轮结果必须保留为负向 pilot。
