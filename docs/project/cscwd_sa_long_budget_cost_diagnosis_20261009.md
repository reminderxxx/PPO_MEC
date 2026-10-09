# CSCWD 长预算后 SA 逾期与成本：独立只读定位

## 审查身份和原件状态

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09；本轮未作新文献检索或 novelty 评价
- `target_venue`: CSCWD 2027（拟投）；本轮不写论文、主稿或论文表
- `artifact_run_id`: `cscwd_causal_strong_baselines_budget_extension_20261009_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scientific_git_commit`: `d25ebcded6b43b69b82bae825b035adc1d6f19c4`
- `causal_interface_commit`: `a08869388f9962149198054b5f8a0d6dd4d07eae`
- `evidence_level`: `E0_UNAVAILABLE`（paper-ready 审查）；开发原件已逐步重放核验，但缺独立 formal/holdout/support，不把本诊断冒充规范中的 `E2_ARTIFACT_AUDITED`
- `verdict`: `RECOMPUTE_DOMINATED_COST_GAP / SHARED_PUBLIC_STATE_ALIAS_CONFIRMED / SA_ADVANTAGE_UNVERIFIED`

B 科学完成回执为 `complete`（115,200 环境步、15,360 optimizer steps、400 learned 评估行、2,811 行行为记录），但 supervisor terminal 为 FAIL：科学结束后的 Python `zip(strict=True)` 后处理非零退出。B 后续独立修复后处理并保留科学原件；本分支没有写入 B 工作树，也没有把 terminal FAIL 改称 PASS。本诊断校验了所读取的科学文件和 20 个 selected checkpoint 的 hash；B 与原因果实现的 environment、agent、encoder、base config/manifest 身份一致。记录动作重放 400 episode / 2,811 步，逐步服务、进展、迁移、传输、重算及 episode 总时延与原件一致；九项互不重叠的成本分量逐步加回 `step_cost_seconds`。每个诊断目录另有 hash 清单：`artifacts/analysis/cscwd_sa_long_budget_cost_diagnosis_20261009_v1/` 的 10 个文件与 `cscwd_sa_long_budget_state_alias_20261009_v2/` 的 2 个文件均通过复核。别名审计 v1 的“SA 内部组”过滤误把混合方法组排除，保留原件但以修正后的 v2 为准；没有重复科学训练。

## 服务覆盖与逐 seed 区域

下表每格为 frozen development 的 8 个已消费实例，顺序是按期/逾期完成/未完成；完整机器表还包括 regression、每 seed、全部三个区域和每项成本。

| 方法 | seed 7 | 17 | 29 | 43 | 61 | 合计完成、按期 |
| --- | --- | --- | --- | --- | --- | --- |
| SA-GHMAPPO | 0/8/0 | 2/6/0 | 1/7/0 | 2/6/0 | 0/7/1 | 39/40、5/40 |
| PPO | 1/7/0 | 0/8/0 | 1/7/0 | 1/7/0 | 1/7/0 | 40/40、4/40 |
| DT-Handoff-PPO | 2/6/0 | 3/5/0 | 1/7/0 | 3/5/0 | 2/6/0 | 40/40、11/40 |

SA 已从短预算的 34/40 frozen 完成恢复到 39/40，但没有同时兑现按期和成本优势。已完成样本的平均 modeled elapsed 为 SA `115.87 s`（39/40 coverage）、PPO `87.25 s`（40/40）、DT `92.87 s`（40/40）。若把 SA 唯一未完成的短 episode 也算入，所有 40 实例平均为 `113.67 s`；不能用这个较低数字掩盖完成覆盖。SA 的 5 个按期样本平均耗时 `42.2 s`、DAG 重算 `10.8 s`、模型网络传输 `3.9 s`；34 个逾期完成样本分别为 `126.7/73.6/16.8 s`。各实例 deadline 不同，此分层只是机制描述，不是“重算独自导致逾期”的反事实证明。

## 完整成本分解与配对覆盖

frozen 全部 40 实例每 episode 平均秒数（SA 包含 1 个未完成；完整逐 seed/region 见机器表）：

| 成本分量 | SA | PPO | DT |
| --- | ---: | ---: | ---: |
| 节点 compute | 28.22 | 28.88 | 28.88 |
| vehicle fallback | 4.20 | 27.00 | 4.80 |
| 失败等待 | 1.50 | 0 | 0.05 |
| 模型本地 load + 网络传输 | 15.71 | 4.41 | 17.89 |
| 状态 restore + 网络传输 | 0.04 | 0 | 0.06 |
| 输入网络传输 | 0.08 | 0.13 | 0.06 |
| DAG 重算 | **63.93** | 26.84 | 41.14 |
| 合计 modeled elapsed | **113.67** | 87.25 | 92.87 |

与同 split/实例/seed 配对时，frozen 的 SA−PPO **双方共同完成 39/40**：平均耗时 `+28.39 s`，其中重算 `+39.15 s`、模型网络 `+10.99 s`、模型 load `+0.63 s`、失败等待 `+0.82 s`，同时少用 vehicle fallback `−23.18 s`。SA−DT 共同完成同为 `39/40`：耗时 `+22.63 s`，重算 `+24.48 s`，模型网络 `−1.85 s`。全部实例和共同完成子集的模型/状态/输入字节、重算及 seed 明细均保留；SA 相对 PPO 的模型传输在共同完成子集多 `500.56 MB/实例`，相对 DT 少 `177.89 MB/实例`。较低字节不能抵消 SA 的完成/按期覆盖及重算时间。

## 动作、状态迁移与首次分叉

SA frozen 共 121 次非 fallback handoff：24 次已有可用准备状态，97 次未就绪，其中 85 次实际付出正 DAG 重算；91 次未就绪此前没有到该 RSU 的成功 state prepare，6 次曾准备但之后节点进展使状态过期。DT 对应 `97/36/61`（handoff/ready/未就绪且正重算）；PPO 的 45 次非 fallback handoff 全部没有准备状态并付重算，但它在 frozen 另有 135 次 vehicle fallback。SA 用较少 fallback 换来更多需重算的 RSU 执行，是本 run 的主要成本结构差异。

SA frozen 有 92 次 action 4、5 次目标 resident 改变、37 次后续已准备目标复用、8 次 victim reload；模型再次传输到同一目标/adapter 的计数为 0，已记录目标与后验首个 handoff 目标不匹配为 0。因而“重复模型准备”“目标预测错位”不是已观察到的主导成本项，不能粗暴封禁 action 4。85 次 SA 正重算 handoff 中，79 次立即前一步预测目标与实际到达 RSU 一致；其中 22 次前一步当前 bundle ready、目标 preview admission 已提交，实际选择 action 3/0/2 分别为 `15/5/2`。这提示可见准备机会未被使用，但 action 4 的即时代价和后续缓存影响尚未在这些状态作公平局部分支，不能把 22 次直接写成可避免的净损失。

SA 与 PPO 在 frozen 40 对中，38 对的首次动作分叉发生于完全相同的公开状态，2 对无动作分叉；与 DT 的 40 对全部在相同公开状态首分叉。此后 cache/prepared-state 历史不同，只可比较各自轨迹代价，不能把后续不同状态下的动作差异称同状态单动作因果。SA frozen 仍有当前 bundle 缺失时 action 3 共 14 步、action 4 共 16 步，集中在 seed 61 的未完成/逾期区域；它们解释部分服务失败，却不能解释其余 39 个完成 episode 的主要重算差距。SA 的 vehicle fallback 只有 21 步，PPO 135、DT 24，也排除了“SA 因频繁 fallback 而慢”的主导解释。

## 公开状态可辨识性：已确认的接口缺口

同一 `observation + semantic_state + mask` hash、同一合法 action 的 handoff 状态中，10 个跨方法组出现 `state_ready` 不同；其中 1 组在 **SA 自身不同 seed** 内也成立。具体在 `regression_08` 第 4 步，SA seeds 17/29/43/61 的公开状态 hash 同为 `86206c…17d62d`，均执行 action 0；17/29 的 state ready、重算 `0 s`，43/61 的 state not ready、重算 `30.2107 s`。前序 action 4 的状态准备历史不同，但 `prepared_state` 有效性没有进入公共 observation/semantic state。这个成对反例证明现有公开状态**不足以唯一判定该执行成本**，所有 feedforward learned 方法都共享此接口限制；它不证明这个缺口单独造成 SA 与 DT 的全部差距。别名证据的来源只在重放后核验，未把真实未来轨迹输入任何 policy。

## 选模边界与唯一后续候选

B 的 checkpoint 选择规则本来把 dev 按期完成排第一、总完成排第二；SA 五个 selected checkpoint 的 dev 按期均 `3/4`，selected updates 为 `48/72/48/48/24`，而 frozen 每 seed 仅 `0/2/1/2/0` 按期。4 个 dev 实例不足以证明选模目标错误；不同 seed 的候选曲线和 frozen 外推差距同时存在，**选模/分布泛化责任未识别**。本轮没有改选 checkpoint，也没有额外评估候选。由于现有 400 episode 的代价账本与同公开状态别名已经给出可执行定位，可选共同状态前向为 `0/1200`，局部分支为 `0/12`；避免新增能力不对称的 oracle 性能矩阵。

**唯一最小候选**：在下一独立实现任务中，把当前 RSU 与公开预测目标 RSU 的 *prepared-state prefix validity/freshness* 作为明确公共字段，并让四种 learned 方法同权访问。该信息来自已经执行的 state migration 和已完成 DAG 前缀，不读取未来 RSU 后缀；属于**共享 observation contract 纠错/已有状态管理技术**，本轮不称算法创新，相关文献水平仍待核验。它针对已证明的同公开状态不同重算成本；不能仅加字段而不检查四种 encoder 是否真正使用。若需改网络宽度，必须另列参数量/推理开销，不能宣称纯输入单变量。

公平 A/B 草案：先冻结字段来源、合法可见时点、prefix-only 与状态别名单测；A 精确复用 B 的原长预算原件，B 新 profile 对四个 learned 方法共同按 `5 seed×5,760` 环境步、相同 4 次 dev 选模机会训练一次，保留全部已消费开发实例、所有失败与正负区域。主要验收为共同完成覆盖、按期、handoff ready/重算与模型/状态/输入字节联合改善；若别名未消除、SA 重算/逾期未改善，或任一方法服务/成本边界恶化，则拒绝。确认性主张仍需新原始区间/车辆独立数据。本轮没有实现候选、训练、调参、改 reward/network/guard/environment 或修改论文。

本地完整原件见上述两个 create-only 目录；Git 仅保存脚本、事前计划、[小型机器摘要](cscwd_sa_long_budget_cost_summary_20261009.json)与本报告。所有 36 个源实例已消费为开发，paper-ready=`Unverifiable`。
