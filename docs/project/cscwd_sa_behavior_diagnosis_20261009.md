# CSCWD 因果强基线：SA 行为差异独立只读诊断

## 审查身份和执行边界

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09；本轮未检索文献，不评价 novelty
- `target_venue`: CSCWD 2027（拟投）；本轮不生成论文表或修改文稿
- `artifact_run_id`: `cscwd_causal_strong_baselines_dev_20261009_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scientific_git_commit`: `a08869388f9962149198054b5f8a0d6dd4d07eae`
- `diagnosis_plan_commit`: `5224756eb4a6746338ef12c4e9068126120c4e4f`
- `diagnosis_script_commit`: `4e33717dc13da61dae0ae97323120c7865365ae7`
- `evidence_level`: `E2_DEVELOPMENT_ARTIFACT_REPLAYED`；仅开发证据
- `verdict`: `PROXIMATE_POLICY_DECISION_FAILURE_WITH_SEED_SENSITIVITY / NO_SA_ADVANTAGE / PAPER_READY_UNVERIFIABLE`

源科学 run 的 terminal 为 PASS、28,800 训练环境步、3,840 optimizer steps、440 评估行、3,308 行行为账本；诊断先核对了输入 hash 与 20 个 selected checkpoint。按原动作重放 440 个 episode / 3,308 步，服务、进展、迁移、传输、重算和原评价完成数逐项一致。共同状态由 train/dev 两条事前规则采集 60 个；20 个 checkpoint 各前向一次，共 1,200 次，所有网络、optimizer、归一化状态前后逐 tensor 不变。`0/12` synthetic 局部反例被使用；训练、正式评估、旧 holdout 读取均为 0。诊断本地完整账本在 `artifacts/analysis/cscwd_sa_behavior_diagnosis_20261009_v1/`，`machine_summary.json` SHA-256 为 `fa4dadeb3171bbf2a1fef59d124364dcc7f3ec715a103b6b5e035861926fc429`，11 个诊断文件通过 hash 复核。Git 只保存[小型机器摘要](cscwd_sa_behavior_diagnosis_summary_20261009.json)、计划、脚本和本报告，不上传 checkpoint 或原始数据。

## 完整 seed 与方法结果

以下是每 seed 20 个已消费开发实例的原始完成数；不是 5×20 个独立来源样本。

| 方法 | seed 7 | 17 | 29 | 43 | 61 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SA-GHMAPPO | 20 | 15 | 18 | 15 | 19 | 87/100 |
| PPO | 15 | 20 | 20 | 20 | 20 | 95/100 |
| controller-MAPPO | 20 | 20 | 20 | 20 | 20 | 100/100 |
| DT-Handoff-PPO | 20 | 20 | 20 | 20 | 20 | 100/100 |

SA 在 regression 为 `53/60` 完成、`23/60` 按期；frozen development 为 `34/40` 完成、`5/40` 按期。DT 分别 `60/60`、`29/60` 与 `40/40`、`12/40`。因此本 run 不支持 SA 服务优势，尤其不能以单个 seed 7 的 20/20 覆盖 seed 17/29/43/61 的差异。SA selected update 为 `24/6/12/18/24`；既有 4 次 dev 选模和训练随机性都可能影响差异，但本轮没有评价未选 checkpoint，不能把 seed 敏感定因于选模。

## 首次分歧、失败链与缓存副作用

SA 的 13 个未完成 episode 在首次服务失败时均缺当前 bundle：8 次执行 action 3，5 次 action 4；失败 episode 共 136 个服务失败/无进展步骤，最长连续无进展达 14。当前 bundle 缺失时，action 3 不准备当前 bundle；action 4 可准备目标缓存但同一步不补当前服务，因此两种动作均可能造成当前节点失败。13 个失败 episode 的动作投影次数为 0。与 PPO/MAPPO 各 100 组的首个动作分歧全部发生在相同公开状态；与 DT 的 100 组中有 94 组首分歧、6 组直到结束无分歧。SA 失败的 13 组对每个比较方法都在相同公开状态先出现动作分歧；这定位到已执行的策略动作，而不把后续不同状态分布误认作同状态因果效应。

60 个事前选取的共同 train/dev 状态中，25 个当前 bundle 缺失。20 个 checkpoint 对这些状态各看一次：SA 选择当前步不服务的 action 1/3/4 为 `51/125`（按 seed `4/21/10/11/5`）；PPO `21/125`、MAPPO `12/125`、DT `10/125`。SA seed 17 的 `21/25` 多为 action 4，seed 29/43 主要为 action 3。共同状态公开了 current readiness，证明观察合同具备区分信息；不能据此推断 SA 网络内部是否充分使用它。共同状态前向中 SA raw action、投影 action 与最终 action 被分列，`15/300` 出现投影；原失败 episode 中为 0，故投影不是本批失败的近端解释。策略选择与 seed 敏感是本轮可执行定位，architecture 或 selection 的单一因果责任仍未知。

**不能封禁所有“失败且无 migration”的 action 4。**SA 全部轨迹中有 89 次 action 4 当前服务失败，其中 11 次仍改变目标缓存 resident；13 个未完成 episode 中也有 2 次此类改变，并在这些 episode 后续观察到 13 次已准备目标复用、4 次 victim reload（这些复用不能全部归因于前述 2 次失败动作）。重放保留每次 admission、victim、后续重载和目标 resident before/after。action 4 的准备可能具有延迟价值，单看当步 `migration_success=false` 或 action 4 总数无法判定收益。

## 成本、完成覆盖与预测口径

全部 100 对实例中，SA 与 PPO/MAPPO/DT 的完成覆盖分别为 `87/95`、`87/100`、`87/100`；双方共同完成子集分别只有 `82/100`、`87/100`、`87/100`。在共同完成子集，SA−PPO 的条件耗时 `-1.65 s`、传输 `+211.35 MB`、重算 `+0.34 s`；SA−MAPPO 为 `-11.85 s`、`-352.92 MB`、`-14.26 s`；SA−DT 为 `+5.64 s`、`-362.95 MB`、`+9.62 s`。这些条件均值不能抵消 SA 较低的完成覆盖，也不能把未完成造成的低传输当作 Pareto 收益。原件包含每 seed 全实例与共同完成配对行；没有将 seed/window 行当独立 cluster 做显著性推断。

预测审计覆盖 3,308 个决策，其中 21 个下一步真实位置截尾；其余 3,287 个中 known 为 `3,155`（`95.98%`），unknown `132`。known 的逐步下一 RSU 正确 `2,874/3,155`（`91.09%`）；其中 stay 正确 `1,625/1,906`（`85.26%`），当步真实 handoff 正确 `1,249/1,249`。在有可观察 horizon 内实际 handoff、且模型预测 handoff 的 `3,091` 个重叠决策点，目标 ID 正确 `3,091/3,091`，首个 handoff 时间仅 `2,645/3,091`（`85.57%`）；另有 105 个 horizon 截尾点。后两项的决策点大量来自同一轨迹/seed，且 RSU 路由按人工模板生成；目标 ID 100% 不是独立现实迁移预测能力的证据，不能宣传为优秀预测器。

## 定位、唯一候选与主张边界

本轮证据支持的**近端定位为策略决策**：当前 bundle 缺失时选择不完成当前服务的动作并出现长时间无进展，且跨 seed 表现不稳。失败 episode 没有执行投影，因而不把主要问题归咎于投影。当前公开状态含 readiness；信息表示是否在 SA 内被有效编码、选模时点对各 seed 的影响，仍是未知，不能从关联直接宣称架构缺陷。

若另立实现任务，**唯一最小候选**是只在 SA 策略侧降低“当前 bundle 缺失时 action 3”的偏好；不改 action 4、目标缓存 admission、reward、环境或比较规则。理由是 13 个未完成 episode 中 8 个首次失败由 action 3 触发，且 action 3 没有目标缓存准备副作用。证伪条件：在事前冻结、匹配预算且不重复本 run 的开发比较中，current-missing 连续无进展未下降，或完成/按期/目标缓存复用/成本任一关键联合边界恶化，则拒绝；还须检查 action 3 的等待迁移是否在特定状态有益。本轮未实现、未训练，也未授权新实验。

全部 36 个实例已作为开发数据消费；无跨 run/车辆独立确认、真实无线、正式支持与外部复现。论文、论文表和主稿保持不动；paper-ready=`Unverifiable`。
