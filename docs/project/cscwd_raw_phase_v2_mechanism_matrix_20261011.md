# CSCWD raw phase v2 冻结开发机制矩阵（2026-10-11）

## 审查身份与判定

- `reviewed_at`: `2026-10-11`；`literature_cutoff`: `not_evaluated_in_development_mechanism_run`；`target_venue`: `IEEE TMC`；`artifact_run_id`: `cscwd_raw_phase_v2_mechanism_matrix_20261011_v1`；`policy_version`: `tmc_review_policy_v3_20260621`；`git_commit`: `61b8200`（执行器）；`evidence_level`: `local_development_artifact_audited / paper E0_UNAVAILABLE`。
- 该矩阵是两条已暴露 NGSIM 开发轨迹与一个已暴露 `dev_01` workload 的有界机制检查。正式 checkpoint、formal/holdout/support 原始结果均缺，因此 paper-ready、优秀基线排名和 SA 贡献判定为 `Unverifiable`，不评分。没有读取 hidden 结果，没有新训练或模型真实调用。
- 事前合同：`cscwd_raw_phase_v2_frozen_contract_20261011.md`；配置 SHA-256 `2a0cee4b4b2ba512f8015296ee1946698dd51066d1ae5744e22b4d428d176343`。结果前已固定两窗×一任务×三条件×v1/v2×五方法=`60` cells，上限 `2,880` real step/`10,000` preview。本次 `60` episode、`522` step、`731` preview 上界；无补格、换窗或调参。

## 原件与独立对账

| 本地 create-only 原件 | SHA-256 |
|---|---|
| `artifacts/analysis/cscwd_raw_phase_v2_mechanism_matrix_20261011_v1/matrix_manifest.json` | `cb581f6871c0928f13f58f14fb77568290277247dc7de8c03ed355aeeca4bc2c` |
| `artifacts/analysis/cscwd_raw_phase_v2_mechanism_matrix_20261011_v1/step_ledger.jsonl` | `bfb2d6743266a24c2a37bf9b714a53f723f0595e6215ec82a8a0d348c9c64028` |
| `artifacts/analysis/cscwd_raw_phase_v2_matrix_audit_20261011_v1/audit.json` | `30d821d4fb7c2e8d26c85378e4824fa4a782f62bf6eb1181510f0fe9bf56db8d` |

执行器核验 raw CSV、source manifest、workload manifest、v1/v2 环境源码、冻结合同、真实预检与公开 fixture 的 SHA。B 公共估计器源码 SHA-256 `a6b49bd2cfd649abe8208864aa5f76935a54f391ab26bba77cc1bcd3b3c61790`，规则只收到 `semantic_state/action_mask/run_metadata`，没有真实 future、原始坐标后缀或 transition。独立只读审计脚本按 `(window, condition, profile, method)` 重新检查 60 个唯一 cell、522 条连续时钟、总/三类传输字节、v2 各相位耗时和 backhaul 字节拆分，以及逐事件/episode 的完成和失败数，`all_checks_passed=true`；重复节点执行计数为 0。

## 全量开发结果

下表每行聚合**同样两个已暴露窗口**，成本是两个 episode 的 `modeled_completion_seconds` 之和，不能视为独立样本均值或正式统计。`completed/on-time` 分母都是 2；`migration` 为成功次数；`failure` 为 service failure 次数。`I`=public immediate，`T`=public two-step，`P`=prepare-if-legal-else-serve。v2 的 T 因缺少可公开重建的下一步二维移动状态，固定降级为同一 immediate score；I/T 结果逐格相同，不算两套独立强基线。

| profile | condition | method | completed | on-time | migration | failure | cost s | model/state/backhaul MB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v1 | default, adverse, ample 各同值 | fallback | 2 | 0 | 0 | 0 | 206.21 | 0/0/0 |
| v1 | default, adverse, ample 各同值 | serve | 2 | 1 | 0 | 5 | 208.18 | 0/0/0 |
| v1 | default, adverse, ample 各同值 | I/T 各同值 | 2 | 1 | 0 | 0 | 198.18 | 0/0/0 |
| v1 | default, adverse, ample 各同值 | P | 2 | 1 | 0 | 5 | 208.18 | 0/0/0 |
| v2 | default | fallback | 2 | 0 | 0 | 0 | 206.21 | 0/0/0 |
| v2 | default | serve | 2 | 1 | 0 | 4 | 190.37 | 0/0/0 |
| v2 | default | I/T 各同值 | 2 | 2 | 1 | 0 | 183.03 | 19.28/0.28/19.56 |
| v2 | default | P | 2 | 2 | 1 | 2 | 179.10 | 19.28/0.28/19.56 |
| v2 | adverse link | fallback | 2 | 0 | 0 | 0 | 206.21 | 0/0/0 |
| v2 | adverse link | serve | 2 | 1 | 0 | 4 | 190.37 | 0/0/0 |
| v2 | adverse link | I/T 各同值 | 2 | 2 | 1 | 0 | 183.66 | 19.28/0.28/19.56 |
| v2 | adverse link | P | 2 | 2 | 1 | 2 | 179.73 | 19.28/0.28/19.56 |
| v2 | ample cache | fallback | 2 | 0 | 0 | 0 | 206.21 | 0/0/0 |
| v2 | ample cache | serve | 2 | 2 | 0 | 0 | 179.05 | 0/0/0 |
| v2 | ample cache | I/T 各同值 | 2 | 2 | 1 | 0 | 182.26 | 0/0.28/0.28 |
| v2 | ample cache | P | 2 | 2 | 1 | 0 | 174.79 | 0/0.28/0.28 |

v1 adverse link 的 backhaul 条件在 v1 物理合同中**没有作用**；v1 三条件各方法结果完全相同，不能把重复格当稳健性复制。v2 共 18 次 action 4 尝试：9 次 `delivered`、9 次 `target_departed_before_delivery`；这些是同一个 US101 来源在三个条件和三种方法上的重复，不能按 18 个独立事件做统计。后者在已完成状态传输/恢复后失败，状态不得记 prepared。v2 中 12 次 `current_bundle_not_ready` service failure、3 次 `prefix_recompute` 完成；本矩阵没有部分传输超时（合成负例另见环境验收），不隐去 9 次交付失败。v1 仍有 30 次 `current_rsu_contact_expires_before_commit`，这是旧整步合同下的事实。

默认条件的两个窗口逐项不同：Lankershim 中 I/T 按期、无迁移、无 service failure，P/serve 更快但各 2 次 service failure；US101 中 I/T/P 均按期、各 1 次成功交付及 1 次目标先离开。P 的两个窗口总成本低于 I/T，但服务失败更多；wide cache 下 serve 成本 179.05 s、I/T 182.26 s，而后者有一次迁移且无失败。不存在由这些开发行支持的单一全面优胜方法。

## 冻结贡献与后续实验边界

本轮可保留的**工程贡献候选**是逐相位物理合同、公开估计/真实执行分离、部分字节与时钟守恒、状态生成后才迁移以及目标离开负例；这只是可复现实验机制，尚未证明新算法优势。`raw_ngsim_event_time_v1` 与 v2 的差别是物理合同版本变化，不得写成同一算法的因果收益。默认/ample cache 改变初始资源，策略随后分叉，不构成单因素 model-prep 消融。

下一轮若要声称恢复优于合法重算或模型预备的边际价值，必须**另行事前冻结**同一个公开决策起点的匹配分支：相同节点、current/target cache、输入/速率/trace，分别只开关状态迁移或目标 model prep；两分支完整计回程/重算、失败等待、后续 victim reload 与 deadline，并固定预算和全部负例。当前 60 格不能事后扩成第四条件或补跑有利窗口。优秀 learned baseline 还需 B 的相同输入/预算/检查点 pilot，最终论文主张需独立 formal/holdout 和统计原件。
