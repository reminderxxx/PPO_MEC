# SA 当前执行—未来准备失衡：单因素有界重训结果（2026-10-06）

## 1. 审查身份与最终判断

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-06`
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `calibrated_workflow_prepare_balance_ablation_20261006_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `run_source_git_commit`: `5709b90b296df63283472dcb2e799cbe796284a9`（运行时 HEAD；本轮实现尚未提交）
- `interface_profile`: `calibrated_workflow_interface_v2`
- `action_contract`: `independent_heads_executed_env_v2`
- `evidence_level`: `E2_ARTIFACT_AUDITED_NONFORMAL_DEVELOPMENT_ABLATION`
- `paper_ready_verdict`: `Unverifiable`
- `decision_category`: **存在其他未定位问题**
- `candidate_disposition`: **拒绝把删除 auxiliary loss 作为过度准备修复；保留原配置，后续若继续应重新设计训练 target，而不是继续删增强。**

单因素候选没有改善 workflow completion，也没有减少 current-missing action 4。相反，在 frozen check 上 action 4、
无效准备、不可行目标准备和 service failure 都增加。它没有退化为始终 action 0；行为向 action 4 与 action 2 转移。
因此本轮主假设被否证：auxiliary target 的 current-readiness 缺口是真实设计冲突，但删除整个 auxiliary loss 不是有效修复，
该 loss 不能被认定为唯一根因。

## 2. 冻结实现与预算

定位原件和事前候选见 `sa_prepare_balance_localization_20261006.md`。A/B 唯一差异：

| 臂 | `auxiliary_coef` | 训练身份 | 其他 SA 参数 |
|---|---:|---|---|
| A 原 SA | 0.1 | 历史精确匹配复用；未伪称重跑 | 不变 |
| B no-auxiliary | 0.0 | 新 3 seeds × 192 episodes | 不变 |

A 复用原 artifact 的 selected checkpoints、576-row 完整学习曲线和 primary results；本轮只重放 selected checkpoint
生成 readiness/feasibility 扩展行为字段。逐 episode 标准指标与历史原结果一致，只有 `~1e-14` 浮点表示差异。
PPO、controller-MAPPO 和 two-step rule 直接复用相同接口、实例和评价身份的历史结果。

B 固定 seeds `7/17/29`，每 seed 192 episodes、每 episode ≤24 steps，候选 checkpoint 为 48/96/144/192，仍只用
4 个 dev instances 按冻结字典序选模。新增理论上限 `13,824` steps；实际 `4,214` steps、72 updates、35.67 s，
selected episodes=`144/192/96`。无自动重试、追加 seed、延长预算或结果导向调参。全部 576 个训练 episode 和全部失败均保留。

训练期描述性 completion 为 A `0.9826`、B `0.9878`；该轻微训练内变化没有转化为开发评价 completion 改善。

## 3. 服务结果与匹配对照

`on-time` 定义为 workflow 完成且 episode deadline violation 为 0。elapsed 只对完成样本平均，并显式列覆盖；任何未完成
episode 都不作为速度优势。

### 已暴露 regression（A/B 与 learned 方法各 36 episodes；rule 12）

| 方法 | completion | on-time | service failure | 完成 elapsed / 覆盖 | transfer MB |
|---|---:|---:|---:|---:|---:|
| A 原 SA | 0.944 | 0.444 | 0.389 | 74.73 / 34/36 | 717.77 |
| B no-auxiliary | 0.944 | 0.389 | 0.389 | 73.44 / 34/36 | 408.01 |
| PPO（复用） | 1.000 | 0.222 | 0.000 | 101.32 / 36/36 | 760.28 |
| controller-MAPPO（复用） | 1.000 | 0.583 | 0.194 | 72.07 / 36/36 | 1211.82 |
| two-step rule（复用） | 1.000 | 0.833 | 0.000 | 52.30 / 12/12 | 169.03 |

### Frozen development check（A/B 与 learned 方法各 24 episodes；rule 8）

| 方法 | completion | on-time | service failure | 完成 elapsed / 覆盖 | transfer MB |
|---|---:|---:|---:|---:|---:|
| A 原 SA | 0.958 | 0.125 | 0.417 | 106.53 / 23/24 | 670.68 |
| B no-auxiliary | 0.958 | 0.250 | 0.583 | 81.60 / 23/24 | 243.58 |
| PPO（复用） | 1.000 | 0.000 | 0.000 | 107.70 / 24/24 | 676.41 |
| controller-MAPPO（复用） | 1.000 | 0.292 | 0.167 | 91.23 / 24/24 | 1084.48 |
| two-step rule（复用） | 1.000 | 0.625 | 0.000 | 51.67 / 8/8 | 65.03 |

候选仍低于 PPO、MAPPO 和 two-step 的 1.000 completion。frozen check 的 on-time 和成本点估计改善，与 service failure
上升并存，不能解释为服务权衡全面改善。完成 elapsed 的 A/B 配对仅覆盖两者都完成的 22 个 episodes，B−A 为
`-21.86 s`；它排除了失败样本，故只作条件描述，不称速度优势。

## 4. Prepare 行为和收益来源

`service-safe prepare` 要求当前 bundle ready 且 action-4 clone 的 target prepare 不因 contact/capacity 失败；其余 action 4
记作 invalid prepare。`realized prepare` 要求实际当前节点完成且 migration 成功。

| split / 臂 | action 4 / 全动作 | current-missing action 4 | realized / action 4 | safe / invalid | target-infeasible |
|---|---:|---:|---:|---:|---:|
| regression / A | 174/294 = 59.2% | 44/83 = 53.0% | 99/174 = 56.9% | 99 / 75 | 42 |
| regression / B | 194/298 = 65.1% | 55/103 = 53.4% | 102/194 = 52.6% | 99 / 95 | 53 |
| frozen / A | 110/201 = 54.7% | 33/66 = 50.0% | 53/110 = 48.2% | 54 / 56 | 34 |
| frozen / B | 149/208 = 71.6% | 45/78 = 57.7% | 62/149 = 41.6% | 63 / 86 | 59 |

候选不是“放弃准备”：frozen action 0 比例从 27.4% 降到 13.0%，action 4 从 54.7% 升到 71.6%，并把原 action 3
大量换成 action 2。恢复成功数 `53→62`，但新增 action 4 更快，成功率 `48.2%→41.6%`；invalid prepare `56→86`。
因此成本降低主要来自不同动作组合和更少大模型传输/重算，不是更好的执行—准备平衡。

以 source window 为 outer unit 的 5,000 次 percentile bootstrap：

- regression completion delta=`0 [0,0]`；invalid prepare=`+0.556 [0.278,0.862]`，target-infeasible prepare=
  `+0.306 [0.083,0.583]`。
- frozen completion delta=`0 [0,0]`；on-time=`+0.125 [0.042,0.250]`，transfer MB=
  `-427.10 [-844.03,-90.60]`，但 invalid prepare=`+1.250 [0.458,2.293]`，target-infeasible prepare=
  `+1.042 [0.250,2.167]`，service failure=`+0.167 [0,0.292]`。
- frozen action-4 attempts/window=`+1.625 [0.458,2.875]`；realized successes/window 仅
  `+0.375 [-0.250,1.167]`。最大连续无进展 delta=`+0.167 [-0.125,0.417]`，没有稳定改善。

所有 handoff pressure、link-error、capacity、sharing 和 topology strata 的 completion 点估计均保持不变；invalid prepare
在每个观察 strata 都增加。失败也没有消失：regression/frozen 的失败 seed 从 A 的 seed 17 转移到 B 的 seed 29。

## 5. 实现正确性与因果边界

- 1,001 条 A/B behavior ledger 的 `policy_log_prob == executed_action_log_prob`，最大差 0；旧 likelihood bug 未复发。
- A/B 均使用 `raw_policy`；margin/sharpening 在运行路径中仍 dormant，不能把候选变化归给推理 bias。
- candidate checkpoint 记录 `auxiliary_coef=0.0`；网络初始 state dict 与 A 同 seed byte-equivalent，环境、reward、数据、
  action mask、interface、episode cap 和 selection rule 未变。
- 单因素结果只说明：在这组配置与开发实例上，删除完整 auxiliary loss 没有改善过度准备。它不能证明 auxiliary 普遍有益，
  也不能排除 hard event target、temporal 子项、event temperature、PPO 信号或有限预算之间的交互。

## 6. 论文结论

不进行最近邻文献核验：候选没有稳定且可解释的正向改进，按事前规则停止。接口纠错、配置中性化和负结果都不是算法创新。
论文可保留：

1. 已验证的 state recovery、dependency safety 和真实 adapter lifecycle 系统证据；
2. 修复接口下 SA 仍有 action-4 policy gap；
3. auxiliary target 存在 current-readiness 信息缺口，但整体移除会加剧无效准备；
4. 强 two-step planner 在当前 workload 仍更简单、更完整、更低成本。

禁止表述：no-auxiliary 改善 completion、auxiliary 是过度准备唯一原因、候选是新算法、成本下降代表服务优势、开发检查是独立泛化。

## 7. 唯一下一步

若继续该学习线，只做一项：**重新设计 event auxiliary target，使 prepare 正标签同时满足 current-node service readiness 与
target prepare feasibility，再按新任务单独冻结验证；不要继续删除整个 auxiliary、调 reward 或扩大预算。** 本轮不实现该设计。

## 8. Artifact 入口

- 诊断：`artifacts/analysis/calibrated_workflow_prepare_balance_diagnosis_20261006_v1/`
- 有界重训：`artifacts/benchmarks/calibrated_workflow_prepare_balance_ablation_20261006_v1/`
- 主比较：`comparison_report.json`、`comparison_summary.csv`、`stratified_ablation.csv`
- 原始记录：`training_curves.json/csv`、`evaluation_rows.json/csv`、`behavior_ledger.csv`、
  `candidate_checkpoint_selection.json`、`candidate_training_summary.json`
- 回执与完整性：`completion_receipt.json`、`analysis_receipt.json`、`run_manifest.json`、`artifact_integrity.json`
- checkpoint：仅本地保留，不提交 Git。
