# CSCWD phase-v2 公开 prepare 区域诊断：一次运行失败记录

- `reviewed_at`: 2026-10-11
- `literature_cutoff`: 2026-10-11（沿用同日既有公开近邻核查；本轮没有新文献结论）
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `cscwd_phase_v2_public_prepare_regions_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`（执行源码）: `82818d97a0cbe6c8f6570a770cb81f8ff73f1144`
- `evidence_level`: 选择清单与失败回执可核对；区域效果为 `E0_UNAVAILABLE`
- `verdict`: `FAIL_NO_SCIENTIFIC_RESULT`；论文与优秀基线判断为 `Unverifiable`

## Evidence inventory 与执行边界

事前方案为 `cscwd_phase_v2_public_prepare_region_plan_20261011.md`。一次 create-only 执行写入本地 `artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_v1/`，仅有下列原件：

| 原件 | SHA-256 | 可支持的结论 |
|---|---|---|
| `start_receipt.json` | `a90a5428ba0aa8bd77883a9bbb25fbff5ac88cb194fa7845c3664faaa899cedc` | 冻结执行身份、来源 hash 与预算 |
| `selection_inventory.json` | `875163d7022ce209746c1e7dd954891d12da2f92f640cac7e815a0604a2d54cb` | 公开前缀的结果盲候选/排除清单 |
| `failure_receipt.json` | `55b826571c4b2474bf1f52ab228892d37855a850425ce20249b942100127e1ad` | `FAIL`、异常、`environment_step_count=0`、无自动重试 |

24 个预设浅层格中 11 个入选、13 个排除：6 格 action3/action4 不同时合法，7 格未见于冻结行为。入选前缀 `[]/[0]/[1]/[4]` 分别 3/3/2/3 个；全部来自同一 US101 开发轨迹，其中 `[4]` 的 3 个是已暴露回归起点，其余 8 个是新覆盖状态，仍非独立来源。Lankershim 无入选格。选择清单不含 branch outcome；它证明覆盖/缺口，不证明迁移效果。

执行在第一个 `env.step` 前抛出 `TypeError: zip() takes no keyword arguments`：脚本在 Python 3.9.6 环境使用了 Python 3.10 才支持的 `zip(..., strict=True)`。因此实际 **0 分支、0 环境步、0 训练、0 formal/holdout**；不存在配对、逐事件、区域汇总或成功交付原件。原始 run root 保持不变；按事前“一次运行、失败即停、不得自动重试”的约束，没有用新 root 补跑。后续源码仅将长度检查显式化以兼容 Python 3.9，并补了局部测试；修复不追认本次运行，也不改变执行 commit 的证据身份。

## 审查与决策

- **Artifact 完整性**：三个文件 SHA 已复算；`summary`、branch/event ledger、completion receipt 不存在，区域效果 `E0_UNAVAILABLE`。
- **Protocol/provenance**：选择账本符合预设 24 格与 11 起点，原始行为 CSV 身份记录在回执；运行失败发生在真实分支前。
- **统计与 baseline 公平性**：没有新的配对 outcome；三个旧回归起点及同一 US101 轨迹的其余状态都不能充当独立正式 cluster。正式 checkpoint/manifest/command log 与 formal/holdout/support 结果不齐全。
- **机制兑现与 claim 边界**：本轮无法判定正区域、无益区域或成本取舍，也不能把“未观察到正例”写作结论。先前 3 起点 action4/action3 局部负例仍只按其独立原件引用，不受本轮失败改写。
- **文献 novelty**：没有新的方法或正式效果证据，贡献仍待证；本轮未新增公开文献。

**Hard blockers**：无分支结果、单一入选来源、formal/holdout/support 缺席。**Major concern**：基线公平性、独立性和正向机制尚未验证。**Minor concern**：执行器 Python 下限未在运行前测试中覆盖。**Scorecard**：七维均 `N/S`，不以失败运行打伪精确分数。**Safe claims**：只可陈述公开筛选覆盖与兼容失败。**Prohibited claims**：正区域存在或不存在、action4 普遍应禁用、SA 优势、优秀基线排名、TMC-ready。**Required actions before re-review**：若另行授权，先以新 run ID 和新事前方案验证执行环境，再执行单次有界诊断；正式晋级另需独立来源与匹配的 formal/holdout/support。**TITS/TVT fit**：本轮无性能证据，不作判断。
