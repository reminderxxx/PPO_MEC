# CSCWD raw phase v2 开发 pilot 独立证据复核（2026-10-11）

## 审查身份与结论

- `reviewed_at`: 2026-10-11；`literature_cutoff`: 2026-10-11（公开一手页面增量检索；不作穷尽性 novelty 结论）。
- `target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `cscwd_phase_v2_causal_public_pilot_20261011_v1`；冻结执行 Git commit：`5a79ffd018d4c260bc7a8417229a440c7d591fe4`。
- `evidence_level`: 该 **development pilot** 的原始记录可独立复算；按顶刊完整审查口径，formal/holdout/support 关键证据缺失，论文结论为 `E0_UNAVAILABLE / Unverifiable`，不把开发审计标作正式 `E2`。
- 开发判定：`MIXED`。候选 SA 在 v2 的同实例配对中减少重算和耗时，但增加状态传输字节；完成率、按期率、服务失败均无改善。**不晋级候选，不固定 SA 算法贡献或优秀基线排名，不更新论文主表。**

本轮只读 B 已完成原件，未执行新训练、原始轨迹环境 step、formal/hidden/holdout 或重跑筛选。机器复算位于本地 create-only `artifacts/analysis/cscwd_phase_v2_pilot_independent_audit_20261011_v3/`；`audit_summary.json` SHA-256 为 `36540a7b4beb1988bd9b81b38e275cba82d80ef471631d5be8635e7fef0ed9a9`。`v1` 为早期不含完整方法行的本地历史输出，`v2` 尚未锁定外层 terminal SHA；以 `v3` 为准。B 原件根在另一 worktree 的 `artifacts/experiments/cscwd_phase_v2_causal_public_pilot_20261011_v1/`；本仓库不提交真实数据、checkpoint 或运行原件。

## 1. Artifact 完整性和 protocol/provenance

只读脚本 `scripts/audit_cscwd_phase_v2_pilot_independent.py` 锁定并逐一验证 B `artifact_integrity.json` 中 **113/113** 个文件的字节数和 SHA-256；`completion_receipt.json`、`run_manifest.json`、`artifact_integrity.json`、`evaluation_rows.csv` 的 SHA 分别为 `88c2ac2c1cd7475b9adb3481d07d70cabe5657b4e0ec3c52fca6eb3fc2b0c8c8`、`8750250575566d68bc4c3030994536da6211db9d8aa65a6ce1a3a2173b84ea28`、`1ecea21b8daddbd79c908edf40a5adfbe38b8e411e9e30ffbc48c8974b95acb2`、`126db6fa0ba9c0c58627bdacbc0ecc64e30f673fe934aabbe1c0c128504071f8`。冻结 workload 与 A 机制矩阵审计身份也从 manifest gate 核对。

训练预算为 20 learned cell、23,040 环境训练步、3,840 optimizer step、80 checkpoint 选择 episode；80 候选检查后对应 20 selected 与 20 update24 checkpoint 身份。最终 320 learned + 18 rule episode 从 2,760 + 144 行逐步行为账本独立重建。全部 episode 核对动作 mask、服务结果、完成/按期/失败分母、每步 phase 时间、输入/模型/状态/模拟 backhaul/partial 字节守恒、重算时间与原奖励六项。`service_failure_rate` 是“该 episode 至少一次服务失败”的二元值，并非失败事件数除以动作数；已在 `full_method_rows.csv` 同时保留失败事件数。

子进程 `completion_receipt.status=complete`、返回码 0，科学原件与上述复算完整；外层 supervisor 原 `terminal_receipt.status=FAIL`，分类 `CHILD_EXECUTION_FAILURE_NO_RETRY`，原收据 SHA-256 `2fbf478e3286cc6f8bf8fc031f81ca17904c19f23df60226329f3765d3879a1e`。原因是原子进程收据缺少外层要求的 `scientific_execution_complete` 字段，外层把收据 schema 不匹配归为失败；这不是通过的外层 terminal，不能回填或改写原件。B 的后续收据修复只适用于未来 run。本报告同时保留两个状态，不把内部完成等同于正式通过。

## 2. 分版本、分视图的完整方法结果

下表 v2 每个 learned 方法/视图 30 行（2 个已暴露来源 × 3 条件 × 5 seeds），为 episode 行算术均值；同源条件/seed 的重复不是独立来源。传输单位为原件的十进制 MB。完成耗时的分母是已完成 episode，本表 v2 全部已完成。完整逐 seed/来源/条件/视图及全部方法见 `full_method_rows.csv`，分层均值见 `stratified_methods.csv`。

| v2 方法 | 视图 | 完成 | 按期 | 失败 episode | 耗时 s | 总传输 MB | 重算 s | 原奖励 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SA control | selected | 1.000 | 1.000 | 0.000 | 90.066228 | 12.292352 | 6.420439 | 12.796978 |
| SA candidate | selected | 1.000 | 1.000 | 0.000 | 88.503388 | 12.395371 | 4.837974 | 12.921957 |
| PPO | selected | 1.000 | 1.000 | 0.000 | 89.886231 | 9.078371 | 6.420439 | 12.812874 |
| MAPPO | selected | 1.000 | 1.000 | 0.000 | 88.715705 | 11.091047 | 5.125695 | 12.905579 |
| SA control | fixed update24 | 1.000 | 1.000 | 0.000 | 90.066228 | 12.292352 | 6.420439 | 12.796978 |
| SA candidate | fixed update24 | 1.000 | 1.000 | 0.000 | 88.503388 | 12.395371 | 4.837974 | 12.921957 |
| PPO | fixed update24 | 1.000 | 1.000 | 0.000 | 89.886231 | 9.078371 | 6.420439 | 12.812874 |
| MAPPO | fixed update24 | 1.000 | 1.000 | 0.000 | 88.325530 | 11.761940 | 4.694113 | 12.936481 |
| public immediate | fixed rule，6 行 | 1.000 | 1.000 | 0.000 | 91.492826 | 9.218851 | 0 | 12.684281 |
| public two-step | fixed rule，6 行 | 1.000 | 1.000 | 0.000 | 91.492826 | 9.218851 | 0 | 12.684281 |

v2 public two-step 因缺少公开的下一步二维移动输入逐步降级为 immediate；动作序列逐 cell 完全相同，不是第二个独立强基线。v1 单独看：四 learned 方法两个视图各 10 行的均值完全相同，完成 1、按期 0.5、失败 episode 1、耗时 104.091943 s、传输 2.554029 MB、奖励 2.179455；v1 immediate/two-step/privileged exact 各 2 行均为完成 1、按期 0.5、失败 0、耗时 99.091943 s、传输 2.554029 MB。`privileged exact` 仅限 v1，不具备 v2 在线公共权限；v1/v2 物理合同不同，不以其差值评估算法改进。

## 3. 配对统计单位与事件归因

`paired_sa_cases.csv` 保存全部 80 个 SA 同实例配对（两个视图各 40；v2 各 30）。v2 两视图的服务/物理成本数值相同，不能视为额外独立复制。Lankershim 三条件的每个 seed 均为相同动作序列、配对服务与物理成本差值为零。US101 默认和 adverse 各 3/5 seeds 于 step0 首次分叉，ample 为 5/5；其余相同。US101 三条件的候选平均每 episode 耗时差分别为 `-2.558846/-2.553451/-4.264743 s`，传输差 `+0.168576/+0.168576/+0.280960 MB`，重算差 `-2.589489/-2.589489/-4.315814 s`；按期和失败差始终为零。原始两个来源曾用于开发：Lankershim 训练、US101 checkpoint 选择，最终评估又复用两者。五 seed 只是这些来源内的重复，无法给出独立来源级 CI 或正式推广结论。

`event_attribution.csv` 的 2,904 行只给**观察到的后续使用**标签；没有从分叉后的不同状态强行断言单动作因果。v2 selected 30 个候选 episode 的 22 次 action4 中，11 次目标交付后观察到目标 RSU 继续计算且无需重算；另 11 次目标先离开，状态仍发送 1,545,280 bytes 而交付失败。候选状态总字节 3,090,560，control 为 0，因此相对 control 的 `+0.103019 MB/episode` 全是状态字节。候选与 control 各自模型传输均为 289,258,320 bytes，拆成同一步服务、后续在已准备 RSU 计算、后续未在已准备 RSU 计算各 96,419,440 bytes；“未后续计算”仅指有限 episode 内的观察，不可直接认定在更长时域无价值。输入字节相同，partial 均为零。PPO 模型传输 192,838,880 bytes、状态 0；MAPPO selected 模型 250,690,544 bytes、状态 2,528,640 bytes（9 次交付、9 次目标先离开）。这些记录说明强基线之间存在成本取舍，没有服务优势排名。

原件一个 US101 default/seed7 分支可追踪该现象：候选先成功传状态，下一节点免重算；随后仍执行 action4 且目标先离开，额外状态字节未形成新交付。control 不传状态而发生后续重算。这是已记录的机制链和失败链；首次动作分叉后的状态已变，不能把其后每项差值都归因到单一动作。独立脚本复算与 B 汇总的 `MIXED` 方向一致；B 汇总混合 v1/v2 的均值不用于本报告的 v2 物理结论。

## 4. Claim 边界与下一项有界确认设计

- 可说：在这两个已暴露来源和固定预算的 v2 开发运行中，候选 SA 出现“更少重算/更短耗时、更多状态字节、服务指标不变”的取舍；同一 US101 来源的重复 action4 有成功交付和目标先离开两类事件。
- 禁说：SA 严格优于 PPO/MAPPO/公开规则、五 seed 构成五个独立移动来源、双 checkpoint 视图构成复现、v2 two-step 是独立基线、状态发送必然避免重算、真实无线 backhaul 已测量、当前结果支持 TMC-ready 或固定算法创新。
- 顶刊审查的 artifact→protocol→统计公平→机制→claim 顺序到此止于 development。缺 formal/holdout/support 原件和独立外层 terminal PASS，完整评分各维均 `N/S`；不作 TMC-ready 或投稿结论。公开近邻增量检索已写入 `literature_reference_table.md`：已有 [TVT 依赖感知跨站迁移](https://doi.org/10.1109/TVT.2023.3310520)、[TMC 移动预测主动迁移](https://doi.org/10.1109/TMC.2023.3271655)及既有 TMC 双依赖缓存/卸载工作。因此“DAG＋迁移＋缓存/预测”组合本身不足以固定 novelty；当前 artifact 更未兑现严格服务优势。该筛查不是穷尽性不存在证明，不对新论文逐篇打水平分。

只提出**一个**后续独立任务：锁定原始失败链中“首次成功交付后下一次 action4 目标先离开”这一重复发送问题，按公共状态前缀去重，至多 12 个合法决策起点。每起点从同一冻结环境状态分别执行 action4 与合法 action3，后续使用同一冻结公共 continuation，最多各 8 步；总上限 24 分支/192 environment steps、0 训练、0 hidden/formal 读取。执行前固定起点身份、动作合法性、继续策略、原始状态/hash、预算与停止规则；两动作输入只能是共同公开字段，环境实际未来仅在执行后用于结果归因。事前主要判据：在服务完成、按期和失败均不退化时，action3 是否减少该起点及后缀的状态/模型/输入总字节；同时报告重算与耗时，所有负例逐一起列。这个设计只检验开发局部是否存在被支配的重复 action4，不复用为独立论文样本；本轮**未执行**。

## 5. 复核命令与本地输出

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q tests/test_cscwd_phase_v2_pilot_independent_audit.py
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/audit_cscwd_phase_v2_pilot_independent.py \
  --run-root /Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/artifacts/experiments/cscwd_phase_v2_causal_public_pilot_20261011_v1 \
  --terminal-receipt /Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/artifacts/experiments/cscwd_phase_v2_causal_public_pilot_20261011_v1_supervisor/terminal_receipt.json \
  --output-root artifacts/analysis/cscwd_phase_v2_pilot_independent_audit_20261011_v3
```

输出目录必须不存在；重算应换用新的 create-only root，不覆盖 `v3`。四张表 SHA-256 分别为 `full_method_rows.csv` `68d1fc07b62170ea73de82bb8a7f6d0dd94af2d1c9e6f7ea71aa4efda6022864`、`stratified_methods.csv` `723065437b7126cf54dc1d933d9f5a95aadd4cba64db666b71f2b51a49bbebf9`、`event_attribution.csv` `48b6e67d0d3dc738872c326a4f250c42a267769a765258fa9c39c806a11df018`、`paired_sa_cases.csv` `b48770f42cef4b3879646b14b7260b1da10cd50a799fc24aca50aea621c35812`。
