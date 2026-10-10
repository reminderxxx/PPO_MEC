# CSCWD 新原始 development 来源与有界可达性独立审查（2026-10-11）

## 审查身份与结论

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（本轮未更新文献近邻；不作 novelty 判断）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_new_development_source_20261011_v2`、`cscwd_new_development_reachability_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: 来源扫描 `14fa246d4ece3cb35d5e38100e3024c96e42d2bc`；可达性执行 `68800dd0e220860d0dc209db54705fc8c00768d7`
- `evidence_level`: `E2_ARTIFACT_AUDITED`，仅指本地来源/可达性原始 ledger 审计；正式论文证据仍不齐
- `verdict`: `DEVELOPMENT SOURCE ELIGIBLE / MECHANISM GATE FAILED / PAPER-READY UNVERIFIABLE`

此轮先以 [事前方案](cscwd_new_development_source_plan_20261011.md)冻结来源、选择规则、工作负载、五个固定方法与上限，再扫描原始 CSV，最后只执行一次有界检查。0 训练、0 checkpoint、0 formal/hidden 结果消费、0 新论文主表行。旧 split 和已消费区间不改写；检查结果不用于重选窗口。旧 40 个 24 帧 development 区间的 `0/40` 资格结论仍有效，它与本轮新增的两段长来源是不同问题。

## Artifact 完整性与来源协议

| 原件或输入 | SHA-256 / 身份 |
|---|---|
| `artifacts/analysis/cscwd_new_development_source_20261011_v2/source_manifest.json` | `34eee3a8d16f2dfe89f3a8d8dcd27f20159c0488e2f1a9ebc77be3aac747abde` |
| 同目录 `candidate_fragments.jsonl` | `560cd95930b6b558b84ddb0e8b4d7973c63f889302219ec3b4a21b720f3f8c08`；13,438 条 |
| `artifacts/analysis/cscwd_new_development_reachability_20261011_v1/reachability_manifest.json` | `481dd42489dbb9bab30f297237b4373fee9b1b3823082031ed308532f2332e0c` |
| 同目录 `step_ledger.jsonl` | `679942ac74cc618127142abc89a5eaa0cd109ff38cad213dfd77dd850a67bacb` |
| 原 NGSIM CSV | `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`；未复制/上传 |
| raw 环境 / calibrated executor / B 公开 estimator 源码 | `30666d7c874717d8464a64702c811116e96b931a5db6b57287c6c11d2dcc6575` / `c6ded6ce84230519160a40f29e276b5823eb204a35e31c35f914d1c6bfee3a59` / `ee6873d9b16e5b1e4ac0dcfab1eacce915d50aeeff8bf8d9a862093202984828` |

来源扫描覆盖 CSV 11,850,526 行。仅纳入有 13 位毫秒 `Global_Time` 且可稳定标识的 US101/Lankershim 共 6,410,252 行；排除时间语义不明确的 Peachtree 与 i80、早期匿名来源前 200,000 行、v27/v28/v71 strict formal/hidden 的原始时间区间及两端各 24 个完整空白帧。`invalid_source_rows=1,408,000` 经独立重算均为 US101 同 `(segment, Vehicle_ID, Global_Time)` 重复行，未发现无效坐标、短时间戳或非 100 ms 对齐；扫描器保守排除它们，入选轨迹无该重复。首轮 `v1` 因数字文本解析/JSON 整数序列化失败只留下部分目录，完整唯一来源原件为 `v2`。

按冻结顺序与 Lankershim 1、US101 2 的配额，实际只找到下列两个合格来源；无结果反馈补位。均为 1,189 个连续 frame、118.8 s。原始 CSV 二次重载与严格 split 的原始区间复算通过：formal/hidden 重叠 0；下表“历史暴露”只描述先前 train/dev 复用，不赋予独立确认性。

| 来源 | Vehicle / Frame_ID | 原始 `Global_Time` ms | 旧 train/dev 暴露数 | 最近 formal/hidden 端点间隔 |
|---|---|---|---:|---:|
| Lankershim | 215 / 1152–2340 | 1118935795100–1118935913900 | 1 | 68,100 ms |
| US101 | 1183 / 5368–6556 | 1118849293800–1118849412600 | 2 | 2,500 ms |

两段来自同一 NGSIM 原始数据集合，且曾与历史 development 原始区间相交。它们只能承载新的开发可行性检查；不能把两窗口、四工作负载、五方法的 40 行当作 40 个独立统计 cluster，也不能作为新的 strict holdout。

## 固定公共可达性结果

固定工作负载为 `regression_05`、`dev_00`、`dev_01`、`regression_04`。五个方法依次是固定 fallback、固定 serve、只用 raw `_info()` 的公开一阶/两步规则、合法时 prepare 否则 serve；规则调用仅接受公开信息，实际 contact 只写在诊断 ledger。执行 2 窗×4 工作负载×5 方法 = 40 episode、320 真实 step、565 模型预览上界，低于冻结的 60/1,440/5,000 上限。40/40 workflow 最终完成且 `terminated=True, truncated=False`，但只有 4/40 按期。

| 方法 | Workflow 完成 | 按期 | 失败服务 | 真实步 | modeled 完成秒范围 |
|---|---:|---:|---:|---:|---:|
| fallback | 8/8 | 0/8 | 0 | 50 | 56.54–103.11 |
| serve | 8/8 | 1/8 | 22 | 72 | 52.51–109.11 |
| public immediate | 8/8 | 1/8 | 13 | 63 | 48.51–109.11 |
| public two-step | 8/8 | 1/8 | 13 | 63 | 48.51–109.11 |
| prepare if legal else serve | 8/8 | 1/8 | 22 | 72 | 52.51–109.11 |

全量逐步 ledger 独立对账 40/40：分项成本加初始 0.1 s 与 episode modeled 完成秒一致，完成节点数、三类传输字节、迁移数、最终 `terminated` 与 episode 清单一致，执行动作全部 mask 合法。96 次请求与执行动作不一致均来自固定 serve/prepare 在 mask 外请求后的投影，已记录，不能当原样决策结果。共 70 次服务失败，其中 68 次 `current_rsu_contact_expires_before_commit`，另 2 次 native 失败；模型/状态传输字节和成功迁移均为 0，输入传输字节为 79,358,038。执行 action 4 共 12 次，12 次均因当前 RSU 接触不足拒绝，故没有成功 prepare/migration。总 step cost 3,178.035209 s，其中 service operation 3,031.417990 s、input transfer 6.617219 s，model load、state restore、recompute 为 0；不能用这些成本分项推断新算法机制收益。

长来源已容纳完整 DAG 终止，旧 2.3 s **数据时长**阻断在本轮 development 来源上解除。但固定模拟 RSU 几何与 raw 环境“当前 RSU 接触必须覆盖整步”准入共同造成 action 4 全部拒绝；迁移/状态准备机制没有被激活。这是当前环境合同的**机制可达性阻断**，不能从 0 迁移断言策略无效，也不能通过本轮结果改选车辆或放宽接触门。需要另立事前协议，明确 phase 分段、跨 RSU 服务/部分传输语义与可观测接触，再用独立开发来源核验；本轮不自动实施该修复或启动 B 训练。

## 公开基线独立复核与主张边界

B 的 `cscwd_public_rule_raw_fairness_20261011_v1` manifest/audit SHA-256 分别为 `21ffc41ac51e16fa58f4e6e5dd0c0ecc5bb9f8bf5f802399499e5230232992b5` / `45f398ee16708bcae256be7fb8a300d380a9933b839243d69260ab70767e0e7b`。A 复核源码与原件：两个只差隐藏未来后缀的合成轨迹有完全相同的 raw `_info()`、mask 与 PPO 输入；两公开规则选 action 0 且不改变环境状态；seed 7、0 update PPO 的数值概率、log-prob 相同。旧 exact-clone immediate/two-step 在慢/快后缀分别选 0/2，action 3 exact preview 成功/拒绝分化，因此其权限应标 `privileged exact-transition reference`。此审计只证明该固定反例的公开输入权限一致；公开规则与 learned reward 的目标函数仍不相同，也无强基线性能资格。B 公开规则 SHA 已锁入上述 40 episode 可达性原件。

**安全表述**：已固定两个长 development 来源，数据时长足够，但当前接触合同下 0 成功迁移、仅 4/40 按期；公开规则通过一个隐藏后缀反例的信息权限检查。**禁止表述**：SA 优于优秀基线、两步规则已是已验证强基线、真实无线迁移有效、formal/holdout 独立确认、TMC-ready 或论文贡献已固定。正式 checkpoint、匹配训练、formal/holdout/support 原始结果及跨数据源统计均不存在于本轮，论文 verdict 为 `Unverifiable`，内部评分 `N/S`。

## 执行与未覆盖风险

- 来源：`/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/freeze_cscwd_new_development_source.py`，参数和全部输入 hash 见本地 `source_manifest.json` 与事前方案。
- 可达性：`/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/audit_cscwd_new_development_reachability.py --raw-csv-path '/Users/howen/Projects/PPO_MEC/data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv' --source-manifest artifacts/analysis/cscwd_new_development_source_20261011_v2/source_manifest.json --public-estimator-path /Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/src/agents/causal_public_action_estimator.py --output-root artifacts/analysis/cscwd_new_development_reachability_20261011_v1`。
- 此轮未测 learned checkpoint、训练稳定性、正式多 seed、独立 holdout、真实 RSU/无线/队列、跨界分段执行及两窗口统计效应；均不得从本轮外推。下一科学执行门保持关闭。
