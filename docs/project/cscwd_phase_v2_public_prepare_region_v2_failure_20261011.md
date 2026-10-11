# CSCWD phase-v2 公开 prepare 区域 v2：唯一执行失败审查

- `reviewed_at`: 2026-10-11
- `literature_cutoff`: 2026-10-11（沿用同日公开近邻核查；本轮没有新文献或 novelty 主张）
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `cscwd_phase_v2_public_prepare_regions_20261011_v2`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`（执行源码）: `d4a3e5b0a8341d86d080da283c6726f872333ba2`（运行前已 push）
- `evidence_level`: 输入/选择/启动/失败回执可核对；区域效果 `E0_UNAVAILABLE`
- `verdict`: `FAIL_NO_COMPLETE_BRANCH_EVIDENCE`；贡献、优秀基线与 paper-ready 均 `Unverifiable`

## Evidence inventory

独立 v2 执行事前绑定 `cscwd_phase_v2_public_prepare_region_v2_execution_plan_20261011.md`、原 v1 科学方案 SHA-256 `472808fc90a3af16f8fe2c4f76acb69a9921a70f50d02184baa1eb40ec9c1def`、原 v1 失败回执 SHA-256 `55b826571c4b2474bf1f52ab228892d37855a850425ce20249b942100127e1ad`，并使用新 manifest SHA-256 `c90e3dce8d58bf823a48a1ec625fe9ca978fe12166003019ee36c61dd8067c13`。Python 3.9.6 零环境步 preflight SHA-256 `8a4c001261f13bad10fe96003f6d21b704c1085f7bbe98a8d942540e9b54193b`：24 格/11 起点/27 计划分支，全部 11 个入选前缀调用实际配对函数，其中 8 个非空，旧选择清单逐项一致。

唯一真实运行的本地 create-only root 为 `artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_v2/`，现存文件只有：

| 文件 | SHA-256 | 事实 |
|---|---|---|
| `start_receipt.json` | `546927737fad974583ac2757f5103a727be05690846944b9264222fdefbe34f9` | 干净执行 commit、来源/方案/manifest hash 和 12/36/288 硬限 |
| `selection_inventory.json` | `875163d7022ce209746c1e7dd954891d12da2f92f640cac7e815a0604a2d54cb` | 与 v1 同 SHA；24 格中 11 个 US101 起点，3 已暴露 `[4]` 回归、8 个同轨迹新覆盖 |
| `failure_receipt.json` | `4a19c502d5c5fb298601c211ed0fdb07d8a6b0b01ff829e164fc4bfda605c088` | `FAIL`、`RuntimeError: selected public states are not unique`、实际 77 个 `env.step`、`automatic_retry=false` |

没有 `starts.json`、`branches.json`、`paired_regions.json`、`event_ledger.jsonl`、`summary.json`、`artifact_integrity.json` 或完成回执。分支循环已产生内部临时记录，但失败路径没有持久化它们；因此**完成分支数、77 步对应的具体 action/成本/服务结果及逐状态全表均不可核验**。不能从总步数还原逐分支效果，也不能把 preflight 的 27 个计划分支写成完成分支。

## Root cause 与停止判定

本次通过了原 Python 3.9 前缀配对位置，随后在重建入选状态时触发执行器原有的全局 `public_state_sha256` 唯一性门。选择规则以 `(source, condition, prefix)` 作为不同候选格，而执行门按 `semantic_state + action_mask` 的公开投影全局去重；两种身份在至少一个入选格上不一致。**确切重复格及其物理原因未在失败原件中保存，不能仅凭报错认定是哪一对或改写选择规则。**此为冻结范围的 first-order protocol blocker，不是已验证的 action4 性能失败。保留原 root，未 retry/resume、换 root、扩数据、调参或修改生产算法。

## 审查结论

1. **Artifact 完整性**：三个已存在文件与 preflight SHA 复算一致；原 v1 失败文件 SHA 未变。缺完整分支/事件/汇总，区域 outcome 为 `E0_UNAVAILABLE`。
2. **Protocol/provenance**：新执行身份、干净 Git commit、旧方案/失败回执和 B pilot 行为 hash 均写入启动/失败回执。选择清单仍是同一 US101 轨迹的相关开发状态；Lankershim 没有 action3/action4 同合法起点。
3. **统计与 baseline 公平性**：原策略、action3、action4 的 27 分支只是计划数；没有可审计配对 outcome，也没有独立来源、formal/holdout/support 原始结果或论文所需统计单位。3 个旧回归不能与 8 个新覆盖状态合并成独立确认。
4. **机制兑现**：本次无法核验 action4 首步交付、后续目标计算使用、重算节省、服务损伤或额外字节。先前 3 起点局部负例仍只能按其独立原件引用。
5. **Claim 边界与 novelty**：不能声称发现或排除正区域、action4 普遍无益、SA 贡献或优秀基线排名；本轮没有新文献结论。

**Hard blockers**：全局公开状态唯一性门未过、完整 branch/event 原件缺失、正式证据缺席。**Major concerns**：同一 US101 相关开发轨迹、选择格与公开状态身份不一致。**Minor concerns**：零步 preflight 能检出前缀配对，却无法检出经历真实前缀后的公开状态碰撞。**Scorecard**：七维均 `N/S`，不打伪精确分数。**Safe claims**：仅可报告选择覆盖、77 个已执行环境步及失败门。**Prohibited claims**：正/无益区域比例、性能或成本配对、在线 gate、论文优势。**Required actions before re-review**：先另立只读诊断确认碰撞身份和归因，再由单独授权决定是否修改选择单位/执行协议；任何新科学运行须新身份及事前冻结。**TITS/TVT fit**：无完整结果，不作判断。
