# CSCWD 长开发来源接触/阶段阻断诊断（2026-10-11）

## 审查身份与裁决

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（未新增外部文献；本轮不评价 novelty）
- `target_venue`: IEEE TMC
- `artifact_run_id`: `cscwd_new_development_contact_ledger_20261011_v1`、`cscwd_new_development_action4_phases_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: 输入 raw/reachability `68800dd0e220860d0dc209db54705fc8c00768d7`；公开 estimator B `6bf5d0b0ba9af2460cdc736f14909b0692126a48`
- `evidence_level`: `E2_ARTIFACT_AUDITED / bounded development diagnostic`，仅对冻结 ledger 与 12 个固定 action 4 起点有效
- `verdict`: `RAW FULL-STEP CONTACT BLOCKS MIGRATION; PHASE-ONLY RELAXATION UNJUSTIFIED; PAPER-READY UNVERIFIABLE`

本轮遵照[事前边界](cscwd_new_development_contact_diagnosis_plan_20261011.md)，不选新窗口、改几何、改算法、训练或读 formal/hidden 结果。A 首先对原 40 episode/320 step 做 0 新环境步的账本审计，然后仅在原 12 次 action 4 起点重建两条已冻结 trace；12 次前缀与 native 预览合计 **20 个额外 `env.step`**，低于事前 288 上限。它们是确定性诊断，不是新策略 episode 或公平性能矩阵。

## 原件及对账

| 本地 create-only 原件 | SHA-256 | 用途 |
|---|---|---|
| `artifacts/analysis/cscwd_new_development_contact_ledger_20261011_v1/contact_ledger_audit.json` | `f35fe1ded86663488568864790c523a24a62ae66ad517c592d7aa75481d76b93` | 320 行逐条重算；0 原始 CSV 重读、0 新 step |
| `artifacts/analysis/cscwd_new_development_action4_phases_20261011_v1/action4_phase_audit.json` | `b3f0634ad0c305fdd9f0cb5780c3b0120c94aa8b4e3600ce175f5fd601d44554` | 12 次 action 4 的公开估计、几何、native 阶段预览和 raw 回滚对照 |
| `artifacts/analysis/cscwd_public_action4_state_20261011_v1/public_info.json` | `4036254c97db4bad48936d8b5f8276931ca2985523250a6385e697b1ab8e8f98` | US101 `dev_01` step0 仅公开 `semantic_state`、`action_mask`、`run_metadata`，供 B 独立 consumer 回归；0 新环境步 |
| 父 `reachability_manifest.json` / `step_ledger.jsonl` | `481dd42489dbb9bab30f297237b4373fee9b1b3823082031ed308532f2332e0c` / `679942ac74cc618127142abc89a5eaa0cd109ff38cad213dfd77dd850a67bacb` | 冻结 40 episode/320 step 输入 |

执行器逐段重放的 step 前后 clock、mask、接触预算、action、拒绝原因与原 ledger 一致，native 预览的 `planned_step_cost_seconds` 与原 12 行逐项一致；公开估计和 native clone 预览均不改变源环境状态。原 CSV、source manifest、raw/base 环境、工作负载与 B 公开 estimator 均按父原件 SHA 核验，未复制/上传原始坐标。
公开状态快照的完整 `_info()` canonical SHA-256 为 `049de83dba0218565999c821d7b053391f9ae1f32f45e4c01079358e12028fa8`，与阶段原件的一阶/两步两行相同；快照不含实际 contact、未来坐标或 native 预览。

## 68 次接触拒绝的全量账本复算

68 行接触拒绝对应 43 个去重 `(window, design, step, action, clock, planned)` 尝试；重复方法在同一状态上形成相关观测。68/68 发生时起点实际 contact **大于 0**，且轨迹剩余时间大于完整预览 step 成本，因此本轮这些拒绝不是轨迹末端不足，也不是起点无覆盖。contact 拒绝按动作分别为 action 0: 24 行/12 去重、action 3: 32 行/21 去重、action 4: 12 行/10 去重。按来源 Lankershim 28、US101 40 行。前两动作共 6 行/3 去重状态出现 `planned <= public contact` 却 `planned > actual contact`，是可见的公开恒速 contact 预测乐观误差；**action 4 中该类误判为 0**。

action 4 全 12 行的公开 contact 范围 `0.497–1.248 s`、隐藏实际 contact `0.420–1.253 s`、native planned 整步 `1.523–6.121 s`。12/12 对 public 和 actual contact 都超时，最小实际超额 `0.270 s`、最大 `4.868 s`。其第一帧之后轨迹还剩 `115.177–118.700 s`，因此仅延长原始车轨迹并不能使当前 RSU 整步门放行。真实 raw 结果 12/12 `current_rsu_contact_expires_before_commit`，模型/状态字节均为 0。

## 12 次 action 4 的阶段和几何

raw profile 的三圆 RSU 几何以轨迹首点为锚，中心沿 y 轴每隔 12 m，半径均为 10 m；这是**模拟配置**，不是 NGSIM 实测 RSU 覆盖。Lankershim 相关起点到当前中心约 `8.619 m`、公开速度 `2.945 m/s`、实际 contact `0.420 s`；US101 两类起点到当前中心约 `0.742/3.550 m`、公开速度 `7.417/7.449 m/s`、实际 contact `1.253/0.869 s`。两个来源在起点均有覆盖，却在计划服务结束前离开当前圆。逐行 RSU、距离、速度与预测/实际预算见阶段原件，不用圆形假设外推真实无线。

| action 4 阶段类别 | 行数 | 去重尝试 | 机制解释 |
|---|---:|---:|---|
| native 模型+状态准备自身超实际 contact | 4 | 4 | Lank `dev_01`: `.724 > .420 s`；另 3 行 `regression_04` 的状态网络约 `87.82 s`，远大于 `.420–1.253 s` |
| native 准备可完成，但当前节点服务失败 | 4 | 4 | native 可暂存模型，当前 bundle 缺失或服务失败，状态迁移未提交；raw 整步仍回滚 |
| native 准备可完成、当前服务完成、预览迁移成功 | 4 | 2 | US101 `dev_00` 1 行、`dev_01` 3 行（后者三方法同起点）；整步 `2.247/6.121 s` 仍超实际 `1.253 s`，raw 合同拒绝并回滚 |

对这 12 个固定状态，公开 action 4 分类为：`current_service yes/no = 7/5`，`target_prepare yes/unknown = 7/5`，`raw_full_step_contact_fit no/unknown = 6/6`，`cost_status known/unknown = 6/6`，`deadline_fit yes/unknown = 6/6`。公开一阶/两步规则各有一次同一 US101 `dev_01` 起点选择 action 4，而该状态其自身 estimator 给出 `raw_full_step_contact_fit=no`；这是规则消费估计字段的具体缺口，B 线应独立最小修复并测试。该修复最多避免此两行明知整步不可行的选择，不能生成本轮不存在的有效迁移。

## First-order 归因与唯一后续建议

1. **数据时间**：本轮两条 118.8 s 来源允许完整 workflow；68 次接触拒绝均非轨迹剩余时间不足。
2. **模拟几何与物理预算**：起点当前圆虽覆盖，但有效接触仅 `.420–3.942 s`（action 4 为 `.420–1.253 s`）；整步成本常超过预算。此处“物理不足”只在冻结三圆几何/速度/原始轨迹合同内成立，不是实测 RSU 网络事实。
3. **保守整步回滚**：8/12 action 4 native 准备可落入接触，4/12 native 预览还会给出迁移成功，但完整当前节点服务跨出当前圆。单边跳过 raw full-step gate 会把跨接触继续服务当作合法，属于未建模物理行为；当前不能作为修 bug 直接实施。
4. **公开估计/消费**：6 次 action 0/3 的公开 contact 乐观误差是预测模型限制；action 4 的 12 次失败无需该误差解释。两公开规则的 action 4 `raw_full_step_contact_fit=no` 却仍选择，属于可最小修的 consumer 问题，但不解除主机制阻断。

**唯一下一物理实现建议**：先冻结逐阶段执行合同——当前/目标 RSU 的真实或声明模拟覆盖、模型与状态传输的可中断/原子边界、节点计算跨接触后的服务归属、commit/rollback 及成本记账；所有 learned 和公开规则共享同一可见输入与动作语义。合同和合成反例测试通过后，再另立预登记的新开发来源可达性与公平训练任务。本轮不修改 raw 环境或论文。B 可另行做公开规则 consumer 的最小修复；该局部修复不充当物理合同或训练门通过。

## 未覆盖和主张边界

无正式 checkpoint、匹配训练、formal/holdout/support 原始结果、独立窗口统计、真实 RSU 网络/无线/队列或跨边界相位引擎；paper-ready/优秀基线排名/SA 贡献均 `Unverifiable`，评分 `N/S`。旧 exact-clone planners 仍是 privileged reference；本阶段诊断只使用公开 estimator 做标签对照，实际 contact 和 native preview 始终留在 privileged 诊断输出。
