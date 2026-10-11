# CSCWD 开发原始窗口资格审计（2026-10-11）

## 结论与范围

按扫描前提交的 `cscwd_development_window_eligibility_plan_20261011.md`（commit `1c18b43`），本轮只检查 v28 strict split 中已经授权、非 sealed 的 20 个 train 和 20 个 dev 原始区间。**40/40 均不具备现有完整 DAG 的最短执行时间；合格 0，门禁 `NO_QUALIFIED_DEVELOPMENT_INTERVAL`。**本轮因此停止在来源资格层，0 个可达性 episode、0 训练步、0 个 checkpoint、0 次 formal/hidden 结果读取。该结论既不评价 SA 与强基线优劣，也不构成论文贡献。

| 元数据 | 值 |
| --- | --- |
| `reviewed_at` | 2026-10-11（Asia/Shanghai） |
| `literature_cutoff` | 不适用；本次未作文献或 novelty 审查 |
| `target_venue` | IEEE TMC（项目目标；本次不判投稿资格） |
| `artifact_run_id` | `cscwd_development_window_eligibility_20261011_v1` |
| `policy_version` | `tmc_review_policy_v3_20260621`（仅作 claim 边界检查） |
| Git commit | 事前方案 `1c18b43`；审计代码/报告见本文件所属提交 |
| evidence level | 来源资格机器清单已直接核验；完整论文 claim 为 `Unverifiable`，不能将本资格清单当作 E2 正式比较包 |

## 原件、计算与结果

- 只读原始 NGSIM CSV 11,850,526 行，SHA-256 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`；父 workload manifest SHA-256 `b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc`；v28 split manifest SHA-256 `246fcacebb04051e5181a1b2eacf702e942532bbf69e5a4c2e6d1194e98ee322`。train/dev 计划各做 SHA 和 window ID 身份校验；formal/hidden 仅消费 split manifest 中 `window_ids` 的区间元数据，hidden 仍 `sealed=true, opened_at=null`。
- 机器清单：本地 create-only `artifacts/analysis/cscwd_development_window_eligibility_20261011_v1/eligibility_manifest.json`，SHA-256 `e0e3dfdbeb35da5f99a4aa00f985983273efe364c5daa`。清单只含身份、帧计数、时长与理由，不含坐标、策略输出或真实数据行；`artifacts/` 忽略 Git，不上传原始文件。
- 40 个区间均有 24 个连续原始时间戳，源时长 2.3 s，首帧作历史后可决策 2.2 s；同段授权区间之间的最小间隔 2.5 s，没有可按规则拼接的区间。与 formal/hidden 元数据区间重叠 0，最近边界间隔 2.5 s。固定车辆规则选第一帧最小数值 `Vehicle_ID`：14/40 在原窗口完整、26/40 不完整；Peachtree 的 2 个短 `Global_Time` 区间另标单位身份待核验。车辆连续性是附加过滤，不影响全部窗口的时长失败结论。
- 36 个现有开发 workload 的纯计算和范围 16.298588–52.032198 s。即使传输、接触、失败和迁移成本全为零，2.2 s 仍比最短 DAG 少 **14.098588 s**；全部 40 个候选触发这一严格必要条件。事前静态工程目标 118.8 s / 1,189 个 100 ms 帧，比现有最长授权区间多 **116.5 s / 1,165 帧**。它含最大计算和、一个最大冷 bundle、一个输入包、一个正常状态包、一次 fallback 和 10% 工程余量；不是原始轨迹测得的服务保证，`state_scale=1,000,000` 的极端迁移另需单独控制。

## 下一步的独立授权方案

若负责人要继续原始时间强基线，应另行冻结**新的 development 原始区间授权**：明确 source segment、闭区间 `Global_Time`、车辆选择规则及与既有 train/dev/formal/hidden 的排斥边界；只接受原始 100 ms 连续、同车 `Frame_ID +1` 的至少 1,189 帧区间。先在未看策略结果的前提下固定候选排序、最多 12 个实例，以及资源宽裕/竞争和接触充足/不足控制；任一门失败保留拒绝记录，不滚动加长至成功。需要先核验 Peachtree 时间单位与 formal/hidden 原始区间身份。**本轮没有为寻找长窗口扫描或启用未授权 NGSIM 区间，也没有创建新 split。**

在来源资格与 B 的公共 phase estimator/executor 一致性两门均通过前，四方法×五 seed 的条件训练保持关闭。若未来获授权，可达性预算上限仍为 60 episode / 1,440 实际步 / 5,000 preview；训练预算上限仍为 23,040 步，必须另有预冻结协议与独立复核。原 `raw_ngsim_event_time_v1` 的几何、成本、奖励、mask、整步接触门均未修改，旧未来接触 clone 权限问题仍是独立 blocker。

复现命令（输出 root 必须不存在）：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/audit_cscwd_development_window_eligibility.py \
  --raw-csv-path '/Users/howen/Projects/PPO_MEC/data/raw/mobility/ngsim/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260329.csv' \
  --output-root artifacts/analysis/cscwd_development_window_eligibility_20261011_v2
```

复算会生成不同 `created_at`，其余来源资格字段应相同；不得覆盖 v1 原件。正式 checkpoint、command log、formal/holdout/support 原始结果尚不存在于本任务，故论文结论和 TMC-ready 判断为 `Unverifiable`，不做评分或 baseline 名次。
