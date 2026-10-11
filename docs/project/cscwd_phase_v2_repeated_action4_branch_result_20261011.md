# CSCWD phase-v2 第二次 action4 有界分支确认结果（2026-10-11）

## 身份与判定

- `reviewed_at`: 2026-10-11；`literature_cutoff`: 2026-10-11（沿用同日公开近邻增量检索）；`target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `cscwd_phase_v2_repeated_action4_branches_20261011_v1`；事前方案与脚本 commit `da2c8b7474b0f06d8c9f1fa0a6389c95faf85714`。源 pilot commit `5a79ffd018d4c260bc7a8417229a440c7d591fe4`。
- `evidence_level`: 仅为 **development 局部分支原件与只读复核**；formal/holdout/support 缺失，论文层面 `E0_UNAVAILABLE / Unverifiable`。局部结果不计作独立来源级统计、不晋级算法/优秀基线、不改论文主表。
- 结论：三个事前固定、按条件去重的 US101 公共起点，第二次 action4 在相同公开 continuation 下均比合法 action3 多传字节而不改善服务；但这不是可直接部署的“见到上次迁移成功便阻止下一次 action4”规则。**当前不修改生产策略**。

## 1. Artifact 完整性、protocol 与执行范围

先前[事前方案](cscwd_phase_v2_repeated_action4_branch_plan_20261011.md)从 B 已冻结 `behavior_rows.csv` SHA-256 `b3975c212bf5e0e9d9f444e2f32367f530a27f84b3f0ec494384f45ea94b0eef` 注册 11 条 `sa_candidate/selected/v2/us_101` 的“step0 action4 交付成功、step1 action4 目标先离开”记录。按同一原始来源、条件、工作负载和前缀动作去重为 **3 个起点**（default 3、adverse_link 3、ample_cache 5 个 seed 记录）。这是回顾性选择的开发问题，不能将 11 或 3 当成独立 mobility cluster。输入 protocol、base config、phase 环境/来源 SHA 锁定；只读扫描原始 NGSIM 11,850,526 行，重建 US101 固定的 1,189 帧/118.8 s 轨迹，未下载或覆盖数据。

一次 create-only 运行从冻结 clean commit 启动，`start_receipt.json` SHA-256 `beb6acb8a415e70ccb5bf61f74e41dcd0a9bcbef22e6478bfefbc2329bf90073`，`completion_receipt.json` SHA-256 `d84ef73379d1e10e45be396b924db6d16c63ce3943dd39a0c9fe6a40b73a0566`，status=`COMPLETE`。实际 **3 个共同前缀 step + 6 个分支各 7 step = 45 个真实环境步**，低于 24 分支/192 步硬上限；0 训练、0 checkpoint 加载、0 formal/holdout 结果读取、0 自动重试。共同首步和 action4 分支首步的动作、字节、时钟、迁移状态及 reward 均与原 pilot 行为一致。分叉前的 `semantic_state + action_mask` 哈希三条件分别为 `5581e312…`、`dd37f05b…`、`e1477d1f…`，两首动作均合法；两个离线分支从相同环境状态复制，但在线选择只看共同公开状态和后续 mask。

本地原件 `artifacts/analysis/cscwd_phase_v2_repeated_action4_branches_20261011_v1/`：`summary.json` SHA-256 `be57ec84b234e31e9866d768b095d2b25ce58645c7f1c8bc179ff27aee8e9a1d`、`paired_branches.json` `2dc40ac12d649dbd0086e1acc4551e03654d3e078483eee61b98f9b7bb92d39a`、`event_ledger.jsonl` `53c089ac328bf288bd77a0d6c135dfc9f4176dc6ce031afb5157b7220e4e19bc`；完整性清单 `2e34a14a8e4037aaf040da7ed5597ed1c1d31f20db5904c27f204cbf8548dcd6`。独立只读审计 `artifacts/analysis/cscwd_phase_v2_repeated_action4_branches_audit_20261011_v1/audit.json` SHA-256 `175bc27a84b9e41d823c1a84aadb2f9862a545511f88391fee37b66da3982f0c`，核对 5/5 payload SHA/size、45/45 phase 时钟/字节/action mask、共同公开起点、六分支汇总和三对差值，`all_checks_passed=true`。

## 2. 三个配对起点的全量结果

表中是**从共同前缀之后**到 DAG 结束的后缀；每分支 7 步，首动作后统一规则为“action3 合法则 3，否则 2”。三个条件的 action4 首步均完成当前节点，但目标 `rsu_2` 在状态交付前离开；其后六步两分支均仅能选 fallback 并最终完成。所有分支按期、0 服务失败、0 后缀迁移成功、0 重算、0 partial bytes。以下差值为 **action3 − action4**，不是跨 seed 平均值。

| 条件（原 pilot 重复行） | action4 后缀总字节 | action3 后缀总字节 | 省模型字节 | 省状态字节 | action3 耗时差 | 服务差 |
|---|---:|---:|---:|---:|---:|---|
| default（3） | 12,240,075 | 2,457,651 | 9,641,944 | 140,480 | −0.411255 s | 完成/按期/失败均相同 |
| adverse_link（3） | 12,240,075 | 2,457,651 | 9,641,944 | 140,480 | −0.724293 s | 完成/按期/失败均相同 |
| ample_cache（5） | 2,598,131 | 2,457,651 | 0 | 140,480 | −0.025536 s | 完成/按期/失败均相同 |

action3 的后缀总字节在 default/adverse 各少 **9,782,424 bytes**，ample_cache 少 **140,480 bytes**。输入传输三个条件均相同，action3 不用为 `rsu_2` 准备本节点模型，也不发送失败的 140,480-byte 状态。默认/adverse 的第二次 action4 为目标模型准备额外传 9,641,944 bytes；ample_cache 已在所有 RSU 驻留该模型，故只节省状态字节。三条件内 action3 的完成/按期/失败均不退化且总字节严格减少，因此满足**预先冻结的局部成本支配判据**。原奖励差分别为 `+0.037456/+0.062499/+0.002108`（action3 − action4），仅反映此物理成本链，不替代服务主指标。

## 3. 公共可识别性、机制与不修改结论

分叉前公开状态显示当前为 `rsu_1`、预测目标为 `rsu_2`，两动作 3/4 均合法；`prepared_state_prefix.current.valid=true`，而 `predicted_target.exists=false/valid=false`。也就是说第二次 action4 是**向新的预测目标准备**，不是向同一 RSU 重发完全相同状态。B 原 pilot 的 public immediate/two-step 在这三个条件同样选 action4，说明当前公开估计器在这些起点预期有准备价值；目标先离开是执行后才观测的真实结果。现有三起点不能证明“current state 已有效”可安全作为一般 abstain 门，亦不能排除其它轨迹中第二次 action4 确有价值。

本轮确认了**这条有限开发轨迹上的真实开销浪费**，但没有建立跨来源、跨 target-delivery 结果的公开判别能力。若现在按事后失败标签直接禁止 action4，会把隐藏未来信息倒灌决策，也可能破坏成功迁移。因此生产算法、reward、action mask、估计器与训练均保持原样。下一轮若另立任务，应先在独立开发来源上冻结并校准仅用观察前缀的目标交付可行性，再做包含成功链和失败链的同能力配对；本轮不启动该工作。B pilot 原 `MIXED` 和外层 terminal `FAIL` 不因局部分支变更，paper-ready 仍 `Unverifiable`。

## 4. 已执行验证

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q tests/test_cscwd_phase_v2_repeated_action4_branch.py tests/test_cscwd_phase_v2_pilot_independent_audit.py
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m py_compile scripts/confirm_cscwd_phase_v2_repeated_action4.py
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/confirm_cscwd_phase_v2_repeated_action4.py \
  --pilot-protocol /Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/configs/experiment/cscwd_phase_v2_causal_public_pilot_v1.json \
  --pilot-run-root /Users/howen/.codex/worktrees/causal-budget-extension/PPO_MEC/artifacts/experiments/cscwd_phase_v2_causal_public_pilot_20261011_v1 \
  --output-root artifacts/analysis/cscwd_phase_v2_repeated_action4_branches_20261011_v1
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/audit_cscwd_phase_v2_repeated_action4.py \
  --source-root artifacts/analysis/cscwd_phase_v2_repeated_action4_branches_20261011_v1 \
  --output-root artifacts/analysis/cscwd_phase_v2_repeated_action4_branches_audit_20261011_v1
```

两个 output root 均为 create-only；已执行的一次真实运行不能原地重试或覆盖。未覆盖风险：仅一个曾用于 checkpoint 选择的 US101 原始来源、回顾性失败起点、固定 serve/fallback continuation、无成功第二次迁移的配对反例、无独立 formal/holdout；因此不能外推成可部署控制规则或算法贡献。
