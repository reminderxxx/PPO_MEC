# 给 CSCWD 2027 A 论文线的主张与证据变更说明

## 身份

- `reviewed_at`: `2026-10-09`（Asia/Shanghai）
- `literature_cutoff`: `2026-10-09`；未新增文献检索
- `target_venue`: `CSCWD 2027`
- `artifact_run_id`: `calibrated_workflow_service_reward_learning_diagnosis_20261009_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_review`: `70a83afb5d3d3b3fb40fcaeaf825a82e4b74c416`
- `evidence_level`: `E2_ARTIFACT_AUDITED_NONFORMAL_OFFLINE_DIAGNOSIS`
- `verdict`: `UNVERIFIED_FOR_CONFIRMATORY_BASELINE_CLAIM`

本说明是可独立合并的论文线输入，没有编辑或并发修改主论文稿。

## 需要收缩的主张

1. 旧 v70 full/mixed 结果继续只作描述性开发观察。A 线审查已经确认计划 Peachtree、实际 Lankershim、原始窗口重叠和
   SA/PPO/MAPPO/DT 训练与选模不匹配；本轮没有修补或重解释旧结果。
2. 2026-10-06 calibrated service-aligned 奖励通过目标排序，但三种 learned method 均有服务退化；它不能写成目标对齐成功、
   SA 稳定领先或算法贡献。action 4 减少不等于完成率提高，no-aux 负结果继续保留。
3. 本轮只支持一个**学习环节候选**：service completion 大回报下 critic 尺度失配与共享全局裁剪可能阻碍 actor 信号。
   这是待证的稳定性改进，不是 SA 专属机制、新颖性或部署收益。
4. two-step rule 保留，但须明确它 clone exact environment transition，枚举前两步，并按 completed nodes、service failure、
   deadline、elapsed、bytes 作字典序选择；其 model-based 能力和目标权限强于 learned scalar actor，不得称 matched-capacity
   learned baseline。

## 可保留的证据

- calibrated run 的动作/状态/奖励账本可精确重放：360 episodes、3,161 steps、0 transition/reward mismatch；90 个 checkpoint
  SHA-256 与原 integrity 一致。旧 checkpoint 只留本地，未上传。
- 当前版本 termination、final observation bootstrap 和 episode-local GAE 接线未发现实现错误。truncation 训练/评价语义有
  边界错配，但 incidence 低，不能定为服务退化主因。
- service-aligned 九个 learned cells 的 last-update explained variance 近 0，固定 dev 梯度中 value:policy ratio=
  `539–7,907`、estimated clip scale=`.00116–.00147`；这足以支持最小 A/B 的动机，不足以支持算法效果。
- 固定 checkpoint 状态前向显示部分 action-4 概率随训练下降但 argmax margin 未翻转；这支持“信号不足/不稳定”的候选解释，
  但不能替代新训练因果检验。

## 后续统一版本和强基线边界

下一次可执行候选应以 `calibrated_workflow_interface_v2 + service_aligned_v1 +
independent_heads_executed_env_v2` 为唯一接口/奖励/动作身份，并整合 commit `6efdde1` 的 frozen mobility window identity
校验或其审计后继；实际 scientific commit 在集成后重新冻结，不能预写。旧 v70 不与该版本合并排名。

学习稳定性 A/B 只含 SA、controller-level MAPPO、PPO，唯一变量为 PopArt critic target/output normalization；配置冻结在
`configs/experiment/calibrated_workflow_value_normalization_ab_v1.json`。它不承担强基线排名主张。

若 A/B 通过后另立论文比较，至少纳入：

- SA-GHMAPPO、PPO、controller-level MAPPO；MAPPO 不冒充 vehicle/RSU-level MARL；
- project-native `dt_handoff_drl`：使用预测 RSU sequence/dwell/confidence/uncertainty/future load/boundary pressure，须标为
  literature-inspired baseline，不是某一论文的精确复现；
- Popularity：显式披露其 stateful adapter count 和 prediction-aware prepare/handoff 规则；
- exact-transition two-step planner：单列 model-based 表，不作 matched-capacity 声称。

当前 registry/单步 probe 能构建上述 learned/heuristic agent 并输出合法动作，但 10 月 6 日 service runner 只接入
SA/MAPPO/PPO/two-step；DT 与 Popularity 尚未接入该训练、选模和 artifact 消费链。该接线缺口必须在独立实现任务解决，
不能在本轮诊断中顺手扩建。

## 数据、预算和晋级门

当前 calibrated manifest 的 36 个实例虽在原始 interval 上两两不重叠，但已全部被 train/dev/regression/frozen-check 消费，
只能作 development。confirmatory 必须把历史 consumption registry 与 10 月 6 日以后新产物取并集，重新冻结 train/dev/
evaluation 的原始 frame/time interval；任何已消费窗口不重新命名为 holdout。

强基线学习方法建议统一为 seeds `7/17/29/43/61`、每 cell 1,440 environment steps、24 update opportunities、每次 60
transitions、4 PPO epochs、batch 32、192 optimizer steps，以及 updates `6/12/18/24` 四个等量选模机会；不做方法专属
超参数搜索。规则方法不训练、不复制成 seed 扩大样本量。若某实现无法满足相同 observation/action mask、交互、更新、搜索
和 checkpoint opportunity，先标 capability mismatch，不运行排名。

共同主表以 on-time/total completion、unfinished-after-deadline、failed-service、连续无进展为主要服务指标；完成条件下时延
必须带覆盖率，再报告 model/state/input bytes、recompute、cache miss/eviction、训练/推理成本和 reward。外层统计单位必须是
不重叠原始窗口，而不是 seed、workflow 或 window ID。

任一 identity/hash/interval gate 失败、预算不等、缺 command log/manifest/checkpoint hash/raw rows，或开发 A/B 未同时通过
机制和服务否证门，即停止 paper-ready 晋级。不得用最终检查集选 checkpoint，也不得靠新增 seed、调 reward 或 auxiliary
消融寻找 SA 获胜。

## 2026-10-09 启动状态追加

PopArt scientific commit `5ee9f1e8071ad8d9e1992e91564ea32801e3d7a7` 和 create-only preflight 已完成；唯一授权后台启动在 run root 创建前结束，日志为空，scientific steps/updates/checkpoints/evaluation rows 均为 0。未重试。因此 A 论文线没有获得新的 A/B 性能、机制或排名证据；上述 `UNVERIFIED_FOR_CONFIRMATORY_BASELINE_CLAIM` 不变。独立状态报告为 `calibrated_workflow_value_normalization_ab_launch_status_20261009.md`，未编辑主论文稿。

后续新授权 v2 已完整结束；本段只保留 v1 历史。最新 A-line 输入以 `calibrated_workflow_value_normalization_ab_claim_change_20261009.md` 为准，overall=`FALSIFIED_OR_NOT_PROMOTED`，仍不支持 formal/paper-ready 或 PopArt 服务优势。
