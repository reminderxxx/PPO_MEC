# 因果强基线统一预算延长协议（2026-10-09）

## 问题与单一干预

本轮仅检验既有因果前缀实现 `a08869388f9962149198054b5f8a0d6dd4d07eae` 在统一增加学习预算后，四种
learned method 的开发指标与学习曲线是否改变。它不是“训练到 SA 获胜”的搜索，也不改变 reward、raw critic、
observation/action contract、网络、optimizer、seed、数据或评价实例。

唯一联合变量为 `uniform_learned_method_training_budget_and_proportionally_scaled_selection_schedule`：每个 method×seed
由 1,440 环境步、24 update、192 optimizer step 延长到 5,760 环境步、96 update、768 optimizer step；候选点随预算
从 `[6,12,18,24]` 等比例移到 `[24,48,72,96]`。因此任何变化必须表述为“预算 + 等比例选模时点”的联合效应，
不能单独归因于训练步数。

## 冻结矩阵与预算

- learned methods：`sa_ghmappo`、`mappo`、`ppo`、`dt_handoff_drl`；seeds：`[7,17,29,43,61]`。
- 每个 cell：60 transitions/update、96 updates、4 epochs、minibatch 32、768 optimizer steps。
- 共 20 cells、115,200 环境步、1,920 updates、15,360 optimizer steps；所有 cell 从初始化重新训练。
- reward=`original_reward_v1`，critic target normalization=`raw_disabled`；禁止 PopArt、reward/cap/search、额外 seed、retry。
- 每个 candidate 只用共同 dev selector；`regression` 与 `frozen_check` 均为已暴露 development，不能用于选择、停止或
  independent-test 声明。
- 固定学习诊断在 update 24 与 96 的 dev rows 分表保存；selected checkpoint 结果与固定端点结果不得混表。

## 短预算与规则结果复用

短预算来源固定为 `cscwd_causal_strong_baselines_dev_20261009_v1`。启动前必须逐文件验证 completion、manifest、
evaluation rows、training summary 与 integrity 的预注册 SHA-256；身份不符即停止。Popularity 与 two-step 只引用该 run
的 40 行既有结果，本轮不得重新评价。two-step 始终标注 exact transition model 与 lexicographic objective 权限；其结果
不是同信息权限 learned baseline。

## 分析与停止

必须完整运行所有 20 cells，不因中间或最终 evaluation 选择、提前停止或追加训练。输出至少包括短→长预算的
method×seed 配对表、方法汇总、selected checkpoint 服务表、update 24/96 固定 dev 表、四候选选模曲线、96-update
optimizer 曲线及 completed-common-subset elapsed 覆盖率。若 SA 未改善，如实记录并停止，不自动提出或运行新算法。

一次 persistent launch，wall cap 2 小时，无 retry。checkpoint、真实数据和行为大表保持本地并由 Git ignore 管理；Git
只提交协议、代码、测试、摘要和小型证据。不得编辑主论文稿、A 审查目录或旧 artifact。

## 主张边界

本轮至多提供“开发数据上的预算敏感性”证据；不支持 formal、holdout、泛化、显著性、paper-ready、SA 优势或机制因果
主张。`reviewed_at=2026-10-09`，`literature_cutoff=2026-10-09`，`target_venue=IEEE TMC`，
`artifact_run_id=cscwd_causal_strong_baselines_budget_extension_20261009_v1`，
`policy_version=docs/project/top_journal_review_policy.md@a088693`，预执行 evidence level=`L1 contract`。
