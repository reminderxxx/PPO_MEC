# CSCWD 强基线开发链最小接线

## 身份与边界

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09；本轮未检索新文献
- `target_venue`: CSCWD 2027（拟投）
- `artifact_run_id`: `none`；只做接线、合成 fixture 验收与只读 preflight
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_review`: 实现基线 `858bc797e23b4e56051663f28d6fd7681f5ee77d`；本轮实现提交见 Git
- `evidence_level`: `E1_INTERFACE_IMPLEMENTED_SYNTHETIC_SMOKE_ONLY`
- `verdict`: `INTERFACE_WIRED / PREDICTION_PERMISSION_PREFLIGHT_BLOCKED / ALGORITHM_ADVANTAGE_UNVERIFIED`

**后续预检勘误（同日）**：完整预测权限审计发现当前 36/36 实例把实际未来 `rsu_sequence` 回退为公共预测；
新增 fail-closed 门阻止本设计启动。下文预算与命令只保留为未执行草案，结论以
`cscwd_2027_strong_baseline_prediction_permission_blocker_20261009.md` 为准。

本轮在独立分支以 B 的 `858bc79` 为基线，保留原有冻结两奖励 runner 和 PopArt A/B runner。新增
`scripts/run_calibrated_workflow_strong_baselines.py` 作为一次开发比较的公共入口，默认配置
`configs/experiment/calibrated_workflow_strong_baselines_development_v1.json` 中 `execution_authorized=false` 且
`scientific_execution_authorized=false`，所以 `--run` 在创建 output root 前拒绝。只读 `--preflight` 检查既有
36-instance manifest 与 base config 的 hash、原始 segment/frame/time 身份、互斥区间和 12/4/12/8 split 数量；
这 36 项已用于开发，不能当独立确认性测试或新 holdout。来源区间门通过之后，新增预测权限门会拒绝
这些实例；接线验收不等于科学可执行。

## 能力与公平性矩阵

| 方法 | 可用观察/决策能力 | 训练、状态和结果身份 |
| --- | --- | --- |
| SA-GHMAPPO | 公共 9 值 observation、`semantic_state`、五动作 mask；图/多头/辅助目标按当前构造器实际生效配置 | 同一 learned 预算；独立 seed、4 次 dev 选模、selected checkpoint 与 hash；参数量/实际 optimizer steps/耗时单列。 |
| controller-level MAPPO | 同公共信息；cache/execution/event controller heads 与 centralized flat critic | 同一 PPO 基类和预算；不称车辆/RSU 级 MAPPO。 |
| PPO | 同公共信息；flat encoder 和单策略 | 同一 PPO 基类和预算；不能把网络结构差异抹成相同算力。 |
| `dt_handoff_drl` | 公共 `semantic_state.predictions` 的 next/target RSU、序列、dwell、confidence/uncertainty、future load 与边界量；同五动作 mask | 项目原生、受文献启发的 digital-twin PPO，不称原论文精确复现。沿用同一 PPO 基类、交互/更新/选模门，逐字段能力披露。 |
| `popularity_cache_heuristic` | 公共节点/RSU/预测与 mask，adapter 计数到阈值 2 可预取 | 每个实例新建 agent，计数只在该 workflow 内累积；无学习、无 checkpoint、无训练 seed。结果 `seed=rule`，行为 ledger 保留规则原因。 |
| `two_step_cost_rule` | 现有前两步 exact-transition clone 与字典序规划 | 维持模型转移能力标签；无学习/seed/checkpoint，不称同能力 learned baseline。 |

公共 wrapper 在调用策略前检查 9 值观察、`semantic_state` 和五动作 mask，调用后检查动作合法；
它不改 actor、reward、环境、五动作或旧 SA/PPO/MAPPO 参数。DT 通过既有 `_build_agent` 获得与
SA/MAPPO 相同的 `independent_heads_executed_env_v2` 构造参数。训练采样复用 B 的精确 transition batch 与
episode-local GAE；开发选模仅看 dev 的预定服务指标，不使用 reward 数值、regression 或 frozen_check。
评估复用公共 `_evaluate_agent`、`_evaluate_rule`，新增规则逐实例重置；原始 row 和行为 ledger 补充
`source_segment_id`、frame offset/length、time start/end、window/workflow ID。已有动态 summary/seed
consumer 读取六种方法的同 schema 行，规则不进入 seed summary；不由这里计算或宣称统计显著性。

## 下一轮待冻结预算及实际差异

当前**未批准的设计草案**为 raw critic、`original_reward_v1`、learned seeds `7/17/29/43/61`，每方法每 seed
1,440 环境步、24 次更新机会、每次 60 transitions、4 PPO epochs、minibatch 32，预期 192 optimizer steps；
在 updates `6/12/18/24` 各给一次 dev 选模，方法专属超参搜索 0。四种 learned 方法都继承同一 PPO 基类；
runner 对每一次实际 optimizer-step receipt、参数量、环境步数、更新数和耗时核对并保存，发现不匹配即失败，
不为对齐数字暗改任何算法。若后续另接非 PPO 算法，须单独冻结交互/选模/search 与优化差异协议。

若上述设计获独立批准，一种 reward profile 对应 4×5=20 learned cells、28,800 环境步、480 更新机会、
预期 3,840 optimizer steps、80 个候选与 20 个 selected checkpoint；regression 12 + frozen_check 8 的
development evaluation 预计 learned 400 行、Popularity 20 行、two-step 20 行。规则只执行每实例一次，
不复制成 5 个训练 seed。所有窗口仍是开发消费区间，不因新接线升级为独立测试。PopArt 默认 disabled/raw；
非 raw 仅留显式 opt-in，须先有另行冻结的晋级决定。本轮不选 PopArt，也不把它写成有效贡献。
B 的开发 A/B 已执行完，本分支尚未独立审查其科学原件或将其结果并入主稿。

只读检查命令（现预期因预测权限非零退出）：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/run_calibrated_workflow_strong_baselines.py --preflight
```

**开发比较命令草案，当前不得执行**；需先单独审查 B 的科学结论、冻结此设计和 Git commit，明确授权字段及
唯一新 output root，再替换占位值：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/run_calibrated_workflow_strong_baselines.py \
  --run \
  --config configs/experiment/calibrated_workflow_strong_baselines_development_v1.json \
  --expected_git_commit <FROZEN_SCIENTIFIC_COMMIT> \
  --output_root artifacts/experiments/<NEW_DEVELOPMENT_RUN_ID>
```

### 本轮验收和未覆盖项

- 合成 fixture 的 4 个 DT transitions、1 次 PPO update、4 个 optimizer-step receipt 仅验证公共链和
  checkpoint save/load/hash；**科学训练步数为 0**，不产生算法比较。
- Popularity 在两个不同合成实例独立执行，重跑动作分布一致；规则的训练 step/checkpoint 计数为 0。
- 原三学习方法的旧 service/PopArt 局部测试与新测试共同通过；旧 runner 输出和科学历史未覆盖或改写。
- 尚无真实开发比较；完整预测字段权限审计已发现阻断，训练/推理耗时实测表、跨 run/车辆独立确认性
  split、正式 checkpoint/command/manifest 或论文主表统计。它们是下一次科学执行与投稿的独立门。
