# 论文现有实验奖励波动与训练趋势图

- `reviewed_at`: 2026-10-21（Asia/Shanghai）
- `literature_cutoff`: 2026-09-28，沿用既有审查，本轮未作新文献或 novelty 评价
- `target_venue`: CSCWD 2027；仅整理训练曲线，不作投稿 readiness 判断
- `artifact_run_id`: `top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `062437d`（本轮实施基线；源码本身 hash 记录于派生 provenance）
- `evidence_level`: `RAW_TRAINING_LOGS_RECEIPT_VERIFIED_DESCRIPTIVE_ONLY`
- `verdict`: confirmatory algorithm advantage / convergence `UNVERIFIED`

## 范围与来源

用户要求其他内容先不变，在保留论文算法优势的基础上取得各算法奖励波动/提升趋势图，并希望末段收敛。
本轮先交付现有实验的真实曲线：未改算法、奖励、环境、数据、历史 run、checkpoint、论文正文或 canonical，
未训练、续训、筛窗或开启 holdout。SAGH 标签映射到代码 agent `sa_ghmappo`。

输入是 v70 原 `seed_checkpoint_manifest.json`，SHA-256
`c54ef4df8bd8ac2685769b5f6c0613c95088a946f9ed77c79a97e21c688b41a6`。
按 checkpoint 父目录读取对应 `train_summary.json`，覆盖十个学习算法、seed 7/13/29 共 30 个 run。
SAGH 每 seed 128 episodes/16 updates，基线每 seed 96 episodes/12 updates。训练窗口、采样方式和
checkpoint 选择规则并不一致；不能把同一张图的高低解释为同条件算法优势。

每个 episode 的 `total_reward` 均与原 `episodes/episode_*.summary.json` 中
`reward_breakdown.total.sum` 对账，共 2,976 条；对源 summary、checkpoint、评估历史及所有输出记录 SHA-256，
逐 episode receipt 的有序名称/内容 hash 链也写入 `provenance.json`。不反序列化/执行 checkpoint。
这证明奖励转录一致，不表示旧窗口身份、独立性或整个实验 protocol 已验证。

## 统一绘图口径

- 每个 seed 先计算完整的过去 16 个 episode 的算术平均，再跨 seed 求均值及样本 SD。
- 前 15 个 episode 保留原始波动，平滑值留空，不用未来点、局部短窗口或端点填充。
- 原始 seed 曲线/均值用浅色展示；单算法及总览图使用相同纵轴范围。阴影表示 seed SD，不是 CI。
- 不裁剪负奖励、不独立归一化、不补造等长尾段、不用累计最大值替换训练记录。
- 启发式没有训练过程，不伪造“训练收敛曲线”。基线没有固定窗口逐 checkpoint 日志时也不补造评估曲线。
- SAGH 另绘原始 `update_eval_history.json` 中的 safety-projected 与 raw-policy 两种评估；不平滑、不混合两种模式。
  逐 checkpoint 核对计划窗口 ID/workflow 列表相同，但真实窗口身份仍受 10/09 审查限制。

## 当前能确认的趋势

以下都是记录中的训练回报，块长统一为 16；算法间不可作公平排名。末块与前块指各算法自己训练末尾的两块。

| 算法 | 每 seed episodes | 首块均值 | 末块均值 | 末块减首块 | 末块减前块 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAGH | 128 | 24.3423 | 31.4894 | +7.1471 | +0.6777 |
| PPO | 96 | -2.1569 | 3.7125 | +5.8694 | -3.6098 |
| MAPPO | 96 | 6.1304 | 5.3077 | -0.8227 | -1.8740 |
| DQN | 96 | -13.6460 | -6.2173 | +7.4288 | -4.4875 |
| Dueling DQN | 96 | -21.5194 | -4.0075 | +17.5119 | -3.3994 |
| QMIX | 96 | -0.8381 | 0.9248 | +1.7629 | -4.2300 |
| Controller MAT | 96 | 5.7881 | 2.3233 | -3.4648 | -4.1200 |
| DAG Offload DRL | 96 | 5.4523 | 6.7883 | +1.3360 | -3.3313 |
| Cache Offload DRL | 96 | 7.4829 | 8.1465 | +0.6635 | -0.7671 |
| DT Handoff DRL | 96 | 8.4998 | 9.7931 | +1.2933 | -1.2825 |

SAGH 的训练回报提升、末段块均值变化减小，可描述为“训练末段趋于稳定”，不能表述为已证实策略收敛。
固定评估 safety-projected 三 seed 均值由初始 32.385729 到末次 29.426840，末四次均值为 29.418898；
raw-policy 由 32.821667 到末次 7.368993，存在先下降再部分恢复。后期平台低于早期最佳。
原 `best_by_reward.pt` 保持原样；单纯延长训练没有已验证的优势保持保证。

其他算法仍有奖励波动/部分末段回落，且缺同窗口逐 checkpoint 评估证据；本轮没有实现或声称“全部收敛”。
若下一轮需要改善收敛，应另冻结同条件训练/选择预算、独立窗口和事前停止规则，保留原始 run 并报告所有结果。
本轮不通过挑 seed、改曲线或自动调参满足预设趋势。

## 交付与验证

产物：`artifacts/analysis/paper_reward_curves_20261021_v2/`。
打包下载：`artifacts/analysis/paper_reward_curves_20261021_v2.zip`（包含全部图、CSV 与来源清单，不含 checkpoint 或原始数据）。

- `reward_curves_all_algorithms.png/.pdf`：十算法原始波动、移动平均与 seed SD 总览。
- `reward_trends_comparison.png/.pdf`：全部学习算法趋势叠图，标明训练条件不匹配。
- `reward_<agent>.png/.pdf`：十个算法各一套，保留每 seed 原始线。
- `checkpoint_evaluation.png/.pdf`、`checkpoint_evaluation.csv`：SAGH 两种评估模式的真实历史。
- `training_rewards_raw.csv`、`training_rewards_rolling.csv`、`tail_diagnostics.csv`、`provenance.json`：原值、平滑值、末段描述性诊断和来源。

验证入口：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ppo_mec_reward_pycache .venv/bin/python -m py_compile scripts/plot_training_rewards.py
.venv/bin/python -m pytest tests/test_training_reward_plots.py -q
PYTHONPATH=/private/tmp/ppo_mec_reward_plot_deps .venv/bin/python scripts/plot_training_rewards.py \
  --manifest artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/seed_checkpoint_manifest.json \
  --output-root artifacts/analysis/paper_reward_curves_20261021_v2 --rolling-window 16
```

专项检查覆盖完整 trailing window、无未来泄漏/填充、日志长度、重复/缺失 episode、非有限奖励、seed/agent 错配、
reward/receipt 不一致拒绝，以及即使平坦曲线也不自动判定收敛。9 项测试通过，真实入口对账 2,976 条奖励。
另独立复算全部 992 行平滑数据，验证 30 个导出文件 hash、13 套 PNG/PDF 与 ZIP CRC；
目视检查总览、趋势叠图和两模式评估图，未见遮挡、裁切或曲线丢失。
v1 曾在缺少 Matplotlib 时留下 CSV，未完成出图；保留为失败导出，不作为结果。

未覆盖：重新训练、基线固定评估、策略参数/分布收敛、公平优势、真实窗口身份修复、独立 holdout、paper-ready。
相关旧 blocker 继续以 `cscwd_2027_baseline_contribution_audit_20261009.md` 为准。
