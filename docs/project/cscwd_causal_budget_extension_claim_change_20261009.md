# 因果预算延长：主张与证据变更说明（2026-10-09）

本文件仅供 A 论文线只读消费；没有编辑论文主稿、论文表或 A 的审查目录。

## 可新增的受限事实

- 在已暴露 development 数据上，四 learned methods 的统一 4× budget run 完整结束；SA completion `87/100→99/100`，
  五 seed 为 `3 positive / 2 unchanged`。
- 相同前 1,440-step 训练前缀得到 20/20 byte-identical update-24 checkpoints；记录差仅 `≤3.55e-15` 浮点文本差。
- SA 的改善不是全面服务改善：frozen on-time `5/40→5/40`，combined service failure 仍为 `.17`，共同完成样本
  recompute 增加 `8.10 s`。
- PPO/MAPPO/DT 同样获得不同收益，因此预算结果不能包装成 SA 独有贡献或领先证据。

## 必须删除或继续禁止的主张

- 禁止“SA 稳定领先”“长预算验证 SA 机制”“更多训练解决服务退化”“已达到 paper-ready”。
- 禁止把 400 evaluation rows 当作 400 个独立样本；五 seed 复用相同 source windows。
- 禁止将 regression/frozen 用于 checkpoint 重选，或把 conditional completed latency 当作全样本 latency。
- two-step 继续标注 exact-transition model-based 权限；本轮只引用旧 40 行，没有重新评价。

## 证据状态

`reviewed_at=2026-10-09T09:22:10Z`；`literature_cutoff=2026-10-09`；`target_venue=IEEE TMC`；
`artifact_run_id=cscwd_causal_strong_baselines_budget_extension_analysis_20261009_v2`；
`policy_version=docs/project/top_journal_review_policy.md@d25ebcd`；
`evidence_level=L2_complete_development_artifact_no_independent_test`。formal/holdout/independent-test 仍为 `Unverifiable`。
