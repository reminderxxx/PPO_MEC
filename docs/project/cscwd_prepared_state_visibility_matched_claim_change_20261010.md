# Prepared-state v4：主张与证据变更说明（2026-10-10）

本文件仅供 A 论文线只读消费；没有编辑论文主稿或论文表。

## 可新增的受限事实

- 共享 v4 观测 contract 消除了一个可复现的 prepared-state 公共状态别名；四 learned 方法同权消费，旧 profile 保留。
- 在已暴露 development frozen windows 上，SA selected completion `39/40→40/40`、on-time `5/40→9/40`，共同完成
  39 对 elapsed `-32.90 s`、recompute `63.93→31.92 s`、transfer `682.92→94.27 MB`。
- 固定 update-96 的 SA completion 同为 `39/40→40/40`，但 on-time `11/40→11/40`、failure episode
  `9/40→9/40`；selected failure episode 反而 `8/40→13/40`，尽管失败尝试总数 `30→19`。
- 成本下降出现在 PPO/MAPPO/DT，不能包装成 SA 专属机制优势。

## 必须禁止的主张

- 禁止“v4 已稳定改善服务”“SA 稳定领先”“prepared-state 字段验证了算法 novelty”或“paper-ready”。
- 禁止只展示 selected on-time 或 conditional latency，省略 fixed-96、failure episode、共同完成 coverage 和总失败次数。
- 禁止把 seed×window 行当独立 source 样本；禁止用 regression/frozen 重选 checkpoint。
- two-step 必须继续标注 exact-transition model-based 与目标函数权限；规则本轮没有重评。

## 当前证据状态

接口纠错可保留；性能候选按预注册双视角 gate 拒绝。下一算法候选必须来自独立只读失败链诊断；尚无证据授权删除整个
auxiliary loss、改变 reward、强制 action、追加 seed 或扫描超参数。

后续只读事件链已把近端失败定位为 current bundle missing 时选择不修复当前服务的合法动作；没有确认 env/ledger bug。
实际生效 event auxiliary label 未检查 action4 的当前服务可行性，但只有 7 个不同公开失败状态且 action branch=`0/24`。
因此只能记录一个 service-feasible target 候选，不能写成已验证算法改进、消融结论或论文贡献。

`reviewed_at=2026-10-10`；`literature_cutoff=2026-10-10`；`target_venue=CSCWD 2027`；
`artifact_run_id=cscwd_causal_prepared_state_visibility_matched_20261010_v1_analysis_v2`；
`policy_version=docs/project/top_journal_review_policy.md@f46ec72`；
`evidence_level=L2_complete_development_artifact_no_independent_test`。formal/holdout/independent-test=`Unverifiable`。
