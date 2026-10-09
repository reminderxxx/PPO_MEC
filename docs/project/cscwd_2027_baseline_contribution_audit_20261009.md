# CSCWD 2027 强基线与贡献证据审查

## 审查元数据

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-09-28（沿用既有近邻文献审查；本轮未评价新文献）
- `target_venue`: CSCWD 2027；TMC 审查政策仅作项目内部严谨性约束
- `artifact_run_id`: `top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_review`: `3881271`；主工作树另有七个受保护的用户修改文件
- `evidence_level`: `E1_PLUS_RAW_PARTIAL`。直接读取了 checkpoint manifest、mixed/full aggregate rows、episode receipt 和统计 JSON；缺完整 command log、全包完整性审计以及可接受的独立窗口统计。
- `verdict`: `UNVERIFIED_FOR_CONFIRMATORY_BASELINE_CLAIM`。旧奖励结果只作描述性开发观察。

本轮是独立只读审查，没有训练、重跑、算法修改、holdout 开启或结果筛选。旧产物不覆盖、不重命名为新实验。

## 原件与一阶问题

原始根目录：`artifacts/experiments/top_journal_closed_loop/top_journal_mechanism_v70_sparse_tail_option_formal_min_20260730/`。四个关键文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `seed_checkpoint_manifest.json` | `c54ef4df8bd8ac2685769b5f6c0613c95088a946f9ed77c79a97e21c688b41a6` |
| mixed `aggregate_summary.json` | `aa521256fb3e68ad2d23f22d56db876851347adfa9d13ddd225ae0b0b4ab667b` |
| full `aggregate_summary.json` | `f5fc9d6c08d3a7ce92fcd46f2f74b0eb1588b98bfb57f565011c2e87285d6a2e` |
| full `paired_statistics.json` | `778c3edd489ce1d5a1cc587436934fa67747b8fdc36122cc7f122fa2014db990` |

1. **窗口身份不一致。** mixed 的 20 个、full 的 48 个 `selected_window_plan` ID 全为 `peachtree`；实际 rows 和 episode receipt 全为 `lankershim`，计划与实际 ID 交集均为 0。相同 frame offset/长度不等于相同原始时间区间。`scripts/benchmark_main_results.py` 传入 `expected_window_id`，但 `src/evaluators/main_results_support.py` 在没有 `formal_window_consumption_contract_path` 时走普通加载分支，不核验此 ID。本轮只定位，不修复。
2. **窗口重叠。** 按原始 segment/frame 闭区间复算，mixed 计划有 61 对重叠，full 计划有 177 对；实际 Lankershim full rows 的 ID/offset 也有 177 对。full 的 48 个窗口只形成 9 个彼此断开的重叠区间组。原统计把 `window_id` 当作 48 个外层独立 cluster，其 BCa CI 与 Holm p 不能作为独立窗口推断；mixed/full 也不能合并增加样本数。
3. **基线训练不匹配。** full rows 中 SA 的 `checkpoint_run_update_count=16`，PPO、MAPPO、DT 均为 12；SA 是 `best_by_reward.pt`，三者是 `latest.pt`。manifest 明示这些基线继承 v69/v67/v55 checkpoint 池。旧比较是已训练配置比较，不是同预算算法归因实验。
4. **业务结果边界。** full 每个方法有 288 行；SA、DT、PPO、MAPPO 的 workflow 完成率均为 `0.958333`。SA/DT 的连续性为 `0.904454/0.904920`，平均 backhaul cost 为 `137.5/137.055556`，平均完成时延字段为 `1374.95833/1373.91667`。SA reward `32.385729` 高于 DT `31.426667` 和 PPO `27.301597`，但不能推导出完成率、时延或流量全面领先。

## 固定的候选贡献及晋级证据

| 候选贡献 | 可写的实现/观察 | 必要的新证据 |
| --- | --- | --- |
| C1：跨 RSU 连续 DAG 中的 adapter/base 依赖、准备时机和 workflow state 问题 | controller-level 原型已实现；共享缓存四配置与两节点后缀恢复分别有 bounded diagnostic。 | action 4 真实 producer/consumer 成本与相同完成量对照；独立机制机会上的 sharing × migration 2×2。62.2% 共享量减少不得归因于 SA 策略。 |
| C2：工作流感知的三头 SA-GHMAPPO 准备控制 | v70 旧开发协议有描述性 reward 排名和 sparse-tail prior；身份是单 controller 的多策略头。 | 同观察、动作、训练/选择预算的 PPO、DT、两步确定性规划；关闭 sparse-tail prior 的匹配消融。若主张预测贡献，还需门控消融与预测校准。 |
| C3：服务质量与成本共同报告 | 旧 rows 有完成、连续、时延、命中、backhaul 和迁移字段。 | 身份和独立性修复后，按同一 paired unit 报告全部指标、原始 rows、命令、checkpoint hash 和完整性账本。 |

这三项仍是**待证候选贡献**，不是三项已验证的论文贡献。世界模型和轻量部署仅作未来方向。

## 下一轮实验的预注册边界

以下阶段按顺序执行；失败即停止 confirmatory 晋级，不能通过调参或筛窗掩盖：

1. **身份修复。** 另立实现任务，使普通 frozen-plan 加载也校验 `window_id`、source segment、原始 frame/time interval、长度和数据身份；错段、错 offset、错长度必须在 rollout 前拒绝。同步检查计划生产端、benchmark 与统计消费者。旧 JSON 只读保留。
2. **独立数据冻结。** 先列出已消费 NGSIM 区间，冻结新的、同 split 不重叠且有机制机会的窗口；train/dev/evaluation/任何 holdout 按原始 interval 互斥。运行前固定 window manifest/hash、科学 commit、seeds、workflows、容量、主 endpoint、缺失值规则、统计 family 和 checkpoint selection。找不到足够独立窗口时写 `inconclusive`，不以 seed/workflow/窗口 ID 数充当独立样本。
3. **同条件强基线。** 在同一冻结环境、请求、观察信息和 `semantic_discrete_5` mask 下比较 SA、PPO、DT、Popularity，以及读取相同剩余工作流信息的两步确定性规划。学习方法分别重训；统一 interaction、更新机会、checkpoint 选择机会和搜索预算，并公开实际 update/gradient-step 数、参数量及训练/推理时间。短期规划不能额外读取未来真实轨迹。
4. **消融与共同主表。** 同预算重训 `no_sparse_tail_prior`；预测门控确实启用且有足够机会时再做 `no_reliability_gate`。主表同时列 workflow completion/failure、continuity、handoff readiness、完成条件下时延、adapter/base load/transfer、backhaul、状态字节、miss/eviction 与 reward。以不重叠原始窗口为外层配对 cluster，seed/workflow 为内层；报告 percentile/BCa CI、window-level sign test、Holm 校正及全部负结果。

这是实验设计边界，不表示新实验已执行、可用 holdout 已建立，或旧 G14 fixed-commit continuation 获准。后者的 `execution_authorized=false` 独立保持。
