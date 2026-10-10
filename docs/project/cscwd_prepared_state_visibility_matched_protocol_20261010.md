# 共同 prepared-state 可见性修复：匹配训练协议（2026-10-10）

## 状态与目的

本协议已冻结科学变量，A 实现交接、B 独立前置门禁、冻结 runner 提交与 push 均已通过；现以独立配置提交
**授权一次启动**：`execution_authorized=true`、`authorization_state=authorized_after_independent_preflight`。授权不包含 retry、
追加 seed、边训练修复或结果导向扩展。

已确认的 first-order 问题是：相同公共 observation/semantic/mask、相同合法 action 的状态可具有不同
prepared-state readiness 与 DAG 重算成本。该别名对所有 feedforward learned methods 共享；它不证明该缺口单独造成
SA 与 PPO/MAPPO/DT 的全部差距。唯一干预是将当前 RSU 和公开预测目标 RSU 的 prepared-state prefix validity/freshness
作为共同公共字段；不是 SA 专属能力或算法创新。

## 原件绑定

- 历史科学 run：`cscwd_causal_strong_baselines_budget_extension_20261009_v1`，commit=`d25ebcd`；20 cells、
  115,200 steps、15,360 optimizer steps、400 selected evaluation rows，119-file integrity 完整。
- 历史后处理：`cscwd_causal_strong_baselines_budget_extension_analysis_20261009_v2`，commit=`c276929`，17-file
  integrity 完整。
- 定位报告：`cscwd_sa_long_budget_cost_diagnosis_20261009.md`，SHA-256=
  `5e26f74493673344fb07f19b8cbdb4f23447dca9d80e68579326cea5fc86e1f9`。
- 精确文件 hash、预算和字段列表见
  `configs/experiment/calibrated_workflow_prepared_state_visibility_matched_protocol_v1.json`。

历史 selected 400 episodes 只按 hash 复用，不重评。历史 update-96 checkpoint 可只读新增评价，不训练、不改选；所有
checkpoint 保持本地，不进入 Git。

## 单一变量与独立验收

A 线只能新增 current RSU / public predicted target RSU 的 prepared-state prefix validity/freshness，并显式启用新 profile。
B 线在训练前独立验收：

1. 字段只来自已完成 DAG prefix、当前 RSU、已执行 migration/prepare 历史和决策时点公开预测；不得读取实际未来 RSU 后缀。
2. 改写未来实际轨迹不改变新字段；新 prefix、节点进展、prepare 失败、RSU 切换、缺失和过期均有负例。
3. 同一动作下 reward、transition、cost 与旧 profile 一致；字段只改变 policy input。
4. SA/PPO/controller-MAPPO/DT 四方法都实际编码并消费字段，不接受只出现在 semantic JSON 中。
5. 旧 profile output/checkpoint contract byte-equivalent。若新 encoder 输入维度导致参数量变化，逐方法披露参数增量和推理
   接口；不得称严格纯信息单变量。

任一门禁失败即退回 A 线。本轮禁止边训练边修复。

实现身份为 `709746bc1f1ea3037f27497bdb51cf6f45c8963c`；交接测试依赖修正为 `517053949d916c476843f48769799e3854810d32`。
B 已复验 43 tests、474 个 public-prefix 后缀篡改点、20 个历史 update-96 checkpoint 与 40 条规则来源；预检为 0 训练、
0 新评价。新输入投影参数增量为 SA `+192`，MAPPO/PPO/DT 各 `+448`，因此不作严格纯信息单变量主张。

## 固定训练与选模

- 新 profile：四 methods × seeds `[7,17,29,43,61]`；每 cell 5,760 environment steps、96 updates、768 optimizer
  steps；总计 115,200 / 1,920 / 15,360。
- 60 transitions/update、4 PPO epochs、minibatch 32；candidate updates=`[24,48,72,96]`。
- `original_reward_v1`、raw critic、Adam learning rate `.0003`、clip `.2`、entropy `.01`、value `.5`、gamma `.99`、
  GAE lambda `.95` 固定；SA auxiliary `.1`，其他方法 auxiliary `0`；不启用 PopArt。
- dev 与 `_selection_score` 完全沿用；regression/frozen 不用于选择、停止或失败重试。
- 单次 durable launch，2 小时上限，无自动重试、补 seed 或结果导向追加预算。

## 预冻结的两个报告视角

主视角为同一 dev 规则各自 selected checkpoint：历史 selected 原件 vs 新 profile selected。辅助视角固定 update 96：历史与新
profile 都评价 update-96，不得在结果出来后选择更有利视角。新增评价上限严格为 1,200 episodes：历史 update-96 400、
新 selected 400、新 update-96 400；每 episode 仍为 24-step cap。若 selected 恰好为 update-96，仍保留角色标签并在汇总
中披露重复 checkpoint 身份，不把重复评价当独立样本。

Popularity 与 two-step 只引用历史 40 行；two-step 继续标注 exact-transition model-based 与 lexicographic objective 权限，
不为本轮重新评价。

## 指标与否证

首要指标固定为 completion、on-time completion、service failure。必须共同报告逐 seed/原始 window、配对覆盖、共同完成
elapsed、DAG recompute、model/state/input bytes、vehicle fallback、prepared-valid/invalid handoff 与 freshness 失效分解；
reward 单列，不能代替服务收益。五 seed 复用相同 window，不能当独立新数据。

报告必须区分：四方法共同接口收益、SA 旧→新变化、SA 相对 PPO/MAPPO/DT、以及 SA 相对有额外 model-based 权限的强规则。
如果别名未消除、SA recompute/逾期未改善、任一方法服务或成本边界恶化，或收益只来自 selected/fixed 两视角中的一个，
则拒绝候选。无优势如实报告，不改 reward、数据、selection 或追加训练追胜。

## Claim 边界

全部 36 实例已暴露，只能形成 development evidence；formal/holdout/independent-test 均为 `Unverifiable`。本轮不改论文、
论文表或主稿。`reviewed_at=2026-10-10`，`literature_cutoff=2026-10-10`，`target_venue=CSCWD 2027`，
`artifact_run_id=cscwd_causal_prepared_state_visibility_matched_20261010_v1`，
`policy_version=docs/project/top_journal_review_policy.md@c276929`，预执行 `evidence_level=L1_FROZEN_PROTOCOL`。
