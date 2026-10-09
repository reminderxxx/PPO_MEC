# CSCWD 因果预测与强基线开发实验冻结合同

## 审查身份与结论边界

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09；本轮未检索新文献
- `target_venue`: CSCWD 2027（拟投）
- `artifact_run_id`: `cscwd_causal_strong_baselines_dev_20261009_v1`（仅开发，启动前为计划 ID）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: 以启动 ACK 和 `run_manifest.json` 的完整 SHA 为准；协议在启动前提交
- `evidence_level`: 启动前为 `E1_SOURCE_AND_CONTRACT_AUDITED`，结果仅能在原始 artifact 完整性通过后评价
- `verdict`: `DEVELOPMENT_ONLY_AUTHORIZED / PAPER_READY_UNVERIFIABLE`

旧 v1 强基线配置因实际未来 RSU 回退仍被 fail-closed 门阻断，历史产物不改写。新 v2 配置指向同一个已消费的 36-instance manifest；其 SHA-256 为 `b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc`。这 36 项只作开发，`regression` 和 `frozen_check` 都不能重新称为独立 holdout。

## 预测与物理合同

`prefix_transition_counts_v1` 只读取训练 split 的 RSU 序列拟合 `(当前 RSU, 连续驻留步数≤4)→下一 RSU` 计数。选择最高频转移，同票按 RSU ID 排序；支持数小于 2 时返回 `unknown`、空序列、零 confidence 和零公共 contact budget。最多递推配置的 horizon；不读取目标实例的未来后缀、总序列长度、design ID、block/pressure 标签或真实 link。训练实例用其余 11 个训练实例留一拟合；dev/regression/frozen_check 用完整 12 个训练实例拟合。模型不加网络或训练超参搜索。训练 source projection hash `c386f0b55a3b82788b9a1792e5841face968516fc725a9d4ca3d6558088034ec`，完整模型 hash `f382b8de8ee8ed8ee3bb60c39acd485fb67d16c4e43bf2f5db6a38c03a37765b`，full+leave-one-out bundle hash `e3de9c40580427abc493bb2e58beddc044d9eb5690477b24c2ed9307b7a6dd75`。配置只提交这些 hash，拟合计数和原始数据留在本地运行时内存，不上传。

新 `calibrated_workflow_interface_v3_prefix_only` 的 observation、`semantic_state.predictions`、contact/dwell、mask 和所有 learned/Popularity 共同使用前缀预测。每个决策保留 prefix 边界、prefix hash、predictor hash、预测序列、confidence、support 和 unknown 标记。旧 `mean_speed_proxy` 是整窗聚合，新版公开 speed 设为 0（缺测），不把完整窗口统计量当决策时点速度。物理执行的接触窗口由实际迁移轨迹判定，仅 real `step()` 使用；decision clone 使用公开预测窗口。此项是独立接口/物理语义修复，旧新版差异不能归因于算法。two-step 仍可复制环境并看到下一步精确转移，是拥有环境模型能力的比较规则，不称同信息能力 learned baseline。474 个原始实例×决策前缀的后缀篡改比较覆盖公共 observation、semantic state 和 mask；保留单元反例。

## 固定开发比较

唯一条件为 `original_reward_v1`、raw critic、SA-GHMAPPO、controller-level MAPPO、PPO、项目原生 DT-Handoff-PPO × seeds `7/17/29/43/61`。每 cell 精确 1,440 环境步、24 更新、60 transitions/更新、4 PPO epochs、minibatch 32、预期 192 optimizer steps；合计 28,800 步。dev 在更新 `6/12/18/24` 选择一次 checkpoint，不读取 regression/frozen_check；方法专属搜索为零。Popularity 每实例重置；two-step 每实例一次；规则无训练、checkpoint 或 seed。20 个评估实例×(20 learned cells+2 rule 方法)=440 原始行。单一 persistent supervisor、2 小时上限、一次启动、无自动重试；结果目录 create-only。

自动分析首先核对 440 行、28,800 training signals、20 cell、原始 ID 与 predictor hash，再给出逐 split/方法/seed 的按期/总完成、失败与失败尝试时长、无进展、所有实例成本、完成样本耗时与 coverage、双方完成子集配对、模型/状态/输入字节、重算和训练/决策耗时；保存逐步预测误差与 unknown。同一窗口不同 seed 不增加独立样本量，不把开发均值/CI 当投稿主表。未完成样本不进入条件耗时，coverage 必须并列。PopArt A/B 的负向结论不改变本条件；不得因结果补 seed、切 reward、改 predictor、调参或重启 run。

贡献冻结为可审查的系统机制候选：跨 RSU 连续 DAG 的 adapter cache 与状态迁移协同接口、物理与预测信息边界、失败区域和开销账本。尚无强基线优势、独立确认 split、真实无线或 paper-ready 证据；只有本次开发结果满足服务、机制和成本联合判据且另获独立来源确认，才可考虑更强主张。历史旧 v1 的泄漏结果不得并表或解释为算法提升。
