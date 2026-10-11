# A 论文线独立主张与证据变更说明：phase-v2 公共候选（2026-10-11）

本说明不编辑主论文稿，只给 A 论文线界定可用主张。

## 可以更新

1. 旧 raw v1 的 full-step current-contact 原子门使迁移机制在已冻结长开发来源上不可达；phase-v2 现在把 input upload、
   source-RSU compute、backhaul model/state、target restore/delivery 显式分相，并在两个真实开发来源通过 reset/step 验收。
2. public estimator/rule 已消费 phase-v2 的公开合同，隐藏实际 future contact 仍不可见；SA/PPO/MAPPO 使用相同 public state、
   mask、reward 与动作权限。
3. 唯一待检验算法候选是 SA 的 public-causal `prepare/serve/abstain` event supervision；schema v3 只反映 phase-v2 输入合同。

## 不可更新

1. 不得写“SA 已优于 PPO/MAPPO/规则”、迁移机制已产生性能收益、或候选 paper-ready。
2. phase-v2 public two-step 在缺少公共二维运动/目标进入时间时退化为 immediate score，不能作为独立两步强基线计数。
3. exact two-step 仍是 privileged model-based reference，只能在 v1 capability view 报告；不得把隐藏 future contact 权限包装为
   information-matched baseline。
4. 两个来源都已用于 mechanism-oriented development，独立单位只有 2；任何 pilot 结果只作描述性开发证据，不做显著性或
   formal/holdout 主张。

## 当前证据变化

`actual-class consumer gate=PASS`，但在一次性 pilot terminal artifact 产生并独立核验前，算法效果仍为 `UNTESTED`、
paper-ready=`Unverifiable`。最终变更只允许引用 create-only run manifest、completion/integrity、selected+update24 双视角和
预注册 verdict；不得以最终评价重选 checkpoint、追加 seed 或调整 reward/auxiliary 权重。
