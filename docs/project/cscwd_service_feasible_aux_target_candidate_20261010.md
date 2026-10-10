# 当前服务可行性一致的 event auxiliary target 候选（2026-10-10）

## 状态

`FROZEN_CANDIDATE_ONLY / execution_authorized=false`。本文件不是实现或训练授权。A 的只读事件链支持“实际生效标签与 action4
当前服务条件存在语义缺口”，但未执行动作分支或训练消融，尚不能证明它单独造成 SA 与 baseline 差距。按照一次只改一个因素
的边界，本轮停在候选说明，不盲目训练。

## 已确认的近端失败机制

- new-v4 selected/update-96 的 SA 首次服务失败分别为 `32/100`、`26/100` episode；全部发生在当前 bundle 缺失时选择
  不修复当前 bundle 的合法动作，action4 分别占 `28/32`、`23/26`。
- 失败 action4 中 `42/46`、`32/36` 已成功在预测目标 stage bundle，但 action4 同时要求服务当前节点；当前 bundle 缺失使
  当前服务失败，因此 prepared state 不 commit。这是合法动作语义下的策略次优，不是已确认 env/ledger bug。
- SA 的 `auxiliary_coef=0.1` 实际生效；3,840/3,840 optimizer step 的 auxiliary loss 和加权辅助梯度均非零。评估使用
  `raw_policy`，没有额外运行时 mechanism logit bias，故影响只能通过训练权重形成。
- 冻结的 24 条首次失败探测全部 current bundle missing；22/24 的当前实现生成 `event_target=1`，21/24 实际执行
  action4，22/24 的 action4 probability 高于 action0。样本只有 7 个不同公开状态，不能当 24 个独立样本或因果证明。

## 当前实现与唯一候选

当前 hard event target 为：event head 启用、公开预测 handoff target 与当前 RSU 不同、目标 RSU 已缓存 required adapter，且
`prepare_window_score >= temporal_prepare_activation_threshold` 时取 1；它不检查当前 RSU 是否已具备执行 action4 当前服务所需的
完整 bundle。

唯一候选只修改 `_build_mechanism_targets` 的 event label 生成：

```text
original_event_target = existing target formula
action4_current_service_feasible = current RSU has the complete current-node bundle
candidate_event_target = original_event_target AND action4_current_service_feasible
candidate_event_soft_target = original_event_soft_target if feasible else 0
```

不得同时改变 slow/fast target、`auxiliary_coef`、各 loss weight、网络、reward、temperature、action mask/guard、运行时 logit、
optimizer、数据或预算；不得移除整个 auxiliary loss。该候选只是机制一致的监督标签，不是 SA 专属信息或新增动作权限。

## 必须先通过的对称反事实门禁

实现或训练前必须另立 create-only preflight，并冻结所有抽样与分母：

1. 明确动作语义：action0 填充当前 bundle 并服务当前节点；action2 以 vehicle fallback 服务当前节点；action4 只准备预测目标
   的 model/state，同时依赖当前 bundle 服务当前节点。当前服务失败时目标 model cache 可变化，但 state 不 commit。
2. 对确定性去重的首次失败公共状态执行环境 clone 的合法 action0/2/4 一步分支；候选只允许作用于“action4 当前服务必失败，
   而至少一个 action0/2 当前服务成功”的状态。若 action4 可完成当前服务，候选门禁失败。
3. 对称正例必须覆盖 current bundle ready + target/timing 成立，保持 `event_target=1`；负例覆盖 current missing、target missing、
   timing 未达阈值、未知目标和 action4 masked。只看失败负例不足以授权训练。
4. 记录 action0/2/4 的当前步服务、target staging、state commit、clock、model/state/input bytes 和 reward 分项。若 bounded suffix
   显示“立即失败的 action4”仍稳定具有更低预声明多步服务代价，则否定把 hard target 置 0 的候选。
5. feature flag 关闭时 target 与 checkpoint 行为 byte-equivalent；打开时除 event hard/soft target 外的训练字段逐项相同。

目前 A 执行的 action branch=`0/24`，因此以上门禁尚未完成，训练保持未授权。

## 门禁通过后的唯一 A/B 草案（未授权）

- 只训练 SA：seeds `[7,17,29,43,61]` × 5,760 steps=`28,800`，96 updates，候选 updates `[24,48,72,96]`。
- 精确复用本轮 v4 原 SA 为 control，以及同 v4 PPO/MAPPO/DT/规则结果；不得重训其他方法。
- 主 selected + 辅助 fixed-96；新增评价最多 200 episodes。相同 reward、raw critic、数据、优化器、预算和选模规则。
- primary：completion 不下降、on-time、failure episode；joint：失败尝试、recompute、model/state/input bytes、共同完成 elapsed
  与 coverage。低传输伴随更多失败不算改善。
- 若 completion 下降、on-time/failure 不在两视角保持方向、只个别 seed 改善，或任一身份字段漂移，则否定候选并停止；不自动
  找第二个改法、追加预算或筛 seed。

全部实例仍为已暴露 development；即使未来 A/B 通过，也不能直接形成 formal/holdout、稳定领先、novelty 或 paper-ready 主张。
