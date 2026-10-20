# 5C零环境数值资格合同

- frozen_at: 2026-10-20；用户“继续下一步”承接前次明确提出的有界数值资格。
- algorithm_source: `2c59719d31c49faaff2158bdd94bcd429008f174`；实际runner由独立clean runtime commit及runner_binding.json绑定。
- artifact_run_id: `sa_credit_numerical_stage5c_20261020`；机器合同SHA `0503b913a610f8efe18cdcaf8416c1bea42ac20b6a0815c164c8c2d1cc853f1a`。
- 这是软件数值/梯度资格；不授权环境、科学训练、历史run恢复、候选算法或实际行为学习run。

## 固定矩阵和费用

三seed7/17/27 × 原版/分组裁剪各固定64 checkpoint，共六组日志off/on私有副本；第七组仅seed7原版的私有更新logit压力fixture，测试原KL拒绝。每组两副本，共最多14个model实例。使用已完成5B相应dev_046 snapshots中九分位索引0/4/8，含idle，不补选。非idle恢复该帧首first-pass捕获的控制memory，idle以相同reset状态测试；这是独立冻结输入评分，不是环境连续回放。

所有副本均使用合成监督：每帧行i的外部returns=8×(0.5+(i mod7)/4)，team target=returns/8；model/state辅助常量0.02/0.03。它们不是原训练MC、原stochastic标签或新性能结果。每次原update最多2epochs。第七组只在update期间向原Categorical的logits加candidate_index×4，两个副本完全相同；不改变生产源码，不生成替代轨迹。

| 独立硬限 | 上限 |
|---|---:|
| model实例 | 14 |
| 原_evaluate调用 | 1024 |
| 内部nn.Module forward调用 | 100000 |
| Categorical sample调用 | 2048 |
| backward调用 | 32 |
| 主/副Adam.step合计 | 64 |
| 私有合成update调用 | 14 |
| 环境构造/reset/step、autograd.grad、科学训练 | 0 |

预算在任何tensor加载/model计算前冻结。所有原调用先扣计数再执行；失败尝试不退款。默认日志单组10000行/64MiB上限，不因限额失败静默丢行。首次资格失败保留claim、counter、partial、failure并停止；本claim不修复后恢复/重复。

## 验收与证据

CPU单线程；同一组off/on从相同冻结checkpoint、输入和三类RNG开始。逐项精确比较动作/rollout/原forward输出、梯度、参数/buffer、两Adam state及学习率、controller、更新统计、序列化载荷和Python/NumPy/Torch RNG。仅日志附加UID及临时observer绑定不属于原输出。

日志on必须保持first执行/second强制重评及实际cache/priority采样顺序；actual mask、padding、raw/normalized credit、MC/10/8、KL与两optimizer计数由独立consumer检查。私有比较原件和日志保留；源/旧资产SHA前后复核。预期六组至少执行一个真实私有梯度步骤；压力组必须KL拒绝且0Adam。源码旧baseline的完整AST一致沿用上一轮独立证据，本轮不另构造历史代码模型。

`bind`先纯只读校验并冻结runner/source/已有资产；`run`只执行一次固定矩阵，exclusive claim拒绝重跑；`status`只读。具体命令及实际结果在独立回收审查中列出。日志资格通过也只适用于本CPU有限输入，GPU、长run容量/耗时、真实stochastic长期信用和退化因果不自动获得验证。

## 晋级边界

本次不修改算法、reward、mask、clip、预算日程、GRU、数据或科学checkpoint。原版SA、固定64、全部整体/seed7/27保护及seed17自身初始化恢复要求保持；分组候选仍不晋级。formal/holdout封存、canonical不变，提醒保持PAUSED。

完成后先独立软件证据审查，再更新长期记录并scoped commit/push。正式论文证据仍E0_UNAVAILABLE / Unverifiable，N/S；沿用literature_cutoff2026-10-17，不作新颖性刷新。若未识别一阶算法缺陷，下一步应提出最小可检验缺口或明确停止，不因工具通过直接启动学习。
