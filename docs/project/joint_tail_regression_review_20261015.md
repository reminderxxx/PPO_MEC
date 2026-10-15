# seed 27 尾部回退与预测消融：独立定位

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：joint_scheduled_control/main/seed27/20261015T053900Z_development；训练Git aa8676a；诊断父Git 3c3a194。
- evidence level：开发固定权重干预原件，论文E0_UNAVAILABLE / Unverifiable。非formal/holdout，无训练/峰值选择。

## Artifact / provenance

原120checkpoint/30模型重放核验保留。新增`artifacts/joint_tail_regression_20261015/audit.json`及`surrogate_heads/audit.json`、逐帧events.jsonl、command log。所有干预使用已冻结seed27权重/8dev，组件替换明确记录；不写checkpoint、不更新参数。所有方法同原环境和费用，未接触formal/holdout。

## 公平性与机制定位

| 固定模型干预 | 八窗奖励 | 完成请求/8 |
|---|---:|---:|
| full96 | 57.009508 | 6.500 |
| full128 | 56.343598 | 6.000 |
| no-pred128（独立训练，非同权重） | 57.674358 | 6.875 |
| full128仅换96 execution head | 56.339848 | 6.000 |
| full128仅换96 priority/event head | 57.745849 | 7.000 |
| full128仅换96 graph | 56.499176 | 6.000 |
| full128关闭全部surrogate logit干预 | 57.837427 | 7.125 |
| full128仅关闭surrogate head0 | 56.343598 | 6.000 |
| full128仅关闭surrogate head1 | 56.343598 | 6.000 |
| full128仅关闭surrogate head2 | 57.837427 | 7.125 |

**已定位的具体原因之一：事件头将准备目标与priority混合编码，surrogate对每个priority候选分别加观测return残差，改变了排队优先级。**它没有按同一prepare target聚合priority；即使没有预测准备，noop的四种priority也被surrogate重新排序。仅关闭head2干预即可复现全部关闭的改善；执行头与图漂移不足以解释该seed的主要损失。不能据此断言所有seed/训练退化只有此一原因。

full128合法分布的每窗logit gap中位数约0.0012–0.0024，很小；global surrogate MSE可靠度不等于动作差值/priority排名的校准。已有固定权重预测开关却为正收益，因此“更多预测信息必然使训练结果更好”不成立，也不能把这次失利归为应删除车辆预测。

## 尾部信用与窗口覆盖

原训练使用gamma=.99而评估为有限时域未折扣总奖励。119帧后的正奖励仅乘约0.3024；代数见证中早期交付1节点的折扣目标1高于末帧交付2节点的0.604809，评估排序相反。这是目标不一致，**尚未证明是本seed每一条坏动作的唯一原因**。全局critic拟合团队return，动作/prefix critic拟合逐请求blend，尺度不同还会共同占用全网络gradient clip；须统一单位并用独立合同核验。

源窗口仅按persistent车辆ID排序取前8，未按迁移机会分层。几何清点（不使用算法回报）原半径为spacing×1.2：20train有29次站覆盖离开，8dev仅4；tight factor .65为42/7，全部仍有至少一个可达站。几何离开不等于实际执行站迁移或有正净收益。需要在原train时间范围内加入更有运动/覆盖变化的车辆cohort，不能从dev/formal按效果选窗口；预测器fit/validation须按原时间区间分组，防止同时间不同cohort泄漏。

## 独立实现任务与claim边界

1. 将surrogate政策干预限定到有净收益的cache/state准备；同prepare target的priority共享残差，noop priority不被其重排。保留GRU和surrogate学习，不用旧模型关闭预测冒充新训练改进。
2. 用共同有限时域未折扣return匹配评估；统一global/per-request critic单位，提供尾部交付见证。原奖励、费用和完成定义保留。
3. 仅从train原区间按外生覆盖变化准备公共训练cohort/场景；冻结原dev与tight压力panel，共同适配全部七算法及no-pred。
4. 各128×3seed、最终128主表，预声明场景权重和support。合理难度用于暴露任务差别，不以算法分数筛场景或削弱baseline。

当前不能声明新实现优势、收敛或cache机制优越性；文献novelty未评估。以上是前置独立审查，后续修复另立实现合同。
