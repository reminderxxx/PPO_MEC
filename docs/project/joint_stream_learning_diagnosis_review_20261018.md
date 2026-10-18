# 第二阶段独立审查：共同学习链与同状态行为

- reviewed_at：2026-10-18；精确UTC时间见机器报告。
- literature_cutoff：2026-10-17（沿用已检索范围；本阶段没有新增文献或novelty评定）。
- target_venue：IEEE TMC内部审查规范，不是录用判断。
- artifact_run_id：`learning_diagnosis_stage2_20261018`；原run `20261016T162100Z_confirmation`；阶段一 `learning_effectiveness_stage1_20261017`。
- policy_version：`tmc_review_policy_v3_20260621`。
- algorithm Git：`35975cb3d73c84ddbabf01f6749bbc2d60c826de`；阶段一评估Git：`4823a1a05ddbd174c3c5079b7b934f46c4d183f6`；本阶段起点Git：`4ee7fd0504c46da0fe12b00039026a3200f47df9`。
- evidence level：局部开发参考轨迹/事件/收据精确重建及独立张量、源码、原日志审查；**正式论文 E0_UNAVAILABLE / Unverifiable**。不能以局部E3式复现替代formal/holdout/support证据。

## 1. 完整性与执行边界

合同[joint_stream_learning_diagnosis_contract_20261018.md](joint_stream_learning_diagnosis_contract_20261018.md)在参考执行前冻结，机器合同SHA为`cde19bc74093926ef9ed5f3ce4150185972fcaec05046548dd2c0e9e56225109`。固定阶段一预列的15个见证，只运行各自最终64模型一次；实际 **15会话 / 7032评估物理帧**，低于7200上限。DQN短案例312帧，其余480帧。新增训练、完整update、optimizer.step均为 **0**；没有算法或reward修改。

123个科学源码指纹、protocol/assets、原8组manifest/dev和48个checkpoint文件SHA核验通过。15格的完整summary字典、逐条canonical events及receipts与原64结果全部精确一致；不是仅奖励相等。采集2160个forward输入（首遍/次遍各1080），保存完整decision、候选、memory/calendar、实际mask、参考prefix和MC标签。所有0/32/64/零head评分输入一致，参考64逐调用重算bit-exact、参数/RNG不变。实际梯度批处理带浮点归约差，64最大logprob误差`2.384185791015625e-7`，在预设容差内。

输出根目录`artifacts/joint_stream_learning/learning_diagnosis_stage2_20261018/`含contract、freeze_receipt、run_started、verification、15组参考档、result/progress、independent_p0、score_analysis、cached_gradients和delivery_validation。原科学run、阶段一、最终checkpoint不覆写；正式split未打开、canonical不变。

## 2. 已确认的一阶问题：DQN回放插入方向

`src/agents/dqn_agent.py`的Stream update为寻找未来决策而逆序遍历轨迹，却直接将transition append到`deque(maxlen=4096)`。当四会话新包超过容量时，FIFO最后保留较早决策、丢掉较新决策，发生在随机采样之前。容量需要丢弃数据本身不是缺陷；**倒序构造和FIFO保留方向相反**才是缺陷。

原源码纯transition构造前缀、默认容量4096和4097条合成时序的反例保留索引0..4095、丢掉最新4096及唯一terminal。没有调用完整update或优化器。对原manifest核验过的DQN训练日志精确重建：

- 48/48次更新新包均溢出。
- 550283条当前决策transition中，353675条在首次学习采样前失去资格。
- 192个训练会话中90个会话的全部当前决策均被淘汰。
- 第64会话seed7/17/27的2047/1905/1798条当前决策均保留0条。

未来return和next-state payload仍可含后续信息，不能写成“后期所有奖励都没有学习”。原实际随机采样索引未记录，本结论是确定性的资格覆盖重建。它不能量化对DQN144/144退化的因果贡献，不能保证修复后胜出。历史结果完整保留，但**不得用当前SA–DQN排序归因主方法机制优势**。独立P0报告在新增参考轨迹前已写入`independent_p0/`；本阶段没有顺手修复。

## 3. 未复现的错误与仍开放的风险

八配置纯合成检查：behavior/current条件logprob、首遍/次遍execution概率误差均0，ratio=1。源码表明PPO保存的prefix和mask能在同参数下正确重算；真实开发状态参考64也通过。原训练完整行为张量没有保存，因此不把此结果冒称全训练概率逐步重放。

团队MC为物理尾回报/10，各车辆信用和value/Q目标再除以物理车辆数8，没有发现重复除8或单位错配。surrogate两个费用输出是单独的链路帧/10辅助标签，不重复扣进reward/logits。DQN不溢出的跨idle、换请求、会话终止合成边界守恒；不能用这些通过项掩盖已确认的溢出覆盖问题。

固定团队尾回报是否提供足够动作区分信用仍开放；参数变化、loss下降和概率一致不能证明学到了有效因果信用。

## 4. 同状态评分：确有动作变化，早期cache覆盖不足

全部比较固定参考64状态/候选/上下文。首遍只看execution；次遍teacher-force参考64完整prefix，保证cache/event条件和mask一致。这是条件评分，不是各checkpoint自主策略的反事实收益。0点仍含公共先验、mask、协调、固定GRU和随机残差，不是随机策略或纯先验。head置零仅为软件控制；cache/event没有数值公共prior，最低有效索引破同分不能称强规则baseline。

| pass/head | 总行 | 单选强制 | 多选 | 0→64动作变更/多选 | 64覆盖prior或首索引/多选 |
|---|---:|---:|---:|---:|---:|
| 首遍execution | 1080 | 890 | 190 | 62/190 | 59/190 |
| 次遍execution | 1080 | 890 | 190 | 62/190 | 59/190 |
| 次遍cache | 1080 | 1072 | 8 | 6/8 | 3/8 |
| 次遍prepare/priority | 1080 | 0 | 1080 | 792/1080 | 288/1080 |

首/次遍execution是同一控制决策的两次计算，不能合并成独立样本。SA seed17 dev046 execution的14个多选中5个相对0点改变、9个覆盖公共prior；相同案例seed7为7/13改变，seed27为4/11改变。learned residual确实能改变候选排序，但成功/失败对照都有改变，不能把“覆盖prior”直接判成错误。

固定最早9帧全为frame0..8：任意head多选即符合采样条件，priority选项足以让早期入选。cache只有8条自由多选，不能靠本样本解释整480帧的cache策略或后期反复加载；不得追加挑选“更显著”的状态。prepare/priority头的变化也不能直接解释成预测预复制或迁移收益。完整逐见证分头、margin和最早分歧见`score_analysis.md/json`。

## 5. 梯度：共同裁剪耦合已兑现，退化归因未成立

源码把critic、prefix_values、action_values与策略头/encoder同置于非surrogate参数组，统一clip norm0.5。`actor_gradient_norm`因此不是纯actor-head梯度范数。surrogate有独立裁剪/优化器；费用surrogate不直接改变greedy logits。

本阶段仅在选中真实开发状态、固定greedy参考动作及事后MC标签上计算PPO形式loss的`autograd.grad`。greedy底层categorical分数不是实际随机行为概率；0/32比值不是有效off-policy估计，本结果也不是原训练梯度或Adam更新模拟。

13个非DQN见证×0/32/64共39组；DQN2格明确跳过PPO梯度。最终64状态批中，policy norm为0.019376..0.043663，value_half norm为0.854577..14.270074，共同裁剪系数为0.035038..0.584322。价值回归主导该诊断batch的共同范数确有证据，策略头同时被缩小；全部参数、optimizer状态、grad buffer、RNG及输入不变。

**反例必须保留**：同dev046，正向SA7/27裁剪系数0.039279/0.036398，比退化SA17的0.188760更小。因此“裁得更重导致seed17更差”不成立；Adam也可能部分抵消统一缩放。本阶段不能直接选“分组裁剪必然改善”为修复结论。下一阶段先加入可观测分组梯度和独立软件合同，若要实验改变需另设预算和公平比较。

## 6. 物理链条与反例

从原结果独立读取54555条最终64事件，固定15格的节点/选站变更/state bytes/model bytes/reward共75项对账0误差。公共execution prior已经包含queue、剩余状态恢复、冷模型/在途加载和peer模型族容量冲突，不能称为“没有考虑状态/cache费用”。

SA17 dev046：304次选站变更、117次同站同base重新准入、99.7901%准入字节为reactive；98次同request/node在旧站付state费用后尚未compute便离站，31次在新站8帧内出现reactive重复base准入。对应SA7/27为38/55次付费离站、4/8条链；cache17为78/10，DAG7为67/12。成功对照同样有这些片段，时序不等于可避免损失，不能全部记为浪费。

相对精确初始化的关键反例：

| SA17案例 | 完成节点变化 | 选站变更变化 | 模型传输量变化 | 状态传输量变化 |
|---|---:|---:|---:|---:|
| dev022（双family） | −62 | +63 | +117 KiB | −902 B |
| dev046（双family） | −28 | +40 | +62.5 KiB | −13896 B |
| dev017（单family） | −25 | +89 | 0 | +103854 B |
| dev020（单family） | −9 | 见机器报告 | 略降 | 略降 |

单family dev017/020的base重准入均0。说明需同时考虑选站/状态连续性、排队与结果送达、family驻留和预取站点寿命，不能只改模型重载或把总state费用上升当统一解释。optional-use和净交付仍需保留此前的对象使用/驻留与实际compute边界，本阶段没有新增预测器因果消融。

## 7. 独立结论与第三阶段任务点

**第二阶段完成：确认DQN回放方向一阶缺陷；在已审计软件合同与固定参考状态内未复现所列概率/单位/边界错误；确认条件动作变化与裁剪耦合；SA17完整退化的唯一因果解释仍未建立。** 未证明收敛、泛化或GNN/surrogate/预测的因果贡献，未修改最终奖励排名。

第三阶段应先做可验收的最小软件修复，不重训：

1. **DQN chronological replay**：保留逆序构造以连接未来，构造完成后按正时间入队。固定容量溢出保留最新索引、terminal尾部可采样；跨idle/换请求/两session bootstrap边界和MC差分守恒。保持replay容量、目标、prior、mask、费用及所有其他算法不变；新增回归先复现后修复，历史原件不覆写。
2. **共同学习可观测性**：分别记录policy头/encoder/value/surrogate梯度、强制与有效多选head覆盖、reference prior margin；明确不把原`actor_gradient_norm`当纯策略梯度。不因本诊断直接改变裁剪、reward或网络。
3. **SA17后续合同设计**：先预列中后期的选站—付费状态—compute/交付及family驻留见证，并冻结状态采样与分支/评估上限，覆盖seed7/27正例。当前最早9帧cache自由度太低，不能据此直接大改缓存架构。
4. **性能确认门禁**：任何未来训练必须另有明确公平预算、固定最终checkpoint和密集快照日程；当前1/1训练已用完。整体reward≥210.3997145、完成率≥88.8888889%、模型≤96.9375KiB且每完成请求≤6.815918KiB；seed7/27各自全部非回退门禁仍按原preservation plan。此处数字仅作展示，执行时使用原计划完整精度与原浮点容差。不能用seed17改善抵消已有优势损失，不挑峰/拼seed。

本次只交付第三阶段任务点，不自动开始第三阶段实现或训练。已有模型、预测和状态共享均保留。原模型不变可保留原面板数值，但公平修复baseline后是否仍排名第一必须重验，不能承诺或通过保留有缺陷baseline维持名次。正式论文七维均N/S：formal/holdout/support关键证据缺失、DQN baseline覆盖缺陷和机制归因不足；不作paper-ready或novelty晋级。

## 8. 验证与复现

以下命令在任务worktree、项目Python执行；原参考入口有exclusive启动锁，**已完成的目录不可再跑**：

```bash
python scripts/audit_joint_stream_learning_chain.py --output_root <新的只读审查目录>
python -m pytest -p no:cacheprovider tests/test_joint_stream_learning_diagnostics.py -q
python scripts/audit_joint_stream_cached_gradients.py --stage2_root artifacts/joint_stream_learning/learning_diagnosis_stage2_20261018 --output <新的梯度审查文件.json>
```

只读链审查通过、46/46软件测试通过；一次性参考入口15/15 summary/events/receipts exact，39组PPO形式梯度纯度通过。交付清单与文档一致性另见`delivery_validation.json`；Git同步实况见`delivery_status.json`，不以本报告文本假称远端成功。
