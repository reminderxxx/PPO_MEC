# 第三阶段：SA学习更新候选与中后期诊断交付

- reviewed_at：2026-10-18；机器回执含UTC精确时间。
- literature_cutoff：2026-10-17，沿用已检索范围，本次无新文献或novelty晋级。
- target_venue：IEEE TMC内部审查规范。
- policy_version：`tmc_review_policy_v3_20260621`。
- artifact_run_id：`sa_learning_stage3_20261018`；历史科学run `20261016T162100Z_confirmation`。
- Git起点：`85fb2a205aadc63f4b30b8be15ebdd9dd94f6927`；原科学源码：`35975cb3d73c84ddbabf01f6749bbc2d60c826de`。交付commit与同步状态见阶段`delivery_status.json`。
- evidence level：局部开发参考轨迹精确复现、软件更新隔离与兼容验收；正式论文仍`E0_UNAVAILABLE / Unverifiable`，七维N/S。

## 1. 本阶段结论

按用户最新优先级，SA算法改进与可检验的奖励稳定性为主线，DQN修复暂缓，既有缺陷继续公开。本阶段完成了一个可单独检验的SA候选：**策略和价值网络独立裁剪梯度**。软件验收通过，尚无新的科学训练或性能结果，不能称奖励提升或已收敛。

第二阶段确认的价值梯度/共同裁剪耦合是真实软件机制，正向seed7/27裁得更重的反例仍保留。当前候选是对该机制的单一干预假设，不是已证实的seed17退化修复。

## 2. 唯一opt-in算法改动

`StreamSAGHMAPPOAgent(separate_policy_value_clip=True)`新增到`src/agents/sa_ghmappo_agent.py`：

| 参数组 | tensors | 裁剪上限 |
|---|---:|---:|
| policy：heads、cache/domain/graph encoder、graph message、graph residual gate | 21 | 0.5 |
| value：critic、prefix_values、action_values | 28 | 0.5 |
| surrogate | 12 | 原独立0.5 |

同一次backward、原同一个non-surrogate Adam及其49参数顺序、原surrogate Adam、loss/MC/学习率/entropy/采样/网络/物理reward和费用、预测/共享全部保留。各组互斥且覆盖原参数。独立0.5使main联合范数上界从0.5变为约0.7071；连surrogate的总体上界从约0.7071变为0.8660。它改变更新约束，不能称总梯度预算相同。

默认False直接调用原父类update/梯度路径，不新增旧options键或旧loss字段；既有最终模型不替换。True新增有限numeric分组norm/clip系数，兼容现有逐值isfinite消费者。checkpoint写options=True及`training_gradient_contract=sa_policy_value_clip_v1`，恢复检查flag/tag配对；旧缺字段为False，True缺tag、未知tag、False带tag、非bool flag均拒绝。网络tensor/action合同保持，`training_resume_supported=False`保持；资格中的私有副本算术不是支持训练恢复。

## 3. 兼容、隔离与预算核验

本阶段合同见[sa_learning_stage3_contract_20261018.md](sa_learning_stage3_contract_20261018.md)。科学训练session与科学优化更新均0，原1/1额度已耗尽。软件预算与实际：

- SA专项23项测试通过；23次合成Adam.step、9次合成update，环境调用0，低于128步上限。覆盖默认与父类逐tensor/Adam/grad/result/RNG一致、True推理一致、组覆盖/交叉路由、裁剪独立、全零/None梯度、去预测默认链、KL stop、surrogate关闭、checkpoint负例。
- 原第二阶段46项纯软件诊断回归通过；不执行环境或优化。
- 额外资格：3seed×0/32/64共9状态，每状态parent/False/候选A/B四份私有模型，用原stage2 dev046每seed72行真实缓存tensor。36次_apply、主/辅助合计72次Adam.step，低于128上限；不调用完整update/环境/rollout。
- False与原父类的一步参数、Adam、裁剪后梯度精确一致。A/B仅将value项放大100倍，policy原始/裁剪后梯度、参数和按参数名提取的Adam状态全部精确一致，说明独立裁剪软件隔离成立；不是因果性能证据。
- 所有48旧checkpoint文件SHA、tensor/actor与加载兼容通过，额外6次SA旧checkpoint恢复、3个精确初始化重建通过。123个原Git源码对象匹配原protocol；当前live仅允许SA文件改变，原件/checkpoint/输入/RNG不变。

合计软件Adam步骤95，全部发生在合成或既有缓存tensor的私有副本上；不能将其计作新的科学训练，也不能隐去真实计算量。软件临时模型不晋级、不上传。

## 4. 中后期参考状态补证

在SA源码修改前冻结并执行5个原最终64参考：SA17 dev022/dev046/dev017、SA7/dev046、SA27/dev046。每格480，共**2400评估帧**，无优化。原123源码、protocol/assets、6个SA checkpoint SHA通过；每格完整summary/events/receipts与stage2已对原科学run验证的归档精确一致。

按原事件冻结127个中后期frame，保留全部活跃车辆/双pass，共1948个捕获调用、974个独立execution决策；已完成车辆不补齐。先验分解使用首遍真实reservations/calendar，次遍保存首遍execution_feature，避免用后续context错误解释选站。raw物理mask、去noop、feature mask、actual mask、旧站可用性和承诺年龄分别保存，0/32/64/零head仅同state/reference prefix评分。

| 固定样本范围 | 数量 |
|---|---:|
| 原始物理mask有多站可选 | 974/974 |
| commitment将多站收窄为单选 | 735/974 |
| 最终自由execution选择 | 239/974 |
| 旧站与其他RSU均可选的真正机会 | 148 |
| 同节点换站；其中旧站仍valid | 64；61 |
| cache实际自由选择 | 5/974 |
| head2多选但只在None prepare下选priority | 974/974 |
| 真实prepare RSU机会/实际prepare | 0/0 |

这些frame围绕既有换站事件富集，不能推为整个480帧的总体频率或独立统计样本。974次head2的0→64变化均为priority-only，不能当成预测预复制或迁移改善。

## 5. 对SA方向有用的正负证据

固定样本中37次“同request/node付state费用后、未compute便换站”，均在旧站仍可选时发生，新站的公共prior均严格高于旧站，未出现learned residual严格压过prior的换站；target-memory年龄为8–11帧。负例及成功对照均存在：

| 见证 | 同节点换站 | 其中已付state未compute离站 | 严格覆盖prior的已付换站 |
|---|---:|---:|---:|
| SA17/dev022 | 13 | 9 | 0 |
| SA17/dev046 | 14 | 7 | 0 |
| SA7/dev046 | 15 | 9 | 0 |
| SA27/dev046 | 12 | 6 | 0 |
| SA17/dev017 | 10 | 6 | 0 |

全部64次同节点换站中有2次严格覆盖prior，来自正向seed7/27，且均不是上述已付state换站；不能把37格的0次覆盖写成所有换站都没有覆盖。

因此，这批负例不支持直接压制learned residual。后续应检验8帧commitment边界后公开剩余成本排序、队列估计与实际驻留/交付的兑现；成功例也有相同片段，不能据年龄边界直接推断延长锁定会改善。费用prior已含剩余state/model/queue/peer family项，已支付量不是应重复扣除的未来成本，不能盲目把sunk cost再加进reward或prior。

本阶段没有在看到补证后增加新滞回/锁定/费用项，也未同时修改多个机制。唯一候选仍是事前合同中的分组裁剪，性能假设可被下一阶段否证。

## 6. 第四阶段：如何得到可信的奖励学习曲线

已准备[下一阶段有界配对合同草案](sa_group_clip_confirmation_draft_20261018.md)，当前**DRAFT，不执行**。建议原版SA与唯一候选各3seed×64会话，共384训练session、最多184320训练帧；0/8/16/24/32/40/48/56/64各完整相同48开发条件，共2592评估session、最多1244160评估帧，资格另≤960帧。原GRU/数据/日程和费用共享，最终固定64，不挑峰/拼seed。

下一阶段任务点：

1. 冻结两臂配置/源码/SHA/唯一执行预算，先完成默认配对资格和exclusive启动门禁；原1/1不能自动续用。
2. 实现两臂相同的密集snapshot消费端，画真实训练raw回报与固定48面板greedy曲线，逐seed和整体分开；保存完整负例、实际帧/决策/更新与Adam参数步长，不只看梯度norm。不能插值伪造8/16点，不能用平滑曲线代替评估。
3. 预定48/56/64尾部工程稳定阈值，草案为reward范围≤1%、完成率≤1pp、模型/完成请求≤2%，按每seed检查。阈值尚待执行合同最终冻结；通过只称“预算内尾部稳定”，不称渐近收敛。持续改善但未平台可判学习有效、尚未稳定，不自动加训。
4. 候选相对同轮legacy的配对学习增益、seed17相对精确初始化的恢复，以及原整体/seed7/27非回退门禁全部核验。保留原最终64，未达门禁不替换；不能保证公平新对照后维持排名。
5. 对公开prior排序和实际驻留的误差，先另立小范围同状态分支诊断合同；本轮不混入第二个算法改动，也不靠DQN弱结果作主张。

## 7. 边界与交付

现有优势保留为固定模型与完整精度/原容差的preservation门禁；不把软件通过称为奖励改善。当前没有新的科学学习曲线，已有三个快照仍不证明收敛。开发时间会话与case选择均已消费、反事实滞留收益未测、预测/GNN/surrogate因果贡献未确定；formal/holdout封存，canonical不变，论文七维N/S。

主要验证命令：

```bash
python -m pytest -p no:cacheprovider tests/test_sa_separate_gradient_clip.py -q
python -m pytest -p no:cacheprovider tests/test_joint_stream_learning_diagnostics.py -q
python scripts/qualify_sa_separate_gradient_clip.py --stage3_root artifacts/joint_stream_learning/sa_learning_stage3_20261018 --output_root <全新资格目录>
```

专项23通过、回归46通过、资格9/9通过、参考5/5全内容一致。中后期参考已有exclusive启动锁，不能重新运行已完成目录。旧冻结工具如需重验原科学123源码，使用原Git35975cb或冻结源码checkout；当前源码已包含显式候选，不追认旧source身份。所有原数据/checkpoint和私有artifact不入Git；本阶段提交仅源脚本/测试/文档，真实同步状态另记。
