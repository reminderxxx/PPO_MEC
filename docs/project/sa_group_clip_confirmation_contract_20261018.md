# 第四阶段：SA分组裁剪有界配对学习与稳定性合同

- authorized_at: 2026-10-18（Asia/Shanghai）
- policy_version: tmc_review_policy_v3_20260621
- target_venue: IEEE TMC 内部审查规范；不代表投稿资格。
- evidence_source: `20261016T162100Z_confirmation`、`learning_effectiveness_stage1_20261017`、`learning_diagnosis_stage2_20261018`。
- frozen_algorithm_reference: `35975cb3d73c84ddbabf01f6749bbc2d60c826de`。
- evidence_level: 局部开发诊断；正式论文 E0_UNAVAILABLE / Unverifiable。
- status: **用户2026-10-18明确授权“继续第四阶段任务”。这是独立的新有界合同，原确认轮1/1历史额度保持已用完；仅本合同允许一次两臂384训练会话。先软件资格、清洁源码冻结及完整性验收，再启动。精确commit/脚本/数据/预算绑定阶段机器contract.json。**

## 1. 问题与可证伪假设

阶段一完整同48条件：SA奖励205.598218→206.179616→210.399715，但seed17为206.201157→203.038582→197.577077。不能用总体均值掩盖seed17退化。

阶段二确认了非surrogate共同范数中value回归占主导，统一0.5裁剪会连带压缩policy梯度；这是**耦合事实**，不是“seed17退化由裁剪造成”的因果证据。正向seed7/27的诊断裁剪更重、Adam可能抵消共同缩放，均须保留。

唯一候选假设：**在相同loss、Adam、样本和预算下，仅将SA的policy/value分开裁剪，可提高真实训练中的策略有效更新，减少seed17相对精确初始化的退化，并保持现有交付/缓存优势。** 若分组梯度变化但固定面板不改善，则该假设未获性能支持；不自动再加学习率、训练量或架构改变。

## 2. 唯一 opt-in 候选与两臂

| 臂 | 唯一区别 |
| --- | --- |
| `sa_legacy_clip` | 当前非surrogate共同clip norm=0.5；surrogate独立clip=0.5。保留原行为。 |
| `sa_group_clip` | 同一backward后，policy组、value组分别clip norm=0.5；surrogate仍独立0.5。 |

- policy：heads、cache/domain encoder、graph encoder/message、graph residual gate；value：critic、prefix_values、action_values；surrogates维持现有独立组。按真实named parameters审查：无重叠、无遗漏；未使用参数允许grad=None。
- 保留原Adam及其参数/超参数、loss权重、return/10/8单位、PPO epoch/采样、entropy计划、4会话更新周期。仅裁剪分组变化；不拆优化器、不改lr、不改网络或信用目标。
- 分组阈值各0.5会改变联合梯度上界；不得描述为“总裁剪预算完全相同”。这是要验证的干预。
- GNN、surrogate、预测器、共享状态、公共prior/mask/预约、真实物理reward和费用、数据/到达/任务、LRU、环境与动作权限保持相同。DQN暂不修、不新增训练；已有DQN缺陷继续公开，未来不得借其弱结果支撑SA主张。
- factory默认仍legacy；候选必须显式opt-in并写入checkpoint options与manifest。默认路径/旧checkpoint的参数、推理和软件行为应保持兼容；不覆盖历史产物或canonical。

## 3. 进入科学轮前的软件门禁

1. 两臂相同seed的初始化tensor/actor、0更新计数完全一致；candidate开关不改变初始推理。
2. legacy开关关闭精确重现原裁剪公式与参数更新；候选分组disjoint/exhaustive，surrogate不混入policy/value。
3. 在价值梯度主导、policy零梯度、forced-only head等固定软件负例上验证裁剪系数、无非有限数及优化路径；软件验证不是算法收益证据，不用其reward挑候选。
4. 两臂共享同一冻结source、protocol、64会话条件序列和初始化；记录 source/config SHA，旧run只读。
5. 新梯度日志不得消耗RNG、改梯度、增加optimizer调用或改变采样次序。检验legacy0/32/64参数与原运行的可追溯一致性；若不一致，先定位，不把有差异的legacy称原对照。

## 4. 已授权预算（唯一固定上限）

| 项目 | 预算 |
| --- | ---: |
| 两臂×3seed×64会话 | 384训练会话 |
| 训练物理帧上限 | 184,320帧（384×480） |
| 更新 | 每4会话一次，每seed16次；不因曲线改变epoch/mini-batch数 |
| seed | 7、17、27；与旧run同一64条件日程 |
| 固定checkpoint | 0、8、16、24、32、40、48、56、64 |
| 每点完整dev | 相同48条件；3size×2arrival×2family×2capacity×2原始时间会话 |
| 两臂×3seed×9point×48 | 2,592个评估会话，最多1,244,160评估帧 |
| 资格核验 | 事前固定不超过2个软件会话/960帧；失败留现场不自动重试 |
| 评估总上限（含资格） | 1,245,120帧 |

全部0点重新测量，作为两臂推理同一性的验收；历史0/32/64仅作不可改锚点，不作为挑选/替换新点的材料。训练与评估分别计数；实际训练帧/决策数随策略不同，报告实际值，不能称严格等决策或等实际计算量。预先写命令、exclusive启动标记和失败策略；不存在64后的自动追加额度。

评估使用新环境、确定性greedy副本，每个会话reset控制器与session cache，保留会话内跨请求缓存；不热启动其他条件。优先独立进程评估；若同进程，保存/恢复Python、NumPy、Torch全部RNG且不修改训练agent/memory/optimizer。最终检查原训练轨迹不被密集评估扰动。

## 5. 必须记录的真实学习诊断

- 每次实际训练backward的policy/value/surrogate裁剪前范数、裁剪系数、裁剪后范数；实际Adam step后各组参数delta，避免把统一缩放误当学习率效应。
- 梯度可用性、多选/forced head覆盖、policy/KL/entropy、值回归/控制变量误差、有效graph gate。诊断按真实训练batch记录，不用开发greedy动作PPO形式梯度冒充训练梯度。
- 每点固定48面的原始reward、完成请求/节点、模型/state字节及slot、每交付成本、选站次数；按seed、size、family、capacity、arrival、原时间会话展开，保留全部配对负例。
- 重点核对seed17/dev022、018、034、046、017、020；同时保留seed7/27的同条件正例。模型重准入、付费后未compute即离站、状态排队、准备站与最终执行站不同分别统计，不能视作已识别的可避免损失或causal reward。
- 科学run存训练/评估行、checkpoint及provenance；固定的尾部和关键负例可以另设已计入预算的完整event/receipt保存，不能因结果不理想额外回放。

## 6. 结果选择与优势保护

主要结果固定为**64**。0和中间点仅用于学习过程及稳定性检验，不挑峰、不按seed拼模型。

执行时直接读并hash绑定 `artifacts/joint_stream_learning/next_stage_preservation_plan_20261017.json` 的完整精度门禁；只允许1e-8浮点容差：

| 范围 | reward下界 | 完成率下界 | 模型KiB上界 | 模型KiB/完成请求上界 |
| --- | ---: | ---: | ---: | ---: |
| 总体 | 210.39971450466592 | 88.88888888888889% | 96.9375 | 6.81591796875 |
| seed7 | 219.4164416097005 | 91.66666666666667% | 89.78125 | 6.121448863636363 |
| seed27 | 214.20562469482425 | 90.49479166666667% | 70.55208333333333 | 4.872661870503597 |

- 候选须同时通过上述历史锚点及同轮legacy64非退化检查；seed17改善不能抵消seed7/27退步。
- “分组裁剪有效”的主要学习判据：候选64总体reward严格高于同轮legacy64（超过数值容差），完成率与模型/交付成本不退化；报告candidate与legacy各自64−0。若仅持平，只能判未获增量效用证据。
- seed17分别报告：候选64相对自身精确0点的reward/完成率恢复布尔值、相对同轮legacy64的增量、重复加载/状态队列/付费后离站链的描述性变化。机制阈值尚无独立依据，不自动输出“退化已解决”；只比旧64改善而仍低于0时明确为部分改善，不能据相关变化作因果归因。
- 任一门禁未过，保留原版本，不替换、不改阈值、不删条件、不延训或筛seed。成功也只晋级为开发候选；论文排名/全基线优势须另有公平完整比较，当前两臂不能代替八基线主表。

## 7. 有界稳定性判据，不能冒称收敛定理

以下阈值在本合同固定为工程判据，结果后不修改：

- 固定完整48条件，在48、56、64三个快照上，分别检查每个seed及三seed均值。
- reward的max−min不超过该尾段平均reward绝对值的1%；完成率max−min不超过1个百分点；模型KiB/完成请求max−min不超过尾段均值的2%。同时64通过第6节保护门禁。
- 若通过，只写“在预定64会话预算及当前固定开发面板上达到有界尾部稳定”；不能写全局/渐近收敛或新seed泛化。若mean平稳但某seed退化，报告seed不稳定。
- 阈值可能拒绝一个仍持续变好的候选：这不等于算法无效，只表示预算内未观察到所定义的稳定性；不因此自动延长预算。
- 小KL、低训练loss、surrogate误差下降或参数变化均不单独构成上述稳定性证据。

## 8. 图与停止规则

保留原始训练reward；short→medium→long轮转造成的尺度锯齿不通过改reward“修掉”。可以另画分size训练线或固定6会话整块均值，需明确横轴/聚合方式，不能据此判收敛。

主要学习图为本轮真实0/8/…/64同48评估点：三seed均值+逐seed曲线/范围，并列完成率与每交付成本；不插值创造未测量策略，不筛掉下降点。三seed范围不是独立时间CI。

训练不因reward达到高点或尾段暂稳提前选模型。到64必停；非有限loss/参数、动作/资源合同违规、source/RNG/初始身份或预算异常则停止两臂并保存原现场，不自动重跑。正常检查按20分钟，不追逐逐回合reward。本合同已由第四阶段授权；不得据此启动第五阶段或另一轮训练。

## 9. 外推边界

48个条件重复在两个相邻peachtree原始时间会话上，且开发集已经消费；不能把144seed×case格冒充独立窗口。long单源、成本合成、环境仍非真实模型部署。该实验只回答裁剪候选对现有SA学习的增量与有限预算稳定性；不回答纯预测/GNN/surrogate贡献、完整MARL、顶刊novelty或formal/holdout泛化。formal/holdout封存、canonical不变；不下载数据、不外发。

## 推荐理由

1. 这是已确认的学习链耦合事实对应的单因素候选，不需要变环境、reward、数据、预测器或网络。
2. 两臂新鲜同预算训练与密集同面板快照，可同时回答“是否改善学习”及“改善是否维持到固定末点”，不靠训练锯齿或平滑说故事。
3. 保留反例、原模型及完整非退化门禁，避免为了修seed17而丢掉seed7/27。
4. 候选有可能无效；预先给出失败交付与停止规则，防止为了想要的上升图无界调参。

## 10. 本轮生产端、消费端与资格冻结

`launch_sa_group_clip_pair.py`承担qualification/prepare/两arm协调和自动审计，`train_sa_group_clip_pair.py`只按冻结contract训练指定arm，`audit_sa_group_clip_pair.py`独立消费原始行/checkpoint/诊断及事件收据。2个worker分别固定arm，每个torch1thread；原算法源码不再改动。训练前每个env.step与评估前均受实际物理帧计数上限保护，源码/非有限/动作/预算失败停止并保留两arm现场，不自动重跑。

资格预算执行前另有只含两固定case的机器qcontract；仅legacy64 seed7/dev000和同权重candidate/dev000，合计最多960评估帧，无optimizer，结果须与原已验证row/事件/收据一致。资格不测候选学习收益，不据其reward选参。新增纯tensor/subprocess模拟测试不运行额外策略环境，最多128次独立软件Adam.step，单列计数，不计为科学学习收益；不执行额外完整环境smoke。严格拒绝伪装软件训练为科学样本。

运行前复制并hash绑定preservation的promotion_gates和原64参考指标；continuation只更新阶段元数据，不改变门禁。source_commit必须清洁本地Git提交；真实尝试push，若GitHub443仍不可达，单独记录待同步，本地完整提交可冻结已授权研究，不假称远端已同步。训练运行中不改源码/合同/预算/reward/数据。

资格、RNG隔离和初始同一性成立只支持软件完整性；当前对候选提升、seed17修复和稳定性的结论均待结果。原始science数据/模型不提交，formal/holdout封存，canonical不变。每20分钟只看状态，状态无变化保持安静，结束后自动独立回收/出图/审查并给出第五阶段任务点。
