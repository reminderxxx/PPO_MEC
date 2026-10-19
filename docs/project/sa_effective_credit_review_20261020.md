# SA第五阶段5A：有效动作与信用分配独立诊断

- reviewed_at: 2026-10-19T16:47:16.813576+00:00（本地日期2026-10-20）
- literature_cutoff: 沿用2026-10-17文献边界；本轮只审现有源码/日志，不作novelty评价。
- target_venue: IEEE TMC内部审查规则；policy_version: `tmc_review_policy_v3_20260621`。
- artifact_run_id: `20261018T082032Z_sa_group_clip_pair`；诊断根 `artifacts/joint_stream_learning/sa_effective_credit_stage5_20261020`。
- Git baseline: `fbd73863fdc8f5c6b2b90e990d702895bd9b8119`；科学源码 `8cacb6de54c1cbde1da6f8089d9060810707958a`。
- evidence: 现有开发日志/源码独立复算；正式论文 `E0_UNAVAILABLE / Unverifiable`；N/S，不打投稿分数。
- 授权：用户2026-10-20要求先5A再同状态核对，保留原版SA与已有优势，不启动新训练。5A已完成；5B须在本报告之后冻结机器合同；5C算法改动和新学习均未授权。

## 1. Artifact、protocol与公平性

130项阶段四冻结source、10项生产端日志SHA、96次真实update、8572次训练forward及其25716条head覆盖记录核对通过。独立动作/信用报告逐臂seed/head计数一致；384训练及2592开发会话的FIFO开始/结束与decision_rows全部核对。根交叉核验158项通过。环境、模型评分、梯度、Adam和新训练均0；未重新扫描阶段四原始gzip。

4epochs重复forward共4286256行/每head，依据每update准确4遍且无KL早停、与decision_rows一致，代数/4为1071564个behavior行。没有逐行UID，不能称逐ID去重；这些行也不是独立时间样本。两个已消费相邻时间session及合成成本边界保留。

## 2. 类别自由度与物理有效动作

| 臂 | seed | behavior行 | 执行站类别多选 | 缓存类别多选 | 准备/优先级类别多选 | MC尾段跨自身完成的行占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版 | 7 | 175198 | 16.2296% | 1.0845% | 100% | 77.6875% |
| 原版 | 17 | 178383 | 16.3934% | 1.2524% | 100% | 77.7019% |
| 原版 | 27 | 179929 | 16.3776% | 1.4622% | 100% | 76.3979% |
| 分组裁剪 | 7 | 178578 | 16.2629% | 1.1384% | 100% | 77.8114% |
| 分组裁剪 | 17 | 179638 | 16.2621% | 1.1250% | 100% | 77.0221% |
| 分组裁剪 | 27 | 179838 | 16.3547% | 1.3017% | 100% | 75.8688% |

forced表示只有一个合法类别；可含必须执行的冷模型准入，不能等同noop。每seed至少4397–5366个非空cache命令属于forced，仍只是命令/尝试下界，不是下载使用收益。prepare非空命令两臂合计229/218次，不代表priority无用。

第三head为5个prepare站（含None）×4个priority，None的四个编号恒合法，因此100%类别多选及约85%的actor/entropy/KL active-pair分母占比是结构事实；**不等于85%的真实梯度、物理机会或因果作用**。

环境priority排序会改变link claim first-wins、cache/state/input/output、共享compute与LRU。固定完整因果前态及其余controls，若改priority仍产生同一全局稳定顺序，则一帧物理状态/费用/下一物理观测同效；event编号及学习轨迹不等价。唯一active请求是可证窄类：训练原版2900/533510（0.54357%）、分组2671/538054（0.49641%）行。多owner同效类的实测频率仍UNVERIFIED，不能把prepare=None全删或将100%多选全称有效。

## 3. 信用分配与误差边界

实际标签为整物理session的团队MC后缀，先/10、再/固定物理fleet8。Tail仅把global returns对齐/8，不是重复归一化。receipt.request_rewards的均匀分配账未被PPO消费。每请求FIFO切换清理请求状态/控制器记忆、共享model cache按session保留；未发现请求重置/算术错误。

两臂约77%的训练决策，其MC后缀越过自身请求完成；后续历时位置约47%，空闲物理帧约3.75%。这些是日历暴露比例，**不是回报/浪费份额或未来输入泄漏**。团队长期目标可能有合理跨请求贡献，信用噪声只是待验假设，不能直接改成逐请求回报。

actor、entropy、KL正确屏蔽真正singleton head；优势逐head在active样本中标准化，然后按所有active-pair统一分母汇总。global/prefix/selected-Q仍对forced回归相同团队MC。Q/辅助预测使用detach特征，不能称value回归直接污染actor encoder；legacy共同clip构成另一耦合路径。

行为时baseline是策略下`.5*(Q+surrogate return)`的加权期望；存储baseline在PPO epochs内固定，surrogate不直接改stream greedy logits。model辅助标签为新准入对象完整名义传输帧/10，state为该event实际state slots/10；不等于本帧真实model费用，也不能把共享下载强分单请求或把sunk cost再罚一次。

原训练缺逐behavior输入、actions/mask、T/B/advantage、逐head损失/梯度方向，故无法恢复实际有效动作是否获正确信用。候选seed7 pooled Q-MSE较低而dev64更差，也不能把低聚合误差称同状态信用改善。5B的greedy事实MC校准也不能冒充原stochastic PPO误差。

## 4. 机制正反例与结论

保留原版SA seed7/27自身0→64奖励+17.967493/+5.061077，以及seed17−8.624080。原版dev046在7/27显著改善，17下降；相同value主导clip并不充分解释退化。候选seed7最差dev046重准入反而减少、seed27正dev021的队列与换目标增加、seed17正dev043奖励改善但完成数少1，反驳单变量充分/必要归因。

**结论：尚未识别分组裁剪退化的唯一原因；已有证据支持精查有效head权重与连续交付排序，不支持直接调clip/lr、删priority、延长锁定或罚已付费用。** 原版、固定64、整体/seed7/27完整精度保护门禁、formal/holdout和canonical不变。不得为补归因重跑阶段四或启动新训练。

## 5. 下一步：已授权5B，同状态有界见证

严格保留既有八条件、三seed×两臂×0/32/64，共144参考会话、最多69120帧；不重估后验子集主效果。按每个原case实际steps预先固定九分位帧，含idle空帧，不另追逐坏事件/选峰。仅重放参考策略，其他六模型/一个prior-only软件条件在相同pass、相同输入、teacher-forced reference prefix上评分，输出允许不同。

先冻源码/合同/命令/单次claim，缓存软件资格≤16已有snapshots/112forward，环境0、反向0、Adam0。参考重放须完整summary及canonical events/receipts与阶段四一致；每次step/评分先扣预算。捕获完整可见decision、第一/二遍上下文、全队controls/优先级资源顺序和物理前态，保留所有正负例；第一遍未计算的Q/V不得伪造。评分保存prior/learned/logits/mask/probability、第二遍完整Q/surrogate/V/baseline及事实MC；状态、RNG、权重和optimizer不变。

独占一次启动，无重试/恢复/追加；明确失败保留现场并停止。5B结束后先独立报告可识别/未识别边界，再提出最小5C任务；本授权不包含算法改动或任何新学习轮。

## Evidence与验证

- `existing_evidence/sa_stage5_effective_action_diagnosis_20261020.json`及helper：96update/8572forward/25716head、source行锚和反例。
- `existing_evidence/sa_stage5_credit_diagnosis_20261020.json`及helper：2976完整会话FIFO/MC历时重建、完整0/8/…/64与八条件正负矩阵。
- `a_review.json`：158项根交叉检查与SHA绑定，source_integrity_pass=true。
- 复核命令：`PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python /tmp/build_sa_stage5_a.py`；结果complete。私有证据归档在诊断根，不提交真实数据/checkpoint/artifact。
- 未覆盖：原stochastic逐行信用、全部多owner动作物理有效率、替代动作长期因果、正式泛化/收敛。固定九分位帧可能错过稀有singleton/FIFO见证，例如原版seed7/32/dev046的frame220/221；未捕获不等于不存在，不补选或扩大预算。
- 新入口匹配验证：`PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sa_stage5_same_state.py`，22 passed；无模型评分/环境/优化。`git diff --check`通过。最终检查记录和源码同步以私有validation/git_sync receipt为准。
