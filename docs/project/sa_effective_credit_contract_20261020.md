# SA第五阶段：有效动作、信用与同状态诊断合同

- authored_at: 2026-10-20（Asia/Shanghai）
- 用户授权：先有效动作和信用分配诊断，再核对同状态决策；保留原版SA及已有优势，禁止新训练。
- 阶段四科学source `8cacb6de54c1cbde1da6f8089d9060810707958a`；原版固定64和整体/seed7/27完整精度保护门禁保留，candidate不晋级。
- 5A独立报告：[sa_effective_credit_review_20261020.md](sa_effective_credit_review_20261020.md)。必须先有source_integrity_pass的complete receipt，才可prepare 5B。

## 1. 固定范围与额度

5A只读现有source/真实训练observer/summary/已审计派生visit，环境/评分/梯度/Adam/新训练均0。原始stochastic逐behavior信用缺失如实标UNVERIFIED，不用greedy形式量替代。

5B仅144参考会话：seeds7/17/27 × 原版sa_legacy_clip/候选sa_group_clip × 原checkpoint0/32/64 × dev000/018/021/022/038/043/045/046。每会话最多480帧、总最多69120；实际步数须与原精确同case相同。该后验富集诊断子集不重估主效果。无替代policy轨迹，无正式formal/holdout，无新训练、backward/autograd.grad、Adam、参数/reward/环境变更。

每个参考的N取原dev row actual steps，冻结 `sorted({i*(N-1)//8 for i in range(9)})`；每个选定帧所有活动vehicle、第一/二pass都保存，空闲帧保留空记录，不补选。最大20736捕获call/10368第二遍行；每call六模型（两臂×0/32/64）及reference prior-only共7条件，最多145152评分call。prior-only是零head-logit软件控制，不是新训练baseline；0点仍有公共prior/mask/协调/GRU，非随机策略。

另单列缓存软件资格：SHA绑定既有stage3 seed7/dev046 capsule，第一/第二pass各按原顺序前8条，共16snapshots/112条件forward；环境0、梯度0、Adam0。只检查同reference64、两个0模型及状态/输入/RNG精确一致，不用reward调参。资格失败停止，不自动重试。

## 2. 冻结与唯一启动

机器plan必须绑定清洁且已实际push的Git commit、全部130旧source、新runner及helpers、原result/audit/arm manifest/dev/ledger、protocol/GRU资产、18个精确checkpoint、固定144身份/实际长度/九帧列表。prepare再绑定同plan软件资格与5A审查receipt。运行前写独占run_started，只有1/1，无resume/重试/追加。step/score执行前扣额度；失败保留现场，不修源码再续原run。

入口：`scripts/diagnose_sa_stage5_same_state.py`，依序contract→qualify→prepare→run→status。科学Python使用现有venv；不下载依赖或数据。冻结命令形状：

```text
PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/diagnose_sa_stage5_same_state.py contract --source_run artifacts/joint_stream_learning/20261018T082032Z_sa_group_clip_pair --output_root <stage5_root>/same_state --qualification_capsule artifacts/joint_stream_learning/sa_learning_stage3_20261018/midlate_reference/02_sa_ghmappo_seed7_dev_046/snapshots.pt
PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/diagnose_sa_stage5_same_state.py qualify --output_root <stage5_root>/same_state
PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/diagnose_sa_stage5_same_state.py prepare --output_root <stage5_root>/same_state --a_receipt <stage5_root>/a_review.json
PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/diagnose_sa_stage5_same_state.py run --output_root <stage5_root>/same_state
```

`<stage5_root>`为continuation.sa_learning_stage5.run_root；实际argv/cwd/python写command.json。原run绝不覆写；输出不入Git。每20分钟只查progress/result/failure，正常安静，不按reward追逐。

## 3. 同状态、信用与物理记录

参考greedy行为完整summary及canonical每条event/receipt须与原阶段四ledger严格一致，含费用与全部原字段。首/二pass分别保存完整实际batch、candidates、masks、原controller context和teacher-forced reference prefix。其他模型只对同一输入/pass评分，不调用它们act_controls。每次评分恢复/核验Python、NumPy、Torch RNG，tensor/optimizer/controller/input不变；参考episode只允许自身controller变化，学习状态和其他模型不变。

被动hook保存真实forward的全部prior/learned/total logits、probability/logprob/entropy、第二遍各valid Q/surrogate三维、V/prefix/baseline。第一遍execution_only未算的Q/V明确None，不加forward伪造。条件概率不是greedy实际抽样概率，greedy事实MC不是原stochastic训练标签；不能补称原训练advantage或梯度。

每个固定帧共享完整可见team decisions、released queue、current node/output、statecopy/remaining transfer、cache/LRU/在途资源、全队controls与physical priority order。该因果ledger不是完整可恢复env快照，不含未来arrivals；更不提供替代动作长程反事实。集合用稳定编码便于独立hash。team_frame_sha绑定写入pre_action_payload_sha256之前的原pre-action字段；后加physical_pre_step及其SHA须单独复核，不能整dict不加区分重哈希。

仅在固定其余co-controls/targets/reservations、同prepare目标且全局priority稳定顺序不变时，把编号划为一帧物理同效充分类；不同序不自动判有用或无用。future optimizer/event编号不在此等价范围。shared model下载不强分单请求，sunk cost不重复罚。

## 4. 回收、边界与后续

自动完整回收原summary/trace精确性、18checkpoint/6初始化及源SHA、所有144单元/计数与费用，严格区分资格成本和参考成本。仅用已保存前态/评分/原事件做离线有效动作类、prior/residual排序翻转、commitment、remaining delivery/queue/family链和事实MC校准；保留全部正/负案例和未见证项，不为补原因新加环境。原stochastic信用与替代动作长期因果未识别时仍UNVERIFIED。

先独立报告再更新PROGRESS/BUGS/ARTIFACT_RECORDS，匹配验证、scoped commit并真实push；不上传真实数据/checkpoint/artifact。用户未授权5C算法改动/新训练；最终交付最小可执行任务点后暂停提醒，失败通知一次并暂停。formal论文仍E0/Unverifiable，不作收敛、泛化、纯预测/GNN贡献或候选晋级结论。
