# 5C1 因果时点、条件概率与信用审计合同

- reviewed_at: `2026-10-20T06:21:17.167551+00:00`
- literature_cutoff: `2026-10-17`，沿用；本次不作 novelty 检索。
- target_venue: `IEEE TMC internal diagnostic/no readiness claim`
- policy_version: `tmc_review_policy_v3_20260621`
- Git commit: `5c73ace0230c39798b332d122be3b5156972b455`
- artifact_run_id: `sa_credit_contract_stage5c_20261020`；5B仅作已完成的背景证据。
- evidence_level: 当前概率/因果结论为源码静态推导；真实 stochastic 行为完整记录尚未取得。正式 `E0_UNAVAILABLE / Unverifiable`，不晋级论文 claim。
- source ledger: 私有本轮目录`sa_stage5c_causal_source_anchors_20261020.json`。10个文件在实施前与上述Git原件精确相同；当前ppo仅增加受守卫观察hook，删除明确日志语句后完整AST与该基线相同。其余原源码、原模型和科学工件保持。
- status: 用户现已授权5C第1、2项：合同、默认关闭的审计工具及零环境纯软件资格。**本合同不授权任何算法候选、环境、真实agent/model构造、forward、backward、Adam、训练或回放。** 项目旧任务文档中的未授权状态不替代本轮用户授权；其他禁行仍有效。

## 1. 核心结论与强度

1. **SOURCE_VERIFIED：实际训练是条件逐head PPO。** `update`使用保存的`head_logs`，逐head构造ratio、分别clip，再按`enabled`加权。保存的`logs.sum(1)`未用于该ratio。不能直接把它称为整队joint PPO，也不能把整个owner tuple视为只给定owner原始观测的独立边缘策略。
2. **SOURCE_VERIFIED：时间顺序是全队first execution，再逐owner second cache/event。** second执行头重用first已抽动作，不再次抽样。固定owner顺序为sorted IDs按frame旋转，与环境priority排序是两种不同顺序。
3. **SOURCE_VERIFIED：固定prepare目标后，priority数值不进入claims、calendar booking或target_memory。** 这支持“后续owner的特征构造/概率kernel在固定prepare及同一其余prefix时不依赖该priority”的受限源码命题；不等于已实测真实PRNG续接一致。
4. **SOURCE_VERIFIED：最后一个second-pass owner的其他controls均已知。** 更早owner的后续priority尚未知。因此5B在完整reference co-controls上形成的order类，通常是事后条件类；并非一律是采样前可用的在线类。
5. **MATHEMATICAL_CONDITIONAL：对priority子因子，未来co-controls满足条件独立后，可研究事后baseline；这不直接适用于整个head2。** 另有较强条件下的完整joint类边缘化或本head未clip surrogate Rao–Blackwell恒等式，见7.1/8.1；它们都不能推出当前逐raw clip/entropy不变。
6. **UNVERIFIED：一帧源代数同效尚不等于完整controller/PRNG续接或冻结策略长期return同效。** 更不能等于未来学习续接同效：原`actions`和`head_logs`进入后续更新，改变raw编号通常就改变训练数据。

可点击源码锚点已映射至本轮当前行；正文未带文件名的历史行范围指基线Git，原件可用该commit直接核验。

## 2. Live dispatch与术语

Live入口是`registry.build_agent(..., control_contract='joint_stream_control_v2')`→`StreamSAGHMAPPOAgent`→`StreamControlPPO`→`DeliveryControlPPO`→`TailControlPPO`→`ScheduledControlPPO`。本合同中的“learn”特指这条`update`链，而不是其他历史agent的`learn`。

源码锚点：

- [registry.py:246](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/registry.py:246)，stream映射在250–263。
- [sa_ghmappo_agent.py:504](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/sa_ghmappo_agent.py:504)，clip flag只改变梯度应用；542–553继续委托相同update链。
- [ppo_agent.py:1032](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/ppo_agent.py:1032)，stream contract为`joint_stream_control_v2`。

记号：owner生成顺序为`v1,...,vm`；`E_i`是execution，`K_i`是cache，`R_i`是prepare target，`P_i`是priority。`H`必须包含实际因果prefix和对应tensor/候选/mask，不能只写“同一state”。所有概率默认针对合法候选的Categorical；物理环境是收到全队controls后才前进一步。

## 3. 精确时间轴与可见性

| 时点 | 已知/已生成 | 尚不可当作已知 | 动作/RNG与保存 |
|---|---|---|---|
| frame开始 | 当前因果decision、已释放FIFO、旧target/request memory、共同预测/共享状态；新request清理旧target/commitments | 当前frame所有新动作 | `request_memory`更新；不产生reward |
| 初始team plan | 各owner旧memory target或当前location/association；共同reactive calendar | 当前frame新E/K/R/P | `_team_targets`与`targets`是同一个dict |
| first owner i前 | `E_<i`已选；`E_>=i`仍是旧/default targets；reservations仅含已选E前缀；初始calendar尚未因所有新E重建 | `E_>i`真实选择；所有K/R/P | 用此刻features采样`E_i`；保留`first[i]`和chosen tensor |
| 所有first结束 | 全队真实E及全部execution reservations | 全队K/R/P | 从完整E重建reactive calendar |
| second owner i前 | 全队E；已生成的`K,R,P`前缀owner<i；该前缀已book的calendar | owner>i的K/R/P | 用fresh f1/f2等替换第一遍对应字段；own/f0/m0/graph/critic等保留第一遍 |
| second head0 | 其概率输入仍是原first execution输入 | 不得用fresh m0或其他新信息重写原E概率 | 强制`execution_actions=chosen[i]`；重评原E logprob，**无新sample** |
| second head1 | second输入、选定`E_i`；cold mask可强制当前family准入 | 当前K及R/P尚未抽 | 抽`K_i`；prefix condition变为选定cache feature |
| second head2 | 上述prefix及已抽`K_i`；actual mask已按E/K冲突调整 | 当前`R_i,P_i`；后续owner K/R/P | **一次Categorical抽联合编号`(R_i,P_i)`**，不是两次runtime抽样 |
| owner i完成 | 构造control、request/node身份、claims、book calendar；target_memory只写E/node/since | 后续owner的真实controls | raw actions/logs/values/baseline进入rolls；不step环境 |
| 全队完成 | 完整controls、全队priority、可计算完整global order | 本frame物理grant/结果/reward | 只此后调用`env.step_controls` |
| episode结束 | 实际全部reward、含idle帧的未来team tail | 这些量在采样时不可见 | 后绑定MC/10、fleet/8信用；不能反填成当时actor观测 |
| 每4session更新 | 原实际行为data/actions/head_logs/baselines；按随机minibatch重评旧prefix | 不生成新的环境状态/动作轨迹 | 逐head conditional ratio；本5C授权不执行此更新 |

核心源码：[ppo_agent.py:1205](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/ppo_agent.py:1205)–1250；request边界1204–1207，第一遍1213–1218，重建calendar1219，第二遍1221–1244。

### 3.1 second execution logprob为何仍可代表first行为因子

`data=dict(first[vid])`只换`f1,m1,node1,f2,m2,node2,opportunity_count`。head0的实际读集是own/f0/m0/node0、graph/adjacency/graph_mask、prior0以及对应policy context；这些保留原first值。SA的context使用f0与graph（Delivery `_policy_context`），不读fresh f1/f2或`opportunity_count`。head0的surrogate logit adjustment在Delivery为0。

所以在frame内权重不变、无随机网络层的当前实现中，second head0是对同一first概率因子的重评。**这是源码命题，未来真实stochastic审计仍必须保存first返回logprob与second存储logprob并精确核对；不得用5B greedy日志补充此项。** 若读集、随机层、数据覆盖或options改变，该命题必须重审。

锚点：[ppo_agent.py:693](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/ppo_agent.py:693)–737、886–902、1223–1226。网络构造195–200及410–413均为Linear/Tanh，无Dropout。

### 3.2 RNG调用预算是语义约束

stochastic路径`deterministic=False`下，每个active owner：first head0 sample一次；second head0强制重用；second head1和head2各sample一次。共`3m`次sample；分布构造共`4m`次。**forced mask也仍执行sample分支**，即使logprob/entropy为0且actor权重为0。

不能为“无效head”跳过原sample，不能把联合head2改成两次sample，也不能为了记录概率再调用一次sample/forward。Categorical kernel在改变支持集/维度后是否消费相同数量PRNG字节，本次未运行或检查库内部，必须记`UNVERIFIED`；不能由“调用次数相同”推出字节RNG状态相同。

训练默认False证据：[train_joint_stream_control.py:37](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/scripts/train_joint_stream_control.py:37)–42；[train_sa_group_clip_pair.py:495](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/scripts/train_sa_group_clip_pair.py:495)及597。抽样分支在`ppo_agent.py:722`，execution-only提前返回在724。

## 4. 行为概率与实际PPO目标

在给定frame外部历史、controller初始状态和完整已实现prefix的条件下，源码的frame动作链分解为：

\[
P_\theta(E,K,R,P\mid\mathcal F_t)=
\prod_{i=1}^{m}\pi^E_\theta(e_i\mid H_i^E)
\prod_{i=1}^{m}\pi^K_\theta(k_i\mid H_i^B,e_i)
                         \pi^{RP}_\theta(r_i,p_i\mid H_i^B,e_i,k_i).
\]

`H_i^E`含first生成前缀；`H_i^B`含全队E及second已生成前缀。后两类history与first不是同一时点。上式是完整**自回归链**的分解；不能将owner的三个乘子边缘化成“owner独立joint policy”。数据中存的owner `logs.sum(1)`只是其三个条件因子的log和；其他owner动作尚需相应因子。

实际learn：令旧实际动作及旧实现prefix为`a_ij,h_ij^old`，

\[
\rho_{ij}(\theta)=\exp(\ell_{ij}(\theta;h_{ij}^{old},a_{ij})-\ell_{ij}^{old}),
\quad
L_\text{policy}=-\frac{\sum_{ij}w_{ij}\min(\rho_{ij}A_{ij},
\operatorname{clip}(\rho_{ij},.8,1.2)A_{ij})}{\sum_{ij}w_{ij}}.
\]

`w=enabled.float()`，`old=cat(head_logs)`；`_evaluate(actions=stored_actions)`teacher-force旧prefix，并不按new argmax重新生成队友controls。KL也是逐head`((rho-1)-logrho)`的enabled均值，阈值`.02`；不是整队joint KL。只有将**所有真实生成位置**的logratio求和，且同一动作路径的history、支持和controller映射均一致，才形成该frame完整动作链的likelihood ratio；当前代码没有用它替代逐headratio，也没有对完整trajectory作joint IS。

源码：[ppo_agent.py:777](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/ppo_agent.py:777)–809，尤其780、793–802。`logs.sum`生产在1244，未被此update的old变量消费。不得将逐headclip改名为“精确joint clip”。

cache维度还会在frame合并与update前扩大padding（`_pad_cache`在574–579，调用1245与778）；新增位置mask=false、node=-1，语义上不新增合法动作。审计应保存原候选宽度、padding后宽度及合法id映射。不同batch/width的float32重算可能有舍入差异，**不能只凭理想概率公式承诺整个update replay与采样时逐位相等**；本轮不额外forward验证。first/second同owner head0的精确读集parity与这种packed-update数值问题是两项检查。

## 5. MC、baseline与“无偏”边界

### 5.1 现有真实单位

物理frame reward是receipt各组件之和。episode suffix `G_t=sum_{u>=t}r_u/10`，含idle帧；初始公开fleet=8时`S_t=G_t/8=physical tail/80`。三个head的credit目标均为S。Tail.update仅将global-value的`returns`临时再除fleet以统一单位，随后恢复原对象；不是对head信用二次除8。模型/state辅助费用分别是slot/10，不能当MC/80。

receipt里的均匀`request_rewards`不是PPO target。

锚点：[train_joint_stream_control.py:37](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/scripts/train_joint_stream_control.py:37)–65，尤其48–49、58–63；[ppo_agent.py:871](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/ppo_agent.py:871)–869；998–1004；[multi_request_stream_env.py:187](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/envs/core/multi_request_stream_env.py:187)–198。

### 5.2 原baseline是当前head动作之前可测的函数

SA旧baseline为

\[
b_j(H_j)=\sum_{a\in M_j(H_j)}p_{old,j}(a\mid H_j)
              \tfrac12[Q_{old,j}(H_j,a)+\widehat R_{old,j}(H_j,a)].
\]

Q/surrogate的valid输入在当前head选定动作之前已由prefix确定；Python调用Q虽在sample之后，其输入/全部候选输出不使用该head已抽动作。j0仍读first输入；j1允许条件于E；j2允许条件于E/K。这使b是当前动作之前的history函数。行为`act_controls`为`no_grad`；旧b保存在rollout，update用旧b构造A，新返回cf未替换该b。

对于未clip、固定history的score-function，任意这种停止梯度的b均有
`sum_a p(a|H) grad log p(a|H) b(H)=0`。不要求Q估计准确、也不要求它是真实干预Q。若用完整支持的未clip条件IS ratio，同一固定history下亦可保持此恒等式。

**实际代码不因此变成严格无偏joint PG**：A按每head的active样本作有限batch均值/std归一化；该统计包含实际动作/return；还有逐headclip、batch有效head分母和熵正则。这些是现有PPO surrogate的组成，不能由baseline恒等式推出整个训练估计严格无偏。也不能仅因Q/b校准差就宣称baseline引入PG偏差。

锚点：[ppo_agent.py:910](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/ppo_agent.py:910)–902；693–733；782–805。`active`只屏蔽actor/entropy/KL，value/Q/surrogate仍拟合：799–802；forced cache准入可以有真实物理贡献。

## 6. priority order类：源代数与因果信息是两项证明

对owner i固定完整非priority control、matched claims、全队其他controls Z和frame旋转tie规则，定义

\[
\mathcal C_{r,z}(p)=\{p':\operatorname{order}(p',r,z)=\operatorname{order}(p,r,z)\}.
\]

环境仅通过排序读取合法priority数值；相同完整order、同claims、同其余controls意味着相同一帧资源分配/物理状态/reward，允许描述性`event.control.priority`不同。相同order只是充分条件；不同order不证明物理效果不同，因为可能资源不冲突。

5B只证明reference prepare下的固定claims类。其他prepare的60个候选仍未识别，不得用reference claims重新贴“等价”标签。源码`control_link_claims`对priority不敏感，故可提出**任意固定prepare的符号命题**，但若要新增其他prepare的实际类，必须给该prepare匹配的decoder/planning/calendar/claims证据；本阶段不回改5B结论。

源锚点：

- [multi_request_target_control_env.py:254](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/envs/core/multi_request_target_control_env.py:254)–313：order282–285、grants288–290、owner执行292后。
- 同文件263–276的intent wire不含priority；232–235及368记录priority仅作描述。原事件不允许删字段，比较物理canonical scope时才明确排除此字段。
- [joint_link_reservation.py:87](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/envs/core/joint_link_reservation.py:87)–96：claims只读execution/prepare/cache/calendar，不读priority。
- [joint_delivery_planning.py:174](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/envs/core/joint_delivery_planning.py:174)–189：booking不读priority。
- [ppo_agent.py:1240](/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/src/agents/ppo_agent.py:1240)–1244：target memory只读E/node/since；原rollout保存raw action是另外的学习副作用。

### 6.1 类是否在线可见

- 最后一个second owner：Z的全部其他controls已生成，可属于head2采样前history。
- 更早owner：Z包含后续真实K/R/P，尚不是actor可见信息。不能将这些值塞进在线mask/feature或声称原行为以它们为条件。
- 可另证明对所有合法未来后缀都保持同order的**robust causal类**，而非取已实现后缀。当前mask为每个合法prepare提供四priority；一般不能用5B某个实现后缀的合并结果代替这个全称命题。
- 在完整joint action生成后作审计不构成在线信息泄漏，但其概率/无偏性仍需以下单独证明。

## 7. prepare/priority分解与未来co-controls条件化

当前一次draw的联合分布记`q_theta(r,p|H)`。纯代数可分解

`q_R(r|H)=sum_p q(r,p|H)`，
`q_P(p|r,H)=q(r,p|H)/q_R(r|H)`。

这不是源码执行了“先抽r再抽p”，也不允许新增一个sample。

令Z是后续实现的peer controls。一般有

\[
P_\theta(r,p\mid H,Z=z)\propto q_\theta(r,p\mid H)
 K_\theta(z\mid H,r,p).
\]

固定r时，当前源码claims/book/target_memory无priority读集，后续owner不读取前owner raw rollouts。因此在独立未来随机数、同其余history且无隐藏priority通道的概率模型下，可得到受限条件
`K_theta(z|H,r,p)=K_theta(z|H,r)`。这支持

`P_theta(p|H,R=r,Z=z)=q_P(p|r,H)`，

但**不支持**`P_theta(r,p|H,Z)=q_theta(r,p|H)`。后续calendar可以随prepare r改变，故R的后验一般会被Z选择。

因此：

1. `sum_{p in C}q(r,p|H)`是给定H的联合编号score mass；对事后Z形成的随机集合，它不能自动称为实际`P(R=r,C|H,Z)`。
2. `sum_{p in C}q_P(p|r,H)`才是满足上述独立性条件后的priority条件类质量；必须明确已条件化R。
3. 当前actor head2的score是`grad log q(r,p|H)`，不是`grad log q_P(p|r,H)`。后者等于`grad log q - grad log q_R`；将只适用于priority子因子的证明套给整个head2会遗漏prepare项。
4. 最后owner的Z若已属于H，不需要未来Z的Bayes补正；仍必须使用相同H和匹配claims构造类，仍没有自动证明长期return等价。

### 7.1 完整joint类边缘化并非原则上不可能

若对固定r的类内priority，suffix kernel不变，完整joint概率可写作

`P_theta(R=r,C,Z=z|H) = [sum_{p in C(r,z)} q_theta(r,p|H)] * K_theta(z|H,r)`。

这是**joint mass**，不是给定Z后的posterior；要作完整joint IS，其ratio还包含`K_new/K_old`及此前实际生成因子的ratio。θ在多owner共享并不允许丢弃suffix项。当前learn未使用这种完整joint IS，不能拿该代数式给现有逐headclip改名。

**可作为纯概率fixture的反例**：令R/P各二元均匀，未来Z=R（R改变calendar导致后续选择）。给定Z=1，R=1后验质量是1，原`sum_p q(1,p)=.5`不是该后验。令baseline b(Z)=Z，Bernoulli R的logit score为`R-.5`；`E[b(Z)(R-.5)]=.25`不为0。故future-co-control baseline不能直接用于整个联合head2。条件于R后，priority score的baseline恒等式可成立，但这是另一项受限结论。

## 8. 合法class logprob/ratio与不可等同的PPO损失

若partition是已明确的固定、合法条件历史h函数，并有相同支持：

\[
P_\theta(C\mid h)=\sum_{a\in C(h)}\pi_\theta(a\mid h),\quad
\rho_C=\frac{P_\theta(C\mid h)}{P_{old}(C\mid h)}
       =\sum_{a\in C}\pi_{old}(a\mid C,h)\rho_a.
\]

不能使用被选raw action的旧logprob充当class分母；不能对raw ratios作简单和/不加权平均。需要相同candidate身份、相同partition、完整old概率向量及因果scope。若只证priority条件类，上式必须用q_P及明确的R/Z条件，而非偷换当前q_RP。

partition须对old/new固定，不可随new argmax、Q排序或结果后挑选改变；不允许用policy-dependent概率阈值悄悄改变类成员。

### 8.1 受限的事后本head未clip Rao–Blackwell恒等式

这与“原行为以future Z为条件”是不同命题。令旧完整行为分布为`q_old(r,p|H)K_old(z|H,r,p)`；在固定`r,z,C`内，若：

1. `K_old(z|H,r,p)`在该类内为同一常数；
2. 所有raw成员的完整actor乘子`U`相同且对new θ停止梯度，例如已证明的冻结策略return/旧baseline/归一化乘子均类内相同；
3. 类与支持固定、完整旧q可得，所有相关质量非0；

则条件类内旧权重为`q_old(r,p)/sum_C q_old`，所以

`E_old[ (q_new(r,p)/q_old(r,p)) * U | H,r,z,C ] = (sum_C q_new / sum_C q_old) * U`。

这使**本head未clip surrogate的条件平均**可以代数收缩，不要求把joint mass误称posterior；但它既不是完整joint IS，也没有证明当前clipped目标/entropy、value/aux损失或未来训练等价。若suffix kernel随priority变化、U随raw成员变化或old向量缺失，就不能使用这个恒等式。5B的一帧类尚未证明完整return及实际stochastic续接条件，因此当前不据此修改任何loss。

即使return在类内相同，**期望的未clip score关系也不等于现有有限batch clipped PPO等价**。纯fixture：old两raw概率(.5,.5)，new(.9,.1)，它们同属一个类，A=1，epsilon=.2。class ratio=1，class目标=1；原raw clipped期望为`.5*1.2+.5*.2=.7`。raw熵从log2折成0也改变了现有正则目标。这是可能的未来算法差异，不是本次允许的日志refactor。

baseline的精确分组恒等式只能用概率加权类内估计：

`b=sum_C P(C)*mean_{a|C,old}[.5*(Q(a)+surrR(a))]`。

用不加权类内均值、任意代表编号或假设Q类内完全相等，通常会改b。5B已观察到非零类内Q/surrR spread；一帧同效不能拿来断言估计器输出相等。对未来priority子因子的事后baseline还要证明：给定H/R/Z时baseline独立P、Z条件独立P及类内完整continuation等价；对prepare score不能直接套用该baseline。

## 9. 一帧、controller、PRNG、return、学习续接分层

| 层 | 本次能作的结论 | 尚需什么 |
|---|---|---|
| 固定controls的一帧物理 | 上述合法priority同完整order是充分条件 | 其他prepare必须匹配decoder/claims；不得外推不同order |
| 本frame其余controller输出 | 固定R时priority不进入claims/book/E-memory，后续feature构造读集不含它 | future实际随机后缀的同分布/耦合、所有隐藏缓存/审计hook无副作用仍需核对 |
| 完整controller+环境续接 | 不承诺byte exact | request/target/commitment/team target/calendar、provider/shared-channel内部状态及物理state完整；event描述差异scope明确；B capsule不是可恢复完整env |
| PRNG续接 | 源码可数sample位置，不能推出kernel随机字节消费 | 未来真实before/after RNG、原sample调用形状/顺序；不得额外抽样验证旧科学run |
| 冻结策略未来return | 同完整续接状态、同外生未来、同future RNG耦合时可作归纳证明 | 目前仅一帧B证据，不足；不能把类内Q spread叫真实反事实Q错误 |
| 后续学习过程 | 不宣称等价，通常不等价 | raw actions进update索引，entropy/clip/selected Q/aux各不同；须另立算法合同，当前禁止 |

后续观测主要读当前物理状态/previous_service_targets，不读取priority描述历史：`multi_request_joint_control_env.py:157–225`；stream队列推进见`multi_request_stream_env.py:191–206`。这缩小了待查读集，但不能代替完整状态与实际随机续接证明。若只能证明同分布而非byte exact，应按该强度标注，不能升级表述。

## 10. 已实现的被动生产/消费合同

入口为`scripts.sa_stage5c_credit_observer.credit_audit(agent,sink)`；只有显式启用的sink才绑定。无训练CLI flag、registry新算法、checkpoint options或自动执行入口。`AuditSink(enable=False)`及disabled binding立即返回，不创建目录、不序列化、不读RNG、不访问或patch agent。Scheduled.update中只有一个private callback查找和八个明确guard；删除这些日志语句后完整ppo文件AST与基线一致。该检查是源级兼容性，不能冒称已做真实模型运行的bitwise对照。

启用时future caller必须另获真实行为授权，使用fresh目录、完整source/provenance及显式`session_id/seed`；`audit.session(...)`内才可由原caller执行原stochastic会话。observer只拦截原调用，不能自己启动环境、update、额外_evaluate/sample/backward或step。Greedy调用在执行前被拒绝，关闭后再调用也先拒绝。旧5B记录不能输入本生产器伪装为真实训练。

### 10.1 四类事件及精确单位

| 事件 | 实际记录与绑定 |
|---|---|
| behavior | 每vehicle/frame真实三head行一个稳定UID；first execution一次、second cache/priority各一次，second execution只重评。包含session/seed/request/node/decision身份、候选、actual conditional masks、actions、old head logprob；first/second实际输入及dtype/shape/JSON SHA、归一化Categorical logits/probs、现有_evaluate返回值、controller因果prefix和选后control。输入SHA为明确JSON语义哈希，不冒充tensor字节哈希。 |
| credit | 同一behavior UID绑定update UID。外部原returns T=physical MC/10、T_team=T/fleet、global-value真实消费目标、旧B、归一前rawA、归一后A、active heads与dtype。保留forced行真实值；原actor屏蔽由active表达。模型closure辅助与state event辅助目标单列slot/10，不强分共享model费用。 |
| minibatch | 实际indices、behavior/credit UID、epoch/mini编号、实际batch输入JSON SHA/shape/dtype；原new logprob/ratio、weight/denominator、KL、Q/surrogate/prefix/globalV/current baseline、loss、实际lr、RNG与是否KL拒绝。原storedB用于A，新cf不是自动替换它。重复epoch不增加行为机会。 |
| update | 实际原update闭合、完整eligible UID、completed_updates计数、主/副optimizer实际调用数、每个原step读到的分组参数变化与postclip梯度norm、复合mini及全update delta。三个分组为policy/value/surrogate；重复读取的norm不是每head梯度份额。主step数不冒充两个优化器调用总数。 |

所有源tensor立即detach成primitive，无计算图或源tensor留存。begin_update逐行核验frame/vehicle/request、storedB、actions/logs及实际输入（允许仅无效cache padding），避免forced同动作把UID错配隐藏。rawA在原归一化前立即复制成数值；原归一化公式与样本std/correction1保持。记录raw与normalized全行可离线重建统计；没有把未拦截的mean/std标为实际独立观测。

整帧act RNG before/after属于原完整动作生成；after状态明确不是actor当时可见信息。prefix抓取时点是after_features_before_evaluate。first-pass未计算的Q/V保持未提供，绝不补forward；raw learned-head logit与逐head梯度分解未捕获，不能由normalized总logits或分组norm倒造。实际forward/梯度的数值无扰动资格本阶段仍UNVERIFIED。

### 10.2 边界、失败与独立消费者

`AuditSink`默认上限10000行、67108864字节（均含header/terminal），primitive树200000节点/深48，单线程且禁止递归emit；不对历史joins逐行deepcopy。未来正式日志须事前另估完整预算，不能沿用本软件默认上限启动大run。硬限到达、非finite、字段/UID/joins错误立即失败，不采样/截断掩盖缺行。独立validator重算envelope顺序、SHA、计数、foreign keys、完整eligible credit集合与step账；KL早停允许部分eligible行从未进入mini，但必须报告never_sampled，而非当作日志损坏或伪补梯度。

消费者CLI只读：`python scripts/sa_stage5c_credit_audit.py <existing_audit_root>`。结构完整fixture/未验证观察链返回0；partial/failed/invalid返回1。header区分synthetic_fixture和unverified_actual_chain，任何情况都保持actual_training_verified=false。结构验证不自证来源真实性、信用算术正确性、loss/梯度因果或性能。

原step在调用前标attempt，返回立即标original_returned，随后参数观察失败另标observation_failed。异常保存最多8192字节独立failure.json（不占events.jsonl上限，单列预算），记总attempt计数和最近两次尝试，明确不是完整step账；原step可能已完成或部分执行。记录失败不回滚参数、不自动恢复/重试；secondary failure-receipt/terminal/stderr错误不得遮盖原异常。若磁盘不可写，文件仍可能缺失/partial，不能声称账已完整。

### 10.3 数值边界

保存原float32概率/logits/logprob，不以log(saved_prob)覆盖原logprob；高公共偏移/subnormal可使逆log不同。identity、source SHA、行为logprob/输入parity及原1e−8效果门禁不放宽。producer的T/fleet float32单位保护使用明确2e−6相对尺度，仅限序列化算术检查，绝非算法效果门禁。old class mass为0/支持不匹配时class ratio未定义，不加epsilon制造行为。本工具不计算或改变class ratio、baseline、loss、mask、clip或reward。

## 11. 零环境纯软件资格矩阵

只使用手写字典/数值向量和现有只读记录；不import live agent、construct模型或执行forward/gradient/Adam。当前操作上限均0。

| fixture | 预期检查 |
|---|---|
| owner first/second token timeline | 所有E先于所有K/RP；标清未来owner；second E是replay无draw；stochastic枚举sample位置3m，分布位置4m |
| sparse/forced masks | 合法id映射、forced logprob0/entropy0/actor weight0；不把“数值确定”改成跳过sample的许可 |
| headwise ratio vsjoint | 每head ratio数组、clip/denominator与源码公式相同；日志字段明确不是整队ratio/KL |
| R/P代数分解 | q_R归一、q_P条件归一、一个runtime联合draw；不能引入第二draw |
| future Z=R反例 | naive joint mass不是posterior；完整head2使用b(Z)的score期望=.25，反驳无偏误用 |
| 条件priority baseline | 在明确`Z independent P | H,R`的人工分布下验证score恒等式；取消该条件应失败 |
| class ratio | 加权raw ratio恒等式、singleton退化精确；selected raw ratio/不加权平均不冒充class ratio |
| PPO/entropy非等价 | (.5,.5)→(.9,.1)纯数值例：raw clipped=.7，class=1；熵也不同 |
| baseline regrouping | 概率加权类均值保持旧b；不加权代表替换故意给出反例 |
| order classes | 单owner、last owner、earlier owner future priorities未定、同order/不同order、reference prepare/其他prepare未匹配claims |
| FIFO/idle/缺字段 | idle无owner无概率；request UID重置；缺旧logprob/完整old向量/时点不伪造补值 |
| 高logit/subnormal | 保留source原值/未定义字段；对数值恒等式注明dtype舍入界，不能传播到hash或效果门禁 |
| default disabled purity | 无额外源调用/文件/RNG/权重/optimizer/controller写入；人工callable counter和不变字典可测，不实例化模型 |

## 12. 启动、拒绝与交付门槛

- **现在执行范围**：完善以上合同、默认关闭审计工具和纯软件fixture；匹配生产/消费schema。允许观察器设计，不允许任何算法flag/candidate、loss/mask/奖励/预算/训练改变。
- **拒绝在线类声明**：使用尚未生成co-controls却声称采样前可知；同一类混入未匹配其他prepare claims；只证明一帧就宣称return Q等价；用raw selected logprob当class分母；对当前整个head2套仅priority子因子的无偏性。
- **拒绝历史行为重建**：把greedy/teacher-forced得分当原stochastic日志、用额外forward补first Q/V、从checkpoint自动重跑旧run补日志。
- **拒绝算法候选启动**：只有编号收缩/熵或校准MSE、尚无因果可用性与正确ratio/credit合同。此时交付`UNIDENTIFIED`和缺口，继续完成已授权工具，不强行提出实现。
- **未来真实行为记录条件**：另行用户授权新的有界行为/训练合同后，才将默认关闭工具开在该真实调用链上；不借阶段四/五已用1/1额度。本合同不启动该运行。
- **本轮交付**：源码anchors、可证明/未证明命题、概率反例、工具schema和零操作软件资格报告。已有原版优势、正负seed和原科学产物全部保留；formal/holdout/canonical不动。

## 13. 本次实际执行记录与交付

reviewed_at: 2026-10-20T06:41:28.507524+00:00。90项匹配纯软件测试通过：28 sink/consumer、22 observer/概率/源码、40原5B回归；测试中step/evaluate等名称均为明示synthetic callback，未构造真实agent/model、未运行实际update/forward/backward/Adam。独立合同另8项有限概率检查、10份源Git原件核对，sink另6项针对检查、observer另完整AST与人工counter检查；这些检查相关且不作为独立科学样本相加。

当前新工具未采集任何实际stochastic行为。实际运行时数值/RNG/optimizer/controller无扰动、GPU/真实I/O成本及真实信用原因仍未验证；进一步资格必须另立有界预算。源码锚点和所有软件receipt保存在本轮私有目录，不上传真实数据/checkpoint/artifact。[本轮交付审查](sa_credit_audit_results_review_20261020.md)。
