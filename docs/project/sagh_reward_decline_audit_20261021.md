# SAGH 长预算奖励下降：只读目标、动作映射与更新审查

- reviewed_at: 2026-10-21 14:19 Asia/Shanghai
- literature_cutoff: 2026-09-28（沿用，未新评novelty）
- target_venue: CSCWD 2027 development
- artifact_run_id: `sagh_reward_decline_audit_20261021_v1`
- parent_run_id: `long_fair_reward_training_20261021_v1`
- policy_version: `tmc_review_policy_v3_20260621`
- Git: clean算法source=`8e154188baa2500daac1890f2504052fa2f8b389`；独立runner release=`2d7cfce`
- evidence_level: `COMPLETED_SEED7_LOGS_AND_PINNED_SOURCE_OFFLINE_AUDIT`
- verdict: 已定位该seed回报下降的动作与奖励账本，并确认概率映射、更新稳定性和上下文缺口；各因素独立因果贡献仍UNVERIFIED。完整三seed/正式优势/holdout仍UNVERIFIED。

## 授权与边界

用户要求只读查奖励目标、动作映射和训练更新，不改正在跑的实验或算法优化主线，继续20分钟监控。
本轮只读取已完成SAGH seed7的全部15,907训练episode、768个learn_info、33个checkpoint、792条验证unit、
现存验证动作trace、冻结计划与clean源码。没有新环境重放、agent.act、learn、backward、optimizer.step或调参；
只用两个合成样本做GAE闭式算术核验。输出新审查文件，不改变运行中的输入。源文件与保护hash前后均核对。

## 直接原因：准备动作减少，不是训练没有执行

该seed的固定验证回报从18.871250降到15.465000，完成率始终0.875，工作流连续性始终0.810521；
回传开销均值从128.666667降到112.666667（约少12.4%）。因此原生reward下降不等同于所有系统指标下降。

初始/最终24个相同unit的轨迹均368步。按同unit/步号对账（后续状态可能受先前动作影响）：

| 初始→最终动作 | 位置数 | 这些位置的原生reward差额合计 |
| --- | ---: | ---: |
| 4 prepare→0 current RSU cache fill | 75 | -77.25 |
| 4 prepare→3 current RSU steady offload | 6 | -6.00 |
| 4 prepare→4 prepare | 30 | +1.50 |
| 0→0、3→3 | 257 | 0.00 |

合计差额-81.75，除24unit得到-3.40625，恰好等于验证均值下降。严格相同初始状态配对是77/368个最终动作改变，
与上述自有轨迹81个步号位置不同，不能混用。

固定验证在98,304交互以前保持初始值，104,448交互首次有动作变化与回报下降；末期option gate在所查368步
全部选择option0 `accept_mappo`，初始仅197步选该option。回报变化的近端链条是：网络和option选择变化，
较多准备动作变成补缓存/普通执行，原生reward下降；不能说增加训练本身必然提高回报。

## 一、奖励账本一致，但学习目标与图中指标并非同一标量

**已核实无原生reward符号/offset/外部shaping对账错误。** 逐episode训练CSV、receipt和原生分项恒等式通过。
训练与验证共用同一core，offset=0；learner输入reward未执行外部temporal shaping。GAE使用gamma=.99、lambda=.95，
因此训练本来就在优化带bootstrap的折扣回报，验证图是未折扣episode total，两者也不能完全等同。

原生core本身含`mechanism_exploration_bonus`：有预测handoff信号且选择prepare/predictive prefetch时加1，
并不要求此步已完成真实handoff收益。见clean `vec_workflow_core_env.py:510`及`:1954`；
该分项在`:2347`标记`shaping_diagnostic`。没有外部shaping不意味着原生reward没有探索奖励。

可直接对账的训练第1/32循环（各6,144真实交互、同unit暴露）的分项为：

| 分项 | 第1循环 | 第32循环 | 变化 |
| --- | ---: | ---: | ---: |
| total reward | 11102.41 | 9853.84 | -1248.57 |
| mechanism exploration bonus | 1882.00 | 669.00 | -1213.00 |
| continuity bonus | 3852.60 | 3578.50 | -274.10 |
| service reward | 13103.80 | 13196.20 | +92.40 |
| cache miss penalty | 973.20 | 859.20 | -114.00 |
| delay penalty | 6227.15 | 6167.30 | -59.85 |
| constraint penalty | 471.80 | 469.00 | -2.80 |
| migration cost | 63.84 | 94.36 | +30.52 |

探索奖励减少1213，约为训练循环净下降1248.57的97.2%，同时存在正负抵消。这是训练分项的精确账本，
不把比例冒充验证分项因果解释：验证trace仅保存总reward，没有每项breakdown，本轮没有重放补测。

learn还叠加event/option PRD、成本dual、教师、行为正则、counterfactual margin、辅助loss等，
见clean `sa_ghmappo_core.py:3548`、`:7818`、`:11366`、`:11404`、`:4154`。
成本dual由初始0.0162在update67到上限1.85；768update的net-utility调整均值约-0.9953，
通过event/option信用进入学习。该方向可能减少有原生探索加分但成本较高的prepare。
原native reward与这些调整不要求同向；这是确认的目标差异，尚未用消融分离各项因果份额。
早期option prior/distill loss系数按原短schedule指数衰减，update408时日志六位舍入已为0；
不是在10万交互处突然新增开关，已有forward prior并未因此全部关闭。

## 二、确认的动作概率口径缺陷：门控前概率被当作门控后动作概率

clean `act`在`:2947–3030`先采样/投影，再经fallback、option gate及最终guard改写动作。
之后`:3030`调用`_env_action_distribution_statistics`，但该函数`:6418`仍只对门控前的五动作score作Categorical，
没有边缘化base action与option映射。`:7818`的env-action PPO读取最终执行action，却使用这一前置概率作old/new ratio。
层级actor也对最终重标的heads做proxy loss，不是完整latent base+option联合行为的似然。

无需新rollout即可从已有初始trace验证一个算术反例：

- window=`window_peachtree_off1296_len24_t1163170200_1163172500`，workflow=`j_3`，step9。
- base action3的概率0.62425494；option0/1/3分别映射3/4/4，条件option选择4的概率合计0.808808。
- 该随机分支产生最终action4的概率贡献已约为0.50490239；完整行为概率还应汇总其他base路径。
- 被登记/重算的门控前action4概率却约为`2.2659e-20`。该例没有idle fallback覆盖，不能用
  “fallback只在确定性验证开启”排除option在随机训练中的同类映射问题。

这些数值来自已存logits/候选映射；是源码随机分支的数学口径核验，并非新增随机训练实测频率。
完整映射概率与前置prob相差巨大是确定缺陷，足以使PPO的行为概率与重要性比率解释失效。
768次训练的per-step action_info/ratio未落盘，不能给出训练受影响transition数或断言它独自导致全部下降。
所查0/98,304/104,448/196,608四个验证点均没有非法mask动作；问题不等同于执行了非法action。
训练与验证还存在既有差异：`:5560`的idle popularity fallback仅deterministic时应用，随机训练中明确跳过，
而option gate在两种模式都会执行，不能把确定性平线直接等同于随机训练策略未变。

## 三、确认存在更新尖峰，PPO clip与现有KL门禁没有覆盖完整目标

**没有漏训练证据：** 768次learn调用均未skip，实际有效epoch共2686/名义上限5376；
Adam参数step均21488（2686×8 minibatches），88个tensor改变，网络初末L2变化9.2824。
全部33个checkpoint参数有限，日志scalar未见NaN/Inf；这不证明每个未保存的中间梯度都健康。

**确有稳定性问题：** env_action_ppo_loss有20次绝对值>100、11次>1e6，最大在update495/126,720交互，
记录值`3.390782751168863e13`。第一次>1e6在update361/92,416交互，早于首次验证下降；
但仅有时间先后不能当因果证明。完整尖峰列表在`loss_spikes.csv`。

源码`:7960`对ratio直接exp；`:7973`之后的额外ratio barrier含**未截断ratio的平方项**，
不是把整个loss都限制在PPO clip范围。actor的负优势unclipped分支也可能变大。总loss同时有多项辅助项。
梯度norm0.5能约束范数，不能保证行为分布、优化方向或全部辅助目标的trust region。

KL早停在epoch后用weighted-head的signed `old_log_prob - new_log_prob`均值与正阈值比较，
不是后置门控完整行为分布的KL，也不逐项控制env-action/option/辅助loss。768次中500次触发早停，
311次记录的KL均值为负；最大尖峰update495的approx_kl=-0.219108、early_stop=false，执行全部7epochs。
负的采样估计单独不构成bug，但结合口径和该例可确认现有保护没有拦住这一数值尖峰。
缺少per-minibatch ratio/gradient日志，尚不能分离barrier、负优势分支和概率错配的具体贡献。

## 四、独立实验入口遗漏window_class，不能宣称完整复现原算法上下文

这是独立实验入口的实现遗漏。旧native trainer `train_sa_ghmappo_real_sample.py:9927`传`window_class`；
本轮`run_long_fair_reward_training.py`的metadata只传cycle/unit/window/workflow等，没有此字段。
15,907条训练receipt metadata全部缺失；所查验证option window_class均为unknown，768update的
mechanism_window_count均0。clean `sa_ghmappo_core.py:6169`的mechanism-window preserve分支及若干
teacher/credit/context分支依赖该字段，因unknown不能照原分类激活。

这不等于窗口没有handoff机制或全部辅助训练失效：retrospective event guidance仍累计39,390条。
共同窗口、reward、交互与更新机会保持匹配，但不能进一步宣称完整保留原方法的上下文语义，
也不能据本run给原方法作正式强弱结论。本轮不补标签或修改冻结plan；分类规则须未来另行冻结验证。

## 截断/GAE、完整性和未确定事项

每episode独立buffer，runner收集到128-step unit尾部时缩短episode上限，trainer对非terminal尾部bootstrap；
每次learn输入256个已独立finalize的transition，不跨重置episode串接GAE。1534个budget_cut标志保留。
对terminated/非terminal时间截断两个合成二步样本，直接调用clean `PPORolloutBuffer.finalize`，
与独立闭式计算一致；没有找到此次下降由GAE符号、跨episode泄漏或漏bootstrap引起的证据。
预算短episode使训练分布与22-step固定验证有差别，这一设计限制继续保留。

审查原始证据：`artifacts/analysis/sagh_reward_decline_audit_20261021_v1/`下`audit_source.py`、
`report.json`、`loss_spikes.csv`，绑定父plan、源CSV/JSONL、33checkpoint hash。执行命令：

```bash
.venv/bin/python /private/tmp/ppo_mec_sagh_decline_audit.py
```

离线核验返回`PASS_READ_ONLY_AUDIT`，意为所述日志/账本/算术核验完成，**不是算法正确性或正式实验通过**。
七个用户文件diff前后仍为`1b10c6e1739314267d12bf86b56069efb4efd2a348ecdd0c519a132733908bb2`；
冻结runner/base/observer hash及clean source一致，当前训练未改、未暂停、未追加预算；
`ppo-mec-3`已只读确认ACTIVE、20分钟间隔。

尚未确定：另外两个seed能否重复；训练中具体哪个transition/梯度产生尖峰；各辅助项和映射缺陷各占多少因果贡献；
固定验证的逐reward分项。未做修复、消融、重新训练、holdout、文献novelty评价或论文修改。
即使完整run通过数据完整性审计，也必须一并呈现本报告的算法/上下文限制，不作为paper-ready或优势证据。
