# 尾部信用与迁移覆盖修复轮：完整开发结果审查

- reviewed_at：2026-10-15T10:06:01.485139+00:00；literature_cutoff：2026-10-15；target_venue：IEEE TMC。
- policy_version：tmc_review_policy_v3_20260621；artifact_run_id：artifacts/joint_tail_control，各group为`<agent>/20261015T074001Z_development`（完整绑定见audit.json.artifact_run_id）；补充run为mobility_support。
- Git implementation commit：1667da3a39b7ae2f724504fdd63697cbda1c1890；审查开始的Git基线d169aab7738df50c4c7b1959de1614dd05189c24，审计helper SHA256：3144412b0c8c6bc1543e19a727a7490783dbe0278f93d27bf8351ffc332dce0e。
- evidence level：development raw/replayed；paper E0_UNAVAILABLE / **Unverifiable**；未审阅未公开论文正文，不作novelty判断。
- 审计原件SHA256：main `de5773c57b6979c2714f108d3e8168ec27585ab5f4b2b463f8f03f8325a73acf`；mobility `e2f1150e900d76b197ec4c984d9c5a2f06ebc0cb99d43454d50778e0f6521a9d`；comparison_plan `5742213a217e1a4d20ea88d08751c3b4e9aaa0dfbde5c13ef7563419e3298af4`。

## 1. Artifact 完整性

十个group全部complete。公共协议与预测器hash、train/dev原始时间互斥、全部预算与循环场景、1920开发曲线单元和480最终单元核验通过；120 checkpoint、409883原始最终事件、55258帧收据核验通过；30个最终模型在base/tight首窗共60次完整事件重放一致。full48次同权重预测关闭原始事件另存`artifacts/joint_tail_control/prediction_probe/20261015T095658Z/events.jsonl`，SHA256 `8debf84b9ec41c51b9910596a7a5e7a8e28bb02b50d7895b39a524a61b4d4506`。各group命令输出存在，审计exit0。

八主配置各128episode×seed7/17/27，3072训练episode；sa_discounted和sa_no_surrogate各384，实际计划3840。工程SA/DQN各4episode合计8单列；本轮真实数据RL训练总计3848，不含以前轮次或pytest toy训练。前置旧模型固定权重定位80次开发评估、新模型预测开关48次均无参数更新。

产物根`artifacts/joint_tail_control/`，公共`protocol/`包含派生轨迹、预测器及manifest；各算法`<agent>/<run_id>/`包含training/dev/train_probe、原始events、checkpoint及manifest；各`<agent>/command.log`保留启动命令执行输出。根comparison_plan.json于2026-10-15T07:40:00Z冻结。原有scheduled产物和失败/工程验证日志保留，不混表。

## 2. Protocol / provenance

训练/环境/数据生产源1667da3a39b7ae2f724504fdd63697cbda1c1890，十组启动时全部Git clean；后续仅完善只读审计/文档，未在训练中改算法、费用、预算或场景。独立前置审查3453fe6定位旧seed27的surrogate事件头干预，见`joint_tail_regression_review_20261015.md`；再按`joint_tail_control_contract_20261015.md`独立实现、冻结训练。

新checkpoint合同tail_parameterized_ppo_v1 / tail_parameterized_dqn_v1。新训练统一有限时域gamma=1，原节点完成奖励及等待、模型/状态/输入/输出/共享费用保留；全局critic与动作/请求critic统一per-initial-request单位。sa_discounted仅gamma=.99，其余配置一致。没有workflow完成bonus。

surrogate继续学习，政策干预限定到正net cache/state准备；同一prepare target四个priority共享残差，noop priority不加偏置。常规执行特征使用观测进度/负载/slack，GRU信息保留供准备，DT保留专用原始预测context。未来calendar及净收益仍是估计，不是实际未来oracle。

### 场景和预测器

- 原20个训练时间区间、8个开发时间区间不改。每个train interval保留原cohort，另按持续覆盖、几何离开次数、位移、ID选择8车mobile_tight cohort；没有使用policy回报排名。原20场景29次几何离站，新增20场景151次，合计180（scene内计数，不作独立事件数）；全部车帧至少一个站可达。几何离站不等于执行站迁移或有益准备。
- 40train scene交替进入128episode；原8dev base保留，相同8dev增加tight半径panel，共16scene，主结果权重base/tight各50%。覆盖半径统一从spacing×1.2变为×.65，站位置、计算/链路/缓存容量、费用不变。预备几何检查比较1.2/.85/.65，仅检查覆盖/离站，没有据算法分数选择场景。两panel共享原始时间，不能当16个独立窗口。
- 公共GRU为horizon8、history8、hidden16、30epoch。16/4原时间区间分组fit/validation（各含两cohort），防止同一帧不同cohort跨组。bundle SHA256：b78be998d13fff145d5c1f249cad5f33af7206a153d6d5259c9d532e16d44ceb。validation RMSE1.010410、CV1.061508；这是新cohort分布，不与旧RMSE作同任务提升比较。
- prepared manifest SHA256：c0199b81f6c6badc108abb827adb95be4aa9133fab0b5c40d75820d29cde2ed8。环境观察与费用来自同一公共prepared protocol；runtime.json记录CPU、Python3.9.6、PyTorch2.8.0、NumPy2.0.2、单进程intraop线程1。

### 独立容量审查与探索性补充

`joint_tail_capacity_review_20261015.md`先于补充评估独立输出：8个DAG最低需要217计算槽、193输入/输出槽。枚举请求子集并按每帧可达站并集计算乐观容量上界，即使忽略模型、状态、DAG顺序和阶段竞争，tight的7/8窗口也已排除全8完成。base没有被该必要条件排除，但不等于已经证明可行。tight仍作为过载调度压力结果完整保留，不改50:50主权重。

补充`mobility_support/`是在本轮训练开始且部分主panel得分已可见后定义的**探索性**分层：保留原8个dev时间区间和原站半径1.2；先要求每车每帧至少2站可达，再按几何离站、位移、ID选8车。条件写入`mobility_support_plan.json`后才运行模型；不读取策略分数筛车辆、不搜索阈值。8个scene共50次几何离站，全部通过上述乐观容量必要检查，仍不构成真实可行调度证明。所有十组最终128模型、三个seed共240次policy评价，full另24次同权重预测关闭；0新增训练，原公共GRU不重训。共享原时间，不能当新独立窗口或替换主表来追认胜出。

补充审计通过：240policy单元+24预测关闭单元，198774事件，30最终模型重放一致，审查时间2026-10-15T10:04:03.122440+00:00。

| 配置 | 奖励 | 完成 /8 |
| --- | --- | --- |
| cache_offload_drl | 59.715258 | 8.000000 |
| sa_discounted | 59.631457 | 7.958333 |
| ppo | 59.544547 | 7.958333 |
| mappo | 59.464020 | 8.000000 |
| dag_offload_drl | 59.408586 | 8.000000 |
| sa_ghmappo | 59.388970 | 8.000000 |
| dt_handoff_drl | 59.355513 | 7.916667 |
| sa_no_prediction | 59.192464 | 7.791667 |
| sa_no_surrogate | 59.117579 | 7.875000 |
| dqn | 49.736023 | 3.166667 |

完整版59.388970/8，独立去预测59.192464/7.791667，奖励+0.196506、请求+0.208333；MAPPO59.464020/8，cache59.715258/8，仍无全基线最高奖励。固定full权重：24单元，奖励on−off +0.022625488，完成+0.000000000；3胜/3负/18平，完成均8/8。预测启用/关闭model槽630/631、state槽791/785，prepare54/0；作用小且有负例。独立重训差与同权重差不能互相替代。以上只是晚定义条件下的探索性证据。

## 3. 最终比较与统计公平性

| 算法 | 奖励均值 | seed均值样本std | 完成请求 /8 |
| --- | --- | --- | --- |
| `cache_offload_drl` | 48.750552 | 0.546023 | 4.708333 |
| `sa_no_prediction` | 48.603467 | 0.288750 | 4.229167 |
| `dt_handoff_drl` | 48.458615 | 0.373764 | 4.500000 |
| `ppo` | 48.442091 | 0.146123 | 4.333333 |
| `sa_ghmappo` | 48.375857 | 0.355044 | 4.062500 |
| `mappo` | 48.132056 | 0.516937 | 4.375000 |
| `dag_offload_drl` | 48.126992 | 0.036597 | 4.208333 |
| `dqn` | 44.643283 | 0.393084 | 2.187500 |

预声明辅助对照（同预算，不并入八主排序）：

| 配置 | 奖励 | 完成 /8 |
| --- | --- | --- |
| sa_discounted | 48.557613 | 4.458333 |
| sa_no_surrogate | 48.697197 | 4.250000 |

full−MAPPO：奖励+0.243801、完成-0.312500；full−no-pred：-0.227610/-0.166667；full−cache：-0.374695/-0.645833。目标未达，cache为本轮八主双指标最高。

| 算法 | base奖励 | base完成 /8 | tight奖励 | tight完成 /8 |
| --- | --- | --- | --- | --- |
| cache_offload_drl | 58.092569 | 7.250000 | 39.408535 | 2.166667 |
| sa_no_prediction | 57.219970 | 6.375000 | 39.986963 | 2.083333 |
| dt_handoff_drl | 57.818177 | 6.875000 | 39.099052 | 2.125000 |
| ppo | 57.208559 | 6.541667 | 39.675622 | 2.125000 |
| sa_ghmappo | 56.658542 | 6.125000 | 40.093172 | 2.000000 |
| mappo | 56.923403 | 6.500000 | 39.340709 | 2.250000 |
| dag_offload_drl | 56.667744 | 6.208333 | 39.586241 | 2.208333 |
| dqn | 50.437114 | 2.583333 | 38.849451 | 1.791667 |

原base中SA56.658542/6.125，亦未胜出；tight虽奖励在主配置中最高40.093172，但完成2.0低于MAPPO2.25，不能写成双指标优势。新主均值含50% tight，不与旧scheduled58分均值直接比较。

主表固定最终128；不取32/64/96峰值，不剔除seed27或困难窗口。每主配置48个seed-scene单元，只有8个原时间区间，且相邻/同源车辆DAG相关，不做独立cluster、显著性或普遍优势推断。报告seed均值及样本标准差，不能把点估计当收敛。

40个训练scene在128episode循环中各出现3–4次，原20scene方案则各6–7次；增强覆盖同时降低单scene重复量。本轮同时改gamma、critic单位、surrogate干预和分布，不能从新旧均值把改善/退化归因单项。训练不足仍是待验证假设，应在合同稳定后做事前固定预算曲线，而非只按最终排名加训。

所有方法交互episode预算相同；实际轨迹长度、PPO和DQN的样本/梯度步数不同，不能称优化计算完全相同。缓存/DAG/DT三个DRL为first-party PPO方向型实现，DQN为replay/target-network/Bellman；不宣称外部论文复现。QMIX/MAT/IPPO/DDQN等仍是旧动作/checkpoint合同，TD3/SAC/MADDPG未稳定，不混入新表。

**基线身份补充（源码事实）**：本轮mappo/ppo/cache/DAG/DT/DQN均消费公共GRU、预约、正净收益门控及队列先验，mappo是共享控制骨架下的预测/缓存增强对照，非裸MAPPO；full相对它主要多GNN/surrogate。本轮“完整”指恢复原仓库七种算法名，不等于覆盖论文全部算法家族。除DQN外大多共用PPO学习骨架，只换context/辅助模块，这层结果首先回答额外模块的增量。普通MAPPO及忠实外部方法/强规则仍须另立定义并补足；增强对照应保留，不能用较弱版本替换本轮结果来追认优势。公开原始信息、派生预测、物理合法mask与算法自身净收益门控应分清；当前公共门控也可能缩小机制间差异，影响大小尚未独立消融。

## 4. 主机制兑现

主panel固定full预测开关：48单元，奖励on−off -0.020602397，完成+0.041666667；1胜/1负/46平。base：24单元，奖励on−off +0.039254557，完成+0.083333333；1胜/0负/23平；tight：24单元，奖励on−off -0.080459351，完成+0.000000000；0胜/1负/23平。48单元只有4次prepare（base1/tight3），不能把180次几何离站当成180次收益机会。

| 配置 | 节点完成 | model槽 | state槽 | 驱逐次数 | 模型传输bytes |
| --- | --- | --- | --- | --- | --- |
| sa_ghmappo | 59.583333 | 18.187500 | 4.979167 | 1.979167 | 12362.666667 |
| mappo | 59.479167 | 47.895833 | 4.437500 | 23.958333 | 28618.666667 |
| sa_no_prediction | 59.729167 | 28.104167 | 3.604167 | 9.708333 | 17589.333333 |
| cache_offload_drl | 59.854167 | 39.687500 | 3.583333 | 17.541667 | 24416.000000 |
| sa_no_surrogate | 59.791667 | 17.625000 | 3.979167 | 2.000000 | 11776.000000 |

SA模型槽18.1875、驱逐1.979167均低，但节点和请求完成也低于cache基线，且state槽更高，不能据此宣布cache整体优越。需交付约束下的成本/尾时延Pareto和cache专门消融。

主预测真实负例：seed17/dev_2760_tight，奖励35.473391对关闭37.404415（−1.931024），节点47对49，完成请求不变。frame11车辆peachtree:244以proxy+.02向rsu_d准备；frame12状态预约占用d槽，peachtree:137的adapter准入被挡；frame13后者仍adapter_not_ready，对照已input_ready。最终137仅送4节点，对照6；准备者244两侧均6。三次prepare并未增加全窗state槽（两侧均4），仍因时序竞争造成损失。实际源站覆盖在frame16失效，预测frame17，故该反例不能归因“无迁移/预测完全错误”；正proxy漏掉别车新冷模型准入和团队机会成本。原件`prediction_negative_case.json`绑定on/off原始事件，未更新任何模型。

固定权重开关仅检验同一策略的预测消费净效应，不能替代独立训练的no-pred消融；成本/完成率/原始prepare共同检查。理论上的额外信息可被最优策略忽略，但有限预算学习器不是最优策略，预测误差和准备竞争也有成本，因此保留预测不保证实验得分必然更高。

原seed27定位证据仍为：full12856.343598/6.0，关闭surrogate事件头干预后57.837427/7.125；关闭其他头无改善。它证明旧模型存在具体控制干预问题，不等于新训练必然全面领先，也不把本轮bundle变化归因单项。

## 5. 后期回退与信用对照

| episode | dev奖励 | dev完成 | 固定train探针奖励 | 探针完成 | seed27奖励 | seed27完成 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 48.391245 | 3.979167 | 52.953014 | 5.333333 | 48.624955 | 4.062500 |
| 64 | 48.769422 | 4.229167 | 52.288884 | 5.083333 | 48.596403 | 4.187500 |
| 96 | 48.541545 | 4.187500 | 52.578408 | 5.250000 | 48.630548 | 4.187500 |
| 128 | 48.375857 | 4.062500 | 52.613086 | 5.333333 | 48.785089 | 4.312500 |

新seed27的96→128由48.630548/4.1875升至48.785089/4.3125；base57.091030/6.375→57.400112/6.625，tight不变。本轮未复现旧seed27尾段回退；旧surrogate干预定位和新保护是工程证据，但场景/目标也改了，不能将恢复唯一归因单项。

新平均64→128仍48.769422→48.375857、完成4.229167→4.0625。seed7的96→128为48.840327/4.375→48.192531/3.8125，变化来自base；seed17的64→128为49.348550/4.5625→48.149951/4.0625。回退未全面消失，主要seed已变化。固定train探针32→12852.953014→52.613086，也非单调提升；96→128仅+.034678，dev−.165688。不能沿用旧轮“训练持续升、开发持续降”的唯一解释。

末32episode三seed平均normalized entropy .941064、approx KL 8.92e−7、KL早停0；Q MSE .008396、value loss .022791、surrogate loss .007956均下降不等于贪心策略收敛。没有证据支持“KL更新过大”是新回退主因。gamma .99辅助反而48.557613/4.458333高于gamma1，no-surrogate48.697197/4.25也高于full；对齐目标/限制旧干预都未足以兑现整体收益。上述独立训练对照是描述性，不将它们当成仅关闭推理模块的因果量。

原训练gamma=.99将119帧后的奖励乘约.3024，与未折扣评价不同；对齐目标不意味着有限样本必然更易学。global critic缩放修复单位不匹配，不能单凭V/Q loss下降证明策略变好。队列变化、预测开关和独立训练对照的证据范围分别报告，未验证的解释标为假设。

### 尚待验证的团队外部性信用风险

本轮仍保留半local、半team/初始请求数的混合信用，而非精确全局边际贡献。N=8时，自己交付1节点的权重为.5625、其他请求交付1节点为.0625（共同reward scale .1不改9:1比例）。代数反例：让出资源使自己−1、其他请求+2，团队净+1，自己的混合credit变化却为−.4375。它说明混合效用与纯team目标可能不同，不证明本轮每个真实动作都出现该反例。后续须先在固定真实轨迹上做资源让步反事实，再检验team-only或central action-Q；不能据代数例直接改本轮结果或声称已找到全部退化原因。本轮已修的是折扣/单位/代理干预，尚未证明全局协同信用问题完全解决。

### 准备门控仍存在节点/整图目标错位

源码核验：`joint_link_reservation.reserved_migration_opportunities`用`workflow_service_frames(state)`（所有未完成节点的计算与IO总量）形成finish_before/after；state等待收益要求整图finish_after落在剩余时域内，crossing也按整图判断。但实际原始奖励按**送达节点**计1分，并无独立整图完成bonus。因而“整图来不及完成、但准备可多送达节点”的动作可能没有正state proxy并被屏蔽。有界公式witness（不更新模型）：基于既有forecast_state fixture，仅将剩余时域设13、状态payload6425B并补齐公共model rates，partial产生正state候选；当前节点预计交付从14提前至12，跨过13帧边界，reserved却返回空。完整状态/候选/源文件hash见`artifacts/joint_tail_control/objective_gap_witness.json`。它验证公式可漏掉节点交付机会，不等于真实轨迹净收益或实际调度可行性。

该条件错位是源码事实；本轮真实窗口有多少动作被此条件错挡、解除后是否净获益尚未做反事实验证，不将其写为唯一退化原因。缓存cache_net另按model_budget估算，也未自动计全其他请求链路机会成本。

## 6. Claim 边界与后续

**开发 verdict：完整性通过，双指标最高/稳定收敛目标未达。** 旧seed27错误有具体干预修复，新seed27尾段恢复；预测保留并在多站移动条件有局部收益，但primary预测净奖励为负，full仍输独立去预测、去surrogate和缓存基线。当前只支持条件化的探索结果，不能写成算法优势或model-cache普遍优越。

下一轮实现应另立冻结合同，优先：(1)准备净收益加入被挤占车辆的新模型/IO事务、节点deadline与目的站计算拥塞，保留预测并允许拒绝负团队收益准备；(2)以送达节点/真实链路费用对齐state gate，而非仅整图完成；(3)对资源让步做同状态反事实，检验team-only/central action-Q信用；(4)保留本轮强增强MAPPO与cache对照，补定义清晰的普通MAPPO/外部强方法；(5)在容量与机会均核验、合同稳定后再做事前固定训练预算曲线。不能靠降低baseline或挑选窗口拉开差距。本报告审查轮不修改算法或继续筛选训练结果。

model-cache友好需要在完成约束下比较加载、驱逐、等待/尾时延并有独立消融/强baseline。当前显式学习驱逐、多base竞争、真实adapter/无线成本、并发DAG/到达、状态存储上限和控制面数据槽竞争仍未覆盖。surrogate是观测return/声明费用近似，不是完整反事实世界模型；罕见覆盖离开也不自动构成正迁移机会。

formal/holdout封存，canonical不更新；论文E0_UNAVAILABLE / Unverifiable。无新增文献检索，不作novelty结论。场景扩充和多次开发迭代不能替代冻结候选后的独立验证。

## 7. 验证与风险

- 相关回归375 passed，原始输出`artifacts/joint_tail_regression_tests.log`。命令：`python -m pytest tests/test_tail_control_learning.py tests/test_scheduled_control_learning.py tests/test_coordinated_control_learning.py tests/test_parameterized_control_learning.py tests/test_multi_request_target_control_env.py tests/test_verify_joint_target_control.py tests/test_joint_model_cache.py tests/test_multi_request_joint_v2_execution.py tests/test_joint_experiment_launcher.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py -q`。
- `python scripts/verify_joint_scheduled_consumers.py`：旧scheduled 11模型首dev完整事件一致，确认opt-in兼容。
- 1epoch predictor的真实链路SA/DQN各4episode；`python scripts/audit_joint_tail_control.py artifacts/joint_tail_preflight_main artifacts/joint_tail_preflight_dqn --smoke`重跑通过：2checkpoint、26924事件、3627收据、base/tight共4次完整重放。首次相对路径输出错误保留日志，修复后未更改原始模型/事件。
- 新增grant/receipt/cache消费者检查在两份工程16scene原件上通过；语法和git diff检查通过。

- `python scripts/audit_joint_tail_control.py artifacts/joint_tail_control`：exit0，120checkpoint、409883事件、55258收据、60次base/tight重放；48次full同权重预测关闭完成。
- `python scripts/joint_tail_mobility_support.py prepare/evaluate/collect --root artifacts/joint_tail_control`（evaluate逐一指定10个agent）：全部exit0，240policy+24off、198774事件、30重放、0训练。
- `python scripts/audit_joint_tail_capacity.py artifacts/joint_tail_control/mobility_support/manifest.json artifacts/joint_tail_control/mobility_support/capacity.json`：8scene无乐观必要容量违规；primary7/8tight违规保留，不把无违规当充分可行性。
- `python -m pytest tests/test_tail_mobility_support.py -q`：2 passed，合计相关377通过。最终审计helper再跑preflight --smoke exit0，2checkpoint/26924事件/3627收据/4重放保持一致。
- 节点/整图门控有界公式witness断言通过，状态与源文件hash存objective_gap_witness.json；不更新模型。真实负例按frame/vehicle对齐on/off，保存prediction_negative_case.json。
- 所有训练/环境/数据生产源码相对1667da3无差异；语法检查与git diff --check通过。未运行formal/holdout评价，原始数据/历史checkpoint未覆盖。

上述python均使用`/Users/howen/Projects/PPO_MEC/.venv/bin/python`。审计原件存在并可回溯，pytest输出日志不是完整shell transcript，未虚构独立重复试验。

未覆盖风险：正式独立窗口/holdout统计、硬件与实际模型成本、多base/学习驱逐、并行workflow/新到达、完整vehicle/RSU多智能体及外部强baseline。工程通过不消除这些研究限制。
