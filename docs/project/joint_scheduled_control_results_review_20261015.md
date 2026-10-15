# 目的站预约修复轮：完整开发对比与后期回退审查

- reviewed_at：2026-10-15T07:12:57.099481+00:00；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- Git训练source：aa8676a、455bc1d（DQN勘误）；审计/消费者be356ed，加本报告的只读训练差值汇总。evidence level：开发原始artifact核验与首窗重放通过，论文E0_UNAVAILABLE / Unverifiable。
- artifact_run_id（完整12份）：

- `artifacts/joint_scheduled_control/main/seed7/20261015T053901Z_development`
- `artifacts/joint_scheduled_control/main/seed17/20261015T053901Z_development`
- `artifacts/joint_scheduled_control/main/seed27/20261015T053900Z_development`
- `artifacts/joint_scheduled_control/baselines/seed7/20261015T053901Z_development`
- `artifacts/joint_scheduled_control/baselines/seed17/20261015T053901Z_development`
- `artifacts/joint_scheduled_control/baselines/seed27/20261015T053900Z_development`
- `artifacts/joint_scheduled_control/stability/seed7/20261015T053901Z_development`
- `artifacts/joint_scheduled_control/stability/seed17/20261015T053900Z_development`
- `artifacts/joint_scheduled_control/stability/seed27/20261015T053900Z_development`
- `artifacts/joint_scheduled_control/dqn_credit/seed7/20261015T060731Z_development`
- `artifacts/joint_scheduled_control/dqn_credit/seed17/20261015T060731Z_development`
- `artifacts/joint_scheduled_control/dqn_credit/seed27/20261015T060730Z_development`

## 1. Artifact 完整性

本轮固定最终128 episode，不按开发峰值、seed或窗口选择。八主配置各128×3 seed，共3072训练episode；constant-update稳定性对照384；独立发现DQN终止信用缺口后保留的legacy诊断384，实际主实验3840。工程入口smoke另计84，总真实数据训练3924；不含先前轮次，也不把smoke纳入效果比较。

产物根目录为`artifacts/joint_scheduled_control/`：`main`包含SA/MAPPO/no-prediction/PPO，`baselines`包含缓存/DAG/DT及旧DQN，`dqn_credit`为修正DQN，`stability`为恒定更新SA。各part/seed保留manifest、command.log、training/dev、原始事件、费用收据、predictor及checkpoint。冻结计划`comparison_plan.json`未改写；信用纠错以`comparison_amendment_1.json`追加。

全部12份manifest完成；120份checkpoint/tensor互异，268517条事件、38056帧费用收据核验通过。30个最终模型独立加载后，首dev窗口完整事件逐条一致。审计原件：`artifacts/joint_coordinated_audit_20261015/joint_scheduled_control_audit.json`，固定贪心/更新诊断：同目录`joint_scheduled_control_training_diagnostics.json`。

## 2. Protocol / provenance

原始source为aa8676a8c87456c87d57ffde0d9af641b9e378dc；DQN纠正为455bc1d8aa5bb15da7d2fae9372bea010f39afc4。全部启动时Git clean。勘误前后仅DQN算法源变化，14个共享环境/预测/PPO/registry模块及`episode()`源码hash一致。先有独立审查`joint_dqn_terminal_credit_review_20261015.md`，再做修复，不以DQN得分选择新旧版本。旧模型默认不消费terminal tail，修正模型显式启用；对应初始tensor相同。

所有方法共享NGSIM+Alibaba原manifest、20train/8dev、八车四RSU、同奖励和资源。GRU统一8帧预测视野、30epoch、history8/hidden16，拟合16train、验证4train；共同bundle hash为`12f8d9193e0ca01bc4d773e5cbf8a268d5a8128fd0191f312b17c15e3d9f4021`。视野由原3帧延到8帧是在开跑前根据约7帧状态传输预算冻结；旧轮与本轮不混表，不把全部增益归因单项修复。

当前一数据槽/RSU预约遵守owner/kind，后台模型传输不抢占；未来预约仍是因果声明估计。共享JSON计原字节费用但不占数据槽。成本为合成帧级预算，不是实测无线/设备成本。真实未来仅在离线影子诊断使用，不进入actor。

## 3. 最终比较与baseline公平性

| 配置 | 奖励均值 | seed均值样本标准差 | 完成请求/8 | seed7/17/27奖励均值 |
|---|---:|---:|---:|---|
| sa_no_prediction | 58.226779 | 0.478411 | 7.250000 | 58.503835/58.502144/57.674358 |
| sa_ghmappo | 57.981931 | 1.420026 | 6.958333 | 58.743024/58.859170/56.343598 |
| mappo | 57.732622 | 0.969583 | 7.041667 | 57.038380/57.319078/58.840409 |
| ppo | 57.661388 | 0.469524 | 6.916667 | 58.152303/57.216670/57.615190 |
| dt_handoff_drl | 57.583436 | 0.529583 | 7.000000 | 57.357787/57.204050/58.188470 |
| dag_offload_drl | 57.210631 | 0.320786 | 6.541667 | 57.526874/57.219532/56.885488 |
| cache_offload_drl | 56.762605 | 0.350327 | 6.375000 | 57.155105/56.651129/56.481581 |
| dqn | 53.964010 | 2.344833 | 4.708333 | 51.607643/56.297137/53.987250 |

独立support（不参与主排名）：

| 配置 | 奖励 | 完成请求/8 |
|---|---:|---:|
| sa_constant_update | 57.070868 | 6.458333 |
| dqn_legacy_terminal_credit | 52.953722 | 4.666667 |

| 完整版减比较组 | 奖励差 | 完成请求差 | 24单元奖励胜/负/平 |
|---|---:|---:|---|
| mappo | +0.249309 | -0.083333 | 16/8/0 |
| sa_no_prediction | -0.244848 | -0.291667 | 15/9/0 |
| dt_handoff_drl | +0.398495 | -0.041667 | 16/8/0 |

完整版在原七种算法中奖励均值第一，但去预测消融奖励更高，且完整版完成率低于MAPPO、DT和去预测版。**完整版同时超越MAPPO与去预测版的奖励/完成率目标：本轮未达到。**

每个主配置都是24个seed-window单元的描述性均值。同环境交互episode数相同；PPO与DQN优化方式/梯度步数不同，不能称梯度预算完全相同。相邻窗口、同批车辆/DAG不作为24个独立cluster；不报告虚构的显著性。

缓存、DAG、DT三种`*_drl`是仓库内PPO方向型实现：分别使用typed-cache池化、DAG统计、原始预测/负载特征，不是外部论文逐式复现。DQN是独立replay/target-network/Bellman实现；PPO、MAPPO也均已接入相同参数化execution/cache/event合同。QMIX/MAT/IPPO/DDQN等仍为旧五动作合同，在manifest逐项标明排除原因；不能拿其旧奖励拼入新表。TD3/SAC/MADDPG仍未稳定，不进live registry。

## 4. 机制兑现

- 预测保留。固定最终full权重，仅开关预测消费，24单元奖励平均+0.0834215、完成请求+0.0833333（合计多2）；1胜1负22平。它是固定策略支持核验，不能替代独立训练的no-prediction消融。
- 七次prepare全部实际推进，共7168B/7链路槽。seed27/offset3120多完成2请求；seed17同窗多用一槽、无完成收益，奖励−0.0100166。正净收益proxy不是实际收益保证；整体预测分支干预不能归因到某一条state动作。
- 真实/预测机会为24/21请求帧，去重后只有1个外生离站事件。说明本开发集对预测机制的覆盖仍弱，不支持广泛的迁移优势。
- 三个full最终checkpoint的首dev初始70节点graph tanh饱和比例均0，归一化后input/output最大0.398171/0.196210。范围是初始图witness，不能推断所有网络/rollout均不饱和。
- 首窗chosen局部return surrogate误差（审计定义）在三seed为RMSE0.646–0.748；费用估计也仍有误差，不能称准确的反事实环境模型。
- 动作条件Q以chosen on-policy return拟合，actor基线为同prefix的policy-average Q；surrogate使用含费用的相对return，避免重复扣费。这是学习的反事实近似，非真实环境oracle，预测误差与信用估计误差仍存在。

最终缓存费用/收益对照（每窗均值）：

| 配置 | 交付节点 | 完成请求 | 模型链路帧 | 状态链路帧 | 未来模型命令 | 驱逐次数 |
|---|---:|---:|---:|---:|---:|---:|
| full | 68.791667 | 6.958333 | 32.500000 | 5.166667 | 14.416667 | 10.333333 |
| MAPPO | 68.791667 | 7.041667 | 31.625000 | 6.791667 | 9.166667 | 8.875000 |
| no-pred | 69.125000 | 7.250000 | 39.083333 | 6.458333 | 21.333333 | 15.083333 |
| cache baseline | 67.916667 | 6.375000 | 25.500000 | 6.583333 | 2.041667 | 4.416667 |

完整SA优于本仓库cache基线的reward/请求，但费用并非全低于MAPPO或cache。提前准备、拉取更少与交付更多是不同指标，尚无独立cache消融/外部强baseline证明cache处理机制优越。

## 5. 训练后期回退

| episode | SA dev奖励/请求 | MAPPO dev奖励/请求 | no-pred dev奖励/请求 | SA greedy train probe奖励/请求 |
|---|---|---|---|---|
| 32 | 58.252568/7.250000 | 56.982484/6.500000 | 58.218707/7.208333 | 59.387543/7.916667 |
| 64 | 58.155290/7.166667 | 57.742424/6.916667 | 58.092291/7.125000 | 59.791635/8.000000 |
| 96 | 58.266082/7.208333 | 57.461136/6.833333 | 58.175236/7.166667 | 59.732479/8.000000 |
| 128 | 57.981931/6.958333 | 57.732622/7.041667 | 58.226779/7.250000 | 59.820370/8.000000 |

训练probe为事先固定的四个train窗口，dev为全部八窗口；各点分别12/24单元，不能把二者数值差直接解释为同分布泛化gap。完整版train probe上升、dev回落的方向仍清晰；去预测最终较32奖励+0.008072、完成请求+0.041667，未出现与完整版相同的整体回退。

完整版96→128时，dev平均计算排队164.125→167.333，服务站变更2.208→2.083，交付节点69.083→68.792；逐seed/window原始差值保存在训练诊断JSON的`final_minus96_paired_dev`，不能把队列相关性写成唯一因果机制。训练与开发方向相反，比“训练总量不够”的单一解释更符合当前证据；未证明扩大同20个训练窗口的重复次数会解决问题。

完整版与constant-update从相同tensor初始化，保存options仅`stability_enabled`不同。完整版最终奖励高0.911063、完成请求高0.5，但完整版自身96→128仍回落0.284151。KL早停次数为0，full最大approx KL=0.00011244，远低于0.02阈值；本次不能声称是KL保护产生了收益。学习率首/末8次更新均值0.000269516/0.000060484，末段normalized entropy仍0.922892，Q MSE和surrogate loss下降也没有保证dev策略改善。四个检查点不足以宣称收敛，高熵和微小更新也不能排除有效学习不足；尚未测量greedy logit margin，不能把argmax脆弱性当已证实原因。

回退主要来自seed27：SA奖励32/64/96/128分别57.334734/57.346930/57.009508/56.343598，完成6.75→6.0；seed7为58.863485→58.743024，seed17为58.559485→58.859170。完整版对去预测的seed均值差+0.239189/+0.357026/−1.330760，15/24奖励胜却均值落后；这首先是跨seed稳定性缺口，不得挑掉seed27。seed27在offset2760/2880/3000后期各多23帧计算队列、少1个交付节点/请求；这些相近窗口不能作独立重复证据。

DQN也有不同形态的回退：32→128 dev奖励−3.100994、固定train probe−5.071478；其TD value loss下降不保证控制回报提高。它不是单纯train改善/dev下降的相同现象，本轮未识别DQN唯一退化原因。

奖励正项按交付节点计，完整workflow没有独立正奖励项；因此奖励排序与完成请求排序不必一致。保持本轮原奖励；不得为追认胜利而事后加completion bonus。缓存传输减少也可能伴随交付减少，需同时检查完成率、节点、费用和deadline，而不能单凭少传字节宣称model-cache更优。

## 6. Claim 边界与下一阶段任务

本轮修复的是可执行目的站链路预约、声明净收益与动作信用、图/候选尺度，并恢复共同合同baseline。主方法双指标领先和收敛仍须以本轮最终表判断。formal/holdout封存、canonical不更新，论文证据E0_UNAVAILABLE / Unverifiable。

后续独立实现应先围绕本轮负例做可复现witness：完整请求尾部交付/计算队列的信用是否足够、预测机会在训练中是否过稀、训练窗口泛化与高熵策略贪心边界是否稳定。优先扩充机制覆盖不同的训练窗口和跨场景验证，保留全部共同资源/费用；不要直接把同一批窗口重复更多次当成解决方案。对deadline/critical-path输入、未使用预约释放或完成目标的改动须另立合同、共同适配baseline并重新冻结，不能在本审查轮修改已有结果。

model-cache友好仍是研究目标。需要显式可学习驱逐/多base竞争、cache强规则或外部强基线、相同完成约束下的加载/命中/尾时延证据。当前单base合成adapter、串行DAG、单批到达、无状态存储上限及控制面不竞争数据槽等限制仍保留。没有新文献检索，本轮不作novelty结论。

## 7. 验证与未覆盖风险

使用项目解释器`/Users/howen/Projects/PPO_MEC/.venv/bin/python`执行：

- `-m pytest tests/test_scheduled_control_learning.py tests/test_coordinated_control_learning.py tests/test_parameterized_control_learning.py tests/test_multi_request_target_control_env.py tests/test_verify_joint_target_control.py tests/test_joint_model_cache.py tests/test_multi_request_joint_v2_execution.py tests/test_joint_experiment_launcher.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py -q`：363 passed；另scheduled目标回归17 passed，覆盖终止tail/相同初始化见证。
- `scripts/verify_joint_scheduled_consumers.py`：11个工程checkpoint的registry加载、fresh初始化及首dev完整事件消费通过。clean AA的SA/DQN各4episode、clean455的DQN tail4episode入口验证完成；全部工程smoke84episode单列。
- `scripts/audit_joint_coordinated_control.py joint_scheduled_control`：exit0，12个manifest、120 checkpoint、268517事件/38056收据、30final首dev完整重放通过。
- `scripts/analyze_joint_scheduled_training.py joint_scheduled_control`：exit0，所有固定train/dev曲线、逐窗口96→128差值、训练机制首/末32段、三组稳定更新初始tensor与唯一option差异核验通过。
- `scripts/probe_joint_coordinated_prediction.py joint_scheduled_control/main`及对应`audit_joint_prediction_effects.py`、`audit_joint_prediction_opportunities.py`、`audit_joint_feature_scale.py`支持核验通过；原件和命令日志保存在同一audit目录，报告见`joint_scheduled_prediction_support_review_20261015.md`。
- 修改模块`py_compile`、`git diff --check`通过。训练源在aa8676a/455bc1d冻结，最终回收阶段只改只读诊断与文档，无新增训练/选峰值。

未覆盖：formal/holdout、独立非重叠外层窗口统计、硬件成本、真实adapter/多base、并发DAG/到达过程、全vehicle/RSU-level MARL及外部论文复现。上述限制不由工程测试通过而消除。
