# 团队节点交付 / 预测准备保护 v2：完整开发结果审查

- `reviewed_at`: 2026-10-16（Asia/Shanghai，客户端日期）；原自动审计UTC时间 `2026-10-15T14:33:07.885010+00:00`。
- `literature_cutoff`: 2026-10-15（沿用既有登记；本轮无新增文献检索，不作novelty评价）。
- `target_venue`: IEEE TMC（规范目标，不代表投稿达标）。
- `artifact_run_id`: `artifacts/joint_delivery_control_v2`，30个最终模型、十配置，具体子run见`audit.json.artifact_run_id`。
- `policy_version`: `tmc_review_policy_v3_20260621`；训练/评价 Git commit `9b9c0d72a1d6740037f6b2785ea5380d257335b2`。
- `evidence_level`: 开发原始结果/账本已审计，关键模型独立事件重放；正式论文证据 `E0_UNAVAILABLE`，verdict **Unverifiable / 不晋级**。
- 审查轮只读算法、费用和原产物；新增零更新诊断/独立汇总，不重训、不调参、不取峰值；canonical及formal/holdout保持原状。

## 1. Artifact 完整性

`launcher_result.json.status=complete`；3840/3840回合、10/10group、主/移动/历史缓存审计exit均0。原始主评估审计120个checkpoint、389044条事件、55741份收据，60次最终模型base/tight首窗重放及逐帧行为概率核验一致。移动补充240policy+24prediction-off单元、182332事件、30次首窗重放；旧cache48/48完整事件匹配。

本次独立从十组`training/dev/train_probe.jsonl`重新重建并核对主均值、分层、seed SD、曲线和paired差：3840训练、480最终单元、1920开发曲线单元、480固定训练probe单元及120 checkpoint SHA均通过；逐个复核冻结生产源码指纹、manifest clean9b、预算、三审计终态。主/补充所有最终policy单元的vehicle fallback均0。

独立汇总与文件SHA目录：`artifacts/joint_delivery_control_v2/post_training_review/analysis.json`；完整事件/费用/真实数据hash与分割审查见原主审计，不能只凭本报告文字认定复现。

新主比较八配置各128×seed7/17/27，共3072训练；no-surrogate / fixed-entropy各384，共3840。每32回合保存/评价，固定最终128、不取dev峰值，不删除困难panel/seed。主表包含48seed-scene单元/配置，十配置480最终单元及1920开发曲线单元。

### 中止试跑与工程成本

v1/cad6509停止原因见独立审查ea81c0e：仅最后节点投影第4帧完成、预测第6帧离站，却因重新计恢复延迟得到+.04准备收益。依据有界实现错误停止，未依据v1策略得分。原root `artifacts/joint_delivery_control/`保留计划/manifest/部分checkpoint和记录316个训练episode；最多10个未完成/未记录另列，不能当完整128对比，不resume或拼接。

工程v1两轮各SA/DQN4回合合计16；v2/f5fe81c与9b9c0d7各8合计16，共32。新主3840已全部完成，本次delivery v1/v2修复及工程核验专项已记录真实RL总4188（3840+316+32），另≤10未完成/未记录试跑交互；不宣称总计算只花3840，不包括pytest toy、零更新重放/评估。v1不完整数据与工程均值均不进主表。

## 2. Protocol / provenance

- 新完整训练source `9b9c0d72a1d6740037f6b2785ea5380d257335b2`；十group启动均clean，训练中生产源码按comparison_plan的文件SHA冻结。
- 同一40train /16dev、原始NGSIM/Alibaba帧与8车配对、4RSU位置/容量/费用、公共GRU30epoch bundle逐字节复制自tail。无新数据挑选/拟合/奖励bonus；GRU SHA `b78be998d13fff145d5c1f249cad5f33af7206a153d6d5259c9d532e16d44ceb`。父manifest及原source SHA沿链验证。
- 主结果base/tight各50%，两panel同一8dev原始时间，不能当16独立窗口。tight7/8必要容量上界不可全8完成，保留过载结果/原权重；无违规的base也不等于证明所有调度可行。
- 训练scene循环与旧tail相同；每scene仅3–4次训练曝光，同预算不等于足量或已收敛。所有方法同episode预算，不宣称DQN/PPO/aux优化计算、sample或墙钟完全相同。
- 新checkpoint `delivery_parameterized_ppo_v2` / `delivery_parameterized_dqn_v2`明确拒绝v1，不是完整resume快照。

### 本轮修复范围

先采样全部执行站，再按实际选择保护reactive模型准入、restore、input与估计compute/output。预测prepare只用未占当前及估计未来槽；保留mandatory未来phase。异步model依赖闭包要求连续可用span；可选LRU预取按观测保护同帧当前依赖，未知remote occupancy不放可选预取。当前必需miss即时加载，未来/远站可选预取保留4帧周期。

净收益按团队节点/原等待和额外物理费用估算，独立cache/state prefix，不把未选model准备并入state收益；原站预计在离站前已交付的prefix保留原收益/时间，所有节点可在离站前结束则无预测准备。source/destination当前节点队列、预计输出增长和下个未交付节点模型成本仍是估计。绝对值<1e−9归零避免8.7e−18数值假正收益；不改raw reward。

全头/critic/surrogate用team MC/N/10，gamma1，DQN早终止保留观察到的完整团队tail。SA吸收cache基线context，零初始化`.25*tanh(gate)`DAG残差；初始actor与cache相同。surrogate保留return/费用学习，作policy-average Q控制变量，不给logit加观测回报bonus；actor/critic和aux独立Adam/.5裁剪。Actor LR3e−4→3e−5，aux LR3e−4；entropy .01→.001，对照保持.01。仍不能据此宣称精确因果边际信用/世界模型。

## 3. Baseline 公平性和固定最终结果

| 算法 | 奖励均值 | 奖励seed SD | 完成请求 /8 | 交付节点 | 模型链路帧 | 状态链路帧 |
| --- | --- | --- | --- | --- | --- | --- |
| dag_offload_drl | 48.645926 | 0.133001 | 4.937500 | 59.520833 | 47.000000 | 4.625000 |
| cache_offload_drl | 48.612738 | 0.150846 | 4.875000 | 59.375000 | 32.833333 | 5.291667 |
| dt_handoff_drl | 48.580330 | 0.247837 | 4.875000 | 59.458333 | 44.541667 | 4.604167 |
| sa_ghmappo | 48.544345 | 0.415356 | 4.875000 | 59.354167 | 44.812500 | 4.645833 |
| sa_no_prediction | 48.433218 | 0.212753 | 4.750000 | 59.208333 | 39.916667 | 5.354167 |
| mappo | 48.181045 | 0.329284 | 4.958333 | 59.062500 | 45.479167 | 3.895833 |
| ppo | 48.041704 | 0.381307 | 4.708333 | 58.958333 | 50.833333 | 3.520833 |
| dqn | 44.962501 | 0.226627 | 3.291667 | 56.458333 | 24.895833 | 12.479167 |

SD由三个seed各自16场景均值计算，非48独立样本的标准误。完整版在八主配置中奖励第4；相对cache奖励−0.068394、请求持平；相对no-pred奖励+0.111126、请求+0.125；相对MAPPO奖励+0.363300、请求−0.083333。相对旧强cache奖励−0.206207、请求+0.166667、节点−0.500000。**没有奖励和完成率同时领先的结果。**

| 算法 | base奖励 | base完成 /8 | tight奖励 | tight完成 /8 |
| --- | --- | --- | --- | --- |
| dag_offload_drl | 57.662858 | 6.916667 | 39.628993 | 2.958333 |
| cache_offload_drl | 57.437992 | 6.833333 | 39.787484 | 2.916667 |
| dt_handoff_drl | 57.682071 | 6.833333 | 39.478588 | 2.916667 |
| sa_ghmappo | 57.817726 | 6.833333 | 39.270963 | 2.916667 |
| sa_no_prediction | 57.197594 | 6.583333 | 39.668842 | 2.916667 |
| mappo | 57.857935 | 7.000000 | 38.504154 | 2.916667 |
| ppo | 57.194013 | 6.541667 | 38.889394 | 2.875000 |
| dqn | 51.414834 | 4.166667 | 38.510168 | 2.416667 |

两panel必须同时披露：完整版base奖励高于cache、tight奖励低于cache；MAPPO在base完成请求更高。保持原50:50权重，不用单个panel替代总表。

| 独立训练辅助配置 | 奖励 | 完成 /8 | 节点 |
| --- | --- | --- | --- |
| sa_no_surrogate | 48.660759 | 4.916667 | 59.416667 |
| sa_fixed_entropy | 48.189376 | 4.666667 | 59.041667 |

去surrogate高于完整版0.116415奖励、0.041667请求；固定entropy低于完整版0.354968奖励、0.208333请求。是三seed开发描述性方向，不能把辅助均值写成显著因果证明。

本轮MAPPO是公共GRU、保护、队列先验增强对照；cache/DAG/DT为first-party PPO方向实现，DQN为Bellman/replay。SA相对cache主要增加有界图残差与surrogate控制变量，相对no-pred保留预测消费。普通MAPPO、忠实外部方法、full vehicle/RSU MARL没有静默替换进来。当前“完整”指原仓七算法＋去预测主配置。

### 额外旧强缓存对照

旧强cache固定最终奖励 **48.750551860**、完成请求 **4.708333333/8**、交付节点59.854166667。48/48单元完整事件精确相等、三seed最终checkpoint hash确认，额外训练0。审查`artifacts/joint_delivery_control_v2/legacy_cache_reference/audit.json`，SHA256 `d54d9324bcdeb7b0628cd706c5dcd7f6a7cfdd7a4bc19641aaa8b11ef6236aec`。

该对照为旧controller/credit合同、旧128×3训练预算、固定最终模型在字节相同物理协议上的重放，0新增更新；和新受保护cache分别标注。不能因新公共保护/目标改变后cache分数变化就宣称主方法胜过原强基线。全部48单元原始事件匹配才确认可比；不混作新训练，也不作为独立统计样本。

## 4. 预测与缓存机制

主评估full：48单元总预测机会0、prepare0，同权重预测on−off全部48平局，奖励/请求/节点/模型帧/状态帧均不变。独立no-pred训练虽低于full，**该差不能解释为主评估预测准备兑现**。训练期间full仍有411条候选机会、194次准备；机制未删除，但训练/主dev的实际可用机会存在落差。机会/prepare计数包含跨帧重复，不是独立迁移样本数；几何handoff不等于有净收益的准备机会。

移动补充full：148条候选、99次prepare、8次prepared-on-arrival；同权重24单元on−off奖励+0.071534、完成请求0，10胜/4负/10平。最大收益seed27/dev3120 +0.812690，状态帧55→39，70节点/8请求不变；最大负值seed17/dev3600 −0.135532，13次prepare、状态帧50→65，仍70节点/8请求，说明净收益估计仍会接受实际无收益准备。prepared-on-arrival也可能来自reactive或既有拷贝，不可当99次都成功利用。

已知资源保护：389044主事件及182332移动事件的prepare当前槽、模型依赖闭包连续span、可选LRU驱逐当前团队模型依赖均经独立raw检查通过；源预计离站前全交付的机会未被放行。该结论限于已知/预约资源与本次运行，未知新需求、延迟快照和实际未来排队未被证明不会受损。主full没有prepare，不能用主零冲突单独证明保护改进；移动99次prepare提供非空检查。

模型cache目标未兑现：full模型帧44.812500 vs cache32.833333，**+36.48%**；未来节点cache命令26.083333 vs13.062500，几近两倍。节点59.354167 vs59.375000、请求均4.875；状态帧4.645833 vs5.291667的小幅节省没有弥补多付模型成本。移动full模型帧31.625 vs cache28.750，且full完成7.958333低于8；不能写成model-cache更优秀。typed-cache仍自动依赖安全LRU，本轮不是学习驱逐/多base真实竞争证据。

目标是保护已知事务并改善团队实际交付，guard依赖共享观测/声明catalog；未知未来、延迟快照、模型/compute队列估计仍可能导致净收益误判。prepare命令/几何离站数不等于独立有益迁移；actual typed-cache成本低也不能在交付下降时单独证明cache优越。

## 5. 后期回退与稳定性

| 回合 | dev奖励 | dev完成 | 固定train奖励 | 固定train完成 | seed7 dev | seed17 dev | seed27 dev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 48.715237 | 4.875000 | 54.564796 | 6.083333 | 48.695972 | 48.495155 | 48.954584 |
| 64 | 48.608666 | 4.958333 | 54.919510 | 6.250000 | 47.886479 | 48.754706 | 49.184814 |
| 96 | 48.404665 | 4.791667 | 54.849431 | 6.250000 | 47.748260 | 48.468106 | 48.997628 |
| 128 | 48.544345 | 4.875000 | 55.020107 | 6.333333 | 48.172389 | 48.468106 | 48.992539 |

完整十组曲线/seed/panel/update指标保存在`audit.json.curves`和独立`post_training_review/analysis.json`。full 64→128：seed7恢复+0.285910，seed17回退−0.286600，seed27回退−0.192275；32→128则seed7下降−0.523584、seed17微降−0.027049、seed27微升+0.037955。96→128 seed7恢复，seed17不变，seed27微降；不能沿用“只seed27出问题”或“回退已解决”。固定train probe32→128上升54.564796→55.020107，但不是全部训练集，也不足以单独证明过拟合。

后期仍未解决。full末段normalized entropy约0.940，KL约4.13e−7，无KL提前停止，actor梯度约0.054；surrogate梯度约0.00294，没有本轮梯度爆炸或探索熵塌缩的直接证据。图残差的平均实际系数约0.001957（日志已是`.25*tanh(gate)`，勿再乘.25），主机制相对cache的图增量很弱。value/surrogate/Q误差下降而dev未持续上升，不能据此称收敛。

有界复核：事后按seed27最大64→128奖励下降选`dev_3120_base`作诊断，**不选最终模型、不改主表**。独立加载原ep64/128，重放两次，原奖励和checkpoint/tensor SHA均一致。首帧peachtree:1689执行站rsu_b→rsu_d；同一前缀下两个最优执行候选概率差由0.001616翻转为0.013361。交付67→65节点、输入排队152→185、状态等待38→55、状态帧18→25，奖励55.581335→53.581141；未完成等待惩罚同为−8.44，净损失主要是2个节点，非预测prepare（两次均0）。此例支持“相近选站概率发生翻转，队列与迁移路径随后变化”的轨迹解释；没有单独干预每个后续动作，不能断言这一个首帧动作是全部损失的唯一因果。证据`post_training_review/late_case.json`、两次trace和命令日志。

共同团队MC目标修正了单位/尾部不一致，仍给同一帧多个请求/控制头共享目标；未识别某次预取/换站对其他车辆节点交付的精确边际作用。surrogate当前是观测回报基线，去surrogate更好加上多余预取，只构成优先校准控制变量及cache信用的诊断依据。固定entropy辅助在seed27最终48.142183更差；简单取消退火不是现有证据支持的完整解法。

固定train probe与dev曲线分开；报告所有三个seed，不沿用旧seed27为唯一原因。surrogate/value/Q loss下降、熵下降、小KL或单点最高均不证明收敛；固定entropy与no-surrogate独立训练对照也不等于同权重推理开关的因果效应。

## 6. 预声明移动分层

| 配置 | 移动奖励 | 完成 /8 | 节点 | 模型帧 | 状态帧 |
| --- | --- | --- | --- | --- | --- |
| mappo | 60.312744 | 8.000000 | 70.000000 | 31.000000 | 25.666667 |
| dt_handoff_drl | 60.243438 | 7.958333 | 69.958333 | 31.291667 | 27.958333 |
| dag_offload_drl | 60.241491 | 8.000000 | 70.000000 | 31.666667 | 27.250000 |
| ppo | 60.174119 | 7.958333 | 69.958333 | 32.041667 | 27.958333 |
| sa_no_prediction | 60.167002 | 8.000000 | 70.000000 | 31.458333 | 27.750000 |
| sa_fixed_entropy | 60.138149 | 8.000000 | 70.000000 | 31.083333 | 33.916667 |
| cache_offload_drl | 60.070319 | 8.000000 | 70.000000 | 28.750000 | 32.875000 |
| sa_ghmappo | 60.028204 | 7.958333 | 69.958333 | 31.625000 | 35.416667 |
| sa_no_surrogate | 59.996675 | 8.000000 | 70.000000 | 28.875000 | 33.375000 |
| dqn | 50.804106 | 4.000000 | 62.708333 | 25.541667 | 37.000000 |

完整版60.028204/7.958333，低于MAPPO60.312744/8、去预测60.167002/8、cache60.070319/8。相对同权重off的小收益不能替代与独立训练基线比较。唯一full未全完成单元是seed7/dev3240_mobile_base，车辆peachtree:560完成8/9节点；预测off同样未完成，不能归咎于预测。

本轮启动前已固定复用旧mobile_base的8个cohort和原宽覆盖、全程≥2站可达条件，0新增训练/重新选车/调阈值。上一轮晚定义exploratory身份保留，当前是预声明secondary dev；同一8原始时间，不是独立新窗口/holdout，不替代主base/tight权重。十组最终30模型全部评估240单元，full另24同权重预测关闭，并核验原始成本/资源约束和30次首窗重放。

## 7. Verdict / claim 边界

**工程检查通过；性能与机制目标未达成；收敛未证实。** 主full超过no-pred的均值但未超过新/旧cache奖励，未超过MAPPO完成率；移动补充也未领先。资源保护在已知预约范围兑现，预测仅在移动补充有微小推理收益、没有请求提升，model-cache成本和surrogate优势未兑现，seed17/27在64→128仍回退，seed7有早段下降后恢复。

安全表述：本轮固定128×3开发对比得到上述描述性结果。禁止表述：预测必然有用、model-cache友好已全面成立、训练已收敛、主方法双指标最优、忠实复现外部baseline或达到TMC-ready。正式formal/holdout证据仍缺，按policy在完整性阶段即判论文`Unverifiable`；novelty/七维评分均N/S（未补正式证据、未做新文献轮），不报伪精确总分。

完整性、机制兑现和性能目标分别判定；研究愿望不是保证。不能靠降低baseline、改变惩罚/资源、删困难样本或选dev峰值制造领先。formal/holdout未评价，canonical不变，paper E0_UNAVAILABLE / Unverifiable；不作novelty/顶刊评分结论，无新增文献检索。

仍未覆盖：真实base/adapter/无线成本、多base真实竞争/学习驱逐、动态新到达/并行DAG、完整vehicle/RSU MARL、外部强baseline及独立formal/holdout。内部轨迹驱动仿真开发结果不能直接当论文普遍优势。

## 8. 验证命令、结果与剩余风险

解释器`/Users/howen/Projects/PPO_MEC/.venv/bin/python`。398回归、旧consumer兼容及clean preflight为训练前已完成验证；本次新增的是原始数值独立重建、两次后期诊断重放和报告逐值验证。

- `python -m pytest tests/test_delivery_control_learning.py tests/test_tail_control_learning.py tests/test_scheduled_control_learning.py tests/test_coordinated_control_learning.py tests/test_parameterized_control_learning.py tests/test_multi_request_target_control_env.py tests/test_verify_joint_target_control.py tests/test_joint_model_cache.py tests/test_multi_request_joint_v2_execution.py tests/test_joint_experiment_launcher.py tests/test_algo_pool_contract.py tests/test_checkpoint_compat.py tests/test_tail_mobility_support.py -q`：398 passed，日志`artifacts/joint_delivery_v2_final_release_tests.log`。
- `python scripts/verify_joint_scheduled_consumers.py`：旧scheduled11模型完整事件一致；只读遗留兼容。
- 最终clean9b SA/DQN各4回合，`python scripts/audit_joint_tail_control.py artifacts/joint_delivery_v2_release_preflight/sa_ghmappo artifacts/joint_delivery_v2_release_preflight/dqn --smoke --delivery`：2checkpoint/25557事件/3674收据/4个base/tight完整事件及逐帧policy logprob重算一致，exit0。工程得分不作结论。
- 旧负例两分支按原动作精确回放47/49节点，新calendar拦三次有害prepare；新有界prefix/finish-before-exit/zero-net/即时miss与v1拒绝等检查通过。公式与单例不能取代新完整比较。

- `python scripts/launch_joint_delivery_comparison.py`：本轮固定入口已完成；原始command JSON/log及`launcher_result.json`保留。
- `python scripts/audit_joint_tail_control.py artifacts/joint_delivery_control_v2 --delivery`：自动收尾已成功，120checkpoint/389044事件/55741收据/60次重放及行为概率通过；没有为本次文字汇总重复跑全审计。
- `python scripts/evaluate_joint_delivery_support.py collect --root artifacts/joint_delivery_control_v2`：自动收尾已成功，240policy+24off/182332事件/30重放通过。
- `python scripts/audit_joint_delivery_cache_reference.py --root artifacts/joint_delivery_control_v2`：旧强cache48完整事件匹配，0训练。
- 本次 `python artifacts/joint_delivery_control_v2/post_training_review/summarize_results.py`：从原始JSONL独立重算主表/分层/曲线/paired及SHA/预算/无fallback，全部通过，日志`post_training_review/reconstruct_command.log`。
- 本次 `python artifacts/joint_delivery_control_v2/post_training_review/replay_late_case.py`：seed27两次原模型独立重放匹配，0更新，原checkpoint/tensor未改变；诊断选择已显式标为post-hoc。

- 本次 `python artifacts/joint_delivery_control_v2/post_training_review/verify_report.py`：主表、panel、辅助、曲线、移动表共32行逐值与独立重建JSON一致，冻结输入hash及报告链接检查通过。

## 9. 下一阶段独立实现任务（本审查不自动启动）

1. **优先补未来模型的团队边际价值和成本**：逐base/adapter候选估计被后续未交付节点使用的时刻、驻留/被驱逐概率、连续依赖闭包传输与占用成本、对同站其他请求交付的影响；不能用当前节点migration proxy给未来adapter假收益，也不能仅按“空闲槽可用”认定值得预取。对照真实收据校准，保留当前保护。验收应同时看团队节点/请求和模型链路成本。
2. **校准surrogate/动作信用**：同一团队交付单位下检查不同头/当前与未来cache的排序与方差；为候选动作构造预算内可追溯的延迟交付/外部性目标，控制变量按误差/方差可靠性消费。保留predictor与共享状态；不把关掉模块当主方法最终方案。先做有界诊断和单变量匹配对照，不凭三seed均值宣称确定因果。
3. **稳定选站决策并增加有效学习覆盖**：首帧近似并列站会被极小更新翻转；先核验候选价值排序/不确定性和跨车队列外部性，再考虑有成本依据的驻留/迁移一致性。不得用评估专用bias锁住“更好”站。128回合在40场景上仅3–4次曝光，高熵/弱图残差表明尚不能排除训练不足；增加预算必须所有方法同步、事前冻结、独立于这批dev的峰值。不是直接承诺多训必胜。
4. **保留预测、按真实机会校准净收益**：训练有机会而主dev无机会，移动support仍有4个负单元；核对剩余节点交付时域、状态增长/重复更新、目的站队列和实际可用时隙的预测误差。先检验校准，再决定训练数据的机会覆盖设计；固定主/补充结果全部保留，不切换权重制造领先。

上述按工程第一因定位排优先级；资源/奖励/数据不为拉开排名而单边改变。任何新实现与实验另立合同并独立提交，不在本只读审查轮边看分数边修算法。

