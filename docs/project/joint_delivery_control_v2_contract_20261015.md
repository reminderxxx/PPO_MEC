# Delivery v2：保留离站前交付的独立修复合同

日期：2026-10-15。先前审查`joint_delivery_completion_guard_review_20261015.md` / ea81c0e已独立输出，现执行用户授权的修复。v1/cad6509在316个已记录训练回合后停止，另最多10个未完成/未记录episode；不resume、不补接，不用其不完整结果选模型或排名。

## 唯一新增机制修复

- 原站投影纳入当前模型剩余加载与已选本站他者当前节点计算队列；若所有节点预计在coverage离开前已交付，不生成准备机会。
- 已在离站前预计送达的prefix保留原时间/奖励，不再重复施加arrival/recovery延迟；仅对离站后剩余节点比较reactive与prepared的交付和原等待效用。
- 按预测时刻的首个未交付节点估计目的站公共模型恢复成本；已经在离站前用完的当前adapter不获得预测预取收益。未来DAG对象仍可由显式cache头选择，其当前节点migration proxy不冒充未来adapter净收益。
- 状态增长考虑ETA内预计生成的输出，当前pending output不重复加入payload增长。全部仍是因果观测+冻结预测的估计，不读轨迹真未来，也不声称精确反事实。
- Utility相减的绝对值<1e−9按0处理，避免数学零净收益因8.7e−18舍入误差被放行；固定数值容差，不改变物理reward，不按policy得分选择。
- 当前执行必需的冷模型miss即时准入，未来/远站可选预取仍按4帧周期；否则已预留的reactive模型槽会因旧缓存时钟空耗。连续span和当前依赖保护照常检查，所有基线共同生效。
- 保留v1两阶段执行站/冷准入/接续IO、不可抢占连续model span、已知当前模型LRU保护、GRU、团队MC信用、缓存＋有界DAG残差、surrogate控制变量与独立裁剪、entropy退火。费用、奖励、容量不变。

## 版本与预算

checkpoint改为`delivery_parameterized_ppo_v2` / `delivery_parameterized_dqn_v2`，明确拒绝v1。Producer标`joint_team_node_delivery_v2`；auditor --delivery为v2显式工厂，验证每个候选的预计source末节点时间大于ETA。v1只可按cad6509历史源码重放，不混合同。

统一入口`python scripts/launch_joint_delivery_comparison.py`，独立root **artifacts/joint_delivery_control_v2/**；旧root joint_delivery_control 是停止的v1历史，不覆写。源必须clean；复制相同40train/16dev与GRU，开跑前冻结comparison_plan/源指纹/补充条件。

八主与两support、128×seed7/17/27、主base/tight各50%、固定最终128全部沿用v1事前设计：3072+768=3840新训练，从头相同预算。无新拟合/窗口选择。预声明固定mobile补充和旧最强缓存48单元重放仍为0新增训练，所有方法都评价，不替代主表。

## 验证与成本透明

新增末节点第4帧可结束而第6帧离站时禁止准备、已交付prefix效用不被重收延迟、真实可观测source拥塞时仍保留预测机会，以及v1 checkpoint拒绝检查。21项新控制检查、完整398相关回归通过，日志artifacts/joint_delivery_v2_final_release_tests.log。先clean commit，再做v2 clean-source真实SA/DQN各4回合和两panel完整事件/概率审计，验证通过才启动完整训练。

工程v1两轮16 real RL episode，停止v1已记录316另至多10个未完成，v2容差修正前后各8个preflight单列；若v2全部完成，本任务已记录真实训练合计4188（3840+316+16+8+8），不把最多10个未完成试跑算作完成，也不包括toy pytest/零更新重放。

预测收益、model-cache优势、对缓存/去预测领先和所有seed后期回退均待完整回收；paper E0_UNAVAILABLE / Unverifiable，formal/holdout封存、canonical不变。v1和失败日志全部保留。本轮无文献检索，不作novelty宣称。

## 2026-10-16收尾状态（原事前合同保留）

source9b9c0d7的3840回合及三审计已完成。完整结果见[joint_delivery_v2_results_review_20261016.md](joint_delivery_v2_results_review_20261016.md)；资源保护有限范围通过，但预测/缓存/双指标领先和收敛目标未达。上文“待回收”属于启动时记录；不更改原预算或历史失败成本，不新增训练/holdout。
