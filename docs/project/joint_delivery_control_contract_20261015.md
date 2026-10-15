# 团队节点交付与预测准备资源保护：独立实现合同

日期：2026-10-15。依据独立审查`joint_tail_control_results_review_20261015.md`（532dca6）；本阶段为用户明确授权的修复/开发实验，审查结论不被追认为性能改进。formal/holdout封存，canonical不变。

## 已定位问题

旧full seed17/dev_2760_tight预测离站基本正确，三次state prepare占用其他请求的冷adapter准入槽，节点47对关闭预测49、奖励−1.931024。旧calendar没有预留新冷模型准入；local/team混合信用N8时自身/他者权重9:1；state proxy按整图完成估值与节点奖励不一致。旧seed27 surrogate priority误干预已有独立冻结干预证据；tail限制后seed27改善，但seed7/17仍回退，不能称全部解决。

## 物理条件与公平性

复用tail同一40train/16dev场景、原始帧/车辆/DAG、费用/容量及30epoch冻结GRU；由`prepare_joint_delivery_control.py`逐字节复制已验证protocol/predictor并记录父manifest/hash，无新数据选择、拟合、资源放大或奖励bonus。base/tight各50%主表不改；tight7/8必要容量不足的负结果保留。原始8时间窗口的两个panel有依赖，不能当16独立窗口。

全部七算法＋去预测统一使用新团队控制合同与预算；MAPPO仍为公共预测/缓存/预约增强对照，方向DRL为本仓PPO适配，DQN是真正Bellman/replay实现；不声称普通MAPPO或外部论文复现。保留强缓存基线，不削弱它来证明SA优势。

## 准备、缓存与团队目标

1. 两阶段控制：第一阶段按公共负载先验选定所有请求的执行站；第二阶段据实际已选站统一预约reactive state/model/input/output，再选cache/prepare/priority。所有头的原始概率与行为特征保存，训练重算可精确核对；不在采样后替换动作。
2. 当前冷base/adapter准入纳入预约，连续phase包括接续input、估计compute release和output。短节点服务排序用于吞吐保护，不再用整个DAG大小决定公共链路先后。已有后台模型传输仍不抢占。
3. State准备只能使用当前空闲且预测ETA前未被已知事务预约的槽；book不会删除自己或他者的mandatory未来phase。
4. 模型准入为异步、不可抢占的依赖闭包。所需连续span必须全部为空或属于自己的model预约；不能以“当前一槽空闲”放行将来会阻断output的base/adapter。calendar长度至少覆盖声明的最大闭包，预测horizon仍8。
5. 可选远站/未来节点模型预取按声明大小、LRU顺序、base依赖模拟淘汰；若会淘汰已选在本站执行的当前请求所需对象，则屏蔽。当前output不需要模型，不享有reactive例外；当前执行必需准入保留。远站缓存容量/占用尚未共享时屏蔽可选预取。此guard针对本合同统一大小合成catalog，未知依赖保守处理，不宣称现实系统全状态安全保证。
6. 状态准备与模型预取分别估算实际所能推进的prefix；不能把未选中的model预取同时计入state收益。节点交付序列尊重已付input、在途output和当前计算进度，按时域内节点数量及原未完成等待计utility；减额外传输与预测失败成本，纳入其他已选在目的站的当前节点计算队列。当前节点migration proxy不复制到另一个未来adapter特征。
7. 已声明reactive事务通过可行域保护，允许动作的已知被挤占成本为0；未知未来需求、共享延迟、计算顺序、版本增长仍是估计。`team_opportunity_cost=0`表示禁止已知挤占，不是测得真实机会成本为零，也不是完整反事实世界模型。会否出现时序级外部性须按原始事件和固定权重开关检验。
8. 保留GRU、预测准备和状态共享。No-pred关闭预测消费/预测准备候选，公共非预测reactive calendar保持相同。

## 学习与后期稳定性

- PPO所有头、global/prefix/action critic与surrogate return标签改用同一`team MC / initial_request_count /10`，有限时域gamma1；移除自身9倍权重，保留全团队尾部。DQN每帧同一团队reward/N，提前结束请求末transition吸收完整已观察团队tail；三个micro-stage仍是Bellman，不改为PPO。
- SA吸收缓存基线cache_encoder；GNN以`.25*tanh(gate)`有界残差补充全图及节点特征，gate零初始化，初始actor概率与cache基线一致。后续额外图信息是否有收益由训练/消融判断，不保证superset在有限训练中更优。
- Surrogate保留team-return和费用学习，作为action-independent的policy-averaged团队Q控制变量；取消观测return直接加logit的干预，不能再据此重排执行/priority。
- Actor/critic和detached-input surrogate分别Adam与grad norm .5，避免辅助loss消耗同一裁剪预算。Actor LR仍3e−4线性至3e−5；aux LR固定3e−4。四epoch/update、512batch、每4episode更新不变；额外辅助优化不声称等计算预算。
- 归一化entropy系数.01线性至.001；预声明sa_fixed_entropy保留.01，其余完全相同。保持KL .02门限及完整曲线，固定最终128、不选dev峰值或按结果回滚模型。减少后期随机漂移是设计目标，不是已验证收敛结论。

## 冻结比较

八主：sa_ghmappo、mappo、sa_no_prediction、ppo、dqn、cache_offload_drl、dag_offload_drl、dt_handoff_drl；各128×seed7/17/27，共3072。两support：sa_no_surrogate与sa_fixed_entropy各384，共3840。训练scene循环与tail相同，每32评价全部16dev及固定train probe0/11/20/31；同一episode预算，不声称PPO/DQN/aux梯度步数或墙钟完全相同。

所有最终full48单元做同权重预测关闭，原始on/off事件保存；所有seed/panel/负例报告，不筛选。预声明复用上一轮mobile_base的固定8开发分层对全部十组最终模型作补充评价，不加训练、不改条件、不替换主表；同权重关闭预测另24单元，仍非独立新时间窗口。

另预声明零训练强对照：上一轮缓存基线128×3的三个最终冻结模型，在字节相同物理协议上重放全部48单元并核验原始事件完全一致。保留其旧controller/credit身份，和新受保护缓存基线并列；新full是否领先必须同时检查这两个缓存对照，不把公共保护带来的baseline变化隐藏。

## 入口、消费者与产物

- `scripts/prepare_joint_delivery_control.py --output_root artifacts/joint_delivery_control/protocol`。
- `scripts/train_joint_delivery_control.py --protocol .../manifest.json --agent <name> --seeds 7 17 27 --episodes 128 --eval_every 32 --output_root artifacts/joint_delivery_control/<name>`；checkpoint `delivery_parameterized_ppo_v1` / `delivery_parameterized_dqn_v1`，非resume快照。
- `scripts/audit_joint_tail_control.py artifacts/joint_delivery_control --delivery`：共享producer/consumer工具的显式新版本factory，验证新credit/entropy标记、全部checkpoint、60次base/tight完整事件及两阶段概率、预测关闭、每个prepare空闲约束、cache闭包span及按同帧其他请求真实节点依赖独立核验可选预取驱逐。旧默认路径保留旧工厂。
- `scripts/audit_joint_delivery_witness.py`精确重放旧两分支（47/49节点），新calendar拦截三次有害准备；0训练，不据此声明新策略收益。
- 新root开跑前冻结comparison_plan.json、公共manifest/GRU SHA、source commit，所有group clean启动；工程preflight、失败日志与旧轮产物单列，不混表。

## 验证状态及限制

完整相关回归394 passed（含17项新控制检查），日志`artifacts/joint_delivery_regression_prelaunch.log`；随机与确定性策略概率局部检查通过。旧scheduled 11模型事件重放一致。两轮真实SA/DQN工程各4episode×2，合计16训练episode单列；最终新消费者核验通过：2checkpoint、27025事件、3757收据、4次两panel完整事件与逐帧策略概率一致。

中间失败日志保留：未知远站snapshot缺occupied_bytes触发KeyError，已按“不观测即禁可选预取”修复，完整回归通过；首审计把DQN确定性策略与epsilon分布混比，已修正审计为同一确定性模式，完整重跑exit0，未更改原模型/事件。工程均值不作算法效果结论。

统一启动：`python scripts/launch_joint_delivery_comparison.py`要求clean source，复制hash核验的协议，开跑前冻结comparison_plan和mobile补充条件，启动十组并等主/补充审计全部成功才结束。程序记录每条命令、source指纹、进度与运行环境；训练进行中不得改冻结生产源码。单组入口保留用于可追溯重放，不可盲目重启整个已存在root。

补充入口：`scripts/evaluate_joint_delivery_support.py prepare/evaluate/collect`字节复用上一轮固定mobile protocol、使用新delivery factory并核验每个事件的prepare/span约束；所有组最终128统一评价，0训练。

仍缺真实base/adapter/无线成本、多base/学习驱逐、并发新到达、完整vehicle/RSU MARL和正式独立holdout。预测收益、cache优势、回退是否减轻均待冻结结果；paper E0_UNAVAILABLE / Unverifiable。无新文献检索，不声称新颖性提升。
