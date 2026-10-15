# v3 参数化学习合同与开发对比

日期：2026-10-14。状态：COMPLETED_NEGATIVE_DEV_RESULT。本文件先冻结实现与预算，再执行开发对比。

## 任务与版本边界

- `build_agent(..., control_contract="joint_parameterized_control_v3")` 对 SA、MAPPO、仓库内缓存 PPO 返回各算法文件中的新版类。默认 registry 与旧 checkpoint 保持原合同。
- 新版 SA 是机制重新实现：DAG 两步消息编码、执行→缓存→状态/优先级的 autoregressive PPO、观察性 return/cache/state-cost surrogate。不是旧 checkpoint 权重移植，也不是旧论文架构完整复现。
- MAPPO 使用每请求共享 actor 与全当前环境的集中 critic；缓存基线额外编码 typed byte-cache 上下文，仍为 first-party cache-augmented PPO，非外部论文 DRL 复现。
- 三方法共同读取同一因果预测、延迟共享缓存状态、已知自身 DAG/执行状态。集中 critic 的未延迟全局缓存只用于价值训练，actor 特征不读取它。

## 动作、概率和控制周期

- 三个条件 categorical：执行站（wait/四 RSU/显式兼容车辆）、缓存（noop/每个未完成已知 DAG 节点在四 RSU 的 base-only 或 adapter+base）、状态准备站与优先级 0..3。
- 联合 log probability 为三个实际条件分布的 log probability 之和；PPO 重算使用冻结的行为特征、掩码及先前已选控制，禁止执行后覆盖动作。
- 公共多时间尺度：执行站选择提交至节点变化或失去覆盖；wait 只提交一帧。缓存每四帧决策。状态准备/优先级每帧决策。reset 清空提交记忆；一个 checkpoint 在新 episode 从空记忆开始。
- 当前学习合同固定四 RSU、最多 32 DAG 节点。合法动作中**显式 victim 选择尚不学习**；typed dependency-safe LRU 为全方法共用。环境显式驱逐仍可用，不能宣称主方法已学会更优替换。
- 完成统计仍需 output 送达；模型、状态、input/output 共用数据链路，费用不加方法特有奖励。

## 预测与 surrogate

- 残差 GRU：8 帧已观察速度，hidden=16，预测未来 3 帧位移；残差叠加常速度。20 train 窗口前 16 用于拟合，后 4 用于预测验证。固定 seed=101、30 epoch；最终模型固定，不按 dev 精度筛选。记录 train validation 的位置 RMSE 与常速度 RMSE。
- 在线推断只接收截至当前帧的车辆位置；延迟到预测目标帧才统计已观测站点命中率。在线 confidence 是滚动命中率，不是校准概率。冷启动历史补零是边界。
- 所有方法使用相同冻结预测器；`sa_no_prediction` 只关闭 actor 中的预测特征，保留其余条件和共享消息费用。
- SA surrogate 由实际训练 rollout 拟合团队 discounted return 和请求命令的即时 cache/state link cost，三个控制阶段分别拟合，0.3 bounded residual 影响 logits。辅助 loss=0.2。它是 observational model，不能将未选择动作的估值视作因果反事实，也没有完整状态转移 imagination。
- cache cost label 当前是该命令占用的初始槽，持续 fetch 槽在团队 return 中记账，不可当完整对象成本估计。

## 冻结对比计划

- 同一 source manifest：`eight_vehicle_joint_v2_mechanism_budget96_20261014/20261014T074430Z_development`；数据 SHA256 与 train/dev 窗口逐字段重建验证。v3 替换动作与 input/output 物理链，不与历史 v2 奖励跨版本比优劣。
- 五配置：SA / MAPPO / cache_offload_drl / SA-no-surrogate / SA-no-prediction；seed=7,17,27。各 128 episode，固定循环相同 20 train 窗口，update=4 episode，每次 4 epoch、batch=512、lr=3e-4、gamma=.99、reward scale=.1、clip=.2、entropy=.01、value=.5。
- 每 32 episode 固定保存 checkpoint 并独立加载评估 8 dev 窗口；恢复训练 RNG，评估不改变训练抽样。仅固定最终 128 checkpoint 用于主开发结果，不按最好 dev checkpoint 选择。
- 固定 reachable-balanced-reactive 规则只作对照，不给 actor imitation teacher。所有失败和负差保留。
- 每 checkpoint 记录文件与 tensor hash、初始 tensor hash、参数量、精确恢复检查。输出 manifest、训练/开发 JSONL、最终 dev 逐请求事件、预测器与 checkpoint。formal/holdout 不执行，paper_claim_ready=false。
- 机制核验：奖励/费用收据、节点与完整请求完成、future-cache/base-only/prepare 命令、实际准备到达、服务站变化、原始 paired window/seed 差值。仅三 seed/八 dev 窗口不足论文统计晋级。

## 验证

新增合同单测覆盖三算法 update、精确 joint-logprob 重算、checkpoint/optimizer 恢复、actor 不读 critic 真值、非法 mask 拒绝、未来轨迹不影响当前预测、跨窗口预测样本边界。已有环境、缓存、legacy registry/checkpoint 回归共同执行。实际训练前另做 4 episode smoke，只确认生产/消费端可运行，不用于性能结论。

## 2026-10-15 回收

本阶段及其后独立productive修复阶段均已完成，原始结果/审查见 `joint_parameterized_results_review_20261015.md`。主机制无优势，327项相关回归通过，论文证据仍E0_UNAVAILABLE。
