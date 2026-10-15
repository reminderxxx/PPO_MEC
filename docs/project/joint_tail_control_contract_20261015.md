# 尾部交付信用与准备专用surrogate：独立实现合同

日期2026-10-15，前置只读审查3453fe6：`joint_tail_regression_review_20261015.md`。用户授权本轮定位后修复/重训；原artifact保留，主目标仍为奖励和完成请求同时领先，尚未验证。

## 学习与预测

- 新opt-in `tail_controls=True`，checkpoint `tail_parameterized_ppo_v1` / `tail_parameterized_dqn_v1`。七算法直接由各自agent文件承载，PPO/MAPPO分开，旧scheduled/五动作默认不变。
- 有限120帧评估是未折扣reward之和，新训练共同gamma=1，原节点完成、等待、全部传输和共享费不变。不添加workflow bonus，不按算法改变reward。DQN跨物理帧Bellman同步gamma=1，三micro-stage内部仍1，终止team tail保留。
- 全局critic目标改成reward scale .1/初始请求数，与local/action/prefix value同单位，避免大单位回归占全网grad clip；ledger仍守恒半local半team，原完整团队returns保留供训练label/审计使用。修正当前input已支付、pending output阶段的剩余服务量估计，增加剩余节点比例与候选deadline slack辨识。
- GRU保留8帧/h16/history8/30epoch；surrogate继续训练，但只影响正净收益cache/state准备。execution不加surrogate；同prepare target四个priority共享一个残差，noop priority干预为0，防止观测return估计重排队列。强制cold-cache不与不可行noop比较。它仍是近似模型，不承诺所有准备正收益。
- 预测定位到准备特征，常规执行观察使用实际进度、负载与deadline；DT保留专用原始预测context。没有真实未来进入actor。

## 公共训练覆盖与压力panel

- 原20train/8dev原始frame/time interval不改，从20train各增加一个8车cohort。仅在同train时间内选持续出现且tight全程覆盖的车辆，按覆盖离开次数、首末位移、ID排序；不读取policy分数。新40个训练scene交替base和mobile_tight，128回合仍相同预算。
- train几何站覆盖离开由原29增加mobile_tight151；所有车帧至少有1个站覆盖。几何事件不是执行迁移/净收益证明；真实训练中的prepare/机会/费用另记。
- tight半径统一为原半径×(.65/1.2)，不改站位置、计算/链路/缓存容量/传输费用。保留原8dev base，再增加相同8dev的tight压力panel。主结果固定两panel各50%，每panel/seed全部报告；共享原区间，不能把16scene当16独立窗口。
- GRU fit/validation按原source/time interval分组16/4，而非40scene截断，避免同时间不同cohort泄漏。所有算法共享一个冻结predictor及prepared protocol，epoch/训练数据一致。
- 不因算法得分调整cohort、几何或panel权重，不改源数据、不打开formal/holdout。场景困难是研究条件的压力检验，不承诺一定扩大或有利于SA的差距。

## 冻结预算

八主配置：sa_ghmappo、mappo、sa_no_prediction、ppo、dqn、cache_offload_drl、dag_offload_drl、dt_handoff_drl；各128episode×seed7/17/27，3072episode。两项预声明support：sa_discounted（仅gamma .99）、sa_no_surrogate，各384episode；实际总3840。最终128主表，无峰值选择；每32保存曲线。固定train probe indices0/11/20/31覆盖两种cohort。

PPO与DQN交互预算相同，优化步数按各自定义公开。三个方向DRL仍是first-party PPO适配，非外部论文复现；旧QMIX/MAT等不混表。

## 产物与消费者

`prepare_joint_tail_control.py`冻结`protocol.pt`（派生真实轨迹/scene）、`predictor.pt`和manifest/hash。`train_joint_tail_control.py`各算法独立group输出原始训练、dev、最终事件、checkpoint和train probe；scenario_id含panel，不能仅以frame_offset去重。`audit_joint_tail_control.py`验证共享protocol、预算、source、时间划分、ledger、链路/容量、所有checkpoint和base/tight首窗模型重放，最后按固定full权重开关预测作零训练support。旧auditor不静默消费新scene schema。

路径：`artifacts/joint_tail_control/protocol/`与`artifacts/joint_tail_control/<agent>/<run_id>/`；根comparison_plan.json在开跑前冻结，最终audit.json同根。工程preflight为`joint_tail_preflight_*`，预测器1epoch、SA/DQN各4episode，仅接口验证。

## 验证与边界

375相关pytest通过；首批真实数据preflight完成，旧scheduled消费者和新schema两panel重放另行核验后启动clean-source正式开发比较。真实成本、多base/学习驱逐、并发DAG/到达、控制面数据槽竞争、formal/holdout等仍未覆盖。不能将本轮工程合同写成论文算法优势。

实现验收追加：375测试通过，旧scheduled的11个checkpoint首窗完整事件仍一致；新SA/DQN两份4episode入口的2checkpoint、base/tight共4次独立事件重放及原始费用核验通过。审计首次输出相对路径解析报错已修复，原失败日志保留，完整重跑exit0；未改原始模型/事件。

完成状态（2026-10-15）：预声明十组3840训练全部完成，120checkpoint/409883事件/55258收据/60重放通过；最终主表及负例见joint_tail_control_results_review_20261015.md。主结果权重未改，双指标最高/收敛目标未达。另有容量独立审查后的零训练移动探索support，明确不替换本合同主表。
