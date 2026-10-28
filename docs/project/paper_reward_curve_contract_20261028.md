# 原始奖励学习曲线：一次有界执行与自动回收（2026-10-28）

状态：`SOFTWARE_QUALIFICATION_BEFORE_SINGLE_LAUNCH`。用户本轮明确要求自主下发、推进并闭环奖励图。
该最新指令授权本独立合同；覆盖此前“不再围绕三个已消费train时段追加训练”的限制，仅用于本次
从随机策略初始化的学习曲线证据，不延长旧run，不把已消费时段恢复为hidden，也不保证正向排名。

## 事前依据与交付目标

[五篇原论文图页核验](paper_reward_figure_reference_20261028.md)检查了ToN/TMC与NeurIPS的实际图页。
本地只读原件审查发现：旧四算法有64×3seed训练回报，但固定策略评价仅ep32/64；
最新意图主方法只有initial/final且仅3次更新。不能以插值、拼接或平滑冒充密集收敛记录。
旧64点已独立输出legacy诊断图，清楚标明旧版本，不作为当前主方法曲线。

本轮交付为当前主方法与固定本地MAPPO/PPO/DQN的原始奖励学习曲线、真实交互帧补图、
原始CSV/JSON、图注、资源账本、checkpoint/逐帧奖励核验与九次前瞻final独立重放。
输出矢量PDF/SVG及600dpi PNG。奖励图本身不等于整体结构必要性、真实部署或顶刊录用证明。

## 冻结的方法、输入与评价

- `main`：现有`IntentAwareStreamSAGHMAPPOAgent`，不再改算法；其节点尺度迁移先验与意图准备限制明确保留。
- `mappo/ppo/dqn`：现有stream factory原样；不改弱规则、不调其他算法。共享规划器、候选特征/信息与
  controller抽象须披露，不称外部论文忠实复现。主方法额外意图限制也不是同一可选策略集合。
- 五seed固定`7/17/27/37/47`；策略全部随机初始化，无策略父checkpoint、旧optimizer或resume。
  共享GRU是旧train拟合后冻结的输入预测器，“from scratch”仅指策略，不能隐去其30epochs拟合暴露。
- 原冻结NGSIM+Alibaba protocol的12个input-selected已消费train条件，8轮固定同顺序，共96完整episode/cell；
  其他数据/原RSU/费用/provider/环境/奖励/预测器均不改，不下载TGSIM，不读取sealed formal/holdout。
- 每4完整episode执行一次原生update，共24次，epochs4。DQN保留原生replay、epsilon和最多32梯度步/调用；
  PPO原生KL停止与辅助优化器不改。精确记录各optimizer.step次数，不声称相同梯度或计算成本。
- ep0及ep8至88每8集在固定`train_000/train_038/train_067`三条件greedy评价；ep96只跑全部12条件，
  其中三probe用于同一曲线最后点，不再重复执行它们。每cell13checkpoint、48评价episode。
- 评价单独加载模型，恢复Python/NumPy/Torch RNG，不推进训练随机流。完整晚到波/idle/终止tail保留；
  不critic截断bootstrap、不填充post-terminal交互、不按奖励峰值选择checkpoint。固定final为ep96。

## 硬预算与一次性执行

| 项 | 声明最大值 |
|---|---:|
| 训练 | 4方法×5seed×96episode×480帧 = 921600帧 |
| 固定策略评价 | 20cell×48episode×480帧 = 460800帧 |
| 冻结gain规则 | 最多15episode / 7200帧；实际原计划12条件各一次，不制造seed重复 |
| 前瞻独立重放 | 9episode / 4320帧 |
| 总物理交互上限 | 1393920帧 |
| 执行器总deadline | 48小时，含自动回收；不等于ETA |
| 科学启动 | 1次campaign，3个worker；每cell一次create-only claim |
| 全campaign新增产物 | 10GiB；每cell最多480MiB，为回收与图件保留余量 |

真实提前终止可少帧。横轴主图用完成的外生训练episode，附图按每个seed实际累计物理帧画线；
不为错位帧预算插值出跨seed均值。资源CSV同时给帧数、更新、精确优化器调用、参数和wall时间。
同会话/上限不意味着相同实际帧、决策行、梯度或compute。历史耗时只作资源量级估计，不承诺wall ETA。

先通过软件资格，再冻结clean commit、完整source SHA、原protocol与protocol.pt SHA、
case/schedule/probe、输出路径、九重放身份与本合同，最后单次启动。运行中不改tracked源码。
任何cell失败保留原件与部分预算并停止其campaign；不自动续训、改参、换seed或扩大额度。
软件回收失败先独立诊断，必要时另立只读消费端恢复任务，原失败不覆盖，不重新训练来填图。

## 统计、图形与自动闭环

每个seed每点先等权平均固定三probe原始native episode return，再跨全部五seed画均值与±1样本SD；
细线保留每seed。主图不平滑，规则只画同probe的固定均值水平线，不制造seed方差或训练过程。
SD不是CI；这些时段已消费，三条件不作为独立时间cluster做显著性或泛化结论。
另报固定final全部12条件结果，不能把它的均值替换三probe最后点。

自动回收逐个检查20cell/260checkpoint、1920训练/960评价、完整原生reward_components、MC/credit尾值、
实际交互账本、方法/class/options、随机初始化tensor、RNG隔离软件资格、决策和原生control对应关系。
九个重放身份在plan中事前固定，核对逐帧奖励和literal控制摘要，不根据奖励选择重放。
同输入的跨方法完全相同行为计数须报告，不通过删除或改写对照处理。

`launch_stream_reward_curve.py`完成训练后自动执行`collect_stream_reward_curve.py`，后者核账、
规则、重放及`plot_stream_reward_curve.py`导出；持续任务回收最终图与负结果，不把子任务结束当论文闭环。
若当前方法没有优势，结论必须明确；后续主方法改进须先独立诊断并另立有界任务，不能回头选图。
全过程保护主目录七修改文件与R20，不修改论文/canonical。`paper_goal_complete=false`。

## 软件验证入口

```bash
python -m pytest -p no:cacheprovider tests/test_stream_reward_curve.py tests/test_stream_reward_curve_collection.py -q
```

软件样例不消费新的科学输入，不按奖励挑参数。独立消费者测试在刷新producer hash后仍应拒绝
篡改原生奖励/MC/credit、删点/删seed、伪造初始化、改变checkpoint选项、遗漏control与花费等错误。
