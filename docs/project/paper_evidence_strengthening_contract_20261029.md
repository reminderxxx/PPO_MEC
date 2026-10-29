# 论文证据强化：独立审查后的两项有界任务（2026-10-29）

用户明确要求广泛对照同方向顶刊顶会、自主优化、分发并迭代验收。本合同把该授权落实为独立审查之后的实现与执行任务，不修改论文、默认registry或canonical。目标稿件为CSCWD2027；TMC/ToN及系统顶会只作为证据对标，不能承诺录用。整体`paper_goal_complete=false`。

## 独立诊断与当前边界

只读审查、机制诊断、文献矩阵和授权原件保存在主目录私有`artifacts/analysis/paper_evidence_strengthening_20261029_v1/`。原奖励campaign和audit的SHA分别为`abd965ec2617cae544ed38f9cf607f9c9c87383e324dda8211783f46824357e1`和`d795a848637b7828214cd41582387075e4ad50a7269682a4d756674165f1a948`，复用单次审计，不重跑原训练、审计、索引或九重放。

全部五seed的main/MAPPO/PPO初始张量逐seed相同，参数73,940；forward、迁移先验和intent mask不同。主方法三probe初始221.324、最终223.365；相对PPO初始差33.375、相对学习增量1.908、最终差35.282。这是描述性恒等式，不是随机化模块归因。主方法17,624次optimizer step包含8,812主优化器和8,812 surrogate，不能写成双倍actor更新。

定向八集诊断显示，旧policy/entropy统一active-row均值使priority权重占77.0%–80.5%，cache仅0.372%–0.873%。等权head是一个有依据但可能放大稀疏样本方差的候选；不预判它改善奖励。KL截停/熵塌缩不是当前记录支持的原因。状态字节不能解释全部负seed回退，且大部分状态链路并无显式prepare；不单独关闭prepare来追排名。

## A：补齐完整12条件的训练前后归因

固定原main五seed`7/17/27/37/47`的ep000 checkpoint。只新评价缺少的9条件×5seed=45集；原三probe×五seed的15集、final60集和gain12集仅按原audit/hash只读引用，不再执行。完整initial矩阵必须恰好5×12，final仍用原固定ep96。

新initial独立重放事前固定为`seed7/train_014`、`seed27/train_031`、`seed47/train_055`，与本任务首次对应initial评价逐帧native reward和literal control摘要相同。每集真实终止、最多480帧，不补post-terminal，不训练、拟合或重新选择策略。

- 新initial最多21,600物理帧；3重放最多1,440帧；总23,040。
- 单次启动、一个worker、3小时绝对deadline、512MiB全部新增产物。
- 原checkpoint、输入protocol、费用、RSU、预测器、规则和环境不变。
- 独立消费者核对原件/矩阵/终止/每帧reward/checkpoint身份/三重放，再报每seed、每条件和原始时间单元的initial→final奖励、完成、状态字节、链路和失败事件。负值完整保留。
- 只回答完整开发条件中训练新增了多少，不证明独立泛化或某个模块必要性。

## B：一个主方法候选，固定原方法为比较锚

启动前先单独完成旧main五cell、final60端点的定向只读比较资格：逐帧receipt累计状态字节/链路，
按v3 `service_target_changed && status in {state_restore,state_link_queue}`重构失败事件。
旧control未保存workflow完成flag/node ID，因此完成请求数只可由`node_completed/vehicle_fallback_completed`
状态次数与冻结请求DAG节点总数推导，并披露该来源边界；不能补造字段。
此任务不重做旧20cell的reward/MC审计，不执行环境/forward/重放，只读取必要旧main原件；
硬上限2GiB压缩原件读取、8GiB解码读取、30分钟、32MiB新输出，所有5seed/12条件必须匹配旧audit。
来源SHA/完整矩阵/成本或完成推导任一不可核验即不启动候选。新plan绑定该独立资格文件SHA，
两侧成本都经原件核对后才能使用非增成本门；旧失败和原件不改写。

新增opt-in `HeadBalancedIntentStreamSAGHMAPPOAgent`，只把policy objective和entropy改为每head内按active行平均，再对本minibatch有active行的heads等权。旧main、MAPPO、PPO、DQN和规则的默认数值路径保持原样。保持advantage标准化、critic、surrogate辅助项、优化器、clip、KL截停、learning rate、先验、mask、环境、费用、奖励、预测器完全相同；独立checkpoint contract拒绝混用。

五seed固定`7/17/27/37/47`，策略从与旧main相同随机张量开始，fresh optimizer；共享GRU仍是旧train拟合后冻结，不能称所有模型从零。原12个已消费train条件同顺序8轮，各96完整episode；每4集更新一次，24次update、epochs4。初始和每8集固定三probe，ep96只评价全12条件；每seed48评价、13checkpoint，保留所有点/seed。旧main e00d5fa已完成的同预算结果作为固定只读锚，不重训旧方法，也不按新结果改对照。

- 训练最大230,400帧，评价最大115,200帧；3事前final重放最大1,440帧，合计347,040。
- final重放固定`seed7/train_014`、`seed27/train_031`、`seed47/train_055`。
- 单次campaign、最多2worker、24小时含回收deadline、3GiB全部新增产物。
- 输出分head active计数/目标权重、policy/entropy值及梯度诊断；消费者从原control独立复算head覆盖，不能只相信producer汇总。
- 新原件额外保留gate需要的native workflow-completion和状态费用字段，逐帧独立重构完成数/状态/失败；
  旧control投影摘要另保留以精确比较initial行为。head浮点复算按原float32与运算顺序，限界为2 ULP，不能用float64重关联误拒合法值。
- 全部checkpoint、原生MC/late/idle/tail、行为概率、literal control、逐帧native奖励和真实训练/评价/更新/主与辅助optimizer/wall/磁盘账单核验。
- 真实物理帧、决策数、梯度和compute可能不同，不能写成计算完全公平。重放只是三集的可复现性范围。

### 事前接受门

固定final ep96、全部五seed×12条件等权。候选仅在以下全部成立时记为“局部开发候选通过”：

1. 完整性、source/输入/checkpoint身份、所有预算及三重放通过。
2. 相对旧main的全60配对native reward均值差严格大于0。
3. 完成请求数均值差不小于0；至少4/5seed的12条件奖励均值差不小于0。
4. 状态传输字节、state-transfer link frames和handoff失败事件三个均值差分别不大于0。

非严格比较采用绝对`1e-8`数值容差。失败即保留原main为比较锚并报告拒绝原因，不扩episode、不换seed、不选中间峰值、不追加候选或调head系数。即使通过，也不晋级canonical、不替换老师稿位或论文主表，不把开发门当formal/hidden证据。

## 合并资源与执行门

A+B新物理帧硬上限370,080，产物分项上限3.5GiB；整个强化授权上限450,000帧/32小时/4GiB，差额不是追加科学预算。最多三个同时工作的科学worker。各任务只启动一次，失败保存现场并先独立诊断；不得重置额度重跑。二者同一clean source冻结后启动，运行中禁止改tracked源码。

软件只用合成样例验证预算、终止、概率/RNG、初始化等价、旧默认等价、稀疏/空head、矩阵缺失、checkpoint/class/options篡改和独立消费者；不能按样例奖励挑系数。合格后冻结clean commit、source hash、protocol/protocol.pt、旧audit、授权、全部输出路径、deadline、接受规则和计划SHA，再单次启动。科学run前的源码变化不回写旧科学版本。

运行时遵守持久continuation，每20分钟至多一次核对身份/completion/failure/完成单元数，不读奖励追逐排名。自动回收完成后交实际结果、负例、CSV和接受判定；失败需反馈并保存独立诊断，不能把一次子任务结束当整个证据工作完成。

## 剩余论文证据按优先级补齐

1. **独立时间/DAG与数据资格**：418未知历史继续隔离，formal/hidden认证0；不能把79候选或不同window ID当独立。后续只针对新来源的已留本地原件做新资格合同，保留缺失值和旧FAIL，不静默插值/丢行。坐标/原RSU/provider/clock/成本合同先于新科学比较。
2. **严格统计消费端**：保留旧1个正例，针对分文件配对、物理outer身份、单条确定规则复用、指标方向/单位、退化CI/效应、source alias和分数时钟7个兼容缺口另行合成资格。当前三已消费时段仍只作描述统计，不发泛化显著性结论。
3. **最近邻强外部对照与模块必要性**：先按公开原算法冻结权限/信息/更新/参数/公共版本映射，再做相同先验和mask的匹配消融。局部类名和不同轨迹不是外部论文忠实复现；推理删除不能替代匹配重训。当前仅一个候选预算，不自动启动额外消融或对照训练。
4. **真实系统与机制成本**：就近验证adapter加载/恢复和质量、请求级迁移失败、队列tail/连续执行、无线/RSU代价；测量与模拟参数映射必须可追溯。共享planner、RMSE、技术恢复成功或奖励提高都不能替代这些证据。

本轮不下载新数据/模型、不读取sealed结果、不向外部上传私有数据/稿件、不向其他聊天或他人发消息。主目录七个用户文件、R20和老师固定稿位保持原值。代码说明匹配验证后scoped commit，单次尝试推送授权仓库分支；原件、checkpoint、私有审查与实验不上传。
