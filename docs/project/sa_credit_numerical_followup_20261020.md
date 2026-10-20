# 5C数值补验下一任务：自由缓存与idle

- authored_at: 2026-10-20；status: **方案与既有见证定位完成，未执行补验、未启动环境/训练或新算法。**
- parent: [本轮独立报告](sa_credit_numerical_review_20261020.md)；七组数值对照通过，但cache全部forced、idle0/21，完整运行资格未齐。
- 本轮已用资格root保持封存；本方案不复用claim、不追加帧、不补跑未覆盖单元。

## 已固定的来源选择

只查已有5B的三seed×两arm×64，按dev000/018/021/022/038/043/045/046、frame升序、原call_index升序，取每单元第一个实际reference second-pass cache mask>1见证；不读取reward作选择。

| seed | 版本 | 既有case/frame/call（零基） | reference合法cache数 |
|---|---|---|---:|
| 7 | 原版 | 固定八case量化帧中无见证 | 未识别 |
| 7 | 分组 | dev000 /20 /25 | 2（None与adapter index32） |
| 17 | 原版 | dev022 /59 /27 | 2（None与base index57） |
| 17 | 分组 | 固定八case量化帧中无见证 | 未识别 |
| 27 | 原版 | dev018 /239 /75 | 2（None与base index191） |
| 27 | 分组 | dev000 /52 /42 | 2（None与adapter index42） |

四见证均有原八车全输入、16个first/second调用及首first控制memory。只读定位成本为24份已有scoring gzip/3020旧记录、4份SHA绑定可信capsule反序列化；新增agent/model/forward/sample/backward/Adam/环境均0。机器计划/完整候选和source SHA在本轮私有根sa_stage5c_cache_witness_plan_20261020.json；helper与失败/无见证单元保留。

## 补验顺序与拒绝条件

1. 先冻结条件时点：这些multi mask成立于保存的reference execution prefix及peer calendar。完整随机act更换execution或前序peer controls后是否仍自由是UNVERIFIED。不得把teacher-forced条件fixture伪装为完整随机行为，或靠重复RNG搜索直到出现多选。
2. 明确两种测试标签：固定reference prefix的条件数值/梯度fixture，及原完整act随机路径。若前者需独立观察适配，只能供应现成输入/输出；任何被强制的动作及样本调用次数都要明示，不能沿用“原stochastic抽样”断言。先审schema/消费者及匹配单元测试，再执行有限资格。
3. idle单列：先查已有空team-frame；若没有，显式synthetic empty decisions可做真实agent零forward/RNG检验，但不能称真实FIFO环境续接。不得新增环境以取得idle。
4. 新软件预算须另立机器合同。可评审草案为四见证×off/on≤8个私有模型、≤32主副Adam/≤16backward、环境0；idle复用同私有副本，不增加环境。具体调用矩阵/seed/条件前缀/计数/输出限额必须先冻结，本文件的上限草案尚未启动或成为新额度。两个无见证单元保持UNVERIFIED；不得拿别seed或不同快照补称完整六单元覆盖。
5. 每个声明的free-cache测试须实际mask>1且actor权重active；若固定完整act未覆盖，作为覆盖失败如实交付，首次失败停止，不在同claim换帧/改RNG/修源码再续。on/off仍逐位比较forward/裁剪前后梯度/两Adam/buffers/controller/RNG/序列化，并独立审原件。

## 与算法推进的关系

软件覆盖补齐只能解决日志的运行可信性，不识别分组裁剪退化原因。原历史stochastic逐行为MC/信用不能由这些greedy输入和合成标签补造。若后续需要真实行为信用，必须先另立无算法差异、完整预算的采集合同；没有已确认一阶缺陷和概率证明，不叠加mask/class ratio、baseline、锁站、sunk-cost惩罚、clip/lr或延长学习。

原版SA及优势门禁保持；分组候选不晋级，formal/holdout/canonical封存，DQN暂缓。任何科学新训练仍须独立授权；本任务交付没有启动它。
