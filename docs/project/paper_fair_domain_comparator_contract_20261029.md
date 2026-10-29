# 方向匹配公平对照：权限与实现合同草案（2026-10-29）

本轮只完成公开方法与本地动作合同映射，未实现、训练或评价新对照。目标稿件为CSCWD2027；TMC/ToN及系统顶会用于对标证据，整体论文未完成。私有计划、逐字段权限和来源SHA保存在 `artifacts/analysis/paper_evidence_strengthening_20261029_v1/fair_domain_comparator_contract/`。复用已冻结12篇一手矩阵，不重复检索或下载。

## 选择理由与外部方法边界

| 原方法 | 可借鉴的受控比较 | 当前不可直接移植的部分 |
|---|---|---|
| [CLIMB](https://proceedings.mlr.press/v306/zhang26dg.html) | 同到达/完成吞吐，readiness与admission特征矩阵，容量安全锚、后台存活与控制开销 | 单站LLM residency调度不包含移动覆盖/连续DAG状态；本地readiness规则不能冠名CLIMB复现 |
| [POLAR](https://arxiv.org/abs/2604.16583) | 固定routing配不同cache规则，质量/延迟/冷加载/churn分账 | 作者页录用与ACM正式条目核验分开；其测量校准仿真不是VEC现场或本地已复现算法 |
| [dLoRA](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang) | 请求与adapter迁移分拆、proactive-only/routing-only和切换成本 | 数据中心GPU serving与移动RSU连续workflow动作不同；不能用简化代理替代原论文方法 |

来源页码、公开PDF hash和已核/未核范围沿用冻结 `literature_comparison.json`，不将外部数字当本项目门槛。上述论文都已在文献表登记，本轮未新增文献。

## 已有比较锚保持原样

`StreamCacheRule('gain_prefetch')`是本地诊断规则：请求内保持执行目标；失去可达目标时按当前分配负载和ID选择；可选预取使用公开DAG/预测与link calendar、protected models及净交付代理。它与main的可行动作、intent支持、node relocation prior和神经forward并不全部相同，不能把main对gain差值完全称为学习收益。

完整main ep000→固定ep96已独立补齐：五seed、全12条件平均训练增量为负。新head等权候选也未通过事前门。两者保留，不追加训练或按峰值替换；这进一步要求明确比较权限，而不是选择较弱对照。

## 建议另立的软件任务：shared_support_prior_greedy_local

这是**未实现、未验收、不可执行的本地启发式草案**。独立审查发现，builder的m0/m1/m2不等于最终动作支持：
`ScheduledControlPPO._evaluate`还按已选execution/cache对后续head作条件mask。修订v2必须匹配最终条件支持，
不能仅比较builder数组。所有既有baseline不动；新规则只允许消费与原intent main相同的当前状态、已释放FIFO DAG、
因果冻结预测、node尺度成本、request memory、两遍执行次序和公开预留日历。未释放job、未来真实位置/dwell和封存结果都不可见。

- 执行head：在相同有序候选和m0上按公开prior0取最大，原候选序号破同分；不调用神经forward。
- 缓存head：在相同typed候选、intent支持及**给定已选execution后的最终条件mask**上选择。原cold分支可能强制当前目标的adapter候选，不能无条件base-first；只有最终mask允许时才按reactive依赖闭包的base/adapter顺序，再按正的`optional_cache_value.net_delivery_proxy`降序、序号破同分。只有no-op仍在最终mask中且没有其他合格项时才可选no-op。
- 状态准备/优先级head：给定已选execution和cache，原最终mask还排除与它们同站的prepare目标。只在最终mask内按`delivery_migration_opportunities`的正`state_net_reward_proxy`排序，阈值固定为该源函数的`prepare_feasible`语义；不使用同时可能代表cache收益的`net_reward_proxy=max(cache_net,state_net)`作为state收益。没有合格项则`(None,0)`；priority固定0会改变环境grant顺序，是显式算法差异，不能称纯学习消融。

顺序也属于合同：request变更重置memory → 原rotated owner次序的第一遍execution及逐owner reservations → 重建calendar → 每owner依次execution条件下cache、execution/cache条件下event → literal control和异构尺寸 → claim/booking → 更新memory。不得先全局排序cache后再回填执行目标，也不得省略较早owner的booking。

选择完动作后仍按真实异构字节和原共享链路容量booking；不能为规则免除冷模型、状态或控制成本。支持一致性只在给定相同原始状态、owner target和已选head前缀条件下验证；轨迹分岔后的内生状态不能假称逐点一致。该控制是domain planner/support的受控对照设计；priority及打分差异均须披露，完整支持内的训练增量仍由初始/最终checkpoint配对回答，不能由本启发式差值单独归因。

## 前置合成验收与后续权限

实施前另立精确纯软件合同，先验证所有head/两遍候选、最终autoregressive mask和prior的条件等价，覆盖cold强制adapter、base不合法、prepare与execution/cache同站冲突、reactive闭包、跨请求/同族需求、保护依赖、满cache、slots冲突、无正值、tie、stale版本和request绑定。前瞻source freeze须绑定原agent、rule、delivery/stream refinement、link reservation与环境费用/完成消费者，不能只锁三个入口。公开ID只用于绑定；规则无神经网络、forward或optimizer。生产端需保留选项/原因/选前选后mask/proxy/预留，消费端逐帧核原始费用与完成；不能只信汇总。

私有v2 helper清单只是最低来源清单；实际freeze还须绑定继承的`multi_request_target_control_env.py`中priority/grant消费者及完整配置依赖。新增来源库存不代表运行时支持/费用已验收。

本轮实现/训练/评价/拟合/下载均为0；当前科学余量不是新比较预算。后续科学任务仍须输入资格、所有对照、公平信息/动作与费用、单次执行及有限原生物理帧合同。模块必要性需匹配重训，不能用推理删模块代替。

最终报告需保留所有到达/未完成/拒绝与tail截尾、请求完成、native reward、真实状态/模型/输入/输出字节、链路/compute占用、重复事件与独立请求失败分母、全部seed和固定最终；分开统计每run尾部再跨seed聚合，组件p99不能相加。独立时间/历史、formal/holdout、忠实外部实现及无线/RSU/任务质量成本仍是未闭合项。
