# 持续请求学习独立回收审查

- reviewed_at: 2026-10-17 (Asia/Shanghai)
- literature_cutoff: 2026-10-15历史检索；本轮未重审novelty
- target_venue: IEEE TMC（规范目标，不代表投稿资格）
- artifact_run_id: joint_stream_learning/20261015T181642Z_development
- policy_version: tmc_review_policy_v3_20260621
- git_commit: e96d831534a7442dc7b21d673563b1a0237eb14f
- evidence_level: 全开发raw/checkpoint审计及24次最终重放；论文E0_UNAVAILABLE / Unverifiable
- verdict: 已回收，未获全基线优势，存在基线重复及安全审计盲点；N/S，不晋级canonical。

## Artifact完整性与protocol

原始audit.json为pass，launcher_result为complete。1536会话，647788实际训练物理帧（上限737280），4301252决策；1152最终单元、2972827事件、461289费用收据、48checkpoint和24最终重放。独立复核88项output/checkpoint SHA通过。最终64固定，全部seed7/17/27、48条件保留；32→64仅同12probe，未选峰。只有2个外层新dev时间会话，不能把144行当独立窗口；long单源、资源合成、formal/holdout封存。

## 三seed描述性结果

| 方法 | 奖励 | 请求完成率 | 模型KiB | 驱逐次数 |
| --- | ---: | ---: | ---: | ---: |
| DAG | 207.295583 | 88.6285% | 127.21875 | 101.77083 |
| DT（与DAG重复） | 207.295583 | 88.6285% | 127.21875 | 101.77083 |
| SA完整 | 205.920336 | 87.9774% | 144.99653 | 107.76389 |
| cache | 201.559963 | 86.5885% | 149.68750 | 117.38194 |
| MAPPO | 199.222386 | 85.8941% | 160.77778 | 136.03472 |
| PPO | 197.816766 | 85.0260% | 152.85764 | 123.45833 |
| 去预测 | 192.103464 | 84.0278% | 169.26736 | 133.92361 |
| DQN | 168.849594 | 77.2135% | 147.07639 | 121.76389 |

## Baseline公平性与独立性blocker

Stream特征构造覆盖Scheduled的DT-specific domain，DT和DAG最终都使用同一DAG domain、相同网络与随机初始化。6组checkpoint全部tensor/actor哈希一致；180dev行、355598event行、56341receipt行在只替换agent/rule标签后逐字节相同。不能当作两份独立算法证据，不能称已战胜或独立评价DT。后续恢复DT的因果预测上下文，保留DAG强基线，不为排名削弱它。方向型baseline仍非外部论文忠实复现。

## 机制诊断

- 全8方法扩展remote-current判定发现11次可选加载驱逐其他当前声明依赖。环境和原auditor都漏了cache_target!=execution_target；原pass不能证明保护完整。原件保留，独立证据见final_forensics_20261017/trace_forensics.json。
- seed7四个long/dual/4096负例解释净奖励亏损87.72%；最差case额外admission字节90.56%来自反应加载。两个family闭包无法共存，选站未显式计完整冷模型/被挤占当前需求。训练漏long/stream/dual。
- seed27后32会话无long；dev040最终比中期少18节点、目标切换多56、state链路多59，model槽仍40。已重放frame11同状态c/d近50% argmax翻转。课程遗忘与薄弱状态迁移成本是候选机制，不假称已完成干预因果证明。
- SA未来/排队/output阶段3248对象仅1144使用；1961未用即驱逐，使用延迟中位20帧而未用存活中位16帧，1331对象驱逐后同站又需要。类别重叠；不是hit rate，也不是所有cache bytes多数浪费。
- PPOseed17没有future/queued/output阶段admission；base admission671 vs SA1709，base family切换474 vs1506。DAGseed17可选对象1881多于SA1491，但反应admission字节4059136低于5857792、family切换879低于1506：不能只按预取数量归因。先修使用时机/容量/站点协调，不删除预测器。
- surrogate和action Q仍以观察团队MC监督，不是识别的动作因果边际贡献；不能把预测误差下降或低KL当收敛证明。

## 独立后续实现范围

1. 修remote-current环境/消费审计判定，禁已输出当前节点的无目的准备。
2. 所有方法共同采用贯穿前后半程、每seed覆盖24条件的均衡交错日程；保持64会话预算。
3. 恢复DT-specific预测上下文并验证不同于DAG；不改变baseline动作权限。
4. 公共选站加入真实最小恢复时隙及冷模型/当前family冲突成本；可选准备按已发布需求使用时间、因果8帧预测驻留、容量风险和净节点交付代理筛选。保持物理reward/数据/费用不变，代理不是实际因果收益。
5. 独立v2 contract拒绝旧checkpoint；先软件验收再另冻一轮全部8方法共同确认，上限不超过737280帧，不再延长/筛seed/选峰。

这是独立审查，本文记录时算法未修改。下一实现另行提交。论文claim、泛化/novelty和TITS/TVT投稿适配本轮均未评分。

## 原始证据

- artifacts/joint_stream_learning/20261015T181642Z_development/{audit.json,launcher_result.json,results.jsonl}
- artifacts/joint_stream_learning/final_forensics_20261017/{independent_review.json,trace_forensics.json,prefetch_objects.jsonl}
- artifacts/joint_stream_learning/deep_pair_review_20261016/diagnosis.json
