# 2026-10-28 奖励图交付结果与证据边界

- `reviewed_at`: `2026-10-28`，Asia/Shanghai。
- `literature_cutoff`: `2026-10-28`；复用本轮五篇原论文实际图页核验，不重复检索。
- `target_venue`: CSCWD2027；TMC/ToN只作为内部图形与证据规范参考。
- `artifact_run_id`: `paper_reward_curve_campaign_20261028_v1`。
- `policy_version`: `tmc_review_policy_v3_20260621`。
- `git_commit`: 科学源码 `e00d5faf9c3f73a689174237245ed6f85394b2f2`。
- `evidence_level`: 已消费train学习曲线原件已审计，九个事前固定final episode逐帧及控制摘要重放一致；整体论文仍为 `E0_UNAVAILABLE / Unverifiable`，不将局部审计等同正式E2。
- `verdict`: 当前奖励学习图已交付；明显学习收敛、独立泛化、formal/holdout/support及顶刊录用均未证实。`paper_goal_complete=false`。

## 完成的范围

按[事前合同](paper_reward_curve_contract_20261028.md)单次启动，20个cell全部成功，无删除seed、重训、峰值选择或算法修改。
四方法各五seed、96完整训练episode；共1920训练、960固定评价、260checkpoint。
实际训练751748帧、固定评价373708帧、规则4482帧和独立重放3494帧，共1133432物理帧，低于1393920上限。
自动回收于2026-10-28T10:58:21Z结束。复用其审计输出，不重复执行原审计、训练或重放。

本机正式产物根为 `artifacts/analysis/paper_reward_curve_campaign_20261028_v1/collection/`：
`figure/reward_learning_curve.{pdf,svg,png}`、`figure/reward_vs_actual_frames.{pdf,svg,png}`、
`figure/seed_points.csv`、`figure/caption.txt`、`plot_source.json`、`resource_bill.csv`和`audit.json`。
PDF/SVG为矢量；PNG为600dpi。全部九个图件/数据文件hash与图清单一致，plot与audit的SHA绑定一致。
两份PDF均经Poppler渲染并视觉检查，字体、坐标、图例无裁切或重叠缺陷。

## 固定三probe学习图

每seed先平均固定 `train_000/train_038/train_067`，再跨全部五seed画均值与一倍样本SD；无平滑，SD不是CI。

| 方法 | 初始均值 | 固定ep96均值 ± seed SD | final减initial |
| --- | ---: | ---: | ---: |
| 当前主方法 | 221.324 | 223.365 ± 3.142 | +2.041 |
| 本地MAPPO | 187.949 | 186.058 ± 14.396 | -1.892 |
| 本地PPO | 187.949 | 188.083 ± 12.988 | +0.134 |
| 本地DQN | 187.949 | 157.195 ± 3.800 | -30.754 |

同三probe的冻结gain规则为201.734；主方法final高10.723%。这是固定条件的描述性差异，不是独立窗口统计显著性。
主方法seed17/47的initial到final变化分别为-4.124/-8.955，其余三个为正；所有回退均保留。
主方法相对学习对照的初始差距已经明显存在；本轮不能将最终整体优势归因于PPO训练增量。
当前图更支持“该完整方法在固定开发条件下奖励较高”，不能写成“从低回报显著学习并收敛”。

## 固定final全部12条件，不能替换曲线最后点

| 方法 | 全部12条件均值 ± seed SD |
| --- | ---: |
| 当前主方法 | 189.499 ± 2.312 |
| 冻结gain规则 | 173.834，无虚构seed重复或SD |
| 本地PPO | 167.692 ± 10.809 |
| 本地MAPPO | 164.716 ± 10.694 |
| 本地DQN | 143.093 ± 2.480 |

主方法相对gain规则为+9.011%，相对本轮最强学习对照PPO为+13.004%。五seed的12条件均值均超过规则，
但60个seed/condition配对中仍有13个负例，不能声称逐条件全面获胜。
主方法平均完成15.300/16个请求，gain为14.083/16；同时主方法平均状态传输27672.117字节、handoff失败计数7.067，
高于gain的8231.083字节和1.083。累计事件计数可能含同一请求多次事件，不能改称独立失败请求率。
奖励与完成的改善不证明所有迁移、状态或无线成本都改善。

## 公平边界与未覆盖风险

全部策略随机初始化，但共享GRU此前已在train拟合；主方法还有节点迁移先验和额外意图动作限制。
同外生会话不等于同可选策略集合、实际帧或梯度成本。主方法/MAPPO/PPO/DQN实际训练帧分别
182681/188920/188995/191152，精确optimizer.step调用分别17624/9852/9884/3840；账本保留各优化器明细。
本地实现共用planner和候选信息，不称外部论文忠实复现。主方法与各对照240组固定评价控制摘要均不同；
MAPPO/PPO/DQN两两仍各有15/240相同控制轨迹，未删除这些结果。

输入均为已消费train时段；五seed和三个probe不构成新的独立时间cluster。未开启formal/hidden，
不计算伪独立显著性、不晋级canonical、不改论文主表。整体论文评分为N/S，不能据奖励图称TMC-ready。
主要缺口仍是独立时间/DAG与holdout/support、公平且足够强的外部对照、机制消融及真实任务/无线/RSU成本校准。
若要进一步归因学习收益，应先冻结同初始化/同动作集合的消融和有界预算，不从本图峰值继续选模型。

## 验证与交付

启动前64项合成软件检查已通过：
`python -m pytest -p no:cacheprovider tests/test_stream_reward_curve.py tests/test_stream_reward_curve_collection.py tests/test_stream_reward_curve_launcher.py -q`。
科学parent自动执行 `collect_stream_reward_curve.py`，核对全cell、checkpoint、逐帧原生奖励、MC/credit、控制及预算，九重放全部一致。
交付时只读复用审计，独立核对图件hash、plot/audit绑定、九重放摘要和固定三probe/final12聚合；两PDF视觉检查通过。
主目录七个用户修改文件和R20哈希均保持原值。原始数据、checkpoint、实验和私有审查不进入Git提交。
