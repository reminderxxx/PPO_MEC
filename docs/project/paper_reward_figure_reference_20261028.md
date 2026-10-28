# 2026-10-28：奖励与学习曲线的一手图页核验

- `reviewed_at`: `2026-10-28`，Asia/Shanghai。
- `literature_cutoff`: `2026-10-28`。
- `target_venue`: 项目论文目标沿用 CSCWD2027；TMC/ToN 在此用于图形与证据规范参考，不作录用承诺。
- `artifact_run_id`: `public_reward_figure_reference_20261028`；本任务没有训练、策略评价、轨迹下载或本地私有材料上传。
- `policy_version`: `tmc_review_policy_v3_20260621`。
- `git_commit`: `dc530c798f9ade5fd77fbb699715c2a8986be4ce`（调查起点）。
- `evidence_level`: `E1_DOCUMENTED`（公开文献与绘图协议参考；没有项目科学 artifact 晋级审查）。相关图页及实验段落已直接核验，不等于复现其数值；`paper_goal_complete=false`。
- `verdict`: 可用作绘图设计依据；不能由本报告判定 PPO_MEC 论文已完成或达到 TMC-ready。

## 实际检查了什么

检索只使用公开论文标题和研究关键词。五篇论文的原始 PDF 均在本机临时目录阅读，使用 `pypdf` 提取全文，再用 Poppler 渲染下表指定整页并视觉核对图号、轴、图例和 caption。公开原图没有复制进仓库，也没有从曲线取点制造本项目数据。页码均为 PDF 从 1 开始的页面；有最终期刊页码时另列。

| 原论文及一手来源 | 实际图页、轴与数据口径 | 重复、阴影和平滑的披露 | 可借鉴与不得推断 |
| --- | --- | --- | --- |
| **Multi-User Layer-Aware Online Container Migration in Edge-Assisted Vehicular Networks**，ToN 32(2):1807–1822，2024。DOI `10.1109/TNET.2023.3330255`。[IEEE](https://ieeexplore.ieee.org/document/10315072/)；[作者最终稿 PDF](https://fangyimou.github.io/papers/ton-2024.pdf#page=14)。 | **Fig.11(c)**，PDF p14 / 期刊 p1820：横轴 `Epoch`，0–1000；纵轴 `Average Total Reward`，约 −11.5 至 −8。OCM、DRL 变体与 Greedy/Kube 同图，规则线为水平线。相邻 (a)(b) 是 policy/network loss；图题与图号按实际 caption，正文误指 Figure 8，不能据误指另造图号。 | p1817 实验段说每个结果重复 **5–6 次**、实线为均值、阴影原文称 `standard variance`；未明确这些重复是否独立训练 seed，也未给该术语计算公式。Fig.11(c) 没有可清楚辨认的 CI 带。相关正文未交代平滑窗或独立 evaluation rollout。 | 可借鉴“奖励 + 损失/系统机制”的布局及规则线；不能把 5–6 次重复写成 6 个 seed，也不能替作者解释成 95% CI。训练相关的奖励图不是独立泛化证明。 |
| **Mobility-aware Seamless Service Migration and Resource Allocation in Multi-edge IoV Systems**，SR-CL；[作者 arXiv 原稿](https://arxiv.org/abs/2503.13494)，2025；[图页 PDF](https://arxiv.org/pdf/2503.13494#page=10)。项目已有 DOI 线索 `10.1109/TMC.2025.3540407`，本次 DOI 页面未打开；最终 TMC 卷页仍不凭二手页面晋级。 | **Fig.8(a)(b)**，作者 PDF p10：横轴 `Training Episodes`，0–1500；(a) `Reward`，主要显示约 −9×10⁴ 至 −5×10⁴；(b) `Migration Frequency`。SR-CL、DDPG、JSR、IDQN 与 GA/NM/AM 同图，规则水平线；有早期/尾部局部放大。p9 说明每 episode **240 个 time slots**。 | 当前作者版本的相关实验段落和 caption 未找到随机 seed / 重复数量、误差带公式或平滑窗；Fig.8 未显示统计置信带。是训练 episode 奖励，未明确另开固定评价池。 | 可借鉴“同一训练进程的奖励 + 迁移频率”双面板，规则与学习方法区分；不能把外形平滑视为多 seed 稳定性，也不能把 1500 episodes 等同本项目固定帧预算。 |
| **Lyapunov-Guided Offloading Optimization Based on Soft Actor-Critic for ISAC-Aided Internet of Vehicles**，TMC 23(12):14708–14721，2024；DOI `10.1109/TMC.2024.3445350`。[作者机构目录](https://cam.tju.edu.cn/homepage/wuhuaming/PDF/)；[作者最终稿 PDF](https://cam.tju.edu.cn/homepage/wuhuaming/PDF/TMC_Liang.pdf#page=11)。 | **Fig.5**，PDF p11 / 期刊 p14718：横轴 `Iteration Number`，0–5000；纵轴 `Average Reward`，约 −40 至 0；LySAC/LyDDPG 两条线带阴影。相邻 Fig.6 用车辆数量对平均系统成本，属于物理指标补证。 | 同页正文把阴影变化归因于 **三个学习率** `0.02, 0.003, 0.005`，并非明确的跨 seed CI。未给独立训练 seed 数、阴影的具体聚合公式或平滑参数。正文称 episodes，图轴写 iteration，二者不可擅自统一。 | 可借鉴白底、两方法清楚比较和旁边的成本图；**不可把不同学习率变异画成跨 seed 误差带**，也不能将其当作本项目公平预算或统计设计模板。 |
| **The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**，NeurIPS 2022 Datasets and Benchmarks。[正式页](https://proceedings.neurips.cc/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)；[正式 PDF](https://papers.nips.cc/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf#page=7)。 | **Fig.2**，PDF p7：横轴 `Timesteps`；Spread 为 `Episode Rewards`，SMAC 为 `Win Rate`。有/无 value normalization 成对比较。p4 说明 MPE 结果平均 **10 seeds**；SMAC 每轮训练后跑 **32 evaluation games**，汇总表为 **6 seeds**。 | p13 checklist 明确训练曲线阴影为 **standard deviation**，不是 CI。Fig.2 caption 不单独重复 seed 数；本次未核补充材料，具体每条消融曲线的 seed 清单仍待核。正文未找到绘图平滑窗。MPE 曲线没有另述独立评价 rollout 口径。 | 采用真实 timesteps、明确阴影来源，分清 reward 与 success 指标。value normalization 是学习目标处理，不能据此把论文图的原始回报归一化成有利范围。 |
| **Deep Reinforcement Learning at the Edge of the Statistical Precipice**，NeurIPS 2021。[正式页](https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)；[正式 PDF](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf#page=9)。 | **Fig.10(right)**，PDF p9：横轴训练 `Number of Frames (in millions)`，0–200；纵轴 **IQM human-normalized score**，展示样本效率。**Fig.9** p8 给 55 个 Atari games 的聚合区间；通常每 game **5 runs**，M-IQN 3、DreamerV2 11。 | Fig.10 caption 明确阴影为 **pointwise 95% percentile stratified bootstrap CIs**。不是单个训练回合波动；未靠 EMA 定义误差带。Fig.6 p6 警示 3 runs 的 bootstrap 可能覆盖不足；Fig.5 p6 与正文指出训练过程最大值和固定 final 口径不可直接比较。 | 借鉴预声明聚合、点态区间和预算轴。PPO_MEC 不能把重叠窗口当独立 Atari tasks，也不能因 IQM 能收窄 CI 就事后改用 IQM，或称点态 95% 带为整条曲线的同时置信带。 |

## 从这些图得出的本项目执行规范

以下是针对 PPO_MEC 的设计决定，**不是期刊官方录用标准**，也不是以上论文共同满足的规则。

1. **先交真实证据图，再补学习过程。** 如果现存研究只保存 initial/final 或仅三次更新，立即画真实端点、每 seed 配对变化及最强规则对照；标注 `Warm-start development evaluation`。不得插值、复制点、平滑或拼接不同实验，制造几百轮的收敛曲线。现有负 seed/时段必须保留。
2. **主学习图以真实环境交互量为轴。** 建议 `Additional training environment frames`；warm-start 起点要明确父 checkpoint 和此前训练资源，不能把继承策略表现当从零学习。episode 数或 optimizer step 可作附轴/收据，但不同方法回合长度和 gradient 次数不同，不能隐去这种公平边界。
3. **纵轴使用冻结的原始奖励定义。** 建议 `Mean evaluation episode return`，评价与训练采样分开。每个 seed、每个预声明 checkpoint 在同一固定 dev 条件池运行，先按条件等权平均，再跨 seed 汇总；不得按不同完成数、帧数、成功回合或最优时段改权，也不得把 shaped training reward 与 unshaped evaluation return 混作一条线。严格 holdout 不用于画每轮挑选曲线。
4. **同图基线的信息和成本合同先冻结。** 固定规则在同一评价池上的均值可以是水平虚线，旁注其评价不确定性；已训练学习对照则需真实 checkpoint 曲线。所有方法共用原奖励、环境、输入时间和评价池，记录实际训练帧、episode、gradient 更新、模型查询和时间账单。规则不需假造训练曲线。不得改弱规则/其他基线来抬高主方法。
5. **把 seed 波动和抽样不确定性说清楚。** 初稿少量 seed 时优先显示每 seed 细线/点 + 均值，若画标准差直接写 `mean ± 1 SD across seeds`，不能写 CI。正式多窗口评价则按项目规范做窗口外层、seed 内层的配对层级 bootstrap，排除重叠窗口伪重复。只在事前冻结的统计量上计算点态 95% 区间；3 seeds 仍明确低功效，不用漂亮阴影代替更多独立证据。
6. **主图默认不平滑。** 固定评价 checkpoint 直接连真实点、附 marker；若训练回报辅助图需要 EMA，必须同时保存未平滑源数据，事前冻结 `alpha` / `span`、初始化和边缘处理，所有方法相同，并在 caption 写明。不得用未来点平滑当前点，或删去训练早期、失败 seed、回退阶段。
7. **奖励之外给净收益。** 建议双面板：(a) 固定池奖励/训练帧；(b) workflow 完成数与无效准备字节，或预声明的迁移/准备成本。主方法增益需能追溯到真实完成与成本，预测 RMSE 和 actor/critic loss 均不能代替它。负收益条件放附图/表而不是隐藏。
8. **输出同时可读、可审。** 白底、统一字体、清晰线型与色盲友好颜色；主方法线宽约 1.8 pt、其他约 1.2 pt、阴影透明度约 0.15–0.2；图例清楚标 full/no-prediction/no-optional/control。导出矢量 PDF/SVG 与 600 dpi PNG，留 CSV/JSON、hash、绘图命令、run/seed/checkpoint 清单和 caption。字号在最终单/双栏尺寸实际渲染复核，不用装饰性渐变或截断坐标放大微小优势。

建议 caption 所需信息：实际训练资源的定义；父 checkpoint 与 warm-start；固定 dev 评价池和每点 episodes 数；独立 seeds 与条件数；中心线统计量；阴影的准确含义；平滑参数或未平滑；规则线的计算；负 seed/失败是否完整保留。缺失任何字段写 `UNVERIFIED`，不能补猜。

## 来源版本与阅读凭据

临时公开 PDF 目录为 `/private/tmp/ppo_reward_reference_20261028`，只用于阅读，不是项目科学原件，不进入提交。下列 SHA 绑定本轮实际看过的文件；公开站点若换版，页码应重新核验。

| 临时文件 | SHA-256 | 实际渲染页 |
| --- | --- | --- |
| `ton_container.pdf` | `7539734954241bae8d75aebe5339d921d21773bdaa7bef2eb48447ad34a19c8b` | 14 |
| `tmc_srcl.pdf` | `90c823859cd73153d4a8035c1ce79cd2247a4db41a585f14d588b2a2c69a508c` | 10 |
| `tmc_isac.pdf` | `c5053e5baa86b7eeb2721401efaf5704729dd62eb3a764264497ccbdd2e57a59` | 11 |
| `mappo.pdf` | `3de7db922d915b4e6d7b3a05b07163a6eeaa5bf0d56fca019eef044b469686ff` | 7 |
| `precipice.pdf` | `7553e280d4a1560860bc5f6be2d70539fc5b510e0e3bb2d556824e51e9a80252` | 9 |

验证：五份 PDF 完整解析、上述五页已渲染并视觉检查；全文检索重复/seed、阴影、平滑与评价协议，并回读相关段落。本报告没有审过这五篇论文的全部补充材料或源代码；未找到的字段只表示当前核验版本未披露/未定位，不证明作者没有做。最终期刊统计充分性、PPO_MEC 数值优势、正式泛化与录用均不由本报告确认。
