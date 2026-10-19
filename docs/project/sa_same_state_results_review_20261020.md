# SA第五阶段完整独立审查：有效动作、信用与同状态决策

- reviewed_at: 2026-10-19T20:02:58.259105+00:00；本地日期2026-10-20。
- literature_cutoff: 2026-10-17沿用，本轮不刷新文献或评价novelty。
- target_venue: IEEE TMC内部诊断；policy_version: `tmc_review_policy_v3_20260621`。
- artifact_run_id: `sa_effective_credit_stage5_20261020/same_state`；原科学run `20261018T082032Z_sa_group_clip_pair`。
- Git/source: `27f6228f5df0e4c9a3a5e9ef3f92a7f84d5d6d08`（已真实push）；算法source `8cacb6de54c1cbde1da6f8089d9060810707958a`。
- evidence/verdict: 开发工件完整审计及离线复算通过；正式论文 `E0_UNAVAILABLE / Unverifiable`，N/S；**未识别唯一退化原因，不晋级分组候选。**

## 1. 完整性、费用与协议

5A先行[只读报告](sa_effective_credit_review_20261020.md)保留：未发现MC重复/8或FIFO重置错误。随后唯一5B已于2026-10-19T19:40:01Z结束。144冻结身份、63944实际参考帧（上限69120）、1296原长度九分位帧（含13idle）；18542两遍捕获、9271第二遍行为记录、129794条件评分、259588 head记录全部回收。资格另列16旧snapshots/112forward，环境0。训练、backward、Adam均0；回收也不构造env/agent、不forward、不重放。

独立完整性2382094项检查通过：130旧source+7helper/runner绑定（133个不同文件）、18checkpoint/6初始化、历史legacy32/64、864会话输出文件SHA；144原summary逐字段exact；479190条event与63944条receipt canonical SHA逐格与阶段四旧ledger一致。原trace与同reference复评分、两0模型、batch/context/prefix/MC标签、RNG/权重/optimizer/controller守卫通过。另执行链审查1032项SHA检查通过；根107项交叉验证通过（其中含图中75个数值）；安装后另168项匹配检查通过。

完整1296分位帧均保留，未按reward或机制补选。18542不是18542独立执行机会：首/二pass的execution对全部7评分条件逐对相同，执行仅计9271条；其中两臂0点重现相同身份，仍不能称9271个独立状态。两相邻已消费时间session、后验富集8case和相关帧，不提供独立时间CI/泛化证据。原全48面板[第四阶段主效果与64会话/9点曲线](sa_group_clip_results_review_20261019.md)保持，不用此子集重估排名、收敛或模块贡献。

## 2. 可选类别与物理同效

| head（第二遍） | 记录 | singleton forced | 类别多选 |
| --- | ---: | ---: | ---: |
| execution | 9271 | 6351 | 2920（31.50%） |
| cache | 9271 | 9250 | 21（0.23%） |
| prepare/priority | 9271 | 0 | 9271（100%） |

forced cache可以完成必须的冷模型准入，非noop。21个自由cache只是固定分位见证，不能替代5A全训练约1.1–1.5%的类别自由率。原stochastic每行信用仍未存，不能从本次greedy推回真实训练梯度。

固定reference prepare、原own claims及其余完整co-controls，只改变4个priority编号，按全局稳定处理顺序划分一帧物理充分类：

| 臂 | seed | 捕获记录 | 1组 | 2组 | 3组 | 4组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版 | 7 | 1486 | 211 | 1275 | 0 | 0 |
| 分组 | 7 | 1535 | 214 | 1321 | 0 | 0 |
| 原版 | 17 | 1543 | 214 | 1329 | 0 | 0 |
| 分组 | 17 | 1562 | 215 | 1347 | 0 | 0 |
| 原版 | 27 | 1542 | 182 | 1178 | 161 | 21 |
| 分组 | 27 | 1603 | 215 | 1388 | 0 | 0 |
| 总计 | — | 9271 | 1251 | 7838 | 161 | 21 |

37084个reference-prepare编号缩至17494个保守组，条件编号收缩19590/37084=52.8260%；另60个其他prepare候选保持UNVERIFIED单独桶，**不同桶不证明物理不同**。9250条记录存在非平凡组，非平凡组共9252个（两条记录各有两组），不能混淆组数与状态数。

该类内真实参考条件概率mass均值0.773448；Q范围均值0.004980、surrogate-return范围0.019336（MC/80单位）。原priority条件熵均值1.387102，保守partition0.504091，类内差0.883011。它们说明参数化对同效编号仍敏感，**不是52.83%的原训练梯度浪费，也不证明Q的长程因果误差**。

关键限制：完整co-controls可能包含该owner采样时尚未生成的后续priority，故离线类不能直接作在线mask/熵项/PPO概率折叠；必须先证明因果可用时点、完整controller续接及正确联合/条件logprob。物理一帧同效不等同event编号、学习状态或长期return-Q等价。33条单owner见证与5A全训练约0.5%的事实保留；分位可能漏已知frame220/221，无见证不等于不存在。

## 3. 同状态排序：有改变，但改变不是收益

每seed所有冻结参考输入分别评分六模型及prior-only；首pass与相同teacher-forced prefix下比较。0点保留公共prior/mask/协调/GRU，非随机baseline。以下每列是0/32/64三份真实checkpoint的计数，非新轨迹：

| 模型臂 | seed | 自由输入数 | 严格覆盖public prior：0/32/64 | 相对自身0的argmax改变：0/32/64 |
| --- | ---: | ---: | --- | --- |
| 原版 | 7 | 964 | 78 / 83 / 81 | 0 / 212 / 210 |
| 分组 | 7 | 964 | 78 / 101 / 105 | 0 / 218 / 176 |
| 原版 | 17 | 987 | 47 / 125 / 146 | 0 / 145 / 161 |
| 分组 | 17 | 987 | 47 / 85 / 84 | 0 / 56 / 44 |
| 原版 | 27 | 969 | 49 / 144 / 122 | 0 / 270 / 238 |
| 分组 | 27 | 969 | 49 / 203 / 187 | 0 / 238 / 169 |

含prior平局换赢家的完整计数、排名及700个确定性首见反例保留在compact明细。不能把平局取不同项统称压倒了先验；cache/priority的零prior更不能称覆盖有信息先验。一个明确反证是dev046/frame0/peachtree:223：seed7与17的batch/candidate/context/decision SHA完全相同，legacy0→64都从rsu_c改排rsu_a，但原版seed7同case学习改善、seed17下降；这种局部翻转不是退化充分条件。

## 4. 已付状态后离站与真实兑现

144原参考全事件含16962次同node RSU换站，8827次满足严格段内paid-before-compute定义（旧连续目标段付过state、未compute且离开前progress=0）。固定九分位采到338次换站、172次此类离站；其中168旧站仍在实际mask，163 age为8–11，163由public prior本身选新，仅9次严格残差覆盖prior。未采到离站的oldtarget资格仍UNVERIFIED；这些是记录段而非物理车辆到站，不能把paid直接叫浪费。

9个严格覆盖包括：legacy7/32/dev022/f59（case自身学习正、节点68交付）；legacy17/32/dev000/f26；group17/32/dev038/f179；group17/64/dev018及dev022/f299；两臂27/0/dev046/f299重复初始化身份；group27/32/dev022/f119；group27/64/dev046/f239。六个负学习cell、一正cell和两个重复0身份全部保留，不能把两个0身份当独立学习反证或用case回报给单动作定因果。

- **group17/64/dev022/f299**：旧rsu_d真实base/adapter和最新state就绪，新rsu_c冷family；prior旧−0.04144、新−0.15470，legacy0/32/64及group0留旧，group32/64选新。302重新加载、306计算、308交付；同帧另一车逐出旧family，换本车目标也会改保护集合，不能称留旧必定成功。
- **group27/64/dev046/f239**：旧rsu_b就绪且state剩0，新rsu_a冷3帧+family冲突2.5；prior旧0、新−0.09129。group32/64转新，其他模型留旧；此前付4819字节state。新站closure227准入、235被另一family替换、240再准入，243算、245交付；共享费用不能强分本车。
- **legacy17/64/dev046/f359**：七条件都选新；双方queue proxy71、旧state剩17而新0、模型均冷，实际403计算、409交付。反驳失败全由残差推翻正确prior，也不证明替代旧站更快。
- **legacy7/64/dev043/f119**：旧段付7168字节state仍转到就绪新站，120计算、122交付；反驳对已付离站统一惩罚。legacy27/dev046/f179残差反而保留就绪rsu_c（191计算、198交付），反驳禁止全部prior覆盖。
- 候选正例仍保留：seed7/dev045最终419.479/12完成（原版403.870/10），seed17/dev046为183.629/6（原版176.205/4），seed27/dev021为403.574/13（原版402.225/13）。原版7/27在dev046自身0→64分别+146.921/+87.281，17在dev022为−66.651，正负角色未改。

现有prior已消费queue、剩余state restore、cold/loading span与peer family冲突，不能再称“完全缺state/model成本”。这些proxy与联合实际兑现不同，尚未识别可保证改善的选站补丁；一律锁站、追加sunk-cost惩罚、压残差或继续调clip/lr均无充分依据。

## 5. 事实信用预测偏差不等于原训练信用错误

标签路径保持physical MC/10，再/公开fleet8；V/prefix/Q/surrogate-return对MC/80。model辅助标签是新准入完整名义link slots/10，state是实际event slots/10，无/8；不是本帧真实model费用，也不再扣进reward。每head baseline均按保存概率重建`.5*(Q+surrogate return)`期望，selected-Q/V/prefix/labels映射通过；第一遍未计算Q/V保持None，未新增forward补造。

仅exact reference模型第二遍对自身greedy事实MC作描述性偏差，其他checkpoint/prior-only不获伪事实误差。固定64自由execution：

| 参考臂 | seed | n | baseline MSE | global V MSE |
| --- | ---: | ---: | ---: | ---: |
| 原版 | 7 | 144 | 2.078650 | 2.285778 |
| 分组 | 7 | 158 | 1.660475 | 1.950908 |
| 原版 | 17 | 160 | 1.179778 | 1.435313 |
| 分组 | 17 | 142 | 1.341793 | 1.680914 |
| 原版 | 27 | 153 | 2.102130 | 2.267856 |
| 分组 | 27 | 181 | 1.155452 | 1.303454 |

单位为(MC/80)²。原版17的两项偏差低于正学习7/27，因此偏差大小不充分解释17退化；分组7/27偏差也可更低而效果更差。状态/target分布随参考轨迹不同，且网络训练来自stochastic行为，greedy事实偏差不是真实on-policy Q误差、原PPO loss/advantage或收敛证据。forced/free及所有head全表、类内预测范围、正负反例在私有明细保留；全局V在三head重复，汇总不按三份独立信息计。

离线consumer修复仅涉及旧资格witness缺arm/episode、float32公共logit偏移的逆log舍入、subnormal半ULP量化边界。两个失败派生目录及反例保留；原科学run无failure、未修改/重跑。最大逆log差0.014824发生在约exp(−100)的subnormal概率，保存logprob仍为原始float32记录；尺度/半ULP界只用于概率/熵恒等式，不放宽源SHA、精确reference parity或1e−8效果门禁。

## 6. 图、claim与下一任务

三幅ReportLab标准静态图（SVG/PNG+75数值）在私有`recovery_20261020/plots/`；用户可见副本 `/Users/howen/.codex/visualizations/2026/10/12/01a12f37-a381-7740-a214-97e84fa45d2c/sa_stage5_final_20261020`：`action_and_priority_classes.png`、`same_input_execution_rankings.png`、`factual_reference_deviation.png`。均标真实0/32/64条件及取样/概率边界，无插值、无新训练曲线；第四阶段原64会话/9点图继续保留。

安全结论：条件动作别名存在、自由head稀疏、MC/强制屏蔽路径成立、具体prior/残差与共享family反应链可追溯。禁止结论：已定位唯一退化原因、已证实不可避免/可避免state浪费、可直接在线折叠52.83%动作、已解决seed17、收敛/泛化/GNN/预测因果或candidate晋级。

选择一个后续优先级：**5C先冻结因果可用的有效动作/信用概率合同及真实行为审计字段，再决定单一opt-in候选；不叠加锁站/换reward/分组clip调参。** [具体5C任务与拒绝条件](sa_learning_stage5c_tasks_20261020.md)仅任务点，本轮未实施、未授权软件Adam或新训练。原版固定64、整体及seed7/27所有完整精度reward/完成率/model/每交付门禁、尾稳标准保持；formal/holdout与canonical不变。

## Evidence与验证

- 私有回收：`/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC/artifacts/joint_stream_learning/sa_effective_credit_stage5_20261020/recovery_20261020`；包含完整性/身份/语义审查、执行forensics、credit summary/compact/entropy、两个consumer-only失败派生目录、root_crosscheck和图。
- 完整性报告SHA `5fdfe5b650c15378829a0a959b4ee8615c0b04f91e1f69203a846e7ade6f4de6`；consumer helperSHA `c0c2f3c273dddaa31c61eb991e970d32097383c02bfd0bf742c62475c20c561c`，语义独立审查同SHA通过。
- 原资格16/112及968恒等式；consumer18纯软件fixture及原runner22项合计40项通过（`PYTHONDONTWRITEBYTECODE=1 /Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sa_stage5_credit_audit.py tests/test_sa_stage5_same_state.py`）。安装后168项检查确认133个冻结source不变、consumer除docstring/六项summary metadata外AST相同、3图75数字exact及语法/链接匹配；root107项包含该75图数值，3幅visual QA通过。安装验证/commit/push以私有同目录`validation.json`/`git_sync.json`记录为准；没有额外环境评估。
- 复核命令在各helper/report中：完整性3条只读命令、consumer `qualification`/`summarize --trusted_tensor_capsules`（只读取已保存tensor）、plot脚本使用现有ReportLab运行时。无新增环境、agent或模型forward。
- 未覆盖：原stochastic行为T/B/advantage/梯度、未采样时刻完整资格、替代动作长期回报、在线可用的联合类概率以及正式外推。
