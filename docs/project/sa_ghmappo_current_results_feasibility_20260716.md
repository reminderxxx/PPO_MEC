# SA-GHMAPPO Current Results Feasibility Report

- `prepared_at`: `2026-07-16`
- `purpose`: 单独展示当前主算法结果，用于证明研究方向和系统机制具备可行性
- `algorithm`: `sa_ghmappo` / SA-GHMAPPO
- `evidence_level`: `E2_ARTIFACT_AUDITED`
- `primary_artifacts`: `strict_full_v8_formal_all_baselines_20260621_v1`, `strict_full_v8_hidden_holdout_20260621_v1`
- `supporting_artifact`: `strict_full_v8_external_lust_grid_20260621_v2`
- `statistics_protocol`: `hierarchical_window_bootstrap_v1_20260621`

## 一句话结论

当前 SA-GHMAPPO 已经证明研究方向可行：在冻结的 NGSIM + Alibaba strict-full 协议上，主算法在 formal 和一次性 hidden holdout 中对全部 learned baselines 的 total reward 均取得正向 hierarchical BCa 95% CI；同时相对 DT handoff baseline 的 workflow continuity 也取得正向 CI。这说明“跨 RSU 连续 DAG workflow + adapter cache + predictive handoff prepare + 多时间尺度控制”的联合问题不是只停留在概念层，而是在真实 mobility/workflow 链路上形成了可复核的性能增益。

但该结论是“可行性已成立”，不是“顶刊已可直接通过”。当前仍存在 handoff failure / backhaul trade-off、未显著超过 popularity heuristic、v8-current support suite 未补齐等边界。

## 1. 结果协议

| 项目 | Formal | Hidden holdout | LuST supporting |
|---|---:|---:|---:|
| Mobility / workflow | NGSIM + Alibaba | NGSIM + Alibaba | LuST + Alibaba |
| Split role | frozen formal | one-time hidden holdout | external low-power support |
| Outer windows | 20 | 20 | 4 |
| Paired rows / baseline | 200 | 200 | 40 |
| Seeds | 7, 13, 29, 41, 53 | 7, 13, 29, 41, 53 | 7, 13, 29, 41, 53 |
| Statistics | window-outer hierarchical BCa 95% CI | window-outer hierarchical BCa 95% CI | window-outer hierarchical BCa 95% CI |
| Evidence role | 主证据 | 候选冻结后一次性验证 | 方向一致辅助证据 |

关键点：

- train/dev/formal/hidden split 已冻结且全局互斥，minimum gap 为 24 frames。
- hidden 在 candidate v2 和 checkpoint manifest 冻结后只开启一次。
- formal / hidden 都使用同一 5-seed checkpoint manifest。
- 统计单位不是单行 episode，而是外层 `window_id`，内层 `seed + workflow_id`。

## 2. 主算法绝对表现

| Split | SA total reward | Workflow continuity | Handoff failure | Backhaul cost | 解释 |
|---|---:|---:|---:|---:|---|
| formal | 88.0168 | 0.896602 | 0.097917 | 143.44 | 主算法在正式 frozen split 上形成稳定综合 reward。 |
| hidden | 83.02555 | 0.860907 | 0.139167 | 126.56 | 一次性 hidden 上 reward 仍保持可迁移正优势。 |
| LuST support | 95.6225 | 1.000000 | 0.000000 | 120.00 | 外部 mobility 子集方向一致，但只有 4 个 outer windows。 |

解读：

- formal 和 hidden 的 absolute reward 都来自同一套 frozen checkpoint，没有 hidden 后调参。
- continuity 在 hidden 上仍保持 0.86 以上，说明方法不是单纯通过破坏 workflow continuity 换 reward。
- handoff failure 与 backhaul 仍是当前需要治理的系统 trade-off，不能写成已全面优化。

## 3. 对全部 learned baselines 的 total reward 优势

| Learned baseline | Formal SA - baseline, BCa 95% CI | Hidden SA - baseline, BCa 95% CI | 可行性判断 |
|---|---:|---:|---|
| PPO | +6.656750 [4.013080, 10.782299] | +3.884200 [1.612143, 7.830383] | 正向，hidden 仍成立 |
| MAPPO | +10.553850 [7.331586, 15.428486] | +9.123500 [5.737617, 13.391786] | 正向 |
| DQN | +30.253300 [24.134327, 35.922997] | +26.042550 [20.358253, 31.357119] | 正向 |
| Dueling DQN | +37.327550 [30.192807, 43.086340] | +32.804050 [26.051377, 38.319775] | 正向 |
| QMIX | +22.859050 [17.700274, 28.437574] | +18.961050 [14.387019, 24.252194] | 正向 |
| Controller-MAT | +10.545250 [7.099470, 16.358703] | +8.741750 [5.262094, 13.067431] | 正向 |
| DAG-Offload-DRL | +15.058100 [10.710342, 20.423193] | +12.915400 [8.527139, 17.160511] | 正向 |
| Cache-Offload-DRL | +11.079250 [6.980455, 16.150704] | +9.474450 [5.063717, 13.802471] | 正向 |
| DT-Handoff-DRL | +17.893100 [12.429725, 24.247976] | +16.323300 [10.347505, 21.426419] | 正向，且幅度最大的一类核心近邻 |

这张表是证明可行性的主证据。它说明 SA-GHMAPPO 不是只超过一个弱 baseline，而是在相同 frozen window、seed、workflow 和 checkpoint manifest 协议下，对 9 个方向匹配 learned baselines 全部取得正 CI。

## 4. 相对 DT handoff baseline 的 continuity 修复

| Split | Metric | SA - DT handoff DRL / signed benefit | BCa 95% CI | 判断 |
|---|---|---:|---:|---|
| formal | total reward | +17.893100 | [12.429725, 24.247976] | 正向 |
| hidden | total reward | +16.323300 | [10.347505, 21.426419] | 正向 |
| formal | workflow continuity | +0.280658 | [0.184057, 0.364369] | 正向 |
| hidden | workflow continuity | +0.266788 | [0.154254, 0.356273] | 正向 |

这证明 v7 阶段的 strict-full DT continuity blocker 已被 v8 实质修复。对论文写作最重要的含义是：当前方法不再只是 total reward 点估计领先，而是对“workflow continuity”这个与 handoff 机制直接相关的系统指标，也能在 formal 和 hidden 上同时给出正 CI。

## 5. 外部 LuST supporting evidence

| Comparator | LuST SA - baseline total reward | BCa 95% CI | 解释 |
|---|---:|---:|---|
| PPO | +6.993500 | [0.454110, 13.443500] | 方向一致 |
| MAPPO | +16.110000 | [8.366929, 24.648996] | 方向一致 |
| Controller-MAT | +16.846750 | [8.722116, 26.236374] | 方向一致 |
| DAG-Offload-DRL | +14.531500 | [7.734000, 21.058574] | 方向一致 |
| Cache-Offload-DRL | +12.043750 | [4.936142, 17.977818] | 方向一致 |
| DT-Handoff-DRL | +10.567000 | [4.321853, 15.050234] | 方向一致 |

LuST 子集说明方法迁移到另一类 mobility trace 时仍有正向趋势；但该证据只有 4 个 outer windows，低于项目审查政策的 12-window 门槛。因此它只能作为 supporting evidence，不能单独证明强泛化。

## 6. 为什么这些结果能证明研究可行

### 6.1 问题设定可行

结果显示，跨 RSU 连续 DAG workflow、adapter warm-state cache、预测 handoff prepare 和多时间尺度控制并不是无法学习的过度复杂问题。SA-GHMAPPO 在 formal / hidden 上都能从该联合状态空间中学到比 PPO、MAPPO、QMIX、Controller-MAT 和领域专项 DRL baseline 更有效的策略。

### 6.2 方法结构可行

与 flat PPO 或单领域 DRL 相比，SA-GHMAPPO 的优势在两个层面体现：

- 对 DAG/cache/DT 专项 learned baselines 全部正 CI，说明单独强化 DAG、cache 或 handoff 维度不足以覆盖联合机制；
- 对 DT-Handoff-DRL 的 continuity 正 CI，说明 event/handoff 机制不是只提升 reward，而确实改善了 workflow continuity。

### 6.3 实验协议可行

当前证据不是 smoke run 或单 seed 结果，而是：

- 5 seeds；
- 20 formal outer windows；
- 20 hidden outer windows；
- formal 后一次性开启 hidden；
- hierarchical BCa 95% CI；
- 完整 artifact integrity audit，11457 个文件，missing reference 为 0。

这足以证明研究路线具备继续冲击高水平论文的实验基础。

## 7. 必须保留的边界

当前不能写成：

- 全面优于所有方法；
- 显著优于 popularity heuristic；
- 全面降低 backhaul；
- handoff failure 全面改善；
- LuST 已证明强跨数据集泛化；
- 已达到 TMC-ready。

必须写成：

- 当前研究可行性已被 strict-full formal / hidden 证据支持；
- learned baseline superiority 成立；
- DT continuity blocker 已修复；
- 但 failure / backhaul / heuristic gap / v8 support suite 仍是 major revision 项。

## 8. 推荐论文结果表述

可直接用于论文或汇报：

> Under the frozen strict-full NGSIM+Alibaba protocol, SA-GHMAPPO consistently outperforms all nine learned baselines in total reward on both the formal split and the one-time hidden holdout. The improvement over PPO is +6.66 on formal and +3.88 on hidden, with BCa 95% CIs entirely above zero. Against the handoff-specialized DT baseline, SA-GHMAPPO further improves workflow continuity by +0.281 on formal and +0.267 on hidden. These results demonstrate the feasibility of jointly controlling continuous DAG execution, adapter warm-state cache, and predictive handoff preparation across RSUs. We report the remaining handoff-failure and backhaul trade-offs separately rather than claiming universal system dominance.

中文版本：

> 在冻结的 NGSIM + Alibaba strict-full 协议下，SA-GHMAPPO 在 formal split 和一次性 hidden holdout 上均显著超过全部 9 个 learned baselines。相对 PPO，formal / hidden 的 total reward 增益分别为 +6.66 和 +3.88，BCa 95% CI 均完全大于 0；相对 handoff 专项 DT baseline，workflow continuity 在 formal / hidden 上分别提升 +0.281 和 +0.267。这说明跨 RSU 连续 DAG 执行、adapter warm-state cache 与预测 handoff prepare 的联合控制具备真实数据链路上的可行性。当前仍需单独报告 handoff failure 与 backhaul trade-off，不将其包装为所有系统指标全面领先。

