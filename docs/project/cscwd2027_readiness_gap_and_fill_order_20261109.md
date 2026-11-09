# CSCWD 2027 投稿缺口评估与补齐顺序（2026-11-09）

用途：记录本轮对「CSCWD 2027 论文证据强度」的独立评估、与三层发表标准的对照、现存优点与缺口补齐顺序。
供后续 agent 或人工复读。**本文件不是论文结论，不晋级任何算法优势、canonical 或 paper-ready 判断。**

复读指引：

1. 先读本文件第二节的实测数字，再读第五节补齐顺序。
2. 需要确认数字来源时，按第八节命令直接从原始 `results.jsonl` 重算，不要引用本文档的表格代替原始产物。
3. 需要动论文表格或 claim 前，仍按 `top_journal_review_policy.md` 与根目录 `agent.md` 执行只读审查轮。

---

## 一、结论摘要

1. 论文现有主张「SAGH 在固定面板上 reward 与完成率最高」在数据上可复算，不是编造；但**证据强度不足以支撑跨种子稳定性**。
2. 核心缺口不是「算法没赢」，而是**统计可辨识性不足**：3 个训练种子、48 个复用条件、无区间估计，且 48 个条件全部嵌套在 2 个时间窗口内。
3. 项目自身 `top_journal_review_policy.md` 已明文禁止把 `(seed, window_id, workflow_id)` 的行冒充独立 cluster；当前 Table II 的口径与该条冲突。
4. 项目工程严格度（frozen manifest、hash、command log、1300+ 测试）**高于**多数顶会论文；统计功效**低于**顶会线。严格性用错了维度。
5. 补齐成本极不对称：8 种子扩展约一晚；G14 typed-cache 的 20 轮 protocol 修复耗时数月且 formal 结果仍为 0。

---

## 二、实测重算（原始产物逐行重算，非引用文档）

来源：`artifacts/joint_stream_learning/20261016T162100Z_confirmation/results.jsonl`，1152 行 = 8 控制器 × 3 seed × 48 dev case。

### 2.1 面板均值（与论文 Table II 一致）

| Controller | reward | completion |
|---|---:|---:|
| sa_ghmappo | 210.400 | 88.889% |
| dag_offload_drl | 205.289 | 87.630% |
| mappo | 203.774 | 86.849% |
| sa_no_prediction | 200.400 | 85.373% |
| cache_offload_drl | 200.245 | 86.068% |
| dt_handoff_drl | 197.817 | 85.069% |
| ppo | 193.407 | 84.505% |
| dqn | 164.064 | 75.521% |

论文报告的 SAGH 210.40 / 88.89% / 96.94 KiB 与本重算一致，说明 Table II 未经过选择性挑选。

### 2.2 按训练种子拆开（关键）

| seed | SAGH reward / 完成率 | MAPPO reward / 完成率 | Δreward | Δ完成率 |
|---|---|---|---|---|
| 7 | 219.42 / 91.67% | 211.74 / 89.19% | +7.67 | +2.47pp |
| 17 | 197.58 / 84.51% | 204.08 / 86.46% | **−6.51** | **−1.95pp** |
| 27 | 214.21 / 90.49% | 195.50 / 84.90% | +18.71 | +5.60pp |

**3 个种子中 1 个方向相反。** 均值优势由 seed 7 与 seed 27 承担。

### 2.3 配对 bootstrap（144 对）与其缺陷

| 对照 | Δreward [95% CI] | Δ完成率 [95% CI] | 胜出对数 |
|---|---|---|---|
| mappo | +6.626 [2.616, 11.148] | +2.04pp [0.65, 3.56] | 83/144 |
| dag_offload_drl | +5.111 [2.881, 7.443] | +1.26pp [0.17, 2.43] | 109/144 |
| ppo | +16.993 [12.821, 21.666] | +4.38pp [2.99, 5.86] | 137/144 |
| sa_no_prediction | +10.000 [5.743, 14.916] | +3.52pp [2.04, 5.12] | 100/144 |

**该 CI 不可作为显著性证据**：48 个 case 由 2 个相邻窗口 × 3 个 DAG 长度类 × 2 种释放制度 × 2 个模型族 × 2 档容量组合而成，两个窗口均来自父训练划分。把 144 行当独立样本会系统性低估不确定性（pseudo-replication）。按 `top_journal_review_policy.md`，外层独立单元是窗口（2 个），seed 与 workflow 只能在窗口内层级重采样；外层窗口少于 12 个必须显式标注低统计功效。

### 2.4 运行确定性（本轮实测）

同一协议、同一 seed 的两次独立 smoke 训练（4 episode + 1 dev case）产出**逐字节相同的 checkpoint**，SHA-256 前缀 `2b992b485a1308769ad967a2937797f4`，reward 逐位一致。

推论：该 pipeline 在固定协议 + 固定 seed 下可复现，run-to-run 噪声约为 0，**全部方差来自训练种子**。因此「重复跑几次取平均」不会产生新信息，加种子是唯一有信息量的实验。

限制：该结论由 4-episode smoke 得出，未在 64-episode 全长训练上验证。8 种子扩展运行会把 seed 7/17/27 排在序列最前，可直接充当全长复现自检。

### 2.5 历史证据盘点

`joint_stream_learning` 下仅两轮：

- `20261015T181642Z_development`：protocol v1，commit `e96d8315`，seeds [7,17,27]
- `20261016T162100Z_confirmation`：protocol v2，commit `35975cb3`，seeds [7,17,27]

两轮种子相同、协议与代码不同，故差异属版本变化，**不能当作同条件重复实验**。历史上从未出现过第 4 个训练种子。

---

## 三、三层发表标准对照

| 维度 | RL 顶会（NeurIPS/ICLR） | 通信顶刊（TMC/TON/TVT） | CSCWD 2027（CCF-C, 6 页） | 当前状态 |
|---|---|---|---|---|
| 种子数 | 5–10 | 3–5 | 3 已算规范 | 3 ⚠️ |
| 区间估计 | 事实必需 | 常见 | 少见 | 无 ❌ |
| 独立测试集 | 必需 | 必需 | 常有 | 无（复用训练窗口）❌ |
| baseline 强度 | 复现公开强方法 | 复现 + 启发式 | 内部对照常见 | 全内部变体 ⚠️ |
| 消融 | 必需 | 必需 | 加分 | 6 个变体 ✅ |
| 固定 checkpoint 不挑峰 | 加分 | 少见 | 少见 | 有 ✅✅ |
| 可追溯性 | 加分 | 少见 | 少见 | 极强 ✅✅✅ |
| 规模 / scalability | — | 必需 | 加分 | 8 车 4 RSU ⚠️ |
| 复杂度分析 | 常有 | 必需 | 加分 | 论文中 deferred |

依据：Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021 Outstanding Paper, arXiv:2108.13264（少 run 的点估计不可靠，主张区间估计、IQM、performance profile）；Colas et al., *How Many Random Seeds?*, arXiv:1806.08295（种子数与统计错误概率绑定，检中等效应通常需 ~10 种子量级）。两者均为 RL 方法学共识，非本项自定标准。

判断：

- **对 CSCWD**：现有设计基本合理且偏严，补 8 种子 + 区间估计即可达到「明显高于中位数」。
- **对 TMC**：会被 major revision，主要卡在独立窗口与 baseline 忠实度，而非种子数。
- **对 NeurIPS/ICLR**：不足，需要多环境、复现公开 SOTA、更大规模。

---

## 四、现存优点（后续修改不得破坏）

1. `primary_checkpoint='fixed_final_no_peak_selection'`：固定最终 checkpoint，不挑峰。这是**高于平均**的做法。
2. protocol 为 `frozen`，带 `source_commit`、`git_dirty`、assets SHA-256、`source_fingerprints`（按文件）、command log。
3. 同时报告 reward、完成率、模型传输量三个指标，且成本计入失败工作流；避免只报有利分母。
4. 六个内部消融变体覆盖预测、缓存、DAG 上下文、handoff 等组件。
5. 论文明确标注内部对照「do not denote faithful reproductions of published algorithms」；诚实但会被顶刊追问。
6. object-lifecycle receipts：1,078 次可选准入、757 次 (70.2%) 记录为使用、传输 880.5 KiB 其中 321.5 KiB 未见后续使用。这是很扎实的机制证据，应保留。
7. 论文 §V-D 已主动声明「descriptive, without independent-window confidence intervals」。补上区间后这段应改写，而不是删除。
8. 工程侧：`top_journal_review_policy.md` 的窗口独立性条款、E0–E3 证据分级、只读审查轮规则，均已是成熟资产。

---

## 五、缺口与补齐顺序

| 优先级 | 动作 | 成本 | 产物 / 判据 | 状态 |
|---|---|---|---|---|
| P0 | 生成 8 种子扩展协议（新目录，不改动冻结 v2 产物） | 0.5 h | `artifacts/joint_stream_learning/<ts>_seed_extension_8/protocol/` | 本轮执行 |
| P1 | 8 种子 × {SAGH, MAPPO} 全长训练 | ~8 h（2 agent × 8 seed × 约 30 min） | 新 `results.jsonl`；seeds 7/17/27 须逐位复现旧值 | 待执行 |
| P2 | 正确统计：窗口外层层级 bootstrap、配对符号检验、per-seed 报告 | 2 h | 替换 Table II 口径的分析脚本 + 区间 | 待执行 |
| P3 | 依判决补 {DAG, PPO} 至 8 种子，或转叙事 | ~8 h 或 0 | 完整 8 种子对照表 | 取决于 P1 |
| P4 | 非重叠窗口评估（父训练划分之外） | 1–2 天 | 独立窗口 results；TMC 软肋、CSCWD 加分 | 待评估数据可得性 |
| P5 | 容量敏感性（进入出现 eviction 的区间） | 0.5 天 | 容量–性能曲线；当前 280/360 MB 两档结果相同，等于容量约束未被测到 | 可选 |
| P6 | 论文改写：Table II 加区间、Fig. 3 加 seed 带、§V-D 改写、摘要与结论同步 | 3–5 天 | 新版 DOCX + 台账轮次 | 待 P1/P2 |
| P7 | 投稿终检：v4 候选合并、版式、作者与基金占位、六页篇幅 | 2–3 天 | 终稿 | 待 P6 |

**不补**（明确排除，避免过度工程）：10 个以上种子、多环境泛化、复现 5 个公开 SOTA、复杂度定理。这些是为冲 NeurIPS/TMC 的量级，与 CSCWD 目标不匹配。

---

## 六、P1 判决树

跑完 8 种子后，按「SAGH 相对 MAPPO 的 seed 级 Δreward 符号」判定：

- **≥6/8 正向**：可写「small but consistent advantage across 8 seeds」。Table II 报 per-seed 均值 + 符号检验 p 值；正文强调一致性而非幅度。
- **4–5/8 正向**：写「outperforms MAPPO on average with high seed variance」，把种子间方差作为发现如实报告，并提供 Fig. 3 的 seed 带。仍是完整论文，但不用 "stable" 一词。
- **≤3/8 正向**：停止追优势叙事。转 `docs/project/system_mechanism_manuscript_working_draft.md` 那条线的写法：机制 + 实现一致性 + 什么条件下准备有益。这不是失败，是换一个更稳的 claim。

任何分支都不得：只报有利 seed、改用挑选后的 checkpoint、或把内部变体写成公开方法。

---

## 七、对既有结论的影响

- 不改变「SAGH 在固定面板上均值最高」这一事实陈述。
- 削弱的是「该优势稳定」这一隐含主张。论文当前未明写 "stable"，但 Table II 无区间的呈现方式容易被读成稳定。
- 不改变任何 typed model cache（G14）系列结论、formal/holdout 状态或 canonical 状态。

---

## 八、复算方法

环境：`/Users/howen/Projects/PPO_MEC/.venv/bin/python`（torch 2.8.0，MPS 可用）。
训练代码位于 `codex/eight-vehicle-simulation` 分支工作树 `~/.codex/worktrees/eight-vehicle-simulation/PPO_MEC`（commit `4bc39b9`，落后 main 89 个提交，尚未合并）。

面板均值与按 seed 拆解：直接读 `results.jsonl`，按 `agent/seed/case_id` 聚合 `reward` 与 `completed_requests/request_count`。
配对区间：对 `(seed, case_id)` 配对差做 percentile bootstrap（20,000 次，seed 0）。**注意该区间不满足窗口独立性要求，仅作对比用途。**
确定性检查：同一 smoke 协议跑两次，比较 checkpoint SHA-256 与逐行 reward。

---

## 九、未验证事项

1. 8 种子全长运行能否在 MPS 上连续 8 小时无中断（smoke 通过不等于长跑通过）。
2. seeds 7/17/27 在 8 种子序列中是否逐位复现旧值（设计上应当复现，未实测）。
3. 非重叠窗口的数据是否在现有 NGSIM/Alibaba 源中可得（P4 前提）。
4. 六个内部对照与公开方法的实际差距（论文未复现，未验证）。
5. 本文件对 CSCWD 录用概率的判断为主观估计，非统计推断。

---

## 十、审查元数据

- `reviewed_at`: 2026-11-09 17:20 Asia/Shanghai
- `literature_cutoff`: 2026-11-09（本轮检索 arXiv:2108.13264、arXiv:1806.08295 摘要页）
- `target_venue`: CSCWD 2027（内部沿用 `tmc_review_policy_v3_20260621` 的 claim 边界）
- `artifact_run_id`: `joint_stream_learning/20261016T162100Z_confirmation`；`20261015T181642Z_development`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `6b0bf3f354cacfdf0402f3359339dae9e25d14c2`（main）
- `evidence_level`: `E2_ARTIFACT_AUDITED`（原始 `results.jsonl` 逐行重算并与论文 Table II 交叉一致；未重跑整包完整性审计，未达 E3）
- 本轮未修改算法、配置、checkpoint 或论文正文；未启动正式训练。
