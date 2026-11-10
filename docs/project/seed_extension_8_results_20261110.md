# 8 种子扩展实验结果（2026-11-10）

用途：记录 `docs/project/cscwd2027_readiness_gap_and_fill_order_20261109.md` 中 P1/P2 的执行结果。
供后续 agent 复读。**本文件报告的是内部对照间的开发面板结果，不是独立窗口泛化证据，也不构成 paper-ready 判定。**

## 一、设计与预注册

- 目标：把 SAGH 与 MAPPO 的训练种子从 3 个扩展到 8 个，检验「优势是否跨种子稳定」。
- 新协议：`runs/seed_ext8_protocol/`（工作区），生成于 `2026-11-09T09:23:23Z`，训练于 `09:23:43Z` 启动。
  **种子列表在看到任何新结果之前已冻结**，属于预注册式扩展，不是事后挑选。
- 种子：`[7, 17, 27, 37, 47, 57, 67, 77]`，新增 5 个与既有分析无关。
- 与冻结 v2 的一致性（逐项核验）：

| 项目 | 结果 |
|---|---|
| `protocol.pt`（实为 sessions+cases+predictor 打包） | **SHA-256 与 v2 相同**（`7099d199…`） |
| `predictor.pt` | **相同**（`05477939…`） |
| seeds 7/17/27 的 `training_schedule` | **逐项相同** |
| train / dev case 数 | 72 / 48，相同 |
| 48 个 dev case 的内容 | 逐条哈希相同 |
| 唯一差异 | manifest 的 `seeds` 与派生计数 |

因此新旧结果之间只差种子，面板未变。旧协议的前三个种子在序列中位置不变，使本次运行同时充当全长复现自检。

## 二、全长复现自检

seed 7 的 48 个 dev case 与 `20261016T162100Z_confirmation` 逐条对比：**48/48 完全一致（差 < 1e-9）**，面板均值 219.4164 / 91.6667% 完全相同。

结论：该 pipeline 在固定协议 + 固定种子下可复现，run-to-run 噪声约为 0。这与 2026-11-09 的 4-episode smoke 结论一致，且已提升到 64-episode 全长。

## 三、结果（8 种子，每 agent 384 个 dev cell）

面板均值：

| Controller | reward | 完成率 |
|---|---:|---:|
| SAGH (`sa_ghmappo`) | **209.458** | **88.36%** |
| MAPPO | 198.700 | 85.68% |

逐种子对比：

| seed | SAGH | MAPPO | Δreward | Δ完成率 |
|---|---:|---:|---:|---:|
| 7 | 219.4 | 211.7 | +7.67 | +2.47pp |
| 17 | 197.6 | 204.1 | **−6.51** | **−1.95pp** |
| 27 | 214.2 | 195.5 | +18.71 | +5.60pp |
| 37 | 212.0 | 193.4 | +18.64 | +5.34pp |
| 47 | 207.7 | 186.2 | +21.46 | +4.95pp |
| 57 | 206.6 | 202.1 | +4.52 | +0.26pp |
| 67 | 208.1 | 204.5 | +3.64 | +1.69pp |
| 77 | 210.0 | 192.1 | +17.92 | +3.12pp |

## 四、统计（seed 为外层单位）

| 指标 | 结果 |
|---|---|
| 正向种子 | **7 / 8**（唯一反向为 seed 17） |
| 平均 Δreward | **+10.76** |
| seed 级 bootstrap 95% CI（reward） | **[+4.07, +17.00]**，不含 0 |
| 平均 Δ完成率 | **+2.69pp** |
| seed 级 bootstrap 95% CI（完成率） | **[+0.93, +4.35] pp**，不含 0 |
| 精确配对符号检验（双侧） | **p = 0.0703** |
| 窗口一致性 | 窗口 3 +7.91、窗口 4 +13.61，两窗口同向 |

说明：n = 8 时符号检验最多只能达到 p = 0.0078（需 8/8 全正）；7/8 对应 p ≈ 0.07，属边际。seed 级区间不含 0 是更强的证据，符号检验因只用方向、丢弃幅度而偏保守。**两项应同时报告，不得只引用区间或只引用 p。**

## 五、判定

按 `cscwd2027_readiness_gap_and_fill_order_20261109.md` 第六节判决树，落于 **≥6/8 正向**分支：可写「小幅但跨种子一致的优势」。

与 3 种子版本的对比：

| | 3 种子 | 8 种子 |
|---|---|---|
| 平均 Δreward | +6.63 | **+10.76** |
| seed 级 95% CI | [−6.51, +18.71]（含 0） | **[+4.07, +17.00]（不含 0）** |
| 正向种子 | 2/3 | **7/8** |
| 结论 | 无证据 | **方向一致** |

新增的 5 个种子全部正向，且使均值上升——不是靠原有 3 个种子撑起来的。

## 六、建议的论文措辞（待作者确认）

> SAGH exceeds the MAPPO internal control on 7 of 8 training seeds, with a mean improvement of 10.8 reward points and 2.7 percentage points of request completion (seed-level bootstrap 95% CI [4.1, 17.0] and [0.9, 4.3]; exact paired sign test p = 0.07).

配套要求：

1. Table II 的均值需重算为 8 种子口径，或保留 3 种子主表并单列 seed 稳健性图表（二选一，不可混用口径）。
2. Fig. 3 应叠加 8 个种子的分布或种子带，不能只画均值曲线。
3. 不得写 "stable convergence"：seed 17 在训练中从 206.20 降至 197.58，退化是真实存在的。
4. 不得写 "all baselines"：本次只扩了两个控制器。

## 七、局限与未验证

1. 只有 2 个时间窗口，窗口层独立单元 n = 2，不构成泛化证据；层级 bootstrap 在 n = 2 时退化，故未作为检验使用。
2. 对照为内部变体，不是公开方法的忠实复现。
3. DAG 与 PPO 的 8 种子扩展在写作时仍在运行（见第八节）；其余四个消融变体仍为 3 种子。若论文主表要统一口径，需全部扩展或明确分表。
4. 未做非重叠窗口评估（原 P4），未做容量敏感性（原 P5）。
5. 本结果不改变 typed model cache（G14）系列的 formal/holdout/canonical 状态。

## 八、产物与复算

| 内容 | 位置 |
|---|---|
| 8 种子协议 | `runs/seed_ext8_protocol/`（manifest SHA-256 `7c2e527f1c21d55f497584ef2570acf14cce85ec9b12f1aedb450654f389fdf0`） |
| SAGH 8 种子输出 | `runs/seed_ext8/sa_ghmappo/`（`dev.jsonl` 480 行 = 8 seed × (12 probe + 48 full)） |
| MAPPO 8 种子输出 | `runs/seed_ext8/mappo/` |
| 统计结果 JSON | `runs/seed_ext8_analysis.json` |
| 生成器 | `tools/prepare_seed_extension.py` |
| 统计脚本 | `tools/analyze_seed_extension.py` |
| 运行脚本 | `tools/run_seed_ext8.sh` |

注：以上均位于本次会话工作区 `/Users/howen/Documents/deepseek-harness/default-workspace/`，尚未迁入仓库 `artifacts/`（`artifacts/` 被 gitignore，迁移需显式执行）。

复算：`python tools/analyze_seed_extension.py --run-root runs/seed_ext8 --protocol runs/seed_ext8_protocol/protocol.pt`，在 `codex/eight-vehicle-simulation` 工作树根目录执行。

## 九、审查元数据

- `reviewed_at`: 2026-11-10 08:40 Asia/Shanghai
- `target_venue`: CSCWD 2027
- `artifact_run_id`: `runs/seed_ext8`（协议 `runs/seed_ext8_protocol`）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `66ca0ad`（文档基线）；实验代码 `4bc39b9`（`codex/eight-vehicle-simulation`，未合并 main）
- `evidence_level`: `E2_ARTIFACT_AUDITED`（原始 dev 行重算 + 与冻结 v2 的资产哈希逐项核对 + 全长 48/48 复现；未做整包 audit 脚本的完整重跑，未达 E3）
- 本轮未修改算法、配置、checkpoint 或论文正文。
