# CSCWD 2027 投稿清单（2026-11-13）

用途：记录从官方站点核实的投稿要求、当前稿件状态与投稿前待办。供后续 agent 与作者复读。

状态更新：2026-11-18。复核了官网和六页阅读候选，官网仍未列明截止的具体时刻和时区。版本与证据见[来源映射](manuscript_source_map.md)，作者决定集中在[需求清单](requirements_backlog.md)。[CSCWD 2027 官网](https://cscwd2027.dailyeliteevents.com.au/)

## 一、官方信息（已从官网核实）

- 会议：30th International Conference on Computer Supported Cooperative Work in Design (CSCWD 2027)，May 26–28 2027，Brisbane, Australia（Hybrid）。
- 官网：`http://2027.cscwd.org`（跳转 `https://cscwd2027.dailyeliteevents.com.au`）。
- 投稿系统：EasyChair `https://easychair.org/conferences/?conf=cscwd2027`。
- **截稿：2026-11-30**（页面标注 "October 31, 2026 → November 30, 2026，Paper submission due (extended)"，即原 10-31 延期至 11-30）。
- 通知：2027-01-31；camera-ready 与注册：2027-02-28。
- **页数：6 页（IEEE proceedings 格式，含参考文献与插图）**。
- 提交格式：PDF（"Electronic submissions in PDF format are strongly recommended"）。
- 出版：IEEE categorized conference proceedings，提交 EI Compendex、DBLP 等检索。

## 二、当前稿件状态

- 正式底稿仍为 R22 标注稿；最新候选为 `artifacts/manuscript/fixed_names_review_20261115/` 中完整题名加 `_marked_candidate.docx`，SHA-256 `f975971cf322bd46f694a908f7747f7ff9f96bd277b97472cf9a2a3c875449b2`，待确认、不赋正式 R23。
- 已合并老师返修、v4、八种子实验与四处叙事优化，并恢复固定小节名称。旧 `r23_20261111` 只是历史候选，目录名不构成正式版本。
- 既有验收记录：阅读 6 页、显示修订 7 页、65 条批注；11-15 已完成逐页目检。11-18 重核候选 hash，未重渲染或逐项点击桌面 Word。
- 11-18 复算了固定面板、种子 bootstrap、两种 Holm 检验族及原始时间区间。原件哈希、checkpoint 张量和 SAGH/缓存对照帧账本已核对，并复算三种子及八种子对象生命周期。稿件仍报告原有三种子机制数据。独立正式验证仍缺。详细结果仅在本地 `artifacts/analysis/cscwd2027_submission_review_20261118_v1/`。

- 新增 I-80 两时间窗、12 个新 DAG 的双模型族复验已审计，适合在结果段追加模型流量补充证据，同时限定完成率及窗口独立性。拟写文字和反例见本地 `artifacts/analysis/cscwd2027_dual_family_transfer_20261118_v1/review.md`。当前候选尚未采用这些结果。

## 三、投稿前必须由作者完成（阻塞项）

1. 审阅并明确确认最新整合标注候选，或定位需修改的具体位置；确认前仍以 R22 为正式底稿。
2. 提供第三/四作者、邮箱、基金及源码公开说明所需真实信息。

以下编辑已完成于候选，不再单独重复索取授权：Model KiB 列与旧 init/32/64 轨迹删除、v4 合并、历史三种子对象收据来源单列。候选沿用未校正 p 的明确标注，不直接宣称 PPO/DQN 显著；七对照 Holm 最小值为 0.0546875。确认候选不代替独立证据补验。

## 四、投稿前技术检查

1. 格式核对：当前 docx 为 Letter 双栏、题名 24pt / 摘要 9pt / 正文 10pt / 参考文献 8pt Times New Roman，与 IEEE 会议样式一致；投稿前建议用 IEEE PDF eXpress（或等效）校验，或按 EasyChair 要求确认。
2. 匿名化：官网未明示 double-blind；若 EasyChair 系统标注 double-blind，需删除作者栏、致谢与基金信息后再投稿。
3. PDF 导出：现有阅读版核验为 6 页 Letter。定稿后移除第二页残留的黄色标记并核对 PDF 作者及主题元数据，再做逐页检查。
4. EasyChair 提交：标题、摘要、关键词、作者信息、PDF 上传；通讯作者邮箱与机构需与论文一致。

## 五、关键风险

- 阅读候选恰为 6 页；作者信息或正文新增后须重新渲染核对，不能沿用旧页数验收。
- 具体截止时刻/时区需核定；不依据附件换算自行延后提交。作者信息与候选确认仍待处理，v4 合并已完成。

## 六、已落地的产物索引

- 8 种子实验：`runs/seed_ext8/`（8 控制器 × 8 种子 dev/training/events/receipts）、`runs/seed_ext8_protocol/`。
- 统计：`runs/seed_ext8_full_panel.json`、`runs/seed_ext8_analysis.json`、`runs/seed_ext8_fig3_curves.json`。
- 文档：`docs/project/cscwd2027_readiness_gap_and_fill_order_20261109.md`、`docs/project/seed_extension_8_results_20261110.md`、本清单。
- 2026-11-18 本地部分快照：`artifacts/analysis/manuscript_support_20261118_v1/`，含路径/hash 清单与复算验证；原大包仍在外部工作区，排除的事件/收据/checkpoint 未归档。
