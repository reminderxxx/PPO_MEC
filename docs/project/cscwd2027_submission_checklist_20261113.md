# CSCWD 2027 投稿清单（2026-11-13）

用途：记录从官方站点核实的投稿要求、当前稿件状态与投稿前待办。供后续 agent 与作者复读。

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

- 候选稿：`artifacts/manuscript/r23_20261111/Multi-agent SAGH reinforcement learning for Workflow-Aware Adapter Preparation in Vehicular Edge Computing_marked_candidate.docx`（SHA-256 `f483e6913c85628379b92da997c5449f4f27d318ea2dc6b00773cdb0dcd730da`）。
- 6 页；Table II 为 8 行×3 列；Fig.3 为 8 种子带图；35 条 Word 批注。
- 数字审计：所有 reward/completion/CI/符号检验数值均已从 `runs/seed_ext8/*/dev.jsonl` 独立重算并逐字比对通过。
- 内容与结构检查通过；**视觉检查未完成**（当前模型不支持读图）。

## 三、投稿前必须由作者完成（阻塞项）

1. 目检 6 页版式（渲染 PDF 已生成：`runs/r23_candidate_v2.pdf` 与 `runs/r23_pages/` 六页 PNG）。
2. 确认 claim 措辞（摘要/结论：显著优于 PPO 8/8、一致优于 MAPPO 7/8、对 cache 无显著优势）。
3. 确认两个删减：删除 Table II 的 Model KiB 列；删除 §V-B/C 的 init/32/64 轨迹句。
4. 作者与基金信息：`Third Aughor`、第四作者邮箱、基金号（当前均为占位）。
5. v4 候选是否合并（Related Work 三线分层 + `IV. SAGH-Based Joint Scheduling Algorithm` 标题）；不合并则以 R22 结构为准。

## 四、投稿前技术检查

1. 格式核对：当前 docx 为 Letter 双栏、题名 24pt / 摘要 9pt / 正文 10pt / 参考文献 8pt Times New Roman，与 IEEE 会议样式一致；投稿前建议用 IEEE PDF eXpress（或等效）校验，或按 EasyChair 要求确认。
2. 匿名化：官网未明示 double-blind；若 EasyChair 系统标注 double-blind，需删除作者栏、致谢与基金信息后再投稿。
3. PDF 导出：已用 LibreOffice Kit 验证可导出 6 页 PDF；正式投稿建议用 Word 导出或按 IEEE 校验工具生成最终 PDF。
4. EasyChair 提交：标题、摘要、关键词、作者信息、PDF 上传；通讯作者邮箱与机构需与论文一致。

## 五、关键风险

- 6 页上限：R23 恰为 6 页，无余量，任何新增内容都会超页。
- 截稿 11-30（约 17 天），时间紧；作者信息与 v4 合并需尽快定。

## 六、已落地的产物索引

- 8 种子实验：`runs/seed_ext8/`（8 控制器 × 8 种子 dev/training/events/receipts）、`runs/seed_ext8_protocol/`。
- 统计：`runs/seed_ext8_full_panel.json`、`runs/seed_ext8_analysis.json`、`runs/seed_ext8_fig3_curves.json`。
- 文档：`docs/project/cscwd2027_readiness_gap_and_fill_order_20261109.md`、`docs/project/seed_extension_8_results_20261110.md`、本清单。
