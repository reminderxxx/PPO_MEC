# CSCWD I-80 窗口来源与独立性有界审计

## 身份与范围

- `reviewed_at`: 2026-10-09（Asia/Shanghai）
- `literature_cutoff`: 2026-10-09；未检索新文献
- `target_venue`: CSCWD 2027（拟投）
- `artifact_run_id`: `cscwd_2027_i80_source_scope_audit_20261009_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit_at_review`: A 基线 `43f288fa00ea1cb8f81035670476d6a0bc9bf65d`；复用 G14R23 审计 `5173234666ddbbb53df977a2d3a95a04f7166300`；本报告最终提交见 Git
- `evidence_level`: `E2_METADATA_SCOPE_AUDITED / NO_NEW_CONFIRMATORY_RESULT`
- `verdict`: `I80_UNKNOWN_SCOPE_CLEARED / NO_CROSS_RUN_INDEPENDENT_SPLIT / POST_G14R23_FULL_HISTORY_NOT_REAUDITED`

机器汇总：`artifacts/analysis/cscwd_2027_i80_source_scope_audit_20261009_v1/mapping_summary.json`。本轮只复用 G14B 的 registry/provider/inventory 和 G14R23 的全历史元数据审计；A 分支自 2026-09-27 后改动的 experiment window plan/manifest 路径为 0，新增只读交叉 B 的 36-instance manifest。没有重新扫描 1,185 万行车辆源、重复遍历全仓 3,957 个历史结果文件、读取封存性能或生成新 split/seal。

## 来源映射与四类判定

G14B 的 NGSIM 原始 CSV 记录 hash 为 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`，大小 `2,118,175,938 B`；本轮直接使用其已记录身份，不重算大文件 hash。`available_interval_inventory.json` 给出 I-80 三个连续 run 的原始 frame/time 边界，`candidate_window_inventory.json` 给出 579 个 result-blind、互不重叠的 24-frame 候选。四份 G14B split plan 中已选择/分配 60 个 exact ID：train 24、dev 12、formal 12、sealed holdout 12；按 run 为 `001:19 / 002:14 / 003:27`。本报告只保存汇总，不复制 sealed 窗口 ID。

| 分类 | 数量和来源证据 | 解释 |
| --- | --- | --- |
| **已消费/分配，不能再作新测试** | I-80 60/579；四份 G14B split plan 与 G14R23 `later_or_uncovered_unique_interval_count=60` 一致 | 包含已分配 sealed 窗口，旧 holdout 不开启或重新命名。 |
| **来源未知，不能反向推成未消费** | G14B 668 个独特历史区间中 418 个 raw interval unknown：224 个只知 Lankershim/Peachtree/US-101 保守范围、181 个 Peachtree、13 个 US-101；G14R23 另隔离 3 个明确非 I-80 的 unresolved 引用 | 原计划/结果缺可置信 segment-run 或原始 frame/time，不能按 window ID/offset 猜测。418 条保守范围均**不含 I-80**；G14R23 后续可能触及 I-80 的 unresolved=0。这是 I-80 scope 排除证据，不是对其他三段恢复成功。 |
| **按既定独立性条件可排除** | 60 个 exact raw interval overlap；余 519 个在 G14R23 审计范围内满足 24-frame 时间隔离，其中 505 个与已用 I-80 观察车辆复现，余 14 个均属 `i_80_run_001`；候选间零车辆复现预览只剩 2 个同 run | 时间不重叠不等于跨车辆、跨 capture session 或跨路段独立。G14R23 的 12-window、至少 3 run、每 run 最多 4 的预先规则无法满足。 |
| **可证明未消费并可进入新确认性 split** | **0 个**符合当前完整证据与相关性门的 I-80 窗口 | 519 是 G14R23 元数据范围内的**时间未占用上限**，不是当前全部历史下已证明 pristine 的 519 个；14/2 也不是外部场景独立样本。 |

三个 I-80 库存 run 映射分别是 `i_80_run_001`：208 候选，raw frame `12–9971`、time `1113433136100–1113434132000`；`run_002`：166，frame `27–7970`、time `1113436769600–1113437563900`；`run_003`：205，frame `1813–11628`、time `1113437746200–1113438727700`。这些是候选的覆盖跨度，中间空隙不能当作可用窗口。G14B provider 连续 run 元数据、原始 source hash、候选中每项的 run/frame/time 字段和 G14R23 60 项后续身份回收相互提供映射证据；本轮没有用 offset 跨段推断。

B 的 `calibrated_continuous_workflow_interface_repair_v3_manifest.json` 36 项分别来自 US-101 28、Lankershim 6、Peachtree 2；每项有明确 `source_segment_id`、`frame_offset/window_length` 和 `time_index_start/end`，与 I-80 候选按原始 segment/run/time 交集为 0。它们自身已全部用于 development，不能转作独立评价。B manifest 只给计划和 sampled DAG 的 source hash，完整 NGSIM CSV hash 仍须由独立 ledger 绑定；不能仅凭不同 window ID 宣称新源身份。

G14R23 已作更广的 metadata-only 审计：3,957 个 plan/result JSON、90,640 个窗口引用，6,995 个配置/checkout 复制引用由原冻结身份消歧；24 个 synthetic/test-only 引用隔离，后续 unresolved=0，另 3 个非 I-80 未知引用隔离。其车辆复现扫描只读 `Vehicle_ID`、`Location`、`Global_Time`，得到 14 个零历史车辆复现窗口，但全部在同一 I-80 run；两窗口预览也仅同 run。上述审计截至 2026-09-27，本轮仅核对已交付的后续 calibrated manifest 和 A 分支 experiment plan/manifest 变化。其他 9 月 27 日之后 artifact 的完整历史没有再跑一次全量扫描，故不给 519 项加盖当前永久未消费证明。

## 具体缺口和最低工作量路线

现有 36 实例仍适合训练接口、奖励/critic 机制诊断和预注册的参数化仿真压力测试；它们不能支持独立算法优势或新道路泛化。可在不看方法效果的前提下先固定 cache capacity、state scale、link estimate、handoff pressure 和失败处理的参数矩阵作合成外推，但须标为 synthetic，不称“未见真实场景”。

按 G14R23 的既定相关性门，当前两个可两两零观察车辆复现的 I-80 预览窗口均在 run_001；若保留它们，至少还需 **10 个**合格窗口来自**至少 3 个其他 segment-runs**（每 run 最多 4），且与所有已用/候选窗口车辆不复现。若要求跨道路或跨 capture session 的场景独立性，应直接准备满足同样元数据与许可证门的新来源，不把旧 I-80 时间切片写成外部泛化。新来源需先记录原始文件 hash、来源/许可与标签、segment/run、frame/time、车辆身份与 workflow 祖先，做全历史消费交叉，再按机制机会盲选 train/dev/evaluation；旧 sealed holdout 永久不再使用。

下一次元数据审计只需补齐 G14R23 之后新增 artifact 的窗口引用和 source identity，优先检查任何可能指向 I-80 的记录；不必重扫已有 3,957 文件或重算 2.1 GB 原始源 hash。若仍只能得到同来源、单 run 的时间外推，结论限定为该相关性范围，不硬凑 12 个“独立”窗口。正式论文仍缺跨 run/车辆独立测试、足够机制机会、DT/Popularity 公平接线、模型与数据许可边界，以及新强基线的原始结果。
