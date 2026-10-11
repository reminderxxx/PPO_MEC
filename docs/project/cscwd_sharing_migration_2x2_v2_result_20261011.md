# CSCWD sharing × migration v2 四臂开发结果与论文证据边界

- `reviewed_at`: 2026-10-11 Asia/Shanghai；`literature_cutoff`: 2026-10-11（本轮未检索新文献）；`target_venue`: IEEE Transactions on Mobile Computing (TMC)。
- `artifact_run_id`: `cscwd_sharing_migration_2x2_development_20261011_v2`；`policy_version`: `tmc_review_policy_v3_20260621`；`git_commit`（执行时）: `dd7bb2afc33bdd7b96a88a1e19b546a2672804e0`。
- `evidence_level`: 局部开发 artifact 的 42/42 文件完整性与逐步账本已核；按 TMC 正式论文证据定义仍为 `E1_DOCUMENTED` / `Unverifiable`，因为正式 checkpoint、formal/holdout/support 和独立统计包不齐。七维评分均 `N/S`。
- `verdict`: **一次有界四臂运行 `COMPLETE`；只能报告已暴露 US101 开发轨迹上的描述性结果，四项论文贡献仍 `UNVERIFIED`。**

## 冻结身份与完整性

零步 preflight receipt SHA-256 `6e296df7dedb682274180a36f1ff960e54bd02d67466b0cb7a6964a8a8842084`；旧 v1 失败 receipt SHA-256 `939bb277c8387c6060849286be01f84eaf9fa65bc6fd427fb723325141c66315`，两者均保持原件。唯一 v2 运行使用原始 NGSIM US101 车辆 1183、`Global_Time` 1118849293800–1118849412600 / `Frame_ID` 5368–6556、`dev_01`、模拟 1000 Mbps backhaul、固定公开 `public_prepare_if_legal_else_serve_v1`。该原始区间与 protected formal/hidden 重叠 0，但历史 train/dev 暴露 2 次；外层独立来源数仅 1。四臂同初始 action mask、同 DAG 节点与输入，未训练、无模型调用、未开启 formal/holdout。

`artifacts/analysis/cscwd_sharing_migration_2x2_development_20261011_v2/` 中 `run_manifest.json` SHA-256 `b4798eabf242aa124b2d71905b160150e204243982f368441d39e5d8d4397c80`，`summary.json` SHA-256 `de9c1375c2767d0571ba6f1a4a658969da9d816ce1339faf9055f6d472439aa7`，`artifact_integrity.json` SHA-256 `ab892cf6f76622723c1f61ed46052959bfa28df2da243f51a3a5eb2da49cbea0`，`completion_receipt.json` SHA-256 `7e07c86ce8c60acac21b639f3bd10293cf3a5ec9e55b41b0403dcda879881d4c`。独立重算 42/42 manifest 文件 SHA、4/4 arm receipt、32/32 逐步记录与 episode 终点对账通过；总 32/96 真实环境步，各臂 8 步自然终止、8/8 节点完成、0 截断、0 service failure。

## 四臂原始终点

S0/S1 表示独立 base / 共享 base；M0/M1 表示关闭 / 开启状态迁移。时间为模拟合同内的 workflow 完成时刻，字节含模型、状态及输入；四臂输入均 2,650,408 bytes。

| arm | 完成/按期 | 总传输 bytes | 模型 bytes | 状态 bytes | 完成秒 | 重算秒 | 迁移交付 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0M0 | 1/0 | 1,027,318,184 | 1,024,667,776 | 0 | 103.893362 | 0 | 0 |
| S1M0 | 1/1 | 21,934,296 | 19,283,888 | 0 | 92.193052 | 4.315814 | 0 |
| S0M1 | 1/0 | 1,027,458,664 | 1,024,667,776 | 140,480 | 103.918897 | 0 | 0 |
| S1M1 | 1/1 | 22,215,256 | 19,283,888 | 280,960 | 87.928309 | 0 | 1 |

全部完成 8 个节点，deadline 99.164509 秒。按同一轨迹作描述性 2×2 计算：共享轴平均少传 1,005,313,648 bytes，迁移轴平均多传 210,720 bytes；总传输 difference-in-differences 为 +140,480 bytes。模拟完成时刻的 difference-in-differences 为 −4.290278 秒；完成率、按期率的交互均为 0。S1M1 相比 S1M0 少 4.264743 秒且多传 280,960 bytes；S0M1 相比 S0M0 多 0.025536 秒及 140,480 bytes。以上是固定策略和模拟物理下的**总系统响应**，没有外层独立样本、置信区间或迁移净收益的泛化解释。

逐步账本显示，S1M1 第一次状态成功交付且后续消费，避免 S1M0 的 4.315814 秒重算；第二次发送 140,480 bytes 后目标先离开。S0M1 唯一一次状态发送同样因目标先离开而未交付。S0 的首次 1,024,667,776-byte 私有模型装入拉长模拟时间，后续只出现 vehicle fallback；S1 各有两次 action 4，后续再 fallback。动作机会本身随因素改变，因此不能把上表的 difference-in-differences 解释为隔离了后续动作路径的直接物理交互。单实例、零共享队列、模拟 RSU/backhaul、无真实模型调用，也不能表述为实测双站部署收益。

## 四项主张与下一道证据门

| 待固定贡献 | 当前证据 | 仍缺的最短证据 |
|---|---|---|
| 真实 RSU 连续执行净收益 | 本轮只验证一条原始车辆轨迹驱动的模拟相位/字节守恒，并保留失败交付 | 独立 RSU/链路测量或明确限定为仿真；匹配的系统成本、队列与失败对照 |
| sharing × migration 协同 | 局部 opt-in S0/M0 四臂可运行，单轨迹描述性交互可算 | 正式消费者、跨 workflow cache 生命周期、多独立原始区间的预冻结四臂及统计 |
| 三头控制优于优秀公平基线 | 本轮无 learned policy 或训练；固定公开策略只是机制探针 | 冻结相同信息权限/预算/物理合同的三头结构消融与强基线训练、checkpoint 和比较包 |
| 独立泛化与论文就绪 | US101 来源历史暴露、唯一外层 cluster，四臂/多条件不能增大独立样本数 | 区间互斥的多来源 formal/holdout/support、窗口层统计与完整 provenance |

Peachtree 本轮仅只读本地元数据：原始 `Location=peachtree` 873,887 行、10,442 个 distinct `Global_Time`，全部 10 位且相邻唯一时间差为 100；23 个版本化 split/window-plan 文件有 77 条 interval 记录、56 个去重 Peachtree 区间（train 12、dev 16、formal 13、hidden 12、rehearsal/other 3）。这不是时间单位或跨来源 clock 语义验证，也不证明新区间独立；不得据此另启科学运行。I-80 sealed performance 未读。

**安全结论**：在这一个已暴露开发例上，base 共享大幅改变模型字节成本；状态迁移既出现一次被后续利用的交付，也出现目标提前离开造成的额外字节。**禁止结论**：独立因果交互、真实双 RSU 净收益、三头算法优势、优秀基线排序、可泛化显著性、canonical 晋级或 paper-ready。
