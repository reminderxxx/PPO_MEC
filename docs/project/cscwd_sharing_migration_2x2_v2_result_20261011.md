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

## 只读归因与同奖励定义复算

从 32 条 `step_*.json` 的原始 `service_completed`、`terminated`、`step_cost_seconds`、四类传输字节、`clock_seconds_after` 和 deadline 独立按冻结 `original_reward_v1` 复算；权重来自 workload v3 所继承的 `configs/experiment/calibrated_continuous_workflow_pilot_v2.json`：每节点 +2、workflow 终止 +4、每 step cost 秒 −0.08、每 GiB 传输 −0.5、失败 −3、终止迟到 −4。逐步六分项、逐步 reward、四臂累加 reward 与 arm receipt 全部在浮点容差 `1e-9` 内一致；四臂传输字节也逐步加总对上终点。初始 clock 为 0.1 秒，故累计 step cost +0.1 秒等于报告的完成时刻。其他奖励 profile 的四臂对照未在此次日志中运行，记 `UNAVAILABLE`。

| arm | 节点+workflow | 时间罚 | 传输罚 | 迟到罚 | 失败罚 | 同定义总 reward |
|---|---:|---:|---:|---:|---:|---:|
| S0M0 | +20 | −8.303469 | −0.478382 | −4 | 0 | 7.218149 |
| S1M0 | +20 | −7.367444 | −0.010214 | 0 | 0 | 12.622342 |
| S0M1 | +20 | −8.305512 | −0.478448 | −4 | 0 | 7.216040 |
| S1M1 | +20 | −7.026265 | −0.010345 | 0 | 0 | 12.963390 |

共享的条件差：M0 时 S1−S0 reward +5.404193（时间 +0.936025、传输 +0.468168、免迟到罚 +4）；M1 时 +5.747350（时间 +1.279247、传输 +0.468103、免迟到罚 +4）。迁移的条件差：S0 时 M1−M0 reward −0.002108（无成功交付，仍多 140,480 bytes）；S1 时 +0.341049（一次交付后免重算，但多 280,960 bytes）。reward 的描述性 difference-in-differences 为 +0.343157；它包含后续状态/动作机会改变，且只有一个已暴露轨迹，**无显著性解释**。S0 与 S1 的固定规则动作序列分别是 `[4,2,2,2,2,2,2,2]` 与 `[4,4,2,2,2,2,2,2]`；各 S 内 M0/M1 的动作编号相同，但因状态和资源改变，不能当作跨四臂纯同状态单动作实验。原 profile 对重算没有单列罚项，重算耗时计入时间罚。以上固定策略 reward 差不能称为 SA 算法优势。

## 给 B 的来源资格元数据交接

仅使用既有来源 manifest、split 元数据与公开开发审查记录，不读取 sealed 性能、不另选训练样本。原始 NGSIM 13 位 `Global_Time` 毫秒和相邻 100 ms / `Frame_ID +1` 的身份已在两段长来源核过；两段各 1,189 帧、118.8 秒，和 strict formal/hidden 原始闭区间重叠 0，但均有历史开发暴露：

| 本地未封存长来源 | 原始区间 / 车辆 | 历史 train/dev 交叠 | 已知后续使用与机制机会 |
|---|---|---:|---|
| Lankershim | 1118935795100–1118935913900，车辆 215，Frame 1152–2340 | 1 | B 既有 pilot 用于训练，后来又用于开发检查/评估；四臂预检首步公开 `action_mask[4]=false`。不能从本轮推断整段 action 4 覆盖率。 |
| US101 | 1118849293800–1118849412600，车辆 1183，Frame 5368–6556 | 2 | B 既有 pilot 用于 checkpoint 选择，后来又用于开发检查/评估；四臂预检首步公开 `action_mask[4]=true`，已有多轮机制探针暴露。 |

既有 v28 strict 的 20 train + 20 dev 原始短区间均仅 24 帧/2.3 秒，对该完整 DAG 资格为 0/40；不能拼接或当成长来源替代。Peachtree 的 10 位绝对时间单位、与其它来源的 clock/历史 split 身份未核；I-80 sealed performance 不读取。**当前没有可交给 B 当作独立确认样本的合格来源。**B 应单独冻结结果盲 coverage gate 与 split；本报告不替 B 选择最终训练样本，也不为制造 prepare 机会改 RSU 布局或成本参数。

最小外部资源需求分两层：若 B 的既有非封存来源经结果盲门仍缺合法 prepare 覆盖，需要另获授权提供原始时间/位置/帧身份可核、与旧 train/dev/formal/hidden 区间可审计的新 development 轨迹；目前不能把 Peachtree 直接投入。要支持真实双 RSU 净收益主张，还需独立站点/链路/队列测量；要支持优秀基线与三头主张，还需匹配训练预算、checkpoint 和独立 formal/holdout/support 包。此处只列资源缺口，不启动采集、下载或科学运行。

## 四项主张与下一道证据门

| 待固定贡献 | 当前证据 | 仍缺的最短证据 |
|---|---|---|
| 真实 RSU 连续执行净收益 | 本轮只验证一条原始车辆轨迹驱动的模拟相位/字节守恒，并保留失败交付 | 独立 RSU/链路测量或明确限定为仿真；匹配的系统成本、队列与失败对照 |
| sharing × migration 协同 | 局部 opt-in S0/M0 四臂可运行，单轨迹描述性交互可算 | 正式消费者、跨 workflow cache 生命周期、多独立原始区间的预冻结四臂及统计 |
| 三头控制优于优秀公平基线 | 本轮无 learned policy 或训练；固定公开策略只是机制探针 | 冻结相同信息权限/预算/物理合同的三头结构消融与强基线训练、checkpoint 和比较包 |
| 独立泛化与论文就绪 | US101 来源历史暴露、唯一外层 cluster，四臂/多条件不能增大独立样本数 | 区间互斥的多来源 formal/holdout/support、窗口层统计与完整 provenance |

Peachtree 本轮仅只读本地元数据：原始 `Location=peachtree` 873,887 行、10,442 个 distinct `Global_Time`，全部 10 位且相邻唯一时间差为 100；23 个版本化 split/window-plan 文件有 77 条 interval 记录、56 个去重 Peachtree 区间（train 12、dev 16、formal 13、hidden 12、rehearsal/other 3）。这不是时间单位或跨来源 clock 语义验证，也不证明新区间独立；不得据此另启科学运行。I-80 sealed performance 未读。

**安全结论**：在这一个已暴露开发例上，base 共享大幅改变模型字节成本；状态迁移既出现一次被后续利用的交付，也出现目标提前离开造成的额外字节。**禁止结论**：独立因果交互、真实双 RSU 净收益、三头算法优势、优秀基线排序、可泛化显著性、canonical 晋级或 paper-ready。
