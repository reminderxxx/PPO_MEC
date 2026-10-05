# 论文成果与问题台账

更新：2026-10-05。主稿：`system_mechanism_manuscript_working_draft.md` v1.2。
本次新增一次事前冻结的 12-call 真实 adapter lifecycle witness；不训练 RL、不下载资源、不读取旧 holdout，也不改写历史 artifact。

## 已完成成果及可用范围

| ID | 已解决的问题 | 可引用成果 | 原件入口 | 限制 |
|---|---|---|---|---|
| E01 | 后缀恢复是否保持执行内容 | 6/6 新配对输入/token/identity 一致，24 次调用 | `independent_recovery_cost_measurement_results_20261005.md` 与对应 artifact root | 单主机、单技术输入；不是交通任务正确性 |
| E02 | 旧成本能否预测新执行方向 | 6/6 选择较低成本 recovery | 同上 `aggregate_summary.json` | 全在 recovery 一侧，不能证明边界分类性能 |
| E03 | 绝对成本估计是否准确 | 中位绝对误差 0.369–1.944 s；保留高估 | 同上逐次测量表 | prepared recovery 相对误差 45.9%–49.7% |
| E04 | 后续驱逐是否影响决策 | 合成原生状态见证；12 点中修正规则匹配参考 10 点 | `eviction_aware_recovery_results_20261005.md` 与对应 artifact root | 开发点，非真实模型 victim→reload |
| E05 | 新规则是否胜过正确两步 | 12/12 持平，未发现性能/开销优势 | 同上 | 不支持原创通用算法或 RL 必要性 |
| E06 | 是否全面改善所有指标 | 汇总时间/传输下降，但多 5 次重跑、53.641216 s 重算 | 同上 | 不可只保留正向指标 |
| E07 | 合法 victim 是否导致真实后续重载 | native 合法驱逐 ALPR 后，PEFT 780 tensors→0；后续从本地文件重载为 780 tensors 并执行 n2；两臂均通过 | `real_cache_victim_reload_results_20261005.md` 与 `artifacts/analysis/real_cache_victim_reload_20261005_v1/` | 单主机、单实例、OS cache 未清、网络模拟 |
| E08 | 无驱逐时是否确实无外部代价 | base+双 adapter 对照的当前/未来请求均 `noop_all_resident`，两臂 action load/unload=0 | 同上 | 仅一个对照，不做统计外推 |
| E09 | 驱逐是否改变 restart/recovery 相对选择 | 两臂 lifecycle 为 1.022245/1.026997 s，因节点依赖相同而共同计费；两条件仍都选 recovery | 同上 | 未跨决策边界，不支持 action reversal |

## 正在解决和未解决的问题

- P01，RESOLVED FOR THIS INSTANCE / HISTORICAL DEFECT FROZEN：已逐路径列出模型准备、输入/状态、前缀/后缀和后续重载；旧公式的 recovery-only `D0` 口径登记为 frozen scientific defect，不重算旧结果。
- P02，RESOLVED BOUNDED：合法 victim→实际 PEFT object/tensor 卸载→后续本地文件重载→n2 执行已闭环；logical resident、runtime 对象、磁盘文件和未控制 OS cache 四层分开。
- P03，OPEN：独立条件是否覆盖恢复/重跑两侧；不得无意义填充状态或 sleep 制造翻转。
- P04，OPEN：远端成本、任务标签及外部代表性。暂不建设共享队列或跨 workflow 平台；是否需要由最终主张决定。
- P05，OPEN：系统组合的新颖性及投稿定位；引用已有成本感知思想，不用润色填补增量价值缺口。

## 本轮收口结论

最小真实生命周期已经得到验证，无需继续同侧重复。若论文只主张 lifecycle 与成本可追溯，P02 已闭环；若主张 eviction 改变
restart/recovery 决策，则必须先证明两条合法路径为何产生不同 resident 转移。不得靠旧公式的不对称计费制造翻转，也不据此扩建
多租户平台、启动 RL 或搜集新模型。

## 后续追加格式

每次完成后在此追加：日期、问题 ID、实际改动、执行 commit、原始 artifact 路径、输入/输出身份、正负结果、已解决/未解决项及论文对应章节。先保存原件，再改总结；不覆盖失败或历史版本。此台账不等同后台监控，也不自动授权任何执行。

## v1.1 编辑维护记录（历史）

- delivery baseline：`510bc43bb4e67cf72803e54c4b391a4518b2b821`。
- reviewed_at/literature_cutoff：2026-10-05；本次未刷新文献或评价 novelty。
- target_venue：IEEE TMC（目标，不是就绪判断）。
- policy_version：`tmc_review_policy_v3_20260621`。
- artifact_run_id：无新增科学 run；证据等级为编辑性文档复核，不重新授予 E2/E3。
- 完成：主稿澄清 RQ 对应证据、全路径成本分解、冻结公式的适用边界；建立本台账。
- 未完成：P02–P05，以及 P01 的生产端核对；paper-ready 未成立。

## v1.2 真实生命周期追加记录

- execution commit：`f31024d957bd07718c4087fa55d8f7bb707e0f6e`；唯一执行 `12/12 generate`，scientific wall
  `64.309679 s`，无重试。
- artifact：`artifacts/analysis/real_cache_victim_reload_20261005_v1/`；22/22 manifest 文件独立 size/hash 复算通过。
- 输入/输出身份：同一技术图像 `83af0966…c307`；状态 `2,195 B`，原图 `192,757 B`；n0、n1、n2 的声明保真检查全通过，任务正确性 unavailable。
- 正结果：ALPR `780 tensors / 154,308,608 B` 实际卸载为 0 后，在后续节点前真实重载；Helmet 同样 `520→0`；权重文件 hash 不变。control 无 action load/unload。
- 负结果：预测仍显著偏高；victim 对两臂是共同成本，未发生决策翻转或边界跨越；网络、queue、任务标签与泛化仍未覆盖。
- claim 更新：支持真实 lifecycle mechanism；削弱并删除 recovery-only eviction penalty、真实 victim 已改变动作、算法原创性和 paper-ready 暗示。
- review identity：`reviewed_at=2026-10-05`，`literature_cutoff=2026-10-05`，`target_venue=IEEE TMC`，
  `artifact_run_id=real_cache_victim_reload_20261005_v1`，`policy_version=tmc_review_policy_v3_20260621`，
  `evidence_level=E2_ARTIFACT_AUDITED`（bounded same-host lifecycle；network assumed）。
