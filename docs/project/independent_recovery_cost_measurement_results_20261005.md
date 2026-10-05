# 独立恢复成本测量结果

## 审查身份与结论

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-10-05`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `independent_recovery_cost_measurement_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 解析基线提交：`de70d5721ea5ccf07ce6325c92cb4ac833b68efc`
- 冻结执行提交：`b7c8c4dadf7a10944ebff8405b5faa56666aeaa0`
- `evidence_level`: `E2_ARTIFACT_AUDITED`（bounded same-host real-model technical execution；网络为假设）
- `verdict`: `DIRECTIONALLY_SUPPORTED_WITHOUT_BOUNDARY_TEST / NOT_PAPER_READY`

唯一一次 clean execution 完成 2 条事前冻结条件、每条 3 次 restart/recovery 配对、24/24 次真实 `generate`；无训练、下载、
old holdout 访问、自动重试或追加预算。scientific wall 为 `243.001998 s`。6/6 配对的完成量、后缀输入、输出 token、模型
identity 和 production action-4 状态消费全部一致。72 个 manifest 文件的 size/SHA-256 独立复算 72/72 通过。

事前成本估计在 6/6 次都选择 recovery，事后按“单机 action wall + 同一冻结模拟动态网络时间”判断也都是 recovery，选错代价均为
0。但全部条件在同一决策侧，因此**没有检验到决策边界**，不能据 6/6 声称分类稳定性或泛化。绝对成本只得到有限支持：四组
condition-arm 的预测均偏高，prepared/recovery 的相对绝对误差尤其达到 `45.9%--49.7%`。

## 计划、信息隔离与成本口径

事前冻结文件为 `configs/acceptance/independent_recovery_cost_measurement_v1.json`，其 SHA-256 为
`0de253a102691e88355e1706929fcae14c0254cd02f01c0e259d240a76dc45f3`；run 内副本 SHA-256 为
`d269f3c8121054aaf7dc9e77e6205732e40b6471c619647897b664ea22820afb`。副本和源文件的字节差异只来自执行前 Git commit
已写入源文件之外的路径语境；执行器记录两者 hash，未按结果修改预测。原始 calibration、执行顺序、arm order、随机种子
`20261005`、评分和预算都在模型输出产生前提交。

> 注：两个 SHA-256 不相等是因为 `write_json` 对 plan 进行 sorted-key 规范化后写入副本，不是内容字段变化；JSON 对象复读相等。

成本分为三层：

1. 单机实测：arm wall、模型加载、n0/n1 推理、状态序列化/保存、校验和输入重建。
2. 字节实测：原输入 `192,757 B`，每次 production state package `2,185 B`；base/adapter 静态目录单独登记。
3. 网络假设：100 Mbps 和一次正动态传输 0.02 s，只用于两路相同规则下的模拟传输时间；不是无线 wall-clock。queue wait 为
   `unavailable`。

`target_model_prepared` 在 action 计时前加载模型；setup load 原始值仍保留。`target_model_requires_local_preparation` 在 action
计时内从既有本地文件加载模型。每个 source/arm 都是新进程，OS 文件缓存未清空且不受控，不能称磁盘冷启动。

## 预测误差与动作一致性

`signed error = frozen prediction - measured scored completion`，因此下表正值表示预测偏高。网络项对 restart/recovery 分别固定为
`0.03542056 s` 和 `0.0201748 s`。

| 条件 | arm | 冻结预测 s | 实测 scored 原始值 s | 中位实测 local wall s | 中位绝对误差 s | 相对绝对误差范围 | 动作匹配 |
|---|---|---:|---|---:|---:|---:|---:|
| prepared | restart | 16.694109 | 15.150372, 15.303677, 15.673569 | 15.268257 | 1.390432 | 6.5%--10.2% | 3/3 |
| prepared | recovery | 5.874839 | 3.923283, 3.931127, 4.027138 | 3.910952 | 1.943712 | 45.9%--49.7% | 3/3 |
| local preparation | restart | 17.833756 | 17.464548, 17.537555, 17.348280 | 17.429127 | 0.369208 | 1.7%--2.8% | 3/3 |
| local preparation | recovery | 7.020647 | 6.138928, 6.131225, 5.828562 | 6.111050 | 0.889423 | 14.4%--20.5% | 3/3 |

逐次 signed errors 为：

- prepared/restart：`+1.543737, +1.390432, +1.020540 s`；prepared/recovery：
  `+1.951556, +1.943712, +1.847702 s`。
- local-preparation/restart：`+0.369208, +0.296201, +0.485476 s`；local-preparation/recovery：
  `+0.881720, +0.889423, +1.192086 s`。

两条件的预测动作均为 recovery，六次实测较优动作也均为 recovery。每次 wrong-choice cost 为 `0.0 s`。这说明旧校准在本次
同机、同模型、同输入的两个合法 residency 条件下保持了动作方向；它没有证明可在成本接近时正确翻转，也没有证明独立 workload、
主机或无线链路上的分类性能。

## 全部重复测量原始值

表中 `load` 在 prepared 条件属于计时前 setup，在 local-preparation 条件属于 action wall。`state` 为
source serialize/save + recovery validation + suffix input rebuild；每次 state package 均为 `2,185 B`。

| seq | 条件/重复 | arm 顺序 | restart wall | restart load | restart n0 | restart n1 | recovery wall | recovery load | recovery n1 | state | 预测/实测 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | local-prep/rep01 | recovery→restart | 17.429127 | 1.157607 | 11.546551 | 3.892442 | 6.118753 | 1.199396 | 3.928551 | 0.003755 | recovery/recovery |
| 2 | prepared/rep01 | recovery→restart | 15.114952 | 1.133000 | 11.107011 | 3.902160 | 3.903108 | 1.172753 | 3.899962 | 0.003536 | recovery/recovery |
| 3 | prepared/rep02 | recovery→restart | 15.268257 | 1.151891 | 11.257937 | 3.902534 | 3.910952 | 1.190704 | 3.907694 | 0.003602 | recovery/recovery |
| 4 | local-prep/rep03 | recovery→restart | 17.502134 | 1.157719 | 11.638865 | 3.876687 | 6.111050 | 1.188684 | 3.926565 | 0.003414 | recovery/recovery |
| 5 | local-prep/rep02 | restart→recovery | 17.312859 | 1.156729 | 11.431668 | 3.897850 | 5.808387 | 1.153611 | 3.919900 | 0.003399 | recovery/recovery |
| 6 | prepared/rep03 | restart→recovery | 15.638148 | 1.135861 | 11.634886 | 3.901007 | 4.006963 | 1.181453 | 4.001440 | 0.005897 | recovery/recovery |

状态子组件原始值（serialize/save, validation, rebuild，单位 s）依次为：

1. `(0.000549875, 0.000357125, 0.002847708)`
2. `(0.000522625, 0.000296000, 0.002717792)`
3. `(0.000487958, 0.000386125, 0.002727791)`
4. `(0.000517375, 0.000288959, 0.002608083)`
5. `(0.000514083, 0.000280791, 0.002604334)`
6. `(0.000516625, 0.002765083, 0.002615250)`

第 6 次 validation 较其余重复高，但总 state component 仍只有 `0.005897 s`，未翻转动作。全部原始 receipt、输入/token 对照、
进程命令和时间戳位于 artifact，不以表格舍入值替代原件。

## 与原 12 点和两步前瞻的关系

原 12 个 d01--d12 是开发/机制验证设计点，不再称未观察测试。它们仍提供三项边界证据：

- 新规则与正确、同信息两步前瞻 12/12 动作及估计成本完全相同，没有算法领先证据。
- d10/d11 保留为估计误差导致的负结果，说明结构账本正确不能替代成本校准。
- 相对原阈值的合成汇总减少时间/字节时，新规则多 5 次重跑并增加 `53.641216 s` 重算；该权衡不得只写改善项。

本轮新真实执行没有 cache victim，因此不能独立检查 victim-induced later reload。它只检查已准备/需本地准备两种可合法核算状态下，
共同模型准备成本、前缀重算、状态恢复与后缀保真的成本预测。

## 文献归因与最强可支持主张

成本感知驱逐/取回并非首创：GreedyDual-Size 已把取回成本、对象大小和 locality 结合；Yao 等在 ICC 2022 直接研究 loading-cost-aware
model caching 与 request routing；Li 等在 TMC 2026 研究模型卸载的内存成本和重载延迟权衡。本项目不能把“考虑驱逐或模型重载
代价”包装为新算法。

**最强可支持主张**：在一个固定真实模型、固定技术输入和已实现 production action-4 状态路径上，原生状态恢复能以完全一致的
后缀输入/输出跳过一次真实前缀执行；用旧测量事前冻结的成本方向在两个合法模型 residency 条件的 6 次新配对执行中均选择了
实测较低成本的 recovery。该结论仅是同机机制/经验性结果；绝对成本存在明显高估，且没有决策边界、真实 victim reload、无线、
queue 或任务正确性证据。

## 最多两个投稿关键缺口与停止条件

1. 缺少生产路径可合法执行的 dependency-safe victim→later reload，以及事前冻结后覆盖 restart/recovery 两侧的独立条件；当前不能
   证明驱逐代价项本身在真实执行中改变正确决策。
2. 缺少远端传输/共享资源和有任务标签的独立 workload；当前只能分列同机墙钟与模拟网络时间，不能给出部署端到端或语义收益。

本轮到此停止扩大实验。现有证据只支持机制/经验性论文定位，不支持继续以“新算法优于两步前瞻”为主线，也不因 6/6 同侧结果追加
条件制造翻转。

## 顶刊政策审查摘要

- Evidence inventory：事前 plan/config/hash、固定执行 commit、18 个 scientific process receipt、6 个 repeat receipt、
  raw JSON/CSV、aggregate、terminal、逐进程 command/timestamp 和 72-file integrity manifest 齐全；无真实无线/queue、
  labeled task、真实 victim reload、formal/holdout/support。
- Hard blockers：与正确两步前瞻无增量算法价值；独立新执行未覆盖 decision boundary 或真实驱逐后重载；没有部署端到端证据。
- Major concerns：单模型/输入/主机、OS cache 未控制、全部同侧、绝对成本系统性高估、小样本无 CI/推断统计。
- Minor concerns：第 6 次 restore validation 高于其余重复但未改变方向；计划副本因 JSON key 规范化具有不同字节 hash。
- Scorecard：`N/S (not scored)`；这不是完整投稿包，不能用 6 个同侧重复给出伪精确 100 分。
- Safe claims / prohibited claims：见“最强可支持主张”及 manuscript artifact ledger。
- Required actions before re-review：只对应上节两个关键缺口；本轮不继续执行。
- TITS/TVT fit note：降低 venue 不消除边界、真实 victim reload 和部署成本缺口；当前仍应按机制/经验性工作定位。

## Artifact inventory

- `[A1]` `artifacts/analysis/independent_recovery_cost_measurement_20261005_v1/`：frozen plan、6 个 repeat 的 18 个进程
  receipt、`all_measurements.json/csv`、aggregate、terminal 和 integrity manifest。
- `[A2]` `configs/acceptance/independent_recovery_cost_measurement_v1.json` 与
  `independent_recovery_cost_measurement_plan_20261005.md`：模型输出前的预测、顺序、预算和评分 freeze。
- `[A3]` `artifacts/analysis/production_action4_independent_repeat_20261005_v1/`：只用于校准的旧 3 次配对。
- `[A4]` `artifacts/analysis/eviction_aware_recovery_validation_20261005_v1/`：开发/机制验证的 12 个合成设计点，不是独立检查。
