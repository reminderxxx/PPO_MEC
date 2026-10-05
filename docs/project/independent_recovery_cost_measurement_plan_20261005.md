# 独立恢复成本测量：事前冻结计划

## 研究问题与解析基线

- 唯一问题：使用事前冻结的成本估计时，现有驱逐代价感知规则能否预测新执行中的 restart/recovery 成本与较优动作？
- 本轮解析并固定从提交 `de70d5721ea5ccf07ce6325c92cb4ac833b68efc` 开始；该提交中的 12 个设计点是开发/机制验证证据，不再称为未观察测试。
- 新规则与正确信息匹配的两步前瞻在原 12 点 12/12 持平；d10/d11 的选错和重算增加继续保留。
- 不训练、不下载、不读取旧 holdout，不增加算法、动作、跨 workflow、队列、无线平台或系统功能。结果出来后不改条件、估计或评分。

## 校准、新检查与假设的分离

- **已有校准**：`production_action4_independent_repeat_20261005_v1` 的 3 次 restart/recovery 配对原始值。它只用于冻结本轮成本估计，不作为新检查结果。
- **本轮新检查**：同一合法 SmolVLM base、ALPR adapter、技术图像、两节点 workflow、production action-4 状态导出/导入路径上的 2 个条件，各 3 次配对重复。
- **仍属假设**：100 Mbps 和一次正动态传输 0.02 s 固定时延。它们只生成单列的模拟网络时间，不称为单机墙钟或真实无线测量。
- 单机实测只覆盖进程内模型加载、输入准备、推理、状态序列化/校验/重建和 arm wall。queue wait 仍为 `unavailable`。

## 条件、顺序与进程状态

1. `target_model_prepared`：合法 base/adapter 在 arm 计时前加载并保持驻留；加载耗时作为 setup 原始值保存，但不计入 action wall。
2. `target_model_requires_local_preparation`：相同本地权重在 arm 计时内加载；不存在下载、覆盖或手动删缓存。

每个 repeat 先执行 source 1 次以创建真实状态包，然后按冻结顺序执行 restart/recovery。随机种子为 `20261005`；六个
condition-repeat 与各自 arm 顺序已完整写入
`configs/acceptance/independent_recovery_cost_measurement_v1.json`。每个 source/arm 都是新进程，OS 文件缓存不清空且不受控，
因此只能称 `new process`，不能称磁盘冷启动。模型驻留只在单个 arm 进程内成立。

合法缓存替换导致真实后续重载未加入：现有真实模型路径只有一个 base/adapter family，生产执行器不能在不增加功能或手动删缓存的
前提下构造 dependency-safe victim 后再真实重载。该条件保持 `uncovered`；既有 native synthetic witness 不升级为实测。

## 事前预测

全部数值已在任何本轮模型输出产生前冻结。完整原始校准数组、静态字节和高精度数值见机器计划。

| 条件 | restart local wall 预测 s | recovery local wall 预测 s | restart + 模拟动态网络 s | recovery + 模拟动态网络 s | 预测动作 |
|---|---:|---:|---:|---:|---|
| target model prepared | 16.658688 | 5.854665 | 16.694109 | 5.874839 | recovery |
| target model requires local preparation | 17.798335 | 7.000473 | 17.833756 | 7.020647 | recovery |

驱逐代价感知规则的比较项在两个条件中相同：`J_restart=11.07072456 s`，`J_recovery=0.0263258 s`。第二个条件中 model
preparation 对两路均为同一合法 base/adapter 的共同成本，必须在完整成本中双边记录并在动作差值中抵消，不能只向 recovery
单边收费。两步前瞻接收相同信息，因此事前动作仍预期持平。

## 调用与时间预算

每个 repeat 的实际调用量为 source 1、restart 2、recovery 1，共 4 次。`2 conditions × 3 repeats × 4 = 24`
次，等于硬上限 24；不会自动增加预算、重试或补跑。每个子进程上限 300 s，总科学执行上限 600 s。

## 评分和停止规则

- 先验证两路完成量、restart/recovery 的后缀输入、token IDs、生产状态包消费和静态 identity 完全一致。
- 每个 repeat 用“本地 action wall + 同一冻结模拟动态网络时间”事后选出较低成本动作；真实模型加载、动态状态、原输入和控制元数据分开记账。
- 对每路报告预测减实测的 signed/absolute/relative error；报告动作一致性和选错代价；保留所有无收益、反向收益与全部原始重复值。
- 网络时间不与本地墙钟混称完整真实动作成本；模型权重没有真实网络传输。
- 六个条件若都落在 recovery 一侧，明确写成未检验到决策边界，不事后造翻转。
- 本轮只作描述性检查。完成后停止扩大实验，不追加第三类条件、训练或调参。
