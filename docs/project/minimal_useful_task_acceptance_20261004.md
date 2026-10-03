# 最小有用任务验收与机制收益试验：输入门禁终态报告

## 审查元数据

- `reviewed_at`: `2026-10-04`
- `literature_cutoff`: `2026-09-30`（本轮未联网检索，不做 novelty 更新）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`（仅用于 claim 边界）
- `artifact_run_id`: `minimal_useful_task_acceptance_20261004_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `scientific_base_commit`: `01d0617d7fb66793329b3c482f794750fb101063`
- `frozen_plan_commit`: `140a2b162411040014c774be7d97fa66d002e151`
- `evidence_level`: `E1_BOUNDED_INPUT_AUDIT_NO_TASK_EXECUTION`
- `verdict`: `DATA_OR_INTERFACE_INSUFFICIENT_CANNOT_JUDGE`

结论先行：本轮没有找到同时具备本地图像、任务匹配的独立标签、可追溯可用许可和无跨组近重复条件的输入，
因此按冻结门禁以 0 次 `generate` 停止。任务正确性尚不可判断，机制 A/B 均未执行；这不是任务失败、零准确率、
零机制收益或算法劣势。

## 1. 任务、意义与依赖判断

候选优先级在模型输出前固定为车牌识别、头盔判断、驾驶场景视觉问题，最多选一个。ALPR 与 Helmet 是两个独立视觉
服务，没有真实上下游依赖，不能包装成 DAG。若未来选择其一，适用的机制仅可能是 B：逐请求重载与复用同一已加载
base/adapter 的资源对照。DriveLM 派生链只有在下游实际消费上游生成输出且任务正确性先通过时才可能采用 A。

本轮没有任务达到输入门禁，所以没有选择主任务或机制。已有两节点后缀恢复只证明 technical workflow 的状态消费
保真，不提供车联网任务正确性，不能替代本门禁。

## 2. 来源、许可、样本与标签依据

仓库 `data/` 中只有 LuST 的两张地图和一张 `TrafficDemands.png` 折线图。三者经文件身份和视觉类型核验，均没有车牌
文本、头盔类别或驾驶问题答案；尤其没有把 TrafficDemands 图冒充实际任务。项目 `artifacts/datasets/` 没有图像。
对 Downloads、Documents、Pictures、Desktop 只做文件名和数据集标记的有界筛查；排除 vendor 与归档工作路径后，
没有出现可审计的 ALPR/Helmet/DriveLM 标注数据集。私人图片内容未被打开或自动纳入研究。

已有 base、ALPR LoRA、Helmet LoRA 均声明 Apache-2.0，权重 size/SHA-256 重新核验一致；但模型许可不等于输入许可，
模型卡也不提供本机任务标签。DriveLM demo 已有固定 revision 与注释身份清单，但 54 张图均未在本机，nuScenes 附加
条款接受仍未解决。其 9 帧也不足以形成 4+8 个无重复场景；不能复制凑数。

因此开发组和锁定检查组均为空，逐样本原始输出为空。`sample_results.json` 明确区分“未执行”与“0 错误”。

## 3. 固定方案与调用预算

计划先以独立提交 `140a2b1` 推送，SHA-256 为
`f5303b20062848ca5e7c78d2849fe16005eb817cff133576571341133f67c79f`。预算为开发最多 8 次、锁定检查 8 次、
单一机制 16 次，总上限 32 次；科学墙钟上限 1,800 秒，单进程 300 秒，不自动重试。无合格输入即 0 次并停止是
冻结条件。本轮实际模型进程、`generate`、训练、formal、holdout、下载和旧实验重跑全部为 0。

## 4. 正确性与逐样本结果

没有样本进入推理，因此没有配置搜索、模型输出、正确/错误/uncertain、耗时或截断记录。不能把此前 adapter 技术验收
对 LuST 图产生的非空文本当正确。本轮结论不是“任务正确性尚未成立”（模型已被有效测试但未通过），而是更前置的
“数据或接口不足，无法判断”。

## 5. 机制完成量—成本表

| 项目 | 重载/重算臂 | 复用/恢复臂 | 解释 |
|---|---:|---:|---|
| 正确完成量 | 未执行 | 未执行 | 任务门禁未通过 |
| 错误/失败 | 未执行 | 未执行 | 不记为 0% 错误 |
| 模型/adapter 加载次数 | 0 | 0 | 当前 run 实际启动 0 个模型进程 |
| 前缀重复计算次数 | 0 | 0 | 未选择机制 A |
| 动态状态字节 | unavailable | unavailable | 未产生任务状态 |
| 静态模型字节 | 仅盘点 | 仅盘点 | base 1,019,895,869 B；ALPR 154,428,630 B；Helmet 9,650,522 B |
| 实测墙钟 | unavailable | unavailable | 6.378 s 仅为输入/完整性审计，不是模型墙钟 |
| 无线传输 | 未测 | 未测 | 不报告传输节省 |

复用的旧后缀恢复证据包含 2,040 B 动态包、目标端预存 1,019,895,869 B base 与 154,428,630 B 单 adapter；
由于其任务正确性 `unavailable`，这里只作为接口能力背景，不纳入机制收益表，也不做字节相减。

## 6. 失败与测量限制

- first-order blocker 是输入数据资格，不是模型加载接口：隔离环境 `pip check` 与固定三份权重完整性均通过。
- 有界扫描不是全盘缺失证明；常用目录可能存在文件名无法识别的数据，但没有可追溯标注/许可清单就不能自动使用。
- 本轮没有评价 ALPR/Helmet/SmolVLM 能力，也没有估计 accuracy、latency、显著性或泛化。
- 没有传输、网络、RSU、冷存储清理或热缓存随机化测量；静态字节只是本地所需资源盘点。
- 未修改算法、权重、production action 或历史 artifact。

## 7. 代码、模型、输入身份与完整性

科学基线为已推送的 `01d0617`；其中执行提交 `1f809cf` 和结果提交 `01d0617` 均已核验，后者已在本轮开始时成功同步到
`origin/codex/workflow-suffix-recovery`。base/ALPR/Helmet 权重 SHA-256 分别为 `d05b567e…91a2`、
`95103f5d…0f83`、`e899c9c1…dd2d`。完整字段、命令摘要与文件 manifest 位于同名 artifact 目录。

主工作区起止均为 `73051ab`，七个用户修改文件的逐文件 SHA-256 完全一致；未从主工作区提交任何内容。

## 8. 最小后续数据需求与停止点

下一次至少需要：同一明确任务的 12 张非重复图像、在模型输出前固定的标签、scene/group ID、来源 revision、许可/条款
记录、每文件字节与 SHA-256。不能同时混用 ALPR 与 Helmet。已知最小的公开缩减候选是 DriveLM demo：9 帧、54 图、
图像 6,114,069 B，加注释 402,384 B，共需下载 6,516,453 B；只能形成 4+5 的缩减划分，且必须先由负责人明确接受
nuScenes 附加条款并授权下载。ALPR/Helmet 尚未选定可用标注源，因此不虚构下载量。

本轮到此停止，交中央决策。只有任务正确性成立、单一机制收益可解释且简单复用仍有实际不足时，才讨论新算法或训练。
