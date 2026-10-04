# 最小机制证据闭环冻结方案

- `frozen_at`: `2026-10-05`
- `target_venue`: `IEEE TMC`（仅使用 claim-boundary policy）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `status`: `FROZEN_BEFORE_NEW_EXECUTION`

本轮只闭合共享缓存公平成本、两节点后缀恢复对照与有界决策空间三类证据，不训练、不下载、不开启 formal/holdout，
也不修改 production action 4。最新 ALPR 对照继续保留为负结果：12 个公开样本上的 base/adapter locked-check 均为
4/8 exact、micro CER 均为 0.2453；它不是新独立测试，也不证明 adapter 任务优势。

机器计划为 `configs/experiment/mechanism_evidence_closure_v1.json` 与
`configs/acceptance/two_node_workflow_reexecution_comparison_v1.json`。A/C 新执行墙钟上限 120 秒；B 使用同一冻结
模型、adapter、输入与 greedy 配置，计划 6 次真实 `generate`、硬上限 8、总墙钟上限 1,200 秒，不自动重试。

实验 A 不重跑四配置请求，只消费已有 576 行 old/candidate 原生账本。所有候选臂保持 72 个完成请求；等待和完成时间
另用明确标为模拟的传输/服务公式给出，不回填成原生实测。旧语义的低字节必须与 36/60 个拒绝同时报告。

实验 B 比较连续执行、中断后从 `n0` 重跑、中断后导入状态只执行 `n1`。状态包、模型权重、输入图像和中间结果分列；
目标预存模型路径做真实同机测量，目标缺模型只做字节与网络公式敏感性。技术输出只用于保真，不用于交通任务正确性。

实验 C 冻结 12 个设计点，覆盖共享开关、136/320 MiB、短/长后缀和低/高状态成本；高成本只用于长尾，因为短尾
没有可解释的状态准备机会。低成本 2,040 B 来自既有 technical package，但不是完整 production state；高成本
160 MiB 是合成压力点。三方法共享合法动作 0--4 与原生约束；完整后缀参考最多枚举每场景 625 条四步序列。
全部格均报告，不能按结果删改。设计点不是独立现实样本。
