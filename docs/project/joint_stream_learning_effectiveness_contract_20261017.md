# 阶段一：Stream 学习有效性验证（2026-10-17）

## 授权和目的

用户要求“按照第一节阶段启动，后续每个阶段结束后给我下一阶段的任务点”。本阶段执行独立开发评估，回答训练是否在已有公共先验之上增加交付收益、哪些 seed/条件回退，以及现有曲线能支持什么结论。阶段完成后交付下一阶段任务点；不自动开始算法修正或新训练。

## 冻结输入

- 原 run：`artifacts/joint_stream_learning/20261016T162100Z_confirmation`；算法源码 `35975cb3d73c84ddbabf01f6749bbc2d60c826de`，`joint_stream_control_v2`。原始 run/checkpoint 不覆写。
- 全部八方法：SA、SA去预测、MAPPO、PPO、DQN、cache、DAG、独立DT；seeds 7/17/27。
- 固定已有完整48开发条件，包含两时间会话、short/medium/long、两arrival mode、单/双family、4096/8192容量；不访问formal/holdout。
- 所有方法使用原环境、reward、模型/状态费用、动作权限、固定GRU和共享状态。0点是零RL更新的现有控制器，已有公共费用先验、mask和协调机制；不是随机动作或零规则baseline。

## 评估预算与复用

| 点 | 每方法每seed条件数 | 来源 |
|---|---:|---|
| 0 | 48 | 按原factory/seed重建；必须匹配manifest记录的初始tensor/actor SHA |
| 32 | 48 | 复用原固定12probe，仅新增缺失36条件 |
| 64 | 48 | 复用原48完整固定最终结果，不选择峰值 |

共3456比较单元，复用1440，新增2016评估session，新增评估最多967680物理帧；**新增训练0、优化更新0**。资格检查另记软件成本：冻结评估器固定SA/seed7/32/dev_000复算1次；定稿前两次完整consumer检查共366帧及一次记录字段检查失败（0物理帧）不混入性能表。不得增加训练、补seed、改变面板、补造8/16/24等不存在的快照。

只使用已存在的0/32/64三个观察点，不声称密集全程评估曲线或渐近收敛。原64session训练回报折线保留为任务异质性/探索诊断。

## 执行和完整性

入口：`scripts/evaluate_joint_stream_learning_effectiveness.py`，四模式`qualify/prepare/run/summarize`。prepare冻结具体输出目录、脚本SHA、输入SHA、条件/预算和并发数（最多2）；run以排他启动标记防重启，不支持自动resume。

原123源码指纹、protocol资产、8个manifest/dev文件、48checkpoint文件SHA须通过。复用1440行既验SHA，又逐字段对原文件；新评估每case重建环境、重置controller，deterministic动作，禁止update且逐case核对完整参数/actor指纹未变。保存逐case原始summary、压缩逐帧选择动作、参数身份、时耗和阶段进度。新结果单独存储，不写回原run。失败保留现场，其余worker在当前case结束后停止，不自动重跑。

软件验收包括24初始化精确重建、固定32probe结果逐字段复算、面板计数/去重/费用汇总、复用篡改拒绝、错误参数身份/帧计数拒绝；不以验收奖励选择配置。

## 阶段结束交付

1. 全八方法×三seed的0/32/64奖励、完成率、完成节点、模型传输/驱逐/每完成请求费用、状态费用和选站变化。
2. 配对0→32、32→64、0→64增量及正/负/平条件数，按size、family、capacity、arrival、时间会话分层；保留所有退化条件。
3. 固定48面板曲线（明确只有三个真实点）和参数/动作变化证据；区分“参数更新”“行为变化”“开发交付增益”“收敛/泛化”。
4. 按`top_journal_review_policy`顺序独立审查并更新PROGRESS/BUGS/ARTIFACT_RECORDS，验证后scoped commit/push，交付下一阶段任务点。

本轮不改变canonical和现有固定最终基线，不以32优于64替换最终结果。既有优势保护合同`artifacts/joint_stream_learning/next_stage_preservation_plan_20261017.json`继续有效：后续候选须通过全48×三seed总均值及seed7/27非回退门禁，失败保留现有基线。

## 预留下一阶段任务方向

须按本阶段结果选择最小独立任务：若零更新先验已强且训练无增益，优先同state/candidate/mask下分解prior/learned logits、梯度分组和动作覆盖；若训练确有收益而seed17退化，优先选站—状态连续性—family驻留的一阶负例。预取站点/驻留与实际使用继续核验。参数变化、小KL、小图gate均不单独构成效果或收敛证据；不预设GNN/surrogate/预测应当有效，不为排名改reward。

## 证据边界

只涉及已消费开发面板，两时间会话不能冒充144个独立窗口。新评估不提供独立泛化或纯模块因果归因；formal/holdout仍封存、论文证据`E0_UNAVAILABLE/Unverifiable`，不晋级paper-ready。本阶段不重新评价novelty；公开文献截止沿用2026-10-17。
