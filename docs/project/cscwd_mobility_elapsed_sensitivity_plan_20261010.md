# 移动时间语义敏感性：执行前冻结计划

## 只读事实

- 现有 `CalibratedContinuousWorkflowEnv` 在 `decision_step_index` 下每个真实决策（失败也包含）令 `step_index += 1`，`_current_rsu_id` 和真实接触预算读 `rsu_sequence[step_index]`；`clock_seconds` 则累加 modeled action cost，并独立判 deadline。`failed_service_seconds=2`、`decision_step_seconds=5`。因此 2s 失败也推进一个合成位置，是已实现的旧合同，并非本轮认定的 bug。
- 冻结 workload 的 `rsu_sequence` 是 NGSIM handoff-pressure 派生的**合成离散序列**，没有每个位置的原始 NGSIM 进入/离开时刻。`source_interval` 只给窗口身份及原始 frame/time 区间，不足以恢复每决策物理停留。故本轮只做“每序列元素固定 5s”**合成敏感性场景**，不称 NGSIM 真实物理时间。
- 不触碰主 checkout 的七个用户修改文件；独立 A worktree、旧默认及旧原件不变。来源 B commit `8176c8c7d96d3d1d360b49a928a9e29cd5ab2aaf` 的候选协议 SHA `61295f49d03ce2f9c4f2b8cf644bc66d6a0436f43b10a61ad9b846776b54b6c4`；原 workload/base SHA 按协议验证。

## 唯一 opt-in 语义假设

- 旧 profile 为原环境，不变。新 `synthetic_elapsed_5s_atomic_v1` 只用于诊断：决策起点的移动 slot = `min(floor(clock_seconds / 5), len(rsu_sequence)-1)`；同一 RSU 连续 slot 表示一段合成接触期。动作的 modeled cost 令 clock 前进，可跨 0/1/多 slot；下一决策读取新的 slot。
- 动作作为**阻塞、原子**操作在起点 RSU 完成；即使服务耗时越过接触边界，当前 service/cache/state/reward/cost 仍按旧代码起点 RSU 原子结算，下一决策才切换位置。action4 同步阻塞，成功时在当前节点完成后提交目标 state；不引入异步预取、中途转发或服务中断。此原子假设没有原始数据支持，只检验时间尺度敏感性；不能据其声称物理正确。
- action1/4 的真实接触预算是当前 slot 剩余时间加后续**连续相同 RSU** slot 的时间；没有后续不同 RSU 或到序列末尾时只给当前 slot 剩余时间，避免编造轨迹末端无限驻留。决策预览与公开 contact 使用因果预测序列和相同剩余 slot 规则。其它成本、cache/state、五动作 mask、deadline、reward、checkpoint 不变。

## 冻结样本、方法与上限

- 只用已消费 development：`dev_01`,`regression_00`（首步 ready/短 compute/接触不变）；`regression_05`,`frozen_check_05`（初始 bundle 缺失，可构造 2s 失败等待）；`dev_00`,`frozen_check_02`（ready 且首步 compute>5s）；`frozen_check_04`,`frozen_check_06`（ready 且 action2 fallback 8s+compute>10s，可跨两个以上 5s slot）。这 8 个 ID 在任何策略结果之前按 manifest 静态状态/耗时条件选定；不根据算法胜负增删。每类另以固定合法动作的局部合同测试确认触发；若真实策略不选该动作，保留“未触发”记录，不换点。
- 固定 seed `7`。旧 v4 SA、PPO、MAPPO 的 `selected` checkpoint SHA 分别 `6fca5f2dac35671b445fa1439e71585d3625a293dc7a5d6d414cb25a9f247feb`、`f93ad95a22a066995bf0deea6f12efe04f6ced2a5f62b021ce082561a531547d`、`5ef457a8403f2b7c83ce299ba9ddc323f9aa949f81ad42e85a2f7ea093d259ce`；条件弃权 SA SHA `9fdd94cd72227c171b23a686163e47631e46158a0d8dfe791c1dabf2472301fe`；two-step 用原规则、无 checkpoint，单列 model-based 两步 clone 能力及不同决策信息权限。
- 8 实例 × 5 方法 × 2 时间 profile = **80 episode**，真实 episode `env.step` 硬上限 `2,880`、每 episode 至多原 `min(24, instance.max_steps)`，0 训练、0 参数更新、0 选模、0 新 seed、0 holdout；two-step 决策模型 clone `step` 另计并披露，不作为实际环境 transition。若公共输入/mask 不合法则记录缺口并停止该 cell，不临时重训。
- 两 profile 对同一冻结 checkpoint 与实例各执行一次 raw deterministic policy；仅称**固定策略跨语义敏感性**，不做公平新排名。逐步记录起止 clock/slot/RSU、接触预算、动作合法性、completion/failure、模型/state/input bytes、所有 step cost 分量及 deadline。核对原生 cache/prepared-state 终态与逐项成本守恒，统计 2s 失败后是否在旧 profile 提前换 RSU、新 profile 是否仍驻留，以及策略结果方向是否依赖语义假设。
- 最多一次执行。无需结果驱动重试。需要共同环境重新匹配训练时只陈述条件，不自动训练；不更新论文、表格、队列/多车/跨 workflow 工程。
