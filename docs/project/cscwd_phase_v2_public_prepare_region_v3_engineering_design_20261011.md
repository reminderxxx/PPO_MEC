# CSCWD phase-v2 公开 prepare 区域 v3：工程修复设计（未执行）

- `designed_at`: 2026-10-11；前置失败为 v1 `55b826571c4b2474bf1f52ab228892d37855a850425ce20249b942100127e1ad` 与 v2 `4a19c502d5c5fb298601c211ed0fdb07d8a6b0b01ff829e164fc4bfda605c088`。
- 本轮仅实现和测试执行器，不读取真实数据进行分支展开，不运行新区域科学实验。原 `v1/v2` root、主仓用户文件、生产环境/策略/reward/五动作合同保持原身份。

## 状态身份 v3.0

保留原结果盲选择全集：两来源×三条件×四浅前缀共 24 格；原有 11 个入选格、13 个排除格及每格理由原样记录。选择格 `(source, condition, prefix)` 是**采样/来源身份**，不是状态相等定义。每个合法前缀需在真实环境按旧动作回放，计为 `env.step`，随后生成两种状态身份：

1. `public_observation_sha256` = 旧 `semantic_state + action_mask` 公共投影的 canonical JSON SHA；同 hash 是在线 actor 无法区分的**观测组**。不同隐含状态/外生未来可能落入同组，这些状态保留作部分可观测对照，绝不因同公开 hash 直接排除。
2. `full_simulation_sha256` = 版本化显式快照的 canonical JSON SHA，覆盖完整外生轨迹（含未来，仅审计/环境使用）、原配置/工作流/物理条件、几何、当前 step/clock、节点进度、cache resident/LRU、prepared/pending state、执行权、累计 metrics/reward 与 deadline 标志。它不能进入在线决策。同一 full hash 才算严格重复：第一格为代表，其余格只标 `full_duplicate_of` 并保留原选择记录，代表仅执行一次分支；没有证据时不猜测重复。

所有 11 个入选格必须先完成身份回放/分类并原子保存 `identity_inventory.json`，**然后**才允许任何比较分支。若真实状态身份不符或预算不足即失败，0 比较分支。真实 prefix 回放至多 8 `env.step`，明确从未来总 **288 步**预算扣除；它是科学环境执行，不称“零步 preflight”。同一 full state 的额外 prefix 回放也计入预算。v2 的 77 步没有逐步账本，不得通过 v3 回放来补写旧失败的碰撞对。

## 持久化与失败口径

每个完成的分支单独写 `branches/<start_id>__actionN.json`：输入的采样/公开/full 身份、原动作与 condition、逐事件 ledger、完整 metric 向量、事后迁移使用。结构含 `payload_sha256`，读回重新计算；使用同目录临时文件、flush/fsync、`os.replace` 原子发布。分支未完成时不产生完成文件；失败回执记录当前分支、已完成文件数、已用 `env.step`、`PARTIAL_UNVERIFIABLE`。全局失败保留所有已完成文件，不能把部分表冒充整轮区域 verdict。最终汇总只从全部完成分支文件读回；任何缺失/hash 不符即失败。

未来 v3 计划仍是原公开动作3/4/原策略首动作、公开 continuation、原成本和区域判据；上限仍为 12 选择格、36 分支、288 步，不新增来源/seed/条件、模型或训练。**当前未知**：11 格中独立 full state 数、公开观测组碰撞结构、full duplicate alias 数、代表分支实际数及正/负效果；这些必须在未来另获授权的新 run ID 中经真实 prefix 身份预检获得，不能凭 v2 失败回执推定。若 full duplicate alias 使比较数少于旧计划 27，完整报告格与代表映射；不按结果补点。v3 仍属同一 US101 开发轨迹，不能形成独立数据、formal/holdout、优秀基线或论文贡献。
