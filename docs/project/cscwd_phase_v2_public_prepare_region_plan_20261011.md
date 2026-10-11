# CSCWD phase-v2 公开 prepare 正负区域有界诊断：事前冻结

- `frozen_at`: 2026-10-11，真实环境分支运行前；`run_id`: `cscwd_phase_v2_public_prepare_regions_20261011_v1`；development-only，0 训练、0 formal/hidden/sealed。
- 源只限 B 冻结 pilot `cscwd_phase_v2_causal_public_pilot_20261011_v1` 的两条已授权 NGSIM 长来源、`dev_01` 和 v2 三条件。沿用前轮固定的 `raw_ngsim_event_time_v2` 物理/缓存/奖励合同与原始 CSV 身份。B `behavior_rows.csv` SHA-256 `b3975c212bf5e0e9d9f444e2f32367f530a27f84b3f0ec494384f45ea94b0eef`；`rule_behavior_rows.csv` SHA-256 `229ba1c059f7dd43b2948cc23ae27a952c9fa20bdfdcafdcb671b25089c8704b`。旧 `cscwd_phase_v2_repeated_action4_branches_20261011_v1` 的三起点仅为**已暴露回归**。

## 结果盲覆盖规则

仅从 B 已存 `selected` learned 行和 `fixed_rule` public-rule 行读取 `source_segment_id`、`condition_id`、`seed`、`arm`、`step`、`executed_action`、**公开** `action_mask`；不读取 reward、迁移状态、未来轨迹或任何分支效果来选起点。对每个 episode 按已执行动作构成决策前 prefix，按 `(source, condition, prefix)` 去重。预注册 prefix class 顺序为 `[]`、`[0]`、`[1]`、`[4]`（即 reset 或一次原生动作后的状态）；仅当 action3 和 action4 在同一公开 mask 都合法时纳入。两来源×三条件×四 class 的 24 个候选格全部列选择清单，未观测、非法、超深度均写排除理由。已存更深合法前缀仍完整列作“深度>1”排除，不按效果替换。若合格去重起点 >12，则按 source、condition、class 固定字典序保留前 12，其余明确列出；不扩 cap。

每个合格状态的“原策略首动作”取该 cell 已记录行中固定优先序 `sa_candidate > sa_control > mappo > ppo > public_rule`，同 arm 先 `selected` 再 `fixed_rule`，同 seed 取最小数值；仅用于选一个**已存动作标签**，不加载 checkpoint。分支动作集合是去重的 `{原策略首动作, action3, action4}`，最多 3 个；所有动作必须在同一原生 action mask 合法。三种臂的**后续**统一为公开 `action3 if legal else action2`，最多 7 步。该能力与前轮一致，完全依据各分支当前公开 mask；并不声称后续复现各 learned 原始 policy。环境由相同外生原始轨迹、初始状态、条件和成本模型复制，真实未来仅由 `env.step` 物理结算与事后归因使用，禁止决策 clone 或在线 oracle。

运行前仅用上述允许字段对 B 固定 CSV 试枚举：24 个浅层候选格中，Lankershim 三条件均无 action3/4 同合法前缀，US101 `ample_cache/[1]` 未观测；其余 **11 个起点**全纳入，`[]/[0]/[1]/[4]` 分别为 3/3/2/3 个。`[4]` 三起点为已暴露回归，其余 8 个是新覆盖状态但仍只来自 US101 同一原始轨迹。原策略首动作去重后共 **27 分支**；包含共同前缀的理论最大真实步数 **224**，低于 288 硬限。这个枚举未读取 branch outcome，运行时仍需重建公开状态 hash 和完整排除账本；任何数量/合法性变化即失败停止。

先完整枚举并写起点/排除清单，再启动分支；每起点 prefix 最多 1 步、每分支最多 8 步，总硬限 **12 去重起点、36 分支、288 个 `env.step`**，单次 create-only root，任何身份/预算/合法性失败即停，保留失败回执，不自动重试/补选/扩 seed。脚本与方案必须先通过局部测试并提交 Git 后运行；运行中不改生产算法、reward、mask、数据或 B 原件。

## 预设结果口径

每分支记录完成、按期、服务失败、模型/输入/状态/backhaul/partial 和总字节、耗时、重算、动作4交付/失败、事件相位、实际目标离开及**后续目标计算使用**。原策略首动作、action3、action4 均按同一状态配对；只按服务向量与逐项资源向量报告，不用任意 reward 权重合成净优越。正区域事前定义为 action4 首步成功交付，后续目标 RSU 实际计算且该步不重算，同时相对 action3 在同公开 continuation 下减少重算或改善完成/按期/失败；若多耗传输必须同列。无益区域是 action4 无服务/重算/时间改善而总字节更高；其余为成本取舍、被截断或不确定。若没有满足正区域的实例，原样报告缺口，不新增轨迹、条件、prefix 深度或调选择规则。

三个 `[4]` 旧回归起点单列，与新覆盖状态不得合并成“新独立证据”。所有新起点若来自同一 US101 trace，也必须明示相关开发样本；任何区域划分仅供定位公开状态缺口，论文/优秀基线/可部署规则均 `Unverifiable`。
