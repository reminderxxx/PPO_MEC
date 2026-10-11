# CSCWD phase-v2 重复 action4 有界分支确认：事前冻结方案

- `frozen_at`: 2026-10-11，真实分支运行前；`scope`: development-only；`artifact_run_id`: `cscwd_phase_v2_repeated_action4_branches_20261011_v1`。
- 源 pilot：`cscwd_phase_v2_causal_public_pilot_20261011_v1`，执行 commit `5a79ffd018d4c260bc7a8417229a440c7d591fe4`；protocol SHA-256 `c2ff08fafee5523a669ec7a201dfe3558bd45b54589bb291383c2ba7a97246c7`、base config SHA-256 `20e82b0ea4ff5c7caf597d987bcef1f7523dc0c1c7ece21deba32aa7d47af923`、`behavior_rows.csv` SHA-256 `b3975c212bf5e0e9d9f444e2f32367f530a27f84b3f0ec494384f45ea94b0eef`。源行为仅用于**运行前**固定起点和核对原动作，不用于在线分支决策。
- 物理/数据合同：`raw_ngsim_event_time_v2`、`dev_01`、US101 原始连续来源 `cscwd_newdev_us_101_v1183_t1118849293800_1118849412600`；既有 phase 合同和 base config 原样加载，三条件恰为 `default`、`adverse_link`、`ample_cache`。输入 SHA 从 pilot protocol 锁定，原始 NGSIM 只读，不下载/改写。

## 起点与匹配规则

原 pilot 的 `sa_candidate/selected/v2/us_101` 行中，恰有 11 个 seed×condition episode 出现 step0 `action4` 且 `delivered`、step1 再次 `action4` 且 `target_departed_before_delivery`。这是回顾性开发异常集合，不是 11 个独立样本。按 `(source_segment_id, condition_id, workload_design_id, step1 前全部动作)` 去重，恰得到三类公共前缀：每条件一次 `action4`。三条件**全部纳入**，不看新分支结果筛选；重建环境后还要保存共同公开 `semantic_state + action_mask` 的规范化 SHA，并检查三个 SHA 唯一、两候选动作 3/4 都合法。若原行为不再满足 11 行/三类、原 step0 真实重放与行为账本不符或任一 action 不合法，则记录失败并停止，不替换起点。

每条件重建一次 reset→step0 `action4` 的前缀，再从完全相同的 post-step0 环境状态复制两个**仅用于离线诊断**的分支。分支 A 首动作 `action4`；分支 B 首动作 `action3`。两个动作均是同一五动作合同下的合法原生动作；无模型/reward/环境修改。分支首动作之后使用相同冻结公共规则：每次只看当前 `action_mask`，若 action3 合法则选 3，否则选 2。该 continuation 不读取原始轨迹后缀、实际未来接触或交付结果，不调用 decision clone/privileged rule。环境 `step` 内的真实未来只用于物理结算和事后事件归因。所有分支无训练、无 checkpoint 载入。

## 预算、指标与判读

- 单次 create-only 运行，三起点×两首动作＝**6 分支**；每分支首动作后最多 8 continuation step，外加每起点一次共同前缀 step0。因此理论最多 **57 个真实 `env.step`**，硬门保留原方案 **≤12 起点、≤24 分支、≤192 环境步**；任何异常不自动重试、不补 seed、不改条件。最多一个输出 root。
- 逐事件保留公开状态 hash、合法 mask、执行动作、服务完成/失败、迁移交付/目标离开、phase 时间、输入/模型/状态/backhaul/partial 字节、重算、reward 与时钟守恒。逐分支统计从共同 post-step0 状态到结束的服务完成、按期、失败事件及三类传输总字节、重算、耗时；同时报告原始 pilot 的 11 次重复动作与其三个去重身份。
- **事前主要判据**：在每个配对起点，若 action3 分支的 workflow 完成、按期和服务失败不劣于 action4，且其 `input + model + state` 总字节严格较少，才称该起点在本冻结 continuation 下存在局部成本支配。若服务退化、总字节未降或两动作的未来状态不可比，直接记录负结果/机制取舍，不以平均值掩盖。报告两方向差值、model/input/state 分解、重复发送是否真的省开销、成功交付是否被破坏；不由此自动修改生产策略。
- 公共可识别性仅依据分叉前 `semantic_state + action_mask` 中当前/预测目标、已公开的 prepared-state prefix 与接触估计；事后 `migration_status`、真实轨迹未来和最终效果只作标签，不能反向成为在线规则。条件与 seed 是同一 US101 来源的重复，任何配对结果均不能给出独立统计或论文 claim。

失败时保留启动/失败回执和已写本地证据；不覆盖源 pilot、不读取 formal/holdout，不生成正式 checkpoint/论文表。完成后先做独立账本核验，再决定是否仅建议一个可实施的公共单变量修复或建议不修改。
