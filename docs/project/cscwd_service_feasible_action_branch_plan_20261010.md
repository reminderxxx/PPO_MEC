# Action0/2/4 对称反事实门禁：执行前冻结规则

- `frozen_at`: 2026-10-10 Asia/Shanghai。新授权来自中央下发的独立有界诊断任务；旧的 `execution_authorized=false` 候选说明只约束旧轮次，不改变本轮 create-only 隔离执行权限。
- 科学来源固定为 `cscwd_causal_prepared_state_visibility_matched_20261010_v1`，commit `f46ec72b15f534ac44768a83ef6316c1cfcb6b58`，必须 terminal PASS、completion complete、119/119 文件 SHA/size 一致；使用 A 已冻结事件链 `auxiliary_probe_rows.csv` 的 **全部 24 条**首次失败映射，不重新筛选。两个 view 的重复记录保留来源身份，不冒充独立样本。只消费已暴露 development，不读 formal/holdout。
- 对照选择在运行任何 action 分支前执行：从 new selected ledger 中按 `(split,design_id,seed,step_index)` 排序，取最多前 6 条 `action4`、当前 complete bundle ready、原 `event_target=1` 且原轨迹 service_completed 与 migration_success 均为 true 的决策。选择只依赖冻结原轨迹与原伪标签，不看分支收益。原始候选/选中名单及 hash 在 preflight receipt 中记录。
- 对 24 条失败及最多 6 条对照，从实例 reset 后只按冻结 ledger 重建目标前缀，不调用 policy。每个目标前缀的**完整可克隆环境状态**（config、instance、RNG state、mask schema、cache resident/LRU、workflow/prefix、clock、deadline、metrics、prepared state 及其他 mutable 字段）连同 observation、semantic_state、mask、run_metadata 做确定性规范化 hash；若 env 字段集合漂移就 fail closed。另记录冻结 SA checkpoint 文件 SHA。实际分支 key=`(full_env_and_policy_input_hash,checkpoint_sha256)`；所有原始行到 key 的映射逐条输出。相同 public hash 不表示完整 env 相同。两个 view/seed 只有在分支 key 相同时才能复用分支；对照与失败分开标识。
- 每个唯一 snapshot 使用 `env.clone()` 分别执行 mask 合法的 action0/2/4。mask 非法一律 `NA`，不调用 env.step，不手工置 ready、搬 cache、推进 DAG 或修改资源。每分支从同一 snapshot 恢复 Python/NumPy/Torch 随机状态及 agent checkpoint，设 `_deterministic_action=True`，后续每一步调用同一冻结 SA checkpoint 的 `agent.act(observation,info)`，只消费当时 public info，按该分支自己的新状态到终止或 `min(instance.max_steps, 原 episode step_cap 24)`。后验 actual suffix 只由环境模拟和事后审计使用，不传入 policy。不得机械重放原动作后缀。
- 硬上限：最多 24 失败源行、6 对照源行、30 唯一 snapshot、90 个 action 分支、额外 2,160 个 env.step 与 2,160 个 policy forward；每分支余下最多 24 step。执行前校验人数、checkpoint、masked actions；超限立即停止并保存已产生输出，不能换样本重跑。无训练、模型生成、下载、外部上传或 B 工作树写入。

## 固定记录和判定

逐状态与逐分支记录 source row、full/public hash、checkpoint SHA、mask、当前/目标 complete bundle、prepared prefix、剩余 deadline、action 及其一步 service、target model admission/stage、state commit、node progress、clock、reward components、model/state/input bytes、recompute；后缀保存逐步 action/info hash/结果，以及终态相对 snapshot 的 completion、on-time、service failure 数、elapsed、三种 bytes、recompute、原 reward 总和。输出原件、映射、所有合法与 `NA` 分支及 manifest hash。任何第一步 action4 与原 ledger 同行不一致，或同一 snapshot 各 clone 的前状态不同，判 provenance FAIL。

先看当前可服务性：受影响分母为 24 条失败映射中原 `event_target=1` 且当前 bundle missing 的**唯一分支 key**；重复来源另报。若其中任何合法 action4 当前服务成功，或合法 action0/2 均不能当前服务成功，则语义门禁 `FAIL`。若原 event 正标签的对照在候选公式下不保留正标签，或其原 action4 不能完成当前服务并提交 state，则正例门禁 `FAIL`。非法 action4 明确 `NA`，不强行开启，也不作为“失败服务”证据。

多步排序不临时调权重：对同一 snapshot 的合法分支，先按 `(workflow_completed, on_time_workflow_completed, -service_failures)` 字典序比较**服务结局**；服务相同再逐项检查 `(elapsed_seconds, model_bytes, state_bytes, input_bytes, recompute_seconds, -original_reward)`，不把不同单位加权相加。另输出完整六维无权重 Pareto 支配/非支配表。若 action4 服务结局严格优于**所有**立即服务成功的合法 action0/2，或 action4 在全指标上仍非支配而不能由可服务替代分支证实多步不劣，门禁为 `MIXED` 并列明反例；不会自动训练。即使立即服务可行性语义通过，若无法建立每个受影响状态至少一个替代分支在服务排序不劣且各成本不高（至少一项严格更好），也只能报 `MIXED / 多步收益不受支持`。只有 provenance、语义、对照、多步四项全通过才给总体 `PASS`；任一硬合同矛盾给 `FAIL`。未受影响的 24 条负例及 6 条对照全部报告，不能删除反例。重复 source 不用于独立性统计。

本门禁最多支持有界可行性与候选合理性，不证明辅助标签是唯一原因、新训练会改善或 SA 优于优秀 baseline。执行中若只发现诊断依赖 bug，可修代码、重新 preflight；已跑出的科学分支原件保留，不能因结果不好改本计划或另换样本。通过与否都向 B 交付 commit/tree/profile、完整报告与 manifest 路径。
