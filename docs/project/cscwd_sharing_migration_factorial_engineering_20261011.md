# CSCWD sharing × migration 最小 opt-in 合同：合成验收

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11；本轮未检索新论文或评定 novelty
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `none`（仅合成单元验收，0 原始 NGSIM 环境步）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: 提交前输入 `425d6944f7e2eee43047f99ff63abcb8df85a981`；本报告与实现一并提交
- `evidence_level`: 合成合同验收；论文主张仍 `E1_DOCUMENTED` / `Unverifiable`
- `verdict`: **四臂环境因素可显式构造，正式四臂科学执行未获许可且未启动**

前置只读缺口见 [独立差距报告](cscwd_sharing_migration_factorial_gap_20261011.md)。本轮只改 `RawNGSIMPhaseV2Env` 的显式 `factorial_profile`；缺省 `None` 走原 v2 路径，原配置、已有矩阵、旧 artifact、`VecWorkflowCoreEnv` 和 formal Protocol 2.9 均未修改。

## 因素和生产/消费合同

调用者必须同时给 `base_sharing_enabled`、`state_migration_enabled` 两个布尔值。S0 在 parent 初始化之前，对每个 adapter 将同一原 base 对象按 ID 复制为私有 resident；每份保留原 `resident_bytes`、`transfer_bytes`、`load_seconds` 和同一 `measured_identity.base_model`，adapter dependency 改指自己的私有副本。DAG 节点、输入、要求的功能 base ID、adapter、执行顺序和五动作均不改。初始 cache 只从原 base+adapter 成对 resident 映射；无配对、无固定模型身份或容量不足即拒绝，不免费挤入、不静默清理。共享与私有的额外 resident、网络和 load 成本由原生 `_admit_bundle`、dependency-safe eviction、phase ledger 消费。

M0 仍允许 action 4 的公开掩码、目标模型准备与当前节点执行；目标准备和当前服务均成功后写 `migration_status=disabled_by_factor`，状态发送/restore/交付为零，之后跨站按原环境规则重算 DAG 前缀。目标准备或节点服务失败则保留原失败状态与字节，M0 的 action 4 不计 migration attempt/failure，服务失败照计。`semantic_state.calibrated_context.factorial_profile`、每步 `transition.factorial_profile` 与 `summary.factorial_profile/private_base_sources` 给后续消费者绑定因素身份；默认调用不新增这些字段。新增 catalog 私有对象的 `source_base_object_id`、`model_function_identity` 可核对来源。这里的“同一模型功能”仅是**同一抽象模型资源与任务映射**，本轮不进行真实推理或质量等价测试。

合成检查覆盖同 DAG/输入/五动作 mask、同初始逻辑 resident 与容量、第二兼容 adapter 进同站时 S0 相比 S1 多计 **1,015,025,832 bytes** base 传输和驻留、M0 action 4 仍完成当前节点/准备同量目标模型而不传状态、下一跨站步骤产生正的重算时间、失败准备保留状态与 partial bytes、四臂相位时间/字节守恒、默认 profile 逐事件回归，以及缺身份/孤立 resident 与私有初始副本超容量拒绝。命令：`/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest tests/test_cscwd_sharing_migration_factorial.py tests/test_raw_ngsim_phase_v2_env.py tests/test_cscwd_region_v3_evidence.py -q`，`23 passed`；仅 synthetic trace，无原始数据、模型调用、训练或 formal/holdout。

## 下一科学执行门保持关闭

机器门位于 `configs/experiment/cscwd_sharing_migration_factorial_gate_20261011.json`，`execution_authorized=false`，source 原始区间、公开策略消费者 hash 与 create-only root 仍为 `null`。同一 `dev_01` 两兼容 adapter DAG、同一 2,194,117,040-byte 容量/初始逻辑资源、默认模拟链路、固定公开 `action_mask` 策略及四因素臂已明写；但独立未暴露轨迹和消费者身份未锁，任何执行器不得仅凭构造器可运行就启动。若后续获得独立授权，仅能先按该门做 1 原始 development 区间 × 1 DAG × 4 臂，最多 4 episode/96 real steps 的一次合同验收；必须先通过 raw frame/time 闭区间、车辆连续性、历史 train/dev/formal/hidden/consumed 暴露与 Peachtree 单位检查。I-80 seal 不打开。若容量、初始资源或状态终点不可比，则报告不可估计，不替换来源或选择有利条件。

正式消费者仍将 `no_base_sharing` 标为 `partially_available`，因为该 opt-in 仅在 raw phase v2；不把合成四臂当正式 2×2。真实跨 RSU 主机/链路、跨 workflow persistence、模型输出质量、独立来源统计、匹配 learned baseline checkpoint、formal/holdout/support 均缺。论文四项贡献保持 `UNVERIFIED`，七维 scorecard `N/S`；安全表述仅为“局部环境因素合同已合成验收”，禁止声称交互为正或 paper-ready。
