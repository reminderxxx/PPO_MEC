# CSCWD base sharing × state migration：执行前独立差距审查

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11；本轮只查本地源码与元数据，不作 novelty 判断
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `none`（只读合同审查，0 个环境步骤）
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `425d6944f7e2eee43047f99ff63abcb8df85a981`（审查输入）
- `evidence_level`: `E1_DOCUMENTED` 用于论文主张；本报告的源码路径已逐项核对
- `verdict`: 四臂尚不可直接公平执行；sharing×migration 交互 `UNVERIFIED`，paper-ready `Unverifiable`

本报告在任何新的 sharing/migration 实现之前形成。旧 phase-v2 v3 结果是一条 US101 开发轨迹的动作分支，不能当作四臂实验。`ample_cache` 同时改变容量和初始 resident，不能作为 sharing-on；旧 `distinct_base` 设计使用人工不同模型身份，不能当作保持同一模型功能的 sharing-off。

## 生产端与消费端现状

| 轴 / 约束 | 已核对路径和事实 | 缺口 |
|---|---|---|
| 共享 base 资源 | `calibrated_continuous_workflow_pilot_v2.json` 的 `alpr`、`helmet_shared` 均依赖 `base:family_a`；`CalibratedContinuousWorkflowEnv._bundle_ids/_admit_bundle/_validate_cache` 按对象 ID 去重、按 resident/transfer bytes 和 load seconds 计费；`RawNGSIMPhaseV2Env` 沿用同一 cache 交易。 | raw phase v2 没有显式同一 base 权重按 adapter 私有副本驻留/传输的 opt-in；直接改容量或换 `family_b` 会混杂功能和资源。 |
| 初始资源与 episode | parent `reset` 从 `instance.initial_residents` 重建每站 cache；phase v2 `ample_cache` 另把全部 bundle 装入目标。 | 两臂初始 resident 必须按同一逻辑对象映射，逐副本计真实 resident bytes，并对账容量；不可把重复份额当免费。跨 workflow 持久缓存仍不可用，因此只能检验单 episode 内复用。 |
| 迁移与重算 | `RawNGSIMPhaseV2Env.step` 的 action 4 同时可能准备目标 bundle、当前节点执行、状态传输、restore 与目标交付；`transition` 保留 model/state/input bytes、step cost、recompute、完成/失败。`VecWorkflowCoreEnv` 另有 opt-in `workflow_state_migration.enabled`，但不是本 raw phase v2 的四臂开关。 | raw phase v2 没有保持 action 4 合法 cache-prepare/当前执行、只关状态交付的 M0 合同；M0 必须让后续 DAG 使用原有合法重算，不能跳过工作。 |
| 信息与策略 | raw `_info()` 输出 `semantic_state` 和 `action_mask`；`calibrated_context` 含 bundle IDs、容量、cache resident、公开估计；实际 trace 后缀仅供执行。 | 四臂需锁同一 ex-ante 固定、非特权策略与决策预算；因素引起的公开 cache 观测差异可以被策略看到，未来后缀、事后交付状态不得入策略。 |
| formal 消费端 | `typed_model_cache_formal_execution.py` 将 `no_base_sharing` 标 `partially_available`；`base_sharing_degree` 写 `unavailable_pre_execution`。 | 本轮即使有 raw phase v2 开关，也不自动接入 Protocol 2.9 formal；正式 ablation 状态不应擅改。 |
| 实物与模型功能 | 配置的 base 文件足迹与 load 时间来自同一已记录 SmolVLM 资源，`family_b` 是人工 distinct identity；raw phase v2 未执行真实 model call。 | 私有副本只能表示同一不可变 base 字节的独立驻留/传输，须带源对象、相同字节/加载参数、依赖与任务一致性的可审计映射；任何模型质量等价仍未由本轮真实调用验证。 |

## Peachtree / I-80 元数据门

只读检查本地 NGSIM CSV 首行的 `Location=peachtree`、`Global_Time=1163368300`，相邻 `Frame_ID=3380/3379/3378` 对应时间差 100；它是 **10 位**值。现有 raw-time 合同只接受已核验的 13 位毫秒 US101/Lankershim。仅凭相邻差 100 不能确定 Peachtree 的绝对单位、跨来源时钟或正式 split 身份，因此 `peachtree` 暂记 `unit_ambiguous`，不得擅自 `/1000` 解释并入新开发或独立泛化来源。v28 split metadata 含 Peachtree train/dev/formal/hidden 区间，不能从新 window ID 推断无暴露；必须以原始 frame/time 闭区间做交叠审计。此轮不读任何 sealed performance。当前 typed Protocol 2.9 index 元数据仍显示 `opened=false`、`consumed_permanently=false`；I-80 seal 不变。未下载、训练或执行模型。

## 冻结的后续四臂合同（本轮不执行）

设 `S=1` 为同一 base 在兼容 adapter 间共享，`S=0` 为每个 adapter 持有同一 immutable base 字节的私有副本；`M=1` 为完整状态交付可用，`M=0` 为 action 4 仍可合法准备模型并服务当前节点，但状态不得交付，后续按同一 DAG 合法重算。必须同一原始轨迹/闭区间、车辆、workflow 输入与节点顺序、adapter 功能、物理链路、cache 容量、初始逻辑资源、五动作/掩码规则、公开观测权限、固定策略、停止条件与预算；只允许因素引起的真实 resident/transfer/load 与状态/重算结果改变。对四臂保留所有零收益、拒绝、失败与截断。

按每个独立原始轨迹区间配对记录 `workflow_completed`、`on_time_workflow_completed`、`service_failures`、`model/state/input/backhaul bytes`、`transfer/model_load/state_restore/recompute/step_cost/completion seconds`、action/mask、cache 交易和状态交付。仅在四臂相同终点、无截断且 endpoint 可比时计算各原始指标的交互 `I=(Y11-Y10)-(Y01-Y00)`；主效应或 reward 差不是交互。不同条件、seed、同源重叠窗口和 workflow 行均不能冒充独立外层 cluster。若四臂中任一 arm 丧失合法模型/任务或容量导致不可比，报告 `interaction_unestimable`，不删除该来源。

最小下一轮预算若获得独立执行门：**1 条事前未用的 development 原始轨迹 × 1 个事前固定且至少含两个兼容 adapter 的 DAG × 4 臂，合计最多 4 episode、96 个真实环境步，0 训练、0 model call、0 formal/holdout**。执行前需有含原始区间、输入 hash、同功能私有副本映射、初始资源账、固定策略/动作权限、完整终点、create-only 输出 root 和单次运行命令的独立 manifest；缺任一项不得启动。这个预算只允许合同验收，不足以支撑统计效应或论文主张。

## 四项主张的当前出口

| 主张 | 当前可做 | 硬资源 / 缺失证据 |
|---|---|---|
| 真实跨 RSU 净收益 | 只可验证模拟相位/字节账本 | 至少两台独立 RSU 主机和可审计链路/队列/时钟实测、真实 payload 与配对执行；本地模拟器不等同真实部署。 |
| base sharing × state migration 因果交互 | 先做同功能开关、合成验收与单次 development 四臂可行性 | 独立原始区间、多 adapter 机会、完整配对终点、正式统计和消费者接入均缺。 |
| 三头优于公平优秀基线 | B 线可独立做匹配开发 pilot | 匹配训练/权限/预算、冻结 checkpoint、正式多来源评估尚缺；A 本轮不编辑或评价 B 的运行。 |
| 独立泛化 / paper-ready | 可先核元数据与区间身份 | Peachtree 单位和暴露未定，I-80 seal 不开；正式 checkpoint、manifest、command log、formal/holdout/support 仍须完整核对。 |

**Hard blockers**：现有 raw phase v2 两个独立因素尚无对称开关、Peachtree 身份与独立性未核定、真实主机/链路及正式证据缺失。**Major concerns**：同一 US101 开发来源重复、模型功能仍是抽象/半合成成本。**Minor concerns**：待未来实现后更新准确 producer/consumer schema 与诊断统计。**Scorecard**：TMC 七维均 `N/S`，论文原件不足不打伪精确分。**Safe claims**：仅可称本地合同缺口已定位。**Prohibited claims**：交互为正、模型功能已由真实推理确认、真实 RSU 净收益、独立泛化或 TMC-ready。**TITS/TVT fit**：不因换 venue 放宽证据门。
