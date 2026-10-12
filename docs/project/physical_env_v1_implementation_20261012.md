# 物理环境 v1：实现与验证边界（2026-10-12）

本页记录本轮代码落地位置和可运行入口。`physical_vec_env_v1` 是显式选择的新环境身份；旧 `VecWorkflowCoreEnv`、正式 Protocol 2.9、历史 checkpoint 与 formal/holdout 结果不自动转入新环境。本文所称物理事件指受容量、带宽和计算槽约束的离散仿真事件；默认参数是 `synthetic`，不是实测 RSU 性能。

## 工作包到代码与验收

| 工作包 | 实现位置 | 本轮可检查结果与边界 |
| --- | --- | --- |
| M01 运行与接口 | `src/runtime/physical_env_config.py`、`configs/environment/physical_env_v1_synthetic.json` | 秒、字节、work unit 单位显式校验；配置有 canonical SHA-256；版本由运行入口携带。 |
| M02 事件内核 | `src/envs/core/physical_runtime.py` | 跨 `step()` 的传输与计算、共享链路和计算槽、取消及确定性事件顺序。 |
| M03 原生缓存 | `src/envs/core/physical_cache_service.py`、`physical_vec_workflow_env.py` | typed 对象、依赖、预留、在途加载、驻留与运行锁定接入新核心；当前驱逐限 LRU。 |
| M04 成本与回放 | `src/runtime/physical_cost_profile.py` | 成本来源和 model/input/generation 身份合同；单节点输出在模拟完成后解锁。真实 cold load、adapter switch、state restore 的测量样本仍缺。 |
| M05 轨迹拓扑 | `physical_vec_workflow_env.py`、既有 NGSIM provider/RSU mapper | 回放帧驱动接入和切换；固定两站场景验证。真实数据入口需提供有效本地 NGSIM 文件。 |
| M06 共享资源 | `physical_runtime.py`、`physical_vec_workflow_env.py` | 模型、输入、状态和结果传输记入有限链路；计算槽排队。固定第二车背景负载改变主请求完成时间。 |
| M07 DAG | `src/envs/core/dag_execution.py` | 前驱输出身份、目标位置、模型就绪、计算排队分开管理；后继等待真实前驱完成。 |
| M08 切换交付 | `dag_execution.py`、`physical_vec_workflow_env.py` | 接入变化与运行位置分离；旧站续算、结果转发和无线交付有事件与 receipt。当前状态转移是任务边界显式输出对象，不等同任意 GPU/KV 进程状态。 |
| M09 预测 | 既有 `src/envs/core/predictor_manager.py` 与新核心预测映射 | 输出下一站候选、handoff ETA、置信度与未校准的候选分布；准备动作使用目标预测。概率校准、误差和决策影响尚需真实多窗口报告。 |
| M10 策略适配 | `src/envs/wrappers/gym_vec_env.py`、`src/envs/specs/action_schema.py`、训练入口 | 五动作共用 wrapper；物理观察编码为版本化 33 维，含最多四站资源摘要；加载中对象不能重复准备。现有 checkpoint 不兼容新编码，必须重训。 |
| M11 账本指标 | `src/metrics/physical_event_metrics.py` | 请求、操作、对象、资源、节点与交付 ID 独立重算；未完成请求 latency 为 unavailable；事件字节守恒校验。 |
| M12 场景回归 | `scripts/run_physical_env_acceptance.py`、`scripts/benchmark_physical_env.py` | 固定场景检查确定性、争用、跨步等待、DAG 与切换；独立多窗口多 seed 非正式入口保存原始事件、指标和身份。 |

## 运行入口

固定工程验收：

```bash
python scripts/run_physical_env_acceptance.py --output-path /tmp/ppo_mec_physical_acceptance.json
python -m pytest -q tests/test_physical_runtime.py tests/test_physical_cache_service.py tests/test_dag_execution.py tests/test_physical_vec_workflow_env.py tests/test_physical_event_metrics.py tests/test_physical_env_acceptance.py tests/test_benchmark_physical_env.py
```

真实 NGSIM＋Alibaba 的非正式比较入口是 `scripts/benchmark_physical_env.py`。必须显式提供 `--physical_env_config`、`--window_plan_path`、`--mobility_csv_path`、`--workflow_csv_path`、`--agents` 和 `--seeds`。学习型策略还需要 `--checkpoint_manifest_path`，每个 checkpoint 的 `environment_version` 和 `physical_environment_config_sha256` 必须匹配。逐 episode 的 `events.json`、`physical_metrics.json`、`summary.json`、`identity.json` 和总表保存在新的 run 目录。该入口标记 `formal=false` 与 `paper_claim_ready=false`。

```bash
python scripts/benchmark_physical_env.py \
  --physical_env_config configs/environment/physical_env_v1_synthetic.json \
  --window_plan_path <frozen_nonformal_window_plan.json> \
  --mobility_csv_path <complete_local_ngsim.csv> \
  --workflow_csv_path <complete_local_alibaba.csv> \
  --agents reactive_greedy --seeds 7 \
  --output_root artifacts/physical_env/nonformal_benchmark
```

训练入口 `scripts/train_algo_pool_real_sample.py`、`scripts/train_sa_ghmappo_real_sample.py` 可通过 `--physical_env_config` 显式选择新核心；旧正式训练合同及 checkpoint 不作为新环境的身份依据。新 checkpoint 记录物理配置哈希。旧 `scripts/benchmark_main_results.py` 保持原正式主表语义，不用于评估新环境。

## 当前不能据此宣称的内容

- 参数文件为合成示例，尚无真实模型 load/hot call/switch、状态恢复、无线与回传链路校准。使用它得到的时间与收益不能作真实部署或论文算法优势结论。
- 单节点回放合同只证明输出在匹配身份且模拟计算完成后可见；多节点真实推理输出、任务语义正确性和动态 KV/tensor 状态未覆盖。
- 预测候选分布由 horizon 序列频率构成，明确标为 `uncalibrated_horizon_vote_share`；不能当作校准后的下一站概率或以此证明预测收益。
- 至多四个 RSU 进入当前固定观察向量；更多节点需要冻结新的观察/动作及 checkpoint 合同。
- 现有长尾 H2/H4、五策略公平性、城市规模和独立外测需按新环境重建输入与 checkpoint，不能沿用旧 formal/holdout 或 smoke 结果。对照蓝图 A01–A34 的逐项签收仍需取得原修订版文件；目前只按用户提供的工作包摘要映射。

## 本轮验收判据

只有固定场景、独立事件重算及新入口身份校验通过，才能称 `physical_vec_env_v1` 工程可运行。只有实测成本、真实模型输出身份、多窗口多 seed 公平比较、原始 frame/time 互斥与独立外测齐备后，才能另行审查研究结论；审查按 `top_journal_review_policy.md` 执行。

## 2026-10-12 本地验证记录

- 物理模块、benchmark 合同、DAG 与旧环境接口回归：`python -m pytest -q tests/test_physical_runtime.py tests/test_physical_cache_service.py tests/test_physical_vec_workflow_env.py tests/test_physical_event_metrics.py tests/test_physical_env_acceptance.py tests/test_physical_env_config.py tests/test_benchmark_physical_env.py tests/test_dag_execution.py tests/test_env_contract.py`，77 passed；`python scripts/smoke_test.py` 通过。
- 固定两站两车验收六项检查通过。新 benchmark 从主工作区完整 NGSIM/Alibaba CSV 只读取样；规则 agent 的 1 seed、1 window、1 workflow 合成快速配置 episode 完成，事件和独立指标可重算。
- PPO 和 SA-GHMAPPO 各完成 1 episode 的物理环境 smoke 训练并生成版本化 checkpoint；两者的 checkpoint 均经新 benchmark 身份核验并实际加载运行。SA 固定样本完成；PPO 在 12 step 预算内截断，主请求 latency 明确为 unavailable。这些都是工程联调，不是训练充分性或算法比较。
