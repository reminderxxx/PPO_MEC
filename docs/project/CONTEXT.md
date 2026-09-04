# Context

更新日期：2026-09-05

用途：记录 PPO_MEC 当前稳定上下文。这里写长期有效事实，不写单次运行细节。

## 项目状态

- 当前唯一 live typed model-cache execution contract 是 Protocol 2.4.0 + Formal Protocol Capability Routing
  Contract 1.0.0；active index 位于
  `configs/experiment/typed_model_cache_formal_protocol_v2_4_20260905/protocol_index.json`。v1.0–v2.3均为
  audit-only，未知版本 fail-closed，active nested preflight 必须消费 outer 持久化 resolved context，不能回退
  default context。Readiness v16只授权未来独立 G14C v14；当前 formal training/checkpoint/performance=0，
  holdout sealed/unopened/unconsumed，不是 formal、G14、TMC 或 paper-ready 证据。

- G14R7A active formal execution baseline：Active Formal Bundle Contract `1.0.0`，Protocol v1.8 semantic
  SHA-256 `9799bf2c2f4b4665b8390c6fc5d5aa235faf11d6525e043eac289c061633b3de`，bundle core/final SHA-256
  `96627ac414cb5dc80785c907ded2c9588dcdcf69469a5821b75fc07dc25e5b65`/
  `793f5106b83f9687044aeeac122179a8c5805688d4a041c0418292345f9138bd`。Readiness v10=
  `READY_FOR_G14C_V8_CLEAN_TRAIN_AND_FORMAL`与ready index原子一致；outer runner在任何run-root写入前从
  唯一index验证完整bundle及clean `HEAD==origin/main`。v1.0–v1.7 audit-only；正式training/checkpoint/
  performance=0，holdout sealed/unopened，未启动G14C v8/G14D/G15，不是formal或paper-ready。

- G14R7 active formal execution baseline：Protocol v1.7 semantic SHA-256
  `5a1c2070529674ecf65c8b836706849f0937853a59b6dfbc3b987d88ac4f50a5`，Formal Agent Order Contract
  `1.0.0` semantic SHA-256 `82e562755dadd4341c950bf71efc488d3527b7f45b7f02512f8064d189b655e0`。
  main agent order严格为5 reactive + 10 learned；mapping key order不构成身份。Scientific config hash与
  dependency fingerprint未变。Readiness v9=`READY_FOR_G14C_V8_CLEAN_TRAIN_AND_FORMAL`仅授权未来独立
  clean run；当前v1.7正式training/checkpoint/performance=0，holdout sealed/unopened，不是formal或paper-ready。

- G14C v7 `typed_model_cache_formal_20260826_233222_g14c_v7`永久无效：150/150 training与1,200 candidates
  已完成，但在dev performance前因mapping order/fairness order冲突失败；dev rows/selection/freeze/formal全为0，
  checkpoint、candidate、partial dev input、ledger与marker均禁止复用。

- G14R2 formal execution baseline：Protocol v1.2 semantic SHA-256
  `718c0f78aabd5d01012df31267626eab74a51b2b621aaa67a535c5b60e655ca9` 已冻结；window contract
  `1.0.0` 将原 60-window split 绑定到完整 11,850,526-row NGSIM prefix、73,871 provider frames 与
  raw frame/time/fingerprint identity，60/60 reachable。Split semantic SHA-256 仍为
  `aa9a7400da2b424d0b1bcd6f1cbfc0a9dd6cfa10e02e847523245afa6608d76a`。150/150 training 与
  30 条 dev/formal/support commands 通过无训练审计；ledger `2.0.0` 完整记录时间、wall-clock、失败枚举
  与 hash chain。Readiness v4=`READY_FOR_G14C_V3_CLEAN_TRAIN_AND_FORMAL`，仅授权未来从 Commit A3
  clean worktree 另立任务。当前 formal checkpoint/episode/performance count=0，holdout sealed/unopened，
  不代表 G14 或 paper-ready 完成。

- G14C v2 `typed_model_cache_formal_20260820_164251_g14c_v2` 永久无效：Protocol v1.1 train command
  未传 source range，默认 1500 raw rows 使 frozen provider offset 在首个 training cell 前不可达。旧 run
  0/150 training、0 checkpoint、0 formal、holdout unopened，禁止 resume/覆盖/删除。

- G14B formal protocol freeze：历史账本 `1.0.0`、split protocol `1.0.0` 与 formal protocol `1.0.0` 已冻结；train/dev/formal/sealed-holdout=`24/12/12/12`，60 个 outer windows 的 1,770 对 raw frame/time/segment-run 审计全部 safe，minimum gap=24。Readiness v2=`READY_FOR_G14C_CLEAN_TRAIN_AND_FORMAL`，evidence level=`E2_PROTOCOL_AND_CONTRACT_VALIDATED_NO_PERFORMANCE_DATA`。Holdout 仍 sealed/unopened；正式 checkpoint、formal result 与 paper-ready evidence 均不存在。G14C 必须从包含 protocol semantic SHA-256 `41fbfab4ac10bae96250d7ead816d907fd6551bb9651ae03210e801c9e2478b4` 的 Commit A clean worktree 另立任务执行。

- G08 live contract：`cache_request_replay_version=1.0.0` + `future_horizon_cache_oracle_contract_v1.0.0`。输入必须是独立policy-neutral DAG/mobility replay；oracle只控制current-RSU placement/admission/eviction，并按当前环境同step admission可命中的真实时序运行exact rolling H=1/3/6/12。matched gap仅为placement opportunity gap，不是causal regret或latency gain；当前validation未运行formal/holdout/hidden。

- 项目：`PPO_MEC`
- 定位：面向 AI-driven VEC 的研究原型
- 主线问题：跨 RSU 连续 DAG workflow 执行、车载 base model 与路侧 adapter cache 协同、handoff 状态迁移、多时间尺度控制
- 当前正式数据主线：`NGSIM + Alibaba`
- 当前数据源声明入口：`docs/project/DATASET_SOURCES.md`
- 当前外部 model-cache registry：G11 `1.0.0` 已核验19个候选；未发现joint VEC或真实adapter request trace。BurstGPT/Qwen-Bailian/Mooncake/Azure/Acme与三个HF size源仅metadata-only声明，不替换benchmark默认cache行为；HF live catalog只保留qwen/cbow/bert三项E类size metadata。
- 当前论文协议：`paper_protocol_v1_20260409`
- 当前正式结果入口：`docs/project/ARTIFACT_RECORDS.md`
- 当前顶刊审查规范：`docs/project/top_journal_review_policy.md`；最新审查为 `docs/project/top_journal_readiness_audit_20260621.md`。
- v7 的 strict-full blocker 已由 v8 修复：冻结 20-window/split 协议下，formal/一次性 hidden 对全部 learned baselines 的 reward CI 为正，对 DT continuity 的 CI 也为正。当前 reviewer verdict 为 `Major revision (78/100)`，证据等级为 `E2_ARTIFACT_AUDITED`；PPO handoff failure/backhaul trade-off、v8-current support suite 与外部样本量仍未达 TMC-ready。
- 2026-07-21 v39-v41 dev-probe 显示：MAPPO-core / delayed-credit / advantage-weighted behavior regularization 可以让 SA-GHMAPPO 在 frozen dev 单 seed/20-window 上略高于 MAPPO、PPO 和 popularity heuristic，但仍未超过 `cache_offload_drl` 与 `dt_handoff_drl`。当前 `reward_positive_offset=5.0` 会按 step 累加，存在 reward ranking 与 workflow completion/continuity 不一致风险；v39-v41 不替代 v8/v20 记录，也不是 paper-ready / all-baseline-winner 结论。
- 当前 live 模型层：主方法 `sa_ghmappo` + 方向匹配型对照算法池；`mappo` 对照采用 controller-level CTDE + `aggregation_reason_weighted_controller_ppo_v3`。
- 当前 predictor 层：默认仍可使用 `baseline_predictor_v2`；代码已新增 `supervised_handoff_predictor_v1` 训练与 runtime 接口，需显式传入冻结 checkpoint。该层定位为短时 handoff anticipation / lightweight DT-style predictive state snapshot，不是完整数字孪生系统；在生成正式 checkpoint、quality report 和 v9 benchmark 前，不自动替代 v8 主结论。
- 当前新增后 live paper-grade learned 对照算法池：`ppo`、`mappo`、`dqn`、`dueling_dqn`、`qmix`、`controller_mat`、`dag_offload_drl`、`cache_offload_drl`、`dt_handoff_drl`
- 当前 optional learned 变体：`ddqn`、`dueling_ddqn`；只有 duplicate trace audit 证明独立后才能作为补充对照。
- 当前 diagnostic / contract-blocked 对照：`ippo`；当前 single-wrapper contract 不支撑独立 IPPO。`mappo` / `qmix` / `controller_mat` 已实现为 controller-level CTDE / value-decomposition / transformer baselines，不是 vehicle-agent / RSU-agent full MARL wrapper。
- 当前非学习启发式对照：`reactive_greedy`、`popularity_cache_heuristic`
- 历史 artifact 路径中仍可能出现 `flat_ppo` / `flat_mappo` run 名称，但它们不再作为 live agent 注册。
- 当前未注册骨架算法：`td3`、`sac`、`maddpg`；后续接入前必须先冻结匹配的 observation/action contract
- 当前 typed model-cache active execution contract 为 Protocol 2.3 + Nullable Metric Aggregation Contract 1.0.0。
  G14C v12 因 nullable consumer 实现错误永久无效；其 256 episodes、8 staging candidates 和 latest checkpoint
  均不是正式证据。G14R12 readiness 只授权未来全新 G14C v13，不代表 formal/G14/TMC/paper-ready。

## 正式入口

- 数据检查：`python scripts/check_data_ready.py`
- NGSIM 检查：`python scripts/run_ngsim_sample.py --max_rows 500`
- Alibaba 检查：`python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20`
- 真实 dry-run：`python scripts/run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12`
- 主结果：`python scripts/benchmark_main_results.py`
- Baseline 闭环：`python scripts/run_baseline_experiment.py --config configs/experiment/baseline/smoke.yaml`
- 对照算法训练：`python scripts/train_algo_pool_real_sample.py`
- 对照算法评估：`python scripts/eval_algo_pool_real_sample.py`
- 消融：`python scripts/benchmark_ablation.py`
- 鲁棒性：`python scripts/benchmark_robustness.py` / `python scripts/benchmark_prediction_robustness.py`
- Supervised predictor：`python scripts/train_supervised_handoff_predictor.py`
- 可扩展性：`python scripts/benchmark_scalability.py`

## 当前可引用结论边界

- 文档化结果：`final_submission_v7_latency_fallback_20260528_v1` 是 legacy paper-ready package；`final_submission_v7_latency_fallback_20260618_rebuild_v1` 已复现旧协议数值，但 strict non-overlap 结果取代 offset-3 作为 readiness 判断依据。
- 当前主结果 claim 边界：可安全表述 v8 strict-full formal/一次性 hidden 对全部 learned baselines 的 reward CI 为正，且对 DT continuity CI 为正；必须同时报告相对 PPO 的 handoff-failure/backhaul trade-off及未超过 popularity heuristic。`mappo` / `qmix` / `controller_mat` 是 controller-level baselines，不是 vehicle-agent / RSU-agent full MARL wrappers。
- 当前 dev-probe claim 边界：v39 update_0005 full-pool SA-GHMAPPO `106.041` 高于 MAPPO/PPO/popularity，但低于 DT/cache；v41 conservative recovery `105.686` 只恢复稳定性，没有扩大 MAPPO 差距。不得把这组结果写为 all-baseline winner、canonical 晋级或投稿主 claim。
- 注意：历史记录可包含旧对比算法；这些只代表归档结果，不代表当前方向匹配算法池的 live 结果。
- 可引用但需说明限制：早期 robustness 最新保留记录，协议早于 frozen main table，不应单独支撑最终主张。
- 不再引用：toy benchmark、tmp quickcheck、LuST micro 激活窗口试验、早期 dry-run、阶段性 reward shaping / recalibration / uncertainty tuning。

## 当前保留原则

- `artifacts/paper/` 只保留当前 canonical paper record。
- `artifacts/benchmarks/` 只保留主线可引用报告和每类最新有效报告。
- `artifacts/training/` 只保留被保留 benchmark 引用的 checkpoint run。
- `docs/project/` 是唯一长期文档目录。
- `maintainable_engineering_docs(1)/` 和旧 `docs/*.md` 阶段文档不再作为事实来源。
