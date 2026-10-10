# CSCWD prepared-state 公共观测修复：A 线交接

- 日期：2026-10-10（Asia/Shanghai）
- 来源科学原件：`cscwd_causal_strong_baselines_budget_extension_20261009_v1`，科学 commit `d25ebcded6b43b69b82bae825b035adc1d6f19c4`
- 来源诊断：`cscwd_sa_long_budget_cost_diagnosis_20261009.md`，A commit `e4598fd147ad19148c0d81f545c1670c2164f284`
- 作用域：共同 observation contract 修复；不是 SA 专属算法贡献。本轮 0 训练、0 新科学评价、0 旧 holdout 读取，论文与论文表未修改。

## 显式 profile 与字段

仅当运行时 `config["interface_profile"] == "calibrated_workflow_interface_v4_prepared_state_prefix"` 时，环境在公共 `semantic_state.calibrated_context.prepared_state_prefix` 添加 `prepared_state_prefix_v1`。旧 v3 profile、九值 raw observation、五动作 mask、reward、transition、物理链路与预测器不变。v4 延续 v3 的 train-only causal-prefix 预测：当前关联 RSU 来源于已到达的当前轨迹前缀，预测目标仅来自公开前缀预测器。B 的 base config 可继续指向已核验的 `configs/experiment/calibrated_continuous_workflow_interface_repair_v3.json`，在完成原 config/manifest hash 核验后显式覆盖 `interface_profile`；A 不改 B 的 runner/protocol。

`current` 和 `predicted_target` 各含 `known`、`exists`、`valid`、`missing_completed_count`、`missing_completed_fraction`。`exists` 表示该 RSU 有已提交的 prepared-state（不是 model resident）。`valid` 与执行时 handoff readiness 同义：相同 workflow，且**当前已完成 DAG 节点集合**包含于 stored prepared-state 前缀。`missing_completed_count = |当前已完成节点 − stored 已完成节点|`；不同 workflow 或无状态时把当前全部已完成节点计为缺失；比例除以 execution order 节点数。不存在公开预测目标时 `known=false`，其余值为零，不能将其解释为“某个真实目标状态无效”。所有值只依赖已发生的 prepare 与已完成前缀，不读取实际未来 RSU、实际未来链路或后验成本。

编码映射：FlatSemanticEncoder 在原 18 维后追加 `current.exists/valid/gap`、`target.exists/valid/gap`、`target.known`，共 25 维；PPO、controller-level MAPPO、DT 均通过该编码器进入 actor/critic，DT 原 14 维 twin 分支不变。SA 的 RSUStateEncoder 在每个 RSU 原 10 维后追加对应 3 值：当前 RSU 取 current，预测目标取 target，其他 RSU 为零，共 13 维；其 current/target/set embedding 进入 SA actor/critic 融合。旧 profile 保留原 18/10 维网络。agent 与编码器对 profile 不匹配 fail closed；checkpoint config 保存 `prepared_state_features_enabled`，新旧互相 load 时在权重写入前报错。

| 方法 | 旧参数 | 新参数 | 增量 | 单线程 CPU 单次 forward 旧→新，毫秒 |
| --- | ---: | ---: | ---: | ---: |
| SA-GHMAPPO | 165,320 | 165,512 | +192 | 0.3746→0.3716 |
| MAPPO | 39,176 | 39,624 | +448 | 0.1149→0.1171 |
| PPO | 22,406 | 22,854 | +448 | 0.0808→0.0833 |
| DT-Handoff-PPO | 44,808 | 45,256 | +448 | 0.1484→0.1523 |

Forward 时间是本机 `torch.set_num_threads(1)`、10 次 warmup、3×100 次单状态 forward 的中位数，仅记录输入宽度开销，不是稳定性能比较。+192/+448 由首层线性投影的 `3×64`/`7×64` 权重得到；核心 actor、critic、auxiliary、mask、reward、数据 split 不变。因此 B 的匹配比较是“共享观测 + 输入投影宽度”的共同改动，不得称纯输入单变量或 SA 创新。

## 最小验收与边界

原 B run 的 `regression_08` 第 4 步，SA seeds 17/43 的已记录动作前缀分别为 `3,3,4,4` / `3,3,0,0`。旧公开 hash 两者同为 `86206ccfda557199bb67a41c0d3a7bcfc3b98a3817aa7de3875e71e6c971d62d`，同选 action 0，实际重算 `0` / `30.21069466716467 s`；新公开字段在动作前分别为 current `exists=true,valid=true` 与 `false,false`。旧新 profile 对两条历史及 action 0 的物理 transition、奖励、成本完全相同；另有混合合法动作序列逐步等价。

合成测试覆盖无状态、valid、实际节点推进后 stale、错误 workflow、prepare 失败、当前/目标不同 RSU、预测目标改变、无预测目标、未来实际后缀及实际 link 扰动不改变当前公开输入、重复序列化稳定、四方法特征/embedding 扰动、rollout/recompute executed logprob 对账、四方法 save/load 与旧 checkpoint 拒绝。命令：

`/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q tests/test_calibrated_workflow_prepared_state_prefix.py tests/test_calibrated_workflow_interface_repair.py tests/test_calibrated_workflow_strong_baselines.py tests/test_calibrated_workflow_service_reward.py tests/test_env_contract.py`（40 passed）；`/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/smoke_test.py`（passed）；`python -m py_compile` 与 `git diff --check`（passed）。B 科学原件四个 seed 7 selected checkpoint 均可由旧 profile 读取，v4 agent 对同一 checkpoint 四次均在写权重前拒绝。

当前仍只在已消费开发实例上有接口证明；B 需独立 preflight 后才可按其事前协议训练。任何服务或成本优势、独立验证、真实无线及 paper-ready 结论均未由此确定。
