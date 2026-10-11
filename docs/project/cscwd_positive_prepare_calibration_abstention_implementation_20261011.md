# CSCWD 正 prepare 校准来源 abstention：最小实现与冻结 A/B（2026-10-11）

## 身份与结论

- `reviewed_at`: 2026-10-11；`literature_cutoff`: 2026-10-11；`target_venue`: IEEE TMC；
  `policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `cscwd_positive_prepare_calibration_abstention_static_audit_20261011_v1`；实现 commit：
  `f500f5ef1e96f3c39b0a99f2a93adb6655c6b75f`；基线审查 commit：`a34100a2290c16022cd04a36846499355f93c794`。
- `evidence_level`: `E2_IMPLEMENTATION_AND_STATIC_RECOMPUTE / no training`。新增 environment steps、optimizer steps、
  checkpoint reads、formal/holdout reads 均为 0。
- 结论：已实现默认关闭的唯一候选 `calibration_provenance_positive_prepare_abstention_v1`。它只在旧 public-causal
  label 已判为 `prepare` 时要求 `causal_provenance.confidence_calibrated is True`；缺失、`false`、字符串 `"true"`、整数
  `1` 均改为 `abstain, supervision_weight=0`。`serve`、slow/fast auxiliary、PPO、reward、mask 和推理权限不变。
- 这属于学习设计候选，不是已证实的算法改进或实现纠错。当前 Raw producer 对所有状态固定输出
  `confidence_calibrated=false`，故在现有训练分布上它等价于**撤销全部正 prepare 辅助监督**，不是细粒度 uncertainty control。

## 实现消费链与身份

| 环节 | 已实现合同 | 验收 |
|---|---|---|
| label producer | opt-in 参数只拦截已判定的正 `prepare`；literal `True` 才放行 | missing/false/string/int/true 与 serve 回归通过 |
| SA training consumer | event CE 与 temporal margin 共用 gated target/weight；slow/fast 不变 | 未校准样本 event logit 梯度归零，slow/fast 梯度逐元素相同 |
| config/builder | `mechanism_aux_positive_prepare_calibration_abstention_enabled=false` 默认兼容 | builder 实际传入 SA；未开启 base public-causal 时 fail-closed |
| checkpoint | 保存独立 flag 与 `causal_public_prepare_advantage_calibration_abstention_v1` 语义 | 新旧语义交叉加载拒绝；旧缺省 checkpoint 按 false 兼容 |
| logs | update receipt 与逐 transition training signal 均记录 gate 和实际 label/weight | 原臂 `prepare/1`，新臂 `abstain/0` 回归通过 |
| runner | 原 `sa_public_v3` 对新 gate 单变量；PPO/MAPPO/public rules 身份不变 | 配置、源码 hash、fixture `prepare→abstain` 静态门通过 |

未改变 estimator 对 target delivery 的估计，也未改 public deterministic rules；因此规则仍可能选 action4，SA actor 也可由
PPO 回报继续学习 action4。本门只撤销未经校准的正 event auxiliary 梯度。

## 静态覆盖与最小见证

静态审计脚本只读取 2 个 public fixture 和 9 个既有 state snapshot，不调用 `env.step()`、不加载 checkpoint、不训练。
产物：`artifacts/analysis/cscwd_positive_prepare_calibration_abstention_static_audit_20261011_v1/audit.json`，SHA-256=
`e4268aa8aec2c981305f6697d15a145d37e151d28f08873150f77d2be0affd39`。

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python \
  scripts/audit_cscwd_positive_prepare_calibration_abstention.py \
  --public-info /Users/howen/.codex/worktrees/cscwd-window-identity/PPO_MEC/artifacts/analysis/cscwd_raw_phase_v2_preflight_20261011_v3/public_fixture.json \
  --public-info /Users/howen/.codex/worktrees/cscwd-window-identity/PPO_MEC/artifacts/analysis/cscwd_public_action4_state_20261011_v1/public_info.json \
  --snapshot-dir /Users/howen/.codex/worktrees/cscwd-window-identity/PPO_MEC/artifacts/analysis/cscwd_service_feasible_action_branches_20261010_v1/state_snapshots \
  --pilot-run-manifest artifacts/experiments/cscwd_phase_v2_causal_public_pilot_20261011_v1/run_manifest.json \
  --output artifacts/analysis/cscwd_positive_prepare_calibration_abstention_static_audit_20261011_v1/audit.json
```

| provenance | 旧 label | 新 label | 数量 |
|---|---|---|---:|
| `literal_false` | prepare | abstain | 1 |
| `literal_false` | abstain | abstain | 1 |
| missing | serve | serve | 7 |
| missing | abstain | abstain | 2 |

总计旧 `prepare/serve/abstain=1/7/3`，新为 `0/7/4`；唯一变化为 `prepare→abstain`，所有 public estimate 和 serve label
保持不变。最小可复现见证是冻结 phase-v2 fixture：旧 label=`prepare, weight=1`，provenance literal false；新 label=
`abstain, weight=0`。

覆盖全历史 pilot 的依据不是逐行回放：旧 run manifest 证明有 23,040 training steps，而相应 Raw producer 源码合同对每次
`_info()` 固定写 literal false；历史 transition 没有逐行保存完整 semantic state。因此可确认该 producer 合同下所有正 prepare
都会被撤销，但不能伪称重建了每个历史 minibatch、old policy 或 optimizer 更新。

## 唯一冻结的下一轮 A/B（未执行）

协议：`configs/experiment/cscwd_phase_v2_positive_prepare_calibration_abstention_ab_v1.json`，当前
`execution_authorized=false`。runner 已支持该协议，但本轮没有运行其 preflight 或 scientific execution。

- 唯一变量：SA 的 `mechanism_aux_positive_prepare_calibration_abstention_enabled=false/true`；两臂都开启原
  `causal_public_prepare_advantage_v3`。
- 匹配参照：同一新 run 内从头训练 PPO、MAPPO；同一 public immediate/two-step rules 重新评价。旧 pilot 仅作描述性历史，
  不充当匹配 control。
- 共同预算：4 arms × 5 seeds；每 cell 1,152 environment steps、24 updates、192 optimizer steps；总计 23,040 steps，
  不超过旧 pilot。checkpoint 候选固定 update 6/12/18/24，同时报告 selected 与 fixed update24。
- 主要服务门：completion、on-time 不下降且 failure 不上升；资源共同报告 failed migration、model/state/input/backhaul bytes、
  recompute、state ready、elapsed。SA 胜过 PPO/MAPPO 不是完成条件。
- 立即否证：任一 selected/fixed 视角服务恶化；Raw 正 prepare 仍有非零 event supervision；服务非劣但 failed migration、
  transfer/recompute 没有至少一个事前指标改善；方向只来自少数 seed、已暴露 US101 或 checkpoint 反选。

该设计仍只有 Lankershim/US101 两个已暴露 development 来源，不是独立 holdout；即使支持，也只能说明“撤销未校准正
prepare 监督”值得继续研究，不能证明自适应置信机制、SA 稳定领先、正式性能或 paper-ready。

## 验证

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q \
  tests/test_cscwd_positive_prepare_calibration_abstention.py \
  tests/test_causal_public_prepare_advantage_auxiliary.py \
  tests/test_causal_public_action_estimator.py
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m py_compile \
  scripts/audit_cscwd_positive_prepare_calibration_abstention.py \
  scripts/run_cscwd_phase_v2_causal_public_pilot.py \
  src/agents/causal_public_action_estimator.py src/agents/sa_ghmappo_core.py
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/smoke_test.py
```

结果：`27 passed`；语法检查通过；smoke 完成 6/6 toy DAG nodes。协议纯静态 validator PASS 且授权门拒绝执行；静态标签
审计 PASS。未覆盖风险是候选的训练效果、跨 seed 稳定性、source-disjoint 泛化和 calibrated=true 生产者行为。
