# CRDCM performance matrix v2：科学条件、执行合同与实现 smoke

- `reviewed_at`: `2026-09-28`
- `literature_cutoff`: `2026-09-28`（本轮没有新增网页检索）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `crdcm_performance_matrix_v2_smoke_20260928`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `implementation_base_commit`: `df6048c0bbcbe83ee95aab6057b6e64f09af0a87`
- `evidence_level`: `E2_ARTIFACT_AUDITED`，仅限实现链路 smoke；性能与 novelty 仍为 `UNVERIFIED`
- `verdict`: `PASS for executable matrix plumbing / UNVERIFIED for CRDCM advantage and novelty`

## 结论

旧文档中的“5 conditions × 3 seeds × 64 = 960”只是预算 proposal，不是一份完整冻结矩阵。v2 不再把
`CRDCM` 和 `matched SA` 写成两个没有实现差异的名字，也不为凑满 960 重复训练。当前冻结为四个独特 learned
conditions、三个事前 seed、每 cell 64 episodes，共 768 training episodes / 15,360-step 上限；固定 checkpoint
在 episode 64 / update 16 的 `latest.pt`，不按结果择点。配对开发评价另计 156 episodes / 3,120-step 上限；训练、
评价和强启发式合计 924 episodes / 18,480-step 上限，低于旧 proposal 的 960 / 19,200 数字。

完整矩阵仍未获执行授权。本轮只执行实现 smoke：8 个训练 episodes、104 个实际训练 steps；四个 checkpoint 各一次
restore evaluation 加一个强启发式 unit，共 5 episodes、45 个实际评价 steps。合计 13 episodes、149 actual steps，
低于 260-step smoke cap。它不进入未来性能结果。

## 科学条件与真实映射

| condition | 实现 | 科学角色 | 每 seed 预算 | 固定终点 |
|---|---|---|---:|---|
| `crdcm_full_sa` | `crdcm_sa_ghmappo`, `full` residual | graph/hierarchical candidate full | 64×20 | episode 64 / update 16 `latest.pt` |
| `crdcm_signal_off_sa` | 同一 wrapper、credit、runtime，`signal_off` 把新增 residual 精确置零 | 贡献消融；不是故意破坏物理系统 | 64×20 | 同上 |
| `crdcm_full_mappo` | `crdcm_mappo`, 同一 28 维 full residual | flat controller-CTDE architecture comparison | 64×20 | 同上 |
| `crdcm_full_ppo` | `crdcm_ppo`, 同一 28 维 full residual | flat PPO architecture comparison | 64×20 | 同上 |

`signal_off` 保留 typed base/adapter cache、容量、LRU、workflow-state migration、五动作 authority、runtime mask/projection、
外生 request replay 和 external-override credit 合同；只令 CRDCM 新 residual 对 actor/value 的增量为零。它因此是
“新增 CRDCM signal/control contribution off”，不是关闭合理服务能力的弱对照。legacy SA 仍可通过基类 semantic state
工作。`full SA vs signal-off SA` 估计新增 CRDCM residual 的主效应；`full SA/MAPPO/PPO` 才是同信息、同物理机制下的
架构比较。不存在另一个仅改名的 `CRDCM` condition。

CRDCM full 的实质增量仍只是 28 维 causal state 到 actor/value logits 的 trainable residual；动作仍为现有五动作。
因此当前不能把它写成对象级 dependency-bundle joint optimizer，novelty 保持 `UNVERIFIED`。

## Seed、训练 exposure 与超参数

- seeds 固定为 `1401/1402/1403`；规则是在任何 CRDCM 训练结果前，从既有 diagnostic seed `1401` 取连续 offsets
  `0/1/2`，不根据结果更换。
- 每个 cell 使用 3 个已观察、raw frame/time 两两不重叠的 NGSIM windows 和 4 个 ordered Alibaba workflows。
  12 个 `(window, workflow)` pair 重复 5 个完整周期，再重复顺序中的前 4 个；所有 condition/seed 顺序完全相同。
- 唯一训练 scenario 为 360 MB、base sharing on、migration on、`semantic_ai_service`、handoff-pressure 主车选择。
- `learning_rate=3e-4`、`clip=0.2`、`entropy=0.01`、`value=0.5`、`gamma=0.99`、`GAE=0.95`、
  `update_every=4`、`batch=32`、`checkpoint_every_updates=4`、`prediction_horizon=3`、reward offset `0`。
- 每个 condition/seed 的 64 episodes 产生 16 updates；只消费共同固定终点，不使用训练指标选择 checkpoint。

这些 windows/workflows 已用于 2026-09-28 mechanism diagnostic/retraining，属于 observed-data development
resubstitution，不是新 split、independent evaluation、formal 或 untouched holdout。旧 consumed holdout 不读取、不重开。

## 固定配对评价与成本

每个 12 个 learned checkpoints 在 4 scenarios × 3 固定 window/workflow pairs 上 deterministic evaluate 一次：

1. `capacity_competition`：280 MB、repeated reuse、migration on；
2. `shared_adapter_reuse_prepare`：360 MB、repeated reuse、migration on；
3. `low_reuse_no_migration_negative`：360 MB、low reuse、migration off；
4. `ample_resource_negative`：1024 MB、low reuse、migration on。

learned evaluation=`12 checkpoints × 12 units = 144 episodes`；强启发式不训练，只跑同 12 units；总评价 156 episodes。
报告 completion、continuity、conditional delay+coverage、right censor/failure、transfer、backhaul、eviction、stale/expired
prefetch 和 migration cost 的向量/Pareto，不生成事后 weighted score。原始 window 是 outer unit；只有 3 个 window，统计
功效很低，不能把 seed×workflow 行当独立 cluster，也不能据此晋级 formal claim。

## Credit、critic 与 checkpoint 语义

`mask_external_override_actor_credit_v1` 现在区分两条路径：

- 实际 executed action 等于 sampled/aggregated policy action 时，样本进入正常 PPO actor+critic update；
- guard/runtime/trainer 外部改写动作时，actor credit 为 0，raw sample 不进入 PPO actor ratio；同一 executed trajectory
  仍进入独立 value-only update，使 critic 不丢掉 override 后的真实 return。

受控真实采样测试先由 agent 产生 raw action，再由 trainer 明确改成不同 executed action；验证
`sampled_from_raw_policy_distribution=false`、actor-specific residual output weights不变、value row改变且全部参数 finite。
真实 8-episode smoke 未自然触发 override（四个 condition 均 `0/26`），因此不能拿 smoke 本身声称已覆盖 runtime override
机会；该分支由定向训练测试覆盖。

checkpoint 记录 observation/action/credit/feature-mode 版本和独立 residual state。`full` checkpoint 不能加载到
`signal_off` runtime；legacy checkpoint 仍 fail-fast。四个 smoke checkpoint 均在一次 update 后 parameter digest 改变、
全部 finite，并成功 restore；评价前后文件 SHA-256 完全相同。

## 资源、入口和失败合同

- Python 必须使用精确入口 `/Users/howen/Projects/PPO_MEC/.venv/bin/python`；不把 symlink resolve 后的系统解释器
  当执行命令。preflight 核对 `sys.prefix=/Users/howen/Projects/PPO_MEC/.venv`、PyTorch/YAML 和两个入口 `--help`。
- NGSIM 实体：size `2118175938`，SHA-256 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`。
- Alibaba 实体：size `802261444`，SHA-256 `6346b0726c6e10466a585c67645af807b425b5be091caf410f5e1aff41a270bc`。
- checkout 中 135/134-byte Git LFS pointers 会在读 CSV 前显式拒绝；不会自动下载或覆盖。HF metadata manifest 不在
  CRDCM producer/consumer 依赖图中，不以假数据替代。
- full root 固定为 `artifacts/training/crdcm_performance_matrix_v2_20260928/`，create-only、单次 background launch、
  no retry；保存完整 argv/manifest、RUNNING state、outer/child logs、checkpoint freeze、evaluation 和 terminal receipt。
- full 启动命令已经可展开，但本轮没有调用 `launch`，所以 full root/receipt/process 均不存在。

可执行入口：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/run_crdcm_performance_matrix.py preflight \
  --config configs/experiment/crdcm_performance_matrix_v2.yaml \
  --data-root /Users/howen/Projects/PPO_MEC

# 仅在中央后续明确批准完整执行后允许调用：
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/run_crdcm_performance_matrix.py launch \
  --config configs/experiment/crdcm_performance_matrix_v2.yaml \
  --data-root /Users/howen/Projects/PPO_MEC \
  --output-root artifacts/training/crdcm_performance_matrix_v2_20260928
```

## Smoke evidence 与资源估计

Smoke artifact：`artifacts/analysis/crdcm_performance_matrix_v2_smoke_20260928/`。四个 child return code 均为 0，
stderr 均为空；completion/evaluation receipt 为 `SUCCEEDED`，无 retry。checkpoint 大小分别约 2.12/2.10/0.57/0.35 MB；
artifact 总计约 12 MB，自排除 integrity manifest 覆盖 48 个文件。

本机 smoke wall-clock 约 8.35 秒，但它只包含 149 actual steps，不能线性保证 full 时间。按 18,480-step cap、12 次
training process 初始化和 156 次评价保守估计 CPU wall-clock 约 12–30 分钟；未实测峰值 RSS/GPU-hours/能耗。
按 smoke episode JSON 均值约 0.64 MB 外推，full root 预计约 0.6–1.0 GB；这些是 capacity planning estimate，
不是完成 receipt。

## 禁止表述与后续停点

- 不得称 960 proposal 曾是完整冻结矩阵；v2 是首次完整 condition/seed/argv/consumer/receipt freeze。
- 不得将本轮 smoke 指标解释为算法效果、收敛、强启发式优势或负控结果。
- 不得称新数据、formal、holdout、generalization、paper-ready 或 TMC-ready。
- 3 seeds、3 observed windows、单一 NGSIM+Alibaba 组合的统计功效明显不足；即使未来 v2 结果正向也只能作为
  observed-data development falsification。
- 下一步停在中央审阅矩阵、预算和实现；完整训练与评价需新的明确执行批准。

## 验证记录

- `python -m pytest -q tests/test_crdcm_performance_matrix.py tests/test_crdcm_decision_contract.py`：19 passed。
- CRDCM + env/checkpoint/algo contract 定向组合：283 passed。
- `python scripts/smoke_test.py`：passed。
- 全仓 `python -m pytest -q tests --tb=short`：1323 passed、2 skipped、22 failed、49 errors。22 failures 全部来自
  checkout 的 129-byte HF manifest LFS pointer；49 errors 来自 134-byte Alibaba / 135-byte NGSIM pointers 被旧 fixtures
  当实体内容使用。它们与此前已知类别一致，本任务 runner 已显式绑定外部实体并在读取前拒绝 pointer；没有自动下载。
- full preflight：12 training commands 全部解析，source identity、venv、三窗口 interval 和预算恒等式通过，且不创建
  full root。`git diff --check` 与受保护七文件 hash audit 通过。
