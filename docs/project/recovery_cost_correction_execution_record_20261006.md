# 恢复成本纠错执行与交付记录

## 身份与保护边界

- `reviewed_at`: `2026-10-06`
- `literature_cutoff`: `2026-10-05`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 起始分支：`codex/manuscript-evidence-v1`
- 起始本地/远端 HEAD：`b3a00b6141039c7ab223eaef8d03cf6c3efdc221`
- 已核对父提交：`f31024d957bd07718c4087fa55d8f7bb707e0f6e`
- 隔离工作树：`/Users/howen/.codex/worktrees/cost-validation/PPO_MEC`
- 主工作区保护：`scripts/train_sa_ghmappo_real_sample.py`、`src/agents/sa_ghmappo_agent.py`、
  `src/agents/sa_ghmappo_core.py`、`src/encoders/fusion_encoder.py`、`src/evaluators/real_eval_support.py`、
  `tests/test_algo_pool_contract.py`、`tests/test_checkpoint_compat.py` 七个修改未触碰。

提交序列：

1. `4e897bf152422beb072fd021c40560e6229faeae`：独立只读缺陷影响报告；
2. `800f0a12f13e34d7de19aee375124813b8817e3a`：对称成本实现、测试、原矩阵与边界配置一次性冻结；
3. `30691177bcac540ec1650d14494fe48326805a8c`：原 12 点纠正版原件与逐点报告；
4. 最终文档/边界 artifact 提交：见本记录所在 Git commit。

## 科学执行命令与预算

原 12 点唯一执行：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python \
  scripts/run_symmetric_recovery_cost_validation.py \
  --config configs/experiment/eviction_aware_recovery_corrected_v2.json \
  --output-root artifacts/analysis/eviction_aware_recovery_corrected_20261006_v2 \
  --expected-git-commit 800f0a12f13e34d7de19aee375124813b8817e3a \
  --historical-results artifacts/analysis/eviction_aware_recovery_validation_20261005_v1/all_method_results.json
```

结果：`COMPLETED`，12 点、48 隔离 path、0 model call、0 RL、0 download、0 old holdout，wall `0.502246 s`。

边界检查的首次命令把 expected SHA 手工误写为 `306911758c…`，在创建 output root 之前由 identity gate 拒绝；科学 path
执行数为 0，没有 artifact、模型调用或结果可覆盖。读取 `git rev-parse HEAD` 后使用真实提交执行：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python \
  scripts/run_symmetric_recovery_cost_validation.py \
  --config configs/experiment/recovery_cost_boundary_check_v1.json \
  --output-root artifacts/analysis/recovery_cost_boundary_check_20261006_v1 \
  --expected-git-commit 30691177bcac540ec1650d14494fe48326805a8c
```

结果：`COMPLETED`，6 点、24 隔离 path、0 model call、0 RL、0 download、0 old holdout，wall `0.232873 s`。这是唯一一次
边界科学执行；没有按输出改点、重试或搜索。

总预算：真实模型 `generate=0`；原矩阵 1 次；边界矩阵 1 次；训练、下载、旧 holdout、跨 workflow 平台和共享队列均为 0。

## 验证命令与结果

语法与受影响合同/回归：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m py_compile \
  src/runtime/symmetric_recovery_cost.py \
  scripts/run_symmetric_recovery_cost_validation.py \
  tests/test_symmetric_recovery_cost.py

/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q \
  tests/test_symmetric_recovery_cost.py \
  tests/test_eviction_aware_recovery.py \
  tests/test_typed_cache_sequential_recompute.py \
  tests/test_production_action4_state.py \
  tests/test_real_cache_victim_reload.py

/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/smoke_test.py
```

结果：`30 passed in 1.59s`；toy smoke 6/6 节点完成，`terminated=True`。另在冻结提交前分组运行 `20 passed` 与
`10 passed`，结果一致。

自动验证覆盖：

- 两条路径共同成本各计一次且 event equality 可审计；
- 四个 preview/score 环境互异，restart/recovery cache state 相互隔离；
- dependency-unsafe path 保守拒绝，既有 native 非法 victim/orphan/容量回归继续通过；
- d04 两次完整 bundle admission 合计恰为 240 MiB，无漏计或重复计费；
- 所有方法完成量一致后才比较成本；restart 主动重算不再伪装成 service failure；
- online record 全部 `realized_score_used=false`，offline 才可读取实际分支评分。

## Artifact 完整性

- `eviction_aware_recovery_corrected_20261006_v2`：7/7 manifest 文件 size/SHA-256 byte-exact；
- `recovery_cost_boundary_check_20261006_v1`：5/5 manifest 文件 size/SHA-256 byte-exact；
- 历史 `real_cache_victim_reload_20261005_v1`：22/22 byte-exact；
- 历史 `eviction_aware_recovery_validation_20261005_v1`：当前 checkout 29/30 byte-exact；唯一 CSV 因 Git
  CRLF→LF 规范化为 8,529 B / `fc3efee6…`，内存恢复 CRLF 后为 manifest 的 8,578 B / `0f1a1624…`。JSON 与语义数值可核验。

## 未覆盖风险与停止

没有实测无线/queue、任务质量、跨 workflow persistent cache、共享资源竞争、独立真实 workload 或统计区间；边界点不是现实
误差分布。真实 lifecycle 仍限一台主机、一个 base、两个 adapter 和未清 OS cache。完成本轮后停止，不自动启动 RL、扩大矩阵
或追加模型测量。
