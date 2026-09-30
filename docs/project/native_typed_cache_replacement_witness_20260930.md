# 原生 typed-cache 合法底座替换候选与最小工作流见证

- `reviewed_at`: `2026-09-30`
- `literature_cutoff`: `2026-09-30`（未检索文献；有界实现诊断）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `native_typed_cache_replacement_20260930_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `execution_git_commit`: `ffa10e598feddac99c2557d994ba094692ede99a`
- `delivered_audit_commit`: `371159dabbacdc0d82acbc63efd588508338d289`
- `audited_native_parent`: `73051ab264aa868e83f2e011b5ced26968eef74b`
- `evidence_level`: `E2_BOUNDED_NATIVE_IMPLEMENTATION_DIAGNOSTIC`
- `verdict`: 候选机制在冻结 synthetic microtrace 上可行；算法优势、formal/holdout 与 paper readiness 不可验证

## 结论与可达性

原有 `semantic_discrete_5` 动作只能请求当前/预测 RSU 缓存当前节点所需 adapter，没有 agent 可输出的 typed-object
remove 动作。`ControlAction` 也没有合法的逐对象删除操作。旧 transaction 在一次静态 resident snapshot 上建立
dependency-safe 候选集合，因此不存在“先删 adapter、再由正常动作删刚解除依赖的 base”的现成路径。这是 G13
冻结语义，不是原生 LRU 的排序错误。

新增候选保持五动作 codec 不变，所有使用共享 `GymVecEnv` / `ActionAdapter` 的算法都能通过同一个 cache-fill 动作
到达；没有 SA 专属 gate。候选仅在显式选择 typed + LRU 时启用：

```text
typed_eviction_semantics = sequential_dependency_recompute_lru_v1
typed_cache_transaction_contract_version = typed_cache_transaction_contract_v1.1.0
```

默认仍是 `static_dependency_safe_v1` + `typed_cache_transaction_contract_v1.0.0`。旧配置的序列化结构与三个冻结
formal runtime SHA-256 未变化。FIFO/LFU/Aging-LFU/Random 与 candidate 组合 fail-fast，不扩展策略矩阵。

## 候选事务语义与观测边界

候选在 transaction shadow resident 上循环：按原生 `LRUEvictionPolicy` 的
`(last_used_step, native_object_id)` 排序取一个 victim，影子移除后重新计算 dependency-safe 可驱逐集合。每轮
记录 shadow residents、eligible candidates、原生 LRU plan、victim 和 remaining required-free。完整 plan 可行后，
才在真实状态上依次提交 hit 更新、eviction callbacks、admission callbacks 与 resident replacement；失败不提交任何
hit、eviction、admission 或 resident 修改。

- transaction 前：真实 resident 与 LRU metadata 是唯一输入；当前请求完整 dependency bundle 受保护；
- shadow planning：不改变真实 resident、LRU metadata、访问计数或其他 policy 状态；
- noop/success：命中更新在 plan 确认后提交；成功后容量与 dependency invariant 再验证；
- failure：resident 与完整 policy export 逐字段等于 transaction 前，实际传输与实际释放均为零。

## 固定两请求验收

原包和映射重新核验：`136 MiB = 142,606,336 bytes`，`b0=96 MiB`，`b1=128 MiB`，adapter=`8 MiB`。
原生 runtime 对 `mb` 数值直接加总，不做 bytes 转换；本轮仍显式采用
`native numeric mb = source bytes / 1,048,576`，不根据字段名猜单位。

| 请求 | pre resident / used | required-free | shadow victims | 实际释放 | admission | 分类型传输 | post resident / used |
|---|---|---:|---|---:|---|---|---|
| `b0.a0` | `[]` / 0 | 0 | `[]` | 0 | committed | base 96 + adapter 8 MiB | `[base:b0, adapter:b0.a0]` / 104 MiB |
| `b1.a0` | `[base:b0, adapter:b0.a0]` / 104 | 104 MiB | round 0 `adapter:b0.a0`; round 1 `base:b0` | 104 MiB | committed | base 128 + adapter 8 MiB | `[base:b1, adapter:b1.a0]` / 136 MiB |

第二轮 shadow 中 base 首轮仍因 adapter 依赖而不可驱逐；影子删除 adapter 后，base 才进入下一轮 eligible set。
完整计划确认后一次提交，`orphan_count=0`。逐轮 policy metadata 与 native plan 在 candidate request log 中保留。
多 adapter 正例从 `[base:b0,a0,a1]` 开始，依次规划 `[a0,a1,base:b0]`，释放112 MiB并原子准入 b1 bundle。

## 负例

| case | 结果 | 说明 |
|---|---|---|
| pinned adapter | `rolled_back_no_mutation` | adapter 不可选，依赖 base 始终受保护 |
| pinned base | `rolled_back_no_mutation` | 可影子规划 adapter，但不足以完成 plan；真实释放为0 |
| non-evictable adapter | `rolled_back_no_mutation` | 对象与其依赖 base 均不能形成完整释放计划 |
| 其他保留 adapter 依赖 base | `rolled_back_no_mutation` | 可规划旧 a0，但 non-evictable a1 保留，base 不进入候选 |
| 135 MiB 下的 b1 bundle | `rolled_back_no_mutation` | 136 MiB bundle 超总容量，shadow planning 不启动 |

所有拒绝的真实 resident 与完整 LRU policy export 均未变化，实际释放/准入/传输为0，容量合法且无 orphan。

## 四配置 old/new 匹配回归

每配置固定72请求；仅重跑指定四配置。旧语义所有计数与已交付审计机器汇总逐字段匹配。

| 配置 | 语义 | committed / noop / rejected | 实际驱逐对象 | 服务成功 / 失败 | base / adapter 传输 |
|---|---|---:|---:|---:|---:|
| interleaved / sharing on | old | 3 / 33 / 36 | 0 | 36 / 36 | 96 / 24 MiB |
| blocked / sharing on | old | 3 / 33 / 36 | 0 | 36 / 36 | 96 / 24 MiB |
| blocked / sharing off | old | 1 / 11 / 60 | 0 | 12 / 60 | 96 / 8 MiB |
| interleaved / sharing off | old | 1 / 11 / 60 | 0 | 12 / 60 | 96 / 8 MiB |
| interleaved / sharing on | candidate | 72 / 0 / 0 | 142 | 72 / 0 | 8,064 / 576 MiB |
| blocked / sharing on | candidate | 72 / 0 / 0 | 94 | 72 / 0 | 2,688 / 576 MiB |
| blocked / sharing off | candidate | 72 / 0 / 0 | 142 | 72 / 0 | 8,064 / 576 MiB |
| interleaved / sharing off | candidate | 72 / 0 / 0 | 142 | 72 / 0 | 8,064 / 576 MiB |

candidate 数值恰好与该 probe reference 的聚合 load/eviction 数相同，但实现没有复制 reference tie-break，不能据此
宣称逐请求排序语义等价。旧语义低传输来自拒绝服务；不能解释为性能收益。两份288行逐请求日志保留全部差异。

## 真实 `env.step()` 最小见证

见证固定为 synthetic diagnostic：一个车辆、一个 RSU、两个有序节点 `b0.a0 → b1.a0`，容量136 MiB，
`max_steps=2`，两种语义都从空 cache reset，均执行合法 Gym action `0` 两次，工作负载、资源和动作规则相同。

- old：step 1 准入并完成 `n0`；step 2 原子拒绝，`hit_source=unserved`、服务失败、`n1` 未执行完成；达到冻结
  step cap 后停止，workflow 未完成。
- candidate：step 1 准入并完成 `n0`；step 2 驱逐 `[adapter:b0.a0, base:b0]`、准入 b1 bundle、服务成功并完成
  `n1`；workflow 在第2步正常完成。
- 两条链均没有 `cloud` hit；`hit_source` 不被当作云端完成证据，也未新增自动回源系统。

## 已证明、未证明与剩余风险

已证明：合法 action 可达；候选在固定容量与大小下完成跨底座替换；要求的保护/超容量/保留依赖负例原子拒绝；
CacheEvent、runtime/fairness binding 与 episode summary 可记录版本/语义；两节点 workflow 经真实 reset/step 推进。

未证明：真实模型 I/O、时延/吞吐/成本收益、reference 全语义等价、非 LRU 策略、训练稳定性、算法优势、formal、
holdout、canonical 晋级或 paper readiness。candidate 当前是显式 non-formal diagnostic；正式 Protocol 2.9 仍绑定旧
v1.0 transaction，不能静默切换。

剩余工程风险：真实 callback 异常的进程级恢复仍沿用既有 transaction 假设；oracle/replay 的正式 candidate 语义尚未
独立冻结，因此本候选不进入 formal oracle；大 catalog 的规划复杂度未做 scalability 评估。

## 执行、完整性与工作区保护

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python \
  scripts/validate_typed_cache_sequential_replacement.py \
  --input-zip /Users/howen/Downloads/vec_mechanism_probe_v0_2.zip \
  --output-root artifacts/analysis/native_typed_cache_replacement_20260930_v1
```

执行时 branch=`codex/typed-cache-replacement`、HEAD=`ffa10e5...`、status为空、worktree diff SHA-256为空内容 hash
`e3b0c442...b855`。输入 ZIP 大小351,734 bytes、SHA-256=`c0b332...bbe`，CRC与38个 manifest member hash通过；
已交付审计的9个 manifest 文件 hash/size再次通过。

主工作区七个用户修改文件在本轮开始与正式诊断结束的逐文件 SHA-256完全相同，combined binary diff SHA-256均为
`1b10c6e1...908bb2`；未修改、暂存、提交、stash、reset或覆盖。完整值见
`main_workspace_seven_file_protection.json`。

机器证据：`artifacts/analysis/native_typed_cache_replacement_20260930_v1/`；integrity manifest 为 pass，含8个
被校验文件。旧审计与历史 artifact 未覆盖。

## 验证命令与结果

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q \
  tests/test_typed_cache_sequential_recompute.py \
  tests/test_typed_model_cache.py \
  tests/test_typed_model_cache_runtime.py \
  tests/test_cache_event_contract.py \
  tests/test_cache_capacity_mb.py \
  tests/test_cache_eviction_policy.py
# 61 passed

/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q tests/test_env_contract.py
# 14 passed

/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/smoke_test.py
# pass，6个 toy 节点完成，terminated=True

/Users/howen/Projects/PPO_MEC/.venv/bin/python -m compileall -q \
  src/envs/core src/envs/specs src/runtime/typed_model_cache_runtime.py \
  src/evaluators/cache_baseline_fairness.py \
  src/evaluators/main_results_support.py \
  scripts/validate_typed_cache_sequential_replacement.py \
  tests/test_typed_cache_sequential_recompute.py
# pass

git diff --check
# pass
```

额外扩大到 `tests/test_typed_runtime_plumbing.py` 的数据依赖运行仍有13个 setup error：隔离 checkout 中
`data/raw/workflow/alibaba2018/batch_task.csv` 的首行少于9列，符合未拉取完整 LFS 数据的表现；未自动下载或覆盖原始数据。
该限制不影响上述 controlled runtime、CacheEvent、capacity、eviction、env contract 与真实 `env.step()` 见证，
但完整 Alibaba plumbing 链仍是本轮未覆盖风险。
