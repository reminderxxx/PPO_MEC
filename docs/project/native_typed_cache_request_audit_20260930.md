# 原生 typed-cache 逐请求一致性与决策空间审计

- `reviewed_at`: `2026-09-30`
- `literature_cutoff`: `2026-09-30`（本轮未做新文献检索；仅审计有界机制）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `native_typed_cache_request_audit_20260930_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `audited_native_git_commit`: `73051ab264aa868e83f2e011b5ced26968eef74b`
- `evidence_level`: `E2_ARTIFACT_AUDITED_FOR_BOUNDED_NATIVE_MICROTRACE_ONLY`
- `verdict`: `Unverifiable for paper readiness; verified bounded native transaction semantics`

## 结论

固定四配置 `blocked/interleaved × sharing on/off` 已各执行 72 个请求，共 288 个请求。请求实际进入
`VecWorkflowCoreEnv._apply_typed_cache_action()`，victim 由原生 `LRUEvictionPolicy` 规划，服务 readiness 与
`CacheEvent 1.3` 也由原生 producer 生成。旁挂脚本只负责输入映射、调用、日志和不变量复核，没有实现第二套缓存、
修改驱逐规则、训练、扩大矩阵、接入 LFU 或读取 holdout。

原生与 reference 不一致。最早按冻结观察顺序出现的差异是 sharing-on/interleaved 的请求 1 `b1.a0`：
reference 驱逐 `[b0.a0, b0]` 后载入 `[b1, b1.a0]`；原生单次 dependency-safe 可行集只包含
`adapter:b0.a0`，可释放 8 MiB，小于所需 104 MiB，因此返回
`rolled_back_no_mutation / insufficient_dependency_safe_evictable_capacity`。resident 和 LRU metadata 在拒绝前后
逐字段相等。该差异保留为独立设计问题；本轮没有放宽原生规则。

## 输入与完整性

- 原包：`/Users/howen/Downloads/vec_mechanism_probe_v0_2.zip`
- 大小：`351,734 bytes`
- SHA-256：`c0b3325874e7427629af8ed36720ed75db5b292c9658d7f3defd39a9deb44bbe`
- ZIP：39 个文件，CRC 通过；`PACKAGE_MANIFEST.json` 列出的 38 个成员逐文件 SHA-256 通过。
- `native_cache_audit_partial_20260930.zip`：本轮不需要、未消费、未据此补字段。
- 主工作区固定 HEAD 与中央观察值一致；7 个用户修改未进入隔离 worktree，未 stash/reset/覆盖。

机器证据位于
`artifacts/analysis/native_typed_cache_request_audit_20260930_v1/`：

- `native_mapping_contract.json`
- `request_comparison.jsonl` / `request_comparison.csv`（288 行）
- `first_observed_native_difference.json`
- `eviction_feasible_set_comparison.json`
- `request_admission_hit_transfer_failure_summary.json`
- `research_claim_boundary.json`
- `commands_and_environment.json`
- `input_integrity_manifest.json`
- `artifact_integrity_manifest.json`

## native_mapping_contract

原包以 bytes 和 MiB 定义大小；原生代码对 `capacity_mb`、`resident_size_mb` 和 `transfer_size_mb` 直接做浮点加总与
比较，内部没有 bytes、SI MB 或 MiB 转换。因此映射不是“字段同名即同单位”，而是显式采用：

```text
native numeric mb value = source bytes / 1,048,576
```

由此得到：容量 `142,606,336 bytes -> 136.0 native mb`，`b0=96.0`，`b1=128.0`，每个 adapter=`8.0`。
这是为保持原包字节约束而冻结的数值映射，不证明原生 `mb` 字段天然表示 MiB。

sharing-on 中，一个 family 的三个 adapter 共同依赖 `base:b0` 或 `base:b1`。sharing-off 中每个 adapter 都映射到
独立 base identity，例如：

```text
b0.a0 -> base:b0::private::b0.a0
b0.a1 -> base:b0::private::b0.a1
b0.a2 -> base:b0::private::b0.a2
```

`b1` 同理。每个 private base 仍通过原生 typed dependency 表达，不用标签模拟共享关闭。四配置均使用单一
`rsu_a`、空初始 cache、LRU、136 MiB、原包固定请求次序；`request_index` 映射为原生 LRU 的 episode step。

## sharing-on/interleaved 前两个请求

### 请求 0：`b0.a0`

- pre-action resident：空，used=`0`
- dependency bundle：`[base:b0, adapter:b0.a0]`，missing 为两者
- required-free：`0`
- transaction：`committed`
- transfer：base=`96 MiB`，adapter=`8 MiB`
- post resident：`[base:b0, adapter:b0.a0]`，used=`104 MiB`
- post-action service：成功；reference 的 `full_hit=false` 是 pre-action hit 字段，不用它替代原生 post-action 服务结果。

### 请求 1：`b1.a0`

- pre-action resident：`[base:b0, adapter:b0.a0]`，used=`104 MiB`
- dependency bundle：`[base:b1, adapter:b1.a0]`，missing 为两者，总计 `136 MiB`
- required-free：`104 MiB / 109,051,904 bytes`
- eligible victims：仅 `[adapter:b0.a0]`
- LRU plan：选择该 adapter，累计只能释放 `8 MiB`，`sufficient=false`
- transaction：`rolled_back_no_mutation`
- rejection：`insufficient_dependency_safe_evictable_capacity`
- transfer：base=`0`，adapter=`0`
- post resident 与 policy metadata：与请求前完全一致
- CacheEvent：`admission_reason=insufficient_dependency_safe_evictable_capacity`、`hit_source=unserved`、
  `service_success=false`
- origin/cloud execution：`false`；本有界 RSU 审计未配置回源 fallback

这三个结果必须分开读：本请求发生了缓存准入拒绝；随后 RSU 服务失败；没有发生回源执行。

## 四配置汇总

所有传输量均是原生实际 transaction 的 per-type `transfer_mb_by_type` 按冻结映射还原成 bytes；低传输量来自大量
准入拒绝，不能解释为节省或性能优势。

| 配置 | 请求 | reference load | native transfer | native committed / noop / rejected | pre-hit | post service success / failure | origin |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocked / sharing on | 72 | 3,264 MiB | 120 MiB | 3 / 33 / 36 | 33 | 36 / 36 | 0 |
| interleaved / sharing on | 72 | 8,640 MiB | 120 MiB | 3 / 33 / 36 | 33 | 36 / 36 | 0 |
| blocked / sharing off | 72 | 8,640 MiB | 104 MiB | 1 / 11 / 60 | 11 | 12 / 60 | 0 |
| interleaved / sharing off | 72 | 8,640 MiB | 104 MiB | 1 / 11 / 60 | 11 | 12 / 60 | 0 |

原生四配置实际 eviction 均为 0。不是 LRU 没被调用：跨 family/private-base 切换时 planner 返回了 insufficient
plan，但原子事务不应用不完整 victim plan；sharing-on 的 `b0 + 三个 adapter=120 MiB` 又能共同驻留，因此同 family
内无需驱逐。

## 两种驱逐可行集与 LRU 语义

reference 的 base candidate 扩张为“base + 全部 resident dependent adapters”的 dependency closure；同一请求循环
可以删掉 adapter，再删掉其 base。其 LRU 用内部 request clock，且请求任一 adapter 时刷新该 base；tie-break 先
adapter 后 base，再按 object ID。

原生首先在当前 resident snapshot 上建立一次可行集：只要还有 resident adapter 依赖 base，该 base 就被排除。
随后 `LRUEvictionPolicy` 按 `(last_used_step, native_object_id)` 选满足 required-free 的最小前缀；不会在选中 adapter
之后重新计算 base eligibility。两者虽都标为 LRU，但可行域不同，因此结果不应被要求机械一致。

独立设计问题为：typed transaction 是否需要 dependency-closure eviction，或在一个原子请求内顺序重算 eligibility。
本轮只记录该问题，不修改 production strategy。

## 不变量

288/288 请求均满足：

- resident used `<=136`；
- `orphan_count=0`；
- `before - evicted + admitted = after`；
- 每个拒绝请求 resident 与 policy state 不变；
- 每个 reference 未观测的 admission/service/origin 字段保持 `null`，未用静态推导补齐。

未触发安全停止条件。

## 可支持与不可支持结论

可支持：在该固定微负载和显式单位映射下，原生 typed-cache 的当前单次 dependency-safe victim 可行集与 reference
closure-eviction 可行集不同；该差异直接导致 b1 切换准入拒绝，并使原生 blocked/interleaved 汇总不再复现 reference
的次序效应。拒绝保持原子性与状态不变量。

不可支持：原生/reference 等价；低传输量代表收益；真实模型加载、时延、吞吐或成本改善；LRU 算法优劣；
SA-GHMAPPO/PPO 训练或性能；formal/holdout/canonical 晋级；TMC-ready；合成 `b0/b1` 是真实 adapter 或部署尺寸。

## 执行与验证

主要审计命令：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/audit_native_typed_cache_probe.py \
  --input-zip /Users/howen/Downloads/vec_mechanism_probe_v0_2.zip \
  --output-root artifacts/analysis/native_typed_cache_request_audit_20260930_v1 \
  --phase full
```

先以 `--phase first-two` 单独保全前两请求，确认无完整性/状态不变量破坏后才执行 full。环境、实际命令和执行时
Git 状态见 artifact。匹配验证结果：

```text
python -m pytest -q tests/test_native_typed_cache_audit.py tests/test_typed_model_cache.py \
  tests/test_cache_capacity_mb.py tests/test_cache_eviction_policy.py tests/test_cache_event_contract.py
50 passed

python -m pytest -q tests/test_env_contract.py
14 passed

python scripts/smoke_test.py
passed; terminated=True, truncated=False, completed nodes=6
```

未训练、未运行 formal/holdout、未调用真实模型。未覆盖风险是 full `env.step()` 的 mobility/workflow progression、
真实 cloud/origin fallback 与真实模型 I/O；固定 request stream 通过原生 transaction、readiness 与 CacheEvent producer
逐请求消费，但没有把服务失败后的 workflow progression 扩展成另一套实验。
