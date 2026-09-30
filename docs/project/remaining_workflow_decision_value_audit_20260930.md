# 剩余工作流信息的决策价值审计

- `reviewed_at`: `2026-09-30`
- `literature_cutoff`: `2026-09-30`（本轮未新增文献；未引入外部部署参数）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `remaining_workflow_decision_value_audit_20260930_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `audited_candidate_git_commit`: `58c3a8900152f03e0d655194e1a02afb50f1f727`
- `candidate_execution_commit`: `ffa10e598feddac99c2557d994ba094692ede99a`
- `delivery_result_commit`: `addd1c5828648e2e3f373f6098f139479cee1849`
- `evidence_level`: `E2_BOUNDED_NATIVE_LEDGER_AND_LOCAL_HOST_MEASUREMENT`
- `verdict`: 剩余信息在有界长尾实例中改变合法准备决策；原生 action-4 见证在相同完成量下少传模型字节，但真实完整状态、adapter 与网络成本仍不可验证，当前不支持算法修改或 paper-ready 结论

## 一句话结论

剩余工作流信息有决策价值，但证据很窄：在 2 个预注册长尾实例中，它把首动作从 action 0 改为 action 1/4；只有
action 4 在固定 4-step 上保持相同节点完成量和服务失败数，同时少传 `104 MiB` 原生 synthetic 模型字节。该环境在
prepare 实现后仍记录 `0` 状态字节，而本机实测的 `732 bytes` 只是明确的应用级 payload、不是完整可迁移状态，故
`104 MiB` 只能作为真实状态 payload 的 byte break-even，不是净时延或部署收益。两步前瞻在全部 4 个实例上与完整
剩余序列作出相同首动作，没有证据需要更强的未来信息算法。

## 证据严格分层

| 层 | 本轮证据 | 可以说明 | 不能替代 |
|---|---|---|---|
| 原生仿真账本 | 已有四配置 576 条 request log；16 个新 bounded native episode | 原生 transaction、服务、加载、驱逐、动作可达性与 step-level 完成 | 硬件加载、完整状态、无线时延 |
| 本机真实测量 | 固定 SmolVLM base、MPS、3 次进程首次加载/推理；显式应用状态 3 次保存恢复 | 该主机、该版本、该 dtype 的本地过程成本 | synthetic 96/128 MiB 对象、车载/RSU、真实 adapter、网络 |
| 假设敏感性 | `state_bytes < avoided_model_bytes` 的符号 break-even | 明确假设下的字节符号和缺口 | 原生实测或网络测量 |

三层数据没有合并成总分，也没有用本机数字回填原生字段。

## 工作 A：Alibaba 数据消费回归

上一轮 13 个 setup error 的 first-order 原因不是 parser 或资源身份放宽，而是隔离 checkout 中的数据仍为 Git LFS
pointer：

| 对象 | 隔离 checkout | 主工作区现有真实文件 |
|---|---:|---:|
| Alibaba `batch_task.csv` | 134 bytes；pointer SHA-256 `14d63f...27a3f` | 802,261,444 bytes；SHA-256 `6346b072...70bc` |
| NGSIM CSV | 135 bytes；pointer SHA-256 `48c89c...d054` | 2,118,175,938 bytes；SHA-256 `ddacb7a0...d10c` |

两份真实文件的 byte size/SHA-256 分别与 LFS pointer 中 `size/oid` 完全一致。测试路径是仓库内常量，故按项目既有
clean-checkout 方式临时 symlink 到主工作区同一原件；未复制、下载、覆盖、mock 或跳过校验。执行：

```text
/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest -q tests/test_typed_runtime_plumbing.py
34 passed in 7.07s
```

测试后 symlink 已移除并恢复原 134/135-byte pointer；隔离 checkout 没有数据文件 diff。该结果只证明真实
NGSIM/Alibaba 数据消费与 typed runtime plumbing 兼容，不证明 adapter、迁移状态或移动模型真实。

## 工作 B：四配置原生账本

输入严格复用上一轮原件：

- `four_config_old_request_log.jsonl`：288 行；
- `four_config_candidate_request_log.jsonl`：288 行；
- `blocked/interleaved × sharing on/off`，每配置 72 请求；
- LRU、136 MiB；candidate=`sequential_dependency_recompute_lru_v1 / transaction v1.1.0`；
- 没有重跑四配置矩阵。

### 相同完成量比较：candidate 四配置

| 配置 | 成功/失败 | base load 次数/MiB | adapter load 次数/MiB | 驱逐对象/字节 | 重复加载字节 | base 复用请求 | 总传输/成功请求 |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocked / sharing on | 72/0 | 24 / 2,688 | 72 / 576 | 94 / 3,128 MiB | 2,992 MiB | 48 | 3,264 MiB / 45.33 MiB |
| interleaved / sharing on | 72/0 | 72 / 8,064 | 72 / 576 | 142 / 8,504 MiB | 8,368 MiB | 0 | 8,640 MiB / 120 MiB |
| blocked / sharing off | 72/0 | 72 / 8,064 | 72 / 576 | 142 / 8,504 MiB | 7,920 MiB | 0 | 8,640 MiB / 120 MiB |
| interleaved / sharing off | 72/0 | 72 / 8,064 | 72 / 576 | 142 / 8,504 MiB | 7,920 MiB | 0 | 8,640 MiB / 120 MiB |

`blocked + sharing on` 相比 `interleaved + sharing on` 在同为 72 完成请求时少 `5,376 MiB` base 传输。共享本身
只有在顺序形成同 family 连续段时兑现：blocked 把 base load 从 72 降到 24并产生 48 个 request-level base reuse；
interleaved 频繁跨 family，sharing on 与 off 的总传输相同。sharing off 中 blocked/interleaved 也相同，因为每个
adapter 使用 private base，顺序无法形成跨 adapter base 复用。

驱逐次数只用于对象生命周期账本。表中没有由驱逐次数推导传输时间或硬件时延。

### 旧语义的低传输不是节省

| 配置 | 成功/失败 | base/adapter 传输 | 驱逐 |
|---|---:|---:|---:|
| sharing on（两种顺序相同） | 36/36 | 96/24 MiB | 0 |
| sharing off（两种顺序相同） | 12/60 | 96/8 MiB | 0 |

旧语义只加载首个可驻留 family/private bundle，随后因静态 dependency-safe 可行集不足而拒绝。它没有与 candidate
相同的完成量，因此低传输不能称为真正节省，也不能用“每成功请求字节”掩盖未服务请求。

四配置 request replay 没有 workflow progression，故 `workflow_completion_count=null`；wait、execution、network
time 均无字段，保持 `unavailable`。没有 handoff/state request，状态成本标为 `not_applicable`，没有补零。

## 工作 C：最小真实模型和状态成本

### 冻结清单与身份

- base：`HuggingFaceTB/SmolVLM-500M-Instruct@a7da5b986cb59b408707209984f360a5f4ad7e47`；
- 13 文件总计 `1,019,893,574 bytes`；`model.safetensors=1,015,025,832 bytes`，SHA-256
  `d05b567e...91a2`，执行前复核一致；
- adapter：限定本地模型根盘点无 `adapter_config.json`，没有可证明兼容的真实 adapter；prompt 不记作 adapter；
- Apple M5 MacBook Air、24 GB unified memory、MPS、PyTorch 2.8.0、Transformers 4.49.0、float16；
- 固定 text-only 输入 `Answer with one word: clear`，greedy 2 tokens；
- base load 1 次 warmup + 3 次测量；每个测量进程内 inference 1 warmup + 1 measurement；120 秒 timeout；
- 不清 OS cache，不改全局配置；因此只称 process-first load。

### 原始结果

| 量 | 三次原始值 | 中位数 | 解释 |
|---|---|---:|---|
| base process-first load | 438.829 / 452.153 / 445.778 ms | 445.778 ms | OS page cache 未知；非 storage-cold |
| minimal inference | 26.728 / 28.992 / 27.709 ms | 27.709 ms | 3/3 成功，均生成 `Clear.` |
| process max RSS raw | 2,368,733,184 / 2,369,306,624 / 2,369,503,232 | — | macOS `ru_maxrss`；进程峰值，不是系统总内存 |
| MPS current allocation after load | 1,014,973,184 bytes（三次相同） | — | PyTorch 当前 allocation，不是硬件峰值 |

adapter 文件字节与 load/switch time 为 `unavailable`，没有下载或伪造 adapter。

应用状态使用预先列出的 `explicit_application_workflow_state_v1`：workflow/version、完成节点、当前节点、后续所需
结构化输出、conversation summary 和 model identity。它不是 KV cache、Python runtime dump 或完整执行上下文。

| 量 | 三次原始值 | 中位数 |
|---|---|---:|
| canonical JSON | 732 bytes（固定） | 732 bytes |
| save + flush + fsync + atomic replace | 0.140 / 0.129 / 0.141 ms | 0.140 ms |
| read + parse restore | 0.027 / 0.024 / 0.026 ms | 0.026 ms |

3/3 canonical payload 相等，3/3 deterministic next-node key 相等。该结果只证明这份应用 payload 可恢复；不能称
完整 workflow-state migration 成本，更不能外推到车载、RSU 或无线网络。

## 工作 D：动作可达性与有界配对见证

### 动作审计

`semantic_discrete_5` 的合法 ActionAdapter 输出为：

- action 0：当前 RSU 缓存当前节点 adapter；
- action 1：预测下一 RSU 预取当前节点 adapter；
- action 2：vehicle fallback；
- action 3：当前 RSU steady offload；
- action 4：预测 handoff target 预取当前节点 adapter，并输出 migration mode=`prepare`。

所有 cache action 的 adapter 都绑定当前节点；不能直接准备未来 adapter。adapter 能输出的 migration mode 只有
`keep/prepare`，没有 `migrate`。也没有“继续原 RSU 执行并转发结果”的动作或结果转发字段。因此本轮停止 explicit
migrate 和 old-RSU forwarding 性能见证，没有手工复制状态、改缓存或推进 DAG。

### 固定设计与预算

设计在运行前保存为 `paired_witness_design.json`。固定当前请求 `b0.a0`、136 MiB、LRU、candidate semantics、
rsu_a→rsu_b 移动、空初始 cache、seed 7、最多 4 steps。两对为：

1. 单节点短尾 vs `b0.a0→b0.a0→b0.a0`；
2. `b0.a0→b1.a0` vs `b0.a0→b0.a0→b1.a0`。

四规则为剩余信息消融 action 0、原生两步前瞻、完整尾部复用 action 1、同信息权限的 action 4 prepare。共
`4 instances × 4 rules = 16 episodes`，低于 24 上限；没有统计显著性结论。

### 结果

短尾/无复用两个实例中四规则首动作均为 action 0，结果完全相同。两个长尾实例中，两步/full-tail action 1 与
full-tail action 4 均从 action 0 翻转，共 6 次 pairwise decision change。

| 长尾实例 | 规则 | 首动作 | 完成节点/总节点 | 成功/失败 step | base+adapter 传输 | 结论 |
|---|---|---:|---:|---:|---:|---|
| b0,b0,b0 | 信息消融 | 0 | 3/3 | 3/1 | 208 MiB | workflow 完成 |
| b0,b0,b0 | 两步/full-tail prefetch | 1 | 2/3 | 2/2 | 104 MiB | 少完成1节点，低传输不是收益 |
| b0,b0,b0 | full-tail prepare | 4 | 3/3 | 3/1 | 104 MiB | 同完成/失败，少104 MiB模型字节 |
| b0,b0,b1 | 信息消融 | 0 | 3/3 | 3/1 | 344 MiB | workflow 完成 |
| b0,b0,b1 | 两步/full-tail prefetch | 1 | 2/3 | 2/2 | 104 MiB | 少完成1节点，低传输不是收益 |
| b0,b0,b1 | full-tail prepare | 4 | 3/3 | 3/1 | 240 MiB | 同完成/失败，少104 MiB模型字节 |

action 4 的 `migration_prepare_realized=true`，但原生事件 `state_migration_size_mb=0`。由于没有状态 payload/object，
这应判为 `unavailable_prepare_realized_but_no_state_payload_bytes_emitted`，不是“零成本迁移”。

### 敏感性而非实测替代

两个长尾 case 的 byte break-even 相同：

```text
true_state_payload_bytes < 109,051,904 bytes (104 MiB)
```

若仅把实测 732-byte 应用 payload 作为明确且不完整的假设，净少传 `109,051,172 bytes`；但它没有包含 KV、模型
runtime、未落盘中间对象或协议开销，不能当真实状态。未测网络，故只给符号公式：

```text
delta_seconds = (true_state_payload_bytes - avoided_model_bytes) * 8 / link_bits_per_second
```

固定协议/排队/无线重传时延均 unavailable。本机 1.015-GB SmolVLM 与 native 96/128-MiB synthetic base 身份不同，
没有把 445.778 ms 乘入仿真，避免混证。

## 问题—证据—限制—下一步

| 问题 | 证据 | 限制 | 下一步 |
|---|---|---|---|
| 共享与顺序是否改变同完成成本？ | candidate 四格均完成72请求；blocked+sharing on 为3,264 MiB，其余8,640 MiB | synthetic size；无时延 | 保留机制见证，不升级部署 claim |
| 哪些成本真实？ | SmolVLM bytes/load/inference；732-byte应用状态保存恢复 | 无真实adapter；状态不完整；Apple M5 only | 取得版本匹配adapter和完整state export/import |
| 剩余信息是否改变决策？ | 两个长尾实例各3种信息/方法比较发生6次首动作变化 | 未来adapter、explicit migrate、result forwarding不可达 | 先冻结动作合同再扩大问题 |
| 改变是否改善权衡？ | action4在两个长尾中同完成/失败少104 MiB模型传输 | state/network成本缺失 | 测完整state与真实传输路径 |
| 简单前瞻是否足够？ | 4/4实例两步与full-tail首动作相同 | 仅4个确定性实例/4-step | 未出现超两步决策差异前不加oracle |
| 是否支持改算法？ | 否；当前 blocker 是cost/action contract而非策略表达 | descriptive、无formal/holdout | 补系统测量与动作语义；另立任务审查 |

## 回答六个研究问题

1. **共享与顺序效应**：共享只在 blocked 顺序创造 base reuse；它把 base load 72→24、总传输 8,640→3,264 MiB。
   interleaved 下 sharing on/off相同；旧语义的低传输来自拒绝服务。
2. **实测与假设**：base文件、进程首次加载、最小推理、明确应用状态保存恢复已实测；真实adapter、完整迁移状态、
   车载/RSU性能、网络传输/时延仍 unavailable。104 MiB 阈值是敏感性条件。
3. **是否改变决策**：是，但只在2个长尾实例；短尾/无复用不变。
4. **是否改善权衡**：action4在原生 bounded witness 中同完成/失败少104 MiB模型字节；加入真实完整状态与网络成本后
   是否仍改善尚不可验证。action1的低字节伴随少完成节点，不能算改善。
5. **两步是否足够**：本轮是。full-tail从未超过两步前瞻改变初始动作或结果。
6. **是否支持下一轮算法修改**：否。至少缺真实兼容adapter、完整state payload/restore、可达future-adapter或explicit
   migrate/forward合同、实际网络测量，以及一个预注册且两步前瞻不足的实例。

## 产物、失败记录与边界

机器证据位于 `artifacts/analysis/remaining_workflow_decision_value_audit_20260930_v1/`：冻结计划、原生账本、动作
可达性、原始本机测量、16-episode raw witness、敏感性综合、执行记录、主工作区保护和完整性 manifest。

开发期 ledger consumer 首次读取 `dependency_bundle.ordered_object_ids` 时误按 object dict 解析，第二次误用
`resident_object_ids` 而非原件的 `native_object_ids`；两次均在生成任何新科学 episode 前失败，随后按原始 schema
最小修正。正式四配置输入未重跑，paired witness 只执行一次16 episodes，本机测量只执行冻结的1 warmup+3次。

未训练、未扩大数据集、未运行 formal/holdout、未重新选模、未检索/新增文献、未下载模型或数据。结果不支持 SA、
PPO 或其他算法优势，也不支持 TMC-ready/canonical 晋级。
