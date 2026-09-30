# 真实 adapter 兼容性与工作流状态恢复成本校准

## 审查元数据

- `reviewed_at`: `2026-09-30`
- `literature_cutoff`: `2026-09-30`
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `adapter_state_recovery_calibration_20260930_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `fixed_execution_commit`: `e6fffb5099563f66641a33a3ff0b70a33c852859`
- `runner_sha256`: `3418b1b063afaa0ce08ee3acf0fb75e0d347eeea6e0aeecd0c5ccc680e3e03e9`
- `result_commit`: `PENDING_DELIVERY_COMMIT`
- `evidence_level`: `E1_BOUNDED_FAILED_CALIBRATION_WITH_NATIVE_LEDGER_REUSE`
- `verdict`: `UNVERIFIED / STOPPED`

本轮固定在上一轮最终提交 `e6fffb5` 的隔离 checkout 执行，没有更换基线、训练、选模、扩展矩阵、运行
formal/holdout、修改 production action codec 或覆盖冻结产物。结论先行：两个目标均未通过验收。真实 adapter
因本地资源不足且本轮没有新增下载授权而为 `unavailable`；状态链在一次预热中保存了声明状态，但目标进程按预先
冻结的正确性条件拒绝该状态，没有完成恢复及最终输出对照。原生 action 4 的两项 104 MiB 合成账本见证保持不变，
但不能据此判断完整真实工作负载的净收益。

## 1. 真实 adapter 资源与兼容性

本机递归检查项目资源和 Hugging Face cache，没有发现 `adapter_config.json` 或 adapter 权重。已存在的真实 base 是
[`HuggingFaceTB/SmolVLM-500M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)，
固定 revision `a7da5b986cb59b408707209984f360a5f4ad7e47`，Apache-2.0，架构
`Idefics3ForConditionalGeneration` / `VLlama3ForCausalLM`。`model.safetensors` 为 1,015,025,832 bytes，
SHA-256=`d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2`；本地仓库 13 个文件合计
1,019,893,574 bytes。

| 资源 | 固定 revision | 声明关系与接口 | 权重字节 | 许可/访问 | VEC 任务解释 | 本轮验收 |
|---|---|---:|---:|---|---|---|
| SmolVLM-500M-Instruct base | `a7da5b9…` | 公共 `AutoModelForVision2Seq`；无需 remote code | 1,015,025,832 | Apache-2.0；已在本机 | 通用视觉语言 base，不单独证明 VEC 任务质量 | 身份/hash 已核验 |
| [ALPR LoRA](https://huggingface.co/Hirai-Labs/FT-SmolVLM-500M-Instruct-ALPR) | `edccd0e…` | 声明同一 base；PEFT LoRA，rank 64，DoRA | 154,423,432 | Apache-2.0；公开、非 gated、无需账号 | 路侧车牌识别可作为 RSU 视觉服务；模型卡不证明质量 | 未下载、未加载、未执行 |
| [Helmet LoRA](https://huggingface.co/revitotan/FT-SmolVLM-500M-Instruct-Helmet) | `96794cb…` | 声明同一 base；PEFT LoRA，rank 4 | 9,641,944 | Apache-2.0；公开、非 gated、无需账号 | 工区头盔合规可作为路侧监测服务；模型卡不证明质量 | 未下载、未加载、未执行 |

两 adapter 新权重最小合计 164,065,376 bytes（156.465 MiB），base 无需重下。两份配置均出现
`task_type=null`，Hub 将其标为配置风险；必须通过真实加载见证才能确认兼容。固定环境缺少 `peft` 与
`accelerate`，且不得修改；后续应使用新的隔离环境。依赖下载大小与固定许可输入尚未确定。本轮没有新增权重或依赖
授权，实际下载、adapter 进程和 adapter 调用均为 0。以上只形成最小资源清单，不形成工程兼容性、输出隔离或任务质量
证据。没有发现需要新增到文献表的论文；SmolVLM 论文已在现有文献表登记。

## 2. 测量前冻结方案

完整方案在 `measurement_plan.json`。方案先于模型调用冻结，并限定：每项最多 1 次预热、3 次测量；单进程
180 s 超时；不因结果改输入、模型、正确性条件或重复次数；失败即停，只有配置错误可以保留记录后做版本化修正。

- 设备：Apple M5 MacBook Air，24 GiB，macOS 26.6.2；Python 3.9.6，PyTorch 2.8.0，Transformers 4.49.0，
  Safetensors 0.5.3；MPS/float16；intra/inter-op thread 均为 1。
- 同步：模型加载与每次 `generate` 的计时边界均调用 `torch.mps.synchronize()`；内存仅记录进程
  `ru_maxrss` 和 MPS current/driver snapshot。
- 冷热措辞：只称 `process-first load`；未清理全局 cache，不能称存储冷启动。
- 固定输入：非安全关键的路侧 `lane status is clear` 文本，两节点均 greedy，最多 3 个新 token，seed 7。
- 正确性：中间 token IDs、解码文本、规范化状态严格相等；状态 canonical bytes/hash、不可变资源身份、已完成/剩余
  DAG 和 next-node 严格校验；最终 token IDs/文本严格相等且规范化输出必须为 `clear`。浮点
  `rtol=atol=0`，非空输出不算通过。
- 路径 A：单独进程 process-first 加载 base，连续执行两个节点。
- 路径 B：source 进程加载 base、执行节点 1、原子保存并退出；此后才启动 target 进程，target 只读取声明状态、冻结
  plan、固定模型/processor 和 runner，重建节点 2 并执行。
- 资源上限：修正后本轮最多 15 个模型进程、20 次模型调用、20 分钟；实际为 3 个进程、3 次成功生成调用。

方案版本历史也保留在机器证据：v1.0 在测量前将四组路径所需进程误算为 10，修正为 v1.1 的 12，不改科学设计；
v1.1 预热完成后，结果 finalizer 将 Python 布尔值写成小写 `false` 而 `NameError`，没有保存预热细节，也没有启动正式
测量对。仅修正 `false`→`False` 后冻结 v1.2；没有改模型、输入、任务、重复数、计时或正确性。v1.1 原失败记录单独
保留，不能用于计时或科学结论。

## 3. 原始执行、计时、内存与失败

v1.2 只执行一组预热，预热失败后正式测量数为 0，没有重试或换 prompt。

| 路径/进程 | PID | process-first load | 推理/状态 | workflow/process | 峰值 RSS | 输出或终态 |
|---|---:|---:|---:|---:|---:|---|
| A continuous | 79354 | 695.009 ms | 节点1 210.802 ms；节点2 103.680 ms | workflow 2,058.571 ms | 2,379,005,952 B | 节点1 ` lane status.` / `[24289,3559,30]`；节点2 ` The upstream lane` / `[378,22302,24289]` |
| B source | 79355 | 566.101 ms | 节点1 153.126 ms；序列化 0.066 ms；durable save 0.394 ms | workflow 1,772.454 ms | 2,378,760,192 B | 保存 3,085 B，SHA-256=`6200a8b7…dbc6` |
| B target | 79356 | 未开始 | 验证后拒绝 | process 34.224 ms | 未记录 | `ValueError: upstream normalized output failed frozen correctness criterion` |

continuous 与 source 的节点 1 完全一致，但都生成 `lane status.`，规范化为 `lanestatus`，不等于预先冻结的
`clear`。target 在模型加载和节点 2 执行前按合同拒绝状态。因此 3,085 bytes、0.066 ms 和 0.394 ms 只是在失败预热中
观察到的诊断值，不是完整已验证迁移状态或正式成本估计。continuous 的第二节点也没有达到最终标准。内存值是各进程
可观测范围，不可相加为系统峰值。

## 4. 状态边界与独立进程恢复对照

选择的边界是 `normalize_event` 完成、`publish_event` 未开始。声明必须传输：workflow/version/boundary、completed/
remaining node IDs、DAG edges、next node/attempt/termination、节点 1 原始文本与 token IDs、规范化状态、固定生成合同、
base 身份、processor/tokenizer/config hashes 和状态内容 hash。

- 随机状态不包含：两节点均 `do_sample=false`，下游不消费 RNG continuation；seed 仍记录为 7。
- KV cache 不包含：这是应用节点边界，节点 2 消费显式上游输出而不是 token continuation。
- tensor state 不包含：没有跨该边界的下游 tensor。
- 目标端预先存在：固定本地 base、processor/tokenizer/config、冻结 plan 和 runner。
- 目标端重建：节点 2 prompt、tokenization tensors 和模型 runtime allocations；方案要求分别计时。
- 网络传输未测；同机文件可见性不等于 RSU 链路。

source 已退出后才启动不同 PID 的 target；target 没有读取 source 内存或未声明临时文件。状态临时文件在失败处理后删除，
其 size/hash 和完整命令保留在结果 JSON。由于 target 在正确性门禁处停止，路径 B 没有“恢复并完成”，也没有最终输出
可与路径 A 比较。本轮只能证明 fail-closed 的独立进程读取/验证边界，不能称恢复见证，更不能称跨 RSU 实测。

## 5. action 4 原生语义

`semantic_discrete_5` 的 action 4 将**当前节点所需 adapter**预取到预测 handoff target，并记录
`migration_action.mode=prepare`；当前 RSU 的 steady 执行仍可进行。后续 handoff 与目标匹配时账本记录
`migration_realized=true`。它不改变当前/未来 workflow node，不序列化、发送或导入应用/runtime 状态，也没有扩展
production action codec。

冻结两个长尾实例各出现一次 `migration_realized`，但 `state_migration_size_mb=0` 是没有解析到 workflow-state
object/payload 字节，不是测得“状态为空”或“迁移免费”。即使本校准器未来恢复成功，也不能写成 action 4 原生已经支持
真实状态迁移。

## 6. 原账本与校准/敏感性分层

### 6.1 原生合成账本（原样保留）

| 冻结实例 | action 0 模型传输 | action 4 模型传输 | 少传 | 完成节点 action0/action4 | 失败 action0/action4 | 状态字节解释 |
|---|---:|---:|---:|---:|---:|---|
| `tail_length__long` | 218,103,808 B | 109,051,904 B | 104 MiB | 3/3 | 1/1 | payload 未记录，非零成本证据缺失 |
| `tail_adapter_structure__long` | 360,710,144 B | 251,658,240 B | 104 MiB | 3/3 | 1/1 | payload 未记录，非零成本证据缺失 |

此层没有改 DAG、容量、移动条件、参数或历史产物。104 MiB 是原合成模型字节条件，不是真实 SmolVLM 或 adapter
传输量；本轮没有用真实模型字节覆盖它。

### 6.2 实测校准

`validated_state_bytes`、正式 save/restore、target rebuild、target model load、adapter switch 和 waiting cost 全部
`unavailable`。失败预热的 3,085 bytes 不能代入为完整迁移状态。模型/状态完整字节收支、时间收支和真实完成/失败收支
因此不可判定。

### 6.3 预先冻结的有限敏感性区间

仅对原合成少传 104 MiB 做网络代数，不称无线实测：

| 假设链路 | 少传 104 MiB 的纯传输时间 | 固定 0/5/20 ms 时的 network-only 状态字节上限 |
|---:|---:|---:|
| 10 Mbps | 87,241.523 ms | 109,051,904 / 109,045,654 / 109,026,904 B |
| 100 Mbps | 8,724.152 ms | 109,051,904 / 108,989,404 / 108,801,904 B |
| 1,000 Mbps | 872.415 ms | 109,051,904 / 108,426,904 / 106,551,904 B |

字节必要条件是 `true_validated_state_bytes < 109051904`。时间盈亏平衡为：

```text
state_save + state_transfer + state_restore + target_rebuild + target_model_load
+ adapter_switch + waiting
< avoided_model_transfer + avoided_model_load + avoided_switch
```

表格尚未加入 save/restore/load/switch/waiting、排队、重传或链路竞争，因此不是充分条件或部署结果。关键量未知时，
结论保持 conditional/unavailable；没有事后调权或综合分数。

## 7. 对现有决策见证的影响

- **工程兼容性**：未建立。公开 adapter 只有声明身份；A→B→A 没有实际加载、执行、参数键/active identity 或切回隔离见证。
- **任务正确性**：未建立。固定 base 在预热中未满足预注册输出标准；未评价 ALPR/Helmet 质量。
- **成本校准**：未完成。状态链在正式测量前失败；本地 process-first load 也不等于模型无线传输。
- **机制效益**：原生 104 MiB 合成字节见证仍完整，但真实净收益从未验证状态下降为仍不可判定；失败恢复和 adapter
  unavailable 削弱进入真实工作负载效益声明的准备度。
- **算法优越性**：未评估，也不由前三项推出；没有 SA、PPO、MAPPO 或其他方法比较。

## 8. 下一轮门禁与停止点

当前**不具备**小规模方法比较条件。中央复核后若明确授权，最小下一步是：

1. 只下载上述两个固定 revision 的 adapter 权重（合计 164,065,376 bytes），另建隔离环境；先审查并处理
   `task_type=null`，不得执行 remote code；固定许可输入及其 hash 后再运行 A→B→A。
2. adapter 验收必须记录真实 active name、加载参数键/非零 tensor、A/B 固定输入输出、切回 A 的 identity 与输出复现；
   任务质量另行验收，不能由加载成功替代。
3. 状态链需要另立、预先冻结的新 calibration task/input；连续与恢复两条路径均先满足任务正确性，再做 1+3 次测量。
   不得把这次失败后换 prompt 当作同一方案重试。
4. 取得已验证状态字节、save/restore/rebuild/load/switch 时间以及独立网络假设/测量后，才可完成净成本表；仍需把
   校准器支持和 production action 4 支持分开。

本轮按“新增资源授权需求或终态失败后停止”收口，交中央复核。

## 9. 机器证据与复算入口

- `artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/adapter_resource_inventory.json`
- `artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/measurement_plan.json`
- `artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/failed_attempt_v1_1.json`
- `artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/workflow_state_recovery_measurements.json`
- `artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/cost_sensitivity_synthesis.json`
- `artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/workspace_protection_snapshot.json`
- `artifacts/analysis/adapter_state_recovery_calibration_20260930_v1/integrity_manifest.json`
- `scripts/calibrate_workflow_state_recovery.py`
- `scripts/synthesize_adapter_state_recovery_calibration.py`
- `tests/test_workflow_state_recovery_calibration.py`

执行时 `HEAD` 固定为上一轮最终提交，新增 runner 以内容 SHA-256 绑定并在本轮交付提交中纳入 Git；因此证据能绑定到
源内容，但不能声称新增 runner 在执行前已存在于一个 clean committed tree。该 provenance 限制保留，不为获得更高证据
等级重跑失败实验。
