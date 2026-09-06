# Formal Checkpoint Nullable Identity Contract

状态：Protocol `2.8.0`，Readiness `20.0.0`。唯一 active index 为
`configs/experiment/typed_model_cache_formal_protocol_v2_8_20260906/protocol_index.json`。

## 失败边界

G14C v15 `typed_model_cache_formal_20260905_213344_g14c_v15` 已完成 150 个 training cell、24 个 dev
evaluation cell 并形成 1,200 条 candidate；`dev_select` 在排序发布前因 candidate 顶层 nullable contract hash
缺失而失败。嵌套 `formal_training_contract` 中该 hash 正确。`dev_selection.json`、checkpoint freeze、formal gate
与 completion 均未发布，holdout 仍 sealed/unopened/capability=false。该 run、checkpoint、candidate 与 partial dev
永久禁止 resume、retry、finalize、salvage 或复用。

## 身份投影与严格消费

`src/runtime/formal_training_identity.py` 提供 capability-aware 的共享 checkpoint identity 字段集合、producer
projection、可信 Protocol/context expected projection，以及 top-level/nested 一致性校验。active checkpoint 的
顶层字段必须真实存在且为非空字符串；禁止从 nested metadata 静默回填。共享 run identity 与
`agent_identity/training_seed/runtime_contract_sha256` per-cell identity 分开验证。

生产端 `scripts/train_algo_pool_real_sample.py` 从已验证的 `ResolvedTrainingContract` 构建 checkpoint 与 training
summary identity；`latest.pt` 和 candidate checkpoint 在 annotate 后立即 read-back 校验。dev runner 在启动昂贵
benchmark child 前检查实际 checkpoint；selector 在排序前对所有 candidate 与可信 Protocol/context identity
比较；freeze 再次读取 selected checkpoint；typed provenance consumer 同样要求完整 top-level/nested identity。

## 验收与结论边界

G14R17 的结构验收对 150 training coordinates × 8 update indices 共 1,200 个 projection 执行生产 builder 与
JSON round-trip；实际序列化验收覆盖 10 个 learned agent、`latest.pt`/`update_0004.pt` 和三档容量，并执行
150-row strict selection/freeze/typed provenance 链。负例覆盖顶层/nested 缺失、冲突、统一错误 nullable hash、
单 candidate 漂移、跨 Protocol/binding/context/bundle、错误 agent/seed/runtime capacity、序列化丢字段和 v15
引用拒绝；pre-benchmark 负例的 child 调用为 0，失败 selection/freeze 不发布。

验收 checkpoint 全部为 test-only，不进入正式 manifest 或论文结果。Protocol 2.8 不改变算法、reward、action、
超参数、数据、split/window、capacity、selection ordering、nullable 数值语义、statistics 或 holdout 语义。
Readiness v20 只授权未来独立任务创建 G14C v16 clean run；本轮未启动 v16、formal、holdout、G14D 或 G15。
