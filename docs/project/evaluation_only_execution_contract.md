# Evaluation-only Execution Contract

- `version`: `1.0.0`
- `goal`: `G14R20-I`
- `status`: `READY_FOR_EVALUATION_ONLY_AUTHORIZATION`
- `formal_execution_authorized`: `false`
- `formal_execution_started`: `false`
- `holdout_opened`: `false`

## 两条身份链

模型来源固定为旧 run `typed_model_cache_formal_20260906_152847_g14c_v16`、科学 commit
`a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d` 和 H 审查列为 `verified` 的 150 个冻结 checkpoint。
`evaluation_model_source_reference.json` 绑定 selection/freeze、6-resource generated registry、三容量 companion、
checkpoint path/size/SHA-256、10 agent × 5 seed × 3 capacity 坐标及原 Protocol 2.9 context/binding。checkpoint
loader 的 expected identity 始终由这条来源链计算。

新评估执行使用不同 run ID、executor commit、command matrix、ledger、lock、staging 和 output。新 run 不拥有也不
声称完成 training、dev selection 或 checkpoint freeze；gate 对这些新 run 计数要求为 0，并另行要求来源侧
150 个冻结模型。普通 generated registry 仍限定 same-run；只有显式、hash-bound 的上述 source reference
允许本轮固定 v16 来源跨 run 只读消费。

## 执行入口与权限

`scripts/prepare_typed_model_cache_evaluation_only.py` 只生成/验证未签发授权申请，不创建 run root、ledger、lock、
staging 或 grant。`scripts/run_typed_model_cache_evaluation_only.py` 只接受绑定精确申请、最终 executor commit、
独立 review、有效期和八阶段的项目 grant；缺失、false、过期、身份漂移或 holdout 扩权均在写入前拒绝。

阶段固定为 `formal_cache_policy → formal_controller → formal_ablation → formal_support → formal_scalability →
formal_statistics → formal_gate → complete_without_holdout`。没有 train/dev/freeze/holdout action，也没有旧 run
recovery/finalize 接口。每次只允许下一个未启动阶段；cell publication、phase ledger、single-writer 和合法 gate
completion 复用现有真实消费者。

## 发布完整性

cell publisher 先验证 producer manifest 与完整 payload，再只允许以下重定位：episode
`run_info.summary_path`、run manifest 的四个 `output_paths` 字段、`request_replay_path`、support `output` 和
`resolved_command.txt` 的绝对命令路径。每个变换记录 before/after SHA-256 与路径归一化后的非路径语义 hash；
metric、request、null、row count、model identity 或其他字段变化均拒绝。

路径变换后重建并验证最终 producer manifest，随后 transaction inventory 将该 manifest 纳入 hash，再写 marker
并原子发布。发布后、重复消费、ledger recovery、formal integrity/gate 都从最终目录重读并同时验证 producer 与
transaction 两层清单。旧 marker 若含 producer manifest 但没有新绑定会 fail-closed；不删除内部 manifest，也不
允许 transaction inventory 掩盖其错误。

## 旧结果和科学边界

旧 288 MB committed cell 只保留为 partial/formal-exposure 证据；576 MB staging 只保留为失败证据。二者路径在
新执行合同中列入排除集，不得进入新 statistics。新任务获批后必须按原 Protocol 2.9 统一重新评估三容量；不得
据旧结果改 checkpoint、窗口、指标或比较规则。formal 已 seen 的事实永久保留；holdout 仍 sealed/unopened，开启
需要另一任务的一次性授权。

本合同只说明执行链已准备，不是正式实验、性能结果、paper-ready 或 holdout 授权。
