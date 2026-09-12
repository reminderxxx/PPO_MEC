# Evaluation-only Execution Contract

- `version`: `1.0.0`
- `goal`: `G14R20-I`
- `status`: `READY_FOR_EVALUATION_ONLY_AUTHORIZATION`
- `formal_execution_authorized`: `false`
- `formal_execution_started`: `false`
- `holdout_opened`: `false`

## 2026-09-13 I3 producer integrity 修复

G14E02 `typed_model_cache_evaluation_only_20260912_g14r20_i2_pending` 已在首个
`formal_cache_policy` cell 以 `failed_terminal` 终止：科学 child rc=0，但 producer manifest 声明 2,706
项、实际 payload 2,708 项，遗漏 `comparison_against_popularity.json` 与
`sa_advantage_diagnosis.json`。0 cell committed，旧 staging、ledger、held lock、grant 和原始证据必须原样
保留；该 run、I2 申请和 grant 均不可恢复、重试、改写或复用。I2 的 bootstrap/handoff 工程验收只保留为
当前回归依据，不再构成执行授权。

I3 仅将 benchmark producer 已写出的两个诊断文件加入其原始 `integrity_files`。producer validator、精确成员
校验、路径白名单重定位、非路径语义证明、transaction inventory 和原子发布均保持不变。真实小型
`benchmark_main_results.py` 非正式验收覆盖 fairness manifest 有/无、发布前及重定位后校验、descriptor→cell
transaction→原子发布→双层重复回读，以及漏成员、缺文件、hash/size 漂移、额外文件、重复成员五类拒绝。
这些执行是受控 non-formal rollout，不是 formal performance；本修复没有训练、选模、freeze、正式评估或
holdout。

新的 unsigned request 必须在最终 executor commit 的持久 clean checkout 上 create-only 生成，使用全新 run ID
和不存在的 run root，并通过真实 `--action validate`。prepare/validate 不签 grant、不创建 run/ledger/lock/
staging，也不启动 G14E03。

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

## 2026-09-11 I2 启动补充

首次 bootstrap 与阶段衔接以 [I2 合同](evaluation_only_bootstrap_handoff_contract.md) 为准。G14E01 已有失败
现场，旧申请/grant 当前不可执行；本文件历史 readiness 描述不构成重试授权。新初始化 marker 和完整锁覆盖
只作用于全新批准 run；本轮只生成新的 unsigned 申请并 validate。
