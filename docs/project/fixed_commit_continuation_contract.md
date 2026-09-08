# Fixed commit continuation contract 1.0.0 — G14R20-A

日期：2026-09-08。开发基线：`c8e67bafaba6a6cfd50f8cebde6c1fbdcd3c7952`。
本文件是独立提出的执行规则修订，**不是 Protocol 2.9 原有授权**，不修改或重新激活旧 Protocol/index/Readiness。
本任务只批准实现、合成测试和真实只读验收；真实 proposal 的 `execution_authorized=false`。

## 状态勘误与作用范围

G14R18 补验任务新增正式训练/执行为零；这不表示整个项目没有既有 v16。
既有 `typed_model_cache_formal_20260906_152847_g14c_v16` 在 execution commit
`a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d` 下已完成 preflight、tests、train、dev_select、checkpoint_freeze：
150 train cells、24 dev cells committed，1,200 candidates、150 selected/frozen checkpoints、6 generated resources。
phase ledger 有 15 条，cell ledger 有 348 条；没有 failed terminal，没有后续 formal 阶段。
恢复受阻不等于 run 失效，不要求重训，不允许新 formal evaluation 使用 checkpoint。

工作树：`/private/tmp/ppo_mec_g14c_v16_a6d1fd8_20260906_152847`。
run root：`/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_formal/typed_model_cache_formal_20260906_152847_g14c_v16`。

追加 correction record 位于
`artifacts/analysis/g14r20_a_continuation_20260908/correction_record.json`，记录原路径、完整 SHA-256、原行号/JSON 字段、
原声明、任务作用范围、核对时间及原始账本证据。原 JSON、日志、manifest、2026-09-06 历史正文均保留。
此勘误不追认原始启动批准。发现的 G14R18 authorization deferral 是暂缓证据，不能冒充启动授权。

## A 新 run 与 B 固定提交 continuation

新 run 继续使用原 public runner 的 clean `HEAD == origin/main` 发布门禁。没有关闭、替换或 monkeypatch。
固定提交 continuation 候选绑定原 execution commit、tracked Git tree 和 tracked src/scripts 内容 hash；
当前 main/origin/main 只出现在独立 observation 中，不进入 checkpoint、Protocol、binding/context 的重绑定。

独立 schema：`configs/experiment/fixed_commit_continuation_v1/continuation_schema.json`。
权威语义 validator：`src/runtime/fixed_commit_continuation.py`，除结构外检查路径、重复证据、ledger 链等跨字段条件。
真实实例：`artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json`。
该 schema 只接收 proposal；不能把批准人、签名、`approved=true` 或新 executor 填入该对象获得执行权。
未来 B 的批准证据必须独立于 proposal 保存，不能修改本轮 proposal 冒充已批准。

身份覆盖：精确 run/root/worktree、原 commit/tree、Protocol/bundle/environment、binding/context/command matrix、
candidate/selection/freeze/registry、150 checkpoint 文件、6 companion 资源、phase/cell prefix。
全量原始产物与旧源码/配置以 `protected_before.json.gz` 的严格 JSON inventory 绑定；gzip 仅压缩存储，
每项保存原路径、size、SHA-256 或 symlink target。它是本轮保护基线，**不是原始用户批准证明**。
所有真实写入范围目前为空。future write scope 仅为精确原 run 下独立批准的合法追加；旧结果不可替换。

## 信任与授权状态

`proposal → structure pass → evidence qualification → independent approval → execution` 是五个不同状态。
结构通过只表示字段合法。兼容检查通过不能证明有原始启动批准、发布证明或 continuation 批准。
本轮未找到独立可验证的原始 launch approval / release attestation，分别报告 unavailable；
现有 preflight、Readiness、Git ancestor、当前 HEAD 或 agent 自算 JSON hash 均不能填补此缺口。
执行器未实现及独立批准不存在分别为 pending。无真实 token、签名或 approved 字段产生。

A 的授权检查始终拒绝真实执行，没有 production trust store。
合成 fixture 的 test-only HMAC 验证依赖测试独立持有的 key/root，绑定完整 proposal、精确 run、scope、expiry/revocation。
它仅返回 `fixture_approval_valid`，执行授权仍为 false；CLI 不接受 key/approval 参数。
真实 v16、无 synthetic 前缀、跨 fixture root、跨 run、scope 修改、自算 hash 冒充签名、过期、撤销均拒绝。
未来真实授权验证机制及批准来源必须在 B 中独立审查；本轮不假装已经实现。

## 账本锚点与合法进度

锚点包含 record_count、byte_count、原始字节 SHA-256、terminal hash、run fingerprint。
每次读取都验证原前缀、完整 successor hash chain、sequence、run identity；prefix 不得截断、插入、重写或分叉。
批准绑定获批起点和合法后继，不要求合法追加后的 tip 永远等于旧 terminal。
phase 后继只能按本文件八个 phase 顺序追加，前 phase completed 后才能进入下一 phase。
既有 train/dev_select/checkpoint_freeze 不得重跑；committed cell ID 不得重复；failed terminal 永不因新批准变为可恢复。
phase/cell 原生 validator 仍运行。原始 committed ledger inventory、marker、当前文件身份相互核验。

只有测试调用可将经过 fixture trust 验证的 phase 列表交给 successor validator。
真实 CLI 没有独立批准，所以任何新增 phase 都拒绝；它不会写 running、锁、失败记录或其它 run 文件。
本轮全量保护快照还用于证明“本次验收完全没有写入”，不能作为未来 B 的永久 tip 锁定规则。
B 必须把获批 prefix 和 immutable committed payload 与允许的新输出分开，并交叉证明 phase/cell 进度一致；
A 的独立 ledger 函数不声称覆盖真实事务调度、跨 ledger 时序、单写者或故障恢复。

## 只读 CLI 与环境边界

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/preflight_fixed_commit_continuation.py \
  --proposal artifacts/analysis/g14r20_a_continuation_20260908/v16_continuation_proposal.json \
  --check evidence
```

`--check structure|evidence|authorization`；默认 evidence。输出只到 stdout，无 execute/dispatch/output-root/approval 参数。
exit 0：结构通过；exit 2：失败或授权拒绝；exit 3：只读兼容项通过但资格证据/执行器/批准仍 unavailable。
原 gate 当前失败在独立项报告，不自动等价于 fixed-source candidate 失败，也不被隐藏。
`require_origin_main_match=False` 只用于独立子进程内的 read-only candidate bundle 检查。
替代条件是固定 source + 原始发布/启动证据 + 独立 continuation 批准；后两项仍缺失。

子进程采用原 context 的绝对 Python、旧 cwd、旧 PYTHONPATH，`-I -B`、关闭 user-site/字节码；
实际 interpreter/cwd/package versions/dependency identity、editable installs、完整 loaded src/scripts origins 和 source hash 均检查。
implementation helper 来自本轮独立路径；科学 validator/benchmark gate 来自旧 clean worktree。
环境探针只读取 metadata，不调用训练、策略 rollout 或 evaluation；torch 强制 weights-only 安全读取。
函数调用监测在环境 step/reset、rollout、train/learn、scripts.main 被调用时立即拒绝；正常报告计数为 0。
这不是未来 executor 动态兼容证明，不能用其零调用替代 B 的真实入口覆盖。

## 原检查保留与 producer/consumer 矩阵

| 原生产端 / 身份 | 本轮只读消费者 | 边界 |
| --- | --- | --- |
| active index/Protocol/Readiness/resource bundle | 旧 `validate_active_formal_bundle`，原门禁与候选分别调用 | 原门禁不变；候选不是批准 |
| 原 Python/environment manifest | 旧 `resolve_execution_environment` + 全模块来源检查 | 不只比较 fingerprint |
| resolved context / command templates | 旧 context validator + `validate_command_templates` | outer 186 command matrix；nested 实际调度交 B |
| scientific config / execution binding | 旧 `validate_execution_binding` | 原环境/commit/matrix/bundle，不重绑定 |
| phase/cell ledger | 前缀 validator + 两个旧原生 validator | 不实例化 writer |
| cell inventory / committed marker | 新只读 reconciliation + 全量文件 hash | 174 committed cells，不重跑 |
| candidates / selection / freeze | 原文件 hash；freeze registry terminal 链 | 不读取 dev 分数作恢复批准决策 |
| registry / six generated resources | 旧 `load_generated_checkpoint_registry` | 原 freeze terminal，禁止 latest-tip 重绑定 |
| checkpoint top-level/nested/17-field envelope | 旧 `validate_benchmark_checkpoint_gate` | nullable、Git/window/runtime/capacity/agent/seed、SHA 和 shared identity |
| invalid-run references | 旧 denylist validator | 不恢复永久 invalid run |
| 原 launch/release approval | 独立证据项 unavailable | readiness/实现声明不得补位 |

## G14R20-B 接口与验收清单（未实现）

B 必须提供独立 executor commit 和完整文件 identities，不能冒充原 execution commit；原科学运行模块仍从旧 clean worktree 加载。
审批必须绑定本 proposal/prefix、executor 身份、允许 phases/写入范围、失效/撤销条件，holdout_capability 永远 false。
B 必须保持原 argv、顺序、coordinates、cwd/env、expected outputs、nested expansion 和 command hash；
证明 registry → phase input → run/cell identity → staging/descriptor → publication → terminal 全链一致。
只保留旧事务已有的合法 staging 重写，不任意改输出约定。证明单写者、幂等、故障恢复、gate/completion 语义未变。

| phase | B 必须补充的真实入口兼容证据 |
| --- | --- |
| formal_cache_policy | outer/child 三容量展开、同一 generated resource 透传、cell input 与发布目录 |
| formal_controller | 原 agent/order/seed/window、controller argv 与 committed-only consumer |
| formal_ablation | setting coordinates 与 output descriptor、原配置差异边界 |
| formal_support | 每个 support setting 的 nested argv、child-output descriptor 与 support provenance |
| formal_scalability | capacity/workload expansion、资源映射与 expected output inventory |
| formal_statistics | 原 committed rows 输入、完整性、nullable/paired/cluster 消费语义 |
| formal_gate | 真实 statistics/registry 身份，passed=false 必须阻止完成 |
| complete_without_holdout | 合法 gate terminal 后才完成；holdout 未开启，无绕过 finalize |

本轮未实现、未调用上述 phase；纯函数和零 rollout 计数不是它们的入口验收。
B 尚需关闭：原始发布/启动证据缺口、独立批准真实性、executor 身份与安全加载、跨 ledger 时序、故障注入、
单写者与恢复锁、8-phase 全链实际 consumer 验收。本轮不启动 G14R20-B、v16-B、G14D 或 G15。
