# Formal Environment Identity Projection Contract

## 冻结身份

- `formal_environment_identity_projection_contract_version=1.0.0`
- `formal_execution_environment_contract_version=1.1.0`
- active Protocol：`2.1.0`
- readiness：`READY_FOR_G14C_V11_CLEAN_TRAIN_AND_FORMAL`

Protocol 2.1 的唯一 scientific environment producer 是
`src.runtime.formal_execution_environment.build_environment_identity_projection`。environment manifest、Protocol、
runtime resolver 与 active bundle validator 不再各自维护字段列表。

## Full projection

Runtime-observable projection 包含 environment contract、Python、platform、architecture、dependency fingerprint、
installed package count、Torch、critical packages、execution commit rule、source-root rule 与 identity rule。
Protocol-bound extension 只允许 endpoint `2.0.0`、exogenous request execution `1.0.0` 和 request exposure trace
`1.0.0`。extension 来自已验证 Protocol，不能由 package probe 推断，也不能从 expected identity 注入额外字段。

Canonical fingerprint 是排除且只排除 `environment_fingerprint` 后，对完整 normalized identity 做 UTF-8、
sorted-key、compact、finite JSON 序列化所得 SHA-256。缺失、未知、重复、别名、错误类型、unsupported major、
NaN/Infinity、逐字段漂移或 fingerprint 漂移均 fail-fast。

Python 绝对路径、worktree 路径、cwd、site-packages、sys.path 与 virtualenv root 仅进入 runtime audit。由于 Git
commit 不能包含自身 hash，manifest 冻结无自引用 commit/source identity rule；实际 40-hex clean HEAD 与 observed
source tree SHA-256 在 active gate、execution binding 和 runtime audit 中记录。旧 `Commit A10/A11` 描述不进入
Protocol 2.1 scientific identity。

## G14C v10 边界

G14C v10 在 durable run 创建前停止，分类为
`PRE-EXECUTION STOP / EXECUTION_IDENTITY_MISMATCH`。clean candidate 是
`/private/tmp/ppo_mec_g14c_v10_8402d2e_20260831_161419`；没有 preflight child、ledger、checkpoint、candidate、
row、formal evidence 或可 resume/salvage 状态。Protocol 2.0/A11 此后 audit-only。

机器证据位于
`artifacts/analysis/typed_model_cache_formal_environment_identity_repair_20260831_g14r10_v1/`。本合同与 rehearsal
不构成性能证据；正式 training/checkpoint/performance 为 0，holdout sealed/unopened，G14C v11 未启动。
