# Formal Training Entrypoint Acceptance Contract

状态：Protocol `2.7.0`，Readiness `19.0.0`。唯一 active index 为
`configs/experiment/typed_model_cache_formal_protocol_v2_7_20260905/protocol_index.json`。

## 根因与失效边界

G14C v14 `typed_model_cache_formal_20260905_185105_g14c_v14` 的首个
`sa_ghmappo / seed 7 / constrained_288mb` training cell 在 episode 0 前触发 `NameError`：
`resolve_training_contract()` 的参数名是 `formal_protocol`，nullable identity 分支却读取未定义 `protocol`。
failure audit SHA-256 为 `d323c122230795585bbadb16f8650f5e395716b145935a4a41cf5fafe21e2608`。
该 run 永久为 `INVALID_PROTOCOL_OR_IMPLEMENTATION`，不得 resume、retry、finalize、salvage 或复用 checkpoint、
binding、context、ledger。有效 episode、interaction、update、checkpoint、performance 均为0；holdout 未打开。

## 最小修复与测试合同

修复仅将 nullable contract identity 的来源改为已验证的 `formal_protocol`。Nullable Metric Aggregation Contract
仍为 `1.0.0`，semantic SHA-256 仍为
`50a9983d21afdc06ec9df309c29f75f5009e062013990bb491ed58c442470889`；科学配置、算法、reward、action、预算、
split/window、catalog/capacity、selection/statistics 与 holdout seal 均不变。

active resolver 测试必须实际进入 `nullable_metric_contract_required=true`，覆盖成功解析/序列化、nullable hash
缺失与漂移、scientific/binding/context mismatch、非法正式预算覆盖，以及 non-formal/historical fixture 排除。
compile/import 只作补充；静态符号检查必须拒绝同类未定义 `protocol` 引用。

## 150-cell 零 episode 验收

验收从 active Protocol 正式 train command expansion 派生 10 agents × 5 seeds × 3 capacities 共150条命令，
保留正式 profile、256 episodes、update interval 8、batch size 64、max_steps 22、32 updates、checkpoint cadence 4
和 SA-GHMAPPO `auxiliary_coef=0.06`，只追加 `--formal_contract_preflight_only`。每条命令必须真实经过 active
bundle/context/resource identity、training resolver、nullable identity、train-window binding、workflow/catalog
初始化、agent 构造与配置审计，然后在 episode 0 前退出。

验收必须在无本地 `.venv` 的 clean detached commit 上使用共享绝对 Python。记录每个 cell 的 coordinates、
命令、return code、resolved budget/config、identity hashes、stdout/stderr hashes，并通过文件扫描确认 episode、
interaction、update、checkpoint 与 performance 均为0。空 episodes/checkpoints 目录不是 checkpoint。

## Readiness 与结论边界

Readiness v19 同时消费 G14R15 的真实 transaction/downstream rehearsal 与 G14R16 的 commit/bundle/nullable-bound
150-cell entrypoint acceptance。前者证明 publication/recovery/gate 链，后者证明正式训练初始化；二者都不证明
完整正式训练、算法性能、formal result、holdout、TMC-ready 或 paper-ready。缺失、不完整、旧版本、跨
commit/bundle 或未经过 nullable 分支的证据必须 fail-closed。
