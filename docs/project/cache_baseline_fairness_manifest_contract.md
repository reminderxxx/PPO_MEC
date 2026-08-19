# Cache Baseline Fairness Manifest Contract

版本：`cache_baseline_fairness_manifest_version = 1.0.0`（G07，2026-08-18）

2026-08-19 G08 以 companion contract 集成：新 builder 可在 `cache_contract.oracle_companion_contract` 写入 replay/oracle version、H、objective、initial-state fingerprint source、action budget和 expected replay fingerprint 派生状态。该字段是 optional consumer-safe 1.x extension；既有 G07 `1.0.0` manifest 不含它仍可校验和消费。实际 replay fingerprint 只能由 policy-neutral producer 生成，不能填入或复用 legacy observed fingerprint。

## 目的与边界

本合同是五个 classical reactive cache baseline 以及后续 G08 oracle 的唯一比较协议来源。对同一 evaluation unit，唯一允许变化的主要实验因素是 eviction policy identity。manifest 校验通过不表示实验已运行；controlled/smoke 运行不构成 formal、holdout、hidden 或算法优劣证据。G07 不实现 G08 oracle、causal eviction regret 或 request-level latency saved。

## Schema

顶层分组固定为 `identity`、`dataset_provenance`、`window_workload_plan`、`seed_plan`、`cache_contract`、`baseline_matrix`、`metrics_aggregation`、`artifact_plan`、`claim_boundary`、`hashes`；未知 critical 顶层字段、缺失字段和不支持的 major version fail-fast。

- Identity：manifest/version/producer/Git commit/dirty audit/protocol status/claim boundary。dirty audit 只记录，不自动提交既有工作区改动。
- Dataset：NGSIM、Alibaba、catalog、window plan 的 logical ID、repo-relative path、文件大小、SHA-256、provider/parser identity；禁止自动下载或覆盖。
- Evaluation unit：`seed + window_id + workflow_id`，必须同时记录 raw frame 和 raw time interval、vehicle/workflow selection、DAG hash、RSU mapper、max steps、termination 与 expected workload fingerprint。
- Seed：environment/workload/policy seed 均为 benchmark run seed；Random 私有 RNG 为 `random.Random(run_seed)`，禁止全局 RNG或隐式默认。
- Cache：capacity enabled/unit/value、initial cache、resident size resolver、fallback、oversized/multi-victim、transfer/migration source，以及 G01/G03/G06 contract versions。
- Metrics：G06 metric set、nullable aggregation、grouping/strata、byte coverage、pollution context、reuse horizons `[1,3,6,12]`；latency saved 固定 unavailable。
- Artifact：预定义 summary/row/aggregate/audit/command/resolved manifest/integrity 路径，路径不改变实验语义。

## Dataset、window 与 request identity

绝对本地路径只用于当前主机文件检查，跨机器 identity 使用 logical dataset ID、repo/data-root-relative path、size 和 content SHA-256。Validator 将 evaluation unit 的 frame/time interval 与 source window plan 逐项对账，不能以 `window_rank_offset`、window 名或不同 rank 代替原始区间证据。

`static_dag_request_plan_v1` 对执行顺序中的 `workflow_id/node_id/base model/adapter/input/output` 计算 pre-run expected workload fingerprint；runtime 对实际构造的 workflow 重算并 fail-fast。CacheEvent 再按 request 顺序生成 `observed_cache_request_stream_v1`，同一 unit 的五个 baseline 必须完全相同。后者是运行后对称性检查，不把 cache outcome、hit、victim 或 reward 混入请求 identity。

## Capacity、catalog 与 initial cache

一个 manifest 只能属于一个 `adapter_slots` 或 `mb` comparison stratum。G07 builder 要求声明的 initial cache 无需 policy-specific trimming 即可装入容量；否则拒绝构建。Catalog 的 explicit `CacheObject.size_mb`、64 MB fallback、adapter transfer size和 state bundle migration size均冻结。Initial per-RSU content和resident-size表都有稳定 identity；oversized object 在 planning 前拒绝，multi-victim 是最小充分 ordered prefix并原子提交。

## Baseline matrix 与差分 allowlist

必须且只能包含：`reactive_lru`、`reactive_fifo`、`reactive_lfu`、`reactive_aging_lfu`、`reactive_random`。每项绑定 agent/config SHA-256、policy name/version/determinism/seed rule、统一 `reactive_current_rsu_admission_v1` 和 action semantics。

10 组 pairwise diff 只允许：agent name/class、eviction policy name/version/determinism/state、Random seed metadata、Aging-LFU policy config，以及由 policy identity 自然产生的 config path/hash/identity字段。Capacity、catalog、workload、reward、admission/control、action semantics 或 agent-specific override 一律失败。完整 config hash 不用于替代字段级 diff。

## Canonical serialization 与 hash

Canonical JSON 使用 UTF-8、递归 key sort、紧凑 separators，并拒绝 NaN/inf。输出同时包含 full manifest SHA-256、semantic protocol SHA-256、dataset/config hashes和 per-unit fingerprint。

Semantic hash 排除 `identity.manifest_id`、`identity.created_at`、`artifact_plan`、本机 absolute path、`hashes` 与 validation report；因此改时间戳或输出目录不改变实验身份。Git commit、content hashes、window/workload/seed/cache/baseline/metrics/claim semantics 均参与；任一关键字段变化必须改变 semantic hash。Full hash包含时间戳、manifest ID与 artifact plan，但排除自身 hash/report，适合文件级 provenance。

## Runtime enforcement

`scripts/build_cache_baseline_fairness_manifest.py` 从显式输入构建 resolved JSON，默认拒绝覆盖并在写出前自动验证。`scripts/validate_cache_baseline_fairness_manifest.py` 输出结构化 report 与 pairwise diff。

`scripts/benchmark_main_results.py --cache_baseline_fairness_manifest_path ...` 在加载数据前重验 manifest，拒绝 CLI 覆盖 agents、seeds、workflow selection、window plan、capacity、max steps、vehicle selection 或 reward offset。当前正式消费者先冻结 `adapter_slots`；MB schema已验证，但在该 runner 增加显式 MB runtime参数前 fail-fast。未传 manifest 的旧 benchmark 保持兼容，并在 row/aggregate/run manifest 中标记 `fairness_manifest_status=unavailable`。

运行后，summary、raw row、aggregate、run manifest、runtime audit、resolved manifest与integrity manifest均记录 manifest ID/full hash/semantic hash及真实 agent/policy identity；aggregate另按 manifest grouping keys生成 fairness stratum。

## G08 reuse

G08 必须直接消费同一 validated manifest与evaluation units，在相同 request plan、capacity/catalog/initial cache、seed和cost model上增加 oracle identity；不得重选窗口、重建请求流或另设容量。G08 输出的 oracle gap/causal regret属于新合同，不能反向写入或修改G07历史artifact。
