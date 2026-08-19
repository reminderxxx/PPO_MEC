# Typed Model Cache Contract

版本：`typed_model_cache_contract_version = 1.0.0`（G13，2026-08-19）

CacheEvent：`1.3.0`

type-aware metrics：`1.1.0`

## 范围与 profile

G13 将 adapter-only cache 扩展为显式对象类型、依赖、容量、传输、命中和生命周期合同，但不改变 `semantic_discrete_5` 的 5 个 action ID、reward 或 RL loss。缺少 profile 的旧 catalog/config 固定解析为 `legacy_adapter_only_v1`；只有显式选择 `typed_base_adapter_state_v1` 才进入 typed runtime。typed runtime 首版只允许 `enabled=true, unit=mb`。

`kv_prefix` 仅保留 enum/interface，G13 强制 disabled，不进入 capacity、CacheEvent request denominator 或正式指标。Qwen-Bailian/Mooncake 仍是 KV-only 候选，不能映射为 adapter。

## 修改前语义审计

- `VehicleBaseModelProfile` 原本是 vehicle capability catalog；它不是 RSU resident cache，也没有 eviction/admission 生命周期。
- 原 RSU 执行先检查 primary vehicle 的 `base_model_id == required_base_model`，原因是环境把 vehicle capability 当作整个服务链的兼容 gate；这不构成 RSU base resident evidence。
- 原 RSU `RSUState.cached_adapter_ids` 只缓存 adapter；RSU 不真正缓存 base model。
- workflow node 同时有 `required_base_model` 和 `required_adapter`，但原 catalog 没有唯一 adapter→base dependency/compatibility map。
- 原 `CacheObject` 只有 `adapter_id`，所以 object identity、size resolver、capacity、victim 和 replay/oracle 都只能表达 adapter。
- `AdapterStateBundle` 是按 adapter 查找的 handoff migration payload/logical continuity token，不是长期共享 resident model object；原实现用固定 32/16 MB估计迁移量。
- adapter admission transfer 与 state migration transfer是两个不同 payload；同 step 同时发生时总 backhaul 相加，不是同一 payload 重复记账。G13 进一步用 per-type transfer 明确分离。
- `vehicle_local` 原事件是 execution-source compatibility 语义，不是 vehicle adapter resident evidence；`current_rsu/target_rsu/neighbor_rsu` 是所选 RSU adapter resident hit；`cloud/unserved` 是 miss。
- 原 `cache_hit` 的 RSU evidence来自 adapter ID resident；vehicle base equality只是 compatibility。原 capacity/occupancy/pollution只覆盖 adapter。
- eviction policy 的形参虽然已名为 `object_id`，环境实际传入 adapter ID，因此 legacy tie-break/victim parity按 adapter ID冻结。
- CacheEvent `object_type=adapter`、singular admitted/evicted adapter字段、G06 pollution、G07 resident-size/initial-cache、G08 replay/oracle、G09 transfer分析均假定 adapter-only。
- typed objects影响 catalog、env readiness/transaction/capacity、eviction key、CacheEvent、recorder/context、G06 reducer、benchmark nullable row、G07 binding、G08 replay/oracle、G09 analyzer。
- 旧 checkpoint 的 observation/action形状和旧 config profile缺省语义必须保持，否则历史 checkpoint、baseline公平性和 artifact可复现性会被破坏；因此 typed字段只作显式 profile 与 1.x optional extension。

## Object schema 与规则

`TypedCacheObject` 至少保存 `object_id/object_type/version/resident_size_mb/transfer_size_mb/source/provenance/base_model_family/base_model_id/required_base_model_id/adapter_id/workflow_identity/shareability_scope/mutability/persistence/evictability/migration_semantics/dependency_ids/dataset_profile_source/license_status/formal_use_status/stable_fingerprint/availability/counts_toward_capacity`。

- `base_model`：具有唯一 `base_model_id/family/version`；多个 compatible adapter 可共享；resident/transfer size独立；禁止 64 MB adapter fallback；evictability显式。
- `adapter`：具有唯一 `adapter_id`，恰好引用一个 required base object；adapter resident不推出base resident；family/version不匹配时不能服务。
- `workflow_state`：绑定 `vehicle/workflow/node/continuity` 逻辑身份；G13 固定为 mutable handoff payload、`counts_toward_capacity=false`，不进入通用 LRU/LFU共享 resident cache。
- `kv_prefix`：reserved/disabled。启用需要未来独立 Goal、contract、denominator 和 capacity语义。

全局 object ID、`(type,version,identity)` 与 adapter映射必须唯一；dependency必须存在且无环；size/transfer必须有限正数；provenance必须 JSON-safe；canonical fingerprint使用 sorted compact JSON SHA-256。未知 license HF metadata不得 formal-ready；BERT size anomaly必须 `blocked_provenance_anomaly`。

## Serving readiness

每个请求输出 `base_ready/adapter_ready/joint_base_adapter_hit/state_required/state_ready/full_service_ready/missing_object_types/incompatibility_reason/compatibility_result/per_object_lookup_results`。

- Vehicle-local：vehicle base equality只是 capability evidence；G13 controlled profile固定 `vehicle_adapter_residency_enabled=false`，因此没有 adapter resident evidence时不能形成 full service hit。
- RSU：service target必须合法，required base与adapter必须同时 resident且 family/version compatible；连续 handoff需要 state ready或同 step合法恢复。
- base hit但adapter miss、adapter present但base missing都不是 full model-service hit；legacy `cache_hit` 在 typed event中对应 full service readiness，分层字段保留部分命中。

## 动作与原子 transaction

每 step最多一个逻辑 cache action。旧“admit required adapter”在 typed profile中解析为最多两个对象的 `[base_model, adapter]` dependency bundle：base已resident时只admit adapter；二者已resident时touch/noop。workflow state仍只由 migration action处理。

transaction顺序冻结为 resolve→validate→single victim plan→commit evictions→admit base→admit adapter。partial admission禁止；任一对象/整个bundle oversized、缺target或没有dependency-safe victims时整体 `rolled_back_no_mutation`。rollback不改变 resident或policy state，不留下orphan。legacy singular admission字段只表示真实 adapter admission；base-only admission不伪造 adapter admission，typed lists保存完整事务。

## Capacity 与 eviction

typed profile中 base+adapter共享每 RSU同一 MB capacity；workflow state与KV不计入。initial typed residents使用同一 catalog validator且必须无需policy-specific trim即可装入。commit后 `used<=capacity` 且 `orphan_count=0`。

LRU/FIFO/LFU/Aging-LFU/Random的 stable key泛化为 typed `object_id`，legacy仍传 adapter ID，因此旧 victim顺序不变。G13选择“resident adapter仍依赖base时禁止evict该base”；pinned/non-evictable同样排除。一次 policy plan可返回 heterogeneous multi-victim；没有可行victim则原子拒绝。Random继续使用私有 seeded RNG。

## CacheEvent 1.3 与 metrics 1.1

1.3只增加 optional字段：contract/profile、typed request/dependency、per-object lookup、base/adapter/joint/state/full readiness、typed admission/eviction、per-type MB、compatibility、typed capacity snapshot、atomic status与orphan count。1.0/1.1/1.2继续读取；旧trace的typed metrics是 `unavailable`，不是零。每 workflow-node request仍只有一个 event；dependency对象不是子请求，不扩大 denominator。

G06 1.1新增：分层 hit/readiness、missing type、compatibility failure、requested/hit/resident/admitted/evicted/transfer MB by type、occupancy share、pinned MB、bundle rejection、adapters per base、base reuse/sharing、严格可重算的 avoided base transfer、orphan、bundle churn与transfer amplification。pollution/context重建支持multi-object typed admission；right-censored仍独立。latency saved保持 unavailable。

## G07/G08/G09

G07 `cache_contract.typed_model_cache`绑定 profile/contract、catalog fingerprint、compatibility map、initial typed state fingerprint、MB capacity、transaction、metric version与oracle compatibility；五baseline仍只允许 eviction policy不同。

G08 typed replay保存完整 `[base,adapter]` dependency bundle、catalog fingerprint和atomic语义。tiny exact oracle在同一MB capacity、initial state、per-type transfer和dependency-safe eviction下穷举；state limit返回 `unknown_state_limit`，不退化为greedy。legacy replay/oracle不变。G09读取total typed transfer、full-service baseline hit和typed initial objects，不把base transfer压成adapter transfer。

## Profile 与数据边界

- `typed_model_cache_controlled.json`：repository-native deterministic controlled profile；2 base、5 compatible perception-family adapters、2 reasoning-family adapters、1 migration-only workflow-state，heterogeneous size、pinned base、3 RSU；每个 base 都可被多个 adapter 共享。它是“controlled synthetic catalog over NGSIM+Alibaba skeleton”，不是真实联合trace。
- `hf_metadata_diagnostic_model_profile.json`：qwen/cbow/bert metadata-only、cross-source、non-formal、no payload downloaded；未知license，BERT anomaly blocked；不进入默认profile或G14正式主表。
- BurstGPT不下载/不进入G13 runtime request stream；仅保留future replay mapping。

## Claim boundary

G13 artifact只证明合同、事务、兼容和最小真实链路可运行，不证明算法优势、latency saving、paper readiness或真实 joint VEC model-cache trace。本轮未执行G14、训练、调参、formal、holdout或hidden benchmark。
