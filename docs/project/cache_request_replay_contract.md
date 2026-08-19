# Cache Request Replay Contract

版本：`cache_request_replay_version = 1.0.0`（G08，2026-08-19）

## 目的与输入边界

G08 oracle 只消费 policy-neutral request replay，不从任一 baseline 的 hit/miss、reward、victim、admission、service result、policy state、cache contents或 aggregate 反推请求。producer 复用 G07 冻结的 Alibaba workflow builder、NGSIM window loader、`RSUMapper`、catalog size resolver和 evaluation unit；DAG 按静态 execution order 推进，不执行 cache/service policy。因此 service failure 或 cache outcome 不会改变后续 replay。

每个 request 记录 request/evaluation-unit/episode identity、step/time/order、vehicle/workflow/node/base-model/object/adapter、resident size/source、pre-action request RSU、post-mobility current service RSU、previous/actual-next RSU、actual/predicted handoff topology、合法 service/cache target RSU 集合和 DAG provenance。prediction 字段允许为 `null`；G08 v1 不控制 prediction、prefetch 或 migration。

## Canonical fingerprint 与验证

完整 replay（除 fingerprint 与 validation 自身）使用 G07 canonical JSON 规则递归排序、紧凑 UTF-8 serialization、拒绝 NaN/inf，再计算 SHA-256。duplicate request ID、非正 size、乱序、unknown major、source manifest ID/hash/Git mismatch、非 canonical RSU list、endogenous producer或 outcome 字段均 fail-fast。JSON round-trip 必须保持完全相等。

同一 G07 evaluation unit 的五个 baseline 必须引用同一 G08 replay fingerprint。legacy `observed_cache_request_stream_v1` fingerprint 只用于 G07 对称性审计，不等于包含 mobility/DAG/provenance 的 G08 replay fingerprint。raw observed `CacheEvent` 可在 replay 已冻结后逐 request 对齐，生成独立 `observed_cache_baseline_outcome_v1.0.0`；不允许反向生成 replay。旧 artifact 缺 replay 时返回 `unavailable`，不能从 aggregate 猜测。

## Producer

```bash
.venv/bin/python scripts/build_cache_request_replay.py \
  --fairness_manifest_path <g07_manifest.json> \
  --evaluation_unit_id <seed/window/workflow> \
  --output_path <new_request_replay.json>
```

入口默认拒绝覆盖。G07 `1.0.0` 仍可读取；新 builder 在 `cache_contract.oracle_companion_contract` 写入可选 consumer-safe 派生信息，旧 manifest 缺该字段仍有效。
