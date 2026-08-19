# G13 Typed Model Cache Validation Report

## Outcome

G13 已冻结：`typed_model_cache_contract_version=1.0.0`、`CacheEvent=1.3.0`、`cache_efficiency_metrics=1.1.0`。legacy adapter-only仍为缺省 profile；typed profile使用MB capacity、原子base+adapter dependency transaction、dependency-safe eviction与分层service readiness。

审查/验证元数据：

- `reviewed_at`: `2026-08-19`
- `literature_cutoff`: `2026-08-19`
- `target_venue`: IEEE TMC mechanism-support；不是 paper-ready verdict
- `artifact_run_id`: `typed_model_cache_validation_20260819_g13_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- baseline Git commit: `695f54fb5324c3ef7e1a6b9b28b660f4a5952bbf`
- evidence level: `E2_ARTIFACT_AUDITED_CONTROLLED_MECHANISM_ONLY`

## 结果

- catalog：9 typed objects，canonical fingerprint `1c0b3d55e957619af8bd1d15cbf1ba61b756b540c2c67ab8e66b507734092d7e`；duplicate/missing/cycle/family/size/license/provenance gates已覆盖。
- readiness：full hit、base-only、adapter-only、state missing、vehicle-local边界均输出分层证据；partial hit不计full service。
- transaction：base+adapter atomic admit、base present只admit adapter、rollback无mutation、heterogeneous two-victim、pinned/dependency-safe无可行victim、oversized bundle均通过；orphan始终0。
- transfer：base、adapter与workflow state分别计费；workflow state不占长期model-cache capacity；CacheEvent仍是一请求一个denominator。
- policies/fairness：LRU/FIFO/LFU/Aging-LFU/Random在相同typed transaction/capacity/catalog下通过；唯一主要变量仍是eviction policy。
- oracle：2-request tiny typed exact oracle在H=1下optimal，首请求原子admit base+adapter，第二请求复用；per-type transfer可重算，capacity/orphan invariant通过。
- legacy：缺profile仍选`legacy_adapter_only_v1`；round-trip环境的reward与CacheEvent完全相同；typed指标为unavailable。
- real minimal：真实NGSIM mobility + Alibaba DAG、3 steps/3 requests、3 typed events、CacheEvent 1.3；1次full service ready。该run是non-formal链路验证，不是算法证据。

Artifact：`artifacts/analysis/typed_model_cache_validation_20260819_g13_v1/`。integrity manifest覆盖其余13个JSON文件，状态pass。入口：

```bash
.venv/bin/python scripts/validate_typed_model_cache.py
```

## 未覆盖风险

- controlled size/compatibility不是production model/adapter trace；HF diagnostic仍受license/provenance blocker。
- workflow state首版是migration-only payload，未建模长期state cache或KV prefix。
- exact typed oracle只保证小状态可运行；大状态达到limit返回unknown。
- G14正式多seed实验、训练、调参、formal/holdout/hidden、真实load/inference latency与latency counterfactual均未执行/未提供。
