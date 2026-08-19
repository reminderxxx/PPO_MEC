# G14A Typed MB Formal Runtime Plumbing Validation Report

- `reviewed_at`: `2026-08-19`
- `literature_cutoff`: `not_applicable_no_literature_review`
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `typed_model_cache_runtime_plumbing_validation_20260819_g14a_v1`
- `policy_version`: `docs/project/top_journal_review_policy.md@2026-08-19`
- execution Git commit: `ba690b3a7b4883d1073901a39464f3e35da386ba`
- evidence level: `contract-validation-only / non-formal rehearsal`
- verdict: `G14A plumbing pass; G14 readiness still blocked`

## Outcome

共享 runtime resolver 已贯通 typed config、fairness、training、checkpoint、evaluation 与 benchmark。320/384 MB resolved runtime hash分别为：

- `e6dfc12dd0c625a385d6eff77e59f55c2179f9faaa38a2c048888d6cba61d48e`
- `d01c2919c24ffaa38e86a31e7e055403d3ce4691d8f8df1e766da2be6c0cf1c8`

两者共享 catalog canonical fingerprint `89c548980b63df733553d748e8db3ca622965b63abcd08ebd4c231790b40a9d6`、initial-state fingerprint `fb0cdbfa761477f4c39bc3416181b475c8884a1c1433edc56d7f2541fc6cac46`、dependency fingerprint `0f8fcd018635426d67eb78af567456d3f7b31a6bac48ac876baee751d09ddcb9` 与 pinned/evictability fingerprint `220f27d6a38d28852e43f1e65e0af8b5aa8399ad6bd785a89246a3de7cd270c7`。

## Fairness 与 checkpoint

320/384 MB typed fairness manifest均经 file/hash/drift/10-pairwise 验证通过；manifest ID分别是 `cbfm-0f6d8ee21dd8ea46` 和 `cbfm-d005fd7b41066ebd`。五个 reactive baseline 共用同一 typed binding，唯一主要差异仍是 eviction policy。

Rehearsal 共创建 8 个明确标注 smoke/non-formal 的 tiny checkpoint：PPO/MAPPO × seed 7/13 × 320/384 MB。8/8 serialization/restore/provenance gate 为 `compatible`；legacy mock checkpoint为 `unavailable_legacy_metadata`。未创建正式 checkpoint。

## End-to-end rehearsal

Run ID：`g14a_rehearsal_20260819_230456_872949`，label：`non_formal_typed_runtime_rehearsal`。使用 repository-controlled catalog、NGSIM + Alibaba 最小骨架、`controlled_non_hidden` G07 smoke window、2 seed、2 capacity、五个 reactive baseline和PPO/MAPPO。两组 typed benchmark共 28 个 one-step episode；不读取 formal/holdout/hidden。

28/28 raw summary满足：request 与 CacheEvent 一一对应、schema 1.3、base+adapter dependency bundle存在、per-object lookup和joint readiness存在、atomic/policy typed字段完整、workflow-state transfer独立、trace context 1.0可用。Metrics 1.1从 raw event独立重算并与 row scalar逐项一致；缺 trace时为 `unavailable`，latency saved保持 `null`。Legacy slot与legacy MB各完成一次旧 benchmark 兼容运行，fairness provenance按合同为 `unavailable`。

Negative cases确认 typed slot、catalog fingerprint mismatch、non-finite MB 和 legacy checkpoint typed impersonation均 fail-fast。专项测试另覆盖 dependency/initial/pinned drift、agent/catalog/capacity/window/SHA mismatch、JSON round-trip和runtime hash resume。

## Artifact

根目录：`artifacts/analysis/typed_model_cache_runtime_plumbing_validation_20260819_g14a_v1/`

顶层包含 producer/consumer matrix、两份 resolved runtime、typed fairness manifest与validation、training/checkpoint/benchmark audit、CacheEvent/metrics reconciliation、legacy compatibility、negative cases、rehearsal manifest、command log和integrity manifest。Integrity manifest记录125个本地 rehearsal 文件；tiny checkpoint只用于本地合同验证，不应提交或复用为正式实验输入。

## Remaining blockers

G14 readiness 仍未通过：G14B 尚未冻结互斥 split/final protocol；没有正式 checkpoint、formal raw trace、holdout/support证据或新 readiness audit。本报告不改变 G14 blocked readiness JSON，不声称 G14 完成或 paper-ready。下一步只能回计划窗口重新执行 readiness gate；不得自动启动 G14B/G15。
