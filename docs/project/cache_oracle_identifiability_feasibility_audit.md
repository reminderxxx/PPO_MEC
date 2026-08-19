# G08 Oracle 可识别性与可行域审计

- `audited_at`: `2026-08-19`
- `baseline_git_commit`: `5d76596aa08c6589e4cc028d78b3383d48b830e6`
- `scope`: G08 Future-Horizon Cache Oracle Contract；不含 G09、formal、holdout、hidden benchmark
- `evidence_level`: source-contract audit（不是性能证据）

## 结论

当前 observed `CacheEvent` request stream 可能受 cache outcome、service failure 和 workflow 推进影响，不能作为 oracle 输入。`VecWorkflowCoreEnv.step()` 仅在 `primary_vehicle + base_model_ok + cache_hit + target_rsu` 全部成立时推进当前 workflow node；miss/stall 会让同一 node 在后续 step 重试。因此 G08 必须从冻结的 DAG execution order 与 mobility frames 独立生成 policy-neutral request replay，并把 observed outcome 放到独立 comparison 输入。

现有环境的真实时序不是“request lookup 后 admission”：step 先在目标 RSU 读取 `pre_execution_cache_hit`，随后执行 `_apply_cache_action()`，最后重新读取 cache 得到用于服务和 CacheEvent 的 `cache_hit`。所以合法的同 step current-RSU admission 可以让当前 request 成为 hit。G08 冻结该代码证据，不采用更有利或更不利的替代时序；同时在 solver trace 中区分 `pre_action_hit` 与 `post_action_hit`。

## 可外生冻结字段

可在 cache/service policy 前冻结：evaluation unit、episode、step/time、request order、vehicle/workflow/node、base model、adapter/object、catalog resident size及来源、raw mobility association、current/next RSU、actual handoff topology、DAG nodes/edges/execution-order provenance、G07 manifest ID/hash、dataset/window/workflow/catalog/seed provenance。确定性 predictor 输出可作为带 producer/as-of 的外生拓扑字段，但 G08 v1 不把 prediction、prefetch 或 migration作为控制变量。

禁止作为 replay 输入：actual hit/miss、victim、admission、policy state、reward、service result、stall、baseline cache contents、execution-dependent node retry。它们只能进入独立 observed outcome。

## 动作与服务可行域

- `semantic_discrete_5` 每个 step 只有一个 cache action；current fill（action 0）或 predicted-next prefetch（action 1）不能同时发生。
- 五个 G07 classical baseline 共享 `reactive_current_rsu_admission_v1`，只在当前关联 RSU 对当前 required adapter 执行至多一次 admission；因此 G08 v1 的 matched oracle 只控制 current-RSU placement/admission、victim 与 noop。
- admission 在当前 step 的服务 lookup 前生效；同 step admission 可命中当前 request。已驻留 admission 是 no-op/touch，不产生 transfer。
- vehicle-local、cloud、migration、handoff state transfer、predicted-target prefetch 和任意 neighbor placement不属于 G08 v1 控制变量。服务 cache lookup 范围由 replay 的 `eligible_service_rsu_ids` 冻结；matched classical stratum 当前只包含 current RSU。
- admission 产生 catalog resident size 等量的 adapter transfer MB。oracle 不允许免费瞬移；state bundle migration 不由 oracle 控制且固定为 0。
- 每个 RSU 独立容量。`adapter_slots` 中每个 resident 占 1 slot；`mb` 使用 `AdapterCatalog.resolve_adapter_resident_size_mb()`。oversized object 在 victim planning 前拒绝，且不得产生非法/部分 eviction。
- oracle 可从该 RSU 当前 residents 选择零个或多个 victim；必须一次性释放足够容量并原子提交。词典序 churn 目标排除多余 victim。
- capacity disabled 是历史无界语义，G08 matched capacity oracle 定义为 `not_applicable` 并 fail-fast，不把它解释为 0 容量或无限上界。

## 可严格优化目标与不可识别项

有限 replay、小状态空间和上述动作合同下，可以 exact 优化词典序目标：future hit MB、future hit count、transfer MB、evicted/churn MB、canonical action tie-break。slot 与 MB 分层求解，不能混合比较。rolling horizon 只能读取从当前 request 起最多 H 个 request；episode 尾部截断。

request-level latency saved 仍不可识别：当前没有同一 request 对齐、单位明确的 observed service latency、cold/cloud counterfactual latency、adapter transfer latency、stall/restart latency。reward、workflow span、service delay aggregate 均不能替代。G08 gap 只能称为同一外生 request replay 与可行域下的 placement opportunity gap，不能称为端到端 latency gain或 causal regret。

## 新增版本化证据

G08 新增 `cache_request_replay_version=1.0.0`、canonical SHA-256 request fingerprint、policy-neutral producer identity、source G07 manifest ID/full hash/semantic hash、initial-state fingerprint、capacity/action-budget contract、per-request topology/DAG provenance，以及独立 observed outcome comparison。G07 `1.0.0` 保持原样读取；G08 使用 companion contract 派生 oracle 字段，不修改旧 manifest required schema。
