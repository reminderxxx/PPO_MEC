# Active Formal Bundle Contract

- contract version：`1.0.0`
- active Protocol：`1.8.0`
- active index：`configs/experiment/typed_model_cache_formal_protocol_v1_8_20260827/protocol_index.json`
- Protocol semantic SHA-256：`9799bf2c2f4b4665b8390c6fc5d5aa235faf11d6525e043eac289c061633b3de`
- active bundle core SHA-256：`96627ac414cb5dc80785c907ded2c9588dcdcf69469a5821b75fc07dc25e5b65`
- active formal bundle SHA-256：`793f5106b83f9687044aeeac122179a8c5805688d4a041c0418292345f9138bd`
- Readiness：v10 `READY_FOR_G14C_V8_CLEAN_TRAIN_AND_FORMAL`
- evidence boundary：pre-execution contract validation；无正式性能结论

## 根因与失效状态

G14R7的v1.7生成器从v1.6 `protocol_index.json`深拷贝，在写入v1.7 Protocol和
`execution_environment_manifest.json`后没有更新index中的environment路径，并固定写入
`PENDING_G14R7_VALIDATION`。v9 finalizer只在analysis artifact中生成ready companion，没有审计、回写或
冻结active index。于是v1.7 index仍指向v1.6 environment fingerprint
`a8126811569850dc77f4d9586df4c20166fe78a16ac3386fc59e6f973e5d7257`，而v1.7实际environment fingerprint为
`55c84dfd45ba8177548adeb1c46ddf45d28a9d1decefcb527b2f68e3c7d25a98`；同时Readiness v9声称ready。

旧outer runner只验证调用者显式传入的Protocol和environment，未读取active index，因此人工传入“正确”的
v1.7 environment可以绕开错误index。v1.7及v1.0–v1.6现全部为audit-only；其状态按
`PRE_EXECUTION_BLOCKED_ACTIVE_BUNDLE_INCONSISTENT`保留，不得执行。

## 单一身份与资源规则

v1.8 index是唯一active authority。它逐项绑定：

- Protocol ID/version/semantic/full SHA与文件content SHA；
- exact execution commit规则：执行时观察clean 40-hex `HEAD`，且必须等于`origin/main`；exact SHA写入
  execution binding和resolved context。index不嵌入包含自身的commit SHA，避免Git commit自引用；
- execution environment logical path/content SHA、environment/dependency fingerprint；
- Scientific Config 2.0.0、Formal Agent Order Contract 1.0.0；
- execution binding schema 1.0.0、resolved context schema 2.0.0；
- portable registry、split/window/catalog/runtime identities；
- formal/dev/support fairness manifests与command-template matrix identity；
- Readiness v10 companion、acceptance evidence hash与holdout seal。

Protocol/environment/scientific/order/binding/context/readiness是v1.8专属资源，必须位于v1.8目录。portable
registry、split/window、runtime和fairness是内容寻址的历史稳定共享资源；每项都在index记录role、content
SHA、size和允许共享理由。未声明跨版本引用、symlink、repository escape、cwd guessing、同名异hash、缺失文件
或content drift全部fail-fast。

## 无自引用哈希图

哈希按单向图生成：

1. current/shared资源身份、Protocol、environment、command matrix、commit binding rule和holdout seal形成
   `active_bundle_core_sha256`；该投影排除Readiness。
2. Readiness v10绑定core与clean acceptance evidence manifest的content SHA。
3. ready index加入Readiness content/evidence SHA，再计算`active_formal_bundle_sha256`；只排除该字段本身。

Readiness不反向嵌入最终index hash，因此不存在迭代碰撞或hash cycle。修改status但不重新生成合法Readiness与
final index hash会被拒绝。

## Finalization与不可降级

普通generator只能生成`PENDING_G14R7A_CLEAN_ACCEPTANCE`。finalizer要求Git-clean、无本地`.venv`的candidate
证据，包括186-command展开、150 training identities、24 dev nested identities、15-agent probe、真实
11,850,526-row/73,871-frame/60-window preflight、tests、smoke、compile/import、diff-check、七个保护文件
hash一致、formal/checkpoint/performance计数全为0和holdout sealed/unopened。

finalizer create-only写Readiness/evidence，以同目录temporary + atomic replace把pending index冻结为ready。
ready后普通generator拒绝降级；finalizer重复执行只在全部身份不变时返回幂等结果，任何证据或资源漂移都拒绝。

## Outer runner门禁与provenance

`scripts/run_typed_model_cache_formal_protocol.py`默认从唯一active index自动解析Protocol和environment；调用者
可以显式传入二者作一致性断言，但不能覆盖index。门禁发生在run root、binding、context、ledger或任何phase
输出创建之前，并在dry-run和真实运行使用同一validator。pending/blocked/missing、旧Protocol、路径漂移、
environment fingerprint漂移、Scientific/Order漂移、Readiness/evidence缺失、dirty HEAD或
`HEAD != origin/main`全部拒绝。

`active_formal_bundle_sha256`直接写入execution binding、resolved context、phase/cell input identity、training
summary/checkpoint metadata、dev candidate/freeze provenance；artifact integrity以binding/context和bundle hash
共同追溯。G14C v1–v7 invalid roots/checkpoints继续永久拒绝。

## 验收与边界

ready candidate使用与真实执行相同的outer gate完成dry-run和`preflight → tests`：11,850,526 rows、73,871
provider frames、60/60 windows、1093 tests均通过；主工作区专项656项、全仓1097项通过。16类负例覆盖index/
Readiness不一致、旧environment、CLI bypass、resource/commit/shared/evidence/symlink/cwd/dry-run/invalid-root和
holdout边界。

Readiness v10只授权未来独立任务新建G14C v8 clean run。G14R7A没有启动G14C v8、正式training、formal、
holdout、G14D或G15；正式training/checkpoint/performance仍为0，holdout保持sealed/unopened，不是G14完成、
算法优势或paper-ready证据。
