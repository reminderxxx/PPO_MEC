# origin_authorization_evidence_review

本轮仅检索项目内 G14R18/v16/A/B 授权候选、原run顶层、两个直接相关原任务。任务列表仅用于定位；不读取无关任务、账户凭证或私人目录。未查阅dev/formal/holdout性能来选择批准。

## 原始 launch approval：verified（限定历史用户任务证据）

来源：任务「执行 G14C v16-A 冻结实验」，task `01a0759d-3398-7f50-855b-aad162b1c21b`，初始 turn `01a0759d-34f2-7a32-ba20-b743191a2b40`，user item `01a0759d-4368-7e53-9e1e-f543ac03f07e`。工具返回的 user 消息明确指向附件 `/Users/howen/.codex/attachments/1b4faf5e-c59f-4c11-b0ba-f42ea9139616/pasted-text.txt`。原任务创建/初始turn时间分别为2026-09-06 15:27:11/15:27:12 +08:00；这是任务记录时间，不伪称附件有独立签名时间。

授权主体：该原任务的用户（工具显示user角色；未独立认证其法定身份）。原文：

> 本消息授权上述阶段的正式执行；不授权 formal evaluation、holdout、G14D 或 G15。

“上述阶段”明确为 preflight → tests → train → dev_select → checkpoint_freeze；唯一 execution commit为a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d，Protocol2.9.0，active bundle 26e5b4fa8d5d33cec0257fbbcdd15ac8dfbace15fa7c22e18e2ce3c32c93aba8；要求全新durable root，开始前实际remote/main等于该commit。附件末尾要求v16-B另行授权。

原授权发出时run尚未命名，授权的是该父目录下唯一新v16；原初始任务记录描述本次worktree和首次创建，原账本/context绑定本任务指定run。现有15条phase证明终止于freeze，不将后来命名的run倒写进原授权原文。附件当前SHA-256及关键摘录见 `origin_source_identity.json`；该新算hash只用于引用完整性，绝非新批准或原签名。

A/B先前unavailable是在其artifact搜索范围内的结论，不是永久不存在。本轮增加直接原用户任务证据，所以改变launch证据判断；不改A proposal，也不追认新的执行范围。

## 原始 release attestation：unavailable；历史发布事实有证据

直接来源：任务「修复checkpoint身份投影验收」，task `01a07557-a437-7281-a735-891fd8cf9d72`，turn `01a07557-a612-71d3-870c-d5d92834ffb5`。任务时间区间2026-09-06 14:11:13—15:17:19 +08:00；原用户要求独立提交/push并区分验收与记录，禁止启动v16。

原执行agent记录的原文：“后续文档提交 `a6d1fd8` 已推送；它只改文档，没有代码变化”；final明确完整commit及834603a（验收）、3a88303（证据）、a6d1fd8（文档）分工。对应push命令记录exitCode=0。Git对象显示a6d1fd8提交时间15:15:30+08:00，parent=3a8830328394fe114caadec02cb34696149d9fd8；834603a→a6d1fd8差异仅文档/审计材料，没有科学源码变化。原launch任务的启动检查记录还陈述remote/main一致。限定摘录见 `release_task_extract.json`。

这些是原时点的agent发布报告/命令记录，可验证历史发布过程；不是本轮新生成hash或当前ancestry推断。但尚未找到独立于实现agent、承担release原件验证责任的主体证明，也没有可验证的历史签名/外部认证release attestation。故保持资格字段unavailable，而不把“发布事实存在”直接升级为生产审批证据。这是本轮对“独立release attestation”的保守资格判定；合同未指定的责任主体/认证标准不得由C临时发明并追认。

## 新 continuation approval：not_issued

原用户授权截止freeze；G14R18暂缓记录只描述该修复任务；A proposal仍false；B approval_request/unsigned contract和test-only HMAC均不是生产批准。C当前请求显式禁止签发。因此不存在本轮可用的continuation生产批准，且原run并不因此永久无效。

## 一次性最小材料清单

1. 由项目授权主体提供或定位**原时点独立release attestation**：原件/认证来源、原时间、责任主体、精确a6d1fd8与834603a验收关系、Protocol/bundle及发布结论。已有两个原任务无需用户重复提供。若历史独立证明确实不存在，明确unavailable；可以另提新的规则修订/现时证据资格方案，但必须注明新日期，不能伪装历史批准，本轮不执行该方案。
2. 未来信任治理须指定谁独立核验launch/release原件、谁有continuation签发权、如何认证该主体、如何保存可审计原件与撤销状态。这是后续资格材料，不是请求用户现在批准真实execute。
3. 修复/独立验收完成后才能提交新的continuation决策。若涉及恢复锁，还需独立现场owner/quiescence证据及其精确hash，不能仅凭锁过期或PID不存在。
