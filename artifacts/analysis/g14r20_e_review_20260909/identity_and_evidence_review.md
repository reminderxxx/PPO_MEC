# identity_and_evidence_review

## 审查元数据

- `reviewed_at`: `2026-09-09T16:53:18+08:00`
- `literature_cutoff`: `2026-09-06`（继承元数据；本轮不评价文献或 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC); administrative release/trust review only`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `artifact_run_id`: `typed_model_cache_formal_20260906_152847_g14c_v16`
- `scientific_commit`: `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`
- `implementation_commit`: `876c369a1fa8cc33b19d782ab1ba230f021965a7`
- `implementation_tree`: `00f4c3e65811aad32b2fb0c1bdd16820262cc1b7`
- `evidence_commit`: `35bec6320c211b4762ec7bee85e83d780424d3eb`
- `evidence_tree`: `2b14c4d6e3d09a792ad36489b99dfb32bf7cec54`
- `evidence_level`: identity/protection 为 `E3_REPRODUCED`；D 全仓/完整合成/150-checkpoint 结果为 `E2_ARTIFACT_AUDITED`；本轮边界测试为 `E3_REPRODUCED`；科学性能为 `N/S`。

## 独立结论

代码验收对象严格固定于 implementation commit，而非后续 evidence commit。独立 checkout 的 HEAD/tree 与 implementation commit 一致且开始审查时 clean；evidence commit 的唯一 parent 正是 implementation commit。22-file executor identity 逐文件复算无差异，canonical identity SHA-256 为 `8d64e52982e58a56f27ec0348fd7c456acc25fc196c00374290e2f872679e2dd`。D 交付包 manifest 的 61 个条目以及 C 包 manifest 的 25 个条目均逐字节复算无差异。

D 相对其实现基线 `e6a73357d08220d34ddc2d6537d140131c60520f` 的完整范围为 17 个文件（902 insertions、65 deletions）：文档/requirements、`authorization.py`、新增 `production_trust.py` 和 test-only fixture、synthetic/native driver 及测试；未改科学 Protocol、科学配置、原科学 worktree、真实 run、账本、checkpoint 或 registry。implementation commit 自身相对 parent `e21302a970cc93c90d9c61a1b06e4d6f2a894222` 只改 4 个文件（12 insertions、2 deletions），收紧非空 `revocation_id`。D 的 `scope_and_diff_check.json` 是已有证据审计，不作为上述独立 Git 复算的替代。

## 科学身份、命令计划和原 run

- 原科学 worktree 当前仍是 detached `a6d1fd8`、tree `eb8e83c6e4532bc45c22fb7a816388e576753e30`、clean。
- D 的 `frozen_command_plans.json` 本轮独立 canonical 复算为 `e68340af18c5195a1c07aa5ef30532ff796ed13f4f4f7e774f2a172960dc9bad`，与未签发 contract 一致。计划由原科学代码生成、150 checkpoint 语义加载和 44 active/6 generated resource 复核仅审计 D 已有只读证据，本轮没有重跑该 1110 秒链路。
- 原 run 两份账本本轮独立读取：phase ledger 15 条（5 running、5 completion_candidate、5 completed），最后为 `checkpoint_freeze/completed`；cell ledger 348 条（174 running、174 committed），没有 formal/holdout 记录被本轮创建。
- 107,320 项保护清单逐文件/链接独立复算：0 mismatch、0 addition。proposal SHA-256 `65f6fbe...13cc8`，两账本 SHA-256 `aa90b53b...a86cc9` / `509fcb07...19e4e`，七个用户文件均与 D 起止证据一致。

## D 已有验收证据的审计边界

D JUnit 原件可解析为 1520 tests、0 failure、0 error、0 skipped；D 完整合成报告记录 8 阶段完成、synthetic child dispatch 25，监控计数为 scientific rollout 0、real v16 dispatch 0、real v16 write 0。命令记录前后均声明 exact implementation commit/tree 且 clean。以上属于“仅审计已有验收证据”，并未在 E 重跑全仓 1520 项或完整 2719 秒合成链；报告内 `pass` 字段没有直接提升为 E 结论。

## 保护状态

本轮没有修改源码、测试、Protocol、A proposal、原科学 worktree/run、账本、checkpoint、registry 或七个用户文件；没有执行真实 train/dev/freeze/formal/holdout，没有创建第二个正式 run。审查探针只使用 `/private/tmp/g14r20_e_startup_probe_*` 下的 synthetic fixture。
