# current_dated_release_verification_proposal

状态：`UNAPPROVED_PROPOSAL`。本文件不是历史 release attestation、不是 continuation approval，也不改变真实授权状态。

## 已能独立验证的历史发布事实

- Git 对象链存在：`834603a266bf06d30a070588a6e1f633eacf70e3`（被验收代码）→ `3a8830328394fe114caadec02cb34696149d9fd8`（验收证据）→ `a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d`（后续文档）。提交时间分别为 2026-09-06 14:53:49、15:12:52、15:15:30 +08:00。
- 原任务「修复checkpoint身份投影验收」的保留记录包含多次 `git push origin main` exit 0，且当时 agent 报告 `HEAD=main=origin/main=a6d1fd8`。这是当时命令/报告的现存记录。
- 原 launch 原件 SHA-256 本轮复算为 `23e4f668...e1bc`；原任务时间为 2026-09-06 15:27:12 +08:00，明确授权 preflight/tests/train/dev_select/checkpoint_freeze，并明确不授权 formal/holdout/G14D/G15。

历史独立、承担 release 责任的 attestation 或可验证签名仍未找到。因此 `origin_release_evidence=unavailable`；当前核验不能倒推 2026-09-06 当时已有新的独立批准、签名或 continuation authority。

## 拟议资格规则

有权 release authority 可选择采用以下规则：允许一份**当前日期的独立 release verification**作为本次 continuation 资格的 release prerequisite，但它必须清楚标注为 2026-09-09 或实际签署日的现时判断，不冒充历史 attestation。

合格记录应同时包含：

1. `historical_event_times`：保留上述原提交/任务/push 记录时间；`verified_at`：记录当前独立核验时间，两者不得混写。
2. 精确对象：`834603a`、`3a88303`、`a6d1fd8`、各 parent/tree、Protocol 2.9/bundle 身份、原 run ID；说明 `a6d1fd8` 是科学执行 checkout 身份，而非声称其时点出现独立批准。
3. 证据：Git 对象、原任务命令记录及其不可变副本/摘要、C 原件提取、当前核验报告；逐项列明哪一事实由哪一证据支持。
4. 责任：独立核验者声明已阅读原件、核实来源/作者/时间/范围与字节；release authority 明确决定这类“现时核验历史事实”是否满足本次资格规则。两项职责可以由有权治理安排兼任，但必须披露，不能由 agent 自行任命。
5. 不可验证项：历史时点是否存在未留存的独立批准、任务用户法定身份、当时远端全局状态、现实组织授权关系，均保留 unavailable/unverified。
6. 若被采用并进入 production trust，认证记录必须由已安装 verifier key 签署并精确覆盖 release role/scope；之后仍需独立 continuation approval。release verification 本身绝不授权执行。

## 当前日期核验的证明边界

它可证明“截至当前，独立核验者根据列明证据判断这些历史 Git/任务/push 事实成立，并由 release authority 现时接受其作为资格前提”。它不能证明“2026-09-06 当时已有独立 release attestation”，不能扩展原 launch 截止 checkpoint_freeze 的范围，也不能证明当前撤销最新、owner quiescent 或 executor 可操作。
