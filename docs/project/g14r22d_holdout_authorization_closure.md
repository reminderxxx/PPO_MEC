# G14R22-D 正式 holdout 后台宿主与授权有效期闭环

- `reviewed_at`: `2026-09-26`
- `literature_cutoff`: `2026-09-21`（沿用 G14A01；本轮不评价 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_g14r22_holdout_execution_contract_20260926_v5`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `executor_commit`: `77935c5fa7fa66f9a86afc940b201bf418ce4c11`
- `executor_git_tree`: `13cc5dd2253a8a5523306f07abd8387d5adb8eeb`
- `evidence_level`: `E2_ARTIFACT_AUDITED_AUTHORIZATION_AND_BACKGROUND_CONTRACT_FIXTURE_ONLY`
- `verdict`: `READY_TO_REQUEST_NEW_EXACT_V5_AUTHORIZATION; NOT AUTHORIZED; HOLDOUT UNOPENED`

本轮只修复 G14H02 原子开启前停止证据中的两个工程缺口：grant 有效期未冻结，以及公共后台宿主只接受
`acceptance_non_holdout` 终态。没有签发正式 grant/token，没有创建正式 output root，没有调用真实 holdout policy，
也没有读取 holdout performance。G14R22 v4 原件保持 audit-only；任何既有 v4 批准均不得迁移到 v5。

## 1. Grant 有效期合同

新版 request/grant 使用 `g14r22_grant_validity_v1`：

- `issued_at` 与 `expires_at` 必须是带显式时区的 ISO-8601 实际时间；naive datetime 拒绝。
- `expires_at` 必须晚于 `issued_at`，且差值最多 `259200` 秒（72 小时）。恰好 72 小时允许；在
  `expires_at` 时刻即已过期。
- `issued_at` 不得早于 exact unsigned request 的 `created_at`。负责人必须填写真实签发时间；合同明确
  `backdating_allowed=false`、`silent_extension_allowed=false`、`automatic_resign_allowed=false`。
- 有效性规则统一为 `issued_at <= checked_at < expires_at`，并在授权检查、授权后派生、后台 launch、
  supervisor 启动 child 前及 atomic rename 前分别 fail-closed。授权检查通过但 atomic open 前到期时，临时 opening
  目录删除，正式 root 不创建，holdout 不 consumed，科学 child 不启动。
- atomic rename 成功后，grant TTL 不再是科学进程 lifetime。即使长 child 运行期间到期，也不得自动 kill、缩减预算、
  retry、resume、reopen、清锁或改变 terminal 判定；已经打开的 execution 永久 consumed，继续按冻结科学预算运行到
  正常/失败/外部中断终态。72 小时只约束“何时允许开始一次性开启”，不是科学 wall-clock 保证。
- 任何 grant 文件变化都会破坏派生 job 中的 file/canonical hash；没有延期或重签代码路径。若到期且尚未打开，只能由
  负责人对新的明确申请另行签发，新签发不能由 executor 自动产生。

## 2. 双身份后台宿主

公共宿主 `g14r22d_fixed_background_host_v3` 只接受三种显式身份：

1. `acceptance_non_holdout`：科学包必须声明 `acceptance_non_holdout=true`，不得携带 formal grant/token，terminal 必须
   `holdout_opened=false`，并继续要求 `passed=true`、outer rc=0、capacity identity 与完整 published inventory。
2. `formal_holdout`：科学包不得带 non-holdout 身份；job 必须由 exact request/package/grant/token references 派生，命令
   必须逐 argv 等于固定 `run_dedicated_public_holdout.py --check execute`。
3. `formal_holdout_fixture`：仅用于隔离状态机验收；命令中禁止 `sealed_holdout`/holdout policy，固定
   `fixture_holdout_policy_runs=0`，不得使用正式 output root。

不能把一个身份改成另一个布尔值绕过：formal→non-holdout 与 non-holdout→formal 均在 launch 前拒绝。
`derive_g14r22_holdout_background_job.py` 对原 unsigned request/package 做前后 byte/hash 比较，只在新的 create-only derived
root 写 job package 与 receipt；它绑定 executor commit/tree、唯一 formal output root、唯一 job root，以及 grant/token 的
path/size/SHA-256。token bytes 只用于内存 hash 校验，绝不复制或序列化。

## 3. Terminal 与失败边界

后台状态保持 `RUNNING / SUCCEEDED / FAILED / INTERRUPTED_OR_UNKNOWN`，无自动 retry/resume/reopen/清锁。

- `SUCCEEDED` 同时要求 outer rc=0、合法 `completed_permanently_consumed` execution receipt、
  `holdout_opened=true`、合法 opening receipt（含 atomic-open grant check）、request/package/executor/output-root 完全绑定，
  以及 published integrity exact inventory pass。缺少科学 receipt 永远不是成功。
- child 非零且存在正式 failure receipt 时记录 `SCIENTIFIC_FAILED` 或
  `SCIENTIFIC_FAILED_PARTIAL_OUTPUT`；不存在 receipt 时记录 `CHILD_NONZERO_NO_SCIENTIFIC_RECEIPT`。所有已打开失败仍
  consumed，禁止恢复。
- outer rc=0 但 receipt 缺失为 `SCIENTIFIC_RECEIPT_MISSING`；receipt/identity/integrity 错误为
  `SCIENTIFIC_RECEIPT_OR_INTEGRITY_INVALID`，均为 `FAILED`。
- SIGTERM/SIGINT/SIGHUP 为 `INTERRUPTED_OR_UNKNOWN`；SIGKILL、断电、宿主崩溃可能无法及时写 terminal receipt，
  `inspect` 只读核对 PID/birth/executable/argv identity 后返回 `INTERRUPTED_OR_UNKNOWN`，不会启动第二个 job。

## 4. v5 unsigned 包与隔离公共验收

v5 root：`artifacts/analysis/typed_model_cache_g14r22_holdout_execution_contract_20260926_v5/`。

- request canonical SHA-256=`a3154ffa735fea670bab02dffa6f7b92cca0f15c6fea63d1356d357a0139e717`
- command package canonical SHA-256=`ec7410d0a03c8d73847ec9d73d612ca1f1612b097b227235e7e0a00baf0fffcc`
- output root=`/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_holdout/typed_model_cache_holdout_20260926_g14r22d_once`
  （验收后仍不存在）
- 科学矩阵、150 checkpoints、12 windows、5 seeds、3 workflows、3 capacities、8,100 rows、84 statistics family、
  10,000 bootstrap 与 2,500 CPU-hours 上限均未改变。

隔离 fixture 从 clean fixed executor 实际执行：synthetic unsigned package + synthetic exact grant/token references →
授权后派生 → 公共后台 `launch` → fixed dedicated runner。正常链 terminal=`SUCCEEDED`、outer rc=0、published inventory
`10/10` exact；child 非零链 terminal=`FAILED / SCIENTIFIC_FAILED_PARTIAL_OUTPUT` 且 consumed/no-retry。TTL 原子边界、
双向身份互换、缺失/错误 receipt、launcher 退出后独立运行、信号/宿主异常与 token 不泄露均有专项覆盖。fixture 标明
真实 holdout policy runs=0，不能作为正式性能、正式 opening 或 paper claim 证据。

## 5. 保护与移交

G14H02 记录的七个用户文件与四个 continuation locks 的 SHA-256 复核不变；既有模型、数据、结果、账本和 G14R22-B/C
receipts 未修改。正式 root absent，grant/token 均未创建。v5 只可交中央窗口和负责人重新审查；负责人若批准，必须针对
exact v5 request/package/executor/output root 签发新的 grant/token，最大有效期不超过签发后 72 小时。当前状态不是授权，
不得 execute。
