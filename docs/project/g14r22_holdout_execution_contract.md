# G14R22 最小 holdout 执行合同与非正式验收

> 2026-09-23 G14R22-A 勘误：本文件所述 non-holdout scientific child 是公式生成 fixture，原 v1 unsigned
> request/package 已 superseded，不能用作真实 benchmark/checkpoint/rollout 验收或后续签发依据。真实消费链缺口与
> 修复见 `g14r22a_non_holdout_benchmark_acceptance.md`；本文件保留历史记录。

- `reviewed_at`: `2026-09-23`
- `literature_cutoff`: `2026-09-21`（沿用 G14A01；本轮不评价 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_g14r22_holdout_execution_contract_20260923_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `baseline_commit`: `a49684216f5d4e2194cc75f9aca12f7a82512908`
- `executor_commit`: `3d34f2b967ac7f9e2111744665cfe6a5bf0c7b42`
- `evidence_level`: `E2_EXECUTION_CONTRACT_VALIDATED_NON_HOLDOUT_ONLY`
- `verdict`: `READY_TO_REQUEST_HOLDOUT_EXECUTION_AUTHORIZATION`

该 verdict 只表示申请包、命令和工程事务已达到可提交独立授权审查的范围。它不是 grant，不是 holdout 已开启，
也不是 paper-ready 或性能结论。`grant_signed=false`、`execution_authorized=false`、`holdout_opened=false`、
`holdout_consumed_permanently=false`。

## 冻结身份与运行量

unsigned request SHA-256=`6383eb01e497b04a75474cfadaada637bb1f88e3a891a91515901443abca61ed`；
command package SHA-256=`81271e69257e4187c4a159ca2cfe05cce20f8b36cfb9840fbfb424225764e5c8`。
唯一未来输出 root 为
`/Users/howen/Projects/PPO_MEC/artifacts/experiments/typed_model_cache_holdout/typed_model_cache_holdout_20260923_g14r22_once`，
当前不存在。

科学矩阵固定为 12 个 sealed outer windows、15 agents（5 reactive + 10 learned）、5 seeds、3 workflows、
3 capacities。执行分成 3 个串行 capacity child，每个应产生 `12 × 15 × 5 × 3 = 2,700` 行，总计 `8,100` 行。
10 learned agents 的实际模型范围为 `10 × 5 × 3 = 150` checkpoints；reactive agents 无 checkpoint。
primary endpoints 固定 6 项，统计固定 candidate=`sa_ghmappo`、14 baselines、window-outer/
seed-workflow-capacity-inner、bootstrap=10,000、seed=1401、BCa/percentile CI、exact sign test 与 84 项固定 Holm family。

G14R21 corrected package 的 files-canonical SHA-256=
`5efee8e5f0991edaa210fc46994994f2b444a2dd01c58862c3d3636cc8f09414`；paired statistics 与 claim map SHA-256
分别为 `a8e51948d106af0822bb0d38d6a3e3a69ef50db193f31a9bdcb1380028cd2a51`、
`e656586f7006142fb8e4eaf07ff345bfef91c02fb1248415b1c9104e3033ab91`。未来 statistics 必须调用该纠正后的
`analyze_top_journal_statistics.py`，不得回落旧统计或缩小 nullable family。

## one-time opening 与永久 consumed

生产入口为 `scripts/run_dedicated_public_holdout.py`。普通 `benchmark_main_results.py` 继续拒绝
`sealed_holdout`；只有同时给出 dedicated opening receipt、request hash 和 command-package hash 才可进入科学 child。
公共入口在任何持久 opening 前完成 grant/review/token、冻结输入、clean executor commit 和 150 checkpoint bytes
检查；任一失败都不创建正式输出 root，也不启动 child。

opening 使用同父目录临时目录构造 receipt/checkpoint audit/request snapshot/command snapshot/ledger，再原子 rename 成
唯一 root。该 rename 即一次性开启点；opening record 当场写 `consumed_permanently=true`。此后：

- child 失败、进程中断或部分 staging output：永久 consumed，保留失败/部分输出，禁止 retry/resume/reopen；
- statistics、publication 或 integrity 失败：同样永久 consumed，不允许换模型、窗口、seed、容量或新建第二 root；
- 成功：发布 3 个科学 payload、84-row statistics、integrity manifest 与 completion receipt，仍永久 consumed；
- `automatic_retry_count=0`。旧 seal 中更宽的 infrastructure-retry 文本在本合同不被使用。

opening ledger 是 fsync 的 append-only SHA-256 chain。科学 child 只在 staging 写入，完整 payload 才原子发布；
producer manifest、最终 published inventory、execution receipt 和 ledger terminal 相互闭合。未新增通用授权平台。

## checkpoint 字节核验

预授权只读核验实际重新打开并散列全部 150 个文件，`150/150` 通过，总字节数 `105,464,376`，坐标覆盖
10 learned agents、5 seeds、3 capacities，aggregate SHA-256=
`2f6c602b8864e331eb70009b73a56984a41f046620b03be74ffd653ea577e976`。首次 shell 观测约 3 秒；随后缓存已热的
package builder 观测为 0.089 秒。两者只说明本机 checkpoint opening gate 的量级。正式 runner 会在 atomic opening
之前再次对同 150 文件逐字节复核；不是复用本轮结果替代 opening-time check。

## 非 holdout 公共入口验收

固定 executor checkout `/Users/howen/.codex/worktrees/g14r22-holdout-executor/PPO_MEC` 位于 clean commit
`3d34f2b...`。非 holdout fixture 通过真实公共 runner 完成 3 个 child、producer payload、原子发布、纠正后统计、
84 项 family 校验、integrity consumer、receipt 与 ledger，实测约 4 秒。验收记录明确
`holdout_opened=false`、`holdout_policy_runs=0`；fixture 中没有 `sealed_holdout` 或 holdout policy 标识。

首次非 holdout 验收因 fixture 写 `agent` 而统计消费者要求 `agent_name`，在三 child 成功后 statistics rc=1；该 root
按合同永久失败且不能恢复。只修 fixture 后使用全新 root 重跑通过，未修改生产科学参数。专项测试还覆盖 exact 150
坐标、request 未授权边界、84 family、opening receipt 和 acceptance 禁止引用 sealed split。

## 工作量、耗时与资源边界

- 准确工作量：150 checkpoint opening hashes；3 serial scientific children；8,100 expected rows；84 statistics rows；
  1 publication closure；1 integrity scan；1 terminal receipt。
- 历史观测仅能作为下界参考：相同 formal cache-policy + controller 约 `5h12m`，statistics/gate 约 `15m`。
- 非 holdout tiny acceptance 约 4 秒；checkpoint hashes 本机观测 0.089–3 秒。
- 完整 holdout 的经过测试 wall-clock 上界：`unknown/null`；不提供 calendar promise。
- 资源：CPU、科学 child 串行度 1、自动重试 0；沿用冻结 maximum total compute budget 2,500 CPU-hours。

## 申请与禁止项

机器包位于 `artifacts/analysis/typed_model_cache_g14r22_holdout_execution_contract_20260923_v1/`，包含 unsigned request、
完整 command package、checkpoint byte audit、integrity checklist 和非 holdout acceptance receipt。未来必须由未参与实现的
独立 reviewer 对 exact request/package/commit 做只读审查，再由 owner 单独签发 exact grant 与 token；本任务不产生二者。

本轮未训练、未选模、未重跑 formal、未读取 holdout performance、未调用 holdout policy、未签 token/grant、未创建
正式 holdout root，也未修改 G14R21 formal 原始结果或七个用户文件。因此结论仅为“可以申请授权”；最终只申请
holdout 执行授权，不自行执行。
