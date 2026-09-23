# G14R22-B 容量身份闭环与低额度后台执行验收

- `reviewed_at`: `2026-09-23`
- `literature_cutoff`: `2026-09-21`（本轮不评价 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_g14r22b_capacity_statistics_20260923_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `evidence_level`: `E3_TARGETED_REPRODUCED_WITH_VERIFIED_CAPACITY_IDENTITY; REAL_PUBLIC_CHAIN_FROZEN_PRELAUNCH`
- `grant_signed`: `false`
- `token_issued`: `false`
- `holdout_opened`: `false`

## A. 容量字段与分组

未来 typed benchmark 只从已解析的 `runtime_config.<capacity_label>` portable resource 取得 label，并对照 runtime
contract 的 enabled/MB/实际数值；288/576/864 MB 任一不匹配即在 rollout 前拒绝。验证身份进入 episode summary、
aggregate、run manifest 与最终 CSV；统计要求 pair=`seed,window_id,workflow_id,capacity_label`，
outer=`source_segment_run_id,window_id`，inner=`seed,workflow_id,capacity_label`。

统计 consumer 对字段缺失、null、blank、错误类型、重复 coordinate、缺少 candidate/baseline pair 和跨 agent cluster
冲突分别报错；输入文件路径不再隐式参与 pair。手算测试的两个容量 delta=2/10 保留两个 pair、两个 inner cluster，
window mean=6，不发生覆盖或合并。

## B. G14R21 影响

旧 8,100 行 CSV 没有容量字段。新包通过 hash-verified recovery handoff coordinates、各 producer integrity manifest、
runtime MB 和 aggregate selected-window plan 在独立 `analysis_inputs/` 补充来源标签；不修改旧 CSV，不按效果或目录名
推断。三份原件 SHA-256 前后相等。

冻结科学预算不变：6 endpoints、14 baselines、84 项 Holm family、10,000 bootstrap、seed 1401、BCa/percentile、
相同模型/窗口/原始行。540 pair coordinates 与 12 outer windows 不变；普通 endpoint inner clusters 180→540，delay
85→220。84 项 mean/effect/window W-T-L/sign p/Holm/coverage 不变；CI low 14 行、CI high 59 行改变，claim transition
为 mixed→mixed 72、contradicted→contradicted 12，Holm<0.05 仍为 0。以上分类从新结果导出，没有硬编码 0/72/12。
旧 G14R21 包保留，仅适用于 signed-direction/window-sign 修正的历史审计，不能再作为 capacity-aware clustering 证据。

## C. 后台宿主保证与限制

宿主只执行一个冻结命令。package 固定 cwd、Python 调用路径及解析目标、环境、request/scientific
package/commit/tree/hash；记录外层
stdout/stderr、child rc、supervisor/child PID + process birth token + live command hash、起止时间和 terminal receipt。
状态仅为 `RUNNING/SUCCEEDED/FAILED/INTERRUPTED_OR_UNKNOWN`。child rc=0 不足以成功；科学 receipt 必须
`passed=true`、`return_code=0`、`holdout_opened=false`，published integrity manifest 还必须逐文件通过。

无自动 retry/resume/recovery/清锁/第二 run，无 AI/API 环境或常驻监控代理。launcher 退出后 supervisor/child 使用
独立 session 继续。SIGTERM/SIGINT/SIGHUP 可捕获并转发，最终为 `INTERRUPTED_OR_UNKNOWN`；SIGKILL、断电或宿主崩溃
可能来不及写 terminal receipt，后续 `inspect` 只读比较 PID/birth/command identity 后推导 unknown。实现不承诺任意
崩溃都即时留下回执。

## D. 申请边界

容量感知历史复算和宿主专项测试已通过。真实 public non-holdout producer→CSV→statistics 长链已冻结在 clean
executor commit `d6c53b4154f84dcb398ceb0f88ace642142c0f23`、tree
`1068b70d642484c9cadadf9380ec2245b640b4e1`；request SHA-256=
`398e0c386104f1e8513a93d671ef419b5265eda20c0ee2b26b8f6f96d5926331`，scientific package SHA-256=
`cd859dc4af0634fe746dc4c87739604976ac01903a6eddc3061ce7a29b243bcd`，job canonical SHA-256=
`1f0965eecc34865421858093581a68a65c9ca90eab2bcb1f8cc06bc59bfe63d6`。前一份错误解析 venv symlink 的包从未启动，
仅留在 `/tmp` 作本机审计，不构成 retry。只有该链完成并经独立只读审查，才生成替代 v3 的新 unsigned 申请并判断
是否可提交授权审查。v3 永久 audit-only，不覆盖旧申请或原件。本任务不签 grant/token、不创建正式 holdout root、
不训练、不选模、不重跑既有正式 rollout。
