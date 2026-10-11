# CSCWD sharing × migration v2：零步修复门与新运行身份

- `frozen_at`: 2026-10-11 Asia/Shanghai；`policy_version`: `tmc_review_policy_v3_20260621`；`target_venue`: IEEE TMC。
- 旧 `cscwd_sharing_migration_2x2_development_20261011_v1` 永久保留 `PRESTEP_FAILED_NO_RETRY`，0 环境步；不修改其 receipt。
- 修复提交 `0da1ed2de5b54368e5d58dfdd506f51350075673` 仅按 `load_traces()` 的真实 `window_identities` wrapper 取身份，并对所有原始 identity 字段 fail-closed；新增真实固定 CSV 的四臂零步构造/开关/共同输入校验。执行器 SHA-256 `455c96d1944278a89c8752002c80c7e897e54e94f2311a6dcc351eb62ee988f7`。
- 当前机器门 `execution_authorized=false`、`READY_ZERO_STEP_ONLY`。唯一预检输出为 create-only `artifacts/analysis/cscwd_sharing_migration_2x2_zero_step_20261011_v1/`；预检必须真实重载已冻结 US101 与 Lankershim 来源并核对 source manifest/protected interval、四臂初态与消费者因素、0 `env.step`。任何失败停止科学执行。
- 若零步门通过，用户已另行授权冻结新提交并只执行一次 `artifacts/analysis/cscwd_sharing_migration_2x2_development_20261011_v2/`：同旧计划的 US101 车辆 1183、`dev_01`、`default` 模拟链路、2,194,117,040-byte 容量、固定公开 `[4,3,2,0,1]` 策略及 `S0M0/S1M0/S0M1/S1M1`。最多 4 episode/96 真实环境步；0 训练/模型调用/formal/holdout。不改来源、预算、指标或策略来激活机制。
- US101 原区间与历史 train/dev 有 2 次暴露；新 root 只表示工程失败后的独立**执行身份**，不表示独立统计来源。任何截断/完成量不等原样保留，成本不得表述为优势。
