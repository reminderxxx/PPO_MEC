# CSCWD sharing × migration 首次单次四臂开发执行方案

- `frozen_at`: 2026-10-11 Asia/Shanghai，任何四臂原始数据 `env.step` 之前
- `target_venue`: IEEE TMC；`policy_version`: `tmc_review_policy_v3_20260621`
- `artifact_run_id`: `cscwd_sharing_migration_2x2_development_20261011_v1`
- `evidence_scope`: 已暴露 development 机制诊断；0 训练/真实模型调用/formal/holdout
- `execution_authority`: 用户本轮继续推进授权；仅此一次、最多 4 episode/96 `env.step`

来源仅取已批准 `cscwd_new_development_source_20261011_v2` 的两个长区间，按其冻结顺序选择**第一个首步公开 `action_mask[4]` 合法**的区间。只读零步预检：Lankershim 车辆 215 为 `[true,false,true,true,false]`；US101 车辆 1183 为 `[true,true,true,true,true]`，故唯一入选 US101 `Global_Time` 1118849293800–1118849412600、`Frame_ID` 5368–6556、1189 连续帧。原 CSV SHA-256 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`；source manifest SHA-256 `34eee3a8d16f2dfe89f3a8d8dcd27f20159c0488e2f1a9ebc77be3aac747abde`。所选原始闭区间与来源 manifest 的保护区间重叠 0，但历史 train/dev 暴露计数 2；它也进入过本项目开发诊断，**不构成新的独立测试来源**。Peachtree 10 位时钟不纳入；I-80 seal 不开。

工作负载固定 `calibrated_continuous_workflow_interface_repair_v3_manifest.json` 的 `dev_01`：8 个 DAG 节点，`helmet_shared`、`alpr` 同需 `base:family_a`；原节点/边、输入、模型功能身份、deadline、奖励、外生轨迹不变。四臂统一 cache 容量 `2,194,117,040 bytes`、原始逻辑 resident、`default` 模拟 backhaul 1000 Mbps；容量已在工程 gate 中事前声明，用来容纳两份 private base，不随结果调整。`S0/S1` 只改变是否重复驻留/传输同一 base；`M0/M1` 只改变状态交付，M0 仍合法准备目标并服务当前节点，后续按原 DAG 重算。四臂顺序 `S0M0,S1M0,S0M1,S1M1`，每臂新环境，从同一初态起步。

所有决策使用同一固定公开规则：只看当前 `_info().action_mask`，按 `[4,3,2,0,1]` 取第一个合法动作；不看真实未来、exact clone 或事后效果。每步记录请求/执行动作、mask、相位 ledger、缓存交易、字节/时钟/节点/失败、终止与截断。逐臂 create-only 原子写入；单次失败停止并保留 partial，不替换来源、不调策略、不追加 run。输出 root 只用 `artifacts/analysis/cscwd_sharing_migration_2x2_development_20261011_v1/`。输入 SHA、代码 commit、消费者 SHA、授权位、来源身份、输出 root 和硬预算必须先在机器 gate 锁定；dirty 或任何不匹配均拒绝启动。

主报四臂 workflow completion、on-time、service failures、total/model/state/input/backhaul bytes、modeled completion/step/transfer/load/restore/recompute seconds、base admission/reuse、迁移成功交付及后续目标服务。对同一 endpoint 给描述性的 sharing/migration 主效应及 `I=(Y11-Y10)-(Y01-Y00)`；若有截断或完成量不等，仍保留四臂原值和描述性差值，但成本不能表述为优势或可比交互。仅一条已暴露来源，不计算独立 CI、不给论文贡献晋级。全零、负例或未激活机制原样保留。
