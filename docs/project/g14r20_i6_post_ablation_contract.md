# G14R20-I6 八-cell 后续评估衔接合同

- 日期：2026-09-21（Asia/Shanghai）
- `reviewed_at`：2026-09-21
- `literature_cutoff`：不适用；本任务仅作执行工程审查
- `target_venue`：IEEE TMC 项目目标；不作 paper-ready 判断
- `artifact_run_id`：G14E06 八-cell handoff；G14E07 尚未创建
- `policy_version`：`tmc_review_policy_v3_20260621`
- `evidence_level`：八-cell 与模型 E2_ARTIFACT_AUDITED；后续正式结果 E0_UNAVAILABLE
- 状态上限：`READY_FOR_POST_ABLATION_EVALUATION_AUTHORIZATION`

## 已核验基线与实际缺口

G14E06 handoff 固定 hash 为 `43792b96592619d18b097ef8969dcfcdd4155dc73e277d25c44f9d44d2d8e3ab`，由原 run 的 3 cache、3 controller 和 recovery run 的 2 ablation 构成；两个消融 attempt 为 3、1。八个结果的 ledger、marker、事务 inventory 和 producer manifest 双层重算均匹配，150 个冻结 checkpoint 与 freeze、seed manifest、provenance manifest 一致。历史 `historical_start_evidence=unavailable`、`historical_protection_verdict=UNVERIFIED` 保持原样。

旧 evaluation-only runner 把前八阶段都当成本 root 已执行；statistics 只查本 root 的 controller rows，gate 只扫描本 root 输出和 cell ledger。scalability 的 oracle replay 还会指向新 root 下不存在的 medium cache cell。I6 以固定 handoff 只读引用解决这些输入，给剩余五阶段建立独立 identity、request、context、ledger 和 create-only root。历史八 cell 仅引用，不复制、不 symlink、不重跑。原 run、recovery run、旧失败 terminal 和 held lock 均保持只读。

原 cache cell 的 handoff `coordinates` 可含 `/ABSOLUTE/FORMAL_OUTPUT_ROOT`，属于历史审计 metadata；它不进入本任务实际执行 argv。申请生成器逐条拒绝未解析 `/ABSOLUTE/` 或花括号，scalability replay 绑定到外部 medium committed cell 的实际路径。

## 冻结阶段依赖

| 阶段 | 冻结 live 数量 | 生产者 | 消费者与前置依赖 |
| --- | ---: | --- | --- |
| 已提交 cache/controller/ablation | 3/3/2 | G14E03 原 run、G14E06 recovery | 新 runner 逐项核对 ledger、marker、事务 inventory、producer manifest；statistics 读三个 controller rows，scalability 读 medium replay，gate 计入 3/3/2 |
| `formal_support` | 11 | 公共 support → nested benchmark → producer manifest → cell transaction | 新 root 的 support cell ledger；依赖冻结 checkpoint/manifest/context |
| `formal_scalability` | 3 | 公共 support → oracle → producer manifest → cell transaction | 新 root 的 scalability ledger；外部 medium request replay |
| `formal_statistics` | 1 | 公共 statistics → analyze | 显式八-cell handoff 中三个 controller rows；冻结统计参数不变 |
| `formal_gate` | 1 | 公共 artifact manager | 外部 3/3/2、本 root 11/3、statistics、生成资源 registry、完整性 manifest |
| `complete_without_holdout` | 0 条 child 命令 | 公共 completion gate | `formal_gate.json` 必须 `passed=true` 且 phase terminal 完整；holdout 保持关闭 |

冻结矩阵中的 support sensitivity（object size、transfer cost、reuse opportunity、base sharing）、typed semantics 四个扩展以及 scalability 的 RSU、车辆、DAG node、typed object 四维均为 `unavailable_pre_execution`，不得转为 live。可执行的 scalability 仅 oracle state limit 1000/10000/100000。

## 授权与验收边界

正式 unsigned request、完整命令包和 create-only 证明保存在 `artifacts/analysis/g14r20_i6_post_ablation_20260921/`。中央窗口须另行审查并签发与该 request hash、executor commit/tree、handoff hash、五阶段及有效期完全一致的 G14E07 grant；本任务不签发 grant、不启动正式矩阵、不打开 holdout。非正式 acceptance root 与 grant fixture 仅供测试，`formal_performance_evidence=false`，不得进入论文统计。

非正式链与回归的实测记录、JUnit、完整性清单和失败根保留在同一分析包；失败日志不是通过证据。正式时间与资源估计按原六 cell 和两消融的实际 duration，以及本轮非正式各阶段 wall time 和内存观察给出区间，不承诺逐 cell 精确完成时刻。
