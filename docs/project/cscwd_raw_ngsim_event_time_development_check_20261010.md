# CSCWD 原始 NGSIM 事件时间：冻结开发检查（2026-10-10）

- `reviewed_at`: `2026-10-10T14:48:00Z`
- `literature_cutoff`: `2026-10-10`；本轮仅核验公开 FHWA NGSIM 字段/采样说明，未做 novelty 评价。
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`；下述结果不是投稿结论。
- `artifact_run_id`: `cscwd_raw_ngsim_event_time_20261010_v4`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: `0704937742ec9218094f380a1b541c4bd2159bcb`（最终运行代码；环境实现主体为祖先提交 `ade1561a5748018b189aeebd9d926c1b7fd2f643`）
- `evidence level`: `E1_DOCUMENTED` 对 paper-ready 问题；本地开发原件和原始文件身份已核验，但无本剖面的正式 checkpoint、formal/holdout/support 原件。
- 结论：**开发时间/接触合同可运行；当前 24 帧窗口不足以检验准备/迁移和优秀基线。算法优势、固定论文贡献与 paper-ready 均为 `Unverifiable`。**

## 完整性与 protocol/provenance

事前输入合同先提交为 `79ce5ab`，B 独立指出 `trace_remaining_seconds` 不应作为公共输入；实现前将其移出 policy semantic。最终还将继承自合成路由的预测 confidence 换成固定未校准 0.5。运行代码为上述 `0704937`，命令是 RUNBOOK 中的 create-only `--output-root ..._v4`，使用本地原始 CSV、冻结 v3 manifest/config、三个已暴露 development ID；没有下载、训练、checkpoint 选择或 holdout 访问。`v1/v2/v3` 是在实现身份、执行动作计数或公共字段收紧前的核对检查，均不用于本报告数字。

本地完整原件：`artifacts/analysis/cscwd_raw_ngsim_event_time_20261010_v4/source_manifest.json`（SHA-256 `c0b64fd64c602ef5351c5686d2f315bbd5784e63ebbd08902b297fc5bacc883f`）与 `summary.json`（SHA-256 `62c1f4c2c7fbb2e2763f6946c0fad29b4cd6a3705ebf01501a7d905d8b1fe30f`）。原始 CSV 为 2,118,175,938 bytes，SHA-256 `ddacb7a0391c6ab80fd4085d1380096733b17882081ae83b40174b8ec662d10c`；11,850,526 行只读扫描，未将坐标、checkpoint 或真实数据提交/上传。父 manifest SHA-256 `b7efe5d6e50ad17b744230c03a126370656d0196b60c7f7e6b5a86bb702effbc`。

| development ID | segment | 原始 Global_Time 闭区间 | 首帧最小 vehicle ID | 连续 Frame_ID | 匹配原始行 |
| --- | --- | --- | --- | --- | ---: |
| `dev_01` | `us_101` | 1118846994600–1118846996900 | 2 | 157–180 | 1,058 |
| `regression_00` | `lankershim` | 1118936085200–1118936087500 | 456 | 4053–4076 | 1,561 |
| `regression_05` | `us_101` | 1118847021000–1118847023300 | 2 | 421–444 | 2,759 |

三者均含 24 个连续 100 ms 时间戳，首个决策在第 2 帧，剩余原始区间仅 2.2 s。两个 US-101 区间不重叠，但共用车辆 ID；它们不能当独立统计 cluster。这里没有 formal/holdout 独立性证据，也没有触碰相关窗口。原始轨迹给出时间/位置，不给 RSU、无线、缓存或 workflow 真值；三圆几何和链路/DAG 成本均为申明的模拟参数。

## 统计/基线公平性与机制兑现

按事前上限执行 3 实例 × 5 个重复固定动作 × 2 剖面 = 30 episode、147 真实 step、30 次内部预览；没有策略训练、调参、补窗或结果择优。五个动作是接口探针，不是 SA/PPO/MAPPO/DT 或强启发式的公平重训。旧 `decision_step_original` 与新 `raw_ngsim_event_time_v1` 的时间/接触语义不同，表中差异仅作边界敏感性。

| 剖面 | episode / step | 轨迹/步数截断 | 完成 DAG 节点 | 完成 workflow | 服务失败 | 模型 / 状态 / 输入传输 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧决策步 | 15 / 117 | 2 | 84 | 13 | 33 | 1,534,440,048 / 1,462,968 / 10,534,166 bytes |
| 原始事件时间 | 15 / 30 | 15 | 2 | 0 | 28 | 全为 0 bytes |

新剖面的 28 次拒绝中，24 次是原始窗口在完整服务提交前结束，4 次是当前 RSU 接触先结束。30 个新剖面决策中，10 次请求被公共 action mask 改写；实际 action1 和 action4 各执行 2 次，仍无迁移成功。全部 15 个 episode 最终为 `truncated=True, terminated=False`，应按非终止 bootstrap；没有一次成功 prepare/migration。独立合成测试覆盖成功节点的时间/字节守恒、失败不偷移时钟也不提交缓存、接触边界、轨迹末端和未来轨迹/终点篡改下公共状态不变。保守准入把模型传输、模型加载、重算、计算、状态恢复与最终提交视作必须留在当前接触内完成的整步；未实现部分传输或跨界远端服务。这是实现范围，不是真实平台证据。

## Claim 边界与下一实验门

本轮只能固定一个可审计的共同事件时间/公共观测合同，不能固定 SA 的算法贡献，也不能宣布任何优秀基线胜负。当前窗口长度与既有 8 s 车辆 fallback、模型加载时长不相配，prepare 和 serve 标签无法同时形成；因此 B 的匹配训练/标签阶段按预冻门停止。本轮不得用 `v1/v2/v3/v4` 中任何结果挑选更长窗口或参数。若继续，先另立结果盲的**较长原始开发区间**选择协议，核对与所有正式/holdout 原始 frame/time 区间互斥并保留完整失败样本，再独立验证公共估计器、跨界能力边界、prepare/serve 可达性；只有通过后才考虑同环境、同预算的 SA/PPO/MAPPO/DT/强规则匹配训练。文献 novelty 此轮未审，不能据此宣布贡献新颖。
