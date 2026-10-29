# 本机组件成本：单次执行、独立审计与交付（2026-10-29）

本任务补到了真实本机组件的可执行性和仪器化成本；没有验证驾驶质量、无线/RSU或当前stream的净准备收益。
全部28个事前attempt成功，自动独立审计只运行一次，终端后复用审计导出，不重跑模型、传输、旧审计或重放。
整体`paper_goal_complete=false`，原main保留，head等权候选拒绝结论不变。

- reviewed_at / literature_cutoff: 2026-10-29；复用原12篇矩阵，无新文献购物
- target_venue: CSCWD2027；TMC仅内部证据规范
- policy_version: `tmc_review_policy_v3_20260621`
- measurement_git_commit: `d5d7e3d97415bd5dab7c8d115c2eb62997c750de`
- artifact_run_id: `paper_evidence_strengthening_20261029_v1/local_cost_execution`
- evidence_level: 本次技术支持原件自动审计及独立交付核对；局部E2，不是整篇E2或E3重放
- plan_sha256: `d517a8b2d305974badd6632d36a7dae4b8ab206eee6f23d1916b2e5871b72332`
- audit_sha256: `4c4508fb05f0decaacd9e2522bb379b99b314a64bf01414c1fe55ffe59ae5e01`
- verdict: `AUDITED_PASS_TECHNICAL_COMPONENT_COST_ONLY`

## 固定测量范围和账单

同一Apple M5、24GiB、CPU float32、batch1、intra/inter-op各1线程；固定旧解释器Python3.9.6与
torch2.8.0/transformers4.49.0/peft0.14.0/accelerate1.2.1/safetensors0.5.3，全部offline。
基座与两个adapter仅使用本地冻结资源。CPU生成greedy、beam1、最多8新token，无prefix/图像读取。
ALPR/Helmet分别rank64 DoRA/rank4非DoRA；技术suffix不具有适用任务标签。

12个模型进程和16个非模型helper均独立、串行并完成清理；另1个自动独立audit。
实际processor/base各12、adapter load16、activation20、generate28、state encode/save/restore/rebuild各4、loopback12。
7个ordinal0资格attempt与21个正式attempt分开；32个组件组各保留3个正式值，不删异常值或选快样。

producer含资源资格及inventory为145.706258秒，consumer0.821651秒，合计146.527909秒，小于5400秒。
自动delivery保留665,022,277字节；全D含launch/control、独立审查及两轮可视化QA后为666,628,155字节
（最终receipt写入前快照），仍低于1GiB；全D日志2,639,355字节，低于64MiB。
payload656,269,664字节、应用协议656,271,008字节；后者多12×112字节header/ACK，不包括TCP/IP或重传。
原A+B科学帧292,286不重置；本任务训练/环境评价/拟合/新候选/下载/hidden全部0。

## 实际组件结果

下表均为3个正式值的均值±1样本SD，单位毫秒，不是CI或跨硬件总体统计。全部3值、资格样本、
嵌套区间、process residual以及未在表中选列的生成/驻留A-B-A各组件均保留在CSV。

| 仪器化组件 | ALPR | Helmet |
|---|---:|---:|
| processor load | 72.294±0.615 | 71.787±0.559 |
| base load | 263.879±1.071 | 263.833±3.828 |
| adapter load | 864.148±3.991 | 376.054±4.843 |
| activation含完整性核验 | 83.233±0.744 | 21.077±0.178 |

| 2,040B应用状态组件 | 均值±样本SD，ms |
|---|---:|
| encode | 0.073445±0.000819 |
| durable save | 0.505472±0.034131 |
| read/validate restore | 0.114583±0.011250 |
| prompt rebuild | 0.197861±0.106117 |

| 仪器化本地文件传输 | payload B | 均值±样本SD，ms |
|---|---:|---:|
| 状态包 | 2,040 | 11.989±1.452 |
| Helmet weight | 9,641,944 | 797.565±124.323 |
| ALPR weight | 154,423,432 | 12,230.443±1,599.025 |

ALPR的实际生成长度为8 token，Helmet为5 token；同一max_new_tokens上限不等于同等实际工作。
T1首生成均值分别3845.474/187.126毫秒、warm分别3840.105/163.119毫秒；只能作为各自技术路径的描述。
不能把其差值归因为adapter大小、rank/DoRA或缓存机制，更不能当任务质量/公平速度优势。

## 解释与尚未覆盖风险

- `process-first`不是清空OS page cache的storage-cold；测量前已对本地资源做hash资格。
- activation包含`set_adapter`、status、参数CPU复制、非零计数和SHA；纯API切换时间没有单独测量。
- 本地transfer包括逐chunk hash/账本、磁盘guard、接收file/fsync、ACK及teardown，不是纯TCP吞吐或无线链路。
- encode/restore等局部时间不含process startup/supervision residual，不能直接相加宣称端到端部署时延；嵌套时间不能重复相加。
- ALPR输出仍是旧技术文本“2010 - 2”，Helmet为“Sales by Month.”；一致性不是驾驶任务正确性，也没有证明KV/tensor恢复。
- 当前stream对象尺寸、provider/RSU/时钟与实测对象没有完成事前映射，未拟合参数，break-even仍`UNAVAILABLE`。
- 全篇独立formal/hidden、强外部公平对照和模块必要性仍缺；技术支持通过不改变全12学习回退或候选拒绝。

下一真实质量任务需要已有、获许可、适用ALPR/Helmet且带独立groundtruth的冻结图像/标签来源；
当前没有合格manifest。独立采集/站点几何/无线费用等同样需真实来源与各自有界合同，不自动增训或下载。

## 图件、CSV与复用

本地产物在主目录私有`artifacts/analysis/paper_evidence_strengthening_20261029_v1/local_cost_execution/`：

```text
plan.json / grant.json / source_freeze.json / launch.json
run/                         # 原始一次执行，不改写
audit.json / delivery.json   # 自动单次审计，禁止重跑补图
independent_result_review.{json,md}
delivery_bindings_verification.json / delivery_complete.json
delivery_exports_v2/
  cost_components.{pdf,svg,png}
  component_summary.csv      # 32组，每组全部3正式值
  native_intervals.csv       # 244行，28attempt含7资格/21正式，嵌套/残差单列
  resource_bill.csv / caption.txt / manifest.json
```

首轮图的最右标签在PDF渲染中截断，原图和失败QA保留；第二轮使用标明十进制MB的短标签及宽x边距。
两轮三份CSV逐字节相同；第二轮PDF渲染150dpi视觉检查通过，无截断/重叠，导出图为600dpi PNG及矢量PDF/SVG。
导出只读消费已绑定audit/delivery，原件、图和私有审查不进入Git。

可复用入口为`scripts/plot_local_adapter_cost_support.py`，须提供精确plan/audit/delivery SHA和fresh输出目录。
验证执行了该导出命令、32组/244行CSV逐项回核、两轮CSV hash相同、PDF渲染视觉检查及AST检查；
没有再次运行生产端、审计器、模型、传输或任何科学重放。
