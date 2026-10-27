# TGSIM查询恢复合同（2026-10-27）

用户在知悉分页HTTP400及错误正文缺失后明确指示“行 你负责 继续做完这件事情”。
本轮另立任务，旧bulk与分页run、收据和失败结论封存，不在原run重试或改写。

先修复并验证HTTPError响应头和有界实体捕获：同一次请求的错误响应最多保留64KiB，
计入body/磁盘预算，标记diagnostic-only并仍然失败；不产生第二个HTTP请求或成功CSV。
旧v2收据不改写；新v3消费者继续按失败状态、原件hash和实际保存body计数读取。

独立诊断阶段最多3个预声明请求、256KiB返回body、120秒；新旧原件/收据/索引累计2GiB。
先观察原冻结查询的一次HTTP400完整错误实体；仅明确解析/语法错误后，执行两个事前固定的
显式字段查询，各源只取1行，字段、站点及原生数值不变。未知错误、认证、限流、传输失败、
schema差异或任一修正查询失败即停止，无重试、来源切换或额外查询购物。
显式字段只用原schema清单，替代混合wildcard；请求顺序仍为系统:id。

2026-10-28诊断已确认服务端编译错误：star selection必须位于select-list开头；
显式原生字段查询在第一源返回HTTP200。随后本地旧ID格式校验误拒了官方点号/下划线ID，
该次诊断以软件校验失败封存。已完成独立诊断再修复，保留其2个观察，不重复请求第一源。
只在新目录执行尚未消费的第三个观察（第二源单行），三观察总预算不扩。

官方行ID是opaque identity；支持点号/下划线/波浪线，不把其编码字典序冒充服务端顺序。
完整恢复阶段保留:id排序请求，并用独立SQLite主键验证跨页系统ID唯一；原生run/id/tick
另作唯一与精确时间检查。版本/count/完整页清单共同核验，仍不声称服务端事务快照。
原始页可协商gzip传输，原样保存encoded body，512MiB明确按保存的传输实体计；
流式解码CSV文本最多1.2GiB、单行65536字符，原生派生CSV和两个身份索引均计入累计2GiB。
CRC损坏、解码/行/字段/原生身份/预算错误均停止，不修复数据。

入口：`scripts/diagnose_tgsim_query.py --plan <frozen_plan.json> --plan_sha <sha256>`。
计划绑定本轮直接人工指令、既有失败诊断、完整URL和新输出目录；从clean commit只执行一次。
软件资格：`python -m pytest -p no:cacheprovider tests/test_tgsim_input_preparation.py tests/test_tgsim_paged_preparation.py tests/test_tgsim_query_diagnosis.py -q`。

诊断成功后才能另行冻结恢复获取的具体执行计划：仍限原两源和16附件、最多512MiB新增body、
累计2GiB磁盘、45分钟、600万原生索引行与144请求；页大小保持50000，失败停且不重试。
所有新耗费单列并与旧任务累计报告，不将本次重新授权说成旧任务自动续跑。
恢复输入不等于自动获得后续科学训练、formal/holdout、provider/坐标/原RSU/费用资格。

恢复入口为 `scripts/recover_tgsim_paged_input.py --plan <new_frozen_plan> --plan_sha <sha>`；
消费者为 `scripts/audit_tgsim_paged_input.py --expected_proposal_sha <same_sha> --analysis <new_run> --output <fresh_audit>`。
42项纯软件资格通过，原收据/失败和科学结果均不改写。

算法、奖励、预测器、环境、规则和其他基线不变；训练、策略评价、模型拟合、hidden开启均为0。
不会用新数据扫描或单行探测声称独立采集、hidden、完整输入或论文目标达成。
