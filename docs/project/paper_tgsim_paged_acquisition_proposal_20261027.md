# TGSIM分页输入获取草案（2026-10-27）

状态：`HUMAN_GRANTED_SOFTWARE_QUALIFIED_NOT_EXECUTED`（2026-10-27）。人工明确答复
“允许按分页新方案继续”，独立grant绑定原始私有方案SHA `71abfd4d…15a64`。
以下原范围不变；原始私有方案和首轮5eb3804失败run继续封存，不续写或重试。
新增分页执行器与独立收据消费者的34项纯软件验证通过；匿名API能力仍未验证，尚未启动真实获取。

## 已核实的失败与取法边界

首份官方CSV返回HTTP200/text-csv/chunked，随后在chunk内部结束。完整轨迹为0，
第二份轨迹和16个几何附件尚未请求；前后版本核验只完成前置部分。
完整私有失败收据、原件、hash和独立诊断留本地。不把部分文件拼接或当新完整输入。
弱ETag、未声明Range且动态导出序列未认证，不能直接信任字节续传。

拟改用同一官方域名、同两个asset的SODA2.1分页CSV接口，减少单次响应体。
官方文档支持[limit/offset分页](https://dev.socrata.com/docs/queries/limit.html)，并要求
[明确稳定排序](https://dev.socrata.com/docs/queries/order.html)；
[`:id`系统字段](https://dev.socrata.com/docs/system-fields.html)可用于行身份。
当前[接口说明](https://dev.socrata.com/docs/endpoints)也指出SODA3查询需要token/认证。
本草案只尝试公开SODA2.1接口；这些资产是否支持匿名分页尚未验证。
若需认证、报限流、返回不支持字段/版本、源身份变化或任何页失败，直接停止保留，
不申请账号/token、不改走其他endpoint、不自动重试。分页不保证成功或独立性。

## 具体新范围与预算

- 仍仅 `9uas-hf8b` 与 `7zjf-a4zf` 的完整轨迹，以及原清单的16个几何附件ID；无新来源。
- 原API metadata与schema/rowsUpdatedAt/附件身份继续绑定；新任务前后各核对两源metadata及row count。
- 轨迹模板为 `https://data.transportation.gov/resource/<asset>.csv`，
  固定 `$select=:id as source_row_id,*`、`$order=:id`、`$limit=50000` 和事前计算的offset。
  行数用同asset的`.json?$select=count(*) as count`核对；参数全部URL编码并记入收据。
- 每页最多50000行、两源最多120个轨迹页；含4个metadata、4个count和16个geometry请求，
  HTTP请求总上限144。每页/附件只请求一次，顺序固定，不换page size或用结果选窗口。
- 新任务下载上限512MiB、45分钟、600万索引行；新增存储**累计**不超过2GiB，含首轮失败原件，
  原始页、派生CSV、索引与新旧收据，不通过删旧失败腾账单。
- 旧任务已存/返回body约178.75MiB；按中断读取保守上界计≤178.82MiB。
  新任务若用满512MiB，两轮累计下载body上界≤690.82MiB。旧耗费单列，不能声称仍只花512MiB。
- 全部写入新 `tgsim_paged_input_qualification_20261027_v1` raw/analysis目录；已有目标即拒绝。
  保留逐页字节/hash、URL、系统行ID和row count。只在完整性全部通过后写完成标记。
- 拼成的native-field CSV是**分页来源的派生组合**，不声称与原bulk文件字节相同。
  验证总行数、系统行ID无重/分页顺序、原生source/run/vehicle/time唯一及原字段；
  不默默删行、平均、补帧、插值或拼接旧失败前缀。版本/计数稳定也不等于服务端事务快照。

## 执行前资格与结果限制

先独立实现并用纯软件样例验证分页边界、版本漂移、重复/缺页、字节上界与硬预算；
人工新答复绑定本草案hash，再从clean commit单次启动。任何失败均保留，不扩预算。
没有新答复不进行能力探测或raw/count查询；仅有官方文档不是这两个资产可用的证据。

仍只检查原生0.1秒网格、持续八车480点及input暴露；原RSU的站点/单位映射、provider与费用兼容
继续独立门禁。所有扫描仍为输入资格暴露，不自动认证hidden或独立采集。
训练、策略评价、模型拟合、环境/奖励/预测器/基线/RSU改动及hidden开启预算均为0。
当前完整formal/holdout/support和整套结构必要性目标不因此完成。
