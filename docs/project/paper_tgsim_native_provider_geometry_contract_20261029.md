# TGSIM Source 1 原生 provider 与站点几何合同（2026-10-29）

- `contract_version`: `tgsim_native_provider_geometry_v1`。
- `reviewed_at` / `literature_cutoff`: `2026-10-29`，Asia/Shanghai；本任务不新增文献检索。
- `target_venue`: CSCWD2027；TMC 只作内部证据规范，不预测录用。
- `artifact_run_id`: `paper_evidence_strengthening_20261029_v1/tgsim_native_contract`。
- `policy_version`: `tmc_review_policy_v3_20260621`。
- `git_baseline`: `fc3f910c18797197b3048d758492925b3308ff3f`。
- `evidence_level`: 已保存输入资格/metadata 的合同核对；不是新原生行验收或正式科学证据。
- `verdict`: 合同可冻结；provider、真实坐标对齐、物理时钟/成本和 formal/hidden 资格未验收。`paper_goal_complete=false`。

## 1. 范围与旧失败

本任务执行论文证据强化队列中的 `tgsim_source1_provider_and_geometry_contract`。只复用已保存的 Source 1 metadata、输入资格及前轮只读报告，不实现或实例化新 provider，不再次读取轨迹、SQLite、旧索引、几何行，不重新扫描时间窗或下载资源。最大7200秒，训练、评价、拟合、下载、新增原生行均为0。

Source 1 固定为官方资产 `9uas-hf8b`（I-90/I-94 Stationary）。原资格记录4,023,022行的原生键、0.1秒网格和有限数值检查通过；完整获取链随后因几何入口失败停止，后置 version/count 未达到。后来任务的16份几何取得及 Source 2 数值失败也原样保留。本合同不把任何旧任务改写为成功，不认证事务快照，不拼接失败前缀。Source 2 `7zjf-a4zf` 不进入本 Source 1 provider 合同。

前轮保存容量是27个八车连续480点的时间候选，run1/2/3/4/5/6分别为2/8/1/8/0/8。它们不是27次独立采集，也不是已分配的12 formal加12 hidden。全部源行已用于输入资格，不能自动称 hidden。具体窗口和 cohort 尚未由新 provider 消费。

## 2. 原生身份与投影字段

未来实现必须同时定义生产端与消费端的 `tgsim_native_frame_v1`，不冒充 NGSIM CSV，不复用 `Global_Time` 的隐含意义。

| 字段 | 冻结语义 | 拒绝条件 |
| --- | --- | --- |
| `source_asset_id` | 必须为 `9uas-hf8b`，绑定保存 metadata 与派生页来源的SHA | 缺失、别名替换或来源不符 |
| `run_index` / `vehicle_native_id` | 由原数值ID按精确十进制解析为整数；不通过float截断 | 非整数、非有限或无法精确解析 |
| `native_tick` | `Decimal(raw_time_seconds) * 10` 必须恰为非负整数 | 不能整除到0.1秒网格；不得四舍五入 |
| `dt_seconds` | 精确有理数 `1/10`；序列时间由整数tick导出 | 偷换成1秒、毫秒或缺少单位 |
| `time_index` | 为适配整数接口时只能表示 `native_tick`，不能宣称Unix时间 | `int(raw_time_seconds)`、跨run排序合并 |
| `sample_ordinal` | 窗内0起始的顺序索引，与原生时间分开保存 | 用ordinal覆盖原native_tick |
| `vehicle_id` | `tgsim:9uas-hf8b:run=<run>:vehicle=<id>`，每辆车持久稳定 | 忽略asset/run导致ID碰撞 |
| `position_x_m` / `position_y_m` | 直接来自 `xloc_kf` / `yloc_kf` 的有限米制数值 | feet数字当meters；静默缩放、旋转、平移 |
| `speed_mps` | 来自 `speed_kf`，单位m/s；不以坐标差值替换或自动取绝对值 | 非有限/负速度按冻结数据资格拒绝并保留原件 |
| `coordinate_frame_id` | `tgsim:9uas-hf8b:run=<run>:reference-image-m` | 跨run默认共用同一空间原点 |
| `source_provenance` | 原始asset/run/key、派生CSV与页清单/metadata身份、投影版本 | 只有重命名后的CSV或不可追踪的临时路径 |

唯一原生样本键为 `(source_asset_id, run_index, vehicle_native_id, native_tick)`。同一键重复必须失败，不能取最后一行、平均、去重后静默继续。ID可相同而run不同；不得据车辆ID把两个run接成一条轨迹。未知字段必须保持未知，不能从ID、排序、文件名或错误计数推断出来。

`base_model_id`、adapter请求、车辆算力、队列、workflow/DAG、release/deadline均不由轨迹提供。后续只能由另一个冻结的合成/跨源工作负载合同附加，记录为外生映射，不能写成原生联合VEC请求。

## 3. 时序、cohort 与终止

一个 provider 实例只服务一个asset、一个run及一个事前固定的窗口。480个连续采样点的首末点跨度是47.9秒，样本间有479个0.1秒转移；“480点”“480次动作”“48秒持续区间”不是可互换的口径。

未来最小接口应使 `reset()` 暴露第一个真实样本，`advance()` 每次严格增加一个原生tick；到最后样本显式标记 `at_end=true`，再advance必须返回明确的 `EndOfNativeWindow`，不能返回复制的末帧。若现有环境仍要求480次物理交互，必须先另立环境时钟/动作区间合同解决479转移与480交互的对应；不能补造第481点、post-terminal padding或隐含零时长动作。

cohort只能按输入覆盖选择，不能看奖励。后续原生行消费者须事前冻结候选窗及确定性ID顺序，从全480点连续存在的车辆中选固定8辆；不能在回合中用新车替换失联车。缺帧、跳tick、重复键、cohort缺样均保存并拒绝，不插值、不前向填充、不补零、不丢行修复。当前摘要只有存在性和可行区间，没有实际cohort ID，因此本合同不宣称具体8辆已验收。

## 4. 物理原点、单位与合成RSU

保存 metadata 说明坐标为参考图像下米制，参考图像原点位于左上。可冻结的是每个run的局部度量坐标系；地理绝对原点、朝向、跨run旋转/平移和经纬度映射在已存证据中未核明，必须保留 `absolute_origin=null`、`orientation_status=unverified`、`cross_run_transform=null`。不能把图像x/y命名为east/north或把0,0当某个地理坐标。

车道中心线中的 `x_<lane>/y_<lane>/ramp_<lane>` 是道路几何，不是RSU位置/覆盖，也不是无线测量。本合同仅引用前轮保存的6个 Source 1 geometry附件身份及header事实，不再次读行；不能用 Source 2 geometry 补 Source 1 空间定义。

未来站点几何必须有独立 `tgsim_site_geometry_v1` manifest，至少包含：

1. `source_asset_id/run_index/coordinate_frame_id`，与原生投影精确一致；`position_unit=m`。
2. 每个合成RSU的唯一ID、有限米制x/y、正且有单位的覆盖半径；实际数值及放置规则在下一任务冻结，本合同不生成实地位置。
3. `placement_kind=synthetic`、`deployment_measured=false`，明确研究者构造的基础设施假设及参数来源；同一配对的所有方法共用同一几何。
4. 最近距离/边界相等/无覆盖/距离并列时的确定性关联规则，及规则版本；在纯合成fixture上验证。
5. 不得以改善覆盖、handoff数量或奖励为由调几何；需变更时另立输入驱动资格与前瞻比较。

原研究四RSU的坐标和762.483数值半径属于原NGSIM数值坐标合同；原provider保留官方feet数值。它们不经证明不能当作新站点meters复用，即使数字偶然落在覆盖圆中。相反，随意把轨迹按比例拉伸到旧圆内也不是单位换算或独立外推证据。本合同既不改旧几何，也不声称当前存在已验收的新合成布局。

## 5. 跨run时钟与统计身份

`time`相对各run起点重置，`run_index`表示发布方的采集过程标签。当前保存资料未给出绝对同步开始/结束、共同采集时钟和clip重叠映射。未来manifest必须保存：

- `clock_kind=run_relative`、`absolute_start_utc=null`、`cross_run_clock_alignment=unknown`。
- `capture_overlap_status=unknown`；不把run标签、不同vehicle ID或不同窗口ID变成独立采集证明。
- 未取得足够采集证据前，将这些run置于同一个“可能相关的来源组”用于保守隔离；不声称该组有12个独立外层统计单位。
- 窗内不重叠/480空tick间隔只能证明给定run时钟下的区间隔离；不能证明跨run不同时录到同一车/时段，也不单独证明随机独立。

27候选可用于后续明示的cross-site input/support资格准备。formal/hidden必须在来源/窗口/模型/统计规则冻结及历史暴露处理后另审，当前认证仍为0。输入先前被扫描不等于已经做策略评价，但也不能在没有once-open协议时自封hidden。

## 6. 物理时钟与费用阻塞

轨迹和centerline没有无线带宽、RTT/丢包、RSU算力、adapter加载、状态序列化/传输或费用的实测值。新provider不能从这些位置字段制造物理成本。必须另立维度一致的消费端合同：

| 类别 | 未来必须显式声明的量 | 当前边界 |
| --- | --- | --- |
| 网络 | bytes/s、传播/排队秒数、每transition可服务字节及小数余额 | 原per-frame常数不能改名直接当0.1秒测量 |
| 计算 | cycles/s或经测量的服务秒数，执行区间与队列时钟 | 不把所有节点单帧完成当真实任务成本 |
| adapter/state | 字节、加载/恢复秒数、依赖/版本、传输完成时刻 | header/技术恢复不等于任务质量或当前stream校准 |
| 费用/reward | 秒/字节/计算资源到原生账本的维度映射和取值依据 | 不为新图修改奖励权重或减少主方法费用 |

例如纯合成fixture中960 bytes/s在0.1秒应提供96字节服务量；1 byte/s的十个0.1秒区间须合计1字节，不能逐帧截断成0。以上只是维度验收例，不是任何实际链路参数。必须保留网络/计算/费用的测量、模型、未核验三种状态；新站点只带来真实移动性输入，不带来真实RSU或联合请求证据。

## 7. 下一轮最小纯软件资格规格

本任务不执行以下provider测试；它们是下一独立实现合同的最低清单，先用合成数据，不消费本地真实行。

| 编号 | 合成见证 | 要求 |
| --- | --- | --- |
| T01 | 时间0.0/0.1/0.2，整数tick0/1/2 | 无小数秒合并；dt精确1/10 |
| T02 | 时间0.15、NaN、负值、ID非整数 | 明确失败且保留错误位置；不修复 |
| T03 | 相同车辆ID在两个run、同一键重复 | run实例隔离；重复样本拒绝 |
| T04 | 480点与最后点再次advance | 479真实转移，显式结束；无末帧复制 |
| T05 | 8辆中1辆中间缺点、第9辆仅后半段出现 | 固定完整cohort，无替补/插值 |
| G01 | 0,0/3,4米距离、覆盖边界、无覆盖、距离并列 | 米制与确定性关联一致 |
| G02 | feet单位、错误frame ID、未冻结几何数值 | 明确拒绝；不套旧NGSIM布局 |
| C01 | 两run绝对时钟未知、相同相对tick | 统计不能判独立或合并时间线 |
| C02 | 同run重叠窗口与留空480tick窗口 | 精确区间判别；不冒充跨run认证 |
| U01 | bytes/s×dt以及不足1字节的服务余额 | 总量守恒；不能用取整少计成本 |
| P01 | 生产/消费 schema、来源SHA、版本和路径篡改 | 生产者与消费端同时拒绝不匹配 |
| P02 | reset、step、checkpoint/source identity | 重新开始不跨run，不读未来信息，不继承另一场景状态 |

下一任务必须冻结fixture数、wall/disk、实现和消费端范围，验收后才讨论单独有界真实行投影。真实行投影还需冻结精确窗口/cohort、行/字节/时间上限、create-only输出、完整性与缺失失败规则；本合同不授予该消费权限，更不授予新训练或策略评价。

## 8. 本轮验收与结论

本轮仅验证保存metadata的asset、字段单位/时钟描述、Source 1几何附件身份与前轮报告一致，绑定输入与合同SHA；不重做原生资格、旧审计、几何小样本或时间容量扫描。私有`task_plan.json/native_contract.json/metadata_validation.json`保存实际允许范围与结果。文档检查包括必需字段、禁止行为、来源链接及`git diff --check`，不是provider运行正确性测试。

可关闭的是“原生键/dt/米制/合成RSU/未知clock的接口规格缺失”。仍真实缺少的是跨run采集时钟与重叠信息、实际新站点RSU参数与其来源、环境动作/物理转移时钟对应、成本校准和DAG请求/质量映射。当前没有新增独立formal/hidden或论文科学收益；旧负结果、未知历史和失败均保留。
