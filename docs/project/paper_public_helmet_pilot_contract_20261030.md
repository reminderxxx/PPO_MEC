# Helmet6：发布方原生标注小样本输入合同（2026-10-30）

人工“允许你下载 你继续推进吧”及最新“你继续找办法”覆盖同一公开来源的小样本输入恢复。
官方bundle调用、固定版本folder树与COCO单文件清单已经单次留存并核账；不再猜接口或下载整包。
原M12请求已全部消费、封存：累计12HTTP、1588374B返回body、中断上界0、图像0。
旧641.769310秒FAIL及此前追加/诊断的独立时间均保留；新900秒不声称整套流程仍在原1800秒内。

本任务另立输入获取范围，最多新增20HTTP、65520490B返回body及不确定读取上界，累计32HTTP/64MiB。
所有新旧raw、私有记录、导出、失败和软件样例累计存储256MiB。新900秒含producer/consumer，最后30秒保留给单次独立消费。
先完成软件/静态独立复核、clean commit、精确plan/grant/SHA冻结及一次push尝试，之后才单次启动计时。
`kern.bootsessionuuid`与共享`clock_gettime(CLOCK_MONOTONIC)`绑定本次时钟，无fallback；不与旧进程clock相减。

唯一数据为EdgeVision V1（j82bnw7gsr/1）。固定COCO `annotations.json`为2065913B，
SHA `5e77fea3d8f1cd93cac42d45065456eeb97a54c31b7534ef453051c9d8853c91`，URL由已保存官方清单给出。
只读取同一images folder的最多3页清单，每页1000条、2MiB；必须末页少于1000且文件名/ID唯一、与COCO images构成精确双射。
COCO相对file_name只允许安全、唯一、区分大小写的basename映射，拒绝绝对路径/..和模糊匹配。

只取Helmet3+NoHelmet3个不同原图，每张不超过8MiB/2000万像素/8192边长。
类别只按原生name去空格/下划线/连字符后的Helmet/NoHelmet匹配，不能猜数字类别。
原生xywh框必须有限、在图内、宽高至少32；完全相同图ID/xywh具有多个原生类别时排除整个冲突组并报告。
固定类别顺序Helmet、NoHelmet；各类按SHA256(`j82bnw7gsr/1|fileID|annotationID|categoryName`)升序取前3，跳过已选原图。
获取COCO/完整清单后、任何图像GET前冻结6个目标、原生标注与selection SHA/clock。下载失败不替换样本，不按模型输出选样。

每个初始文件URL必须与官方metadata字面一致，只接受identity传输；原始body完整保存并核发布方size/SHA。
只对annotation/image资产允许一次字面Location到发布方public-files S3的HTTPS跳转，每跳单独计请求/bytes；
metadata页不跳转，拒绝同站登录/API跳转、第三跳、认证/Cookie/Range、自动重试或换源。成功annotation允许JSON/octet-stream，
metadata页仅JSON，图像仅image/*或octet-stream；实际图片格式/完整性由Pillow独立验证。
read1逐次扣新总余量；中断请求的已返回body与最多65536B不确定读取上界分账，不能当精确wire流量。
HTTP/API/schema/来源/尺寸/哈希/时间/存储/身份失败保存停止，旧raw不续写、不删改。

`prepare_public_helmet_pilot.py`负责有界下载与事前选择；只共享consumer的规则常量，不共享选择/核账实现。
`audit_public_helmet_pilot.py`不联网，独立从COCO、全部metadata、选择freeze、每次原件/收据重建资格与账单。
无论producer成功或失败只启动一次consumer，不修复重跑。只有consumer exit0、无consumer_failure且audit/terminal SHA配对同时成立才可接受。
输出`helmet_samples.csv`/`helmet_groundtruth.json`保留原生图像/框/类别与外部标注源；不制作模型标签，不裁图/调用模型。

软件资格分别保留：consumer首2合成case与后续MIME/redirect/grant的AST增量；producer首3合成case和后续静态增量。
旧版本通过不追认为最终版本完整链路重测，不重跑旧suite或真实审计来填工作。
raw新路径`data/raw/task_quality/public_helmet_pilot_20261030_v1/`；私有包`quality_helmet_pilot_acquisition_20261030_v1/`。

成功也只认证Helmet6输入。全COCO与metadata的2392量级标签暴露另计，不称只见过6个标签、hidden、独立采集或模型训练无重叠。
发布方CC BY声明不是底层权利链完整认证。ALPR的文字/无牌GT仍缺，12例总方案不完整。
Q/模型/训练/评价/拟合/hidden预算均0；不能据此声称适配器有效、物理费用兼容或论文完成。
