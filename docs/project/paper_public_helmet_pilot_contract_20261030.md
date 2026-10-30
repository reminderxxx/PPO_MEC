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

## 首次0HTTP存储拒绝与原deadline恢复

clean b619c88首次启动被全局旧software_fixture的pytest `*current`符号链接拦下；producer/consumer各一次封存，新增0HTTP/0body/0图像，原失败audit不改写或重跑。独立诊断后另立最小实现：全局计账只计链接本体`lstat().st_size`，不跟随链接目标；source、raw body、GT等输入仍必须regular/no-symlink，不删除旧fixture。
新增1个微型tree案例（普通文件、目录别名、文件链接）单轮核producer和consumer计账一致，旧5case不重跑。
fresh `public_helmet_pilot_20261030_v2/`绑定首失败plan/audit/terminal/producer_failure/control SHA且验证0attempt；只接受这一直接首失败，拒绝递归恢复。
v2继承首run原`claim.start_clock`和同boot，仍是原900秒绝对deadline，连修复/等待都扣除，绝不开新的899/900秒。producer最多用到原870秒，最后30秒仍给consumer。原HTTP/body余额因0新增不变，所有失败原件继续计磁盘。

## 原生206分页与只读COCO恢复

v2完成302跳转134B和完整COCO 2065913B，发布方SHA匹配；COCO实际2392images/8275annotations、原生Helmet/NoHelmet类已核。
images清单合法返回HTTP206、`Content-Range: items 0-999/2392`、Content-Length754940，旧客户端误当错误，仅保留65536B后主动停止；不是服务器断流。旧consumer也误拒了HTTP合法重复Set-Cookie。三收据与失败audit封存。
独立诊断后另立v3：仅metadata分页可接受206，必须JSON、精确原生items start/end/2392与完整页/末页/count核对；附件仍只200+固定一次publicS3跳转。只有Set-Cookie多值可合并保留，绝不发送；长度/范围/Location冲突仍拒。
完整COCO原件只读SHA引用，不再下载、复制、改写；不完整metadata前缀不拼接。重新获取完整页的费用另计，不续写原run。
前缀累计15HTTP/3719957B返回body/读取上界0；另外保守扣未读声明实体754940−65536=689404B，不能把旧upper0改写或称精确wire总量。
新上限17HTTP/62699503B（累计32HTTP/64MiB含该额外声明上界），存储256MiB。仍继承最初900秒绝对deadline，修复等待也扣除，不另开时间。
只允许固定1d336三收据/原件及其b619零请求祖先链，逐SHA核对并拒绝重复恢复；新raw `public_helmet_pilot_20261030_v3/`、私有`native_pagination_v3/`。两纯内存header/算术检查和AST增量通过，0新fixture，不重跑旧case/audit。模型/Q/训练仍0。

## 最终固定文件收尾：额外独立300秒，不恢复旧时间

v3的3页清单已完整取得，原生2392文件与COCO对应，6目标及其selection SHA已冻结。第0图的发布方302包含两条合法`Server-Timing`统计头，被本地泛化冲突门拒绝；原图本体未请求。失败收据0body/65536读取上界及全部旧失败原封保留，原900秒范围不重置或追认为成功。
用户持续要求自主找办法并已明确允许下载；据此另立只补固定6文件的有限收尾任务：额外独立300秒（producer240/consumer60），不是原900秒复活，不声称整套历时≤900秒。此范围不新增来源、metadata/COCO查询、样本或科学授权。
先独立诊断并修已见的合法sideband头，保留raw header pairs；Set-Cookie/Server-Timing可多值，Content-Length/Type/Encoding/Range/Location等冲突仍拒。Cookie不发送，原body/source no-symlink不变。
只读引用完整COCO、3完整metadata及旧selection，不下载或复制这些原件。第0图仅跟随已留302的字面publicS3 Location，收据redirect_from=-1明确指历史已SHA绑定的parent；其他5图只用原metadata URL并最多1次publicS3 Location，预期11新HTTP。任何失败保存停止，不替换目标，不再自动开下一套网络任务。
前缀累计19HTTP/5526965B返回body/65536读取上界，另有689404B声明未读实体上界；总已收费6281905B。新最多13HTTP/60826959B，仍累计32HTTP/64MiB/256MiB原资源上限，全部负结果/原件/新软件与导出计账。
`finish_public_helmet_pilot.py`复用有界Capture，仅允许plan把请求/字节/时间cap缩小；`audit_public_helmet_delivery.py`独立验证全部原生GT/文件identity、继承账单、旧冻结选择、6图片/Pillow完整性与300秒收尾。新raw=`public_helmet_delivery_20261030_v4/`，私有`final_delivery_v4/`。consumer仅单次，exit0/无failure/audit-terminal SHA同时成立才资格；模型/Q/ALPR/hidden/论文完成均不因此获准。

## 实际交付：Helmet6输入资格通过（2026-10-30）

clean `a5ecc585d2aaf389a79dec7a4724185d73f82440`固定输入收尾只启动一次，实际11HTTP/10643077B、33.830951秒（独立consumer0.489504秒）。六张不同原图总10642407B，Helmet/NoHelmet原生ROI各3例；全部publisher SHA/size、Pillow格式/尺寸、框边界、完整清单及事前选择均通过。producer/consumer exit0、无consumer_failure、audit-terminal SHA配对通过，另作一次只读交付层接受，未重跑审计。

累计实际30HTTP/16170042B返回body、65536B中断读取上界、689404B独立声明未读实体上界，收费合计16924982B（约16.14MiB），低于32HTTP/64MiB。真实wire总量仍不可核验；不能把body或上述界限称全部互联网精确流量。原900秒范围的3个Helmet失败（v1/v2/v3）、更早M超时/API错误均封存；新额外300秒范围单列，不将旧失败改为成功。

交付位于主目录私有`paper_evidence_strengthening_20261029_v1/quality_helmet_pilot_acquisition_20261030_v1/final_delivery_v4/`：`helmet_samples.csv`、`helmet_groundtruth.json`、`audit.json`、`delivery_verification.json`、`independent_delivery_acceptance.json`；6原图仍为worktree新raw的opaque body文件，不上传GitHub。

本次只完成Helmet6输入。完整2392图COCO/8275标注及2392 metadata已暴露并记录；六原图不是六个新独立采集，公开许可声明也不认证底层权利链或模型训练无重叠。未来Q必须固定原生ROI/相同预处理/全部样本及成对基线，另立模型评价范围；本次模型、Q、训练/拟合/hidden均0。ALPR文字/明确NONE六例尚未资格，12例总计划不完整，任务质量收益及论文目标仍未验证。

软件验证谱系保留：首producer3/consumer2合成案例、1个22B存储微例及各次纯内存/AST增量分别绑定源码；不声称最终版本全部旧case重跑。最终输入端到端运行和单次独立审计真实通过，不是训练/benchmark结果。主目录七个用户修改文件SHA均未变，R20/老师稿位未编辑。
