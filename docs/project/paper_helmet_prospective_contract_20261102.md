# Helmet24事前样本与固定选择评分合同（2026-11-02）

用户要求解决旧失败并继续实验、取得证据，既有公开数据下载和自主分发授权继续适用。
本轮与旧六例诊断分开，私有包为主目录`artifacts/analysis/paper_helmet_prospective_20261102_v1/`。
CSCWD2027为稿件目标；TMC只作内部规范，paper_goal_complete=false，不修改manuscript/canonical。

## 先冻选择与评价，不据六例成绩选图改题

已在读取当前六例4generate+56score结果前冻结selection SHA
`388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743`，
及evaluation_scope SHA `a6a1755b10d35de9d6d6b60708b25504f1c283e7bbd1e91e3f8d612657680be2`。
仅使用已经保存的官方COCO2392图/8275标注及完整三页metadata，不重新购物来源或扫描历史实验。
按SHA256(`helmet-prospective-20261102-v1|image_id|annotation_id`)全局排序，
类未满且image未占用才接受；实际接受后才占用图像，固定Helmet/NoHelmet各12、每图1ROI。
排除旧六image及同publisher SHA；native整数框宽高64..256，文件≤4MiB，必须有发布方SHA/字面URL。
不得按成绩重抽、补例或换种子。24个不同图像/文件hash不认证不同主体、独立采集或无训练重叠。

## 单次输入获取

acquisition_scope SHA `06c9b094aa0eaad416c48f875c1251e41440132c52f4816ea9588e7cdaab9baf`。
只取原24字面download_url，各最多一次发布方public-files S3 Location；不再查询列表、换来源/endpoint、取整包、Range或重试。
固定原图合计15,347,859B；新scope上限48HTTP、64MiB body及中断读取上界、1500秒（producer1380/consumer120）、128MiB全包。
先离线实现/验收再开启时间预算；失败前缀、未知读取和控制日志均保留计费，外层监督整个CLI及进程组。
旧累计公共输入账单16924982B（含上界）、所有失败及旧时限均保留，本轮不是旧run续写或预算复活。

原图按发布方size/SHA原样保留；60Mpixel上限、EXIF唯一指定显示方向、RGB及不变native框，元数据清除后生成opaque PNG。
原始COCO raw width/height只记录，不作为额外筛选理由。任一HTTP/格式/hash/CRC/范围/预算错误保存并停，不替换该例。
新独立consumer只运行一次，重核原件/收据/三页及COCO身份、24GT、EXIF/像素、全账单，并输出manifest/GT/contact sheet。
逐张视觉复核全部24裁剪且冻结接受记录后才可加载模型；遇到不合格输入先独立诊断，不静默转框、改标签或补例。
数据发布页声明CC BY4.0有既有本地引用；底层权利链与训练重叠仍未核验，不将发布方声明扩大为其他认证。

## 固定模型矩阵

只读24个已资格display PNG；两臂为原base和原Helmet adapter，全部weight/runtime/LoRA状态沿用旧固定身份，不训练或下载模型。
同六例诊断已事前固定的presence题、两种A/B类别对调、单token` A`=330/` B`=389，全部24×2×2=96显式评分。
CPUfloat32、threads1、batch1、seed0、offline、eval/inference、use_cache=False；无generate或旧六例重测。
每次保留完整float32 vocab logits，consumer独立float64 softmax；主分
`0.5+0.5*(pA_map0-pA_map1)`，大于0.5为Helmet，小于为NoHelmet，精确等于为ABSTAIN按错计。
完整vocab的A+B概率质量和映射冲突单独报告；条件分数不是校准正确概率。
模型只收图像tensor和固定prompt，不收publisher文件名、GT或类别字段；两臂实际token/所有视觉tensor/options须逐一相同。

额外上限2模型进程、2processor/2base/1adapter load/1activation、96score/96底层native forward、0generate，
85分钟5100秒（producer4800/worker每个2400/consumer300）、256MiB全评价包、32MiB日志。
绝对起点在CLI入口、前置身份核验也计时；外层同一5100秒护栏，失败不重试，不重置当前或旧花费。
新增输入128MiB与评价256MiB分包限额，实际新增及历史输入/旧12generate/本轮六例1033.433秒均累计列示。
旧主线292286物理帧不变，新环境frame/episode、训练/拟合、hidden预算全为0。

## 接收、统计与交付

两个新consumer分别单次自动调用；已有终端只复用，禁止再审计/再生成来救分。
输入完整性、调用完整性与科学准确率分开，不以高分决定是否成功收回数据。
报告全部24逐例/两映射分数、每类12混淆、ABSTAIN、两臂配对正确/错误计数，保留全负例。
由于主体/采集与pretraining/adapter训练交集未认证，不报告确认性p或总体CI，不推自然发生率准确率。
无峰值选择、prompt shopping、换权重或追加候选；任何后续修复先诊断、另立有限任务。

软件验收各自有界：输入4微例和评分4微例各单轮，0预训练/HTTP；随后身份/计时/输出接口必要增量只AST，分版记录不追认最终全链。
独立静态接收后，匹配验证、scoped commit、单次push、clean source/plan/grant冻结后才真实启动。
真实原图、logits、checkpoint、私有review和原始实验永不上传。

实际结果必须交图/CSV/账单/风险；本轮可以反驳旧“没有任何真实Q输入”的泛化，但不能补齐总体任务质量提升、
无线/RSU/费用映射、main reward收益、强外部公平对照、formal/holdout或整篇论文完成。
