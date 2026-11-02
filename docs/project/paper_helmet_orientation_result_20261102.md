# Helmet方向修复实验实际结果（2026-11-02）

执行源码`2fa20d5ad93a9e6382f339159bf7222b8fc15538`，plan SHA `520b0183361b86025a10ea6422378e84c385d8a2b939fab03448763feed81799`。
私有原件：主目录`artifacts/analysis/paper_helmet_repair_20261102_v1/`；自动一次audit与terminal一致，未重跑旧run或audit。
CSCWD2027为稿件目标，TMC只是内部证据规范，`paper_goal_complete=false`。

## 确认的修复与实际负结果

旧q02/q03 EXIF6方向漏处理，原ROI裁到背景。唯一按EXIF指定方向处理、不改native框，才得到头部ROI。
这是已确认的输入缺口；旧原件、0/6严格格式分数和全部账单不变，旧“六例语义ROI均合格”解释撤回。

新4次生成只覆盖这两例×两模型，原prompt/options不变。base两个原始suffix改变，Helmet两个suffix未变；四次均未通过旧严格格式门。
不能把方向修复等同于识别修好，也不能事后放宽parser宣布旧分数提高。

新56次显式forward固定14图像条件×两映射×两模型。六个display主例两臂均3/6，且全部预测Helmet：三正例对、三负例错。
base六例、adapter五例在两个标签映射下类别不一致，表现出明显字母选择偏好。
adapter的q02 display相对灰图的条件Helmet分数增加约0.262284，说明该指定干预下输出对图像有变化；
它没有带来本组六例分类正确数增加，不能据此证明头盔语义机制或adapter质量优势。
灰图/raw是干预对照，不新增有标签独立样本。分数为A/B条件相对分数，非校准正确概率。

## 完整性与资源

2个新模型进程、2processor/2base/1adapter load/1activation、4generate、56显式score；实际底层native forward82，低于184上限。
parent实际1033.432835秒；0HTTP、0训练拟合、0环境episode，原主线科学292286物理帧不变。
全部28条件、56映射、4生成配对、full-vocab logits、输入tensor与参数SHA保留。
图为全0–1轴、不平滑的六例分数与同尺寸灰图比较；未画成“旧0提高到50%”的等价准确率对比。

软件谱系：producer4小case单轮通过，其后的执行前校验只做AST增量；consumer4微例单轮通过并绑定最终消费者SHA。
不是同一最终源版本八项全链路重跑。独立纯静态接受与真实一次audit分别保存。

## 下一步及边界

在读取本轮模型输出前，已另冻24个不同原图（12+12）的hash选择与同一两映射规则，排除旧6及同publisher SHA。
下一轮独立获取、EXIF显示ROI逐例资格后才运行，保持全部负例，不按六例成绩换prompt、换图或追加权重。
当前六例已消费，只是输入/接口/图像干预诊断；训练重叠、同主体/采集相关性未知。
不能据此补齐formal/holdout、真实任务质量提升、无线/RSU成本、净系统收益或整篇论文完成。
