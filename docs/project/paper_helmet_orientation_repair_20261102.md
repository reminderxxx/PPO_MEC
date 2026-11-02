# Helmet输入方向修复与选择评分诊断（2026-11-02）

用户本轮明确要求“找办法解决，继续进行实验，拿到证据”。目标仍为CSCWD2027，TMC规范仅内部证据要求。
新任务私有包为主目录`artifacts/analysis/paper_helmet_repair_20261102_v1/`，与旧Q、旧科学和所有失败分目录保存。

## 独立首因诊断先于新实验

只读复核已确认：旧q02/q03 JPEG为EXIF Orientation=6，COCO记录的width/height是存储方向，
但原native bbox在EXIF显示方向才落到相应带Helmet头部/无Helmet头发；旧raw ROI分别落到栏杆树叶和草地。
不变bbox、唯一由EXIF指定的90°方向变换即可复现这两个定位差异，未试旋转角度、改框或按模型结果选图。
旧input的哈希/尺寸/边界通过不足以认证语义ROI正确，应撤回“6个均为合格head ROI”的解释。
旧12个原始输出、0/6严格格式结果、账单和audit全部保留，不改判分或续写；不能据含两例错位输入的旧六例评定adapter识别能力。
独立诊断还核到实际PEFT启用、原参数匹配、两臂部分原始输出不同，旧输出已EOS结束；无证据把首因归为未激活或截断。

## 新固定实验

全部使用原6个已消费source images及native框，不下载、不训练、不拟合，不增加独立样本数。
对原图统一`ImageOps.exif_transpose`后转RGB、按原整数bbox裁剪、去元数据以固定PNG参数保存。
q00/q01/q04/q05像素不变，q02/q03随EXIF修正；display六例均已核对可见头部，旧原件只读。

A：仅q02/q03的display ROI×base/Helmet，共4次生成；旧prompt、greedy/max32参数及旧runtime/resources保持不变。
只读引用同例旧raw结果作配对；不重生成其余4例，也不混充新6例成绩。
这回答“单独修正图像方向会怎样改变原接口输出”。旧严格格式门不救分。

B：固定新object-presence选择题，非原wearing自由生成题的等价重评分：

- `Look at this cropped head region. Does it contain a motorcycle helmet? A: helmet present. B: no helmet present. Reply with A or B only.`
- 第二条只交换A/B对应的两个类别，全部样本两条都评，不按结果选提示词或标签顺序。

固定候选` A`=330和` B`=389，已对原tokenizer离线验证各为单token，保留共同前导空格风格。
每次单独显式forward、use_cache=False，在attention_mask最后有效token位置取完整float32 vocab logits并保存；
独立消费者用float64稳定二选softmax计算pA，主分采用事前固定的中心化均值
`pHelmet=0.5+0.5*(pA_map0-pA_map1)`，严格0.5为ABSTAIN并按错计，除此>0.5为Helmet、<0.5为NoHelmet。
该写法令相同A偏好在对调映射下精确相消，避免舍入造成假分类。另报完整vocab中的A/B总质量与映射冲突。
这些是条件相对分数，非校准正确概率。

输入矩阵固定为display6、同尺寸RGB(128,128,128)灰图6、q02/q03原raw背景对照2；
14输入×2映射×2模型=56显式评分forward。主类别结果只按6个display图片计；gray/raw不赋予新的质量标签。
gray只回答指定灰替换下的视觉敏感性，raw/display只在q02/q03同尺寸对比，不据其单独证明头盔语义利用。
两个模型实际input IDs、全部视觉tensor、candidate位置/配置必须匹配；原base/Helmet权重和完整runtime身份不刷新。

## 预算、实现与回收

单次2新模型进程、2processor/2base/1adapter load/1activation；4generate最多32新token，56显式score forward。
在实际Idefics3顶层入口注册一组pre/post hook记录底层forward；每score恰1，每generate与生成token数对应，
最多184=56+4×32，不把PEFT wrapper与base重复计数。所有begin失败也收费；无重试/补测/丢例。
旧12生成与约217.449秒等账单不重置，本轮新增账单独立列出。

CPUfloat32、单线程、batch1、seed0、offline、eval/inference_mode，复用旧T固定Python/资源。
45分钟包含producer2400秒、每worker1200秒及consumer剩余最多300秒；全新增256MiB、metadata/log32MiB。
受控写前、运行中全树/日志及PG/reap监管复用已验收旧软件harness；运行中冻结tracked源码。
软件仅新增producer4+consumer4小合成case单轮，最多8小图/8小数组、0预训练模型/网络；后续纯静态增量分版披露。

生产端`run_helmet_orientation_choice.py`和独立`audit_helmet_orientation_choice.py`须匹配验收，
随后clean commit、输入/源文件/plan/grant SHA冻结、单次push尝试后才启动；自动一次新audit，已有终端禁止重跑。
全部模型输出、完整logits、hook事件、参数/输入、原图及原审计引用保留；私有原件不上传。

## 接受及claim边界

完整矩阵、原件/旧引用、实际调用/预算和独立核验通过才能接受诊断完整性；科学结果可为负，不能以预测得分触发重跑。
本轮最多确认6已消费例上的输入修正与选择/视觉干预现象，B改变题义与接口，不能声称只修格式产生的因果收益。
不会把6例、两映射、三种图像条件当独立样本；不能认证总体准确率、训练无重叠或正式泛化。
原adapter训练来源与外部独立新测试、ALPR质量、物理成本/无线映射、公平对照及正式/holdout仍未闭合。
不改论文或老师稿位、不晋级canonical，`paper_goal_complete=false`。
